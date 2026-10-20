// V1 adaptations of the existing stable CUB router and fused native TopK epilogue.
// All shapes/addresses are capture-stable; lengths, tables and take counts are device data.
#include "topk_v2.cuh"
#include <cub/block/block_radix_sort.cuh>
#include <math_constants.h>
namespace {
template<int THREADS,int ITEMS,int BITS,typename Index>
__global__ void v1_route_main(const float* scores,const int32_t* lens,const int32_t* takes,
    Index* selected,int pages,int take_stride,int64_t score_stride,const int32_t* rt,int32_t* ft,const int32_t* tail_rows,
    int fine_pages,int capacity,bool fill_table){
    int row=blockIdx.x, active=max(lens[row]-1,0)/1024*16, take=takes[row];
    if(fill_table){
        for(int j=take+threadIdx.x;j<fine_pages;j+=THREADS)
            ft[int64_t(row)*fine_pages+j]=capacity+tail_rows[row]*68+min(j-take,67);
    }
    if(take>=active){
        for(int i=threadIdx.x;i<take_stride;i+=THREADS){
            selected[int64_t(row)*take_stride+i]=i;
            if(fill_table && i<take) ft[int64_t(row)*fine_pages+i]=rt[int64_t(row)*pages+i];
        }
        return;
    }
    using Sort=cub::BlockRadixSort<float,THREADS,ITEMS,int,BITS>;
    __shared__ typename Sort::TempStorage storage;
    float values[ITEMS];int ids[ITEMS];
    constexpr int BASE_ITEMS=THREADS*ITEMS/128;
    #pragma unroll
    for(int i=0;i<ITEMS;i++){
        int old_rank=threadIdx.x*ITEMS+i;
        int p=(old_rank/BASE_ITEMS)+(old_rank%BASE_ITEMS)*128;
        float best=-CUDART_INF_F;
        if(p<active){
            const float* s=scores+int64_t(row)*score_stride+p*8;
            // Match the tuned old router's two vector loads per page.
            const float4 a=reinterpret_cast<const float4*>(s)[0];
            const float4 b=reinterpret_cast<const float4*>(s)[1];
            best=fmaxf(fmaxf(fmaxf(a.x,a.y),fmaxf(a.z,a.w)),
                       fmaxf(fmaxf(b.x,b.y),fmaxf(b.z,b.w)));
        }
        values[i]=best;ids[i]=p;
    }
    Sort(storage).SortDescendingBlockedToStriped(values,ids);
    #pragma unroll
    for(int i=0;i<ITEMS;i++){
        int rank=threadIdx.x+i*THREADS;
        if(rank<take_stride)selected[int64_t(row)*take_stride+rank]=ids[i];
        if(fill_table && rank<take)ft[int64_t(row)*fine_pages+rank]=rt[int64_t(row)*pages+ids[i]];
    }
}
struct V1Map {
    const int32_t *members,*rt,*nt,*lens,*takes;
    const void* selected;
    int32_t *physical,*logical,*raw;
    int pages,take_stride,window;
    bool mask;
    bool bucket_from_rank;
    int32_t* expanded;
    int expansion_stride;
    const int32_t* fine_table;
    int fine_pages;
};
template<int Level,bool Expand,bool Physical,typename Index> TOPK_KERNEL void v1_topk_main(const __grid_constant__ TopKLaunchParams params,
    const __grid_constant__ V1Map map){
    device::enable_smem_spilling();
    auto problem=params.problem(blockIdx.x);
    __shared__ impl::MaxSmem<Register2::Smem,Register4::Smem,Streaming::Smem> smem;
    __shared__ int32_t indices[kMaxTopK];
    int64_t row=blockIdx.x;
    int n=map.lens[row],take=map.takes[row],sink=min(n,64);
    int tail=max(sink,min(max(n-1,0)/1024*1024,max(sink,n-map.window)));
    if(map.mask){
        // Native TopK owns this row exclusively. Mask only invalid entries;
        // do not read and rewrite all valid logits in a separate GPU pass.
        device::PDLWaitPrimary<true>();
        float* scores=const_cast<float*>(problem.in);
        // Every reordered page belongs to one immutable 1024-position bucket.
        // Interior buckets cannot overlap sink/tail, so inspect their bucket
        // identity once per warp instead of reading all 64 member IDs.
        const int lane=threadIdx.x%32,warp=threadIdx.x/32;
        for(int r=warp;r<take;r+=kBlockSize/32){
            int page=0,rank=0,bucket=0;
            if(lane==0){
                rank=static_cast<const Index*>(map.selected)[row*map.take_stride+r];
                if(map.bucket_from_rank){
                    // State.table registers exactly 16 grouped pages per 1K
                    // bucket. This is a logical page rank, never a physical ID.
                    bucket=(rank/16)*1024;
                }else{
                    page=map.rt[row*map.pages+rank];
                    bucket=(map.members[int64_t(page)*64]/1024)*1024;
                }
            }
            bucket=__shfl_sync(0xffffffff,bucket,0);
            if(bucket<sink || bucket+1024>tail){
                if(map.bucket_from_rank && lane==0)
                    page=map.rt[row*map.pages+rank];
                page=__shfl_sync(0xffffffff,page,0);
                #pragma unroll
                for(int i=0;i<2;i++){
                    int offset=lane+i*32;
                    int logical=map.members[int64_t(page)*64+offset];
                    if(logical<sink || logical>=tail) scores[r*64+offset]=-CUDART_INF_F;
                }
            }
        }
        __syncthreads();
    }
    problem.out=indices;
    if(problem.seq_len<=problem.topk){
        device::PDLWaitPrimary<true>();
        for_each_item(problem.topk,[&](uint32_t tx,uint32_t){indices[tx]=tx<problem.seq_len?int(tx):-1;});
    }else if constexpr(Level==0){
        __builtin_assume(problem.seq_len<=kReg2MaxSeqLen);Register2::forward<true>(problem,&smem);
    }else if constexpr(Level==1){
        __builtin_assume(problem.seq_len<=kReg4MaxSeqLen);Register4::forward<true>(problem,&smem);
    }else{
        if(problem.seq_len<=kReg4MaxSeqLen)Register4::forward<true>(problem,&smem);
        else Streaming::forward<true>(problem,&smem);
    }
    device::PDLTriggerSecondary<true>();__syncthreads();
    for_each_item(problem.topk,[&](uint32_t tx,uint32_t){
        int raw=indices[tx],logical=-1,physical=-1;
        if(raw>=0 && raw<problem.seq_len && problem.in[raw]!=-CUDART_INF_F){
            if(raw<take*64){
                int page;
                if(map.fine_pages){
                    page=map.fine_table[row*map.fine_pages+raw/64];
                }else{
                    int rank=static_cast<const Index*>(map.selected)[row*map.take_stride+raw/64];
                    page=map.rt[row*map.pages+rank];
                }
                logical=map.members[int64_t(page)*64+raw%64];
            }else{
                int off=raw-take*64;logical=off<sink?off:tail+off-sink;
            }
            if(logical>=0 && logical<n){
                if constexpr(Physical)physical=map.nt[row*map.pages+logical/64]*64+logical%64;
            }else logical=-1;
        }
        int64_t dst=row*problem.topk+tx;
        map.logical[dst]=logical;
        if constexpr(Physical){map.physical[dst]=physical;map.raw[dst]=raw;}
        if constexpr(Expand){
            const int base=logical*4;
            reinterpret_cast<int4*>(map.expanded+row*map.expansion_stride)[tx]=
                logical<0?make_int4(-1,-1,-1,-1):make_int4(base,base+1,base+2,base+3);
        }
    });
    if constexpr(Expand){
        for(int j=problem.topk*4+threadIdx.x;j<map.expansion_stride;j+=kBlockSize)
            map.expanded[row*map.expansion_stride+j]=-1;
    }
}
struct V1QueryKernel {
    static void route(tvm::ffi::TensorView scores,tvm::ffi::TensorView lens,
        tvm::ffi::TensorView takes,tvm::ffi::TensorView selected,int64_t pages,
        tvm::ffi::TensorView rt,tvm::ffi::TensorView ft,tvm::ffi::TensorView tail_rows,
        int64_t capacity,bool fill_table){
        using namespace host;
        auto B=SymbolicSize{"batch"},L=SymbolicSize{"score_cols"},S=SymbolicSize{"score_stride"},K=SymbolicSize{"take"};
        auto dev=SymbolicDevice{};dev.set_options<kDLCUDA>();
        TensorMatcher({B,L}).with_strides({S,1}).with_dtype<float>().with_device(dev).verify(scores);
        TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(lens);
        TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(takes);
        const bool selected32=selected.dtype().bits==32;
        if(selected32) TensorMatcher({B,K}).with_dtype<int32_t>().with_device(dev).verify(selected);
        else TensorMatcher({B,K}).with_dtype<int64_t>().with_device(dev).verify(selected);
        TensorMatcher({B,pages}).with_dtype<int32_t>().with_device(dev).verify(rt);
        TensorMatcher({B,-1}).with_dtype<int32_t>().with_device(dev).verify(ft);
        TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(tail_rows);
        RuntimeCheck(capacity>0 && ft.size(1)>=K.unwrap(),"Invalid fine page table");
        RuntimeCheck(pages>=64&&pages<=4096&&L.unwrap()>=pages*8&&K.unwrap()<=pages&&S.unwrap()%4==0,"Invalid V1 route capacity");
        #define ROUTE_TYPED(T,I,R,INDEX) LaunchKernel(B.unwrap(),T,dev.unwrap()).launch(v1_route_main<T,I,R,INDEX>,static_cast<const float*>(scores.data_ptr()),static_cast<const int32_t*>(lens.data_ptr()),static_cast<const int32_t*>(takes.data_ptr()),static_cast<INDEX*>(selected.data_ptr()),int(pages),int(K.unwrap()),S.unwrap(),static_cast<const int32_t*>(rt.data_ptr()),static_cast<int32_t*>(ft.data_ptr()),static_cast<const int32_t*>(tail_rows.data_ptr()),int(ft.size(1)),int(capacity),fill_table)
        #define ROUTE(T,I,R) do { if(selected32){ROUTE_TYPED(T,I,R,int32_t);}else{ROUTE_TYPED(T,I,R,int64_t);} } while(0)
        if(pages<=128){ROUTE(64,2,4);}else if(pages<=256){ROUTE(64,4,4);}
        else if(pages<=512){ROUTE(64,8,4);}else if(pages<=1024){ROUTE(128,8,4);}
        else if(pages<=2048){ROUTE(128,16,4);}else{ROUTE(128,32,5);}
        #undef ROUTE
        #undef ROUTE_TYPED
    }
    static void topk(tvm::ffi::TensorView scores,tvm::ffi::TensorView fine_lens,
        tvm::ffi::TensorView members,tvm::ffi::TensorView rt,tvm::ffi::TensorView nt,
        tvm::ffi::TensorView selected,tvm::ffi::TensorView lens,tvm::ffi::TensorView takes,
        tvm::ffi::TensorView physical,tvm::ffi::TensorView logical,tvm::ffi::TensorView raw,int64_t window,bool mask,tvm::ffi::TensorView expanded,bool physical_output,bool bucket_from_rank,tvm::ffi::TensorView fine_table,bool direct_fine){
        using namespace host;
        auto B=SymbolicSize{"batch"},L=SymbolicSize{"cols"},S=SymbolicSize{"stride"},P=SymbolicSize{"pages"},K=SymbolicSize{"take"},T=SymbolicSize{"topk"};
        auto dev=SymbolicDevice{};dev.set_options<kDLCUDA>();
        TensorMatcher({B,L}).with_strides({S,1}).with_dtype<float>().with_device(dev).verify(scores);
        TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(fine_lens);
        TensorMatcher({-1,64}).with_dtype<int32_t>().with_device(dev).verify(members);
        TensorMatcher({B,P}).with_dtype<int32_t>().with_device(dev).verify(rt);
        TensorMatcher({B,P}).with_dtype<int32_t>().with_device(dev).verify(nt);
        const bool selected32=selected.dtype().bits==32;
        if(selected32) TensorMatcher({B,K}).with_dtype<int32_t>().with_device(dev).verify(selected);
        else TensorMatcher({B,K}).with_dtype<int64_t>().with_device(dev).verify(selected);
        TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(lens);
        TensorMatcher({B}).with_dtype<int32_t>().with_device(dev).verify(takes);
        TensorMatcher({B,T}).with_dtype<int32_t>().with_device(dev).verify(physical);
        TensorMatcher({B,T}).with_dtype<int32_t>().with_device(dev).verify(logical);
        TensorMatcher({B,T}).with_dtype<int32_t>().with_device(dev).verify(raw);
        TensorMatcher({B,-1}).with_dtype<int32_t>().with_device(dev).verify(expanded);
        TensorMatcher({B,-1}).with_dtype<int32_t>().with_device(dev).verify(fine_table);
        RuntimeCheck(!direct_fine || fine_table.size(1)>=K.unwrap(),"Invalid direct fine table");
        const int expansion_stride=expanded.size(1);
        RuntimeCheck(expansion_stride==0 || expansion_stride==T.unwrap()*4+64,"Invalid expansion buffer");
        RuntimeCheck(B.unwrap()>512&&T.unwrap()<=2048&&T.unwrap()>0&&window>=0&&window<=4096,"Unsupported fused V1 TopK");
        const auto params=TopKLaunchParams{.scores=static_cast<const float*>(scores.data_ptr()),.seq_lens=static_cast<const int32_t*>(fine_lens.data_ptr()),.page_table=nullptr,.page_indices=static_cast<int32_t*>(raw.data_ptr()),.raw_indices=nullptr,.metadata=nullptr,.score_stride=S.unwrap(),.page_table_stride=0,.topk=uint32_t(T.unwrap()),.page_bits=6,.cluster_floor=std::numeric_limits<uint32_t>::max()};
        const auto map=V1Map{static_cast<const int32_t*>(members.data_ptr()),static_cast<const int32_t*>(rt.data_ptr()),static_cast<const int32_t*>(nt.data_ptr()),static_cast<const int32_t*>(lens.data_ptr()),static_cast<const int32_t*>(takes.data_ptr()),selected.data_ptr(),static_cast<int32_t*>(physical.data_ptr()),static_cast<int32_t*>(logical.data_ptr()),static_cast<int32_t*>(raw.data_ptr()),int(P.unwrap()),int(K.unwrap()),int(window),mask,bucket_from_rank,static_cast<int32_t*>(expanded.data_ptr()),expansion_stride,static_cast<const int32_t*>(fine_table.data_ptr()),direct_fine?int(fine_table.size(1)):0};
        #define LAUNCH_TYPED(LEVEL,EXPAND,PHYSICAL,INDEX) LaunchKernel(B.unwrap(),kBlockSize,dev.unwrap()).config({.use_pdl=true}).launch(v1_topk_main<LEVEL,EXPAND,PHYSICAL,INDEX>,params,map)
        #define LAUNCH(LEVEL,EXPAND,PHYSICAL) do { if(selected32){LAUNCH_TYPED(LEVEL,EXPAND,PHYSICAL,int32_t);}else{LAUNCH_TYPED(LEVEL,EXPAND,PHYSICAL,int64_t);} } while(0)
        #define RUN(LEVEL,EXPAND) do { if(physical_output){LAUNCH(LEVEL,EXPAND,true);}else{LAUNCH(LEVEL,EXPAND,false);} } while(0)
        if(expansion_stride){
            if(L.unwrap()<=kReg2MaxSeqLen){RUN(0,true);}else if(L.unwrap()<=kReg4MaxSeqLen){RUN(1,true);}else{RUN(2,true);}
        }else{
            if(L.unwrap()<=kReg2MaxSeqLen){RUN(0,false);}else if(L.unwrap()<=kReg4MaxSeqLen){RUN(1,false);}else{RUN(2,false);}
        }
        #undef RUN
        #undef LAUNCH
        #undef LAUNCH_TYPED
    }
};
}
