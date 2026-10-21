// Adapted from V1 v1_route_main: preserve eight-center max and CUB page sort.
// Only output/metadata contract changes: published pages, physical IDs, actual takes.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
#include <cfloat>

template<int ITEMS>
__global__ void route_physical(const float* scores, const int32_t* published,
    const int32_t* budgets, const int32_t* rt, int32_t* selected, int32_t* takes,
    int pages, int take_stride, int64_t score_stride) {
    constexpr int THREADS=128;
    int row=blockIdx.x;
    int active=min(max(published[row],0),pages);
    int take=min(min(max(budgets[row],0),active),take_stride);
    if(threadIdx.x==0) takes[row]=take;
    // Fill every output entry on every replay; padding remains a safe page ID.
    if(take==0 || take==active) {
        for(int i=threadIdx.x;i<take_stride;i+=THREADS)
            selected[int64_t(row)*take_stride+i]=i<take ? rt[int64_t(row)*pages+i] : 0;
        return;
    }
    using Sort=cub::BlockRadixSort<float,THREADS,ITEMS,int>;
    __shared__ typename Sort::TempStorage storage;
    float values[ITEMS]; int ids[ITEMS];
    constexpr int BASE_ITEMS=ITEMS;
    #pragma unroll
    for(int i=0;i<ITEMS;i++) {
        int old_rank=threadIdx.x*ITEMS+i;
        int p=old_rank/BASE_ITEMS+(old_rank%BASE_ITEMS)*128;
        float best=-CUDART_INF_F;
        if(p<active) {
            const float* s=scores+int64_t(row)*score_stride+p*8;
            const float4 a=reinterpret_cast<const float4*>(s)[0];
            const float4 b=reinterpret_cast<const float4*>(s)[1];
            best=fmaxf(fmaxf(fmaxf(a.x,a.y),fmaxf(a.z,a.w)),
                       fmaxf(fmaxf(b.x,b.y),fmaxf(b.z,b.w)));
            // Active all-masked pages still precede invalid capacity padding.
            best=best==-CUDART_INF_F ? -FLT_MAX : best;
        }
        values[i]=best; ids[i]=p;
    }
    Sort(storage).SortDescendingBlockedToStriped(values,ids);
    #pragma unroll
    for(int i=0;i<ITEMS;i++) {
        int rank=threadIdx.x+i*THREADS;
        if(rank<take_stride)
            selected[int64_t(row)*take_stride+rank]=rank<take ? rt[int64_t(row)*pages+ids[i]] : 0;
    }
}

void route_cuda(torch::Tensor scores,torch::Tensor published,torch::Tensor budgets,
                torch::Tensor rt,torch::Tensor selected,torch::Tensor takes) {
    TORCH_CHECK(scores.is_cuda() && scores.scalar_type()==torch::kFloat32 && scores.dim()==2 && scores.is_contiguous(), "scores must be contiguous CUDA float32 [R,P*8]");
    TORCH_CHECK(rt.dim()==2 && selected.dim()==2, "rt/selected must be matrices");
    int64_t rows=scores.size(0), pages=rt.size(1), width=selected.size(1);
    TORCH_CHECK(rows>0 && pages>0 && pages<=4096 && width>0 && width<=pages, "invalid capacity (1..4096 pages, 1..pages output width)");
    TORCH_CHECK(scores.size(1)==pages*8 && rt.dim()==2 && rt.size(0)==rows && selected.dim()==2 && selected.size(0)==rows, "shape mismatch");
    for(auto t:{published,budgets,rt,selected,takes}) {
        TORCH_CHECK(t.is_cuda() && t.device()==scores.device() && t.scalar_type()==torch::kInt32 && t.is_contiguous(), "metadata must be contiguous int32 on the scores device");
    }
    TORCH_CHECK(published.numel()==rows && budgets.numel()==rows && takes.numel()==rows,"row metadata mismatch");
    c10::cuda::CUDAGuard guard(scores.device());
    auto stream=at::cuda::getCurrentCUDAStream();
    int items=1;while(items*128<pages)items*=2;
#define LAUNCH(I) route_physical<I><<<rows,128,0,stream>>>(scores.data_ptr<float>(),published.data_ptr<int32_t>(),budgets.data_ptr<int32_t>(),rt.data_ptr<int32_t>(),selected.data_ptr<int32_t>(),takes.data_ptr<int32_t>(),pages,width,scores.stride(0))
    switch(items) {
        case 1: LAUNCH(1);break; case 2: LAUNCH(2);break;
        case 4: LAUNCH(4);break; case 8: LAUNCH(8);break;
        case 16: LAUNCH(16);break; case 32: LAUNCH(32);break;
    }
#undef LAUNCH
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
