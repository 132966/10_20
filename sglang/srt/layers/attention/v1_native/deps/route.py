"""Exact stable page ranking with alternate CTA layouts (experimental).

Preserves the old 128-thread blocked-input tie order, even with different CTA
sizes. No score truncation, P specialization, or candidate changes.
"""
from functools import lru_cache
from torch.utils.cpp_extension import load_inline

@lru_cache(None)
def extension(threads=256, bits=4):
    assert threads in (64,128,256,512) and bits in (4,5,6)
    source=r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/block/block_radix_sort.cuh>
#include <cuda_runtime.h>
#include <math_constants.h>
template<int ITEMS> __global__ void route_kernel(const float* scores,int* table,
 int pages,int take,int score_stride,int table_stride,unsigned char* bitmap){
 constexpr int THREADS=THREADS_VALUE;
 constexpr int BASE_ITEMS=THREADS*ITEMS/128;
 using Sort=cub::BlockRadixSort<float,THREADS,ITEMS,int,BITS_VALUE>;
 __shared__ typename Sort::TempStorage storage;
 float values[ITEMS];int ids[ITEMS];
 if(bitmap)for(int p=threadIdx.x;p<pages;p+=THREADS)bitmap[int64_t(blockIdx.x)*pages+p]=0;
 const float* row=scores+int64_t(blockIdx.x)*score_stride;
 #pragma unroll
 for(int i=0;i<ITEMS;++i){
   int old_rank=threadIdx.x*ITEMS+i;
   int p=(old_rank/BASE_ITEMS)+(old_rank%BASE_ITEMS)*128;
   float best=-CUDART_INF_F;
   if(p<pages){
     float4 a=reinterpret_cast<const float4*>(row+p*8)[0];
     float4 b=reinterpret_cast<const float4*>(row+p*8)[1];
     best=fmaxf(fmaxf(fmaxf(a.x,a.y),fmaxf(a.z,a.w)),fmaxf(fmaxf(b.x,b.y),fmaxf(b.z,b.w)));
   }
   values[i]=best;ids[i]=p;
 }
 __syncthreads();
 Sort(storage).SortDescendingBlockedToStriped(values,ids);
 #pragma unroll
 for(int i=0;i<ITEMS;++i){int rank=threadIdx.x+i*THREADS;
   if(rank<take){table[int64_t(blockIdx.x)*table_stride+rank]=ids[i];
     if(bitmap)bitmap[int64_t(blockIdx.x)*pages+ids[i]]=1;
   }
 }
}
void route_impl(torch::Tensor s,torch::Tensor t,int64_t pages,int64_t take,torch::Tensor b){
 TORCH_CHECK(s.is_cuda()&&t.is_cuda()&&s.device()==t.device());
 TORCH_CHECK(s.scalar_type()==at::kFloat&&t.scalar_type()==at::kInt);
 TORCH_CHECK(s.dim()==2&&t.dim()==2&&s.size(0)==t.size(0));
 TORCH_CHECK(pages>THREADS_VALUE/2&&pages<=4096&&take>0&&take<=pages&&t.size(1)>=take);
 TORCH_CHECK(s.size(1)>=pages*8&&s.stride(1)==1&&s.stride(0)%4==0&&t.stride(1)==1);
 c10::cuda::CUDAGuard guard(s.device());auto stream=at::cuda::getCurrentCUDAStream();
 const int rows=s.size(0),ss=s.stride(0),ts=t.stride(0);
 unsigned char* bitmap=b.defined()?b.data_ptr<unsigned char>():nullptr;
 #define LAUNCH(I) route_kernel<I><<<rows,THREADS_VALUE,0,stream>>>(s.data_ptr<float>(),t.data_ptr<int>(),pages,take,ss,ts,bitmap)
 DISPATCH_VALUE
 #undef LAUNCH
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
void route(torch::Tensor s,torch::Tensor t,int64_t p,int64_t k){route_impl(s,t,p,k,torch::Tensor());}
void route_mask(torch::Tensor s,torch::Tensor t,int64_t p,int64_t k,torch::Tensor b){
 TORCH_CHECK(b.is_cuda()&&b.scalar_type()==at::kByte&&b.is_contiguous()&&b.size(0)==s.size(0)&&b.size(1)==p);
 route_impl(s,t,p,k,b);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("route",&route);m.def("route_mask",&route_mask);}
'''
    # At least 128 slots preserves the baseline input order. Unsupported tiny
    # page counts are deliberately left to the public old route, not padded to
    # a different sorting network and silently changing ties.
    sizes=[n for n in (128,256,512,1024,2048,4096) if n>=threads]
    dispatch=''.join(('if' if i==0 else 'else if')+f'(pages<={n}){{LAUNCH({n//threads});}}' for i,n in enumerate(sizes))
    source=source.replace('THREADS_VALUE',str(threads)).replace('BITS_VALUE',str(bits)).replace('DISPATCH_VALUE',dispatch)
    return load_inline(name=f'native_qwen_v1_route_t{threads}_b{bits}_20261013',cpp_sources='',cuda_sources=source,extra_cuda_cflags=['-O3','--use_fast_math'],verbose=False)
