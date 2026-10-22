#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cub/cub.cuh>

template<int N>
__global__ void sort_members(const int32_t* p,int32_t* o,const float* s,const int32_t* flag) {
 if(*flag>=0)return;
 constexpr int THREADS=128,ITEMS=(N+127)/128;
 // CUB radix sort is stable: carry members as values rather than encoding
 // position into another 32 key bits. Equal keys retain input order.
 using Sort=cub::BlockRadixSort<unsigned,THREADS,ITEMS,int>;
 __shared__ typename Sort::TempStorage temp;
 unsigned keys[ITEMS];int values[ITEMS];int base=blockIdx.x*N;
 #pragma unroll
 for(int j=0;j<ITEMS;j++) {
  int i=threadIdx.x*ITEMS+j;
  unsigned key=0;int member=0;
  if(i<N) {
   float score=s[base+i];if(score==0.0f)score=0.0f;
   unsigned bits=__float_as_uint(score);
   unsigned ordered=bits^((bits&0x80000000u)?0xffffffffu:0x80000000u);
   key=ordered;member=p[base+i];
  }
  keys[j]=key;values[j]=member;
 }
 Sort(temp).SortDescending(keys,values);
 #pragma unroll
 for(int j=0;j<ITEMS;j++) {
  int i=threadIdx.x*ITEMS+j;
  if(i<N) {
   o[base+i]=values[j];
  }
 }
}
void sort_cuda(torch::Tensor p,torch::Tensor o,torch::Tensor s,torch::Tensor flag,int n) {
 TORCH_CHECK(p.is_cuda() && p.scalar_type()==torch::kInt32 && p.is_contiguous() && p.numel()>=1024,"P must have 1024 contiguous CUDA int32 entries");
 TORCH_CHECK(o.device()==p.device() && o.scalar_type()==torch::kInt32 && o.is_contiguous() && o.numel()>=1024,"invalid O");
 TORCH_CHECK(s.device()==p.device() && s.scalar_type()==torch::kFloat32 && s.is_contiguous() && s.numel()>=1024,"invalid S");
 TORCH_CHECK(flag.device()==p.device() && flag.scalar_type()==torch::kInt32 && flag.is_contiguous() && flag.numel()>=1,"invalid FLAG");
 TORCH_CHECK(p.data_ptr()!=o.data_ptr(),"P and O must be distinct non-overlapping buffers");
 c10::cuda::CUDAGuard guard(p.device());auto stream=at::cuda::getCurrentCUDAStream();
#define RUN(N) sort_members<N><<<1024/N,128,0,stream>>>(p.data_ptr<int32_t>(),o.data_ptr<int32_t>(),s.data_ptr<float>(),flag.data_ptr<int32_t>())
 switch(n){case 1024:RUN(1024);break;case 512:RUN(512);break;case 256:RUN(256);break;case 128:RUN(128);break;case 64:RUN(64);break;case 32:RUN(32);break;case 16:RUN(16);break;default:TORCH_CHECK(false,"unsupported group size");}
#undef RUN
 C10_CUDA_KERNEL_LAUNCH_CHECK();
}
