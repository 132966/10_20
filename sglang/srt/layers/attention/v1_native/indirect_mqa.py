"""Signed DeepGEMM SM90 arithmetic with replay-updatable TMA descriptors.

Like native DSA, lengths and schedule metadata are refreshed in-place before
replay. Device-resident descriptors additionally permit one query graph to serve
different layers/heads without copying their KV cache into a staging buffer.
"""
from functools import lru_cache
from pathlib import Path
import torch
from torch.utils.cpp_extension import load_inline


@lru_cache(None)
def extension():
    import deep_gemm
    include = Path(deep_gemm.__file__).parent / 'include'
    here = Path(__file__).parent / 'deps'
    source = r'''
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda.h>
#include "sm90_indirect_mqa.cuh"

static void encode(CUtensorMap* out, CUtensorMapDataType type, int rank,
                   void* ptr, uint64_t* dims, uint64_t* strides,
                   uint32_t* box, CUtensorMapSwizzle swizzle) {
  uint32_t element_strides[3] = {1,1,1};
  auto err = cuTensorMapEncodeTiled(out, type, rank, ptr, dims, strides, box,
      element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  TORCH_CHECK(err == CUDA_SUCCESS, "cuTensorMapEncodeTiled failed: ", int(err));
}

void descriptors(torch::Tensor q, torch::Tensor weights, torch::Tensor cache,
                 torch::Tensor cpu_out) {
  TORCH_CHECK(q.is_cuda() && weights.is_cuda() && cache.is_cuda());
  TORCH_CHECK(q.is_contiguous() && weights.is_contiguous() && cache.is_contiguous());
  TORCH_CHECK(q.dim()==3 && q.size(1)==32 && q.size(2)==128 && q.element_size()==1);
  TORCH_CHECK(weights.dim()==2 && weights.scalar_type()==at::kFloat && weights.size(0)==q.size(0) && weights.size(1)==32);
  TORCH_CHECK(weights.device()==q.device() && cache.device()==q.device());
  TORCH_CHECK(cache.scalar_type()==at::kByte && cache.dim()==4 && cache.size(1)==64 && cache.size(2)==1 && cache.size(3)==132);
  TORCH_CHECK(!cpu_out.is_cuda() && cpu_out.is_contiguous() && cpu_out.scalar_type()==at::kByte && cpu_out.numel()==512);
  c10::cuda::CUDAGuard guard(q.device());
  auto maps = reinterpret_cast<CUtensorMap*>(cpu_out.data_ptr());
  uint64_t q_dims[] = {128, uint64_t(q.size(0)*32)}, q_stride[] = {128};
  uint32_t q_box[] = {128, 32};
  encode(maps, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, q.data_ptr(), q_dims, q_stride, q_box, CU_TENSOR_MAP_SWIZZLE_128B);
  uint64_t k_dims[] = {128,64,uint64_t(cache.size(0))}, k_stride[] = {128,8448};
  uint32_t k_box[] = {128,64,1};
  encode(maps+1, CU_TENSOR_MAP_DATA_TYPE_UINT8, 3, cache.data_ptr(), k_dims, k_stride, k_box, CU_TENSOR_MAP_SWIZZLE_128B);
  uint64_t s_dims[] = {64,uint64_t(cache.size(0))}, s_stride[] = {8448};
  uint32_t s_box[] = {64,1};
  encode(maps+2, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, (char*)cache.data_ptr()+8192, s_dims, s_stride, s_box, CU_TENSOR_MAP_SWIZZLE_NONE);
  uint64_t w_dims[] = {32,uint64_t(q.size(0))}, w_stride[] = {128};
  uint32_t w_box[] = {32,1};
  encode(maps+3, CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 2, weights.data_ptr(), w_dims, w_stride, w_box, CU_TENSOR_MAP_SWIZZLE_NONE);
}

void launch(torch::Tensor maps, torch::Tensor lengths, torch::Tensor table,
            torch::Tensor schedule, torch::Tensor output, int64_t sms) {
  TORCH_CHECK(maps.is_cuda() && maps.scalar_type()==at::kByte && maps.numel()==512);
  TORCH_CHECK(lengths.scalar_type()==at::kInt && table.scalar_type()==at::kInt && schedule.scalar_type()==at::kInt);
  TORCH_CHECK(output.is_cuda() && output.dim()==2 && table.dim()==2);
  TORCH_CHECK(output.scalar_type()==at::kFloat && output.stride(1)==1 && table.stride(1)==1);
  TORCH_CHECK(lengths.device()==output.device() && table.device()==output.device()
              && schedule.device()==output.device() && maps.device()==output.device());
  TORCH_CHECK(maps.is_contiguous() && lengths.is_contiguous() && schedule.is_contiguous());
  TORCH_CHECK(output.size(0)>0 && table.size(0)==output.size(0)
              && lengths.numel()==output.size(0));
  TORCH_CHECK(output.size(1)%256==0 && output.stride(0)>=output.size(1)
              && output.size(1)>=table.size(1)*64,
              "Paged MQA output must cover full 256-key scheduling tiles");
  TORCH_CHECK(sms>0);
  c10::cuda::CUDAGuard guard(output.device());
  auto kernel = deep_gemm::sm90_indirect_paged_mqa_logits<1,32,128,64,true,false,3,3,256,128,512,float>;
  constexpr int smem = 131072;
  C10_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
  kernel<<<sms,640,smem,at::cuda::getCurrentCUDAStream()>>>(
      output.size(0), output.stride(0), table.stride(0),
      reinterpret_cast<uint32_t*>(lengths.data_ptr<int>()), output.data_ptr<float>(),
      reinterpret_cast<uint32_t*>(table.data_ptr<int>()), nullptr,
      reinterpret_cast<uint32_t*>(schedule.data_ptr<int>()),
      reinterpret_cast<cute::TmaDescriptor*>(maps.data_ptr()));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {m.def("descriptors",&descriptors);m.def("launch",&launch);}
'''
    return load_inline(name='v1_indirect_signed_mqa_20261014', cpp_sources='',
                       cuda_sources=source, extra_include_paths=[str(include), str(here)],
                       extra_cuda_cflags=['-O3', '-std=c++17', '--expt-relaxed-constexpr',
                                          '-gencode=arch=compute_90a,code=sm_90a'],
                       extra_ldflags=['-lcuda'], verbose=False)


def descriptors(q, weights, cache):
    cpu = torch.empty(512, dtype=torch.uint8, pin_memory=True)
    extension().descriptors(q, weights, cache, cpu)
    return cpu
