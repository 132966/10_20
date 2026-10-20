"""Capture-safe native query fusions, adapted from the previously tuned V1 kernels."""
from sglang.kernels.jit.utils import cache_once, load_jit

@cache_once
def module():
    return load_jit(
        'v1_gqa_bucket1k_query_fused',
        cuda_files=['deepseek_v4/v1_query.cuh'],
        cuda_wrappers=[('route', 'V1QueryKernel::route'), ('topk', 'V1QueryKernel::topk')],
    )
