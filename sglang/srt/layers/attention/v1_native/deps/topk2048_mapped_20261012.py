"""Experiment-local Top2048 fused original-position mapping; no x4 expansion."""
from pathlib import Path
from functools import lru_cache
import torch
from sglang.kernels.jit.utils import load_jit

@lru_cache(None)
def module():
    return load_jit('native_qwen_top2048_mapped_20261013', external_cuda_files=[
        str(Path(__file__).parent/'topk_mapped_vendor_20261012/topk_v2.cuh')],
        cuda_wrappers=[('topk_transform','TopKKernel::transform'),('topk_plan','TopKKernel::plan')])

def mapped_top2048(scores, table, original_ids, return_raw=False):
    assert original_ids.dtype==torch.int32 and original_ids.is_contiguous()
    rows,width=scores.shape
    lengths=torch.full((rows,),width,device=scores.device,dtype=torch.int32)
    out=torch.empty((rows,2048),device=scores.device,dtype=torch.int32)
    plan=lengths.new_empty(rows+1,2)
    module().topk_plan(lengths,plan,0)
    raw=torch.empty_like(out) if return_raw else None
    module().topk_transform(scores,lengths,table,out,64,plan,raw,original_ids)
    return (out,raw) if return_raw else out
