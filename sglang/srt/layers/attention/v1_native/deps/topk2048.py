"""Use the existing universal SGLang TopK-v2 kernel; do not edit upstream."""
import torch
from pathlib import Path
from functools import lru_cache
from sglang.kernels.jit.utils import load_jit


@lru_cache(None)
def module():
    return load_jit('native_qwen_signed_topk2048_20261013', external_cuda_files=[
        str(Path(__file__).parent / 'topk_vendor/topk_v2.cuh')], cuda_wrappers=[
            ('topk_transform', 'TopKKernel::transform'), ('topk_plan', 'TopKKernel::plan')])


def topk2048(scores, need_values=True):
    assert scores.ndim == 2 and scores.is_cuda and scores.dtype == torch.float32
    rows, width = scores.shape
    assert width >= 2048
    lengths = torch.full((rows,), width, dtype=torch.int32, device=scores.device)
    # Identity physical pages preserve raw candidate-slot indices. V1 applies
    # its own candidate-slot -> original-token mapping after this operation.
    tables = torch.arange((width + 63) // 64, device=scores.device, dtype=torch.int32)
    # Kernel accepts a row stride of zero: one immutable identity table is
    # shared by all rows instead of copying it thousands of times.
    tables = tables[None].expand(rows, -1)
    out = torch.empty((rows, 2048), dtype=torch.int32, device=scores.device)
    raw = torch.empty_like(out)
    plan = lengths.new_empty(rows + 1, 2)
    module().topk_plan(lengths, plan, 0)
    module().topk_transform(scores, lengths, tables, out, 64, plan, raw)
    indices = raw.long()
    return (scores.gather(1, indices) if need_values else None), indices
