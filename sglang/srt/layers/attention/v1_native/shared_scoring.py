"""Original V1 math, but page tables address shared physical index pages."""
import torch
from .grouping import quantize
from .kernels import mask_scores
from .deps.route import extension
from .deps.topk2048_mapped_20261012 import mapped_top2048


def select(scorer, q, state, head, prefix, offset, total, fixed, scale):
    arena = state.arena
    state.wait()
    fixed.record_stream(torch.cuda.current_stream(q.device))
    rows, qheads, dim = q.shape
    if (qheads, dim) != (4, 128):
        raise ValueError("Expected four Q heads of dimension 128")
    small, qs = quantize(q)
    q8 = torch.zeros((rows, 32, dim), device=q.device, dtype=small.dtype)
    weights = torch.zeros((rows, 32), device=q.device, dtype=torch.float32)
    q8[:, :4] = small
    weights[:, :4] = qs * (scale / 4)
    positions = torch.arange(prefix + offset, prefix + offset + rows,
                             device=q.device, dtype=torch.int32)
    ids = select_tensors(scorer, q8, weights, arena, head, state.table[head],
                         fixed[head], positions, state.full_bucket_append,
                         state.fixed_start)
    arena.record()
    return ids


def select_tensors(scorer, q8, weights, arena, head, page_ids, fixed_ids,
                   positions, full_bucket_append, fixed_start, decode_meta=None):
    """Query kernels only; stream dependencies and lifecycle stay with caller."""
    rows = len(q8)
    pages, take = page_ids.numel(), scorer.config.budget(page_ids.numel())
    selected = torch.empty((rows, take), device=q8.device, dtype=torch.int32)
    if take == pages:
        selected[:] = torch.arange(pages, device=q8.device, dtype=torch.int32)
    else:
        ct = page_ids[None].expand(rows, -1).contiguous()
        coarse = scorer.paged(q8, weights, arena.centers[head], ct, pages * 64)
        coarse = coarse.reshape(rows, pages, 64)[:, :, :8].contiguous()
        valid = arena.valid[head, page_ids.long()]
        coarse.masked_fill_(~valid[None], -torch.inf)
        coarse = coarse.reshape(rows, pages * 8)
        if 32 < pages <= 4096:
            extension(64, 4).route(coarse, selected, pages, take)
        else:
            selected[:] = coarse.reshape(rows, pages, 8).amax(-1).topk(take, sorted=False).indices.int()
    table = torch.cat((page_ids[selected.long()],
                       fixed_ids[None].expand(rows, -1)), 1).contiguous()
    scores = scorer.paged(q8, weights, arena.fine[head], table, table.shape[1] * 64)
    if decode_meta is None:
        mask_scores(scores, table, arena.mapping[head], positions,
                    take * 64 if full_bucket_append else 0,
                    fixed_start if full_bucket_append else 0)
    else:
        from .decode_query_graph import mask_decode_scores
        mask_decode_scores(scores, table, arena.mapping[head], decode_meta, take)
    # Return physical reordered storage indices, not original logical IDs.
    # Causal checks above still use original logical positions.
    ids = mapped_top2048(scores, table, arena.identity)
    return ids
