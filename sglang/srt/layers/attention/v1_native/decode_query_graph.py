"""Compatibility helpers for the shared, capacity-bucketed query graph.

Graph ownership belongs to SharedIndexPool, never an individual request/head.
"""
import torch
import triton as tr
import triton.language as tl
from .config import PAGE

@tr.jit
def _mask_decode_scores(SCORES, TABLE, MAPPING, META,
                        WIDTH: tl.constexpr, SELECTED: tl.constexpr,
                        BLOCK: tl.constexpr):
    col = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    page = tl.load(TABLE + col // 64, col < WIDTH, other=0)
    logical = tl.load(MAPPING + page * 64 + col % 64, col < WIDTH, other=-1)
    position = tl.load(META)
    fixed_start = tl.load(META + 1)
    fixed_count = tl.load(META + 2)
    invalid = ((logical < 0) | (logical > position)
               | (col >= SELECTED + fixed_count * 64)
               | ((col < SELECTED) & ((logical < 64) | (logical >= fixed_start))))
    tl.store(SCORES + col, -float("inf"), (col < WIDTH) & invalid)


def mask_decode_scores(scores, table, mapping, meta, take):
    assert scores.shape[0] == 1 and scores.stride(1) == 1
    _mask_decode_scores[(tr.cdiv(scores.shape[1], 512),)](
        scores, table, mapping, meta, scores.shape[1], take * PAGE, 512)


def select_decode(scorer, q, state, head, prefix, fixed, scale):
    pool = state.query_graph_pool
    if pool is None:
        raise RuntimeError("Shared query graph pool is not enabled")
    return pool.select(scorer, q, state, head, prefix, 0,
                       prefix + len(q), fixed, scale)
