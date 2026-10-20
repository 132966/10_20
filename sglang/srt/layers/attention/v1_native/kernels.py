"""Slot-aware IO. Full-history BF16 K/V is never materialized for routing."""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _pack_slots(K, REQ, LOGICAL, CACHE8, CACHE32, MAP,
                COUNT: tl.constexpr, DEST: tl.constexpr, HEADS: tl.constexpr,
                KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
                CS: tl.constexpr, MS: tl.constexpr):
    i, h = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    logical = tl.load(LOGICAL + i)
    valid = logical >= 0
    physical = tl.load(REQ + logical, valid, other=0)
    x = tl.load(K + physical * KS0 + h * KS1 + d * KS2, valid, other=0).to(tl.float32)
    maximum = tl.maximum(tl.max(tl.abs(x), 0), 1.e-4)
    scale = tl.exp2(tl.ceil(tl.log2(maximum / 448.)))
    dst = DEST + i
    page, slot = dst // 64, dst % 64
    tl.store(CACHE8 + h * CS + page * 8448 + slot * 128 + d,
             (x / scale).to(tl.float8e4nv))
    tl.store(CACHE32 + h * (CS // 4) + page * 2112 + 2048 + slot, scale)
    tl.store(MAP + h * MS + dst, logical)


def pack_slots(k_pool, request_slots, logical_ids, cache, mapping, dest=0):
    assert k_pool.ndim == 3 and k_pool.shape[-1] == 128
    assert logical_ids.is_contiguous() and request_slots.stride(0) == 1
    if logical_ids.numel() == 0:
        return
    if dest + logical_ids.numel() > mapping.shape[1]:
        raise ValueError("Index write exceeds reserved capacity")
    _pack_slots[(logical_ids.numel(), k_pool.shape[1])](
        k_pool, request_slots, logical_ids,
        cache.view(torch.float8_e4m3fn), cache.view(torch.float32), mapping,
        logical_ids.numel(), dest, k_pool.shape[1], *k_pool.stride(),
        cache.stride(0), mapping.stride(0), num_warps=4)


@tr.jit
def _mask_scores(S, TABLE, MAP, POS, WIDTH: tl.constexpr,
                 SS: tl.constexpr, TS: tl.constexpr, BLOCK: tl.constexpr,
                 DEDUP_COLS: tl.constexpr = 0, FIXED_START: tl.constexpr = 0):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    page = tl.load(TABLE + row * TS + col // 64, col < WIDTH, other=0)
    logical = tl.load(MAP + page * 64 + col % 64, col < WIDTH, other=-1)
    pos = tl.load(POS + row)
    tl.store(S + row * SS + col, -float("inf"),
             (col < WIDTH) & ((logical < 0) | (logical > pos) |
              ((col < DEDUP_COLS) & ((logical < 64) | (logical >= FIXED_START)))))


def mask_scores(scores, table, mapping, positions, dedup_cols=0, fixed_start=0):
    _mask_scores[(len(scores), tr.cdiv(scores.shape[1], 512))](
        scores, table, mapping, positions, scores.shape[1],
        scores.stride(0), table.stride(0), 512, dedup_cols, fixed_start)


@tr.jit
def _gather_slots(K, V, REQ, IDS, KG, VG, ROWS: tl.constexpr, TOP: tl.constexpr,
                  HEAD: tl.constexpr, KS0: tl.constexpr, KS1: tl.constexpr,
                  KS2: tl.constexpr, VS0: tl.constexpr, VS1: tl.constexpr,
                  VS2: tl.constexpr, IS0: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = tl.arange(0, 128)
    logical = tl.load(IDS + (t // TOP) * IS0 + t % TOP, t < ROWS * TOP, other=-1)
    valid = (t < ROWS * TOP) & (logical >= 0)
    physical = tl.load(REQ + logical, valid, other=0)
    k = tl.load(K + physical[:, None] * KS0 + HEAD * KS1 + d[None, :] * KS2,
                valid[:, None], other=0)
    v = tl.load(V + physical[:, None] * VS0 + HEAD * VS1 + d[None, :] * VS2,
                valid[:, None], other=0)
    tl.store(KG + t[:, None] * 128 + d[None, :], k, (t < ROWS * TOP)[:, None])
    tl.store(VG + t[:, None] * 128 + d[None, :], v, (t < ROWS * TOP)[:, None])


class GatherWorkspace:
    """Backend-owned bounded scratch, reused across requests/heads/layers."""
    def __init__(self, rows, device, dtype):
        self.rows, self.device, self.dtype = rows, device, dtype
        self.k = self.v = None

    def release_request(self, slot):
        # Shared scratch has no request-owned contents and can be reused.
        pass

    def clear(self):
        # ReqToTokenPool.clear is also reached by framework KV memory release.
        # Sparse attention runs eagerly, so no captured graph owns these pointers.
        self.k = self.v = None

    def ensure_allocated(self):
        if self.k is None:
            k = torch.empty((self.rows, 2048, 1, 128),
                            device=self.device, dtype=self.dtype)
            v = torch.empty_like(k)
            self.k, self.v = k, v

    def gather(self, k_pool, v_pool, request_slots, ids, head):
        rows = len(ids)
        if rows > self.rows:
            raise ValueError("Gather workspace microbatch exceeded")
        self.ensure_allocated()
        k, v = self.k[:rows], self.v[:rows]
        _gather_slots[(tr.cdiv(rows * 2048, 32),)](
            k_pool, v_pool, request_slots, ids, k, v, rows, 2048, head,
            *k_pool.stride(), *v_pool.stride(), ids.stride(0), 32, num_warps=4)
        return k, v


@tr.jit
def _append_flat(K, REQ, OUT, SCALE, START: tl.constexpr,
                 KS0: tl.constexpr, KS1: tl.constexpr, KS2: tl.constexpr,
                 CAP: tl.constexpr):
    logical, h = START + tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, 128)
    physical = tl.load(REQ + logical)
    x = tl.load(K + physical * KS0 + h * KS1 + d * KS2).to(tl.float32)
    scale = tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(x), 0), 1.e-4) / 448.)))
    tl.store(OUT + (h * CAP + logical) * 128 + d, (x / scale).to(tl.float8e4nv))
    tl.store(SCALE + h * CAP + logical, scale)


def append_flat(k_pool, request_slots, k8, scales, start, end):
    if end > start:
        _append_flat[(end - start, k_pool.shape[1])](
            k_pool, request_slots, k8, scales, start,
            *k_pool.stride(), scales.shape[1], num_warps=4)
