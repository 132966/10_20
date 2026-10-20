"""GPU-only KV payload IO for opt-in active-request page sharing.

Only small fingerprints/member IDs return to the CPU control plane. Full K/V
are never offloaded. Hash hits are checked bit-for-bit before sharing.
"""
import torch
import triton as tr
import triton.language as tl


@tr.jit
def _fingerprint(K, V, REQ, IDS, OUT, N: tl.constexpr, IS: tl.constexpr,
                 KS0: tl.constexpr, KS1: tl.constexpr, VS0: tl.constexpr,
                 VS1: tl.constexpr):
    page, h = tl.program_id(0), tl.program_id(1)
    i = page * 64 + tl.arange(0, 64)
    d = tl.arange(0, 128)
    logical = tl.load(IDS + h * IS + i, i < N, other=-1)
    good = (i < N) & (logical >= 0)
    slot = tl.load(REQ + logical, good, other=0)
    k = tl.load(K + slot[:, None] * KS0 + h * KS1 + d[None, :],
                good[:, None], other=0).to(tl.uint16, bitcast=True).to(tl.uint32)
    v = tl.load(V + slot[:, None] * VS0 + h * VS1 + d[None, :],
                good[:, None], other=0).to(tl.uint16, bitcast=True).to(tl.uint32)
    weight = ((i[:, None] + 1).to(tl.uint32) * 2654435761
              + (d[None, :] + 1).to(tl.uint32) * 2246822519)
    base = (h * tr.cdiv(N, 64) + page) * 4
    tl.store(OUT + base, tl.sum(tl.sum(k, 1), 0).to(tl.int64))
    tl.store(OUT + base + 1, tl.sum(tl.sum(v, 1), 0).to(tl.int64))
    tl.store(OUT + base + 2, tl.sum(tl.sum(k * weight, 1), 0).to(tl.int64))
    tl.store(OUT + base + 3, tl.sum(tl.sum(v * weight, 1), 0).to(tl.int64))


def fingerprints(k, v, slots, ids):
    assert k.dtype == v.dtype == torch.bfloat16
    assert k.stride(-1) == v.stride(-1) == 1
    heads, count = ids.shape
    out = torch.empty((heads, tr.cdiv(count, 64), 4), device=k.device, dtype=torch.int64)
    _fingerprint[(tr.cdiv(count, 64), heads)](
        k, v, slots, ids, out, count, ids.stride(0),
        k.stride(0), k.stride(1), v.stride(0), v.stride(1))
    return out


@tr.jit
def _copy_page(K, V, REQ, IDS, RK, RV, MAP, H: tl.constexpr,
               KS0: tl.constexpr, KS1: tl.constexpr,
               VS0: tl.constexpr, VS1: tl.constexpr):
    i = tl.arange(0, 64)
    d = tl.arange(0, 128)
    logical = tl.load(IDS + i)
    good = logical >= 0
    slot = tl.load(REQ + logical, good, other=0)
    k = tl.load(K + slot[:, None] * KS0 + H * KS1 + d[None, :],
                good[:, None], other=0)
    v = tl.load(V + slot[:, None] * VS0 + H * VS1 + d[None, :],
                good[:, None], other=0)
    tl.store(RK + i[:, None] * 128 + d[None, :], k)
    tl.store(RV + i[:, None] * 128 + d[None, :], v)
    tl.store(MAP + i, logical)


def copy_page(k, v, slots, ids, rk, rv, mapping, head):
    _copy_page[(1,)](k, v, slots, ids, rk, rv, mapping, head,
                    k.stride(0), k.stride(1), v.stride(0), v.stride(1))


@tr.jit
def _matches(K, V, REQ, RK, RV, MAP, TABLE, OUT,
             P: tl.constexpr, CAP: tl.constexpr,
             KS0: tl.constexpr, KS1: tl.constexpr,
             VS0: tl.constexpr, VS1: tl.constexpr):
    p, h = tl.program_id(0), tl.program_id(1)
    page = tl.load(TABLE + h * P + p)
    i = tl.arange(0, 64)
    d = tl.arange(0, 128)
    base = (h * CAP + page) * 64
    logical = tl.load(MAP + base + i)
    good = logical >= 0
    slot = tl.load(REQ + logical, good, other=0)
    k = tl.load(K + slot[:, None] * KS0 + h * KS1 + d[None, :],
                good[:, None], other=0).to(tl.uint16, bitcast=True)
    v = tl.load(V + slot[:, None] * VS0 + h * VS1 + d[None, :],
                good[:, None], other=0).to(tl.uint16, bitcast=True)
    old_k = tl.load(RK + (base + i[:, None]) * 128 + d[None, :]).to(tl.uint16, bitcast=True)
    old_v = tl.load(RV + (base + i[:, None]) * 128 + d[None, :]).to(tl.uint16, bitcast=True)
    bad = tl.sum(tl.sum(((k != old_k) | (v != old_v)).to(tl.int32), 1), 0)
    tl.store(OUT + h * P + p, bad)


def matches(k, v, slots, arena, table):
    out = torch.empty_like(table)
    _matches[(table.shape[1], table.shape[0])](
        k, v, slots, arena.k, arena.v, arena.mapping, table, out,
        table.shape[1], arena.capacity, k.stride(0), k.stride(1),
        v.stride(0), v.stride(1))
    return not bool(out.any().item())


@tr.jit
def _match_page(K, V, REQ, RK, RV, MAP, OUT, H: tl.constexpr,
                KS0: tl.constexpr, KS1: tl.constexpr,
                VS0: tl.constexpr, VS1: tl.constexpr):
    i = tl.arange(0, 64)
    d = tl.arange(0, 128)
    logical = tl.load(MAP + i)
    good = logical >= 0
    slot = tl.load(REQ + logical, good, other=0)
    k = tl.load(K + slot[:, None] * KS0 + H * KS1 + d[None, :],
                good[:, None], other=0).to(tl.uint16, bitcast=True)
    v = tl.load(V + slot[:, None] * VS0 + H * VS1 + d[None, :],
                good[:, None], other=0).to(tl.uint16, bitcast=True)
    old_k = tl.load(RK + i[:, None] * 128 + d[None, :]).to(tl.uint16, bitcast=True)
    old_v = tl.load(RV + i[:, None] * 128 + d[None, :]).to(tl.uint16, bitcast=True)
    bad = tl.sum(tl.sum(((k != old_k) | (v != old_v)).to(tl.int32), 1), 0)
    tl.store(OUT, bad)


def matches_page(k, v, slots, arena, head, page):
    out = torch.empty((), device=k.device, dtype=torch.int32)
    _match_page[(1,)](
        k, v, slots, arena.k[head, page], arena.v[head, page],
        arena.mapping[head, page * 64:(page + 1) * 64], out, head,
        k.stride(0), k.stride(1), v.stride(0), v.stride(1))
    return out.item() == 0


@tr.jit
def _gather(K, V, IDS, KG, VG, N: tl.constexpr, IS: tl.constexpr,
            CAP: tl.constexpr, H: tl.constexpr):
    i = tl.program_id(0) * 32 + tl.arange(0, 32)
    d = tl.arange(0, 128)
    index = tl.load(IDS + i // 2048 * IS + i % 2048, i < N, other=-1)
    good = (i < N) & (index >= 0) & (index < CAP * 64)
    off = ((H * CAP * 64 + index[:, None]) * 128 + d[None, :])
    k = tl.load(K + off, good[:, None], other=0)
    v = tl.load(V + off, good[:, None], other=0)
    tl.store(KG + i[:, None] * 128 + d[None, :], k, (i < N)[:, None])
    tl.store(VG + i[:, None] * 128 + d[None, :], v, (i < N)[:, None])


def gather(arena, ids, head, workspace):
    arena.wait()
    if len(ids) > workspace.rows:
        raise ValueError("Shared gather exceeds workspace capacity")
    workspace.ensure_allocated()
    stream = torch.cuda.current_stream(arena.device)
    workspace.k.record_stream(stream)
    workspace.v.record_stream(stream)
    k, v = workspace.k[:len(ids)], workspace.v[:len(ids)]
    _gather[(tr.cdiv(ids.numel(), 32),)](
        arena.k, arena.v, ids, k, v, ids.numel(), ids.stride(0), arena.capacity, head)
    arena.record()
    return k, v
