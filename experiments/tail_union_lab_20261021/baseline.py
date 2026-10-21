import triton as tr
import triton.language as tl
@tr.jit
def stage_tail_shared(
    NATIVE,
    NS,
    NT,
    LENS,
    K,
    KS,
    TAIL_ROWS,
    CAP: tl.constexpr,
    NP: tl.constexpr,
    SCRATCH: tl.constexpr,
    SINK: tl.constexpr,
    WINDOW: tl.constexpr,
    ROWS: tl.constexpr = 0,
    GROUP: tl.constexpr = 1,
):
    group, p = tl.program_id(0).to(tl.int64), tl.program_id(1)
    rows = group * GROUP + tl.arange(0, GROUP)
    valid = rows < ROWS if ROWS > 0 else tl.full((GROUP,), True, tl.int1)
    owners = tl.load(TAIL_ROWS + rows, valid, other=-1)
    pending = tl.where(valid & (owners == rows), rows, 2147483647)
    row = tl.min(pending, 0)
    while row < 2147483647:
        n = tl.load(LENS + row)
        sink = tl.minimum(n, SINK)
        tail = tl.maximum(sink, tl.minimum(tl.maximum(n-1,0) // 1024 * 1024, tl.maximum(sink, n - WINDOW)))
        count = sink + n - tail
        i = p * 64 + tl.arange(0, 64)
        d = tl.arange(0, 128)
        if (p * 64 < count) | (p == 0):
            logical = tl.where(i < sink, i, tail + i - sink)
            page = tl.load(NT + row * NP + logical // 64, i < count, other=0).to(tl.int64)
            val = tl.load(
                NATIVE + page[:, None] * 8448 + (logical % 64)[:, None] * 128 + d[None, :],
                (i < count)[:, None],
                other=0.0,
            )
            scale = tl.load(NS + page * 2112 + 2048 + logical % 64, i < count, other=0.0)
            dst = CAP + row * SCRATCH + p
            tl.store(
                K + dst.to(tl.int64) * 8448 + (i % 64)[:, None] * 128 + d[None, :], val
            )
            tl.store(KS + dst * 2112 + 2048 + i % 64, scale)

        pending = tl.where(pending == row, 2147483647, pending)
        row = tl.min(pending, 0)



