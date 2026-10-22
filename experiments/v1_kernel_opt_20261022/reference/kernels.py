"""FP8 page64 kernels. Use 64-bit offsets: large query scratch exceeds 2 GiB.
Native pages contain 8192 K bytes followed by 64 FP32 scales.
"""

import triton as tr
import triton.language as tl


@tr.jit
def initialize(K, KS, TABLE, META, UNIT, PERM, FLAG):
    active = tl.load(META + 2)
    if tl.program_id(0) == 0:
        tl.store(FLAG, tl.where(active != 0, -1, 0))
    if active:
        i = tl.program_id(0) * 16 + tl.arange(0, 16)
        d = tl.arange(0, 128)
        page = tl.load(TABLE + i // 64).to(tl.int64)
        scale = tl.load(KS + page * 2112 + 2048 + i % 64)
        x = (
            tl.load(K + page[:, None] * 8448 + (i % 64)[:, None] * 128 + d[None, :]).to(
                tl.float32
            )
            * scale[:, None]
        )
        norm = tl.sqrt(tl.sum(x * x, 1))
        tl.store(
            UNIT + i[:, None] * 128 + d[None, :], x / tl.maximum(norm[:, None], 1.0e-12)
        )
        tl.store(PERM + i, i)


@tr.jit
def assemble(K, KS, TABLE, META, IDS, TMP, TMPS, CENTERS, CS, MAP):
    if tl.load(META + 2):
        page, g = tl.program_id(0), tl.program_id(1)
        i = g * 8 + tl.arange(0, 8)
        d = tl.arange(0, 128)
        src = tl.load(IDS + page * 64 + i)
        native = tl.load(TABLE + src // 64).to(tl.int64)
        k = tl.load(K + native[:, None] * 8448 + (src % 64)[:, None] * 128 + d[None, :])
        scale = tl.load(KS + native * 2112 + 2048 + src % 64)
        tl.store(TMP + page * 8448 + i[:, None] * 128 + d[None, :], k)
        tl.store(TMPS + page * 2112 + 2048 + i, scale)
        tl.store(MAP + page * 64 + i, tl.load(META) + src)
        mean = tl.sum(k.to(tl.float32) * scale[:, None], 0) / 8.0
        cs = tl.exp2(
            tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(mean), 0), 1.0e-4) / 448.0))
        )
        c = page * 8 + g
        tl.store(
            CENTERS + (c // 64) * 8448 + (c % 64) * 128 + d,
            (mean / cs).to(tl.float8e4nv),
        )
        tl.store(CS + (c // 64) * 2112 + 2048 + c % 64, cs)


@tr.jit
def fingerprint(TMP, MAP, META, HASH):
    if tl.load(META + 2):
        p = tl.program_id(0)
        i = tl.arange(0, 16384)
        x = tl.load(TMP + p * 8448 + i, i < 8448, other=0).to(tl.int64)
        m = tl.load(MAP + p * 64 + i, i < 64, other=0).to(tl.int64)
        a = tl.sum(x * (i.to(tl.int64) + 1), 0) + tl.sum(m * 104729, 0)
        b = tl.sum(x * ((i.to(tl.int64) + 7) * 1000003), 0) + tl.sum(
            m * (i.to(tl.int64) + 1) * 65537, 0
        )
        tl.store(HASH + p * 2, a)
        tl.store(HASH + p * 2 + 1, b)


@tr.jit
def match_pages(
    TMP,
    MAP,
    HASH,
    META,
    FINE,
    MEMBERS,
    HASHES,
    REFS,
    MATCH,
    CAP: tl.constexpr,
    SCAN: tl.constexpr,
):
    if tl.load(META + 2):
        p = tl.program_id(0)
        j = tl.arange(0, SCAN)
        refs = tl.load(REFS + j, j < CAP, other=0)
        h0 = tl.load(HASHES + j * 2, (j < CAP) & (refs > 0), other=0)
        h1 = tl.load(HASHES + j * 2 + 1, (j < CAP) & (refs > 0), other=0)
        good = (
            (j < CAP)
            & (refs > 0)
            & (h0 == tl.load(HASH + p * 2))
            & (h1 == tl.load(HASH + p * 2 + 1))
        )
        dst = tl.min(tl.where(good, j, CAP), 0)
        found = -1
        byte = tl.arange(0, 16384)
        a = tl.load(TMP + p * 8448 + byte, byte < 8448, other=0)
        ma = tl.load(MAP + p * 64 + byte, byte < 64, other=0)
        while (dst < CAP) & (found < 0):
            b = tl.load(FINE + dst.to(tl.int64) * 8448 + byte, byte < 8448, other=0)
            mb = tl.load(MEMBERS + dst * 64 + byte, byte < 64, other=0)
            same = (
                tl.sum((a != b).to(tl.int32), 0) + tl.sum((ma != mb).to(tl.int32), 0)
                == 0
            )
            found = tl.where(same, dst, -1)
            good = good & (j != dst)
            dst = tl.min(tl.where(good, j, CAP), 0)
        tl.store(MATCH + p, found)


@tr.jit
def allocate(META, MATCH, FREE, COUNT, REFS, TABLES, NEW):
    if tl.load(META + 2):
        i = tl.arange(0, 16)
        dst = tl.load(MATCH + i)
        fresh = dst < 0
        rank = tl.cumsum(fresh.to(tl.int32), 0) - 1
        needed = tl.sum(fresh.to(tl.int32), 0)
        count = tl.load(COUNT)
        tl.device_assert(count >= needed, "V1 DeepSeek page arena exhausted")
        allocated = tl.load(FREE + count - 1 - rank, fresh, other=0)
        dst = tl.where(fresh, allocated, dst)
        tl.atomic_add(REFS + dst, 1, sem="relaxed")
        tl.store(COUNT, count - needed)
        tl.store(TABLES + tl.load(META + 1) * 16 + i, dst)
        tl.store(NEW + i, fresh)


@tr.jit
def publish(
    TMP, TC, TCS, MAP, HASH, META, TABLES, NEW, FINE, CENTERS, CS, MEMBERS, HASHES
):
    if tl.load(META + 2):
        p = tl.program_id(0)
        if tl.load(NEW + p):
            dst = tl.load(TABLES + tl.load(META + 1) * 16 + p)
            i = tl.arange(0, 16384)
            tl.store(
                FINE + dst.to(tl.int64) * 8448 + i,
                tl.load(TMP + p * 8448 + i, i < 8448, other=0),
                i < 8448,
            )
            tl.store(
                MEMBERS + dst * 64 + i,
                tl.load(MAP + p * 64 + i, i < 64, other=-1),
                i < 64,
            )
            tl.store(
                HASHES + dst * 2 + i, tl.load(HASH + p * 2 + i, i < 2, other=0), i < 2
            )
        # Centers belong to the bucket record, independently of shared fine pages.
        # Eight consecutive center pages per record enable native paged GEMM.
        x = tl.arange(0, 1024)
        src_c = p * 8 + x // 128
        dst_c = tl.load(META + 1).to(tl.int64) * 128 + src_c
        val = tl.load(TC + (src_c // 64) * 8448 + (src_c % 64) * 128 + x % 128)
        tl.store(CENTERS + (dst_c // 64) * 8448 + (dst_c % 64) * 128 + x % 128, val)
        g = tl.arange(0, 8)
        src_c = p * 8 + g
        dst_c = tl.load(META + 1).to(tl.int64) * 128 + src_c
        val = tl.load(TCS + (src_c // 64) * 2112 + 2048 + src_c % 64)
        tl.store(CS + (dst_c // 64) * 2112 + 2048 + dst_c % 64, val)


@tr.jit
def release(TABLE, REFS, FREE, COUNT):
    i = tl.arange(0, 16)
    page = tl.load(TABLE + i)
    old = tl.atomic_add(REFS + page, -1, sem="relaxed")
    dead = old == 1
    rank = tl.cumsum(dead.to(tl.int32), 0) - 1
    n = tl.sum(dead.to(tl.int32), 0)
    base = tl.atomic_add(COUNT, n, sem="relaxed")
    tl.store(FREE + base + rank, page, dead)


@tr.jit
def coarse(
    Q,
    W,
    C,
    CS,
    TABLE,
    LENS,
    SCORES,
    MAXP: tl.constexpr,
    H: tl.constexpr,
    RELU: tl.constexpr,
):
    row, block = tl.program_id(0).to(tl.int64), tl.program_id(1)
    pages = (tl.maximum(tl.load(LENS + row)-1,0) // 1024) * 16
    pp = block * 8 + tl.arange(0, 8)
    if block * 8 < pages:
        head = tl.arange(0, H)
        d = tl.arange(0, 128)
        i = tl.arange(0, 64)
        page = tl.load(TABLE + row * (MAXP // 8) + block).to(tl.int64)
        ci = page * 64 + i
        k = tl.load(
            C + (ci[None, :] // 64) * 8448 + (ci[None, :] % 64) * 128 + d[:, None]
        )
        ks = tl.load(CS + (ci // 64) * 2112 + 2048 + ci % 64)
        q = tl.load(Q + row * (H * 128) + head[:, None] * 128 + d[None, :])
        w = tl.load(W + row * H + head)
        dots = tl.dot(q, k).to(tl.float32)
        if RELU:
            dots = tl.maximum(dots, 0.0)
        scores = tl.sum(dots * w[:, None], 0) * ks
        values = tl.max(tl.reshape(scores, (8, 8)), 1)
        tl.store(
            SCORES + row * MAXP + pp,
            tl.where(pp < pages, values, -float("inf")),
            pp < MAXP,
        )
    else:
        tl.store(SCORES + row * MAXP + pp, -float("inf"), pp < MAXP)


@tr.jit
def query_lengths(
    LENS,
    FLENS,
    TAKES,
    MAXTAKE: tl.constexpr,
    TOPK: tl.constexpr,
    NUM: tl.constexpr,
    DEN: tl.constexpr,
    SINK: tl.constexpr,
    WINDOW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    n = tl.load(LENS + row)
    pages = tl.maximum(n-1,0) // 1024 * 16
    take = tl.minimum(
        pages, tl.maximum((pages * NUM + DEN - 1) // DEN, tr.cdiv(TOPK, 64))
    )
    sink = tl.minimum(SINK, n)
    tail = tl.maximum(sink, tl.minimum(tl.maximum(n-1,0) // 1024 * 1024, tl.maximum(sink, n - WINDOW)))
    tl.store(FLENS + row, take * 64 + sink + n - tail)
    tl.store(TAKES + row, take)


@tr.jit
def fine(
    Q,
    W,
    K,
    KS,
    NATIVE,
    NS,
    MEMBERS,
    RT,
    NT,
    SELECTED,
    LENS,
    TAKES,
    SCORES,
    MAXP: tl.constexpr,
    NP: tl.constexpr,
    TAKE: tl.constexpr,
    WIDTH: tl.constexpr,
    SINK: tl.constexpr,
    WINDOW: tl.constexpr,
    H: tl.constexpr,
    RELU: tl.constexpr,
):
    row, block = tl.program_id(0).to(tl.int64), tl.program_id(1)
    n = tl.load(LENS + row)
    take = tl.load(TAKES + row)
    sink = tl.minimum(n, SINK)
    tail = tl.maximum(sink, tl.minimum(tl.maximum(n-1,0) // 1024 * 1024, tl.maximum(sink, n - WINDOW)))
    count = take * 64 + sink + n - tail
    i = block * 64 + tl.arange(0, 64)
    d = tl.arange(0, 128)
    h = tl.arange(0, H)
    if block * 64 < count:
        q = tl.load(Q + row * (H * 128) + h[:, None] * 128 + d[None, :])
        w = tl.load(W + row * H + h)
        if block < take:
            rank = tl.load(SELECTED + row * TAKE + block)
            page = tl.load(RT + row * MAXP + rank).to(tl.int64)
            k = tl.load(K + page * 8448 + (i[None, :] % 64) * 128 + d[:, None])
            scales = tl.load(KS + page * 2112 + 2048 + i % 64)
            logical = tl.load(MEMBERS + page * 64 + i % 64)
            good = (logical >= sink) & (logical < tail)
        else:
            offset = i - take * 64
            logical = tl.where(offset < sink, offset, tail + offset - sink)
            valid = i < count
            native_pages = tl.load(NT + row * NP + logical // 64, valid, other=0).to(
                tl.int64
            )
            k = tl.load(
                NATIVE
                + native_pages[None, :] * 8448
                + (logical[None, :] % 64) * 128
                + d[:, None],
                valid[None, :],
                other=0.0,
            )
            scales = tl.load(
                NS + native_pages * 2112 + 2048 + logical % 64, valid, other=0.0
            )
            good = valid
        scores = tl.dot(q, k).to(tl.float32)
        if RELU:
            scores = tl.maximum(scores, 0.0)
        score = tl.sum(scores * w[:, None], 0) * scales
        tl.store(
            SCORES + row * WIDTH + i,
            tl.where(good & (logical < n), score, -float("inf")),
            i < WIDTH,
        )
    else:
        tl.store(SCORES + row * WIDTH + i, -float("inf"), i < WIDTH)


@tr.jit
def output_ids(
    RAW,
    SCORES,
    MEMBERS,
    RT,
    NT,
    SELECTED,
    LENS,
    TAKES,
    OUT,
    LOGICAL,
    FT,
    MAXP: tl.constexpr,
    NP: tl.constexpr,
    TAKE: tl.constexpr,
    WIDTH: tl.constexpr,
    TOPK: tl.constexpr,
    SINK: tl.constexpr,
    WINDOW: tl.constexpr,
    MAP_PHYSICAL: tl.constexpr = True,
    DIRECT_FINE: tl.constexpr = False,
    FP: tl.constexpr = 0,
):
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * 256 + tl.arange(0, 256)
    raw = tl.load(RAW + row * TOPK + j, j < TOPK, other=-1)
    take = tl.load(TAKES + row)
    n = tl.load(LENS + row)
    sink = tl.minimum(n, SINK)
    tail = tl.maximum(sink, tl.minimum(tl.maximum(n-1,0) // 1024 * 1024, tl.maximum(sink, n - WINDOW)))
    valid = (raw >= 0) & (raw < WIDTH) & (j < TOPK)
    if DIRECT_FINE:
        page = tl.load(FT + row * FP + raw // 64, valid & (raw < take * 64), other=0)
    else:
        rank = tl.load(
            SELECTED + row * TAKE + raw // 64, valid & (raw < take * 64), other=0
        )
        page = tl.load(RT + row * MAXP + rank, valid & (raw < take * 64), other=0)
    member = tl.load(
        MEMBERS + page * 64 + raw % 64, valid & (raw < take * 64), other=-1
    )
    offset = raw - take * 64
    logical = tl.where(
        raw < take * 64, member, tl.where(offset < sink, offset, tail + offset - sink)
    )
    score = tl.load(SCORES + row * WIDTH + raw, valid, other=-float("inf"))
    valid = valid & (score != -float("inf")) & (logical >= 0) & (logical < n)
    if MAP_PHYSICAL:
        physical = (
            tl.load(NT + row * NP + logical // 64, valid, other=0) * 64 + logical % 64
        )
        tl.store(OUT + row * TOPK + j, tl.where(valid, physical, -1), j < TOPK)
    tl.store(LOGICAL + row * TOPK + j, tl.where(valid, logical, -1), j < TOPK)


@tr.jit
def fine_table(
    RT,
    SELECTED,
    TAKES,
    LENS,
    FT,
    CAP: tl.constexpr,
    MAXP: tl.constexpr,
    TAKE: tl.constexpr,
    FP: tl.constexpr,
    SCRATCH: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, B)
    take = tl.load(TAKES + row)
    rank = tl.load(SELECTED + row * TAKE + j, (j < take) & (j < FP), other=0)
    page = tl.load(RT + row * MAXP + rank, (j < take) & (j < FP), other=0)
    # Keep even padded table entries in-bounds for native tile prefetches.
    page = tl.where(
        j < take,
        page,
        CAP + row * SCRATCH + tl.minimum(tl.maximum(j - take, 0), SCRATCH - 1),
    )
    tl.store(FT + row * FP + j, page, j < FP)


@tr.jit
def stage_tail(
    NATIVE,
    NS,
    NT,
    LENS,
    K,
    KS,
    CAP: tl.constexpr,
    NP: tl.constexpr,
    SCRATCH: tl.constexpr,
    SINK: tl.constexpr,
    WINDOW: tl.constexpr,
):
    row, p = tl.program_id(0).to(tl.int64), tl.program_id(1)
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


@tr.jit
def mask_fine(
    SCORES,
    SELECTED,
    RT,
    MEMBERS,
    LENS,
    TAKES,
    MAXP: tl.constexpr,
    TAKE: tl.constexpr,
    STRIDE: tl.constexpr,
    WIDTH: tl.constexpr,
    SINK: tl.constexpr,
    WINDOW: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    j = tl.program_id(1) * 256 + tl.arange(0, 256)
    n = tl.load(LENS + row)
    take = tl.load(TAKES + row)
    sink = tl.minimum(n, SINK)
    tail = tl.maximum(sink, tl.minimum(tl.maximum(n-1,0) // 1024 * 1024, tl.maximum(sink, n - WINDOW)))
    count = take * 64 + sink + n - tail
    fine = j < take * 64
    rank = tl.load(SELECTED + row * TAKE + j // 64, fine & (j < WIDTH), other=0)
    page = tl.load(RT + row * MAXP + rank, fine & (j < WIDTH), other=0)
    logical = tl.load(MEMBERS + page * 64 + j % 64, fine & (j < WIDTH), other=-1)
    good = (j < count) & (~fine | ((logical >= sink) & (logical < tail)))
    score = tl.load(SCORES + row * STRIDE + j, good & (j < WIDTH), other=-float("inf"))
    tl.store(SCORES + row * STRIDE + j, score, j < WIDTH)


@tr.jit
def reduce_centers(LOGITS, LENS, SCORES, MAXP: tl.constexpr, STRIDE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    p = tl.program_id(1) * 32 + tl.arange(0, 32)
    g = tl.arange(0, 8)
    pages = tl.maximum(tl.load(LENS + row)-1,0) // 1024 * 16
    values = tl.load(
        LOGITS + row * STRIDE + p[:, None] * 8 + g[None, :],
        (p < pages)[:, None],
        other=-float("inf"),
    )
    tl.store(SCORES + row * MAXP + p, tl.max(values, 1), p < MAXP)

@tr.jit
def fine_table_shared(
    RT,
    SELECTED,
    TAKES,
    LENS,
    FT,
    TAIL_ROWS,
    CAP: tl.constexpr,
    MAXP: tl.constexpr,
    TAKE: tl.constexpr,
    FP: tl.constexpr,
    SCRATCH: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, B)
    take = tl.load(TAKES + row)
    rank = tl.load(SELECTED + row * TAKE + j, (j < take) & (j < FP), other=0)
    page = tl.load(RT + row * MAXP + rank, (j < take) & (j < FP), other=0)
    # Keep even padded table entries in-bounds for native tile prefetches.
    page = tl.where(
        j < take,
        page,
        CAP + tl.load(TAIL_ROWS + row).to(tl.int64) * SCRATCH + tl.minimum(tl.maximum(j - take, 0), SCRATCH - 1),
    )
    tl.store(FT + row * FP + j, page, j < FP)



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



@tr.jit
def query_metadata(LENS, NT, FLENS, TAKES, DG_LENS, CENTER_LENS, TAIL_ROWS,
                   NP: tl.constexpr, TOPK: tl.constexpr, NUM: tl.constexpr,
                   DEN: tl.constexpr, SINK: tl.constexpr, WINDOW: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    n = tl.load(LENS + row)
    pages = tl.maximum(n-1,0) // 1024 * 16
    take = tl.minimum(pages, tl.maximum((pages * NUM + DEN - 1) // DEN, tr.cdiv(TOPK, 64)))
    sink = tl.minimum(SINK, n)
    tail = tl.maximum(sink, tl.minimum(tl.maximum(n-1,0) // 1024 * 1024, tl.maximum(sink, n - WINDOW)))
    fine = take * 64 + sink + n - tail
    tl.store(FLENS + row, fine)
    tl.store(TAKES + row, take)
    tl.store(DG_LENS + row, tl.maximum(fine, 1))
    tl.store(CENTER_LENS + row, tl.maximum(pages * 8, 1))
    # Share bytes only after comparing current device lengths AND all involved
    # physical page IDs. This remains correct after migration, growth and replay.
    i = tl.arange(0, 128)
    page = tl.where(i == 0, 0, tail // 64 + i - 1)
    valid = (n > 0) & ((i == 0) | ((i > 0) & (page < ((n + 63) // 64)))) & (page < NP)
    mine = tl.load(NT + row * NP + page, valid, other=-1)
    first = tl.load(NT + page, valid, other=-1)
    share = (n == tl.load(LENS)) & (tl.sum((mine != first).to(tl.int32), 0) == 0)
    tl.store(TAIL_ROWS + row, tl.where(share, 0, row))


@tr.jit
def expand_logical(LOGICAL, OUT, TOPK: tl.constexpr, STRIDE: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    j = tl.arange(0, BLOCK)
    logical = tl.load(LOGICAL + row * TOPK + j // 4, j < TOPK * 4, other=-1)
    value = tl.where((j < TOPK * 4) & (logical >= 0), logical * 4 + j % 4, -1)
    tl.store(OUT + row * STRIDE + j, value, j < STRIDE)
