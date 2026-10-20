"""Fixed-capacity eager/graph implementations for DeepSeek-layout V1.

Public contract: native packed FP8 cache, compact native page table, FP8 Q and
scale-folded head weights, per-query lengths. Build only complete 1024 buckets (isolated GQA benchmark specialization).
Native MLA KV is never copied. All outputs use the caller's CURRENT native table.
"""

from dataclasses import dataclass
from fractions import Fraction
import math
import threading
import torch
import triton as tr
from . import kernels as k
from ..v1_native import fused_grouping as fg
from sglang.kernels.ops.attention.dsv4.topk import plan_topk_v2, topk_transform_512_v2


@dataclass(frozen=True)
class Config:
    max_context: int = 65536  # Index positions (compressed positions for C4).
    heads: int = 32
    max_requests: int = 1
    topk: int = 512
    page_fraction: float = 0.5
    q_buckets: tuple = (1, 4, 16, 64)
    sink: int = 64
    window: int = 2048
    score_mode: str = "relu"  # DSA score rule; signed V1 routing is explicit.
    graphs: bool = True
    budget_bytes: int = 2 << 30
    query_optimizations: bool = True
    query_mask_fusion: bool = True
    query_bucket_rank_mask: bool = True
    query_direct_fine_map: bool = False  # Optional: standalone speedup was not consistent.
    query_int32_selected: bool = True
    query_tail_compaction: bool = True
    query_inplace_metadata: bool = False  # Experimental: slower on the tested SM90.

    query_output_mapping_optimization: bool = True
    expand_output: bool = False  # Return four logical positions per compressed index + 64 padding.
    query_route_table_fusion: bool = False  # Optional; no stable gain in L3/Q1024 tests.

    def __post_init__(self):
        if (
            self.max_context < 4096
            or self.max_context % 4096
            or (self.expand_output and self.max_context > (1 << 29))
            or self.max_requests < 1
            or not 0 < self.topk <= 2048
            or not 0 < self.page_fraction <= 1
            or self.heads not in (32, 64)
            or self.score_mode not in ("relu", "signed")
            or self.sink != 64
            or not 0 <= self.window <= 4096
            or not self.q_buckets
            or tuple(sorted(set(self.q_buckets))) != self.q_buckets
            or self.q_buckets[0] != 1
            or self.budget_bytes <= 0
        ):
            raise ValueError("Invalid DeepSeek V1 geometry/budget")

    @property
    def pages(self):
        return self.max_context // 64

    @property
    def ratio(self):
        return Fraction(str(self.page_fraction)).limit_denominator(1000000)

    @property
    def take(self):
        return min(
            self.pages,
            max(math.ceil(self.pages * self.page_fraction), tr.cdiv(self.topk, 64)),
        )

    @property
    def width(self):
        return tr.cdiv(self.take * 64 + 64 + 4096, 256) * 256

    @property
    def reservation(self):
        # Includes graph-private topk temporaries and allocator rounding headroom.
        cap = self.pages * self.max_requests
        arena = (
            cap * (8448 + 1056 + 256 + 32)
            + self.q_buckets[-1] * 68 * 8448
            + self.max_requests * self.pages * 4
        )
        scratch = 32 << 20
        query = sum(
            (8 << 20) + r * (self.width * 24 + self.pages * 64 + 32768)
            for r in self.q_buckets
        )
        return arena + scratch + query + (sum(self.q_buckets) * (self.topk * 4 + 64) * 4 if self.expand_output else 0)


class Execution:
    def __init__(self, device, use_graph):
        self.device = device
        self.use_graph = use_graph
        self.graph = None
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(device))
        self.uploads = []
        self.replays = 0

    def capture(self):
        if not self.use_graph:
            return
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(2):
                self.compute()
        stream.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph, stream=stream):
            self.compute()
        stream.synchronize()
        self.done.record(stream)

    def begin(self):
        torch.cuda.current_stream(self.device).wait_event(self.done)
        self.uploads.clear()

    def upload(self, dst, values):
        src = torch.tensor(values, dtype=dst.dtype, pin_memory=True)
        dst.copy_(src, non_blocking=True)
        self.uploads.append(src)

    def execute(self):
        if self.graph is None:
            self.compute()
        else:
            self.graph.replay()
        self.replays += 1

    def finish(self):
        self.done.record(torch.cuda.current_stream(self.device))

    def close(self):
        self.done.synchronize()
        if self.graph is not None:
            self.graph.reset()
        self.graph = None
        self.uploads.clear()


class Build(Execution):
    def __init__(self, owner):
        self.owner = owner
        super().__init__(owner.device, owner.config.graphs)
        d = owner.device
        self.source = torch.zeros(16, device=d, dtype=torch.int32)
        self.meta = torch.zeros(3, device=d, dtype=torch.int32)
        self.unit = torch.empty((1024, 128), device=d)
        self.perm = [torch.empty(1024, device=d, dtype=torch.int32) for _ in range(2)]
        self.anchor = torch.empty((64, 2), device=d, dtype=torch.int32)
        self.scores = torch.empty(1024, device=d)
        self.flag = torch.zeros((), device=d, dtype=torch.int32)
        self.tmp = torch.empty((16, 8448), device=d, dtype=torch.uint8)
        self.centers = torch.empty((2, 8448), device=d, dtype=torch.uint8)
        self.mapping = torch.empty((16, 64), device=d, dtype=torch.int32)
        self.hash = torch.empty((16, 2), device=d, dtype=torch.int64)
        self.matches = torch.empty(16, device=d, dtype=torch.int32)
        self.new = torch.empty(16, device=d, dtype=torch.bool)
        self.capture()  # meta.active=0: capture cannot allocate/mutate live pages.

    def compute(self):
        o = self.owner
        k.initialize[(64,)](
            o.native.view(torch.float8_e4m3fn),
            o.native.view(torch.float32),
            self.source,
            self.meta,
            self.unit,
            self.perm[0],
            self.flag,
            num_warps=4,
        )
        current = 0
        for n in (1024, 512, 256, 128, 64, 32, 16):
            for mode in (0, 1, 2):
                fg._dots[(64, 1)](
                    self.unit,
                    self.perm[current],
                    self.anchor,
                    self.scores,
                    self.flag,
                    n,
                    0,
                    mode,
                    num_warps=4,
                    enable_fp_fusion=False,
                )
                if mode < 2:
                    fg._anchor[(1024 // n, 1)](
                        self.perm[current],
                        self.anchor,
                        self.scores,
                        self.flag,
                        n,
                        mode,
                        num_warps=4,
                    )
            fg._sort[(1024 // n, 1)](
                self.perm[current],
                self.perm[1 - current],
                self.scores,
                self.flag,
                n,
                num_warps=4 if n <= 1024 else 8,
            )
            current = 1 - current
        k.assemble[(16, 8)](
            o.native.view(torch.float8_e4m3fn),
            o.native.view(torch.float32),
            self.source,
            self.meta,
            self.perm[current],
            self.tmp.view(torch.float8_e4m3fn),
            self.tmp.view(torch.float32),
            self.centers.view(torch.float8_e4m3fn),
            self.centers.view(torch.float32),
            self.mapping,
            num_warps=4,
        )
        k.fingerprint[(16,)](self.tmp, self.mapping, self.meta, self.hash, num_warps=8)
        k.match_pages[(16,)](
            self.tmp,
            self.mapping,
            self.hash,
            self.meta,
            o.fine,
            o.members,
            o.hashes,
            o.refs,
            self.matches,
            o.capacity,
            tr.next_power_of_2(o.capacity),
            num_warps=8,
        )
        k.allocate[(1,)](
            self.meta,
            self.matches,
            o.free,
            o.free_count,
            o.refs,
            o.tables,
            self.new,
            debug=True,
        )
        k.publish[(16,)](
            self.tmp,
            self.centers,
            self.centers.view(torch.float32),
            self.mapping,
            self.hash,
            self.meta,
            o.tables,
            self.new,
            o.fine,
            o.centers,
            o.centers.view(torch.float32),
            o.members,
            o.hashes,
            num_warps=8,
        )

    def run(self, table, start, lease):
        self.begin()
        self.source.copy_(table[start // 64 : start // 64 + 16])
        self.upload(self.meta, [start, lease, 1])
        self.execute()
        self.finish()
        table.record_stream(torch.cuda.current_stream(self.device))


class Query(Execution):
    def __init__(self, owner, rows):
        self.owner, self.rows = owner, rows
        c = owner.config
        d = owner.device
        super().__init__(d, c.graphs)
        self.q = torch.zeros((rows, c.heads, 128), device=d, dtype=torch.float8_e4m3fn)
        self.weights = torch.zeros((rows, c.heads), device=d)
        self.lens = torch.zeros(rows, device=d, dtype=torch.int32)
        self.native_table = torch.zeros((rows, c.pages), device=d, dtype=torch.int32)
        self.reordered_table = torch.zeros_like(self.native_table)
        self.center_table = torch.zeros(
            (rows, c.pages // 8), device=d, dtype=torch.int32
        )
        self.coarse = torch.empty((rows, c.pages), device=d)
        selected_dtype = (
            torch.int32 if c.query_int32_selected and c.query_optimizations
            and c.score_mode == "relu" and c.pages <= 4096 else torch.int64
        )
        self.selected = torch.empty((rows, c.take), device=d, dtype=selected_dtype)
        self.selected_values = torch.empty((rows, c.take), device=d)
        self.scores = torch.empty((rows, c.width), device=d)
        self.takes = torch.zeros(rows, device=d, dtype=torch.int32)
        self.fine_lens = torch.zeros_like(self.takes)
        self.plan = torch.empty((rows + 1, 2), device=d, dtype=torch.int32)
        self.raw = torch.empty((rows, c.topk), device=d, dtype=torch.int32)
        self.transformed = torch.empty_like(self.raw)
        self.output = torch.empty_like(self.raw)
        self.logical = torch.empty_like(self.raw)
        self.no_expansion = torch.empty_strided(
            (rows, 0), (0, 1), device=d, dtype=torch.int32
        )
        self.expanded = (
            torch.empty((rows, c.topk * 4 + 64), device=d, dtype=torch.int32)
            if c.expand_output else self.no_expansion
        )
        # Large candidate sets showed no stable gain from the fused epilogue;
        # use the lightweight expansion kernel there to limit TopK complexity.
        self.fuse_expansion = c.expand_output and c.width <= 32768
        self.map_physical = not (c.expand_output and c.query_output_mapping_optimization)
        self.identity = torch.arange(c.width // 64, device=d, dtype=torch.int32)[
            None
        ].expand(rows, -1)
        self.dg = None
        if c.score_mode == "relu":
            import deep_gemm

            if "vendor_norelu" in deep_gemm.__file__:
                raise RuntimeError("DSA ReLU scoring requires original DeepGEMM")
            from sglang.kernels.ops.attention.dsa.paged_mqa_logits import (
                deepgemm_paged_mqa_logits_split,
            )

            self.dg = deep_gemm
            self.dg_split = deepgemm_paged_mqa_logits_split
            self.fine_table = torch.zeros(
                (rows, c.width // 64), device=d, dtype=torch.int32
            )
            self.center_schedule = None
            self.center_lens = torch.ones((rows, 1), device=d, dtype=torch.int32)
            self.schedule = None
            self.dg_lens = torch.ones((rows, 1), device=d, dtype=torch.int32)
        self.fused = None
        self.full_query = c.query_optimizations and self.dg is not None and c.ratio >= 1
        self.fused_topk = c.query_optimizations and rows > 512
        self.fused_route = c.query_optimizations and self.dg is not None and c.pages <= 4096
        self.tail_rows = torch.arange(rows, device=d, dtype=torch.int32)
        if (self.fused_route or self.fused_topk) and not self.full_query:
            from .query_fused import module
            self.fused = module()  # compilation finishes BEFORE capture
        self.metadata_module = None
        if self.dg is not None and c.query_optimizations and c.query_inplace_metadata:
            from sglang.kernels.ops.attention.dsv4.attn import _jit_metadata_module
            self.metadata_module = _jit_metadata_module()
            sms = torch.cuda.get_device_properties(d).multi_processor_count
            self.schedule = torch.empty((sms + 1, 2), device=d, dtype=torch.int32)
            self.center_schedule = torch.empty_like(self.schedule)
        self.stage_lengths()
        self.capture()

    def stage_lengths(self):
        c = self.owner.config
        if c.query_optimizations and self.dg is not None:
            k.query_metadata[(self.rows,)](
                self.lens, self.native_table, self.fine_lens, self.takes,
                self.dg_lens, self.center_lens, self.tail_rows, c.pages, c.topk,
                c.ratio.numerator, c.ratio.denominator, c.sink, c.window,
            )
        else:
            k.query_lengths[(self.rows,)](
                self.lens,
                self.fine_lens,
                self.takes,
                c.take,
                c.topk,
                c.ratio.numerator,
                c.ratio.denominator,
                c.sink,
                c.window,
            )
        if not self.fused_topk or self.full_query:
            from sglang.kernels.ops.attention.dsv4.topk import _jit_topk_v2_module
            _jit_topk_v2_module().topk_plan(
                self.lens if self.full_query else self.fine_lens, self.plan, 0
            )
        if self.dg is not None:
            if self.full_query:
                self.dg_lens.copy_(self.lens[:, None].clamp_min(1))
            elif not c.query_optimizations:
                self.dg_lens.copy_(self.fine_lens[:, None].clamp_min(1))
            if self.metadata_module is not None:
                self.metadata_module.run(self.dg_lens.view(-1), self.schedule)
                if not self.full_query:
                    self.metadata_module.run(self.center_lens.view(-1), self.center_schedule)
                return
            schedule = self.dg.get_paged_mqa_logits_metadata(
                self.dg_lens,
                64,
                torch.cuda.get_device_properties(self.device).multi_processor_count,
            )
            if not c.query_optimizations:
                self.center_lens.copy_(((self.lens[:, None]-1).clamp_min(0) // 1024 * 128).clamp_min(1))
            center_schedule = self.dg.get_paged_mqa_logits_metadata(
                self.center_lens,
                64,
                torch.cuda.get_device_properties(self.device).multi_processor_count,
            )
            if self.center_schedule is None:
                self.center_schedule = center_schedule
            else:
                self.center_schedule.copy_(center_schedule)
            if self.schedule is None:
                self.schedule = schedule
            else:
                self.schedule.copy_(schedule)

    def compute(self):
        self.stage_lengths()
        o = self.owner
        c = o.config
        if self.full_query:
            self.scores = self.dg_split(
                self.dg.fp8_paged_mqa_logits, self.q, o.native, self.weights,
                self.dg_lens, self.native_table, self.schedule, c.max_context,
                q_offset=self.rows,
            )
            topk_transform_512_v2(
                self.scores, self.lens, self.native_table, self.output, 64,
                self.plan, self.logical,
            )
            self.expand_fallback()
            return
        if self.dg is not None:
            center_logits = self.dg_split(
                self.dg.fp8_paged_mqa_logits,
                self.q,
                o.centers,
                self.weights,
                self.center_lens,
                self.center_table,
                self.center_schedule,
                c.pages * 8,
                q_offset=self.rows,
            )
            if self.fused_route:
                self.fused.route(center_logits, self.lens, self.takes, self.selected, c.pages,
                                 self.reordered_table, self.fine_table, self.tail_rows,
                                 o.capacity, c.query_route_table_fusion)
            else:
                k.reduce_centers[(self.rows, tr.cdiv(c.pages, 32))](
                    center_logits, self.lens, self.coarse, c.pages, center_logits.stride(0)
                )
        else:
            k.coarse[(self.rows, tr.cdiv(c.pages, 8))](
                self.q,
                self.weights,
                o.centers.view(torch.float8_e4m3fn),
                o.centers.view(torch.float32),
                self.center_table,
                self.lens,
                self.coarse,
                c.pages,
                c.heads,
                c.score_mode == "relu",
                num_warps=4,
            )
        if not self.fused_route:
            torch.topk(
                self.coarse,
                c.take,
                dim=-1,
                sorted=True,
                out=(self.selected_values, self.selected),
            )
        if self.dg is not None:
            if c.query_optimizations:
                if not (self.fused_route and c.query_route_table_fusion):
                    k.fine_table_shared[(self.rows,)](
                        self.reordered_table,
                        self.selected,
                        self.takes,
                        self.lens,
                        self.fine_table,
                        self.tail_rows,
                        o.capacity,
                        c.pages,
                        c.take,
                        c.width // 64,
                        68,
                        tr.next_power_of_2(c.width // 64),
                    )
                tail_group = 32 if c.query_tail_compaction and self.rows > 512 else 1
                k.stage_tail_shared[(tr.cdiv(self.rows, tail_group), 68)](
                    o.native.view(torch.float8_e4m3fn),
                    o.native.view(torch.float32),
                    self.native_table,
                    self.lens,
                    o.fine.view(torch.float8_e4m3fn),
                    o.fine.view(torch.float32),
                    self.tail_rows,
                    o.capacity,
                    c.pages,
                    68,
                    c.sink,
                    c.window,
                    ROWS=self.rows,
                    GROUP=tail_group,
                    num_warps=4,
                )
            else:
                k.fine_table[(self.rows,)](
                    self.reordered_table,
                    self.selected,
                    self.takes,
                    self.lens,
                    self.fine_table,
                    o.capacity,
                    c.pages,
                    c.take,
                    c.width // 64,
                    68,
                    tr.next_power_of_2(c.width // 64),
                )
                k.stage_tail[(self.rows, 68)](
                    o.native.view(torch.float8_e4m3fn),
                    o.native.view(torch.float32),
                    self.native_table,
                    self.lens,
                    o.fine.view(torch.float8_e4m3fn),
                    o.fine.view(torch.float32),
                    o.capacity,
                    c.pages,
                    68,
                    c.sink,
                    c.window,
                    num_warps=4,
                )
            self.scores = self.dg_split(
                self.dg.fp8_paged_mqa_logits,
                self.q,
                o.fine,
                self.weights,
                self.dg_lens,
                self.fine_table,
                self.schedule,
                c.width,
                q_offset=self.rows,
            )
            if not (self.fused_topk and c.query_mask_fusion):
                k.mask_fine[(self.rows, tr.cdiv(c.width, 256))](
                    self.scores,
                    self.selected,
                    self.reordered_table,
                    o.members,
                    self.lens,
                    self.takes,
                    c.pages,
                    c.take,
                    self.scores.stride(0),
                    c.width,
                    c.sink,
                    c.window,
                )
        else:
            k.fine[(self.rows, c.width // 64)](
                self.q,
                self.weights,
                o.fine.view(torch.float8_e4m3fn),
                o.fine.view(torch.float32),
                o.native.view(torch.float8_e4m3fn),
                o.native.view(torch.float32),
                o.members,
                self.reordered_table,
                self.native_table,
                self.selected,
                self.lens,
                self.takes,
                self.scores,
                c.pages,
                c.pages,
                c.take,
                c.width,
                c.sink,
                c.window,
                c.heads,
                c.score_mode == "relu",
                num_warps=4,
            )
        if self.fused_topk:
            self.fused.topk(
                self.scores, self.fine_lens, o.members, self.reordered_table,
                self.native_table, self.selected, self.lens, self.takes,
                self.output, self.logical, self.raw, c.window,
                c.query_mask_fusion and self.dg is not None,
                self.expanded if self.fuse_expansion else self.no_expansion,
                self.map_physical, c.query_bucket_rank_mask,
                self.fine_table if self.dg is not None else self.reordered_table,
                c.query_direct_fine_map and self.dg is not None,
            )
            if not self.fuse_expansion:
                self.expand_fallback()
            return
        topk_transform_512_v2(
            self.scores,
            self.fine_lens,
            self.identity,
            self.transformed,
            64,
            self.plan,
            self.raw,
        )
        k.output_ids[(self.rows, tr.cdiv(c.topk, 256))](
            self.raw,
            self.scores,
            o.members,
            self.reordered_table,
            self.native_table,
            self.selected,
            self.lens,
            self.takes,
            self.output,
            self.logical,
            self.fine_table if self.dg is not None else self.reordered_table,
            c.pages,
            c.pages,
            c.take,
            self.scores.stride(0),
            c.topk,
            c.sink,
            c.window,
            MAP_PHYSICAL=self.map_physical,
            DIRECT_FINE=c.query_direct_fine_map and self.dg is not None,
            FP=self.fine_table.shape[1] if self.dg is not None else 0,
        )

        self.expand_fallback()

    def expand_fallback(self):
        if self.owner.config.expand_output:
            k.expand_logical[(self.rows,)](self.logical, self.expanded, self.owner.config.topk,
                                          self.expanded.shape[1], tr.next_power_of_2(self.expanded.shape[1]))

    def run(self, q, w, native_tables, lengths, states, return_logical=False):
        self.begin()
        n = len(q)
        self.q[:n].copy_(q)
        self.weights[:n].copy_(w)
        self.lens[:n].copy_(lengths)
        self.native_table[:n, : native_tables.shape[1]].copy_(native_tables)
        if native_tables.shape[1] < self.native_table.shape[1]:
            self.native_table[:n, native_tables.shape[1] :].zero_()
        if n < self.rows:
            self.q[n:].zero_()
            self.weights[n:].zero_()
            self.lens[n:].zero_()
            self.native_table[n:].zero_()
            self.reordered_table[n:].zero_()
            self.center_table[n:].zero_()
        row = 0
        while row < n:
            end = row + 1
            while end < n and states[end] is states[row]:
                end += 1
            self.reordered_table[row:end].copy_(
                states[row].table[None].expand(end - row, -1)
            )
            self.center_table[row:end].copy_(
                states[row].center_table[None].expand(end - row, -1)
            )
            row = end
        self.execute()
        result = (self.expanded if self.owner.config.expand_output else self.output)[:n].clone()
        if return_logical:
            result = (result, self.logical[:n].clone())
        self.finish()
        for tensor in (q, w, native_tables, lengths):
            tensor.record_stream(torch.cuda.current_stream(self.device))
        return result


class State:
    def __init__(self, owner):
        self.table = torch.zeros(
            owner.config.pages, device=owner.device, dtype=torch.int32
        )
        self.center_table = torch.zeros(
            owner.config.pages // 8, device=owner.device, dtype=torch.int32
        )
        self.leases = []
        self.length = 0


class Indexer:
    """One layer's native cache + V1 arena. Calls are serialized on a CUDA event.

    owners: hashable request identities, preferably (request_slot,generation).
    lengths_cpu: scheduler-owned lengths, must match lengths tensor; no GPU->CPU
    synchronization is introduced in the regular query path.
    """

    def __init__(self, native_cache, config=Config(), storage=None):
        if config.reservation > config.budget_bytes:
            raise ValueError(
                f"V1 reservation {config.reservation} exceeds budget {config.budget_bytes}"
            )
        if (
            native_cache.dtype != torch.uint8
            or not native_cache.is_cuda
            or not native_cache.is_contiguous()
            or native_cache.ndim not in (2, 4)
            or native_cache.numel() % 8448
            or (native_cache.ndim == 2 and native_cache.shape[1] != 8448)
            or (native_cache.ndim == 4 and native_cache.shape[1:] != (64, 1, 132))
        ):
            raise ValueError("Expected native contiguous FP8 page64 packed byte cache")
        if torch.cuda.get_device_capability(native_cache.device)[0] != 9:
            raise ValueError("Initial V1 DeepSeek kernels target SM90")
        self.config, self.native, self.device = (
            config,
            native_cache.view(-1, 64, 1, 132),
            native_cache.device,
        )
        self.capacity = config.pages * config.max_requests
        self.max_leases = self.capacity // 16
        d = self.device
        self.fine = storage['fine'] if storage is not None else torch.empty(
            (self.capacity + config.q_buckets[-1] * 68, 64, 1, 132),
            device=d, dtype=torch.uint8,
        )
        self.centers = storage['centers'] if storage is not None else torch.empty(
            (tr.cdiv(self.capacity, 8), 64, 1, 132), device=d, dtype=torch.uint8
        )
        self.members = storage['members'] if storage is not None else torch.full((self.capacity, 64), -1, device=d, dtype=torch.int32)
        if storage is not None:self.members.fill_(-1)
        self.refs = torch.zeros(self.capacity, device=d, dtype=torch.int32)
        self.free = torch.arange(self.capacity, device=d, dtype=torch.int32)
        self.free_count = torch.full((), self.capacity, device=d, dtype=torch.int32)
        self.hashes = torch.zeros((self.capacity, 2), device=d, dtype=torch.int64)
        self.tables = torch.empty((self.max_leases, 16), device=d, dtype=torch.int32)
        self.center_pages = torch.arange(
            self.max_leases * 2, device=d, dtype=torch.int32
        ).view(-1, 2)
        self.free_leases = list(range(self.max_leases - 1, -1, -1))
        self.states = {}
        self.queries = {}
        self.build = None
        self.lock = threading.RLock()
        self.closed = False
        self.done = torch.cuda.Event()
        self.done.record(torch.cuda.current_stream(d))
        self.build_capture_count = 0
        self.query_capture_count = 0

    def _begin(self):
        if self.closed:
            raise RuntimeError("Indexer is closed")
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "V1 index subgraphs must execute outside full-model capture"
            )
        if (
            threading.current_thread() is not threading.main_thread()
            and self.config.graphs
        ):
            raise RuntimeError("Use the model thread for V1 index dispatch/capture")
        torch.cuda.current_stream(self.device).wait_event(self.done)

    def _finish(self):
        self.done.record(torch.cuda.current_stream(self.device))

    def _release(self, owner):
        state = self.states.pop(owner, None)
        if state:
            for lease in state.leases:
                k.release[(1,)](
                    self.tables[lease], self.refs, self.free, self.free_count
                )
                self.free_leases.append(lease)
            state.table.record_stream(torch.cuda.current_stream(self.device))
            state.center_table.record_stream(torch.cuda.current_stream(self.device))

    @torch.inference_mode()
    def prepare(self, owner, native_table, length):
        with self.lock:
            self._begin()
            try:
                return self._prepare(owner, native_table, length)
            finally:
                self._finish()

    def _prepare(self, owner, native_table, length):
        if not 0 <= length <= self.config.max_context:
            raise ValueError("Index context capacity exceeded")
        if (
            native_table.dtype != torch.int32
            or native_table.device != self.device
            or native_table.ndim != 1
            or native_table.numel() < tr.cdiv(length, 64)
            or not native_table.is_contiguous()
        ):
            raise ValueError("Invalid compact native page table")
        state = self.states.get(owner)
        if state is not None and length < state.length:
            self._release(owner)
            state = None
        if state is None:
            if len(self.states) >= self.config.max_requests:
                raise ValueError("V1 request capacity exceeded")
            state = self.states[owner] = State(self)
        complete = length // 1024
        if complete > len(state.leases) and self.build is None:
            self.build = Build(self)
            self.build_capture_count += int(self.config.graphs)
        for bucket in range(len(state.leases), complete):
            if not self.free_leases:
                raise RuntimeError("V1 bucket records exhausted")
            lease = self.free_leases.pop()
            self.build.run(native_table, bucket * 1024, lease)
            state.leases.append(lease)
            state.table[bucket * 16 : (bucket + 1) * 16].copy_(self.tables[lease])
            state.center_table[bucket * 2 : (bucket + 1) * 2].copy_(
                self.center_pages[lease]
            )
        state.length = length
        return state

    @torch.inference_mode()
    def query(
        self,
        q,
        weights,
        native_tables,
        lengths,
        owners,
        lengths_cpu,
        return_logical=False,
    ):
        n = len(q)
        c = self.config
        if (
            q.shape != (n, c.heads, 128)
            or q.dtype != torch.float8_e4m3fn
            or not q.is_contiguous()
            or weights.shape != (n, c.heads)
            or weights.dtype != torch.float32
            or not weights.is_contiguous()
            or native_tables.ndim != 2
            or native_tables.shape[0] != n
            or native_tables.shape[1] > c.pages
            or native_tables.dtype != torch.int32
            or not native_tables.is_contiguous()
            or lengths.numel() != n
            or lengths.dtype != torch.int32
            or not lengths.is_contiguous()
            or len(owners) != n
            or len(lengths_cpu) != n
            or any(
                t.device != self.device for t in (q, weights, native_tables, lengths)
            )
        ):
            raise ValueError("Unsupported V1 FP8/page64 query inputs")
        cpu = [int(x) for x in lengths_cpu]
        if any(
            x < 0 or x > c.max_context or (x + 63) // 64 > native_tables.shape[1]
            for x in cpu
        ):
            raise ValueError("Query lengths exceed page-table/context capacity")
        with self.lock:
            self._begin()
            try:
                if not n:
                    empty = torch.empty(
                        (0, c.topk * 4 + 64 if c.expand_output else c.topk), device=self.device, dtype=torch.int32
                    )
                    return (empty, torch.empty((0,c.topk),device=self.device,dtype=torch.int32)) if return_logical else empty
                # Build to each request's maximum visible length before querying
                # earlier rows. Each row still masks to its own complete buckets.
                max_rows = {}
                for i, key in enumerate(owners):
                    if key not in max_rows or cpu[i] > cpu[max_rows[key]]:
                        max_rows[key] = i
                for key, i in max_rows.items():
                    self._prepare(key, native_tables[i], cpu[i])
                outputs = []
                for start in range(0, n, c.q_buckets[-1]):
                    end = min(n, start + c.q_buckets[-1])
                    rows = next(r for r in c.q_buckets if r >= end - start)
                    if rows not in self.queries:
                        self.queries[rows] = Query(self, rows)
                        self.query_capture_count += int(c.graphs)
                    outputs.append(
                        self.queries[rows].run(
                            q[start:end],
                            weights[start:end],
                            native_tables[start:end],
                            lengths.reshape(-1)[start:end],
                            [self.states[x] for x in owners[start:end]],
                            return_logical,
                        )
                    )
                if len(outputs) == 1:
                    return outputs[0]
                if return_logical:
                    return tuple(torch.cat([x[i] for x in outputs]) for i in (0, 1))
                return torch.cat(outputs)
            finally:
                self._finish()

    @torch.inference_mode()
    def release_request(self, owner):
        with self.lock:
            self._begin()
            try:
                self._release(owner)
            finally:
                self._finish()

    @torch.inference_mode()
    def clear(self):
        with self.lock:
            if self.closed:
                return
            self._begin()
            for key in list(self.states):
                self._release(key)
            self._finish()
            self.done.synchronize()
            if self.build:
                self.build.close()
            for graph in self.queries.values():
                graph.close()
            self.build = None
            self.queries.clear()

    def close(self):
        self.clear()
        self.closed = True
