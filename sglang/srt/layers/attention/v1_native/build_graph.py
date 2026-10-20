"""One reusable complete-4K build graph per local geometry, across all layers.

Native KV and arena addresses, bucket start and lease are live GPU metadata.
Whole-bucket lookup/CPU lease ownership stay outside. Grouping, page matching,
GPU page allocation and final writes are captured, including shared-page hits.
"""
import threading
import torch
import triton as tr
import triton.language as tl
from . import fused_grouping as fg
from . import gpu_page_meta as gm
from .bulk_build import _write
from .grouping import build_bucket
from .graph_runtime import GraphEntry
from .config import shared_build_graph_bytes


@tr.jit
def ptr(P, i: tl.constexpr, dtype: tl.constexpr):
    return tl.load(P + i).to(tl.pointer_type(dtype))


@tr.jit
def init(P, M, U, O, H: tl.constexpr):
    if tl.load(M + 2):
        fg._init(ptr(P, 0, tl.bfloat16), ptr(P, 2, tl.int32), U, O,
                 ptr(P, 3, tl.int32), tl.load(M), H*128, 128)


@tr.jit
def dots(P, M, U, O, A, S, N: tl.constexpr, MODE: tl.constexpr):
    if tl.load(M + 2):
        fg._dots(U, O, A, S, ptr(P, 3, tl.int32), N, tl.load(M), MODE)


@tr.jit
def anchor(P, M, O, A, S, N: tl.constexpr, SLOT: tl.constexpr):
    if tl.load(M + 2):
        fg._anchor(O, A, S, ptr(P, 3, tl.int32), N, SLOT)


@tr.jit
def sort(P, M, O, OUT, S, N: tl.constexpr):
    if tl.load(M + 2):
        fg._sort(O, OUT, S, ptr(P, 3, tl.int32), N)


@tr.jit
def gather_reference(P, M, K, H: tl.constexpr):
    if tl.load(M + 2):
        i = tl.program_id(0)*16 + tl.arange(0, 16)
        h = tl.program_id(1)
        d = tl.arange(0, 128)
        slots = tl.load(ptr(P, 2, tl.int32) + tl.load(M) + i).to(tl.int64)
        x = tl.load(ptr(P, 0, tl.bfloat16) + slots[:, None]*H*128 + h*128 + d[None, :])
        tl.store(K + i[:, None]*H*128 + h*128 + d[None, :], x)


@tr.jit
def absolute_ids(P, M, IDS, OUT, N: tl.constexpr):
    if tl.load(M + 2):
        if tl.load(ptr(P, 3, tl.int32)) < 0:
            i = tl.program_id(0)*256 + tl.arange(0, 256)
            tl.store(OUT+i, tl.load(IDS+i, i<N, other=0)+tl.load(M), i<N)


@tr.jit
def fingerprint(P, M, IDS, H: tl.constexpr):
    if tl.load(M + 2):
        gm.fingerprint_if_needed(ptr(P,0,tl.bfloat16),ptr(P,1,tl.bfloat16),
            ptr(P,2,tl.int32),IDS,ptr(P,4,tl.int64),ptr(P,3,tl.int32),
            H*128,128,H*128,128)


@tr.jit
def match(P, M, IDS, H: tl.constexpr, CAP: tl.constexpr, GCAP: tl.constexpr, SCAN: tl.constexpr):
    if tl.load(M + 2):
        gm.page_match(ptr(P,0,tl.bfloat16),ptr(P,1,tl.bfloat16),ptr(P,2,tl.int32),
            IDS,ptr(P,4,tl.int64),ptr(P,5,tl.int64),ptr(P,6,tl.int32),
            ptr(P,13,tl.bfloat16),ptr(P,14,tl.bfloat16),ptr(P,17,tl.int32),
            ptr(P,12,tl.int32),ptr(P,3,tl.int32),CAP,GCAP,SCAN,H*128,128,H*128,128)


@tr.jit
def allocate(P, M, H: tl.constexpr, GCAP: tl.constexpr):
    if tl.load(M + 2):
        gm.allocate(ptr(P,6,tl.int32),ptr(P,7,tl.int32),ptr(P,8,tl.int32),
            ptr(P,4,tl.int64),ptr(P,5,tl.int64),ptr(P,12,tl.int32),ptr(P,3,tl.int32),
            ptr(P,9,tl.int32),ptr(P,10,tl.int32),ptr(P,11,tl.int1),tl.load(M+1),H,GCAP)


@tr.jit
def write(P, M, IDS, H: tl.constexpr, CAP: tl.constexpr):
    if tl.load(M + 2):
        _write(ptr(P,0,tl.bfloat16),ptr(P,1,tl.bfloat16),ptr(P,2,tl.int32),IDS,
            ptr(P,10,tl.int32),ptr(P,13,tl.bfloat16),ptr(P,14,tl.bfloat16),
            ptr(P,15,tl.float8e4nv),ptr(P,15,tl.float32),
            ptr(P,16,tl.float8e4nv),ptr(P,16,tl.float32),ptr(P,17,tl.int32),
            ptr(P,18,tl.int1),CAP,4096,H*128,128,H*128,128,ptr(P,11,tl.int1),True)


class BuildGraph(GraphEntry):
    def __init__(self, arena, ledger):
        self.heads, self.cap, self.gcap = arena.heads, arena.capacity, arena.gcap
        self.mode = fg.grouping_mode()
        super().__init__(arena.device, ledger, shared_build_graph_bytes(self.heads, self.mode))
        try:
            h, d, b = self.heads, self.device, self.buffers
            b['ptrs'] = torch.zeros(19, device=d, dtype=torch.int64)
            b['meta'] = torch.zeros(3, device=d, dtype=torch.int64)  # start, lease, active
            b['p0'] = torch.zeros((h,4096), device=d, dtype=torch.int32)
            b['p1'] = torch.zeros_like(b['p0'])
            if self.mode == 'reference':
                b['keys'] = torch.zeros((4096,h,128), device=d, dtype=torch.bfloat16)
            else:
                b['unit'] = torch.empty((h,4096,128), device=d, dtype=torch.float32)
                b['anchor'] = torch.empty((h,256,2), device=d, dtype=torch.int32)
                b['scores'] = torch.empty((h,4096), device=d, dtype=torch.float32)
            self.capture()  # active=0: warmup/capture cannot mutate any real arena.
        except BaseException:
            self.close()
            raise

    def compute(self):
        b, h = self.buffers, self.heads
        p, m = b['ptrs'], b['meta']
        if self.mode == 'reference':
            gather_reference[(256,h)](p,m,b['keys'],h,num_warps=4)
            ids = build_bucket(b['keys'],0,ids_only=True)
            absolute_ids[(tr.cdiv(h*4096,256),)](p,m,ids,b['p1'],h*4096)
        else:
            init[(256,h)](p,m,b['unit'],b['p0'],h,num_warps=4,enable_fp_fusion=False)
            current = 0
            for n in (4096,2048,1024,512,256,128,64,32,16):
                source = b[f'p{current}']
                for mode in (0,1,2):
                    dots[(256,h)](p,m,b['unit'],source,b['anchor'],b['scores'],n,mode,
                                  num_warps=4,enable_fp_fusion=False)
                    if mode < 2:
                        anchor[(4096//n,h)](p,m,source,b['anchor'],b['scores'],n,mode,num_warps=4)
                sort[(4096//n,h)](p,m,source,b[f'p{1-current}'],b['scores'],n,
                                  num_warps=4 if n<=1024 else 8)
                current = 1-current
        ids = b['p1']
        fingerprint[(64,h)](p,m,ids,h,num_warps=8)
        match[(64,h)](p,m,ids,h,self.cap,self.gcap,tr.next_power_of_2(self.gcap),num_warps=8)
        allocate[(h,)](p,m,h,self.gcap,debug=True)
        write[(h*64,8)](p,m,ids,h,self.cap,num_warps=4)

    def run(self, arena, k, v, slots, start, lease, return_ids=False):
        if k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
            raise ValueError('Build graph requires native BF16 K/V')
        if (k.shape[1:] != (self.heads,128) or v.shape != k.shape
                or not k.is_contiguous() or not v.is_contiguous()
                or slots.dtype != torch.int32 or not slots.is_contiguous()
                or start < 0 or start+4096 > slots.numel()):
            raise ValueError('Invalid native KV layout or 4K range for build graph')
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Index build graphs must run outside full-model capture')
        tensors = [k,v,slots,arena.flag,arena.fp,arena.fingerprints,arena.grefs,
                   arena.gfree,arena.gcount,arena.tables,arena.jobs,arena.new,arena.matches,
                   arena.k,arena.v,arena.fine,arena.centers,arena.mapping,arena.valid]
        with self.lock, torch.inference_mode():
            stream = self.begin()
            arena.wait()
            self.upload('ptrs', [t.data_ptr() for t in tensors])
            self.upload('meta', [start,lease,1])
            self.graph.replay()
            ids = self.buffers['p1'].clone() if return_ids else None
            self.finish(stream,tensors)
            arena.record()
            return ids


class BuildGraphPool:
    def __init__(self, ledger, drain=None):
        self.ledger, self.drain = ledger, drain
        self.entries = {}
        self.capture_count = 0

    def prepare(self, arena):
        key = (str(arena.device),arena.heads,arena.capacity,arena.gcap,fg.grouping_mode())
        if key not in self.entries:
            if threading.current_thread() is not threading.main_thread():
                raise RuntimeError('Build graphs must be prepared before launching index workers')
            if self.drain is not None:
                self.drain()
            with torch.inference_mode():
                self.entries[key] = BuildGraph(arena,self.ledger)
            self.capture_count += 1
        return self.entries[key]

    def close(self):
        for entry in self.entries.values():
            entry.close()
        self.entries.clear()
