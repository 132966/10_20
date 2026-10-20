"""Opt-in, GPU-resident immutable pages shared only by active requests.

Original SGLang KV ownership is unchanged. This pool owns additional BF16 K/V,
FP8 routing copies, centers and logical maps. No unreferenced index is cached.
"""
from dataclasses import dataclass
import logging
import os
from contextlib import contextmanager
import torch
import torch.nn.functional as F
from .config import SINK, WINDOW, BUCKET, PAGE, bucket_plan, Ledger, build_graph_enabled, query_graph_enabled
from .grouping import build_bucket, pack
from . import shared_kernels as io

logger = logging.getLogger(__name__)


def capacity(config, requests):
    # One extra bucket for transactional boundary replacement.
    extra = 2 * BUCKET // PAGE if os.environ.get("SGLANG_V1_FULL_BUCKET_APPEND", "0") == "1" else 0
    gpu_extra = BUCKET // PAGE if os.environ.get("SGLANG_V1_GPU_METADATA", "0") == "1" else 0
    return (config.fine_pages + extra) * requests + BUCKET // PAGE + gpu_extra


def arena_bytes(config, heads, requests):
    count = capacity(config, requests)
    # BF16 K+V, packed FP8 fine+centers, logical IDs, center validity,
    # and one head-independent physical-index identity map for mapped TopK.
    return heads * count * (64 * 128 * 4 + 2 * 64 * 132 + 64 * 4 + 8) + count * 64 * 4


@dataclass(eq=False)
class Page:
    head: int
    index: int
    key: object
    refs: int = 1


@dataclass(eq=False)
class Bucket:
    key: object
    pages: list
    table: torch.Tensor
    refs: int = 1


class Arena:
    def __init__(self, config, heads, requests, device):
        self.capacity = capacity(config, requests)
        self.bulk_build = os.environ.get("SGLANG_V1_BULK_BUILD", "0") == "1"
        self.batch_metadata = os.environ.get("SGLANG_V1_BATCH_METADATA", "0") == "1"
        self.heads, self.device = heads, device
        shape = (heads, self.capacity, 64, 128)
        self.k = torch.empty(shape, device=device, dtype=torch.bfloat16)
        self.v = torch.empty_like(self.k)
        self.fine = torch.empty((heads, self.capacity, 64, 1, 132),
                                device=device, dtype=torch.uint8)
        # Eight useful centers per physical page. Padded layout permits the
        # unchanged paged DeepGEMM kernel; optimizing this padding is separate.
        self.centers = torch.empty_like(self.fine)
        self.valid = torch.zeros((heads, self.capacity, 8), device=device, dtype=torch.bool)
        self.mapping = torch.full((heads, self.capacity * 64), -1,
                                  device=device, dtype=torch.int32)
        self.identity = torch.arange(self.capacity * 64, device=device, dtype=torch.int32)
        self.free = [list(range(self.capacity - 1, -1, -1)) for _ in range(heads)]
        self.page_cache, self.bucket_cache = {}, {}
        self.live_pages = 0
        self.event = None
        self.stats = dict(bucket_hits=0, page_hits=0, page_builds=0, bucket_builds=0)
        self.record()  # Publish initialization to later callers on other streams.

    def wait(self):
        stream = torch.cuda.current_stream(self.device)
        if self.event is not None:
            stream.wait_event(self.event)
        for tensor in (self.k, self.v, self.fine, self.centers, self.valid,
                       self.mapping, self.identity):
            tensor.record_stream(stream)

    def record(self):
        self.event = torch.cuda.Event()
        self.event.record(torch.cuda.current_stream(self.device))

    @contextmanager
    def operation(self):
        self.wait()
        try:
            yield
        finally:
            # Also fence queued writes if a later Python operation fails.
            self.record()

    def allocate(self, head, key=None):
        if not self.free[head]:
            raise RuntimeError("Shared V1 page arena exhausted")
        self.live_pages += 1
        return Page(head, self.free[head].pop(), key)

    def release_page(self, page):
        if page.refs <= 0:
            raise RuntimeError("Shared page refcount underflow")
        page.refs -= 1
        if page.refs:
            return
        if page.key is not None:
            entries = self.page_cache[page.key]
            entries.remove(page)
            if not entries:
                del self.page_cache[page.key]
        # All users enqueue after wait(); allocator reuse is stream ordered.
        self.free[page.head].append(page.index)
        self.live_pages -= 1

    def release_bucket(self, bucket):
        if bucket.refs <= 0:
            raise RuntimeError("Shared bucket refcount underflow")
        bucket.refs -= 1
        if bucket.refs:
            return
        if bucket.key is not None:
            entries = self.bucket_cache[bucket.key]
            entries.remove(bucket)
            if not entries:
                del self.bucket_cache[bucket.key]
        for pages in bucket.pages:
            for page in pages:
                self.release_page(page)

    def write_page(self, page, kp, vp, slots, ids, centers=True):
        h, p = page.head, page.index
        io.copy_page(kp, vp, slots, ids, self.k[h, p], self.v[h, p],
                     self.mapping[h, p * 64:(p + 1) * 64], h)
        self.fine[h, p].copy_(pack(self.k[h, p])[0])
        if centers:
            counts = (ids.reshape(8, 8) >= 0).sum(-1)
            mean = self.k[h, p].float().reshape(8, 8, 128).sum(1) / counts.clamp_min(1)[:, None]
            self.centers[h, p].copy_(pack(F.pad(mean, (0, 0, 0, 56)))[0])
            self.valid[h, p].copy_(counts > 0)

    def acquire_page(self, kp, vp, slots, ids, head, members, digest, pending=None, source=0):
        key = (head, tuple(members), tuple(digest))
        for page in self.page_cache.get(key, ()):
            # Fingerprints are only an accelerator, never proof of equality.
            if io.matches_page(kp, vp, slots, self, head, page.index):
                page.refs += 1
                self.stats["page_hits"] += 1
                return page
        page = self.allocate(head)
        try:
            if pending is None:
                self.write_page(page, kp, vp, slots, ids)
            else:
                pending.append((head, source, page.index))
        except Exception:
            self.release_page(page)
            raise
        page.key = key
        self.page_cache.setdefault(key, []).append(page)
        self.stats["page_builds"] += 1
        return page

    def acquire_bucket(self, kp, vp, slots, start, end):
        whole = end - start == BUCKET
        key = None
        raw_digest = None
        if whole:
            logical = torch.arange(start, end, device=self.device, dtype=torch.int32)
            logical = logical[None].expand(self.heads, -1)
            raw_digest = io.fingerprints(kp, vp, slots, logical).flatten()
            # On a cold range no existing bucket can match: defer the GPU->CPU
            # digest transfer until grouping and page metadata are also ready.
            possible_hit = any(k[0] == start and k[1] == end for k in self.bucket_cache)
            if not self.batch_metadata or possible_hit:
                key = (start, end, tuple(raw_digest.cpu().tolist()))
            for bucket in self.bucket_cache.get(key, ()):
                bucket.table.record_stream(torch.cuda.current_stream(self.device))
                if io.matches(kp, vp, slots, self, bucket.table):
                    bucket.refs += 1
                    self.stats["bucket_hits"] += 1
                    return bucket
        # Only a miss clusters. No V snapshot and no reordered KV copy on hit.
        keys = kp[slots[start:end].long()]
        grouper = getattr(self, "grouper", None)
        if grouper is not None and whole:
            ids = grouper.run(keys, start)
        elif self.bulk_build:
            ids = build_bucket(keys, start, ids_only=True)
        else:
            ids, _, _, _ = build_bucket(keys, start)
        del keys
        if self.batch_metadata:
            page_digest = io.fingerprints(kp, vp, slots, ids)
            parts = [ids.flatten().long(), page_digest.flatten()]
            if whole and key is None:
                parts.append(raw_digest)
            metadata = torch.cat(parts).cpu()
            members = metadata[:ids.numel()].reshape(self.heads,-1,64).tolist()
            finish = ids.numel()+page_digest.numel()
            digests = metadata[ids.numel():finish].reshape(page_digest.shape).tolist()
            if whole and key is None:
                key = (start,end,tuple(metadata[finish:].tolist()))
        else:
            members = ids.reshape(self.heads, -1, 64).cpu().tolist()
            digests = io.fingerprints(kp, vp, slots, ids).cpu().tolist()
        pages = [[] for _ in range(self.heads)]
        pending = [] if self.bulk_build else None
        try:
            for h in range(self.heads):
                for p in range(len(members[h])):
                    pages[h].append(self.acquire_page(
                        kp, vp, slots, ids[h, p * 64:(p + 1) * 64],
                        h, members[h][p], digests[h][p], pending, p))
            if pending:
                from .bulk_build import write_pages
                write_pages(self, kp, vp, slots, ids, pending)
            table = torch.tensor([[p.index for p in head] for head in pages],
                                 device=self.device, dtype=torch.int32)
        except Exception:
            for head in pages:
                for page in head:
                    self.release_page(page)
            raise
        bucket = Bucket(key, pages, table)
        if whole:
            self.bucket_cache.setdefault(key, []).append(bucket)
        self.stats["bucket_builds"] += 1
        return bucket


class SharedState:
    def __init__(self, arena, config):
        self.arena, self.config, self.heads = arena, config, arena.heads
        self.full_bucket_append = os.environ.get("SGLANG_V1_FULL_BUCKET_APPEND", "0") == "1"
        self.buckets = {}
        self.fixed = [[] for _ in range(self.heads)]
        self.committed, self.ready_prefix, self.flat_until = SINK, -1, 0
        if self.full_bucket_append:
            self.committed = 0
        self.pages = 0
        self.decode_query_graphs = {}
        self.decode_query_ledger = None
        self.decode_query_drain = None
        self.query_graph_pool = None
        self.table = torch.empty((self.heads, 0), device=arena.device, dtype=torch.int32)
        self.table_storage = (torch.empty((self.heads, ((config.context+4095)//4096)*64),
            device=arena.device,dtype=torch.int32) if hasattr(arena,'grefs') else None)

    def wait(self):
        self.arena.wait()
        self.table.record_stream(torch.cuda.current_stream(self.arena.device))

    def reset(self):
        self.wait()
        # Shared graphs stage new addresses before every replay. These are
        # diagnostic aliases, not owners; another request/layer can use them.
        self.decode_query_graphs.clear()
        for bucket in self.buckets.values():
            self.arena.release_bucket(bucket)
        self.buckets.clear()
        for pages in self.fixed:
            for page in pages:
                self.arena.release_page(page)
        self.fixed = [[] for _ in range(self.heads)]
        self.committed, self.ready_prefix, self.pages = SINK, -1, 0
        if self.full_bucket_append:
            self.committed = 0
        self.table = torch.empty((self.heads, 0), device=self.arena.device, dtype=torch.int32)
        self.arena.record()

    def prepare(self, kp, slots, prefix, asynchronous=False, *, v_pool):
        # Shared publication is synchronous in this first opt-in version.
        # CUDA work stays on the caller stream; no CPU KV offload occurs.
        with self.arena.operation():
            self.wait()
            if prefix == self.ready_prefix:
                return
            if prefix < self.ready_prefix:
                self.reset()
            if self.full_bucket_append:
                committed = prefix // BUCKET * BUCKET
                plan = [(start, start + BUCKET, True)
                        for start in range(self.committed, committed, BUCKET)]
            else:
                plan, committed = bucket_plan(self.committed, prefix)
            for start, end, _ in plan:
                observer = getattr(self, "build_observer", None)
                if observer:
                    observer[0](start, end)
                new = self.arena.acquire_bucket(kp, v_pool, slots, start, end)
                if observer:
                    observer[1]()
                old = self.buckets.get(start)
                self.buckets[start] = new
                if self.table_storage is not None:
                    self.table_storage[:,start//64:end//64].copy_(new.table)
                if old is not None:
                    self.arena.release_bucket(old)
            self.committed, self.ready_prefix = committed, prefix
            if self.table_storage is not None:
                self.table = self.table_storage[:,:committed//64]
                self.pages = self.table.shape[1]
                return
            entries = [b.table for _, b in sorted(self.buckets.items())]
            for table in entries:
                table.record_stream(torch.cuda.current_stream(self.arena.device))
            self.table = (torch.cat(entries, 1) if entries else
                          torch.empty((self.heads, 0), device=kp.device, dtype=torch.int32))
            self.pages = self.table.shape[1]

    def prepare_fixed(self, kp, slots, prefix, total, *, v_pool):
        with self.arena.operation():
            self.wait()
            self.fixed_start = min(self.committed, max(SINK, prefix - WINDOW)) if self.full_bucket_append else prefix - WINDOW
            logical = torch.cat((torch.arange(min(SINK, total), device=kp.device),
                                 torch.arange(max(SINK, self.fixed_start), total, device=kp.device))).int()
            logical = F.pad(logical, (0, (-len(logical)) % 64), value=-1)
            count = len(logical) // 64
            for h in range(self.heads):
                while len(self.fixed[h]) > count:
                    self.arena.release_page(self.fixed[h].pop())
                while len(self.fixed[h]) < count:
                    self.fixed[h].append(self.arena.allocate(h))
                for p, page in enumerate(self.fixed[h]):
                    self.arena.write_page(page, kp, v_pool, slots,
                                          logical[p * 64:(p + 1) * 64], centers=False)
            return torch.tensor([[p.index for p in head] for head in self.fixed],
                                device=kp.device, dtype=torch.int32)


class SharedIndexPool:
    def __init__(self, requests, config, limit):
        self.requests, self.config = requests, config
        self.states, self.arenas = {}, {}
        self.groupers = {}
        self.ledger = Ledger(limit)
        mode = os.environ.get("SGLANG_V1_SHARED_BUILD_MODE", "inline")
        directory = os.environ.get("SGLANG_V1_INDEX_PROFILE_DIR")
        self.controller = None
        if mode != "inline" or directory:
            from .build_overlap import BuildController
            self.controller = BuildController(mode, directory)
        self.build_graph_pool = self.query_graph_pool = None
        drain = self.controller.join_all if self.controller else None
        if build_graph_enabled():
            if (os.environ.get('SGLANG_V1_GPU_METADATA','0') != '1'
                    or os.environ.get('SGLANG_V1_FULL_BUCKET_APPEND','0') != '1'):
                raise ValueError('Shared build graphs require GPU_METADATA=1 and FULL_BUCKET_APPEND=1')
            from .build_graph import BuildGraphPool
            self.build_graph_pool = BuildGraphPool(self.ledger,drain)
        if query_graph_enabled(decode=True):
            from .shared_query_graph import QueryGraphPool
            self.query_graph_pool = QueryGraphPool(self.config,self.ledger,drain)
        requests.add_native_index_listener(self)

    def get(self, slot, layer, heads, device):
        if self.controller:
            self.controller.join_layer(layer)
        generation = int(self.requests.req_generation[slot])
        key = (slot, generation, layer)
        for old in list(self.states):
            if old[0] == slot and old[1] != generation:
                self._drop(old)
        if key not in self.states:
            group_key = (heads, str(device))
            if os.environ.get("SGLANG_V1_GROUP_GRAPH", "0") == "1" and group_key not in self.groupers:
                from .group_graph import GroupGraph, RESERVATION
                self.ledger.acquire(RESERVATION)
                try:
                    self.groupers[group_key] = GroupGraph(heads, device)
                except Exception:
                    self.ledger.release(RESERVATION)
                    raise
            created_arena = layer not in self.arenas
            if created_arena:
                size = arena_bytes(self.config, heads, self.requests.size)
                gpu_metadata = os.environ.get("SGLANG_V1_GPU_METADATA", "0") == "1"
                arena_type = Arena
                if gpu_metadata:
                    if os.environ.get("SGLANG_V1_GROUP_GRAPH", "0") == "1":
                        raise ValueError('GPU metadata uses its own grouping workspace; disable GROUP_GRAPH')
                    from .gpu_owned_arena import GPUOwnedArena, scratch_bytes
                    arena_type = GPUOwnedArena
                    size += scratch_bytes(self.config, heads, self.requests.size)
                self.ledger.acquire(size)
                try:
                    arena = arena_type(self.config, heads, self.requests.size, device)
                    arena.reservation_bytes = size
                except Exception:
                    self.ledger.release(size)
                    raise
                self.arenas[layer] = arena
            arena = self.arenas[layer]
            if self.build_graph_pool is not None:
                try:
                    arena.build_graph = self.build_graph_pool.prepare(arena)
                except BaseException:
                    if created_arena:
                        del self.arenas[layer]
                        arena.wait()
                        self.ledger.release(arena.reservation_bytes)
                    if not self.states:
                        self._close_graphs()
                    raise
            arena.grouper = self.groupers.get(group_key)
            if arena.heads != heads:
                raise RuntimeError("Shared index head layout changed")
            self.states[key] = SharedState(arena, self.config)
            self.states[key].decode_query_ledger = self.ledger
            self.states[key].decode_query_drain = (
                self.controller.join_all if self.controller else None)
            self.states[key].query_graph_pool = self.query_graph_pool
        return self.states[key]

    def _drop(self, key):
        if self.controller:
            self.controller.join_layer(key[2])
        state = self.states.pop(key)
        state.reset()
        layer = key[2]
        if not any(k[2] == layer for k in self.states):
            arena = self.arenas.pop(layer)
            arena.wait()
            assert arena.live_pages == 0 and not arena.page_cache and not arena.bucket_cache
            assert not getattr(arena,'leases',{})
            validation = getattr(arena,'validation_mismatches',None)
            if validation is not None:
                # Diagnostics-only opt-in, never part of the no-D2H serving path.
                mismatches = int(validation.item())
                logger.info('GPU grouping validation layer=%s differing_positions=%s',layer,mismatches)
                if mismatches:
                    logger.info('Grouping permutation differs from reference; fused mode permits reduction-order differences')
                invalid=int(arena.validation_invalid.item())
                logger.info('GPU grouping integrity layer=%s invalid_positions=%s',layer,invalid)
            self.ledger.release(getattr(arena,'reservation_bytes',arena_bytes(self.config, arena.heads, self.requests.size)))
            if validation is not None and invalid:
                raise RuntimeError('GPU grouping lost, duplicated, or corrupted bucket members')
        logger.info("Shared V1 release slot=%s remaining_states=%s reserved_bytes=%s "
                    "bucket_hits=%s page_hits=%s live_pages=%s",
                    key[0], len(self.states), self.ledger.used,
                    state.arena.stats["bucket_hits"], state.arena.stats["page_hits"],
                    state.arena.live_pages)

    def release_request(self, slot):
        if self.controller:
            self.controller.join_request(slot)
        for key in list(self.states):
            if key[0] == slot:
                self._drop(key)
        if self.controller:
            self.controller.export(slot)
        if not self.states and self.groupers:
            from .group_graph import RESERVATION
            for grouper in self.groupers.values():
                grouper.close()
                self.ledger.release(RESERVATION)
            self.groupers.clear()
        if not self.states:
            self._close_graphs()

    def _close_graphs(self):
        for pool in (self.query_graph_pool,self.build_graph_pool):
            if pool is not None:
                pool.close()

    def clear(self):
        if self.controller:
            self.controller.join_all()
        for slot in {key[0] for key in self.states}:
            self.release_request(slot)
        self._close_graphs()

    @property
    def early_build(self):
        return self.controller is not None and self.controller.mode == "worker_early"

    def prepare_next(self, slot, layer, state, kp, vp, slots, total, stage="attention_done", kv_ready=None):
        if self.controller:
            return self.controller.submit(slot, int(self.requests.req_generation[slot]),
                                   layer, state, kp, vp, slots, total, stage, kv_ready=kv_ready)
        else:
            state.prepare(kp, slots, total, asynchronous=True, v_pool=vp)
