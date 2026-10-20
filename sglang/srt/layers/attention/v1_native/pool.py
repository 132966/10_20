"""Persistent index ownership integrated with ReqToTokenPool lifecycle.

Original attention K/V remain in the native pool. Only a <=4096-token
bucket snapshot is read for each new/rebuilt bucket; no full-history V copy.
"""
import logging
import torch
import torch.nn.functional as F
from .config import SINK, WINDOW, PAGE, bucket_plan, Ledger
from .grouping import build_bucket, pack
from .kernels import pack_slots, append_flat

logger = logging.getLogger(__name__)


class IndexState:
    def __init__(self, config, heads, device):
        self.config, self.heads, self.device = config, heads, device
        fine_pages = config.fine_pages if config.mode == "v1" else 0
        self.fine = torch.zeros((heads, fine_pages, 64, 1, 132),
                                device=device, dtype=torch.uint8)
        self.mapping = torch.full((heads, fine_pages * PAGE), -1,
                                   device=device, dtype=torch.int32)
        capacity = config.fine_pages * PAGE if config.mode == "dsa" else 0
        self.flat_k = torch.zeros((heads, capacity, 128), device=device, dtype=torch.float8_e4m3fn)
        self.flat_scale = torch.zeros((heads, capacity), device=device, dtype=torch.float32)
        self.centers = torch.zeros((heads, config.center_pages, 64, 1, 132),
                                   device=device, dtype=torch.uint8)
        self.valid = torch.zeros((heads, config.center_pages * PAGE),
                                  device=device, dtype=torch.bool)
        self.committed, self.ready_prefix, self.flat_until = SINK, -1, 0
        self.pages = 0
        self.ready = None
        self.builder = torch.cuda.Stream(device=device) if config.async_index else None

    def wait(self):
        current = torch.cuda.current_stream(self.device)
        if self.ready is not None:
            current.wait_event(self.ready)
            self.ready = None
        for tensor in (self.fine, self.mapping, self.centers, self.valid, self.flat_k, self.flat_scale):
            tensor.record_stream(current)

    def prepare(self, k_pool, slots, prefix, asynchronous=False):
        if self.config.mode != "v1" or prefix == self.ready_prefix:
            if not asynchronous:
                self.wait()
            return
        self.wait()
        # Handles resumed/retracted requests without reusing a future index.
        if prefix < self.ready_prefix:
            self.committed = SINK
        plan, committed = bucket_plan(self.committed, prefix)
        stream = self.builder if asynchronous and self.builder is not None else None
        main = torch.cuda.current_stream(self.device)
        # All native KV reads happen on main before returning to the scheduler.
        # Async work holds copies, never slots that may be freed/reassigned.
        copies = []
        for start, end, _ in plan:
            physical = slots[start:end].long()
            copies.append((start, k_pool[physical]))
        if stream is not None:
            stream.wait_stream(main)

        def build():
            for start, keys in copies:
                ids, ordered, centers, valid = build_bucket(keys, start)
                dest = start - SINK
                center_dest = dest // 8
                length = ids.shape[1]
                cg_pad = (-centers.shape[1]) % 64
                centers = F.pad(centers, (0, 0, 0, cg_pad))
                valid = F.pad(valid, (0, cg_pad), value=False)
                for h in range(self.heads):
                    self.fine[h, dest // PAGE:(dest + length) // PAGE].copy_(pack(ordered[h]))
                    self.mapping[h, dest:dest + length].copy_(ids[h])
                    cp = pack(centers[h])
                    self.centers[h, center_dest // PAGE:center_dest // PAGE + len(cp)].copy_(cp)
                    self.valid[h, center_dest:center_dest + valid.shape[1]].copy_(valid[h])
            self.committed = committed
            self.pages = (max(0, prefix - WINDOW - SINK) + PAGE - 1) // PAGE
            self.ready_prefix = prefix

        if stream is None:
            build()
        else:
            for _, keys in copies:
                keys.record_stream(stream)
            for tensor in (self.fine, self.mapping, self.centers, self.valid):
                tensor.record_stream(stream)
            with torch.cuda.stream(stream):
                build()
                self.ready = torch.cuda.Event()
                self.ready.record(stream)

    def prepare_fixed(self, k_pool, slots, prefix, total):
        self.wait()
        logical = torch.cat((
            torch.arange(SINK, device=self.device),
            torch.arange(prefix - WINDOW, total, device=self.device),
        )).int()
        logical = F.pad(logical, (0, (-len(logical)) % PAGE), value=-1)
        first = self.config.middle_pages
        pack_slots(k_pool, slots, logical, self.fine, self.mapping, first * PAGE)
        return torch.arange(first, first + len(logical) // PAGE,
                            device=self.device, dtype=torch.int32)

    def append_flat(self, k_pool, slots, total, prefix):
        self.wait()
        # A retracted/recomputed suffix may have different values even when its
        # new total length exceeds the old cached total.
        self.flat_until = min(self.flat_until, prefix)
        append_flat(k_pool, slots, self.flat_k, self.flat_scale, self.flat_until, total)
        self.flat_until = total


class NativeIndexPool:
    def __init__(self, request_pool, config, limit):
        self.requests, self.config = request_pool, config
        self.ledger = Ledger(limit)
        self.states = {}
        request_pool.add_native_index_listener(self)

    def get(self, slot, layer, heads, device):
        generation = int(self.requests.req_generation[slot])
        key = (slot, generation, layer)
        # Generation is CPU-owned; never synchronize a device tensor here.
        if key not in self.states:
            stale = [k for k in self.states if k[0] == slot and k[1] != generation]
            for old in stale:
                self._drop(old)
            size = self.config.state_bytes(heads)
            self.ledger.acquire(size)
            try:
                self.states[key] = IndexState(self.config, heads, device)
            except Exception:
                self.ledger.release(size)
                raise
        state = self.states[key]
        if state.heads != heads:
            raise RuntimeError("Head layout changed within one request/layer")
        return state

    def _drop(self, key):
        state = self.states.pop(key)
        state.wait()  # Order allocator reuse after all background writes.
        self.ledger.release(self.config.state_bytes(state.heads))

    def release_request(self, slot):
        released = 0
        for key in list(self.states):
            if key[0] == slot:
                self._drop(key)
                released += 1
        if released:
            logger.info("Native index release slot=%s layers=%s remaining_states=%s bytes=%s",
                        slot, released, len(self.states), self.ledger.used)

    def clear(self):
        for key in list(self.states):
            self._drop(key)
