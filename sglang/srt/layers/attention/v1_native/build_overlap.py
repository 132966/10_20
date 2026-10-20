"""Opt-in build scheduling and CUDA-event timing for the shared V1 backend.

Background host workers handle shared-page hash/equality CPU dependencies;
a CUDA stream alone does not release the model thread. Default: one worker.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import time
import threading
import torch


def event():
    return torch.cuda.Event(enable_timing=True)


class BuildController:
    def __init__(self, mode, directory):
        if mode not in ("inline", "worker_late", "worker_early"):
            raise ValueError("SGLANG_V1_SHARED_BUILD_MODE must be inline/worker_late/worker_early")
        self.mode = mode
        self.workers = int(os.environ.get("SGLANG_V1_BUILD_WORKERS", "1"))
        if self.workers not in (1, 2, 4):
            raise ValueError("SGLANG_V1_BUILD_WORKERS must be 1, 2 or 4")
        self.local = threading.local()
        self.directory = Path(directory) if directory else None
        self.executor = None if mode == "inline" else ThreadPoolExecutor(
            max_workers=self.workers, thread_name_prefix="v1-index")
        self.stream = None
        self.pending = {}
        self.jobs, self.waits, self.epochs = [], [], {}

    def submit(self, slot, generation, layer, state, kp, vp, slots, total, stage, kv_ready=None):
        key = (slot, generation)
        if self.stream is None and self.executor is not None:
            self.stream = torch.cuda.Stream(device=kp.device)
        producer = event()
        # A scheduler row can be reused; keep a private GPU slot-map snapshot.
        snapshot = slots[:total].clone() if self.executor is not None else slots
        producer.record()
        self.epochs.setdefault(key, kv_ready if kv_ready is not None else producer)
        job = dict(slot=slot, generation=generation, layer=layer, total=total,
                   stage=stage, producer=producer, submitted=time.perf_counter(),
                   state=state, buckets=[], kv_ready=kv_ready)
        if self.executor is None:
            self._run(job, kp, vp, snapshot)
        else:
            if (slot, layer) in self.pending:
                raise RuntimeError("Previous build must be joined before resubmission")
            future = self.executor.submit(self._run, job, kp, vp, snapshot)
            self.pending[slot, layer] = (future, job)
        return job

    def _run(self, job, kp, vp, slots):
        with torch.inference_mode(), torch.cuda.device(kp.device):
            if self.executor is not None and self.workers > 1:
                if not hasattr(self.local, 'stream'):
                    self.local.stream = torch.cuda.Stream(device=kp.device)
                stream = self.local.stream
            else:
                stream = self.stream or torch.cuda.current_stream(kp.device)
            with torch.cuda.stream(stream):
                stream.wait_event(job["producer"])
                for tensor in (kp, vp, slots):
                    tensor.record_stream(stream)
                state = job["state"]
                start, end = event(), event()
                job["start"], job["end"] = start, end
                job["started"] = time.perf_counter()
                start.record(stream)
                def before(begin, finish):
                    record = dict(members=finish - begin, start=event(), end=event(),
                                  host_start=time.perf_counter())
                    record["start"].record(stream)
                    job["buckets"].append(record)
                def after():
                    record = job["buckets"][-1]
                    record["end"].record(stream)
                    record["host_ms"] = (time.perf_counter() - record.pop("host_start")) * 1000
                state.build_observer = (before, after) if self.directory else None
                try:
                    with torch.cuda.nvtx.range(f"V1_BUILD_L{job['layer']}_T{job['total']}"):
                        state.prepare(kp, slots, job["total"], v_pool=vp)
                finally:
                    state.build_observer = None
                    end.record(stream)
                    job["finished"] = time.perf_counter()
                    self.jobs.append(job)

    def _join(self, slot_layer, reason):
        future, job = self.pending.pop(slot_layer)
        begin, end = event(), event()
        begin.record()
        wall = time.perf_counter()
        try:
            future.result()  # Also propagates worker failures; never use stale data.
        finally:
            job["state"].wait()
            end.record()
            self.waits.append(dict(slot=job["slot"], generation=job["generation"],
                                   layer=job["layer"], total=job["total"],
                                   reason=reason, start=begin, end=end,
                                   host_ms=(time.perf_counter() - wall) * 1000))

    def join_layer(self, layer):
        # CPU page ownership is shared between requests in this layer.
        for key in list(self.pending):
            if key[1] == layer:
                self._join(key, "next_layer_use")

    def join_request(self, slot):
        errors = []
        for key in list(self.pending):
            if key[0] == slot:
                try:
                    self._join(key, "request_release")
                except Exception as exc:
                    errors.append(exc)
        if errors:
            raise errors[0]

    def join_all(self):
        errors = []
        for key in list(self.pending):
            try:
                self._join(key, "pool_clear")
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise errors[0]

    def export(self, slot):
        jobs = [j for j in self.jobs if j["slot"] == slot]
        waits = [w for w in self.waits if w["slot"] == slot]
        if jobs and self.directory:
            # Done only at request release, never per bucket for timing.
            torch.cuda.current_stream(jobs[0]["state"].arena.device).synchronize()
            self.directory.mkdir(parents=True, exist_ok=True)
            for generation in sorted({j["generation"] for j in jobs}):
                epoch = self.epochs[slot, generation]
                rows = []
                for job in jobs:
                    if job["generation"] != generation:
                        continue
                    rows.append(dict(layer=job["layer"], total=job["total"], stage=job["stage"],
                        host_ms=(job["finished"] - job["started"]) * 1000,
                        queue_ms=(job["started"] - job["submitted"]) * 1000,
                        gpu_span_ms=job["start"].elapsed_time(job["end"]),
                        gpu_start_ms=epoch.elapsed_time(job["start"]),
                        gpu_end_ms=epoch.elapsed_time(job["end"]),
                        producer_ms=epoch.elapsed_time(job["producer"]),
                        kv_ready_ms=(epoch.elapsed_time(job["kv_ready"]) if job["kv_ready"] is not None else None),
                        attention_done_ms=(epoch.elapsed_time(job["attention_done"]) if "attention_done" in job else None),
                        buckets=[dict(members=b["members"], host_ms=b["host_ms"],
                                      gpu_span_ms=b["start"].elapsed_time(b["end"]))
                                 for b in job["buckets"]]))
                joined = [dict(layer=w["layer"], total=w["total"], reason=w["reason"],
                               host_ms=w["host_ms"],
                               main_stream_wait_span_ms=w["start"].elapsed_time(w["end"]))
                          for w in waits if w["generation"] == generation]
                report = dict(mode=self.mode, slot=slot, generation=generation,
                              workers=self.workers,
                              max_total=max(j["total"] for j in rows), jobs=rows, waits=joined)
                ordered = sorted(rows, key=lambda r: (r['total'], r['layer']))
                windows = []
                for current, following in zip(ordered, ordered[1:]):
                    if (current['total'] == following['total'] and following['layer'] == current['layer'] + 1
                            and current['kv_ready_ms'] is not None and following['kv_ready_ms'] is not None):
                        begin, deadline = current['kv_ready_ms'], following['kv_ready_ms']
                        windows.append(dict(total=current['total'],layer=current['layer'],
                            kv_to_next_kv_ms=deadline-begin,
                            kv_to_attention_done_ms=current['attention_done_ms']-begin,
                            post_attention_to_next_kv_ms=deadline-current['attention_done_ms'],
                            build_complete_after_kv_ms=current['gpu_end_ms']-begin,
                            lateness_ms=max(0,current['gpu_end_ms']-deadline),
                            finished_by_next_layer=current['gpu_end_ms']<=deadline))
                report['layer_windows'] = windows
                report['window_note'] = 'Observed main-stream intervals include index-induced stalls; not an index-free compute budget. Last layer excluded.'
                (self.directory / f"index_{os.getpid()}_{slot}_{generation}_{report['max_total']}_{time.time_ns()}.json").write_text(
                    json.dumps(report, indent=2))
        # Events/job state references must not keep a finished request's arena.
        self.jobs[:] = [j for j in self.jobs if j["slot"] != slot]
        self.waits[:] = [w for w in self.waits if w["slot"] != slot]
        for key in list(self.epochs):
            if key[0] == slot:
                del self.epochs[key]
