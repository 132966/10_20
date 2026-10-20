"""Native sparse prefill and length-gated V1 decode, with native KV ownership.

No monkey-patching, AST rewrites, or full-history BF16 gather. Persistent
index state is owned by the backend and released by ReqToTokenPool hooks.
"""
import logging
import os
import torch
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.runtime_context import get_parallel
from .config import (Config, reserve_bytes, local_index_geometry, decode_query_graph_budget,
                     build_graph_enabled, query_graph_enabled, shared_build_graph_bytes)
from .kernels import GatherWorkspace
from .pool import NativeIndexPool
from .scoring import Scorer

logger = logging.getLogger(__name__)


class NativeSparseBackend(FlashAttentionBackend):
    def __init__(self, runner, mode):
        args = runner.server_args
        # Fail closed: these need different lifecycle/attention contracts.
        rejected = {
            "MLA": runner.use_mla_backend,
            "encoder-decoder": runner.model_config.is_encoder_decoder,
            "disaggregated serving": getattr(args, "disaggregation_mode", "null") not in ("null", None),
            "context parallelism": getattr(args, "attn_cp_size", 1) != 1,
            "DP attention padding": bool(getattr(args, "enable_dp_attention", False)),
            "decode context parallelism": getattr(args, "dcp_size", 1) != 1,
            "two-batch overlap": bool(getattr(args, "enable_two_batch_overlap", False)),
            "overlapped PP microbatches": getattr(args, "pp_async_batch_depth", 0) > 0,
            "PD multiplexing": bool(getattr(args, "enable_pdmux", False)),
        }
        for feature, active in rejected.items():
            if active:
                raise ValueError(f"Native Qwen sparse backend does not support {feature}")
        # Dynamic index construction is an eager attention graph boundary.
        # Breakable/piecewise graphs capture the rest of the model, not indexes.
        graph = args.cuda_graph_config.prefill.backend
        if getattr(graph, "value", graph) == "full":
            raise ValueError("Native sparse needs breakable/tc_piecewise prefill graphs, not full")
        super().__init__(runner, fa_impl_ver=3)
        if self.use_sliding_window_kv_pool or self.kv_cache_dtype != torch.bfloat16:
            raise ValueError("Native sparse currently requires ordinary BF16 GQA KV pool")
        self.config = Config(mode, self.max_context_len, int(args.chunked_prefill_size),
                             fraction=float(os.environ.get("SGLANG_V1_PAGE_FRACTION", ".5")),
                             microbatch=int(os.environ.get("SGLANG_V1_QUERY_TILE", "1024")),
                             decode_threshold=int(os.environ.get("SGLANG_V1_DECODE_THRESHOLD", "524288")),
                             share_active_pages=os.environ.get("SGLANG_V1_SHARE_ACTIVE_PAGES", "0") == "1",
                             async_index=os.environ.get("SGLANG_V1_ASYNC_INDEX", "1") == "1")
        if mode == "v1":
            self.needs_cpu_seq_lens = True
            if (self.max_context_len > self.config.decode_threshold + 1
                    and args.get_attention_backends() != ("qwen_v1_native", "qwen_v1_native")):
                raise ValueError("Long V1 decode requires qwen_v1_native for both prefill and decode")
            if args.speculative_algorithm and self.max_context_len > self.config.decode_threshold + 1:
                raise ValueError("V1 decode above 512K does not yet support speculative tree verification")
        # Match admission capacity to reserved persistent memory, before serving.
        hf = runner.model_config.hf_config
        tp = get_parallel().attn_tp_size
        # PP stages own disjoint layers. KV offload changes physical slots but
        # not this rank-local index capacity or its logical position mapping.
        heads, local_layers = local_index_geometry(
            hf.num_attention_heads, hf.num_key_value_heads,
            getattr(hf, "head_dim", None) or hf.hidden_size // hf.num_attention_heads,
            tp, self.token_to_kv_pool.layer_num)
        if heads != runner.model_config.get_num_kv_heads(tp):
            raise ValueError("Model KV head partition differs from native index partition")
        required = self.config.state_bytes(heads) * local_layers * self.req_to_token_pool.size
        if self.config.share_active_pages:
            from .shared_pool import SharedIndexPool, arena_bytes
            required = arena_bytes(self.config, heads, self.req_to_token_pool.size) * local_layers
            if os.environ.get("SGLANG_V1_GPU_METADATA", "0") == "1":
                from .gpu_owned_arena import scratch_bytes
                required += scratch_bytes(
                    self.config, heads, self.req_to_token_pool.size) * local_layers
        if (build_graph_enabled() or query_graph_enabled(decode=True)) and (
                mode != 'v1' or not self.config.share_active_pages
                or os.environ.get('SGLANG_V1_FULL_BUCKET_APPEND','0') != '1'):
            raise ValueError('Index graphs require shared V1 and FULL_BUCKET_APPEND=1')
        if build_graph_enabled() and os.environ.get('SGLANG_V1_GPU_METADATA','0') != '1':
            raise ValueError('Build graphs require GPU_METADATA=1')
        build_graph_required = 0
        if build_graph_enabled():
            from .fused_grouping import grouping_mode
            build_graph_required = shared_build_graph_bytes(heads,grouping_mode())
            required += build_graph_required
        query_graph_required = decode_query_graph_budget(
            self.config, heads, local_layers, self.req_to_token_pool.size,
            enabled=query_graph_enabled(decode=True),
            full_bucket_append=os.environ.get("SGLANG_V1_FULL_BUCKET_APPEND", "0") == "1")
        required += query_graph_required
        if required > reserve_bytes():
            raise ValueError(f"Index reservation too small for admission capacity: "
                             f"need {required / 2**30:.3f} GiB "
                             f"(including {query_graph_required / 2**30:.3f} GiB "
                             f"for shared query graphs, {build_graph_required / 2**30:.3f} GiB "
                             "for shared build graphs); increase "
                             "SGLANG_V1_INDEX_POOL_GIB or reduce --max-running-requests")
        pool_type = SharedIndexPool if self.config.share_active_pages else NativeIndexPool
        self.index_pool = pool_type(self.req_to_token_pool, self.config, reserve_bytes())
        self.scorer = Scorer(self.config, self.device)
        self.workspace = GatherWorkspace(self.config.microbatch, self.device, torch.bfloat16)
        self.req_to_token_pool.add_native_index_listener(self.workspace)
        self._requests = None
        self._batch = None
        self._logged_sparse_decode = False

    @property
    def supports_full_cuda_graph_chunked_prefix(self):
        return False

    def init_forward_metadata_out_graph(self, forward_batch, in_capture=False):
        if forward_batch.forward_mode.name in ("EXTEND", "MIXED"):
            raise RuntimeError("Native sparse prefill graph capture is not implemented")
        return super().init_forward_metadata_out_graph(forward_batch, in_capture)

    def init_forward_metadata_in_graph(self, forward_batch):
        if forward_batch.forward_mode.name in ("EXTEND", "MIXED"):
            raise RuntimeError("Native sparse prefill graph capture is not implemented")
        return super().init_forward_metadata_in_graph(forward_batch)

    def init_forward_metadata(self, batch):
        super().init_forward_metadata(batch)
        self._batch = batch
        self._requests = None
        if (batch.forward_mode.is_decode() and self.config.mode == "v1"
                and batch.seq_lens_cpu is not None and not self._logged_sparse_decode
                and any(self.config.sparse(int(n) - 1, decode=True) for n in batch.seq_lens_cpu)):
            logger.info("Native V1 sparse decode active: threshold=%s, eager routing",
                        self.config.decode_threshold)
            self._logged_sparse_decode = True
        if batch.native_index_capture_dummy:
            return
        if batch.forward_mode.name in ("TARGET_VERIFY", "DRAFT_EXTEND", "DRAFT_EXTEND_V2"):
            # Tentative draft tokens never enter the persistent prefill index.
            return
        if batch.forward_mode.is_extend() and not batch.forward_mode.is_decode():
            if batch.forward_mode.name not in ("EXTEND", "MIXED"):
                raise ValueError(f"Unsupported sparse forward mode: {batch.forward_mode}")
            # Scheduler-owned CPU slots avoid a device-to-host synchronization.
            slots = batch.native_req_pool_indices_cpu
            decode_mask = batch.native_decode_mask
            if slots is None or decode_mask is None or any(slot is None for slot in slots):
                raise RuntimeError("Missing native scheduler request metadata")
            self._requests = list(zip(slots, batch.extend_prefix_lens_cpu,
                                      batch.extend_seq_lens_cpu, decode_mask))
            if any(decode_mask) and not all(decode_mask):
                logger.info("Native mixed attention metadata: prefill=%s decode=%s",
                            len(decode_mask) - sum(decode_mask), sum(decode_mask))

    def _dense_one(self, q, kp, vp, slots, total, layer):
        ps = self.page_size
        table = (slots[:total:ps] // ps).int()[None].contiguous()
        lengths = torch.tensor([total], device=q.device, dtype=torch.int32)
        return flash_attn_with_kvcache(
            q[None], kp.view(-1, ps, layer.tp_k_head_num, 128),
            vp.view(-1, ps, layer.tp_k_head_num, 128),
            page_table=table, cache_seqlens=lengths, causal=True,
            softmax_scale=layer.scaling)[0]

    def can_run_decode_cuda_graph(self, batch):
        if self.config.mode != "v1" or not batch.forward_mode.is_decode():
            return True
        if self.config.context <= self.config.decode_threshold + 1:
            return True
        if batch.seq_lens_cpu is None:
            return False
        return not any(self.config.sparse(int(total) - 1, decode=True)
                       for total in batch.seq_lens_cpu)

    def forward_decode(self, q, k, v, layer, forward_batch,
                       save_kv_cache=True, **kwargs):
        fb = forward_batch
        # Captured short-decode graphs keep the original FA3 implementation.
        # The graph runner must reject replay for batches containing long rows.
        if (self.config.mode != "v1" or
                self.config.context <= self.config.decode_threshold + 1):
            return super().forward_decode(q, k, v, layer, fb,
                                          save_kv_cache=save_kv_cache, **kwargs)
        if fb.seq_lens_cpu is None:
            raise RuntimeError("V1 decode routing requires scheduler CPU sequence lengths")
        lengths = [int(n) for n in fb.seq_lens_cpu]
        if not any(self.config.sparse(n - 1, decode=True) for n in lengths):
            return super().forward_decode(q, k, v, layer, fb,
                                          save_kv_cache=save_kv_cache, **kwargs)
        slots = fb.native_req_pool_indices_cpu
        if slots is None or len(slots) != len(lengths) or any(s is None for s in slots):
            raise RuntimeError("Missing native request ownership for sparse decode")
        if self._batch is not fb:
            raise RuntimeError("Sparse decode must execute eagerly with live metadata")
        self._requests = [(slot, total - 1, 1, True) for slot, total in zip(slots, lengths)]
        return self.forward_extend(q, k, v, layer, fb, save_kv_cache=save_kv_cache,
                                   **kwargs)

    def forward_extend(self, q, k, v, layer, forward_batch,
                       save_kv_cache=True, **kwargs):
        fb = forward_batch
        if (fb.native_index_capture_dummy or fb.forward_mode.name in
                ("TARGET_VERIFY", "DRAFT_EXTEND", "DRAFT_EXTEND_V2")):
            return super().forward_extend(q, k, v, layer, fb,
                                          save_kv_cache=save_kv_cache, **kwargs)
        if not save_kv_cache or k is None or v is None:
            raise ValueError("Sparse prefill requires native KV writes")
        if any(value is not None for value in kwargs.values()):
            raise ValueError("Extra attention modifiers are not supported by native sparse")
        if layer.qk_head_dim != 128 or layer.v_head_dim != 128:
            raise ValueError("Native sparse head dimension must be 128")
        if layer.tp_q_head_num != 4 * layer.tp_k_head_num:
            raise ValueError("Expected four Q heads per KV head")
        if getattr(layer, "sliding_window_size", None) not in (None, -1):
            raise ValueError("Model sliding-window attention is not supported")
        if getattr(layer, "logit_cap", 0) not in (None, 0):
            raise ValueError("Attention logit softcap is not supported")
        if getattr(layer, "is_cross_attention", False):
            raise ValueError("Cross attention is not supported")
        if self._batch is not fb or self._requests is None:
            raise RuntimeError("Native sparse metadata not initialized; no prefill graph capture")
        if any(count > self.config.chunk or prefix + count > self.config.context
               for _, prefix, count, _ in self._requests):
            raise ValueError("Request exceeds configured context/chunk capacity")
        all_dense = all(not self.config.sparse(prefix, decode)
                        for _, prefix, _, decode in self._requests)
        if (all_dense and self.config.share_active_pages and self.index_pool.early_build):
            # No current-query index reads in this branch. Publish KV first,
            # submit the future index, then enqueue normal FA3 on the main stream.
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(fb.out_cache_loc, self.forward_metadata.swa_out_cache_loc),
                k, v, layer.k_scale, layer.v_scale)
            kv_ready = torch.cuda.Event(enable_timing=True)
            kv_ready.record()
            build_jobs = []
            kp = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
            vp = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
            for slot, prefix, count, decode in self._requests:
                if decode:
                    continue
                state = self.index_pool.get(slot, layer.layer_id, layer.tp_k_head_num, q.device)
                if prefix == 0 and state.ready_prefix > 0:
                    state.reset()
                job = self.index_pool.prepare_next(slot, layer.layer_id, state, kp, vp,
                                             self.req_to_token[slot], prefix + count,
                                             stage="kv_ready", kv_ready=kv_ready)
                if job is not None:
                    build_jobs.append(job)
            result = super().forward_extend(q, k, v, layer, fb, save_kv_cache=False)
            attention_done = torch.cuda.Event(enable_timing=True)
            attention_done.record()
            for job in build_jobs:
                job['attention_done'] = attention_done
            return result
        if all_dense:
            result = super().forward_extend(q, k, v, layer, fb, save_kv_cache=True)
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(fb.out_cache_loc, self.forward_metadata.swa_out_cache_loc),
                k, v, layer.k_scale, layer.v_scale)
            result = torch.empty_like(q)
        kp = self.token_to_kv_pool.get_key_buffer(layer.layer_id)
        vp = self.token_to_kv_pool.get_value_buffer(layer.layer_id)
        # Native BF16 slot-major pool; never materialize a full request KV copy.
        if kp.ndim != 3 or vp.shape != kp.shape:
            raise ValueError("Unsupported native KV pool layout")
        q3 = q.reshape(-1, layer.tp_q_head_num, 128)
        output = result.view_as(q3)
        start = 0
        for slot, prefix, count, decode in self._requests:
            total = prefix + count
            slots = self.req_to_token[slot]
            if decode and not self.config.sparse(prefix, decode=True):
                if not all_dense:
                    output[start:start + count] = self._dense_one(
                        q3[start:start + count], kp, vp, slots, total, layer)
                start += count
                continue
            state = self.index_pool.get(slot, layer.layer_id, layer.tp_k_head_num, q.device)
            prepare_kwargs = {"v_pool": vp} if self.config.share_active_pages else {}
            if prefix == 0 and (state.ready_prefix > 0 or state.flat_until > 0):
                # Slot generations normally make this unnecessary; explicit
                # reset also handles scheduler recomputation from zero.
                state.wait()
                if self.config.share_active_pages:
                    state.reset()
                state.committed, state.ready_prefix, state.flat_until = 64, -1, 0
                if getattr(state, "full_bucket_append", False):
                    state.committed = 0
            if not self.config.sparse(prefix, decode):
                if not all_dense:
                    output[start:start + count] = self._dense_one(
                        q3[start:start + count], kp, vp, slots, total, layer)
            else:
                if self.config.mode == "v1":
                    state.prepare(kp, slots, prefix, **prepare_kwargs)
                    fixed = state.prepare_fixed(kp, slots, prefix, total, **prepare_kwargs)
                else:
                    state.append_flat(kp, slots, total, prefix)
                    fixed = None
                for head in range(layer.tp_k_head_num):
                    for offset in range(0, count, self.config.microbatch):
                        rows = min(self.config.microbatch, count - offset)
                        qs = q3[start + offset:start + offset + rows, head * 4:(head + 1) * 4].contiguous()
                        select = (getattr(self.scorer, "select_decode", self.scorer.select)
                                  if decode else self.scorer.select)
                        ids = select(qs, state, head, prefix, offset, total, fixed, layer.scaling)
                        if self.config.share_active_pages:
                            from .shared_kernels import gather
                            kg, vg = gather(state.arena, ids, head, self.workspace)
                        else:
                            kg, vg = self.workspace.gather(kp, vp, slots, ids, head)
                        out = flash_attn_with_kvcache(qs[:, None], kg, vg,
                                                     causal=False, softmax_scale=layer.scaling)
                        output[start + offset:start + offset + rows, head * 4:(head + 1) * 4] = out[:, 0]
            if self.config.mode == "v1":
                # Snapshot only new/boundary buckets, then build on a side stream.
                # Native KV reads finish on the producer stream before return.
                if self.config.share_active_pages:
                    self.index_pool.prepare_next(slot, layer.layer_id, state, kp, vp, slots, total)
                else:
                    state.prepare(kp, slots, total, asynchronous=True)
            elif prefix < self.config.threshold:
                # Quantize/append only new slots; no full-history per-query repack.
                state.append_flat(kp, slots, total, prefix)
            start += count
        return result
