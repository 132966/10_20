"""Opt-in C4 integration; native Q/K preparation and MLA attention stay native.

SGLANG_V1_DEEPSEEK_CONFIG is a JSON object with Config fields plus mandatory
`total_budget_bytes` (all local layers). Outer model graphs must be disabled;
Config.graphs controls the inner build/query graphs independently.
"""

import json
import os
import torch
from .runtime import Config, Indexer


class C4Adapter:
    def __init__(self, backend, config, total_budget_bytes):
        self.backend = backend
        self.config = config
        self.total_budget_bytes = total_budget_bytes
        self.layers = {}
        backend.req_to_token_pool.add_native_index_listener(self)

    @classmethod
    def from_environment(cls, backend):
        raw = os.environ.get("SGLANG_V1_DEEPSEEK_CONFIG")
        if not raw:
            return None
        data = json.loads(raw)
        total = int(data.pop("total_budget_bytes"))
        if "q_buckets" in data:
            data["q_buckets"] = tuple(data["q_buckets"])
        model_heads = backend.model_runner.model_config.hf_text_config.index_n_heads
        data.setdefault("heads", model_heads)
        config = Config(**data)
        if config.heads != model_heads:
            raise ValueError("V1 heads must match native Index Q heads")
        args = backend.model_runner.server_args
        if not args.disable_cuda_graph or not args.disable_prefill_cuda_graph:
            raise ValueError(
                "V1 C4 subgraphs require --disable-cuda-graph --disable-prefill-cuda-graph; inner graphs remain enabled"
            )
        if (
            getattr(args, "tp_size", 1) != 1
            or getattr(args, "pp_size", 1) != 1
            or getattr(args, "speculative_algorithm", None)
            or getattr(args, "enable_prefill_context_parallel", False)
            or getattr(args, "enable_dsa_prefill_context_parallel", False)
            or backend.enable_deepseek_v4_fp4_indexer
            or backend.hisparse_coordinator is not None
        ):
            raise ValueError(
                "V1 C4 currently supports single-GPU FP8 without speculation or HiSparse"
            )
        if config.topk != backend.c4_topk:
            raise ValueError("V1 topk must match the native C4 attention topk")
        local_layers = sum(
            item is not None and item.compress_ratio == 4
            for item in backend.token_to_kv_pool.layer_mapping
        )
        required = local_layers * config.reservation
        if total < required:
            raise ValueError(
                f"Total V1 budget {total} is below all-layer reservation {required}"
            )
        if config.max_context * 4 < backend.max_context_len:
            raise ValueError(
                "V1 index capacity must cover configured model context / 4"
            )
        free_bytes, _ = torch.cuda.mem_get_info(backend.device)
        if required > free_bytes:
            raise ValueError(
                "Insufficient free GPU memory for V1; lower native mem-fraction-static"
            )
        return cls(backend, config, total)

    def row_metadata(self, batch, rows):
        slots = batch.native_req_pool_indices_cpu
        if slots is None:
            raise ValueError("V1 C4 requires scheduler CPU request slots")
        owners = []
        lengths = []
        pool = self.backend.req_to_token_pool
        if batch.forward_mode.is_decode():
            if batch.seq_lens_cpu is None:
                raise ValueError("Missing scheduler CPU decode lengths")
            for slot, n in zip(slots, batch.seq_lens_cpu.tolist()):
                owners.append((slot, int(pool.req_generation[slot])))
                lengths.append(max(0, int(n) // 4))
        elif batch.forward_mode.is_extend():
            prefixes = batch.extend_prefix_lens_cpu
            counts = batch.extend_seq_lens_cpu
            if prefixes is None or counts is None:
                raise ValueError("Missing scheduler CPU prefill lengths")
            for slot, prefix, count in zip(slots, prefixes, counts):
                owner = (slot, int(pool.req_generation[slot]))
                owners.extend([owner] * count)
                lengths.extend(
                    max(0, n // 4) for n in range(prefix + 1, prefix + count + 1)
                )
        else:
            raise ValueError("V1 C4 does not support speculative/idle query dispatch")
        if len(owners) != rows:
            raise ValueError("V1 C4 requires unpadded scheduler-owned query rows")
        return owners, lengths

    def query(self, c4_indexer, batch, q, weights, metadata, output, raw_output=None):
        if c4_indexer.use_fp4_indexer or metadata.c4_page_size != 64:
            raise ValueError("V1 C4 expects FP8 Index K with page_size=64")
        owners, lengths_cpu = self.row_metadata(batch, len(q))
        layer = c4_indexer.layer_id
        # Access through the native accessor on every call, preserving any native
        # readiness checks; stable graph pointers are checked explicitly below.
        native = self.backend.token_to_kv_pool.get_index_k_with_scale_buffer(
            layer_id=layer
        )
        if layer not in self.layers:
            if (
                len(self.layers) + 1
            ) * self.config.reservation > self.total_budget_bytes:
                raise ValueError(
                    "All-layer V1 index reservation exceeds total_budget_bytes"
                )
            self.layers[layer] = Indexer(native, self.config)
        engine = self.layers[layer]
        if native.data_ptr() != engine.native.data_ptr():
            raise RuntimeError(
                "Native Index K buffer changed address; flush V1 before rebinding"
            )
        table = metadata.page_table
        if table.shape[1] > self.config.pages:
            table = table[:, : self.config.pages].contiguous()
        result = engine.query(
            q.contiguous(),
            weights.float().contiguous(),
            table.contiguous(),
            metadata.c4_seq_lens.reshape(-1).contiguous(),
            owners,
            lengths_cpu,
            return_logical=raw_output is not None,
        )
        if raw_output is None:
            output.copy_(result)
        else:
            physical, logical = result
            output.copy_(physical)
            raw_output.copy_(logical)

    def release_request(self, slot):
        for engine in self.layers.values():
            for owner in list(engine.states):
                if owner[0] == slot:
                    engine.release_request(owner)

    def clear(self):
        # Flush drops arena bindings too, so a replacement native pool can rebind.
        for engine in self.layers.values():
            engine.close()
        self.layers.clear()
