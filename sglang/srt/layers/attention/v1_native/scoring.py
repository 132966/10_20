"""Signed FP8 paged scoring + existing page route and fused mapped Top2048."""
import importlib
import os
from pathlib import Path
import torch
from .grouping import quantize
from .kernels import mask_scores
from .deps.route import extension
from .deps.topk2048_mapped_20261012 import mapped_top2048
from .deps.topk2048 import topk2048
from .config import query_graph_enabled


def load_signed_deepgemm():
    dg = importlib.import_module("deep_gemm")
    # Refuse an accidental original ReLU build; never mutate sys.modules.
    if "vendor_norelu_20261011" not in Path(dg.__file__).parts:
        raise RuntimeError("Native signed routing requires the isolated no-ReLU "
                           "DeepGEMM fork; use launch_native_v1.sh")
    if torch.cuda.get_device_capability()[0] != 9:
        raise RuntimeError("The preserved signed DeepGEMM kernels require SM90")
    return dg


class Scorer:
    def __init__(self, config, device):
        self.config = config
        self.dg = load_signed_deepgemm()
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.query_graph = query_graph_enabled()
        self.decode_query_graph = query_graph_enabled(decode=True)

    def select_decode(self, q, state, head, prefix, offset, total, fixed, scale):
        """Optional query graph; gather and FA3 remain outside this subgraph."""
        if self.decode_query_graph and self._graph_eligible(state):
            return self._select_graph(q,state,head,prefix,offset,total,fixed,scale)
        return self.select_eager(q,state,head,prefix,offset,total,fixed,scale)

    def _graph_eligible(self,state):
        return (self.config.mode=='v1' and self.config.share_active_pages
                and getattr(state,'full_bucket_append',False) and state.pages>0)

    def _select_graph(self,q,state,head,prefix,offset,total,fixed,scale):
        if state.query_graph_pool is None:
            raise RuntimeError('Query graph pool was not initialized with the backend')
        return state.query_graph_pool.select(self,q,state,head,prefix,offset,total,fixed,scale)

    def paged(self, q8, weights, cache, table, count):
        lengths = torch.full((len(q8), 1), count, device=q8.device, dtype=torch.int32)
        meta = self.dg.get_paged_mqa_logits_metadata(lengths, 64, self.sms)
        return self.dg.fp8_fp4_paged_mqa_logits(
            (q8[:, None], None), cache, weights, lengths, table, meta, count,
            clean_logits=False)

    def select(self, q, state, head, prefix, offset, total, fixed, scale):
        if self.query_graph and self._graph_eligible(state):
            return self._select_graph(q,state,head,prefix,offset,total,fixed,scale)
        return self.select_eager(q,state,head,prefix,offset,total,fixed,scale)

    def select_eager(self, q, state, head, prefix, offset, total, fixed, scale):
        if self.config.share_active_pages:
            from .shared_scoring import select
            return select(self, q, state, head, prefix, offset, total, fixed, scale)
        rows, qheads, dim = q.shape
        if dim != 128 or qheads != 4:
            raise ValueError("This backend preserves Qwen's four-Q-head routing group")
        small, qs = quantize(q)
        q8 = torch.zeros((rows, 32, dim), device=q.device, dtype=small.dtype)
        weights = torch.zeros((rows, 32), device=q.device, dtype=torch.float32)
        q8[:, :4] = small
        weights[:, :4] = qs * (scale / 4)
        positions = torch.arange(prefix + offset, prefix + offset + rows,
                                 device=q.device, dtype=torch.int32)
        if self.config.mode == "dsa":
            scores = self.dg.fp8_mqa_logits(
                q8, (state.flat_k[head, :total], state.flat_scale[head, :total]),
                weights, torch.zeros_like(positions), positions + 1, True)
            return topk2048(scores, need_values=False)[1].int()
        else:
            pages = state.pages
            centers = pages * 8
            center_pages = (centers + 63) // 64
            ct = torch.arange(center_pages, device=q.device, dtype=torch.int32)[None].expand(rows, -1).contiguous()
            coarse = self.paged(q8, weights, state.centers[head], ct, centers)
            coarse = coarse[:, :centers]
            coarse.masked_fill_(~state.valid[head, :centers][None], -torch.inf)
            take = self.config.budget(pages)
            table = torch.empty((rows, take + len(fixed)), device=q.device, dtype=torch.int32)
            table[:, take:] = fixed[None]
            if take == pages:
                table[:, :take] = torch.arange(pages, device=q.device, dtype=torch.int32)
            elif 32 < pages <= 4096:
                extension(64, 4).route(coarse, table, pages, take)
            else:
                table[:, :take] = coarse.reshape(rows, pages, 8).amax(-1).topk(take, sorted=False).indices.int()
        scores = self.paged(q8, weights, state.fine[head], table, table.shape[1] * 64)
        mask_scores(scores, table, state.mapping[head], positions)
        # Output is logical ORIGINAL token position, not physical KV slot and not x4.
        return mapped_top2048(scores, table, state.mapping[head])
