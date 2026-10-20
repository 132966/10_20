"""Paired live GQA prefill probe. All modifications belong to the isolated copy."""
import json,math,os,time,statistics
from pathlib import Path
from types import SimpleNamespace
from dataclasses import dataclass
import torch
import triton as tr
import deep_gemm
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.srt.layers.attention.flashattention_backend import FlashAttentionBackend
from sglang.srt.mem_cache.memory_pool import KVWriteLoc
from sglang.srt.layers.attention.v1_deepseek import runtime as rt
from sglang.srt.layers.attention.v1_native.kernels import GatherWorkspace
from sglang.srt.layers.attention.v1_native.deps.topk2048 import module as topk_module
from bench_v1_dynamic_budget_metadata import BudgetMetadata
from qwen_gqa_probe_kernels import pack_new,pack_q,gather_index
ROOT=Path(__file__).resolve().parent

@dataclass(frozen=True)
class BudgetConfig(rt.Config):
 selected_pages:int=1
 @property
 def take(self):return self.selected_pages

class QwenGQAProbeBackend(FlashAttentionBackend):
 def __init__(self,runner):
  super().__init__(runner,fa_impl_ver=3)
  self._v1_deepseek=True  # Request CPU ownership metadata from ForwardBatch.
  assert not runner.use_mla_backend and self.page_size==64
  assert runner.server_args.disable_cuda_graph
  hf=runner.model_config.hf_config
  assert hf.num_attention_heads==32 and hf.num_key_value_heads==8 and hf.head_dim==128
  self.layers=hf.num_hidden_layers;self.heads=8;self.groups=self.layers*8
  self.capacity=tr.cdiv(self.max_context_len,4096)*4096;self.pages=self.capacity//64
  self.native_pages=tr.cdiv(self.token_to_kv_pool.size+64,64)
  self.threshold=int(os.environ.get('PROBE_THRESHOLD','65536'))
  self.repeat=int(os.environ.get('PROBE_REPEATS','3'))
  self.tag=os.environ.get('PROBE_TAG','main')
  self.lean=os.environ.get('PROBE_LEAN','0')=='1'
  self.lean_records=[]
  self.policy={x['prefix_k']:x for x in json.loads((ROOT/'chunk_budgets.json').read_text())}
  self.native=torch.empty((self.groups*self.native_pages,64,1,132),device=self.device,dtype=torch.uint8)
  self.fine=torch.empty((self.groups*self.pages+1024*68,64,1,132),device=self.device,dtype=torch.uint8)
  self.centers=torch.empty((self.groups*self.pages//8,64,1,132),device=self.device,dtype=torch.uint8)
  self.members=torch.full((self.groups*self.pages,64),-1,device=self.device,dtype=torch.int32)
  self.build_stream=torch.cuda.Stream(device=self.device)
  self.owner={};self.query={};self.budget=torch.zeros(1,device=self.device,dtype=torch.int32)
  rt.k.query_metadata=BudgetMetadata(self.budget)
  self.workspace=GatherWorkspace(1024,self.device,torch.bfloat16)
  self.q8=torch.empty((1024,32,128),device=self.device,dtype=torch.float8_e4m3fn)
  self.weights=torch.empty((1024,32),device=self.device)
  self.flat_k=torch.empty((self.capacity,128),device=self.device,dtype=torch.float8_e4m3fn)
  self.flat_s=torch.empty(self.capacity,device=self.device)
  self.names=('v1','dsa','dsa_paged')
  self.ids={name:torch.empty((1024,2048),device=self.device,dtype=torch.int32) for name in self.names}
  self.raw=torch.empty_like(self.ids['dsa']);self.plan=torch.empty((1025,2),device=self.device,dtype=torch.int32)
  self.identity=torch.arange(self.pages,device=self.device,dtype=torch.int32)[None].expand(1024,-1)
  self.starts=torch.zeros(1024,device=self.device,dtype=torch.int32)
  self.records=[];self.build_records=[];self.dense_records=[];self.consume_records=[];self.pending={};self.chunk_start=None;self.scratch_warm=set()
  self.timeline=torch.cuda.Event(enable_timing=True);self.timeline.record()
  self.request_build_count=0
  self.chunk_events=[]
  install_chunk_timer()
  self.req_to_token_pool.add_native_index_listener(self)
  self.report_path=ROOT/'results'/f'{self.tag}_layers.jsonl'
  print('GQA_PROBE_INIT',self.tag,'groups',self.groups,'original_slots',self.token_to_kv_pool.size,'page64','Top2048','eager_only',flush=True)

 def release_request(self,slot):
  self.build_stream.synchronize()
  torch.cuda.current_stream().synchronize()
  for owner in self.owner.values():owner.release_request(slot)
  self.flush_records()
  self.flush_chunk_events()
 def clear(self):
  self.build_stream.synchronize()
  torch.cuda.current_stream().synchronize()
  for owner in self.owner.values():owner.clear()
  self.flush_records()
 def flush_records(self):
  self.request_build_count+=len(self.build_records)
  self.build_records.clear();self.dense_records.clear();self.lean_records.clear()
  self.consume_records.clear();self.records.clear()

 def get_owner(self,gid):
  if gid not in self.owner:
   c=rt.Config(max_context=self.capacity,q_buckets=(1,),topk=2048,graphs=False,page_fraction=.5,budget_bytes=1<<30)
   storage={'fine':self.fine[gid*self.pages:(gid+1)*self.pages],
            'centers':self.centers[gid*self.pages//8:(gid+1)*self.pages//8],
            'members':self.members[gid*self.pages:(gid+1)*self.pages]}
   self.owner[gid]=rt.Indexer(self.native,c,storage=storage)
  return self.owner[gid]

 def get_query(self,total):
  tier=min(self.capacity,tr.cdiv(total,65536)*65536)
  if tier not in self.query:
   max_budget=max([x['pages'] for x in self.policy.values() if x['prefix_k']*1024<tier] or [self.policy[64]['pages']])
   max_budget=min(tier//64,max(32,max_budget))
   cfg=BudgetConfig(max_context=tier,q_buckets=(1,1024),topk=2048,graphs=False,expand_output=False,selected_pages=max_budget,budget_bytes=32<<30)
   proxy=SimpleNamespace(config=cfg,device=self.device,native=self.native,fine=self.fine,centers=self.centers,members=self.members,capacity=self.groups*self.pages)
   work=rt.Query(proxy,1024);work.map_physical=False
   self.query[tier]=work
  return self.query[tier]

 def previous_snapshot(self,layerid,slot,prefix):
  jobs=[]
  for h in range(8):
   gid=layerid*8+h;o=self.get_owner(gid)
   if slot not in o.states:o.states[slot]=rt.State(o)
   assert len(o.states[slot].leases)*1024==prefix,('publication length',layerid,h,prefix)
   event=self.pending.get(gid,o.done)
   jobs.append((h,event))
  return jobs

 def submit_build(self,layerid,slot,slots,total,ready):
  assert total%1024==0
  for h in range(8):
   gid=layerid*8+h;o=self.get_owner(gid)
   old=len(o.states[slot].leases) if slot in o.states else 0
   assert old+1==total//1024,('incremental bucket',layerid,h,old,total)
   # Record after owner initialization as well as the original K write.
   initialized=torch.cuda.Event();initialized.record()
   with torch.cuda.stream(self.build_stream):
    self.build_stream.wait_event(ready);self.build_stream.wait_event(initialized)
    table=(slots[:total:64]//64+gid*self.native_pages).int().contiguous()
    a=torch.cuda.Event(enable_timing=True);z=torch.cuda.Event(enable_timing=True);a.record()
    o.prepare(slot,table,total);z.record()
   self.pending[gid]=z
   self.build_records.append((dict(layer=layerid,head=h,prefix=total-1024,total=total,new_buckets=1,bucket_tokens=1024),a,z))

 def forward_extend(self,q,k,v,layer,forward_batch,save_kv_cache=True,**kwargs):
  fb=forward_batch
  if not fb.forward_mode.is_extend_without_speculative():return super().forward_extend(q,k,v,layer,fb,save_kv_cache=save_kv_cache,**kwargs)
  if fb.native_index_capture_dummy:raise RuntimeError('CUDA Graph forbidden in this benchmark')
  assert fb.batch_size==1 and len(fb.extend_prefix_lens_cpu)==1
  method=fb.rids[0].split(':')[0]
  assert method in ('v1','dsa','dsa_paged','dense')
  if method=='dense':return super().forward_extend(q,k,v,layer,fb,save_kv_cache=save_kv_cache,**kwargs)
  prefix=int(fb.extend_prefix_lens_cpu[0]);rows=int(fb.extend_seq_lens_cpu[0]);total=prefix+rows
  assert 0<rows<=1024 and total<=self.capacity
  # Final partial chunk has no complete 1K bucket to publish. With Radix off
  # and dense decode, run this tail through native FA3 and leave prior indices
  # alive until the normal request listener drains and releases them.
  if prefix==0 and rows<1024:
   return super().forward_extend(q,k,v,layer,fb,save_kv_cache=save_kv_cache,**kwargs)
  slot=fb.native_req_pool_indices_cpu[0];layerid=layer.layer_id
  q3=q.view(rows,32,128);slots=self.req_to_token[slot]
  self.token_to_kv_pool.set_kv_buffer(layer,KVWriteLoc(fb.out_cache_loc,self.forward_metadata.swa_out_cache_loc),k,v,layer.k_scale,layer.v_scale)
  kp=self.token_to_kv_pool.get_key_buffer(layerid);vp=self.token_to_kv_pool.get_value_buffer(layerid)
  pack_new[(rows,8)](kp,fb.out_cache_loc,self.native,rows,kp.stride(0),kp.stride(1),layerid*8,self.native_pages)
  ready=torch.cuda.Event();ready.record()
  jobs=self.previous_snapshot(layerid,slot,prefix) if method=='v1' else [(h,None) for h in range(8)]
  if prefix<self.threshold:
   da=torch.cuda.Event(enable_timing=True);dz=torch.cuda.Event(enable_timing=True);da.record()
   result=super().forward_extend(q,k,v,layer,fb,save_kv_cache=False,**kwargs)
   dz.record()
   if method=='v1' and rows==1024:self.submit_build(layerid,slot,slots,total,ready)
   self.dense_records.append((dict(prefix=prefix,layer=layerid),da,dz))
   if layerid==self.layers-1:
    self.flush_records();print('PROBE_DENSE_CHUNK',prefix,total,flush=True)
   return result
  policy=self.policy[prefix//1024];budget=policy['pages'];self.budget.fill_(budget)
  work=self.get_query(total)
  visible=torch.arange(prefix+1,total+1,device=self.device,dtype=torch.int32)
  work.lens.zero_();work.lens[:rows].copy_(visible)
  out=torch.empty_like(q3);head_stats=[]
  for h,event in jobs:
   if method=='v1':
    torch.cuda.current_stream().wait_event(event)
    gid=layerid*8+h;o=self.get_owner(gid);state=o.states[slot]
   else:gid=layerid*8+h
   qh=q3[:,4*h:4*h+4]
   native_table=(slots[:total:64]//64+gid*self.native_pages).int().contiguous()
   if method=='v1':
    work.native_table.zero_();work.native_table[:rows,:len(native_table)].copy_(native_table[None].expand(rows,-1))
    cp=work.owner.config.pages
    work.reordered_table.copy_((state.table[:cp]+gid*self.pages)[None].expand(1024,-1))
    work.center_table.copy_((state.center_table[:cp//8]+gid*self.pages//8)[None].expand(1024,-1))
   def prepare_q():pack_q[(1024,32)](qh,self.q8,self.weights,rows,qh.stride(0),qh.stride(1),layer.scaling)
   def select_v1():
    prepare_q();work.q.copy_(self.q8);work.weights.copy_(self.weights);work.compute();self.ids['v1'][:rows].copy_(work.logical[:rows])
   def select_dsa():
    prepare_q()
    gather_index[(tr.cdiv(total*128,1024),)](self.native,native_table,self.flat_k.view(torch.uint8),self.flat_s,total,1024)
    scores=deep_gemm.fp8_mqa_logits(self.q8[:rows],(self.flat_k[:total],self.flat_s[:total]),self.weights[:rows],self.starts[:rows],visible,clean_logits=True)
    mod=topk_module();mod.topk_plan(visible,self.plan[:rows+1],0)
    mod.topk_transform(scores,visible,self.identity[:rows],self.ids['dsa'][:rows],64,self.plan[:rows+1],self.raw[:rows])
   paged_table=native_table[None].expand(rows,-1)
   def select_dsa_paged():
    prepare_q()
    lengths=visible[:,None]
    schedule=deep_gemm.get_paged_mqa_logits_metadata(lengths,64,torch.cuda.get_device_properties(self.device).multi_processor_count)
    scores=work.dg_split(deep_gemm.fp8_paged_mqa_logits,self.q8[:rows],self.native,self.weights[:rows],lengths,paged_table,schedule,total,q_offset=rows)
    mod=topk_module();mod.topk_plan(visible,self.plan[:rows+1],0)
    mod.topk_transform(scores,visible,self.identity[:rows],self.ids['dsa_paged'][:rows],64,self.plan[:rows+1],self.raw[:rows])
   selected={'v1':select_v1,'dsa':select_dsa,'dsa_paged':select_dsa_paged}
   def attn(name):
    ids=self.ids[name][:rows]
    if ':smoke' in fb.rids[0]:
     assert bool(((ids>=0)&(ids<visible[:,None])).all()),(method,'range')
     ordered=ids.sort(dim=-1).values
     assert bool((ordered[:,1:]!=ordered[:,:-1]).all()),(method,'duplicates')
    kg,vg=self.workspace.gather(kp,vp,slots,ids,h)
    return flash_attn_with_kvcache(qh.contiguous()[:,None],kg,vg,causal=False,softmax_scale=layer.scaling)[:,0]
   def chain(name):selected[name]();return attn(name)
   live_value=chain(method)
   out[:,h*4:h*4+4].copy_(live_value)
  if method=='v1' and rows==1024:self.submit_build(layerid,slot,slots,total,ready)
  if layerid==self.layers-1:
   self.flush_records();print('CHUNK4',method,prefix,total,flush=True)
  return out.reshape_as(q)

 def flush_chunk_events(self):
  assert self.chunk_events
  meta=[m for m,a,z in self.chunk_events]
  assert len({m['rid'] for m in meta})==1
  with (ROOT/'results'/('coverage_gpu'+os.environ.get('CUDA_VISIBLE_DEVICES','0')+'.jsonl')).open('a') as f:
   f.write(json.dumps(dict(rid=meta[0]['rid'],method=meta[0]['method'],chunks=meta,build_tasks=self.request_build_count))+'\n')
  self.chunk_events.clear();self.request_build_count=0

def install_chunk_timer():
 from sglang.srt.model_executor.model_runner import ModelRunner
 if getattr(ModelRunner,'_chunk4_timer',False):return
 original=ModelRunner.forward
 def timed(runner,forward_batch,*args,**kwargs):
  fb=forward_batch
  if not fb.forward_mode.is_extend_without_speculative():return original(runner,fb,*args,**kwargs)
  prefix=int(fb.extend_prefix_lens_cpu[0])

  a=z=None
  result=original(runner,fb,*args,**kwargs)
  runner.attn_backend.chunk_events.append((dict(rid=fb.rids[0],method=fb.rids[0].split(':')[0],prefix=prefix,rows=int(fb.extend_seq_lens_cpu[0])),a,z))
  return result
 ModelRunner.forward=timed;ModelRunner._chunk4_timer=True
