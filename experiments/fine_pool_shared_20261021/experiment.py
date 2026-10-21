"""Unchanged DeepGEMM paged-MQA, same single fine-pool ABI in both modes.
Independent experiment; production SGLang and its kernels are never modified.
"""
import torch
import triton as tr
import deep_gemm as dg
from tail_union import PreparedStage
from baseline_published import stage_tail_shared
from kernels import tables,normalize,map_topk

def bounds(n,s,w,end):
    s=min(n,s);t=max(s,min(min(max(end,0),n),max(s,n-w)));return s,t,s+n-t

class Data:
    def __init__(self,history=65536,rows=4096,sink=64,window=2048,max_take=40,min_end=None):
        self.history=history;self.rows=rows;self.sink=sink;self.window=window;self.max_take=max_take
        self.min_end=history if min_end is None else min_end
        self.np=max(1,tr.cdiv(history+rows,64));self.cap=max(64,max_take)
        self.native=torch.empty((self.np,8448),device='cuda',dtype=torch.uint8)
        self.native[:,:8192]=torch.randn((self.np,8192),device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
        self.native.view(torch.float32)[:,2048:]=torch.rand((self.np,64),device='cuda')*.25+.05
        self.nt=torch.randperm(self.np,device='cuda',dtype=torch.int32)
        self.nt_rows=self.nt.repeat(rows,1).contiguous()
        self.lens=torch.arange(history+1,history+rows+1,device='cuda',dtype=torch.int32)
        self.end=torch.tensor([history],device='cuda',dtype=torch.int32)
        self.takes=torch.full((rows,),max_take,device='cuda',dtype=torch.int32)
        self.selected=((torch.arange(max_take,device='cuda')[None,:]+torch.arange(rows,device='cuda')[:,None])%self.cap).int()
        # Reordered pages contain distinct original logical members, including
        # some sink/window members so overlap masking is actually tested.
        if history<self.cap*64:raise ValueError('use >=4096 history for this fine-pool fixture')
        self.members=torch.randperm(history,device='cuda')[:self.cap*64].view(self.cap,64).int()
        self.members.view(-1)[:64]=torch.arange(64,device='cuda',dtype=torch.int32)
        # Keep member IDs unique after forcing sink fixtures.
        rest=torch.randperm(history-64,device='cuda')[:self.cap*64-64]+64
        self.members.view(-1)[64:]=rest.int()
        logical=self.members.long();page=self.nt[logical//64].long();off=logical%64
        self.reordered=torch.empty((self.cap,8448),device='cuda',dtype=torch.uint8)
        self.reordered[:,:8192]=self.native[page[:,:,None],off[:,:,None]*128+torch.arange(128,device='cuda')[None,None,:]].reshape(self.cap,8192)
        self.reordered.view(torch.float32)[:,2048:]=self.native.view(torch.float32)[page,2048+off]
        self.q=torch.randn((rows,1,32,128),device='cuda').to(torch.float8_e4m3fn)
        self.weights=torch.rand((rows,32),device='cuda')/32
        c=bounds(history+rows,sink,window,self.min_end)[2]
        self.scratch=max(1,tr.cdiv(c,64))
        t=bounds(history+1,sink,window,self.min_end)[1]
        self.union_cap=min(self.np,tr.cdiv(sink,64)+tr.cdiv(history+rows,64)-t//64+2)
        # Worst-case shared leading padding at BOTH sink and tail boundaries.
        self.width=tr.cdiv(max_take*64+c+128,256)*256
        self.top=2048

class Runner:
    def __init__(self,data,shared):
        self.d=data;self.shared=shared
        d=data;extra=d.union_cap if shared else d.rows*d.scratch
        self.fine=torch.empty((d.cap+extra,8448),device='cuda',dtype=torch.uint8)
        self.fine[:d.cap].copy_(d.reordered)
        self.fine4=self.fine.view(-1,64,1,132)
        self.fine_fp8=self.fine.view(torch.float8_e4m3fn);self.fine_scale=self.fine.view(torch.float32)
        self.native_fp8=d.native.view(torch.float8_e4m3fn);self.native_scale=d.native.view(torch.float32)
        self.desc=torch.zeros(4,device='cuda',dtype=torch.int32)
        self.meta=torch.empty((d.rows,3),device='cuda',dtype=torch.int32)
        self.table=torch.empty((d.rows,d.width//64),device='cuda',dtype=torch.int32)
        self.dglens=torch.empty((d.rows,1),device='cuda',dtype=torch.int32)
        self.clens=torch.empty((d.rows,),device='cuda',dtype=torch.int32)
        self.owners=torch.arange(d.rows,device='cuda',dtype=torch.int32)
        self.scores=torch.empty((d.rows,d.width),device='cuda')
        self.values=torch.empty((d.rows,d.top),device='cuda')
        self.ids=torch.empty((d.rows,d.top),device='cuda',dtype=torch.int64)
        self.logical=torch.empty((d.rows,d.top),device='cuda',dtype=torch.int32)
        self.slots=torch.empty_like(self.logical)
        self.sms=torch.cuda.get_device_properties(0).multi_processor_count
        self.group=1
        if shared:self.stage=PreparedStage(d.native,d.nt,d.lens,d.end,self.fine[d.cap:],self.meta,self.desc,d.sink,d.window)
    def prepare(self):
        d=self.d
        if self.shared:self.stage()
        else:
            stage_tail_shared[(tr.cdiv(d.rows,self.group),d.scratch)](self.native_fp8,self.native_scale,d.nt_rows,d.lens,d.end,self.fine_fp8,self.fine_scale,self.owners,d.cap,d.np,d.scratch,d.sink,d.window,d.rows,self.group,num_warps=4)
        tables[(d.rows,)](d.lens,d.end,d.takes,d.selected,self.desc,self.table,self.dglens,self.clens,d.cap,d.max_take,d.scratch,d.width//64,d.sink,d.window,tr.next_power_of_2(d.width//64),self.shared,num_warps=4)
    def fine_score(self):
        d=self.d
        # Exactly the original DeepGEMM interface: ONE pool, per-Q tables/lens.
        self.schedule=dg.get_paged_mqa_logits_metadata(self.dglens,64,self.sms)
        self.raw=dg.fp8_paged_mqa_logits(d.q,self.fine4,d.weights,self.dglens,self.table,self.schedule,d.width,clean_logits=False)
        normalize[(d.rows,tr.cdiv(d.width,256))](self.raw,d.lens,d.end,d.takes,d.selected,d.members,self.scores,d.max_take,d.width,self.raw.stride(0),d.sink,d.window,256,self.shared,num_warps=4)
    def select(self):
        # A common preallocated PyTorch TopK harness in both arms, not the
        # production fused V1 TopK kernel. Canonical layout is compatible.
        d=self.d
        torch.topk(self.scores,d.top,dim=-1,out=(self.values,self.ids))
        map_topk[(d.rows,)](self.ids,self.values,d.lens,d.end,d.takes,d.selected,d.members,d.nt,self.logical,self.slots,d.max_take,d.top,d.sink,d.window,tr.next_power_of_2(d.top),num_warps=4)
    def run_fine(self):self.prepare();self.fine_score()
    def full(self):self.run_fine();self.select()
