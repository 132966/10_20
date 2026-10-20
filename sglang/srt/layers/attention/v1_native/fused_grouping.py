"""Fixed-4K balanced cosine grouping, preallocated GPU scratch.

Same split/left-tie policy; reduction-order equivalence is tested separately.
Fused mode allows reduction-order differences from reference mode.
In fused mode, a whole-bucket hit makes every grouping kernel a no-op.
"""
import torch
import os
import triton as tr
import triton.language as tl

def grouping_mode():
    mode=os.environ.get('SGLANG_V1_GROUPING_MODE')
    if mode is None:
        mode=('fused' if os.environ.get('SGLANG_V1_EXPERIMENTAL_FUSED_GROUPS','0')=='1'
              else 'reference')
    if mode not in ('reference','fused'):
        raise ValueError('SGLANG_V1_GROUPING_MODE must be reference or fused')
    return mode

@tr.jit
def _publish_exact(SRC,DST,FLAG,N:tl.constexpr):
    if tl.load(FLAG)<0:
        i=tl.program_id(0)*256+tl.arange(0,256)
        tl.store(DST+i,tl.load(SRC+i,i<N,other=0),i<N)

@tr.jit
def _init(K,REQ,U,P,FLAG,START,KS0:tl.constexpr,KS1:tl.constexpr):
    if tl.load(FLAG)<0:
        i=tl.program_id(0)*16+tl.arange(0,16);h=tl.program_id(1);d=tl.arange(0,128)
        slot=tl.load(REQ+START+i)
        x=tl.load(K+slot[:,None]*KS0+h*KS1+d[None,:]).to(tl.float32)
        norm=tl.sqrt(tl.sum(x*x,1))
        unit=x/tl.maximum(norm[:,None],1.e-12)
        tl.store(U+(h*4096+i[:,None])*128+d[None,:],unit)
        tl.store(P+h*4096+i,START+i)

@tr.jit
def _dots(U,P,A,S,FLAG,N:tl.constexpr,START,MODE:tl.constexpr):
    if tl.load(FLAG)<0:
        i=tl.program_id(0)*16+tl.arange(0,16);h=tl.program_id(1);d=tl.arange(0,128)
        group=i//N
        ids=tl.load(P+h*4096+i)-START
        if MODE==0:
            anchor=tl.load(P+h*4096+group*N)-START
        else:
            anchor=tl.load(A+(h*256+group)*2+(MODE-1))-START
        x=tl.load(U+(h*4096+ids[:,None])*128+d[None,:])
        y=tl.load(U+(h*4096+anchor[:,None])*128+d[None,:])
        value=tl.sum(x*y,1)
        if MODE==2:
            value=value-tl.load(S+h*4096+i)
        tl.store(S+h*4096+i,value)

@tr.jit
def _anchor(P,A,S,FLAG,N:tl.constexpr,SLOT:tl.constexpr):
    if tl.load(FLAG)<0:
        group,h=tl.program_id(0),tl.program_id(1);i=tl.arange(0,N)
        score=tl.load(S+h*4096+group*N+i)
        pos=tl.argmin(score,0,tie_break_left=True)
        chosen=tl.load(P+h*4096+group*N+pos)
        tl.store(A+(h*256+group)*2+SLOT,chosen)

@tr.jit
def _sort(P,O,S,FLAG,N:tl.constexpr):
    if tl.load(FLAG)<0:
        group,h=tl.program_id(0),tl.program_id(1);i=tl.arange(0,N)
        score=tl.load(S+h*4096+group*N+i)
        score=tl.where(score==0,0.,score)
        bits=score.to(tl.uint32,bitcast=True)
        ordered=bits^tl.where((bits&0x80000000)!=0,0xffffffff,0x80000000).to(tl.uint32)
        key=(ordered.to(tl.uint64)<<32)|(0xffffffff-i.to(tl.uint32)).to(tl.uint64)
        sorted_key=tl.sort(key,descending=True)
        pos=0xffffffff-sorted_key.to(tl.uint32)
        value=tl.load(P+h*4096+group*N+pos)
        tl.store(O+h*4096+group*N+i,value)

class FusedGroups:
    def __init__(self,heads,device):
        self.heads=heads
        self.mode=grouping_mode()
        self.unit=torch.empty((heads,4096,128),device=device,dtype=torch.float32)
        self.perm=[torch.empty((heads,4096),device=device,dtype=torch.int32) for _ in range(2)]
        self.anchor=torch.empty((heads,256,2),device=device,dtype=torch.int32)
        self.scores=torch.empty((heads,4096),device=device,dtype=torch.float32)

    def run(self,k,slots,start,flag):
        if k.ndim!=3 or k.shape[1:]!=(self.heads,128) or k.stride(-1)!=1:
            raise ValueError('GPU grouping requires [slots, heads, 128] with contiguous dimensions')
        if slots.ndim!=1 or not slots.is_contiguous() or start<0 or start+4096>slots.numel():
            raise ValueError('GPU grouping requires a complete 4096-key range and contiguous slot map')
        # Full fusion intentionally permits FP32 reduction-order differences.
        # Preserve the original grouping in reference mode. This
        # fallback still stays entirely on GPU, but computes grouping even on
        # a whole-bucket cache hit (publication/page writes remain skipped).
        if self.mode=='reference':
            from .grouping import build_bucket
            ids=build_bucket(k[slots[start:start+4096].long()],start,ids_only=True)
            _publish_exact[(tr.cdiv(ids.numel(),256),)](ids,self.perm[1],flag,ids.numel())
            return self.perm[1]
        _init[(256,self.heads)](k,slots,self.unit,self.perm[0],flag,start,k.stride(0),k.stride(1),num_warps=4,enable_fp_fusion=False)
        current=0
        for n in (4096,2048,1024,512,256,128,64,32,16):
            p=self.perm[current]
            for mode in (0,1,2):
                _dots[(256,self.heads)](self.unit,p,self.anchor,self.scores,flag,n,start,mode,num_warps=4,enable_fp_fusion=False)
                if mode<2:
                    _anchor[(4096//n,self.heads)](p,self.anchor,self.scores,flag,n,mode,num_warps=4)
            _sort[(4096//n,self.heads)](p,self.perm[1-current],self.scores,flag,n,num_warps=4 if n<=1024 else 8)
            current=1-current
        return self.perm[current]
