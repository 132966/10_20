"""Demonstration fine-score consumer: 1 K head, 32 Q heads.
Uses shared K directly; never expands shared bytes to per-Q buffers.
Not a replacement for the project's DeepGEMM/TopK pipeline.
"""
import triton as tr
import triton.language as tl
@tr.jit
def score_tail(Q,W,K,KS,LENS,DESC,OUT,SINK:tl.constexpr,WINDOW:tl.constexpr,SCRATCH:tl.constexpr,SHARED:tl.constexpr):
    r,p=tl.program_id(0),tl.program_id(1)
    n=tl.load(LENS+r);s=tl.minimum(n,SINK)
    t=tl.maximum(s,tl.minimum(tl.maximum(n-1,0)//1024*1024,tl.maximum(s,n-WINDOW)))
    count=s+n-t
    i=p*64+tl.arange(0,64);d=tl.arange(0,128);h=tl.arange(0,32)
    valid=i<count
    if SHARED:
        logical=tl.where(i<s,i,t+i-s)
        sp=tl.load(DESC);tp=tl.load(DESC+1)
        lp=logical//64
        page=tl.where(lp<sp,lp,sp+lp-tp).to(tl.int64)
        offset=logical%64
    else:
        page=tl.full((64,),r*SCRATCH+p,tl.int64)
        offset=i%64
    raw=tl.load(K+page[None,:]*8448+offset[None,:]*128+d[:,None],valid[None,:],other=0)
    k=raw.to(tl.float8e4nv,bitcast=True).to(tl.bfloat16)
    scale=tl.load(KS+page*2112+2048+offset,valid,other=0.)
    q=tl.load(Q+r*32*128+h[:,None]*128+d[None,:])
    w=tl.load(W+r*32+h)
    dot=tl.dot(q,k)
    score=tl.sum(tl.maximum(dot,0.)*w[:,None],0)*scale
    tl.store(OUT+r*(SCRATCH*64)+i,tl.where(valid,score,-float('inf')))
