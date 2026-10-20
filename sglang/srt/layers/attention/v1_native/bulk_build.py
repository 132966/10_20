"""Opt-in page assembly fusion; leaves grouping and exact sharing unchanged."""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def _write_one(K,V,REQ,IDS,JOBS,RK,RV,F8,F32,C8,C32,MAP,VALID,
           CAP:tl.constexpr, IS:tl.constexpr, KS0:tl.constexpr,KS1:tl.constexpr,
           VS0:tl.constexpr,VS1:tl.constexpr):
    job,g=tl.program_id(0),tl.program_id(1)
    h=tl.load(JOBS+job*3); src=tl.load(JOBS+job*3+1); dst=tl.load(JOBS+job*3+2)
    i=g*8+tl.arange(0,8); d=tl.arange(0,128)
    logical=tl.load(IDS+h*IS+src*64+i)
    good=logical>=0
    slot=tl.load(REQ+logical,good,other=0)
    k=tl.load(K+slot[:,None]*KS0+h*KS1+d[None,:],good[:,None],other=0)
    v=tl.load(V+slot[:,None]*VS0+h*VS1+d[None,:],good[:,None],other=0)
    page=h*CAP+dst
    tl.store(RK+page*8192+i[:,None]*128+d[None,:],k)
    tl.store(RV+page*8192+i[:,None]*128+d[None,:],v)
    tl.store(MAP+page*64+i,logical)
    x=k.to(tl.float32)
    scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(x),1),1.e-4)/448.)))
    tl.store(F8+page*8448+i[:,None]*128+d[None,:],(x/scale[:,None]).to(tl.float8e4nv))
    tl.store(F32+page*2112+2048+i,scale)
    count=tl.sum(good.to(tl.int32),0)
    mean=tl.sum(x,0)/tl.maximum(count,1)
    cs=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(mean),0),1.e-4)/448.)))
    tl.store(C8+page*8448+g*128+d,(mean/cs).to(tl.float8e4nv))
    tl.store(C32+page*2112+2048+g,cs)
    tl.store(VALID+page*8+g,count>0)
    # Match pack(F.pad(mean, ..., 56)) exactly, including padded scales.
    for n in range(1,8):
        tl.store(C8+page*8448+(g+n*8)*128+d,0.)
        tl.store(C32+page*2112+2048+g+n*8,2.384185791015625e-7)

@tr.jit
def _write(K,V,REQ,IDS,JOBS,RK,RV,F8,F32,C8,C32,MAP,VALID,
           CAP:tl.constexpr, IS:tl.constexpr, KS0:tl.constexpr,KS1:tl.constexpr,
           VS0:tl.constexpr,VS1:tl.constexpr,NEW=None,HAS_NEW:tl.constexpr=False):
    if not HAS_NEW or tl.load(NEW+tl.program_id(0)):
        _write_one(K,V,REQ,IDS,JOBS,RK,RV,F8,F32,C8,C32,MAP,VALID,
                   CAP,IS,KS0,KS1,VS0,VS1)

def write_pages(arena,k,v,slots,ids,jobs):
    if not jobs:
        return
    assert k.dtype == v.dtype == torch.bfloat16
    assert k.stride(-1)==v.stride(-1)==1
    table=torch.tensor(jobs,device=k.device,dtype=torch.int32)
    _write[(len(jobs),8)](k,v,slots,ids,table,arena.k,arena.v,
        arena.fine.view(torch.float8_e4m3fn),arena.fine.view(torch.float32),
        arena.centers.view(torch.float8_e4m3fn),arena.centers.view(torch.float32),
        arena.mapping,arena.valid,arena.capacity,ids.stride(0),
        k.stride(0),k.stride(1),v.stride(0),v.stride(1),num_warps=4)
