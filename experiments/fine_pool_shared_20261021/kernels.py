"""Single-pool page tables and canonical score restoration for unchanged DeepGEMM.
Shared mode only changes preparation/layout, not GEMM's K-pool ABI.
"""
import triton as tr
import triton.language as tl

@tr.jit
def bounds(n,end,S:tl.constexpr,W:tl.constexpr):
    s=tl.minimum(n,S)
    t=tl.maximum(s,tl.minimum(tl.minimum(tl.maximum(end,0),n),tl.maximum(s,n-W)))
    return s,t,s+n-t

@tr.jit
def tables(LENS,END,TAKES,SELECTED,DESC,FT,DGLENS,CLENS,
           CAP:tl.constexpr,MAXTAKE:tl.constexpr,SCRATCH:tl.constexpr,
           FP:tl.constexpr,S:tl.constexpr,W:tl.constexpr,B:tl.constexpr,SHARED:tl.constexpr):
    r=tl.program_id(0);j=tl.arange(0,B)
    n=tl.load(LENS+r);end=tl.load(END);take=tl.load(TAKES+r)
    s,t,c=bounds(n,end,S,W)
    selected=tl.load(SELECTED+r*MAXTAKE+j,(j<take)&(j<MAXTAKE),other=0)
    k=j-take
    if SHARED:
        sp=(s+63)//64;tp=tl.maximum(sp,t//64)
        ep=(n+63)//64
        global_sp=tl.load(DESC);global_tp=tl.load(DESC+1)
        tail_page=CAP+global_sp+tp-global_tp+k-sp
        page=tl.where(k<sp,CAP+k,tail_page)
        pages=sp+tl.maximum(ep-tp,0)
        span=tl.where(ep>tp,sp*64+n-tp*64,n)
    else:
        page=CAP+r*SCRATCH+k
        pages=(c+63)//64;span=c
    page=tl.where(j<take,selected,tl.where((k>=0)&(k<pages),page,0))
    tl.store(FT+r*FP+j,page,j<FP)
    tl.store(DGLENS+r,tl.maximum(take*64+span,1))
    tl.store(CLENS+r,take*64+c)

@tr.jit
def normalize(RAW,LENS,END,TAKES,SELECTED,MEMBERS,OUT,
              MAXTAKE:tl.constexpr,WIDTH:tl.constexpr,STRIDE:tl.constexpr,
              S:tl.constexpr,W:tl.constexpr,B:tl.constexpr,SHARED:tl.constexpr):
    # Restore original compact candidate order: selected pages, sink, exact tail.
    # Boundaries/causality/overlap are masked BEFORE TopK.
    r=tl.program_id(0);j=tl.program_id(1)*B+tl.arange(0,B)
    n=tl.load(LENS+r);end=tl.load(END);take=tl.load(TAKES+r)
    s,t,c=bounds(n,end,S,W);hist=take*64
    fine=j<hist;x=j-hist
    page=tl.load(SELECTED+r*MAXTAKE+j//64,fine&(j<WIDTH),other=0)
    member=tl.load(MEMBERS+page*64+j%64,fine&(j<WIDTH),other=-1)
    logical=tl.where(x<s,x,t+x-s)
    if SHARED:
        sp=(s+63)//64;tp=tl.maximum(sp,t//64)
        tail_slot=tl.where(logical<sp*64,logical,sp*64+logical-tp*64)
        source=tl.where(fine,j,hist+tail_slot)
    else:
        source=j
    valid=(j<WIDTH)&tl.where(fine,(member>=s)&(member<t)&(member<n)&(member<end),(x>=0)&(x<c))
    value=tl.load(RAW+r*STRIDE+source,valid,other=-float('inf'))
    tl.store(OUT+r*WIDTH+j,value,j<WIDTH)

@tr.jit
def map_topk(IDS,VALUES,LENS,END,TAKES,SELECTED,MEMBERS,NT,LOGICAL,SLOTS,
             MAXTAKE:tl.constexpr,TOP:tl.constexpr,S:tl.constexpr,W:tl.constexpr,B:tl.constexpr):
    r=tl.program_id(0);i=tl.arange(0,B)
    rank=tl.load(IDS+r*TOP+i,i<TOP,other=0).to(tl.int32)
    value=tl.load(VALUES+r*TOP+i,i<TOP,other=-float('inf'))
    take=tl.load(TAKES+r);n=tl.load(LENS+r);end=tl.load(END)
    s,t,c=bounds(n,end,S,W);fine=rank<take*64
    page=tl.load(SELECTED+r*MAXTAKE+rank//64,(i<TOP)&fine,other=0)
    member=tl.load(MEMBERS+page*64+rank%64,(i<TOP)&fine,other=-1)
    x=rank-take*64
    logical=tl.where(fine,member,tl.where(x<s,x,t+x-s))
    valid=(i<TOP)&(value!=-float('inf'))&(logical>=0)&(logical<n)
    physical=tl.load(NT+logical//64,valid,other=0)
    tl.store(LOGICAL+r*TOP+i,tl.where(valid,logical,-1),i<TOP)
    tl.store(SLOTS+r*TOP+i,tl.where(valid,physical*64+logical%64,-1),i<TOP)
