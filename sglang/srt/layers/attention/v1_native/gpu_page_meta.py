"""Experimental GPU-owned page metadata for complete 4K buckets.

CPU knows lease IDs/ranges, never reads members, hashes, matches or page IDs.
Arena operations serialize access; no global allocator locks inside kernels.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def whole_compare(K,V,REQ,RK,RV,MAP,TABLE,CAND,BAD,
                  CAP:tl.constexpr,H:tl.constexpr,KS0:tl.constexpr,KS1:tl.constexpr,
                  VS0:tl.constexpr,VS1:tl.constexpr):
    p,h,c=tl.program_id(0),tl.program_id(1),tl.program_id(2)
    lease=tl.load(CAND+c);page=tl.load(TABLE+(lease*H+h)*64+p)
    i=tl.arange(0,64);d=tl.arange(0,128);base=h*CAP+page
    logical=tl.load(MAP+base*64+i);slot=tl.load(REQ+logical)
    k=tl.load(K+slot[:,None]*KS0+h*KS1+d[None,:]).to(tl.uint16,bitcast=True)
    v=tl.load(V+slot[:,None]*VS0+h*VS1+d[None,:]).to(tl.uint16,bitcast=True)
    oldk=tl.load(RK+base*8192+i[:,None]*128+d[None,:]).to(tl.uint16,bitcast=True)
    oldv=tl.load(RV+base*8192+i[:,None]*128+d[None,:]).to(tl.uint16,bitcast=True)
    bad=tl.sum(tl.sum(((k!=oldk)|(v!=oldv)).to(tl.int32),1),0)
    tl.atomic_max(BAD+c,bad,sem='relaxed')

@tr.jit
def choose_whole(CAND,BAD,FLAG,N:tl.constexpr,B:tl.constexpr):
    i=tl.arange(0,B);bad=tl.load(BAD+i,i<N,other=1)
    winner=tl.min(tl.where((i<N)&(bad==0),i,0x7fffffff),0)
    lease=tl.load(CAND+winner,winner<N,other=-1)
    tl.store(FLAG,lease)

@tr.jit
def page_match(K,V,REQ,IDS,FP,OLD_FP,REF,RK,RV,MAP,MATCH,FLAG,
               CAP:tl.constexpr,GCAP:tl.constexpr,SCAN:tl.constexpr,
               KS0:tl.constexpr,KS1:tl.constexpr,VS0:tl.constexpr,VS1:tl.constexpr):
    if tl.load(FLAG)<0:
        p,h=tl.program_id(0),tl.program_id(1)
        pages=tl.arange(0,SCAN)
        live=tl.load(REF+h*GCAP+pages,pages<GCAP,other=0)>0
        ok=live&(pages<GCAP)
        for j in range(4):
            old=tl.load(OLD_FP+(h*GCAP+pages)*4+j,pages<GCAP,other=0)
            val=tl.load(FP+(h*64+p)*4+j)
            ok=ok&(old==val)
        candidate=tl.min(tl.where(ok,pages,0x7fffffff),0)
        match=tl.full((),-1,tl.int32)
        i=tl.arange(0,64);d=tl.arange(0,128)
        logical=tl.load(IDS+h*4096+p*64+i)
        while (candidate<GCAP)&(match<0):
            original=tl.load(MAP+(h*CAP+candidate)*64+i)
            same=tl.sum((logical!=original).to(tl.int32),0)==0
            if same:
                slot=tl.load(REQ+logical)
                k=tl.load(K+slot[:,None]*KS0+h*KS1+d[None,:]).to(tl.uint16,bitcast=True)
                v=tl.load(V+slot[:,None]*VS0+h*VS1+d[None,:]).to(tl.uint16,bitcast=True)
                oldk=tl.load(RK+(h*CAP+candidate)*8192+i[:,None]*128+d[None,:]).to(tl.uint16,bitcast=True)
                oldv=tl.load(RV+(h*CAP+candidate)*8192+i[:,None]*128+d[None,:]).to(tl.uint16,bitcast=True)
                if tl.sum(tl.sum(((k!=oldk)|(v!=oldv)).to(tl.int32),1),0)==0:
                    match=candidate
            ok=ok&(pages!=candidate)
            candidate=tl.min(tl.where(ok,pages,0x7fffffff),0)
        tl.store(MATCH+h*64+p,match)

@tr.jit
def allocate(REF,FREE,COUNT,FP,OLD_FP,MATCH,FLAG,TABLE,JOBS,NEW,
             LEASE,H:tl.constexpr,GCAP:tl.constexpr):
    h=tl.program_id(0);whole=tl.load(FLAG);top=tl.load(COUNT+h)
    for p in range(64):
        if whole>=0:
            physical=tl.load(TABLE+(whole*H+h)*64+p);fresh=False
        else:
            physical=tl.load(MATCH+h*64+p);fresh=physical<0
            if fresh:
                top-=1
                tl.device_assert(top>=0,'GPU page arena exhausted')
                physical=tl.load(FREE+h*GCAP+top)
                for j in range(4):
                    tl.store(OLD_FP+(h*GCAP+physical)*4+j,tl.load(FP+(h*64+p)*4+j))
        tl.atomic_add(REF+h*GCAP+physical,1,sem='relaxed')
        tl.store(TABLE+(LEASE*H+h)*64+p,physical)
        job=h*64+p
        tl.store(JOBS+job*3,h);tl.store(JOBS+job*3+1,p);tl.store(JOBS+job*3+2,physical)
        tl.store(NEW+job,fresh)
    tl.store(COUNT+h,top)

@tr.jit
def release(REF,FREE,COUNT,TABLE,LEASE,H:tl.constexpr,GCAP:tl.constexpr):
    h=tl.program_id(0);top=tl.load(COUNT+h)
    for p in range(64):
        physical=tl.load(TABLE+(LEASE*H+h)*64+p)
        before=tl.atomic_add(REF+h*GCAP+physical,-1,sem='relaxed')
        tl.device_assert(before>0,'GPU page refcount underflow')
        if before==1:
            tl.store(FREE+h*GCAP+top,physical);top+=1
    tl.store(COUNT+h,top)

@tr.jit
def fingerprint_if_needed(K,V,REQ,IDS,OUT,FLAG,KS0:tl.constexpr,KS1:tl.constexpr,VS0:tl.constexpr,VS1:tl.constexpr):
    if tl.load(FLAG)<0:
        p,h=tl.program_id(0),tl.program_id(1);i=p*64+tl.arange(0,64);d=tl.arange(0,128)
        logical=tl.load(IDS+h*4096+i);slot=tl.load(REQ+logical)
        k=tl.load(K+slot[:,None]*KS0+h*KS1+d[None,:]).to(tl.uint16,bitcast=True).to(tl.uint32)
        v=tl.load(V+slot[:,None]*VS0+h*VS1+d[None,:]).to(tl.uint16,bitcast=True).to(tl.uint32)
        # Hash page-local order, not its location in the bucket. Identical
        # pages may move when a different part of the bucket changes.
        weight=(i[:,None]%64+1).to(tl.uint32)*2654435761+(d[None,:]+1).to(tl.uint32)*2246822519
        base=(h*64+p)*4
        tl.store(OUT+base,tl.sum(tl.sum(k,1),0).to(tl.int64))
        tl.store(OUT+base+1,tl.sum(tl.sum(v,1),0).to(tl.int64))
        tl.store(OUT+base+2,tl.sum(tl.sum(k*weight,1),0).to(tl.int64))
        tl.store(OUT+base+3,tl.sum(tl.sum(v*weight,1),0).to(tl.int64))
