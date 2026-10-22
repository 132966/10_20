"""Independent V1 candidates. Fixed 1K bucket, one Index K head, original ABI.
32-bit page views are bit-preserving; no change to hashes or scale packing.
"""
import triton as tr
import triton.language as tl

@tr.jit
def split_fused(U,P,O,A,S,FLAG,N:tl.constexpr):
    if tl.load(FLAG)<0:
        group=tl.program_id(0);i=tl.arange(0,N);d=tl.arange(0,128)
        ids=tl.load(P+group*N+i)
        x=tl.load(U+ids[:,None]*128+d[None,:])
        seed=tl.load(P+group*N)
        y=tl.load(U+seed*128+d)
        s0=tl.sum(x*y[None,:],1)
        bp=tl.argmin(s0,0,tie_break_left=True)
        b=tl.load(P+group*N+bp)
        yb=tl.load(U+b*128+d)
        sb=tl.sum(x*yb[None,:],1)
        ap=tl.argmin(sb,0,tie_break_left=True)
        a=tl.load(P+group*N+ap)
        ya=tl.load(U+a*128+d)
        sa=tl.sum(x*ya[None,:],1)
        score=sa-sb
        tl.store(A+group*2,b);tl.store(A+group*2+1,a)
        tl.store(S+group*N+i,score)
        score=tl.where(score==0,0.,score)
        bits=score.to(tl.uint32,bitcast=True)
        ordered=bits^tl.where((bits&0x80000000)!=0,0xffffffff,0x80000000).to(tl.uint32)
        key=(ordered.to(tl.uint64)<<32)|(0xffffffff-i.to(tl.uint32)).to(tl.uint64)
        key=tl.sort(key,descending=True)
        pos=0xffffffff-key.to(tl.uint32)
        value=tl.load(P+group*N+pos)
        tl.store(O+group*N+i,value)

@tr.jit
def fingerprint_words(TMP32,MAP,META,HASH):
    if tl.load(META+2):
        p=tl.program_id(0);i=tl.arange(0,4096)
        w=tl.load(TMP32+p*2112+i,i<2112,other=0).to(tl.uint32)
        b0=(w&255).to(tl.int64);b1=((w>>8)&255).to(tl.int64)
        b2=((w>>16)&255).to(tl.int64);b3=((w>>24)&255).to(tl.int64)
        sums=b0+b1+b2+b3
        weighted=sums*(4*i.to(tl.int64)+1)+b1+2*b2+3*b3
        xw=tl.sum(weighted,0);xs=tl.sum(sums,0)
        j=tl.arange(0,64);m=tl.load(MAP+p*64+j).to(tl.int64)
        a=xw+tl.sum(m,0)*104729
        b=(xw+6*xs)*1000003+tl.sum(m*(j.to(tl.int64)+1),0)*65537
        tl.store(HASH+p*2,a);tl.store(HASH+p*2+1,b)

@tr.jit
def match_words(TMP32,MAP,HASH,META,FINE32,MEMBERS,HASHES,REFS,MATCH,CAP:tl.constexpr,SCAN:tl.constexpr):
    if tl.load(META+2):
        p=tl.program_id(0);j=tl.arange(0,SCAN)
        ref=tl.load(REFS+j,j<CAP,other=0)
        h0=tl.load(HASHES+j*2,(j<CAP)&(ref>0),other=0)
        h1=tl.load(HASHES+j*2+1,(j<CAP)&(ref>0),other=0)
        good=(j<CAP)&(ref>0)&(h0==tl.load(HASH+p*2))&(h1==tl.load(HASH+p*2+1))
        dst=tl.min(tl.where(good,j,CAP),0);found=-1
        w=tl.arange(0,4096);m=tl.arange(0,64)
        a=tl.load(TMP32+p*2112+w,w<2112,other=0)
        ma=tl.load(MAP+p*64+m)
        while (dst<CAP)&(found<0):
            b=tl.load(FINE32+dst.to(tl.int64)*2112+w,w<2112,other=0)
            mb=tl.load(MEMBERS+dst*64+m)
            same=(tl.sum((a!=b).to(tl.int32),0)==0)&(tl.sum((ma!=mb).to(tl.int32),0)==0)
            found=tl.where(same,dst,-1)
            good=good&(j!=dst);dst=tl.min(tl.where(good,j,CAP),0)
        tl.store(MATCH+p,found)

@tr.jit
def publish_words(TMP32,TC32,MAP,HASH,META,TABLES,NEW,FINE32,CENTERS32,MEMBERS,HASHES):
    if tl.load(META+2):
        p=tl.program_id(0);lease=tl.load(META+1).to(tl.int64)
        if tl.load(NEW+p):
            dst=tl.load(TABLES+lease*16+p).to(tl.int64)
            w=tl.arange(0,4096)
            data=tl.load(TMP32+p*2112+w,w<2112,other=0)
            tl.store(FINE32+dst*2112+w,data,w<2112)
            m=tl.arange(0,64);v=tl.load(MAP+p*64+m);tl.store(MEMBERS+dst*64+m,v)
            h=tl.arange(0,2);v=tl.load(HASH+p*2+h);tl.store(HASHES+dst*2+h,v)
        x=tl.arange(0,256);src=p*8+x//32;dst=lease*128+src
        val=tl.load(TC32+(src//64)*2112+(src%64)*32+x%32)
        tl.store(CENTERS32+(dst//64)*2112+(dst%64)*32+x%32,val)
        g=tl.arange(0,8);src=p*8+g;dst=lease*128+src
        val=tl.load(TC32+(src//64)*2112+2048+src%64)
        tl.store(CENTERS32+(dst//64)*2112+2048+dst%64,val)

@tr.jit
def dots_tc(U,P,A,S,FLAG,N:tl.constexpr,BM:tl.constexpr,MODE:tl.constexpr):
    if tl.load(FLAG)<0:
        group,tile=tl.program_id(0),tl.program_id(1)
        i=tile*BM+tl.arange(0,BM);d=tl.arange(0,128);c=tl.arange(0,16)
        ids=tl.load(P+group*N+i,i<N,other=0)
        x=tl.load(U+ids[:,None]*128+d[None,:],(i<N)[:,None],other=0)
        if MODE==0:
            a=tl.load(P+group*N);b=a
        elif MODE==1:
            a=tl.load(A+group*2);b=a
        else:
            a=tl.load(A+group*2+1);b=tl.load(A+group*2)
        anchors=tl.where(c==0,a,b)
        y=tl.load(U+anchors[None,:]*128+d[:,None],(c<2)[None,:],other=0)
        z=tl.dot(x,y,input_precision='tf32x3')
        sa=tl.sum(tl.where(c[None,:]==0,z,0.),1)
        if MODE==2:
            sb=tl.sum(tl.where(c[None,:]==1,z,0.),1)
            sa=sa-sb
        tl.store(S+group*N+i,sa,i<N)

@tr.jit
def pack_bmm(U,P,A,X,Y,N:tl.constexpr):
    g,t=tl.program_id(0),tl.program_id(1);i=t*16+tl.arange(0,16);d=tl.arange(0,128)
    ids=tl.load(P+g*N+i);x=tl.load(U+ids[:,None]*128+d[None,:])
    tl.store(X+(g*N+i[:,None])*128+d[None,:],x)
    if t==0:
        c=tl.arange(0,2);anchor=tl.load(A+g*2+1-c)
        y=tl.load(U+anchor[None,:]*128+d[:,None])
        tl.store(Y+g*256+d[:,None]*2+c[None,:],y)

@tr.jit
def finish_bmm(Z,S):
    i=tl.program_id(0)*256+tl.arange(0,256)
    a=tl.load(Z+i*2);b=tl.load(Z+i*2+1);tl.store(S+i,a-b)

@tr.jit
def anchored_dots(U,P,PREV,OUT,A,FLAG,N:tl.constexpr,MODE:tl.constexpr):
    # Separate PREV/OUT prevents inter-program read/write races.
    if tl.load(FLAG)<0:
        tile=tl.program_id(0);i=tile*16+tl.arange(0,16);d=tl.arange(0,128)
        group=(tile*16)//N;j=tl.arange(0,N)
        prev=tl.load(PREV+group*N+j)
        pos=tl.argmin(prev,0,tie_break_left=True)
        anchor=tl.load(P+group*N+pos)
        ids=tl.load(P+i)
        x=tl.load(U+ids[:,None]*128+d[None,:]);y=tl.load(U+anchor*128+d)
        score=tl.sum(x*y[None,:],1)
        if MODE==2:score=score-tl.load(PREV+i)
        tl.store(OUT+i,score)
        if (tile*16)%N==0:tl.store(A+group*2+MODE-1,anchor)
