"""Single-request causal chunk; nonnegative nondecreasing device LENS.
ONE shared request page table, packed pages = 8192 FP8 bytes + 256 scale bytes.
No SGLang dependency. No host readback, allocation or numerical conversion inside launch.
"""
import torch
import triton as tr
import triton.language as tl

@tr.jit
def bounds(n,S:tl.constexpr,W:tl.constexpr):
    s=tl.minimum(n,S)
    t=tl.maximum(s,tl.minimum(tl.maximum(n-1,0)//1024*1024,tl.maximum(s,n-W)))
    return s,t,s+n-t

@tr.jit
def stage_union(SRC,NT,LENS,DST,META,DESC,R:tl.constexpr,S:tl.constexpr,W:tl.constexpr,B:tl.constexpr,PAGE:tl.constexpr):
    pid=tl.program_id(0)
    first=tl.load(LENS);last=tl.load(LENS+R-1)
    _,tail,_=bounds(first,S,W)
    sp=(tl.minimum(last,S)+63)//64
    tp=tl.maximum(sp,tail//64);ep=(last+63)//64
    total=sp+tl.maximum(ep-tp,0)
    x=tl.arange(0,B)
    if PAGE:
        page=pid;off=x;valid=(page<total)&(off<2112);target=page*2112+off
    else:
        target=pid*B+x;page=target//2112;off=target%2112;valid=target<total*2112
    lp=tl.where(page<sp,page,tp+page-sp)
    physical=tl.load(NT+lp,page<total,other=0).to(tl.int64)
    data=tl.load(SRC+physical*2112+off,valid,other=0)
    tl.store(DST+target,data,valid)
    if pid<tr.cdiv(R,128):
        r=pid*128+tl.arange(0,128);n=tl.load(LENS+r,r<R,other=0)
        s,t,c=bounds(n,S,W)
        tl.store(META+r*3,s,r<R);tl.store(META+r*3+1,t,r<R);tl.store(META+r*3+2,c,r<R)
    if pid==0:
        tl.store(DESC,sp);tl.store(DESC+1,tp);tl.store(DESC+2,ep);tl.store(DESC+3,total)

@tr.jit
def reconstruct_for_test(U,US,M,D,O,OS,SCRATCH:tl.constexpr):
    # Test-only consumer, NOT required/timed by optimized stage.
    r,p=tl.program_id(0),tl.program_id(1)
    s=tl.load(M+r*3);t=tl.load(M+r*3+1);c=tl.load(M+r*3+2)
    sp=tl.load(D);tp=tl.load(D+1)
    i=p*64+tl.arange(0,64);d=tl.arange(0,128)
    logical=tl.where(i<s,i,t+i-s);lp=logical//64
    up=tl.where(lp<sp,lp,sp+lp-tp).to(tl.int64)
    if (p*64<c)|(p==0):
        v=tl.load(U+up[:,None]*8448+(logical%64)[:,None]*128+d[None,:],(i<c)[:,None],other=0)
        scale=tl.load(US+up*2112+2048+logical%64,i<c,other=0.)
        dst=(r*SCRATCH+p).to(tl.int64)
        tl.store(O+dst*8448+(i%64)[:,None]*128+d[None,:],v)
        tl.store(OS+dst*2112+2048+i%64,scale)

def launch(src,nt,lens,dst,meta,desc,sink,window,block=512,page=False,warps=4):
    jobs=dst.shape[0] if page else tr.cdiv(dst.numel()//4,block)
    stage_union[(max(jobs,tr.cdiv(lens.numel(),128)),)](src.view(torch.int32),nt,lens,dst.view(torch.int32),meta,desc,lens.numel(),sink,window,block,page,num_warps=warps)

class PreparedStage:
    """Precreate alias views and launch parameters; call after mutating lens/nt.
    Caller preallocates dst large enough for ALL replay lengths, and keeps the
    source pages alive. This class deliberately avoids hot-path .item()/sync.
    """
    def __init__(self,src,nt,lens,dst,meta,desc,sink,window,block=None,warps=4):
        assert src.dtype==dst.dtype==torch.uint8
        assert src.is_contiguous() and dst.is_contiguous()
        assert src.ndim==dst.ndim==2 and src.shape[1]==dst.shape[1]==8448
        assert nt.ndim==lens.ndim==1 and lens.numel()>0
        assert nt.is_contiguous() and lens.is_contiguous()
        assert nt.dtype==lens.dtype==meta.dtype==desc.dtype==torch.int32
        assert meta.shape==(lens.numel(),3) and desc.numel()==4
        assert meta.is_contiguous() and desc.is_contiguous()
        assert all(x.device==src.device for x in [nt,lens,dst,meta,desc]) and src.is_cuda
        assert sink>=0 and window>=0
        b=block or (256 if lens.numel()<=2048 else 512)
        self.args=(src.view(torch.int32),nt,lens,dst.view(torch.int32),meta,desc,lens.numel(),sink,window,b,False)
        self.grid=(max(tr.cdiv(dst.numel()//4,b),tr.cdiv(lens.numel(),128)),)
        self.warps=warps
    def __call__(self):
        stage_union[self.grid](*self.args,num_warps=self.warps)
