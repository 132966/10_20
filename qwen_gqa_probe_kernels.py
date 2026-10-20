import triton as tr
import triton.language as tl

@tr.jit
def pack_new(K,LOC,NATIVE,COUNT:tl.constexpr,KS0:tl.constexpr,KS1:tl.constexpr,HEAD_BASE:tl.constexpr,HEAD_PAGES:tl.constexpr):
 i=tl.program_id(0);h=tl.program_id(1);d=tl.arange(0,128)
 slot=tl.load(LOC+i).to(tl.int64)
 x=tl.load(K+slot*KS0+h*KS1+d).to(tl.float32)
 scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(x),0),1.e-4)/448.)))
 page=(HEAD_BASE+h)*HEAD_PAGES+slot//64
 tl.store((NATIVE+page*8448+slot%64*128+d).to(tl.pointer_type(tl.float8e4nv)),(x/scale).to(tl.float8e4nv))
 tl.store((NATIVE+page*8448+8192+slot%64*4).to(tl.pointer_type(tl.float32)),scale)

@tr.jit
def pack_q(Q,OUT,W,ROWS:tl.constexpr,QS0:tl.constexpr,QS1:tl.constexpr,SCALE:tl.constexpr):
 row=tl.program_id(0);h=tl.program_id(1);d=tl.arange(0,128)
 x=tl.load(Q+row*QS0+h*QS1+d,(row<ROWS)&(h<4),other=0).to(tl.float32)
 scale=tl.exp2(tl.ceil(tl.log2(tl.maximum(tl.max(tl.abs(x),0),1.e-4)/448.)))
 tl.store(OUT+(row*32+h)*128+d,(x/scale).to(tl.float8e4nv))
 tl.store(W+row*32+h,tl.where((h<4)&(row<ROWS),scale*SCALE/4,0.))

@tr.jit
def gather_index(NATIVE,TABLE,K,S,N:tl.constexpr,B:tl.constexpr):
 x=tl.program_id(0)*B+tl.arange(0,B);token=x//128;d=x%128
 page=tl.load(TABLE+token//64,token<N,other=0).to(tl.int64)
 val=tl.load(NATIVE+page*8448+(token%64)*128+d,token<N,other=0)
 tl.store(K+x,val,token<N)
 v=tl.load((NATIVE+page*8448+8192+(token%64)*4).to(tl.pointer_type(tl.float32)),(token<N)&(d==0),other=0.)
 tl.store(S+token,v,(token<N)&(d==0))
