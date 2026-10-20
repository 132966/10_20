"""DSA-style fixed query buckets and in-place replay metadata for V1.

One capacity per Q bucket is shared by every request/layer/KV head. Lengths,
page tables, schedules and device TMA descriptors are updated before replay.
Only real index pages are scored; unused capacity and padded Q rows are masked.
"""
import threading
import weakref
import torch
import triton as tr
import triton.language as tl
from .config import (PAGE, query_graph_sizes, query_graph_capacity,
                     query_graph_fixed_pages, shared_query_graph_bytes)
from .graph_runtime import GraphEntry
from .grouping import quantize
from .indirect_mqa import extension, descriptors
from .deps.topk2048 import module as topk_module


@tr.jit
def stage_tables(PAGES, FIXED, M, COARSE, FIXED_OUT, CLEN, FLEN,
                 PC: tl.constexpr, FC: tl.constexpr, BLOCK: tl.constexpr):
    row=tl.program_id(0)
    i=tl.program_id(1)*BLOCK+tl.arange(0,BLOCK)
    pages=tl.load(M);take=tl.load(M+1);fixed=tl.load(M+2);rows=tl.load(M+5)
    dummy=tl.load(FIXED)
    value=tl.load(PAGES+i,i<pages,other=dummy)
    tl.store(COARSE+row*PC+i,value,i<PC)
    if row==0:
        value=tl.load(FIXED+i,i<fixed,other=dummy)
        tl.store(FIXED_OUT+i,value,i<FC)
    if tl.program_id(1)==0:
        tl.store(CLEN+row,tl.where(row<rows,pages*64,0))
        tl.store(FLEN+row,tl.where(row<rows,(take+fixed)*64,0))


@tr.jit
def extract_centers(PTRS,M,TABLE,SCORES,OUT,PC:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);i=tl.program_id(1)*B+tl.arange(0,B)
    pages=tl.load(M);rows=tl.load(M+5)
    good=(i<pages*8)&(row<rows)
    physical=tl.load(TABLE+row*PC+i//8,good,other=0)
    valid_ptr=tl.load(PTRS).to(tl.pointer_type(tl.int1))
    valid=tl.load(valid_ptr+physical*8+i%8,good,other=False)
    score=tl.load(SCORES+row*PC*64+(i//8)*64+i%8,good,other=-float('inf'))
    tl.store(OUT+row*PC*8+i,tl.where(valid,score,-float('inf')),i<PC*8)


@tr.jit
def candidates(M,SELECTED,COARSE,FIXED,TABLE,PC:tl.constexpr,TC:tl.constexpr,
               WIDTH:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);i=tl.program_id(1)*B+tl.arange(0,B)
    take=tl.load(M+1);fixed=tl.load(M+2)
    rank=tl.load(SELECTED+row*TC+i,i<take,other=0)
    page=tl.load(COARSE+row*PC+rank,i<take,other=0)
    tail=tl.load(FIXED+i-take,(i>=take)&(i<take+fixed),other=0)
    dummy=tl.load(FIXED)
    value=tl.where(i<take,page,tl.where(i<take+fixed,tail,dummy))
    tl.store(TABLE+row*WIDTH+i,value,i<WIDTH)


@tr.jit
def mask_scores(PTRS,M,TABLE,SCORES,WIDTH:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);i=tl.program_id(1)*B+tl.arange(0,B)
    take=tl.load(M+1);fixed=tl.load(M+2);position=tl.load(M+3)+row
    start=tl.load(M+4);rows=tl.load(M+5)
    live=(row<rows)&(i<(take+fixed)*64)
    page=tl.load(TABLE+row*(WIDTH//64)+i//64,live,other=0)
    mapping=tl.load(PTRS+1).to(tl.pointer_type(tl.int32))
    logical=tl.load(mapping+page*64+i%64,live,other=-1)
    invalid=(~live)|(logical<0)|(logical>position)|((i<take*64)&((logical<64)|(logical>=start)))
    tl.store(SCORES+row*WIDTH+i,-float('inf'),(i<WIDTH)&invalid)


@tr.jit
def physical_ids(RAW,TABLE,OUT,WIDTH:tl.constexpr,B:tl.constexpr):
    row=tl.program_id(0);i=tl.program_id(1)*B+tl.arange(0,B)
    index=tl.load(RAW+row*2048+i,i<2048,other=-1)
    page=tl.load(TABLE+row*WIDTH+index//64,(index>=0)&(index<WIDTH*64)&(i<2048),other=0)
    tl.store(OUT+row*2048+i,tl.where(index>=0,page*64+index%64,-1),i<2048)


class QueryGraph(GraphEntry):
    def __init__(self,pool,scorer,rows,pages,q,state,head,prefix,offset,fixed,scale):
        self.rows,self.pages,self.capacity_pages=rows,pages,pages
        self.config=pool.config
        self.take=self.config.budget(pages)
        self.fixed_capacity=query_graph_fixed_pages(self.config)
        # SM90 writes complete four-page (256-key) scheduling tiles.
        self.width=tr.cdiv(self.take+self.fixed_capacity,4)*256
        self.sms=scorer.sms
        self.dg=scorer.dg
        self.mod=extension()
        self.topk=topk_module()
        self.descriptor_cache=weakref.WeakKeyDictionary()
        super().__init__(q.device,pool.ledger,shared_query_graph_bytes(self.config,pages,rows))
        try:
            b,d=self.buffers,self.device
            b['q']=torch.zeros((rows,4,128),device=d,dtype=q.dtype)
            b['q8']=torch.zeros((rows,32,128),device=d,dtype=torch.float8_e4m3fn)
            b['weights']=torch.zeros((rows,32),device=d,dtype=torch.float32)
            b['scale']=torch.ones(1,device=d,dtype=torch.float32)
            b['meta']=torch.zeros(6,device=d,dtype=torch.int64)
            b['ptrs']=torch.zeros(2,device=d,dtype=torch.int64)
            b['coarse_table']=torch.zeros((rows,pages),device=d,dtype=torch.int32)
            b['fixed']=torch.zeros(self.fixed_capacity,device=d,dtype=torch.int32)
            b['coarse_lens']=torch.zeros((rows,1),device=d,dtype=torch.int32)
            b['fine_lens']=torch.zeros((rows,1),device=d,dtype=torch.int32)
            b['coarse_maps']=torch.empty(512,device=d,dtype=torch.uint8)
            b['fine_maps']=torch.empty(512,device=d,dtype=torch.uint8)
            b['coarse_scores']=torch.empty((rows,pages*64),device=d,dtype=torch.float32)
            b['centers']=torch.empty((rows,pages,8),device=d,dtype=torch.float32)
            b['table']=torch.empty((rows,self.width//64),device=d,dtype=torch.int32)
            b['scores']=torch.empty((rows,self.width),device=d,dtype=torch.float32)
            b['identity']=torch.arange(self.width//64,device=d,dtype=torch.int32)
            b['raw']=torch.empty((rows,2048),device=d,dtype=torch.int32)
            b['topk_out']=torch.empty_like(b['raw'])
            b['output']=torch.empty_like(b['raw'])
            b['plan']=torch.empty((rows+1,2),device=d,dtype=torch.int32)
            self.stage(q,state,head,prefix,offset,fixed,scale)
            self.capture()
        except BaseException:
            self.close()
            raise

    def stage(self,q,state,head,prefix,offset,fixed,scale):
        b=self.buffers
        if state.pages>self.pages or fixed.shape[1]>self.fixed_capacity or len(q)>self.rows:
            raise ValueError('Query exceeds captured capacity')
        if len(q)<1 or state.pages<1 or fixed.shape[1]<1:
            raise ValueError('Empty sparse query metadata')
        state.wait()
        b['q'].zero_()
        b['q'][:len(q)].copy_(q)
        self.upload('meta',[state.pages,self.config.budget(state.pages),fixed.shape[1],
                            prefix+offset,state.fixed_start,len(q)])
        self.upload('ptrs',[state.arena.valid[head].data_ptr(),state.arena.mapping[head].data_ptr()])
        self.upload('scale',[scale/4],torch.float32)
        stage_tables[(self.rows,tr.cdiv(max(self.pages,self.fixed_capacity),256))](
            state.table[head],fixed[head],b['meta'],b['coarse_table'],b['fixed'],
            b['coarse_lens'],b['fine_lens'],self.pages,self.fixed_capacity,256)
        cache=self.descriptor_cache.setdefault(state.arena,{})
        if head not in cache:
            cache[head]=(descriptors(b['q8'],b['weights'],state.arena.centers[head]),
                         descriptors(b['q8'],b['weights'],state.arena.fine[head]))
        b['coarse_maps'].copy_(cache[head][0],non_blocking=True)
        b['fine_maps'].copy_(cache[head][1],non_blocking=True)
        # Mirror DSA's replay preparation: refresh schedules outside capture,
        # preserving the addresses already referenced by the captured graph.
        for kind in ('coarse','fine'):
            meta=self.dg.get_paged_mqa_logits_metadata(b[kind+'_lens'],64,self.sms)
            key=kind+'_schedule'
            if key not in b:
                b[key]=meta
            else:
                b[key].copy_(meta)

    def compute(self):
        b=self.buffers
        small,scales=quantize(b['q'])
        b['q8'].zero_();b['weights'].zero_()
        b['q8'][:,:4]=small
        b['weights'][:,:4]=scales*b['scale']
        b['coarse_scores'].fill_(-torch.inf)
        self.mod.launch(b['coarse_maps'],b['coarse_lens'],b['coarse_table'],
                        b['coarse_schedule'],b['coarse_scores'],self.sms)
        extract_centers[(self.rows,tr.cdiv(self.pages*8,256))](b['ptrs'],b['meta'],
            b['coarse_table'],b['coarse_scores'],b['centers'],self.pages,256)
        selected=b['centers'].amax(-1).topk(self.take,sorted=True).indices
        candidates[(self.rows,tr.cdiv(self.width//64,256))](b['meta'],selected,
            b['coarse_table'],b['fixed'],b['table'],self.pages,self.take,self.width//64,256)
        b['scores'].fill_(-torch.inf)
        self.mod.launch(b['fine_maps'],b['fine_lens'],b['table'],
                        b['fine_schedule'],b['scores'],self.sms)
        mask_scores[(self.rows,tr.cdiv(self.width,512))](b['ptrs'],b['meta'],b['table'],b['scores'],self.width,512)
        lengths=b['fine_lens'].view(-1)
        self.topk.topk_plan(lengths,b['plan'],0)
        self.topk.topk_transform(b['scores'],lengths,b['identity'][None].expand(self.rows,-1),
                                 b['topk_out'],64,b['plan'],b['raw'])
        physical_ids[(self.rows,8)](b['raw'],b['table'],b['output'],self.width//64,256)

    def run(self,q,state,head,prefix,offset,fixed,scale):
        with self.lock,torch.inference_mode():
            stream=self.begin()
            self.stage(q,state,head,prefix,offset,fixed,scale)
            self.graph.replay()
            output=self.buffers['output'][:len(q)].clone()
            self.finish(stream,(q,state.table,fixed))
            state.arena.record()
            return output

    def close(self):
        super().close()
        self.descriptor_cache.clear()


class QueryGraphPool:
    def __init__(self,config,ledger,drain=None):
        self.config,self.ledger,self.drain=config,ledger,drain
        self.sizes=query_graph_sizes()
        self.entries={}
        self.lock=threading.RLock()
        self.capture_count=0

    def select(self,scorer,q,state,head,prefix,offset,total,fixed,scale):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('V1 query graphs execute outside full-model capture')
        if (q.ndim!=3 or q.shape[1:]!=(4,128) or q.dtype!=torch.bfloat16
                or not state.full_bucket_append or not 0<=head<state.heads
                or offset<0 or prefix+offset+len(q)>total):
            raise ValueError('Unsupported shared graph query layout')
        if len(q)>self.sizes[-1]:
            return torch.cat([self.select(scorer,q[i:i+self.sizes[-1]],state,head,
                prefix,offset+i,total,fixed,scale) for i in range(0,len(q),self.sizes[-1])])
        rows=next(n for n in self.sizes if n>=len(q))
        key=(str(q.device),rows,q.dtype)
        with self.lock,torch.inference_mode():
            entry=self.entries.get(key)
            if entry is not None and state.pages>entry.pages:
                entry.close()
                del self.entries[key]
                entry=None
            if entry is None:
                if threading.current_thread() is not threading.main_thread():
                    raise RuntimeError('Query capture must run on the model thread')
                if self.drain is not None:
                    self.drain()
                capacity=query_graph_capacity(state.pages,self.config.context)
                entry=QueryGraph(self,scorer,rows,capacity,q,state,head,prefix,offset,fixed,scale)
                self.entries[key]=entry
                self.capture_count+=1
            state.decode_query_graphs[head]=entry  # Non-owning diagnostics alias.
            return entry.run(q,state,head,prefix,offset,fixed,scale)

    def close(self):
        with self.lock:
            for entry in self.entries.values():
                entry.close()
            self.entries.clear()
