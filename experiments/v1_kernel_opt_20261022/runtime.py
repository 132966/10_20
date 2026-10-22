"""Preallocated, single-head 1K build adapter. Original implementation is copied.
No Tensor Core approximation is enabled. Call prepare/warmup before Graph capture.
Streams that share allocator state must be serialized, just as in the original.
"""
import torch,triton
from common import fg,v
import optimized as opt
from sort_cub import sort as cub_sort, module as cub_module

class Build:
    def __init__(self,native,table,meta,capacity=4096,records=2,optimized=True):
        assert native.is_cuda and native.dtype==torch.uint8 and native.shape[1]==8448
        assert table.shape==(16,) and table.dtype==torch.int32
        assert meta.shape==(3,) and meta.dtype==torch.int32
        assert capacity>=16 and records>=1
        self.optimized=optimized;self.capacity=capacity
        if optimized:cub_module()  # compile before capture, not inside the hot path
        self.native=native;self.table=table;self.meta=meta
        device=native.device
        def empty(shape,dtype):return torch.empty(shape,device=device,dtype=dtype)
        self.k=native.view(torch.float8_e4m3fn);self.ks=native.view(torch.float32)
        self.unit=empty((4096,128),torch.float32)
        self.perm=[empty((4096,),torch.int32) for _ in range(2)]
        self.scores=empty((4096,),torch.float32)
        self.s0=empty((4096,),torch.float32);self.s1=empty((4096,),torch.float32)
        self.anchor=empty((256,2),torch.int32);self.flag=empty((1,),torch.int32)
        self.tmp=empty((16,8448),torch.uint8);self.tmp8=self.tmp.view(torch.float8_e4m3fn);self.tmps=self.tmp.view(torch.float32);self.tmp32=self.tmp.view(torch.int32)
        self.tc=empty((2,8448),torch.uint8);self.tc8=self.tc.view(torch.float8_e4m3fn);self.tcs=self.tc.view(torch.float32);self.tc32=self.tc.view(torch.int32)
        self.mapping=empty((16,64),torch.int32);self.hash=empty((16,2),torch.int64)
        self.match=empty((16,),torch.int32);self.new=empty((16,),torch.bool)
        self.fine=torch.zeros((capacity,8448),device=device,dtype=torch.uint8);self.fine32=self.fine.view(torch.int32)
        self.centers=torch.zeros((records*2,8448),device=device,dtype=torch.uint8);self.cs=self.centers.view(torch.float32);self.c32=self.centers.view(torch.int32)
        self.members=torch.full((capacity,64),-1,device=device,dtype=torch.int32)
        self.hashes=torch.zeros((capacity,2),device=device,dtype=torch.int64)
        self.refs=torch.zeros(capacity,device=device,dtype=torch.int32)
        self.free=torch.arange(capacity,device=device,dtype=torch.int32);self.count=torch.tensor([capacity],device=device,dtype=torch.int32)
        self.tables=torch.full((records,16),-1,device=device,dtype=torch.int32)
        self.record_tables=[self.tables[i] for i in range(records)]
    def initialize(self):v.initialize[(64,)](self.k,self.ks,self.table,self.meta,self.unit,self.perm[0],self.flag)
    def grouping(self):
        current=0
        for n in [1024,512,256,128,64,32,16]:
            p=self.perm[current];o=self.perm[1-current];g=1024//n
            if self.optimized and n<=64:
                warps={64:4,32:16,16:8}[n]
                opt.split_fused[(g,)](self.unit,p,o,self.anchor,self.scores,self.flag,n,num_warps=warps)
            elif self.optimized:
                fg._dots[(64,1)](self.unit,p,self.anchor,self.s0,self.flag,n,0,0,num_warps=4)
                opt.anchored_dots[(64,)](self.unit,p,self.s0,self.s1,self.anchor,self.flag,n,1,num_warps=4)
                opt.anchored_dots[(64,)](self.unit,p,self.s1,self.scores,self.anchor,self.flag,n,2,num_warps=4)
                if n==1024:cub_sort(p,o,self.scores,self.flag,n)
                else:fg._sort[(g,1)](p,o,self.scores,self.flag,n,num_warps=4)
            else:
                for mode in [0,1,2]:
                    fg._dots[(64,1)](self.unit,p,self.anchor,self.scores,self.flag,n,0,mode,num_warps=4)
                    if mode<2:fg._anchor[(g,1)](p,self.anchor,self.scores,self.flag,n,mode,num_warps=4)
                fg._sort[(g,1)](p,o,self.scores,self.flag,n,num_warps=4)
            current=1-current
    def assemble(self):v.assemble[(16,8)](self.k,self.ks,self.table,self.meta,self.perm[1],self.tmp8,self.tmps,self.tc8,self.tcs,self.mapping)
    def pages(self):
        if self.optimized:
            opt.fingerprint_words[(16,)](self.tmp32,self.mapping,self.meta,self.hash)
            opt.match_words[(16,)](self.tmp32,self.mapping,self.hash,self.meta,self.fine32,self.members,self.hashes,self.refs,self.match,self.capacity,triton.next_power_of_2(self.capacity))
        else:
            v.fingerprint[(16,)](self.tmp,self.mapping,self.meta,self.hash)
            v.match_pages[(16,)](self.tmp,self.mapping,self.hash,self.meta,self.fine,self.members,self.hashes,self.refs,self.match,self.capacity,triton.next_power_of_2(self.capacity))
        v.allocate[(1,)](self.meta,self.match,self.free,self.count,self.refs,self.tables,self.new)
        if self.optimized:
            opt.publish_words[(16,)](self.tmp32,self.tc32,self.mapping,self.hash,self.meta,self.tables,self.new,self.fine32,self.c32,self.members,self.hashes)
        else:
            v.publish[(16,)](self.tmp,self.tc,self.tcs,self.mapping,self.hash,self.meta,self.tables,self.new,self.fine,self.centers,self.cs,self.members,self.hashes)
    def run(self):self.initialize();self.grouping();self.assemble();self.pages()
    def release(self,record=0):v.release[(1,)](self.record_tables[record],self.refs,self.free,self.count)
    def cycle(self):self.run();self.release(0)
