"""Opt-in complete-bucket GPU metadata arena; original Arena remains intact."""
import torch
import os
import triton as tr
from .shared_pool import Arena,Bucket,capacity
from .fused_grouping import FusedGroups
from .bulk_build import _write
from . import gpu_page_meta as gm
from .config import build_graph_enabled

def scratch_bytes(config,heads,requests):
    cap=capacity(config,requests)
    leases=max(1,requests*((config.context+4095)//4096)+2)
    # Conservative charge for GPU metadata, grouping scratch, lease tables.
    # Include peak temporary tensors of exact PyTorch grouping fallback.
    return heads*(cap*64+4096*128*32+4096*16+leases*64*8)+leases*16+65536

class GPUOwnedArena(Arena):
    def __init__(self,config,heads,requests,device):
        super().__init__(config,heads,requests,device)
        if not __import__('os').environ.get('SGLANG_V1_FULL_BUCKET_APPEND','0')=='1':
            raise ValueError('GPU metadata currently requires full-bucket append')
        self.gcap=self.capacity-requests*(config.fixed_pages+64)
        if self.gcap<requests*((config.context+4095)//4096)*64:
            raise ValueError('GPU metadata capacity must cover maximum complete buckets')
        self.free=[list(range(self.capacity-1,self.gcap-1,-1)) for _ in range(heads)]
        self.max_leases=max(1,requests*((config.context+4095)//4096)+2)
        if self.gcap<self.max_leases*64:
            raise ValueError('GPU arena must accommodate every outstanding bucket lease')
        self.free_leases=list(range(self.max_leases-1,-1,-1))
        self.leases={};self.by_start={}
        self.grefs=torch.zeros((heads,self.gcap),device=device,dtype=torch.int32)
        self.gfree=torch.arange(self.gcap,device=device,dtype=torch.int32)[None].repeat(heads,1)
        self.gcount=torch.full((heads,),self.gcap,device=device,dtype=torch.int32)
        self.fingerprints=torch.zeros((heads,self.gcap,4),device=device,dtype=torch.int64)
        self.tables=torch.empty((self.max_leases,heads,64),device=device,dtype=torch.int32)
        self.candidates=torch.empty((self.max_leases,),device=device,dtype=torch.int32)
        self.bad=torch.empty_like(self.candidates)
        self.flag=torch.empty((),device=device,dtype=torch.int32)
        self.fp=torch.empty((heads,64,4),device=device,dtype=torch.int64)
        self.matches=torch.empty((heads,64),device=device,dtype=torch.int32)
        self.jobs=torch.empty((heads*64,3),device=device,dtype=torch.int32)
        self.new=torch.empty((heads*64,),device=device,dtype=torch.bool)
        self.groups=None if build_graph_enabled() else FusedGroups(heads,device)
        self.build_graph=None  # Installed/prewarmed by the pool on the main thread.
        self.validation_mismatches = (torch.zeros((),device=device,dtype=torch.int64)
            if os.environ.get('SGLANG_V1_VALIDATE_GPU_GROUPS','0')=='1' else None)
        self.validation_invalid = (torch.zeros((),device=device,dtype=torch.int64)
            if self.validation_mismatches is not None else None)
        self._scratch=[self.grefs,self.gfree,self.gcount,self.fingerprints,self.tables,
                       self.candidates,self.bad,self.flag,self.fp,self.matches,self.jobs,self.new,
                       ]
        if self.groups is not None:
            self._scratch.extend((self.groups.unit,*self.groups.perm,self.groups.anchor,self.groups.scores))
        if self.validation_mismatches is not None:
            self._scratch.extend((self.validation_mismatches,self.validation_invalid))
        self.record()

    def wait(self):
        super().wait()
        for tensor in getattr(self,'_scratch',()):
            tensor.record_stream(torch.cuda.current_stream(self.device))

    def acquire_bucket(self,k,v,slots,start,end):
        if end-start!=4096:
            raise ValueError('GPU-owned metadata only handles completed 4096-key buckets')
        if not self.free_leases:
            raise RuntimeError('GPU bucket lease capacity exceeded')
        candidates=self.by_start.get(start,[])
        self.flag.fill_(-1)
        if candidates:
            self.candidates[:len(candidates)].copy_(torch.tensor(candidates,device=k.device,dtype=torch.int32))
            self.bad[:len(candidates)].zero_()
            gm.whole_compare[(64,self.heads,len(candidates))](k,v,slots,self.k,self.v,self.mapping,
                self.tables,self.candidates,self.bad,self.capacity,self.heads,
                k.stride(0),k.stride(1),v.stride(0),v.stride(1),num_warps=8)
            gm.choose_whole[(1,)](self.candidates,self.bad,self.flag,len(candidates),tr.next_power_of_2(len(candidates)))
        lease=self.free_leases.pop()
        if self.build_graph is not None:
            try:
                ids=self.build_graph.run(self,k,v,slots,start,lease,
                                         return_ids=self.validation_mismatches is not None)
            except BaseException:
                self.free_leases.append(lease)
                raise
        else:
            if self.groups is None:
                self.free_leases.append(lease)
                raise RuntimeError('Build graph was not prepared by SharedIndexPool')
            ids=self.groups.run(k,slots,start,self.flag)
        if self.validation_mismatches is not None:
            from .grouping import build_bucket
            reference=build_bucket(k[slots[start:end].long()],start,ids_only=True)
            self.validation_mismatches.add_(torch.count_nonzero((ids!=reference)&(self.flag<0)))
            expected=torch.arange(start,end,device=k.device,dtype=ids.dtype)
            self.validation_invalid.add_(torch.count_nonzero(
                (ids.sort(-1).values!=expected[None,:])&(self.flag<0)))
        if self.build_graph is None:
            gm.fingerprint_if_needed[(64,self.heads)](k,v,slots,ids,self.fp,self.flag,
                k.stride(0),k.stride(1),v.stride(0),v.stride(1),num_warps=8)
            gm.page_match[(64,self.heads)](k,v,slots,ids,self.fp,self.fingerprints,self.grefs,
                self.k,self.v,self.mapping,self.matches,self.flag,self.capacity,self.gcap,tr.next_power_of_2(self.gcap),
                k.stride(0),k.stride(1),v.stride(0),v.stride(1),num_warps=8)
            gm.allocate[(self.heads,)](self.grefs,self.gfree,self.gcount,self.fp,self.fingerprints,self.matches,
                self.flag,self.tables,self.jobs,self.new,lease,self.heads,self.gcap,debug=True)
            _write[(self.heads*64,8)](k,v,slots,ids,self.jobs,self.k,self.v,
                self.fine.view(torch.float8_e4m3fn),self.fine.view(torch.float32),
                self.centers.view(torch.float8_e4m3fn),self.centers.view(torch.float32),self.mapping,self.valid,
                self.capacity,4096,k.stride(0),k.stride(1),v.stride(0),v.stride(1),self.new,True,num_warps=4)
        bucket=Bucket(None,[],self.tables[lease])
        bucket.gpu_lease=lease;bucket.start=start
        self.leases[lease]=bucket
        self.by_start.setdefault(start,[]).append(lease)
        return bucket

    def release_bucket(self,bucket):
        if bucket.refs!=1:
            raise RuntimeError('GPU bucket lease already released')
        lease=bucket.gpu_lease
        gm.release[(self.heads,)](self.grefs,self.gfree,self.gcount,self.tables,lease,self.heads,self.gcap,debug=True)
        bucket.refs=0
        self.by_start[bucket.start].remove(lease)
        if not self.by_start[bucket.start]:del self.by_start[bucket.start]
        del self.leases[lease]
        self.free_leases.append(lease)
