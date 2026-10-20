"""Small real Triton tests: shared page lifecycle and shared tail staging."""
import importlib.util
from pathlib import Path
import unittest
import torch
spec=importlib.util.spec_from_file_location('page_test_kernels',Path(__file__).with_name('kernels.py'))
k=importlib.util.module_from_spec(spec); spec.loader.exec_module(k)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class PageExamples(unittest.TestCase):
    def tensor(self,x,dtype=torch.int32): return torch.tensor(x,device='cuda:0',dtype=dtype)
    def equal(self,a,b): torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_page_lifecycle(self):
        # allocate/release are fixed at 16 pages (one 1K bucket).
        cap=16
        tmp=torch.arange(16,device='cuda:0',dtype=torch.uint8)[:,None].expand(16,8448).contiguous()
        mapping=torch.arange(1024,device='cuda:0',dtype=torch.int32).reshape(16,64)
        meta=self.tensor([0,0,1]); hashes=torch.empty((16,2),device='cuda:0',dtype=torch.int64)
        centers=torch.zeros((2,8448),device='cuda:0',dtype=torch.uint8)
        centers[:,:8192]=7; centers.view(torch.float32)[:,2048:]=1.25
        fine=torch.zeros_like(tmp); members=torch.full_like(mapping,-1)
        live_hash=torch.zeros_like(hashes); refs=torch.zeros(cap,device='cuda:0',dtype=torch.int32)
        free=torch.arange(cap,device='cuda:0',dtype=torch.int32); count=self.tensor(cap)
        tables=torch.full((2,16),-1,device='cuda:0',dtype=torch.int32)
        new=torch.empty(16,device='cuda:0',dtype=torch.bool)
        match=torch.empty(16,device='cuda:0',dtype=torch.int32)
        center_pool=torch.zeros((4,8448),device='cuda:0',dtype=torch.uint8)
        def lookup():
            k.match_pages[(16,)](tmp,mapping,hashes,meta,fine,members,live_hash,refs,match,cap,16)
        def alloc_publish():
            k.allocate[(1,)](meta,match,free,count,refs,tables,new,debug=True)
            k.publish[(16,)](tmp,centers,centers.view(torch.float32),mapping,hashes,meta,tables,new,
                            fine,center_pool,center_pool.view(torch.float32),members,live_hash)
        k.fingerprint[(16,)](tmp,mapping,meta,hashes)
        lookup(); self.assertTrue(bool((match==-1).all()))
        alloc_publish()
        self.equal(tables[0],torch.arange(15,-1,-1,device='cuda:0',dtype=torch.int32))
        self.assertEqual(count.item(),0); self.assertTrue(bool(new.all()))
        self.equal(fine[tables[0].long()],tmp); self.equal(members[tables[0].long()],mapping)
        self.equal(center_pool[:2],centers); self.assertTrue(bool((refs==1).all()))
        # Even a matching fingerprint must pass exact bytes and member comparison.
        tmp[0,0]=99; lookup(); self.assertEqual(match[0].item(),-1); tmp[0,0]=0
        mapping[0,0]=9999; lookup(); self.assertEqual(match[0].item(),-1); mapping[0,0]=0
        # Second bucket record shares exactly the same 16 pages; no extra capacity.
        meta[1]=1; lookup(); self.equal(match,tables[0])
        alloc_publish()
        self.equal(tables[1],tables[0]); self.assertFalse(bool(new.any()))
        self.assertEqual(count.item(),0); self.assertTrue(bool((refs==2).all()))
        self.equal(center_pool[2:],centers)
        k.release[(1,)](tables[0],refs,free,count)
        self.assertEqual(count.item(),0); self.assertTrue(bool((refs==1).all()))
        self.equal(fine[tables[1].long()],tmp)
        k.release[(1,)](tables[1],refs,free,count)
        self.assertEqual(count.item(),16); self.assertTrue(bool((refs==0).all()))
        self.equal(free.sort().values,torch.arange(16,device='cuda:0',dtype=torch.int32))
        print('page example: allocate16 -> share(ref=2) -> release A(ref=1) -> release B(free=16)')

    def test_shared_tail(self):
        # Four Q rows; identical requests/lengths pairwise share the same staging owner.
        rows,np,cap,scratch=4,18,1,3
        native=torch.zeros((np,8448),device='cuda:0',dtype=torch.uint8)
        for page in range(np):
            native[page,:8192].view(torch.float8_e4m3fn).fill_(page+1)
            native.view(torch.float32)[page,2048:]=torch.arange(64,device='cuda:0')+1
        nt=torch.arange(np-1,-1,-1,device='cuda:0',dtype=torch.int32)[None].expand(rows,-1).contiguous()
        lens=self.tensor([1100,1100,1101,1101]); owners=self.tensor([0,0,2,2])
        out=torch.full((cap+rows*scratch,8448),255,device='cuda:0',dtype=torch.uint8)
        k.stage_tail_shared[(1,scratch)](native.view(torch.float8_e4m3fn),native.view(torch.float32),nt,lens,
             out.view(torch.float8_e4m3fn),out.view(torch.float32),owners,cap,np,scratch,64,64,ROWS=rows,GROUP=4,num_warps=4)
        for row,n in ((0,1100),(2,1101)):
            # tail=min(floor((n-1)/1024)*1024,n-window)=1024
            ids=torch.cat((torch.arange(64,device='cuda:0'),torch.arange(1024,n,device='cuda:0')))
            expected=torch.zeros((scratch,8448),device='cuda:0',dtype=torch.uint8)
            pages=nt[row,ids//64].long(); offsets=ids%64
            expected.view(scratch,8448)[torch.arange(len(ids),device="cuda:0")[:,None]//64,(torch.arange(len(ids),device="cuda:0")[:,None]%64)*128+torch.arange(128,device="cuda:0")[None,:]]=native[:,:8192].reshape(np,64,128)[pages,offsets]
            for j in range(len(ids)):
                expected.view(torch.float32)[j//64,2048+j%64]=native.view(torch.float32)[pages[j],2048+offsets[j]]
            self.equal(out[cap+row*scratch:cap+(row+1)*scratch],expected)
        # Non-owner rows have no copies; their page tables point to owner scratch.
        for row in (1,3): self.assertTrue(bool((out[cap+row*scratch:cap+(row+1)*scratch]==255).all()))
        self.assertTrue(bool((out[0]==255).all()))
        rt=torch.zeros((rows,np),device='cuda:0',dtype=torch.int32)
        selected=torch.zeros((rows,1),device='cuda:0',dtype=torch.int32)
        takes=self.tensor([0]*rows); ft=torch.empty((rows,scratch),device='cuda:0',dtype=torch.int32)
        k.fine_table_shared[(rows,)](rt,selected,takes,lens,ft,owners,cap,np,1,scratch,scratch,4)
        self.equal(ft,self.tensor([[1,2,3],[1,2,3],[7,8,9],[7,8,9]]))
        print('tail example: Q0/Q1 share pages [1,2,3]; Q2/Q3 share pages [7,8,9]')

if __name__=='__main__': unittest.main()
