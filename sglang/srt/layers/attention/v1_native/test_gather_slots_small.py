"""Real Triton K/V gather tests, compared with PyTorch indexing."""
import importlib.util
from pathlib import Path
import unittest
import torch
spec=importlib.util.spec_from_file_location('gather_impl',Path(__file__).with_name('kernels.py'))
mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

@unittest.skipUnless(torch.cuda.is_available(),'CUDA required')
class GatherTests(unittest.TestCase):
    def setUp(self):
        # Physical pool: 8 slots, 2 heads, 128 dimensions. Distinct K/V values.
        x=torch.arange(8*2*128,device='cuda:0',dtype=torch.float32).reshape(8,2,128)
        self.k=x.to(torch.bfloat16)
        self.v=(-x-16).to(torch.bfloat16)
        self.req=torch.tensor([5,1,7,0,6,2],device='cuda:0',dtype=torch.int32)

    def reference(self,ids,head,k=None,v=None):
        k=self.k if k is None else k; v=self.v if v is None else v
        valid=ids>=0
        slots=self.req[ids.clamp_min(0).long()].long()
        a=k[slots,head].masked_fill(~valid[...,None],0).unsqueeze(2)
        b=v[slots,head].masked_fill(~valid[...,None],0).unsqueeze(2)
        return a,b

    def test_small_kernel(self):
        # A padded row stride and noncontiguous pool dimensions test real addressing.
        backing=torch.tensor([[0,2,-1,99],[4,1,4,99]],device='cuda:0',dtype=torch.int32)
        ids=backing[:,:3]
        kp=torch.empty((8,2,256),device='cuda:0',dtype=torch.bfloat16)[:,:,::2]; kp.copy_(self.k)
        vp=torch.empty((8,4,128),device='cuda:0',dtype=torch.bfloat16)[:,::2,:]; vp.copy_(self.v)
        before_k=kp.clone();before_v=vp.clone()
        out_k=torch.empty((2,3,1,128),device='cuda:0',dtype=torch.bfloat16)
        out_v=torch.empty_like(out_k)
        for head in (0,1):
            mod._gather_slots[(1,)](kp,vp,self.req,ids,out_k,out_v,2,3,head,
                *kp.stride(),*vp.stride(),ids.stride(0),32,num_warps=4)
            expected_k,expected_v=self.reference(ids,head,kp,vp)
            torch.testing.assert_close(out_k,expected_k,rtol=0,atol=0)
            torch.testing.assert_close(out_v,expected_v,rtol=0,atol=0)
        torch.testing.assert_close(kp,before_k,rtol=0,atol=0)
        torch.testing.assert_close(vp,before_v,rtol=0,atol=0)
        print('small example: logical [[0,2,-1],[4,1,4]] -> physical [[5,7,padding],[6,1,6]]')

    def test_real_workspace_reuse(self):
        # Production wrapper fixes TOP=2048; only two Q rows are needed.
        ids=(torch.arange(4096,device='cuda:0')%6).to(torch.int32).reshape(2,2048)
        work=mod.GatherWorkspace(2,'cuda:0',torch.bfloat16)
        a,b=work.gather(self.k,self.v,self.req,ids,0)
        expected=self.reference(ids,0)
        torch.testing.assert_close(a,expected[0],rtol=0,atol=0)
        torch.testing.assert_close(b,expected[1],rtol=0,atol=0)
        addresses=(a.data_ptr(),b.data_ptr())
        changed=ids.flip(1).contiguous()
        a,b=work.gather(self.k,self.v,self.req,changed,1)
        self.assertEqual(addresses,(a.data_ptr(),b.data_ptr()))
        expected=self.reference(changed,1)
        torch.testing.assert_close(a,expected[0],rtol=0,atol=0)
        torch.testing.assert_close(b,expected[1],rtol=0,atol=0)
        work.clear();self.assertIsNone(work.k);self.assertIsNone(work.v)
        print('workspace PASS: [2,2048,1,128], changed IDs/head, buffers reused, clear releases references')

if __name__=='__main__':unittest.main()
