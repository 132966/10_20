"""Small, independent tests of every Triton kernel in fused_grouping.py.
Run with the original torch/triton environment: python test_fused_grouping_small.py -v
No model, service, pytest, or SGLang package import is needed.
"""
import importlib.util
from pathlib import Path
import unittest
import torch

spec = importlib.util.spec_from_file_location("tested_fused_grouping", Path(__file__).with_name("fused_grouping.py"))
fg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fg)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU required")
class SmallKernelTests(unittest.TestCase):
    def setUp(self):
        self.device = "cuda:0"
        self.start = 100
        self.flag = self.tensor(-1, torch.int32)

    def tensor(self, values, dtype=torch.float32):
        return torch.tensor(values, dtype=dtype, device=self.device)

    def exact(self, actual, expected):
        actual = actual.cpu()
        expected = torch.as_tensor(expected, dtype=actual.dtype)
        self.assertTrue(torch.equal(actual, expected), f"actual={actual}, expected={expected}")

    def test_publish_exact(self):
        src = self.tensor([9, 2, 7, 4, 1], torch.int32)
        dst = torch.full((8,), -99, device=self.device, dtype=torch.int32)
        fg._publish_exact[(1,)](src, dst, self.flag, 5)
        self.exact(dst, [9, 2, 7, 4, 1, -99, -99, -99])
        self.flag.fill_(0)
        src.fill_(88)
        fg._publish_exact[(1,)](src, dst, self.flag, 5)
        self.exact(dst, [9, 2, 7, 4, 1, -99, -99, -99])

    def test_init(self):
        # One program initializes 16 members. Head=0 avoids fixed 4K head strides.
        k = torch.zeros((16,1,128), device=self.device)
        k[:,0,0] = torch.arange(16,device=self.device)*3
        k[:,0,1] = torch.arange(16,device=self.device)*4
        slots = torch.arange(15,-1,-1,device=self.device,dtype=torch.int32)
        unit = torch.full((16,128), -99.,device=self.device)
        perm = torch.full((16,), -99,device=self.device,dtype=torch.int32)
        fg._init[(1,1)](k,slots,unit,perm,self.flag,0,k.stride(0),k.stride(1),num_warps=4,enable_fp_fusion=False)
        expected = torch.zeros((16,128))
        expected[:15,0]=0.6; expected[:15,1]=0.8
        torch.testing.assert_close(unit.cpu(),expected)
        self.exact(perm, list(range(16)))
        self.flag.fill_(0); unit.fill_(-99); perm.fill_(-99)
        fg._init[(1,1)](k,slots,unit,perm,self.flag,0,k.stride(0),k.stride(1),num_warps=4,enable_fp_fusion=False)
        self.assertTrue(bool((unit==-99).all())); self.exact(perm,[-99]*16)

    def test_dots_three_modes(self):
        # Nontrivial P: member IDs, rather than positions, select vectors.
        unit=torch.zeros((16,128),device=self.device)
        signs=self.tensor([1,-1]*8); unit[:,0]=signs
        order=[2,1,4,3,6,5,8,7,10,9,12,11,14,13,0,15]
        perm=self.tensor([self.start+i for i in order],torch.int32)
        anchor=self.tensor([[self.start+1,self.start+2]],torch.int32) # b negative, a positive
        scores=torch.empty(16,device=self.device)
        for mode in (0,1,2):
            with self.subTest(mode=mode):
                # MODE=2 consumes the previous dot(K,b), not an empty buffer.
                scores.copy_(-signs[self.tensor(order,torch.int64)] if mode==2 else torch.zeros_like(scores))
                fg._dots[(1,1)](unit,perm,anchor,scores,self.flag,16,self.start,mode,num_warps=4,enable_fp_fusion=False)
                expected=[(1 if i%2==0 else -1)*([1,-1,2][mode]) for i in order]
                self.exact(scores,expected)
                self.flag.fill_(0); scores.fill_(77)
                fg._dots[(1,1)](unit,perm,anchor,scores,self.flag,16,self.start,mode,num_warps=4,enable_fp_fusion=False)
                self.exact(scores,[77]*16); self.flag.fill_(-1)

    def test_anchor_left_tie(self):
        perm=self.tensor([115,101,102,100]+list(range(103,115)),torch.int32)
        scores=self.tensor([4,-3,2,-3]+[5]*12)
        anchor=torch.full((1,2),-99,device=self.device,dtype=torch.int32)
        for slot in (0,1):
            fg._anchor[(1,1)](perm,anchor,scores,self.flag,16,slot,num_warps=4)
        # Leftmost minimum is position 1 -> member 101, not smaller ID 100.
        self.exact(anchor,[[101,101]])
        self.flag.fill_(0); anchor.fill_(-99)
        fg._anchor[(1,1)](perm,anchor,scores,self.flag,16,0,num_warps=4)
        self.exact(anchor,[[-99,-99]])

    def test_sort_stable_and_signed_zero(self):
        values=[2.,-1.,2.,0.,-0.,3.,-3.,1.,1.,2.,-1.,0.,4.,4.,-4.,0.]
        ids=[115,102,108,101,109,103,114,100,113,104,112,105,111,106,110,107]
        perm=self.tensor(ids,torch.int32); scores=self.tensor(values)
        out=torch.full((16,),-99,device=self.device,dtype=torch.int32)
        fg._sort[(1,1)](perm,out,scores,self.flag,16,num_warps=4)
        expected=[ids[i] for i in sorted(range(16),key=lambda i:(-values[i],i))]
        self.exact(out,expected)
        self.exact(perm,ids) # input permutation stays unchanged
        self.flag.fill_(0); out.fill_(-99)
        fg._sort[(1,1)](perm,out,scores,self.flag,16,num_warps=4)
        self.exact(out,[-99]*16)

    def test_one_split_chain(self):
        # Actual three dots/two anchors/one sort sequence: 16 -> two groups of 8.
        unit=torch.zeros((16,128),device=self.device); unit[:,0]=self.tensor([1,-1]*8)
        perm=self.tensor(list(range(100,116)),torch.int32)
        anchor=torch.empty((1,2),device=self.device,dtype=torch.int32)
        scores=torch.empty(16,device=self.device)
        out=torch.empty_like(perm)
        for mode in (0,1,2):
            fg._dots[(1,1)](unit,perm,anchor,scores,self.flag,16,100,mode,num_warps=4,enable_fp_fusion=False)
            if mode<2:
                fg._anchor[(1,1)](perm,anchor,scores,self.flag,16,mode,num_warps=4)
        fg._sort[(1,1)](perm,out,scores,self.flag,16,num_warps=4)
        self.exact(anchor,[[101,100]])
        self.exact(out,list(range(100,116,2))+list(range(101,116,2)))
        print("\n16-member split:",out.cpu().reshape(2,8).tolist())


if __name__ == "__main__":
    unittest.main()
