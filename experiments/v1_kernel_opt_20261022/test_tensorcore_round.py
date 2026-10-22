from common import *
import optimized as opt

def main():
 torch.manual_seed(909);records=[]
 u=torch.empty((4096,128),device='cuda');p=torch.arange(4096,device='cuda',dtype=torch.int32)
 a=torch.empty((256,2),device='cuda',dtype=torch.int32);s=torch.empty(4096,device='cuda');o=torch.empty_like(p);flag=tensor([-1])
 for n in [1024,128,16]:
  g=1024//n;bm=min(n,32)
  def old():
   for mode in range(3):
    fg._dots[(64,1)](u,p,a,s,flag,n,0,mode,num_warps=4)
    if mode<2:fg._anchor[(g,1)](p,a,s,flag,n,mode,num_warps=4)
   fg._sort[(g,1)](p,o,s,flag,n,num_warps=4)
  def tc():
   for mode in range(3):
    opt.dots_tc[(g,triton.cdiv(n,bm))](u,p,a,s,flag,n,bm,mode,num_warps=4)
    if mode<2:fg._anchor[(g,1)](p,a,s,flag,n,mode,num_warps=4)
   fg._sort[(g,1)](p,o,s,flag,n,num_warps=4)
  changes={}
  for kind in ['random','near_ties']:
   mismatch=0
   for seed in range(10):
    if kind=='random':u.normal_()
    else:u.copy_(torch.randn((1,128),device='cuda')+torch.randn_like(u)*1e-6)
    u/=u.norm(dim=-1,keepdim=True).clamp_min(1e-12);p[:1024]=torch.randperm(1024,device='cuda',dtype=torch.int32)
    old();ref=o[:1024].clone();tc();mismatch+=not torch.equal(ref,o[:1024])
   changes[kind]=mismatch
  u.normal_();u/=u.norm(dim=-1,keepdim=True).clamp_min(1e-12)
  r={'N':n,'mismatch_trials_out_of_10':changes,'timing':bench({'original':old,'tensorcore':tc})}
  records.append(r);print(r,flush=True)
 save('tensorcore_round_results.json',records)
if __name__=='__main__':main()
