from common import *
import optimized as opt

def main():
 torch.manual_seed(432);records=[]
 u=torch.randn((4096,128),device='cuda');u/=u.norm(dim=-1,keepdim=True).clamp_min(1e-12)
 p=torch.zeros(4096,device='cuda',dtype=torch.int32);p[:1024]=torch.randperm(1024,device='cuda',dtype=torch.int32)
 a=torch.empty((256,2),device='cuda',dtype=torch.int32);s=torch.empty(4096,device='cuda');s0=torch.empty_like(s);s1=torch.empty_like(s);o=torch.empty_like(p);flag=tensor([-1])
 for n in [1024,512,256,128,64,32,16]:
  g=1024//n
  def ref():
   for mode in (0,1,2):
    fg._dots[(64,1)](u,p,a,s,flag,n,0,mode,num_warps=4)
    if mode<2:fg._anchor[(g,1)](p,a,s,flag,n,mode,num_warps=4)
   fg._sort[(g,1)](p,o,s,flag,n,num_warps=4)
  def partial():
   fg._dots[(64,1)](u,p,a,s0,flag,n,0,0,num_warps=4)
   opt.anchored_dots[(64,)](u,p,s0,s1,a,flag,n,1,num_warps=4)
   opt.anchored_dots[(64,)](u,p,s1,s,a,flag,n,2,num_warps=4)
   fg._sort[(g,1)](p,o,s,flag,n,num_warps=4)
  fns={'original':ref,'partial':partial}
  if n<=256:
   for w in [4,8,16]:
    fns['full_w'+str(w)]=lambda w=w:opt.split_fused[(g,)](u,p,o,a,s,flag,n,num_warps=w)
  for seed in range(10):
   u.normal_();u/=u.norm(dim=-1,keepdim=True).clamp_min(1e-12)
   if seed==8:u.zero_()
   if seed==9:u.zero_();u[:,0]=1
   p[:1024]=torch.randperm(1024,device='cuda',dtype=torch.int32)
   ref();ro=o[:1024].clone();ra=a[:g].clone();rs=s[:1024].clone()
   for name,fn in fns.items():
    fn()
    assert torch.equal(ro,o[:1024]) and torch.equal(ra,a[:g]) and torch.equal(rs,s[:1024]),(n,seed,name)
  u.normal_();u/=u.norm(dim=-1,keepdim=True).clamp_min(1e-12)
  t=bench(fns);records.append({'N':n,'timing':t,'exact':True});print(records[-1],flush=True)
 save('tuned_group_results.json',records)
if __name__=='__main__':main()
