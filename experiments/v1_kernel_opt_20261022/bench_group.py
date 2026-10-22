from common import *
import optimized as opt

def main():
 torch.manual_seed(123)
 u=torch.randn((1,4096,128),device='cuda');u/=u.norm(dim=-1,keepdim=True).clamp_min(1e-12)
 p=torch.zeros(4096,device='cuda',dtype=torch.int32);p[:1024]=torch.randperm(1024,device='cuda',dtype=torch.int32)
 a=torch.empty((256,2),device='cuda',dtype=torch.int32);s=torch.empty(4096,device='cuda');o=torch.empty_like(p);flag=tensor([-1])
 records=[];tc=[]
 for n in [1024,512,256,128,64,32,16]:
  g=1024//n
  def ref():
   for mode in (0,1,2):
    fg._dots[(64,1)](u,p,a,s,flag,n,0,mode,num_warps=4)
    if mode<2:fg._anchor[(g,1)](p,a,s,flag,n,mode,num_warps=4)
   fg._sort[(g,1)](p,o,s,flag,n,num_warps=4)
  def fused():opt.split_fused[(g,)](u,p,o,a,s,flag,n,num_warps=8 if n>=256 else 4)
  exact=True;maxerr=0.
  for kind in range(6):
   if kind<4:u.normal_();u/=u.norm(dim=-1,keepdim=True).clamp_min(1e-12)
   elif kind==4:u.zero_()
   else:u.zero_();u[0,:,0]=1
   p[:1024]=torch.randperm(1024,device='cuda',dtype=torch.int32)
   ref();ro=o[:1024].clone();ra=a[:g].clone();rs=s[:1024].clone()
   fused();exact &= torch.equal(ro,o[:1024]) and torch.equal(ra,a[:g]) and torch.equal(rs,s[:1024])
   maxerr=max(maxerr,(rs-s[:1024]).abs().max().item())
  u.normal_();u/=u.norm(dim=-1,keepdim=True).clamp_min(1e-12)
  timing=bench({'original':ref,'fused':fused})
  records.append({'N':n,'exact':exact,'score_max_error':maxerr,'timing':timing})
  print('GROUP',records[-1],flush=True)
  flag.fill_(0);o.fill_(-99);fused();assert bool((o==-99).all());flag.fill_(-1)
  # Tensor Core and library candidate: MODE2 with valid fixed anchors.
  ref();gold=s[:1024].clone()
  def dots_ref():
   fg._dots[(64,1)](u,p,a,s,flag,n,0,1,num_warps=4)
   fg._dots[(64,1)](u,p,a,s,flag,n,0,2,num_warps=4)
  def tc_fn():return opt.dots_tc[(g,triton.cdiv(n,min(n,32)))](u,p,a,s,flag,n,min(n,32),2,num_warps=4)
  kernel=tc_fn();err=(gold-s[:1024]).abs().max().item();same=torch.equal(gold,s[:1024])
  x=torch.empty((g,n,128),device='cuda');y=torch.empty((g,128,2),device='cuda');z=torch.empty((g,n,2),device='cuda')
  def lib_fn():
   opt.pack_bmm[(g,n//16)](u,p,a,x,y,n,num_warps=4)
   torch.bmm(x,y,out=z)
   opt.finish_bmm[(4,)](z,s,num_warps=4)
  lib_fn();liberr=(gold-s[:1024]).abs().max().item()
  t=bench({'original_two_dots':dots_ref,'tensorcore_tf32x3':tc_fn,'bmm_with_packing':lib_fn})
  tc.append({'N':n,'tensorcore_exact':same,'max_error':err,'bmm_max_error':liberr,'mma_in_ptx':'mma.sync' in kernel.asm['ptx'],'timing':t})
  print('TC',tc[-1],flush=True)
 save('group_results.json',{'gpu':torch.cuda.get_device_name(0),'group':records,'tensorcore':tc})
if __name__=='__main__':main()
