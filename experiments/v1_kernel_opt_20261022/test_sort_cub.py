from common import *
from sort_cub import sort,module

def main():
 module();torch.manual_seed(2026);records=[]
 p=torch.randperm(4096,device='cuda',dtype=torch.int32);o=torch.empty_like(p);ref=torch.empty_like(p)
 s=torch.randn(4096,device='cuda');flag=tensor([-1])
 for n in [1024,512,256,128,64,32,16]:
  def old():fg._sort[(1024//n,1)](p,ref,s,flag,n,num_warps=4)
  def new():sort(p,o,s,flag,n)
  for seed in range(10):
   if seed<5:s.normal_()
   else:s.copy_(torch.tensor([0.,-0.,1.,1.,-1.,-float('inf'),float('inf'),0.],device='cuda').repeat(512))
   old();new();assert torch.equal(o[:1024],ref[:1024]),(n,seed)
  flag.fill_(0);o.fill_(-99);new();assert bool((o==-99).all());flag.fill_(-1)
  g=torch.cuda.CUDAGraph()
  with torch.cuda.graph(g):new()
  s.normal_();g.replay();old();assert torch.equal(o[:1024],ref[:1024])
  t=bench({'triton':old,'cub':new});records.append({'N':n,'timing':t});print(records[-1],flush=True)
 save('sort_results.json',records)
if __name__=='__main__':main()
