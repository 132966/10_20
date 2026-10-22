from common import *
from runtime import Build

def check(a,b):
 for name in ['unit','mapping','tmp','tc','hash','match','tables','new','refs','count','free','fine','centers','members','hashes']:
  x=getattr(a,name);y=getattr(b,name)
  if name=='unit':x=x[:1024];y=y[:1024]
  assert torch.equal(x,y),name
 assert torch.equal(a.perm[1][:1024],b.perm[1][:1024]),'perm'

def main():
 torch.manual_seed(88);records=[]
 for cap in [32,4096]:
  native=torch.empty((64,8448),device='cuda',dtype=torch.uint8)
  native[:,:8192]=torch.randn((64,8192),device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
  native.view(torch.float32)[:,2048:]=torch.rand((64,64),device='cuda')+.01
  table=torch.randperm(64,device='cuda',dtype=torch.int32)[:16].contiguous();meta=tensor([4096,0,1])
  old=Build(native,table,meta,cap,optimized=False);new=Build(native,table,meta,cap,optimized=True)
  for i in range(6):
   table.copy_(torch.randperm(64,device='cuda',dtype=torch.int32)[:16]);meta[0]=i*1024;meta[1]=0
   old.run();new.run();check(old,new)
   # Same bucket content under another lease must share pages.
   meta[1]=1;old.run();new.run();check(old,new)
   assert not bool(new.new.any()) and int(new.count)==cap-16
   old.release(0);new.release(0);old.release(1);new.release(1)
   assert int(new.count)==cap and bool((new.refs==0).all())
  meta[1]=0
  # Capture complete build/release lifecycle. Dynamic source table and metadata.
  graphs=[]
  for plan in [old,new]:
   plan.cycle();torch.cuda.synchronize()
   g=torch.cuda.CUDAGraph()
   with torch.cuda.graph(g):plan.cycle()
   graphs.append(g)
  for i in range(4):
   table.copy_(torch.randperm(64,device='cuda',dtype=torch.int32)[:16]);meta[0]=16384+i*1024
   graphs[0].replay();graphs[1].replay();check(old,new)
   assert int(new.count)==cap
  # Inactive build skips all writes except FLAG, without release of nonexistent work.
  meta[2]=0
  snapshots={name:getattr(new,name).clone() for name in ['tmp','hash','refs','count','fine','centers']}
  new.run()
  for name,x in snapshots.items():assert torch.equal(x,getattr(new,name)),name
  meta[2]=1
  old.initialize();new.initialize()
  # reset permutation inside each grouping timing so every replay starts identically
  def old_group():old.initialize();old.grouping()
  def new_group():new.initialize();new.grouping()
  timing=bench({'original':old.cycle,'optimized':new.cycle},repeats=5)
  gt=bench({'original':old_group,'optimized':new_group},repeats=5)
  records.append({'cap':cap,'build_and_release':timing,'initialize_and_grouping':gt})
  print('PASS BUILD',records[-1],flush=True)
 save('build_results.json',{'gpu':torch.cuda.get_device_name(0),'records':records,'exact':True,'graph_replay_dynamic':True,'note':'build timing includes release for a repeatable fresh allocation lifecycle; no CPU uploads inside timed run'})
if __name__=='__main__':main()
