from common import *
import optimized as opt

def main():
 torch.manual_seed(77);records=[]
 for cap in [32,257,4096]:
  tmp=torch.randint(0,256,(16,8448),device='cuda',dtype=torch.uint8);mp=torch.randint(-1,262144,(16,64),device='cuda',dtype=torch.int32)
  meta=tensor([4096,1,1]);h=torch.empty((16,2),device='cuda',dtype=torch.int64);h2=torch.empty_like(h)
  tmp32=tmp.view(torch.int32)
  def old_hash():v.fingerprint[(16,)](tmp,mp,meta,h)
  def new_hash():opt.fingerprint_words[(16,)](tmp32,mp,meta,h2)
  for seed in range(5):
   tmp.random_(0,256);mp.random_(-1,262144);old_hash();new_hash();assert torch.equal(h,h2)
  records.append({'op':'fingerprint','cap':cap,'timing':bench({'original':old_hash,'words':new_hash})})
  fine=torch.randint(0,256,(cap,8448),device='cuda',dtype=torch.uint8);fine32=fine.view(torch.int32)
  members=torch.zeros((cap,64),device='cuda',dtype=torch.int32);hashes=torch.zeros((cap,2),device='cuda',dtype=torch.int64)
  refs=torch.zeros(cap,device='cuda',dtype=torch.int32);m0=torch.empty(16,device='cuda',dtype=torch.int32);m1=torch.empty_like(m0)
  def old_match():v.match_pages[(16,)](tmp,mp,h,meta,fine,members,hashes,refs,m0,cap,triton.next_power_of_2(cap))
  def new_match():opt.match_words[(16,)](tmp32,mp,h,meta,fine32,members,hashes,refs,m1,cap,triton.next_power_of_2(cap))
  for mode in ['miss','hit','collision']:
   refs.zero_();hashes.zero_()
   if mode!='miss':
    fine[1:17].copy_(tmp);members[1:17].copy_(mp);hashes[1:17].copy_(h);refs[1:17]=1
   if mode=='collision':
    fine[0].copy_(tmp[0]);fine[0,0]^=1;members[0].copy_(mp[0]);hashes[0].copy_(h[0]);refs[0]=1
   old_match();new_match();assert torch.equal(m0,m1)
   expected=torch.full_like(m0,-1) if mode=='miss' else torch.arange(1,17,device='cuda',dtype=torch.int32)
   assert torch.equal(m0,expected)
   records.append({'op':'match_'+mode,'cap':cap,'timing':bench({'original':old_match,'words':new_match})})
  tc=torch.randint(0,256,(2,8448),device='cuda',dtype=torch.uint8);tc32=tc.view(torch.int32);tcs=tc.view(torch.float32)
  tab=torch.zeros((2,16),device='cuda',dtype=torch.int32);tab[1]=torch.arange(1,17,device='cuda')
  new=torch.ones(16,device='cuda',dtype=torch.bool)
  pools=[]
  for _ in range(2):
   f=torch.full_like(fine,19);c=torch.full((4,8448),19,device='cuda',dtype=torch.uint8);m=torch.full_like(members,-99);hh=torch.full_like(hashes,-99)
   pools.append((f,c,m,hh,f.view(torch.int32),c.view(torch.int32),c.view(torch.float32)))
  def old_pub():
   f,c,m,hh,*_=pools[0]
   v.publish[(16,)](tmp,tc,tcs,mp,h,meta,tab,new,f,c,pools[0][6],m,hh)
  def new_pub():
   f,c,m,hh,fw,cw,_=pools[1]
   opt.publish_words[(16,)](tmp32,tc32,mp,h,meta,tab,new,fw,cw,m,hh)
  for mode in ['all_new','mixed','reuse']:
   new.fill_(mode!='reuse')
   if mode=='mixed':new[::2]=False
   for pool in pools:
    pool[0].fill_(19);pool[1].fill_(19);pool[2].fill_(-99);pool[3].fill_(-99)
   old_pub();new_pub()
   for x,y in zip(pools[0][:4],pools[1][:4]):assert torch.equal(x,y)
   records.append({'op':'publish_'+mode,'cap':cap,'timing':bench({'original':old_pub,'words':new_pub})})
  # Inactive graph replay must leave outputs intact.
  new_hash();new_match();new_pub();torch.cuda.synchronize()
  graph=torch.cuda.CUDAGraph()
  with torch.cuda.graph(graph):new_hash();new_match();new_pub()
  meta[2]=0;h2.fill_(-99);m1.fill_(-99)
  before=[x.clone() for x in pools[1][:4]];graph.replay()
  assert bool((h2==-99).all()) and bool((m1==-99).all())
  for x,y in zip(before,pools[1][:4]):assert torch.equal(x,y)
  meta[2]=1;tmp.random_(0,256);old_hash();graph.replay();assert torch.equal(h,h2)
  print('PASS pages capacity',cap,flush=True)
 save('page_results.json',{'gpu':torch.cuda.get_device_name(0),'records':records})
 print(json.dumps(records),flush=True)
if __name__=='__main__':main()
