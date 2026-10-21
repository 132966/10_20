import json,statistics,hashlib
from pathlib import Path
import torch
import triton as tr
from baseline import stage_tail_shared
from union import launch,reconstruct_for_test
ROOT=Path(__file__).parent
torch.manual_seed(20261021)
CONFIGS=[(256,False,4),(512,False,4),(1024,False,4),(2048,False,4),(4096,True,4),(4096,True,8)]
def bounds(n,s,w):
    s=min(n,s);t=max(s,min(max(n-1,0)//1024*1024,max(s,n-w)));return s,t,s+n-t
class Case:
    def __init__(self,h,r,s,w):
        self.args=(h,r,s,w);self.r=r;self.s=s;self.w=w
        self.lens=torch.arange(h+1,h+r+1,device='cuda',dtype=torch.int32)
        self.np=max(1,tr.cdiv(h+r,64))
        self.pool=torch.empty((self.np,8448),device='cuda',dtype=torch.uint8)
        self.pool[:,:8192]=torch.randn((self.np,8192),device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
        self.pool.view(torch.float32)[:,2048:]=torch.rand((self.np,64),device='cuda')+.01
        self.nt=torch.randperm(self.np,device='cuda',dtype=torch.int32)
        self.nts=self.nt.repeat(r,1).contiguous()
        self.bs=[bounds(n,s,w) for n in self.lens.cpu().tolist()]
        self.scratch=max(1,tr.cdiv(max(b[2] for b in self.bs),64))
        self.old=torch.full((r*self.scratch,8448),165,device='cuda',dtype=torch.uint8)
        self.restored=torch.full_like(self.old,165)
        cap=min(self.np,tr.cdiv(s,64)+tr.cdiv(w+1024+r,64)+2)
        self.u=torch.empty((cap,8448),device='cuda',dtype=torch.uint8)
        self.m=torch.empty((r,3),device='cuda',dtype=torch.int32);self.d=torch.empty(4,device='cuda',dtype=torch.int32)
        self.owners=torch.arange(r,device='cuda',dtype=torch.int32)
    def baseline(self,g=1):
        stage_tail_shared[(tr.cdiv(self.r,g),self.scratch)](self.pool,self.pool.view(torch.float32),self.nts,self.lens,self.old,self.old.view(torch.float32),self.owners,0,self.np,self.scratch,self.s,self.w,self.r,g,num_warps=4)
    def opt(self,cfg=(512,False,4)):
        launch(self.pool,self.nt,self.lens,self.u,self.m,self.d,self.s,self.w,*cfg)
    def reconstruct(self):
        reconstruct_for_test[(self.r,self.scratch)](self.u,self.u.view(torch.float32),self.m,self.d,self.restored,self.restored.view(torch.float32),self.scratch,num_warps=4)
    def validate(self,cfg):
        self.old.fill_(165);self.restored.fill_(165);self.baseline();self.opt(cfg);self.reconstruct()
        torch.testing.assert_close(self.old,self.restored,rtol=0,atol=0)
        assert self.m.cpu().tolist()==[list(x) for x in self.bs]
        sp,tp,ep,total=self.d.cpu().tolist();ids=list(range(sp))+list(range(tp,ep))
        torch.testing.assert_close(self.u[:total],self.pool[self.nt[ids].long()],rtol=0,atol=0)
        for n,(s,t,c) in zip(self.lens.cpu().tolist(),self.bs):
            ids=list(range(s))+list(range(t,n))
            assert len(ids)==len(set(ids))==c and all(0<=j<n for j in ids)
def eager(fn):
    for _ in range(5):fn()
    torch.cuda.synchronize();times=[]
    for _ in range(5):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record()
        for _ in range(50):fn()
        b.record();b.synchronize();times.append(a.elapsed_time(b)/50)
    return statistics.median(times)
def graph(fn):return tr.testing.do_bench_cudagraph(fn,rep=50)
def main():
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'triton':tr.__version__,'tests':[],'results':[]}
    for args in [(0,1,64,2048),(0,65,17,0),(17,129,65,128),(1000,65,64,128),(2048,1024,64,128),(65536,1024,64,2048),(65549,1024,17,2049),(131072,1024,0,0),(262144,4096,64,2048)]:
        c=Case(*args)
        for cfg in CONFIGS:c.validate(cfg)
        report['tests'].append({'case':args,'configs':len(CONFIGS),'status':'PASS'})
        print('PASS',args,flush=True)
    c=Case(65536,1024,64,2048);c.opt();torch.cuda.synchronize()
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):c.opt()
    c.lens.sub_(37);c.nt.copy_(c.nt.flip(0));c.nts.copy_(c.nt.expand_as(c.nts))
    g.replay();c.baseline();c.reconstruct()
    bb=[bounds(n,64,2048) for n in c.lens.cpu().tolist()]
    for r,(_,_,cnt) in enumerate(bb):
        lo=r*c.scratch;hi=lo+max(1,tr.cdiv(cnt,64))
        torch.testing.assert_close(c.old[lo:hi],c.restored[lo:hi],rtol=0,atol=0)
    assert c.m.cpu().tolist()==[list(x) for x in bb]
    report['dynamic_graph_replay']='PASS';print('PASS dynamic graph replay',flush=True)
    for args in [(65536,1024,64,2048),(131072,1024,64,2048),(262144,1024,64,2048),(65549,1024,17,2049),(65536,4096,64,2048),(65536,1024,64,128)]:
        c=Case(*args)
        timings=[(graph(lambda cfg=cfg:c.opt(cfg)),cfg) for cfg in CONFIGS];best=min(timings)[1]
        bt=[(graph(lambda g=g:c.baseline(g)),g) for g in [1,2,4,8]];bg=min(bt)[1]
        old=lambda:c.baseline(bg);new=lambda:c.opt(best)
        og=graph(old);ng=graph(new);oe=eager(old);ne=eager(new);new();torch.cuda.synchronize()
        rr={'case':args,'best_config':best,'baseline_group':bg,'old_graph_us':og*1000,'new_graph_us':ng*1000,'graph_speedup':og/ng,'old_eager_us':oe*1000,'new_eager_us':ne*1000,'eager_speedup':oe/ne,'old_reserved_bytes':c.old.numel(),'new_reserved_bytes':c.u.numel()+c.m.numel()*4+16,'old_written_bytes':sum(max(1,tr.cdiv(b[2],64))*8448 for b in c.bs),'new_written_bytes':int(c.d[3])*8448+c.r*12+16,'tuning_graph_us':[(t*1000,cfg) for t,cfg in timings],'baseline_tuning_graph_us':[(t*1000,g) for t,g in bt]}
        report['results'].append(rr);(ROOT/'results.json').write_text(json.dumps(report,indent=2));print(json.dumps(rr),flush=True)
    p=json.loads((ROOT/'provenance.json').read_text());assert hashlib.sha256(Path(p['source']).read_bytes()).hexdigest()==p['sha256']
    report['original_source_unchanged']=True;(ROOT/'results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
