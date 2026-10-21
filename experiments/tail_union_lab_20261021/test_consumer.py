import json
import torch
from test_bench import Case,graph,eager,ROOT
from consumer import score_tail
c=Case(65549,1024,17,2049)
q=torch.randn((1024,32,128),device='cuda',dtype=torch.bfloat16)
w=torch.rand((1024,32),device='cuda')/32
out0=torch.empty((1024,c.scratch*64),device='cuda');out1=torch.empty_like(out0)
def score(shared):
    k=c.u if shared else c.old
    score_tail[(1024,c.scratch)](q,w,k,k.view(torch.float32),c.lens,c.d,out1 if shared else out0,c.s,c.w,c.scratch,shared,num_warps=4)
c.baseline();c.opt();score(False);score(True)
torch.testing.assert_close(out0,out1,rtol=0,atol=0)
# Independent PyTorch FP32 reference for first/middle/last Q; variable candidate counts.
for r in [0,511,1023]:
    s,t,count=c.bs[r];ids=torch.tensor(list(range(s))+list(range(t,int(c.lens[r]))),device='cuda')
    pp=c.nt[ids//64].long();off=ids%64
    raw=c.pool[pp[:,None],off[:,None]*128+torch.arange(128,device='cuda')[None,:]].contiguous()
    k=raw.view(torch.float8_e4m3fn).float()
    scales=c.pool.view(torch.float32)[pp,2048+off]
    ref=(torch.relu(q[r].float()@k.T)*w[r,:,None]).sum(0)*scales
    torch.testing.assert_close(out1[r,:count],ref,rtol=2e-5,atol=2e-5)
    assert torch.isneginf(out1[r,count:]).all()
def old():c.baseline();score(False)
def new():c.opt();score(True)
a=graph(old);b=graph(new)
e=eager(old);f=eager(new)
result={'case':c.args,'Q_shape':list(q.shape),'exact_old_new_scores':'PASS','pytorch_reference':'PASS','graph_old_us':a*1000,'graph_new_us':b*1000,'graph_speedup':a/b,'eager_old_us':e*1000,'eager_new_us':f*1000,'eager_speedup':e/f,'scope':'staging + demonstration tail scoring only, not complete query or attention'}
(ROOT/'consumer_results.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
