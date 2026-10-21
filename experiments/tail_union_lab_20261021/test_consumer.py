import json
import torch
from test_bench import Case,graph,eager,ROOT
from consumer import score_tail
results=[]
for args in [(65549,1024,17,2049),(65536,4096,64,2048)]:
    c=Case(*args);rows=c.r
    q=torch.randn((rows,32,128),device='cuda',dtype=torch.bfloat16)
    w=torch.rand((rows,32),device='cuda')/32
    out0=torch.empty((rows,c.scratch*64),device='cuda');out1=torch.empty_like(out0)
    def score(shared):
        k=c.u if shared else c.old
        score_tail[(rows,c.scratch)](q,w,k,k.view(torch.float32),c.lens,c.end,c.d,out1 if shared else out0,c.s,c.w,c.scratch,shared,num_warps=4)
    c.baseline();c.opt();score(False);score(True)
    torch.testing.assert_close(out0,out1,rtol=0,atol=0)
    for r in [0,rows//2,rows-1]:
        s,t,count=c.bs[r];ids=torch.tensor(list(range(s))+list(range(t,int(c.lens[r]))),device='cuda')
        pp=c.nt[ids//64].long();off=ids%64
        raw=c.pool[pp[:,None],off[:,None]*128+torch.arange(128,device='cuda')[None,:]].contiguous()
        k=raw.view(torch.float8_e4m3fn).float();scales=c.pool.view(torch.float32)[pp,2048+off]
        ref=(torch.relu(q[r].float()@k.T)*w[r,:,None]).sum(0)*scales
        torch.testing.assert_close(out1[r,:count],ref,rtol=2e-5,atol=2e-5)
        assert torch.isneginf(out1[r,count:]).all()
    def old():c.baseline();score(False)
    def new():c.opt();score(True)
    a=graph(old);b=graph(new);e=eager(old);f=eager(new)
    result={'case':c.args,'indexed_end':c.end_cpu,'Q_shape':list(q.shape),'exact_old_new_scores':'PASS','pytorch_reference':'PASS','graph_old_us':a*1000,'graph_new_us':b*1000,'graph_speedup':a/b,'eager_old_us':e*1000,'eager_new_us':f*1000,'eager_speedup':e/f,'scope':'corrected per-Q staging + demonstration tail scoring, not complete query or attention'}
    if rows==4096:
        # A high-score unindexed token outside last Q's window must reach TopK.
        logical=65537;phys=int(c.nt[logical//64]);off=logical%64
        c.pool[phys,off*128]=torch.tensor(448.,device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
        c.pool.view(torch.float32)[phys,2048+off]=1.
        q[-1].zero_();q[-1,:,0]=1.;w[-1].fill_(1/32)
        c.baseline();c.opt();score(False);score(True)
        torch.testing.assert_close(out0,out1,rtol=0,atol=0)
        idx=torch.topk(out1[-1],2048).indices
        s,t,count=c.bs[-1]
        chosen=torch.where(idx<s,idx,t+idx-s)
        assert (chosen==logical).any()
        assert int(out1[-1].argmax())==s+logical-t
        result['unindexed_65537_reaches_last_Q_top2048']='PASS'
    results.append(result);print(json.dumps(result),flush=True)
(ROOT/'consumer_results.json').write_text(json.dumps({'results':results},indent=2))
