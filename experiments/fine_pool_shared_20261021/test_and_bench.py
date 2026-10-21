import gc,json,statistics,time
from pathlib import Path
import torch,triton as tr
from experiment import Data,Runner,bounds
ROOT=Path(__file__).parent
torch.manual_seed(20261021)

def check(a,b,reference=True):
    d=a.d;a.full();b.full();torch.cuda.synchronize()
    torch.testing.assert_close(a.clens,b.clens,rtol=0,atol=0)
    assert torch.equal(torch.isfinite(a.scores),torch.isfinite(b.scores))
    torch.testing.assert_close(a.scores,b.scores,rtol=1e-5,atol=2e-5)
    torch.testing.assert_close(a.values,b.values,rtol=1e-5,atol=2e-5)
    # Both restore the SAME candidate order before running the same TopK.
    if torch.equal(a.scores,b.scores):assert torch.equal(a.logical,b.logical) and torch.equal(a.slots,b.slots)
    n=d.lens[:,None];valid=b.logical>=0
    assert ((b.logical<n)|~valid).all()
    physical=d.nt[(b.logical.clamp_min(0)//64).long()]*64+b.logical.clamp_min(0)%64
    assert torch.equal(b.slots,torch.where(valid,physical,-1).int())
    assert int(a.table.min())>=0 and int(a.table.max())<a.fine.shape[0]
    assert int(b.table.min())>=0 and int(b.table.max())<b.fine.shape[0]
    if not reference:return
    end=int(d.end[0])
    for row in sorted(set([0,d.rows//2,d.rows-1])):
        n=int(d.lens[row]);take=int(d.takes[row]);s,t,c=bounds(n,d.sink,d.window,end)
        # Independent union oracle for the tail, plus separately gathered history.
        tail_ids=sorted(set(range(min(n,d.sink)))|set(range(max(0,n-d.window),n))|set(range(min(end,n),n)))
        assert tail_ids==list(range(s))+list(range(t,n)) and len(tail_ids)==c
        sel=d.selected[row,:take].long()
        hist_k=a.fine[sel,:8192].contiguous().view(torch.float8_e4m3fn).reshape(-1,128).float()
        hist_scale=a.fine.view(torch.float32)[sel,2048:].reshape(-1)
        ids=torch.tensor(tail_ids,device='cuda',dtype=torch.long)
        pp=d.nt[ids//64].long();off=ids%64
        tk=d.native[pp[:,None],off[:,None]*128+torch.arange(128,device='cuda')[None,:]].contiguous().view(torch.float8_e4m3fn).float()
        ts=d.native.view(torch.float32)[pp,2048+off]
        k=torch.cat([hist_k,tk]);sc=torch.cat([hist_scale,ts])
        ref=(torch.relu(d.q[row,0].float()@k.T)*d.weights[row,:,None]).sum(0)*sc
        members=d.members[sel].reshape(-1)
        keep=(members>=s)&(members<t)&(members<n)&(members<end)
        ref[:take*64].masked_fill_(~keep,-float('inf'))
        torch.testing.assert_close(b.scores[row,:ref.numel()],ref,rtol=2e-4,atol=2e-4)
        assert torch.isneginf(b.scores[row,ref.numel():]).all()
        good=b.logical[row];good=good[good>=0]
        assert good.unique().numel()==good.numel()

def eager(fn):
    for _ in range(3):fn()
    torch.cuda.synchronize();ts=[]
    for _ in range(5):
        start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(20):fn()
        end.record();end.synchronize();ts.append(start.elapsed_time(end)/20)
    return statistics.median(ts)*1000

def graph(fn):return tr.testing.do_bench_cudagraph(fn,rep=40)*1000

def measure(a,b,mode):
    aa=getattr(a,mode);bb=getattr(b,mode)
    # Alternate order and take median to reduce order/clock bias.
    old=[];new=[]
    for rep in range(3):
        if rep%2: new.append(graph(bb));old.append(graph(aa))
        else:old.append(graph(aa));new.append(graph(bb))
    oe=eager(aa);ne=eager(bb)
    return {'baseline_graph_us':statistics.median(old),'shared_graph_us':statistics.median(new),'graph_speedup':statistics.median(old)/statistics.median(new),'baseline_eager_us':oe,'shared_eager_us':ne,'eager_speedup':oe/ne}

def main():
    report={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,'tests':[],'results':[]}
    for h,r,s,w in [(4096,4,17,0),(65536,1024,64,2048),(65549,1024,17,2049),(65536,4096,64,2048)]:
        d=Data(h,r,s,w);a=Runner(d,False);b=Runner(d,True)
        check(a,b);report['tests'].append({'case':[h,r,s,w],'status':'PASS'})
        print('PASS single-pool DeepGEMM, score/reference/TopK/mapping',h,r,s,w,flush=True)
        if r==4096:
            token=65537;page=int(d.nt[token//64]);off=token%64
            d.native[page,off*128]=torch.tensor(448.,device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
            d.native.view(torch.float32)[page,2048+off]=1.
            qq=d.q[-1,0].view(torch.uint8);qq.zero_();qq[:,0]=torch.tensor(1.,device='cuda').to(torch.float8_e4m3fn).view(torch.uint8)
            d.weights[-1].fill_(1/32)
            check(a,b)
            assert (b.logical[-1]==token).any()
            report['unindexed_65537_last_Q_top2048']='PASS'
        del a,b,d;gc.collect();torch.cuda.empty_cache()
    # Capture includes page preparation, dynamic schedule, unchanged GEMM,
    # boundary restoration, TopK and physical-slot mapping.
    d=Data(65536,1024,17,2049,min_end=61440);a=Runner(d,False);b=Runner(d,True)
    check(a,b)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):b.full()
    for delta,end,budget in [(0,65536,40),(-13,61440,17),(0,65536,0),(0,65536,40)]:
        d.lens.copy_(torch.arange(65537+delta,66561+delta,device='cuda',dtype=torch.int32));d.end.fill_(end)
        d.takes.fill_(budget)
        if budget:d.takes[::3]=max(1,budget//2)
        d.selected.copy_((d.selected+7)%d.cap)
        saved=d.native.clone();oldnt=d.nt.clone();newnt=oldnt.flip(0)
        d.native.index_copy_(0,newnt.long(),saved[oldnt.long()]);d.nt.copy_(newnt);d.nt_rows.copy_(newnt.expand_as(d.nt_rows))
        g.replay();torch.cuda.synchronize()
        captured_scores=b.scores.clone();captured_slots=b.slots.clone()
        a.full();torch.testing.assert_close(a.scores,captured_scores,rtol=1e-5,atol=2e-5)
        if torch.equal(a.scores,captured_scores):assert torch.equal(a.slots,captured_slots)
        check(a,b)
        print('PASS dynamic SAME graph',delta,end,budget,flush=True)
    d.lens.zero_();d.takes.zero_();g.replay();torch.cuda.synchronize()
    assert torch.isneginf(b.scores).all() and (b.logical==-1).all() and (b.slots==-1).all()
    report['same_graph_dynamic_lengths_budget_pages_published_end_and_empty']='PASS'
    del g,a,b,d;gc.collect();torch.cuda.empty_cache()
    for h,r,s,w in [(65536,1024,64,2048),(65536,4096,64,2048),(65549,4096,17,2049)]:
        d=Data(h,r,s,w);a=Runner(d,False);b=Runner(d,True);check(a,b)
        tuning=[]
        for group in [1,2,4,8]:
            a.group=group;tuning.append((graph(a.prepare),group))
        a.group=min(tuning)[1]
        rr={'case':[h,r,s,w],'K_heads':1,'Q_heads':32,'selected_pages':40,'baseline_group':a.group,
            'fine_pool_bytes_baseline':a.fine.numel(),'fine_pool_bytes_shared':b.fine.numel(),
            'staging_and_tables':measure(a,b,'prepare'),
            'staging_tables_schedule_DeepGEMM_mask':measure(a,b,'run_fine'),
            'plus_common_torch_TopK_and_mapping':measure(a,b,'full')}
        report['results'].append(rr);print(json.dumps(rr),flush=True)
        (ROOT/'results.json').write_text(json.dumps(report,indent=2))
        del a,b,d;gc.collect();torch.cuda.empty_cache()
    (ROOT/'results.json').write_text(json.dumps(report,indent=2))
if __name__=='__main__':main()
