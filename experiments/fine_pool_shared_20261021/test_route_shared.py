import json
import torch
from route_shared import route,module

def check(scores,published,budgets,rt,selected,takes):
    # Reference checking stays on CPU after copying results; not part of graph/timing.
    scores,published,budgets,rt,selected,takes=[x.detach().cpu() for x in (scores,published,budgets,rt,selected,takes)]
    pages=rt.shape[1];width=selected.shape[1]
    actual=budgets.clamp(min=0).minimum(published.clamp(0,pages)).clamp(max=width)
    assert torch.equal(takes,actual)
    reduced=scores.view(scores.shape[0],pages,8).amax(-1)
    # Compare selected scores with reference top-k multiset (ties may permute).
    for r in range(scores.shape[0]):
        k=int(actual[r]);a=int(published[r].clamp(0,pages))
        if k:
            ids=selected[r,:k]
            eq=ids[:,None]==rt[r,:a][None,:]
            assert bool(eq.any(1).all()) and ids.unique().numel()==k
            ranks=eq.int().argmax(1)
            got=reduced[r,ranks].sort(descending=True).values
            want=reduced[r,:a].topk(k).values
            torch.testing.assert_close(got,want,rtol=0,atol=0)
        assert bool((selected[r,k:]==0).all())

def fixture(rows,pages,width):
    scores=torch.randn(rows,pages*8,device='cuda')
    rt=torch.stack([torch.randperm(pages,device='cuda',dtype=torch.int32)+100 for _ in range(rows)])
    pub=torch.full((rows,),pages,device='cuda',dtype=torch.int32)
    budget=torch.full_like(pub,width)
    out=torch.empty(rows,width,device='cuda',dtype=torch.int32);takes=torch.empty_like(pub)
    return scores,pub,budget,rt,out,takes

def main():
    torch.manual_seed(20261021);module();cases=[]
    for p in [1,17,64,129,1024,3904,4096]:
        args=fixture(8,p,min(40,p));s,pub,b,rt,out,t=args
        pub.copy_(torch.tensor([p,p//2,0,p,p+5,-1,p,p],device='cuda',dtype=torch.int32))
        b.copy_(torch.tensor([40,100,4,0,1,8,-1,p],device='cuda',dtype=torch.int32))
        route(*args);check(*args)
        s.fill_(-float('inf'));route(*args);check(*args)
        cases.append(p)
    args=fixture(1024,64,40);route(*args);check(*args)
    g=torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):route(*args)
    for active,budget in [(64,40),(17,40),(64,0),(0,40),(64,17)]:
        args[0].normal_();args[1].fill_(active);args[2].fill_(budget)
        args[3].copy_(args[3].flip(1))
        g.replay();check(*args)
    print('PASS standalone route: capacity branches, ties, padding, 1024 Q, same-Graph dynamic pages/budget/table',flush=True)
    # Actual shared fine pipeline, 1 K head and 32 Q heads.
    from experiment import Data,Runner
    d=Data(history=65536,rows=1024,min_end=61440);r=Runner(d,True)
    scores=torch.randn(d.rows,d.cap*8,device='cuda')
    pub=torch.full((d.rows,),d.cap,device='cuda',dtype=torch.int32)
    budget=torch.full_like(pub,40)
    rt=torch.arange(d.cap,device='cuda',dtype=torch.int32).flip(0).repeat(d.rows,1).contiguous()
    def full():
        route(scores,pub,budget,rt,d.selected,d.takes)
        r.full()
    full();torch.cuda.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):full()
    for a,k,end in [(64,40,65536),(17,40,61440),(64,0,65536),(64,17,65536)]:
        # Fixed capacity covers both published-end snapshots.
        pub.fill_(a);budget.fill_(k);d.end.fill_(end);scores.normal_();rt.copy_(rt.flip(1))
        graph.replay();torch.cuda.synchronize()
        check(scores,pub,budget,rt,d.selected,d.takes)
        expected_table=d.selected.clone();shared_scores=r.scores.clone();logical=r.logical.clone();slots=r.slots.clone()
        # Independent per-Q staging baseline with identical physical selected pages.
        reference=Runner(d,False);reference.full();torch.cuda.synchronize()
        torch.testing.assert_close(shared_scores,reference.scores,rtol=2e-4,atol=2e-4)
        assert torch.equal(torch.isfinite(shared_scores),torch.isfinite(reference.scores))
        assert torch.equal(logical,reference.logical) and torch.equal(slots,reference.slots)
        assert torch.equal(r.table[:,:int(d.takes[0])],expected_table[:,:int(d.takes[0])])
    result={'status':'PASS','device':torch.cuda.get_device_name(0),'rows':1024,'key_heads':1,'query_heads':32,
      'route_capacities':cases,'graph_dynamic':['published page count','budget','request page table','center scores','indexed end in shared pipeline'],
      'pipeline':'route -> stage_union -> tables -> DeepGEMM -> normalize -> torch.topk -> map_topk',
      'limitations':'center scores supplied as fixture; not full SGLang inference; no fused.topk integration; no performance claim'}
    from pathlib import Path
    Path(__file__).with_name('route_test_results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
