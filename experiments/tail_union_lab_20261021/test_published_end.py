"""Semantic regressions independent of the old bucket-boundary formula."""
import json
import torch
from test_bench import Case,bounds,ROOT,CONFIGS
from union import PreparedStage

def verify(c,end):
    c.bs=[bounds(n,c.s,c.w,end) for n in c.lens.cpu().tolist()]
    c.end_cpu=end
    c.baseline();c.reconstruct()
    torch.testing.assert_close(c.old,c.restored,rtol=0,atol=0)
    assert c.m.cpu().tolist()==[list(x) for x in c.bs]
    for n,(s,t,count) in zip(c.lens.cpu().tolist(),c.bs):
        got=set(range(s))|set(range(t,n))
        expected=set(range(min(n,c.s)))|set(range(max(0,n-c.w),n))|set(range(min(end,n),n))
        assert got==expected and count==len(got)

cases=[(65536,4096,64,2048,65536),(65536,4096,64,2048,61440),
       (2051,65,17,0,1025),(17,129,65,128,0),(0,65,0,0,0)]
for h,r,s,w,end in cases:
    c=Case(h,r,s,w,indexed_end=end)
    for cfg in CONFIGS:c.validate(cfg)
    print('PASS independent unindexed coverage',h,r,end,flush=True)
# Explicit reported bug: last Q must include 65537 even outside its 2K window.
c=Case(65536,4096,64,2048);c.opt()
assert c.m[-1].cpu().tolist()==[64,65536,4160]
assert c.d.cpu().tolist()==[1,992,1088,97]
s,t,count=c.m[-1].cpu().tolist();assert 65537 in range(t,69632)
old_tail=max(64,min((69632-1)//1024*1024,69632-2048))
assert old_tail==67584 and not (old_tail<=65537<69632)
# One graph, same lens, changing published end independently of lengths.
c=Case(65536,4096,64,2048,indexed_end=65536,min_indexed_end=61001)
plan=PreparedStage(c.pool,c.nt,c.lens,c.end,c.u,c.m,c.d,c.s,c.w);plan()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):plan()
for end in [65536,61440,61001,65536]:
    c.end.fill_(end);c.nt.copy_(c.nt.flip(0));c.nts.copy_(c.nt.expand_as(c.nts))
    c.old.fill_(165);c.restored.fill_(165)
    g.replay();verify(c,end)
    print('PASS same graph dynamic indexed_end',end,flush=True)
(ROOT/'published_end_results.json').write_text(json.dumps({'independent_coverage_cases':cases,'all_six_configs':'PASS','last_Q_meta':[64,65536,4160],'old_formula_misses_65537':True,'new_formula_includes_65537':True,'dynamic_indexed_end_same_graph':[65536,61440,61001,65536],'status':'PASS'},indent=2))
