import torch,json
from test_bench import Case,eager,graph,ROOT,bounds
from union import PreparedStage
results=[]
for args in [(65536,1024,64,2048),(131072,1024,64,2048),(262144,1024,64,2048),(65549,1024,17,2049),(65536,4096,64,2048),(65536,1024,64,128)]:
    c=Case(*args);plan=PreparedStage(c.pool,c.nt,c.lens,c.u,c.m,c.d,c.s,c.w)
    plan();c.baseline();c.reconstruct();torch.testing.assert_close(c.old,c.restored,rtol=0,atol=0)
    groups=[(graph(lambda g=g:c.baseline(g)),g) for g in [1,2,4,8]];bg=min(groups)[1]
    old=lambda:c.baseline(bg)
    og=graph(old);ng=graph(plan);oe=eager(old);ne=eager(plan)
    rr={'case':args,'baseline_group':bg,'old_graph_us':og*1000,'prepared_graph_us':ng*1000,'graph_speedup':og/ng,'old_eager_us':oe*1000,'prepared_eager_us':ne*1000,'eager_speedup':oe/ne}
    results.append(rr);print(rr,flush=True)
# Empty lengths then growing/shrinking in same graph, including bucket crossing.
c=Case(65536,1024,64,2048);plan=PreparedStage(c.pool,c.nt,c.lens,c.u,c.m,c.d,c.s,c.w);plan()
g=torch.cuda.CUDAGraph()
with torch.cuda.graph(g):plan()
for history in [-1,65536,65499,65536]:
    if history<0:c.lens.zero_()
    else:c.lens.copy_(torch.arange(history+1,history+1025,device='cuda',dtype=torch.int32))
    c.nt.copy_(c.nt.flip(0));c.nts.copy_(c.nt.expand_as(c.nts));c.old.fill_(165);c.restored.fill_(165)
    g.replay();c.baseline();c.reconstruct()
    torch.testing.assert_close(c.old,c.restored,rtol=0,atol=0)
    assert c.m.cpu().tolist()==[list(bounds(n,64,2048)) for n in c.lens.cpu().tolist()]
(ROOT/'prepared_results.json').write_text(json.dumps({'results':results,'dynamic_zero_grow_shrink_replay':'PASS'},indent=2))
