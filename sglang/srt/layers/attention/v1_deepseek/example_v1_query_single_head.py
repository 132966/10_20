"""Single KV head, real CUDA route/topk, synthetic Q/K and independent references.
Run from the SGLang torch/triton/CUDA environment. No model or service required.
"""
import torch
from sglang.srt.layers.attention.v1_deepseek.query_fused import module


def main():
    device='cuda:0'
    # B>512 is required by this exact fused TopK implementation.
    B, P, CAP, TAKE, N, TOP, WINDOW = 1024, 128, 128, 40, 4097, 2048, 64
    def ints(x): return torch.tensor(x,device=device,dtype=torch.int32)
    gen=torch.Generator().manual_seed(19)
    native_pages=torch.randperm(P,generator=gen).to(device=device,dtype=torch.int32)
    reorder_pages=torch.randperm(CAP,generator=gen).to(device=device,dtype=torch.int32)
    nt=native_pages[None].expand(B,-1).contiguous()
    rt=reorder_pages[None].expand(B,-1).contiguous()
    # One Q head, one K head, D=128. Unique nonnegative dot products avoid TopK ties.
    k=torch.zeros((N,128),device=device)
    k[:,0]=torch.arange(N,device=device)/8192
    q=torch.zeros((B,128),device=device); q[:,0]=1
    # Emulate already-built 1K buckets: reverse members within each bucket,
    # keeping every page within its original bucket.
    logical_order=torch.arange(4096,device=device,dtype=torch.int32).view(4,1024).flip(1).reshape(64,64)
    members=torch.full((CAP,64),-1,device=device,dtype=torch.int32)
    members[reorder_pages[:64].long()]=logical_order
    reordered_k=k[logical_order.long()]
    centers=reordered_k.reshape(64,8,8,128).mean(2)
    scores=torch.full((B,P*8),float('-inf'),device=device)
    scores[:,:64*8]=q@centers.reshape(-1,128).T
    lens=torch.full((B,),N,device=device,dtype=torch.int32)
    takes=torch.full((B,),TAKE,device=device,dtype=torch.int32)
    selected=torch.full((B,TAKE),-1,device=device,dtype=torch.int32)
    fine_table=torch.full((B,TAKE+68),-1,device=device,dtype=torch.int32)
    tail_rows=torch.zeros(B,device=device,dtype=torch.int32)
    mod=module()
    mod.route(scores,lens,takes,selected,P,rt,fine_table,tail_rows,CAP,True)
    reference_pages=scores[:,:512].view(B,64,8).amax(-1).topk(TAKE,dim=-1).indices
    torch.testing.assert_close(selected.long(),reference_pages,rtol=0,atol=0)
    torch.testing.assert_close(fine_table[:,:TAKE],rt.gather(1,reference_pages),rtol=0,atol=0)
    torch.testing.assert_close(fine_table[:,TAKE:],(CAP+torch.arange(68,device=device,dtype=torch.int32))[None].expand(B,-1),rtol=0,atol=0)
    # Build precisely the fine candidate layout consumed by this CUDA mapper:
    # selected reordered pages, sink, then visible tail.
    sink=64
    tail=max(sink,min((N-1)//1024*1024,max(sink,N-WINDOW)))
    chosen=members[fine_table[:,:TAKE].long()].reshape(B,-1)
    fixed=torch.cat((torch.arange(sink,device=device),torch.arange(tail,N,device=device))).to(torch.int32)
    candidate_ids=torch.cat((chosen,fixed[None].expand(B,-1)),dim=1)
    # Actual Q.K; efficient because all rows intentionally use the same Q and candidates.
    fine_scores=(q@k[candidate_ids[0].long()].T).contiguous()
    fine_lens=torch.full((B,),fine_scores.shape[1],device=device,dtype=torch.int32)
    valid=torch.ones_like(candidate_ids,dtype=torch.bool)
    valid[:,:TAKE*64]=(chosen>=sink)&(chosen<tail)
    reference_scores=fine_scores.masked_fill(~valid,float('-inf'))
    expected_raw=reference_scores.topk(TOP,dim=-1).indices
    expected_logical=candidate_ids.gather(1,expected_raw)
    physical=torch.empty((B,TOP),device=device,dtype=torch.int32)
    logical=torch.empty_like(physical); raw=torch.empty_like(physical)
    no_expansion=torch.empty_strided((B,0),(0,1),device=device,dtype=torch.int32)
    # Exercise both ordinary mapping and direct fused fine-page-table mapping.
    for direct in (False,True):
        work=fine_scores.clone() # fused mask updates its score input
        mod.topk(work,fine_lens,members,rt,nt,selected,lens,takes,
                 physical,logical,raw,WINDOW,True,no_expansion,True,True,fine_table,direct)
        torch.cuda.synchronize()
        # Kernel output order is unspecified: compare sets, and verify mapping per entry.
        torch.testing.assert_close(raw.sort(1).values.long(),expected_raw.sort(1).values,rtol=0,atol=0)
        torch.testing.assert_close(logical,candidate_ids.gather(1,raw.long()),rtol=0,atol=0)
        torch.testing.assert_close(logical.sort(1).values,expected_logical.sort(1).values,rtol=0,atol=0)
        expected_physical=nt.gather(1,(logical//64).long())*64+logical%64
        torch.testing.assert_close(physical,expected_physical,rtol=0,atol=0)
        assert bool((logical.sort(1).values[:,1:]!=logical.sort(1).values[:,:-1]).all())
        torch.testing.assert_close(work,reference_scores,rtol=0,atol=0)
        print(f'PASS Top2048 + fused mask + logical/physical mapping, direct_fine={direct}')
    print('PASS page max-of-8 + page TopK + fused fine-table write')
    print('single KV head / single Q head; Q:',tuple(q.shape),'K:',tuple(k.shape))
    print('first selected logical page ranks:',selected[0,:8].cpu().tolist())
    print('corresponding physical reordered pages:',fine_table[0,:8].cpu().tolist())
    print('fine scores:',tuple(fine_scores.shape),'output:',tuple(logical.shape))
    print('first 8 outputs (logical, native physical):',list(zip(logical[0,:8].cpu().tolist(),physical[0,:8].cpu().tolist())))


if __name__=='__main__': main()
