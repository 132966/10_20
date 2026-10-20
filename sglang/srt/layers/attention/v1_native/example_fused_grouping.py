"""CPU teaching example; --gpu additionally exercises real FusedGroups kernels."""
import argparse
import importlib.util
import os
from pathlib import Path


def explain():
    # Normalized 128D vectors with only dimension 0 nonzero.
    x = [1, -1, 1, -1, 1, -1, 1, -1]
    perm = list(range(8))
    for n in (8, 4):
        out = []
        for begin in range(0, 8, n):
            group = perm[begin:begin+n]
            seed = group[0]
            score0 = [x[i]*x[seed] for i in group]
            b = group[min(range(n), key=lambda j: score0[j])]
            score1 = [x[i]*x[b] for i in group]
            a = group[min(range(n), key=lambda j: score1[j])]
            margin = [x[i]*x[a]-x[i]*x[b] for i in group]
            order = sorted(range(n), key=lambda j: (-margin[j], j))
            result = [group[j] for j in order]
            print(f"n={n} P={group} s={seed} b={b} a={a}")
            print(f"  dots0={score0}, dots1={score1}, dots2={margin}")
            print(f"  O={result}, split={result[:n//2]} | {result[n//2:]}")
            out.extend(result)
        perm = out
    print("Teaching final groups:", [perm[i:i+2] for i in range(0,8,2)])


def gpu_test(device):
    import torch
    os.environ['SGLANG_V1_GROUPING_MODE'] = 'fused'
    source = Path(__file__).with_name('fused_grouping.py')
    spec = importlib.util.spec_from_file_location('grouping_example_impl', source)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    start, count, heads = 64, 4096, 1
    gen = torch.Generator().manual_seed(123)
    # Deliberately scatter logical positions across native physical slots.
    slots = torch.randperm(start+count, generator=gen).to(device=device, dtype=torch.int32)
    k = torch.zeros((start+count, heads, 128), device=device, dtype=torch.bfloat16)
    ids = torch.arange(count, device=device)
    signs = torch.where(ids % 2 == 0, 1., -1.)
    # Different norms: grouping should use directions, not magnitude.
    magnitudes = (ids % 7 + 1).float()
    k[slots[start:].long(), 0, 0] = (signs*magnitudes).to(k.dtype)
    original = k.clone()
    flag = torch.full((), -1, device=device, dtype=torch.int32)
    build = mod.FusedGroups(heads, device)
    result = build.run(k, slots, start, flag).clone()
    torch.cuda.synchronize(device)
    expected = torch.cat((ids[::2], ids[1::2])).to(torch.int32)+start
    assert torch.equal(result[0], expected), 'unexpected grouping or tie order'
    assert torch.equal(result[0].sort().values, torch.arange(start,start+count,device=device,dtype=torch.int32))
    grouped_signs = signs[(result[0]-start).long()].view(512,8)
    assert bool((grouped_signs == grouped_signs[:,:1]).all())
    assert torch.equal(k, original), 'source K was modified'
    # In fused mode FLAG >= 0 skips all grouping writes.
    for buffer in build.perm:
        buffer.fill_(-777)
    flag.fill_(0)
    skipped = build.run(k, slots, start, flag)
    assert bool((skipped == -777).all()), 'inactive grouping wrote output'
    print('GPU PASS: permutation, direction grouping, stable ties, source preservation, inactive skip')
    print('shape:', tuple(result.shape), 'groups:', tuple(result.view(heads,512,8).shape))
    print('first group:', result[0,:8].cpu().tolist())
    print('first negative group:', result[0,2048:2056].cpu().tolist())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    explain()
    if args.gpu:
        gpu_test(args.device)
