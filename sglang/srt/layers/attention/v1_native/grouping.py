"""Same balanced cosine grouping as the preserved Qwen experiment."""
import torch
import torch.nn.functional as F


def build_bucket(keys, start, ids_only=False):
    k = keys.transpose(0, 1).contiguous()
    heads, count, dim = k.shape
    unit = F.normalize(k.float(), dim=-1)
    head = torch.arange(heads, device=k.device)[:, None, None]

    def split(ids):
        h, batch, n = ids.shape
        if n <= 8:
            return ids
        u = unit[head, ids]
        b = (u * u[:, :, :1]).sum(-1).argmin(-1)
        bi = b[:, :, None, None].expand(-1, -1, 1, dim)
        ub = u.gather(2, bi)
        a = (u * ub).sum(-1).argmin(-1)
        ua = u.gather(2, a[:, :, None, None].expand_as(bi))
        margin = (u * ua).sum(-1) - (u * ub).sum(-1)
        ordered = ids.gather(-1, torch.argsort(margin, dim=-1, descending=True, stable=True))
        cut = ((n + 7) // 8 // 2) * 8
        if cut * 2 == n:
            return split(ordered.reshape(h, batch * 2, cut)).reshape(h, batch, n)
        return torch.cat((split(ordered[:, :, :cut]), split(ordered[:, :, cut:])), -1)

    perm = split(torch.arange(count, device=k.device).expand(heads, 1, count)).reshape(heads, count)
    pad = (-count) % 64
    ids = F.pad(perm + start, (0, pad), value=-1).int()
    if ids_only:
        return ids
    ordered = F.pad(k.gather(1, perm[:, :, None].expand(-1, -1, dim)).float(),
                    (0, 0, 0, pad))
    valid = (ids.reshape(heads, -1, 8) >= 0).sum(-1)
    centers = ordered.reshape(heads, -1, 8, dim).sum(2) / valid.clamp_min(1)[:, :, None]
    return ids, ordered, centers, valid > 0


def quantize(x):
    x = x.float()
    scale = 2. ** torch.ceil(torch.log2(x.abs().amax(-1).clamp_min(1e-4) / 448))
    return (x / scale[..., None]).to(torch.float8_e4m3fn), scale


def pack(keys):
    """One head, aligned count; byte layout: 64x128 FP8 then 64 FP32 scales."""
    k8, scale = quantize(keys)
    pages = len(keys) // 64
    return torch.cat((k8.view(torch.uint8).reshape(pages, 8192),
                      scale.contiguous().view(torch.uint8).reshape(pages, 256)), 1
                     ).reshape(pages, 64, 1, 132)
