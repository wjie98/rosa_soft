"""Small definition-level oracles used only by the production tests."""

import math

import torch
import torch.nn.functional as F


def sign(x):
    return torch.where(x > 0, torch.ones_like(x), -torch.ones_like(x))


def soft_sign(x):
    z = x / (1 + x.abs())
    return sign(x) + z - z.detach()


def routes(q, k):
    """Return latest end of the longest historical suffix for [B,T,H,D]."""

    qb = (q > 0).cpu()
    kb = (k > 0).cpu()
    b, t, h, _ = q.shape
    out = torch.full((b, t, h), -1, dtype=torch.int64)
    for z in range(b):
        for a in range(h):
            for i in range(t):
                best_len = 0
                best_end = -1
                for j in range(i):
                    length = 0
                    while length <= j and torch.equal(
                        qb[z, i - length, a], kb[z, j - length, a]
                    ):
                        length += 1
                    if length > best_len or (
                        length == best_len and length > 0 and j > best_end
                    ):
                        best_len = length
                        best_end = j
                out[z, i, a] = best_end
    return out


def hard(q, k, v):
    route = routes(q, k)
    b, t, h = route.shape
    hv = v.size(2)
    values = sign(v).repeat_interleave(h // hv, dim=2).permute(0, 2, 1, 3)
    index = (route + 1).to(v.device).permute(0, 2, 1)[..., None]
    index = index.expand(-1, -1, -1, v.size(3))
    out = torch.gather(values, 2, index).permute(0, 2, 1, 3)
    return out.masked_fill(route.to(v.device)[..., None] < 0, 0), route


def bitflip_vjp(q, k, v, dy):
    """Enumerate independent sign edits; contract output differences before STE."""
    q, k, v, dy = (x.detach().cpu().double().contiguous() for x in (q, k, v, dy))
    base, route = hard(q, k, v)
    grads = [torch.zeros_like(x) for x in (q, k, v)]
    for side, x in enumerate((q, k)):
        for p in range(x.numel()):
            edited = x.clone()
            edited.flatten()[p] = -1 if x.flatten()[p] > 0 else 1
            out = hard(edited if side == 0 else q, edited if side == 1 else k, v)[0]
            delta = ((out - base) * dy).sum()
            grads[side].flatten()[p] = (
                -0.5 * sign(x.flatten()[p]) * delta / (1 + x.flatten()[p].abs()).square()
            )
    b, t, h = route.shape
    for a in range(b):
        for i in range(t):
            for head in range(h):
                end = int(route[a, i, head])
                if end >= 0:
                    vh = head // (h // v.size(2))
                    grads[2][a, end + 1, vh] += (
                        dy[a, i, head] / (1 + v[a, end + 1, vh].abs()).square()
                    )
    return tuple(grads)


def _mask(t, device):
    i = torch.arange(t, device=device).view(t, 1)
    a = torch.arange(t, device=device).view(1, t)
    return (a == 0) | ((a > 0) & (a <= i))


def _hash(x):
    mask = (1 << 32) - 1
    x &= mask
    x ^= x >> 16
    lo, hi = x & 0xFFFF, x >> 16
    x = (lo * 0x7FEB352D + ((hi * 0x7FEB352D & 0xFFFF) << 16)) & mask
    x ^= x >> 15
    lo, hi = x & 0xFFFF, x >> 16
    x = (lo * 0x846CA68B + ((hi * 0x846CA68B & 0xFFFF) << 16)) & mask
    return (x ^ (x >> 16)) & mask


def _dropout(p, seed, batch, h, t, device):
    if p == 0:
        return 1.0
    coords = (
        torch.arange(batch, device=device, dtype=torch.int64).view(batch, 1, 1, 1),
        torch.arange(h, device=device, dtype=torch.int64).view(1, h, 1, 1),
        torch.arange(t, device=device, dtype=torch.int64).view(1, 1, t, 1),
        torch.arange(t, device=device, dtype=torch.int64).view(1, 1, 1, t),
    )
    x = _hash(coords[3] ^ 0x68E31DA4)
    for c, salt in zip(coords[2::-1], (0xB5297A4D, 0x63D83595, 0xA511E9B3)):
        x = _hash(x ^ c ^ salt)
    x = _hash(x ^ (seed & ((1 << 32) - 1)))
    x = _hash(x ^ ((seed >> 32) & ((1 << 32) - 1)) ^ 0x9E3779B9)
    keep = ((x >> 8).float() * 2**-24) >= p
    return keep / (1 - p)


def carrier(q, k, v, scale=1.0, dropout_p=0.0, mismatch_scale=3.0, seed=0):
    """Dense differentiable carrier; its forward value is not ROSA output."""

    b, t, h, _ = q.shape
    mask = _mask(t, q.device)
    qs = soft_sign(q.permute(0, 2, 1, 3))
    ks = soft_sign(k.permute(0, 2, 1, 3)[..., :-1, :])
    mismatch = 0.5 * (1 - qs.unsqueeze(-2) * ks.unsqueeze(-3)).mean(-1)
    gate = F.pad(torch.exp(-mismatch_scale * mismatch), (1, 0)) * mask

    previous = gate.new_zeros(b, h, t)
    rows = []
    for i in range(t):
        current = previous if i == 0 else F.pad(
            gate[..., i, 1 : i + 1] * (1 + previous[..., :i]),
            (1, t - i - 1),
        )
        rows.append(current)
        previous = current
    suffix = torch.stack(rows, dim=-2)
    score = (math.sqrt(2) + 1) * (torch.sqrt(1 + suffix) - 1)
    score = score.clone()
    score[..., 0] = 0.5

    route = torch.arange(t, device=q.device).view(1, 1, 1, t)
    nonnull = mask.view(1, 1, t, t) & (route > 0)
    count = nonnull.sum(-1, keepdim=True).clamp_min(1)
    logits = scale * score - torch.where(
        nonnull, count.to(score.dtype).log(), torch.zeros((), device=q.device)
    )
    logits = logits.masked_fill(~mask, -torch.inf)
    probability = logits.softmax(-1)
    probability = probability * _dropout(dropout_p, seed, b, h, t, q.device)

    values = soft_sign(v).repeat_interleave(h // v.size(2), dim=2)
    values = values.permute(0, 2, 1, 3)
    values = torch.where(route.transpose(-2, -1) > 0, values, 0)
    return torch.einsum("bhta,bhad->bhtd", probability, values).permute(0, 2, 1, 3)
