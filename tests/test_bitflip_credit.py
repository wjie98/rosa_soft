"""Definition-level DP checks across the row and value-cache boundaries."""
import pytest
import torch

from rosa_soft import rosa_bitflip

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def routes(q, k):
    # Integer symbols, independent scalar DP, latest position wins a length tie.
    prev = [0] * len(k)
    result = []
    for i, code in enumerate(q):
        curr = [0] * len(k)
        best = (0, -1)
        for j in range(i):
            if code == k[j]:
                curr[j] = 1 + (prev[j - 1] if j else 0)
                best = max(best, (curr[j], j))
        result.append(best[1])
        prev = curr
    return result


def output(q, k, v):
    ends = torch.tensor(routes(q, k))
    return v[ends + 1] * (ends >= 0)[:, None]


def inputs(t, d, dv, dtype):
    q, k = [0] * t, [0] * t
    q[1::5] = [2] * len(q[1::5])
    k[::5] = [2] * len(k[::5])
    rng = torch.Generator().manual_seed(171)
    v = (2 * torch.randint(0, 2, (t, dv), generator=rng) - 1).double()
    dy = torch.randint(-2, 3, (t, dv), generator=rng).double()
    xs = [torch.tensor([[(code >> z & 1) * 2 - 1 for z in range(d)] for code in seq],
                       dtype=dtype, device="cuda")[None, :, None].requires_grad_()
          for seq in (q, k)]
    xs.append(v.to(device="cuda", dtype=dtype)[None, :, None].requires_grad_())
    return q, k, v, dy, xs


@pytest.mark.parametrize("dv", [28, 29, 32, 33, 60, 61, 64, 65,
                               481, 508, 509, 512, 513, 1024])
def test_full_bit_edits_large_row(dv):
    q, k, v, dy, xs = inputs(65, 2, dv, torch.float32)
    y = output(q, k, v)
    refs = [torch.zeros(65, 2, dtype=torch.float64) for _ in range(2)]
    for side, seq in enumerate((q, k)):
        for p in range(len(seq)):
            for bit in range(2):
                changed = seq.copy()
                changed[p] ^= 1 << bit
                yy = output(changed, k, v) if side == 0 else output(q, changed, v)
                sign = (seq[p] >> bit & 1) * 2 - 1
                refs[side][p, bit] = -sign * ((yy - y) * dy).sum() / 8
    gv = torch.zeros_like(v)
    for i, end in enumerate(routes(q, k)):
        if end >= 0:
            gv[end + 1] += dy[i] / 4
    out = rosa_bitflip(*xs, rows=33)
    gs = torch.autograd.grad(out, xs, dy.to("cuda").float()[None, :, None])
    torch.testing.assert_close(out.detach().cpu()[0, :, 0].double(), y, atol=0, rtol=0)
    for g, r in zip(gs, [*refs, gv]):
        torch.testing.assert_close(g.cpu()[0, :, 0].double(), r, atol=0, rtol=0)


@pytest.mark.parametrize("dv", [511, 512, 513, 768, 1024])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_sampled_large_row(dv, dtype):
    q, k, v, dy, xs = inputs(577, 8, dv, dtype)
    y = output(q, k, v)
    out = rosa_bitflip(*xs, rows=128)
    gs = torch.autograd.grad(out, xs, dy.to(device="cuda", dtype=dtype)[None, :, None])
    torch.testing.assert_close(out.detach().cpu()[0, :, 0].double(), y, atol=0, rtol=0)
    for side, seq in enumerate((q, k)):
        for p, bit in [(0, 0), (31, 1), (128, 7), (256, 1), (512, 0), (576, 1)]:
            changed = seq.copy()
            changed[p] ^= 1 << bit
            yy = output(changed, k, v) if side == 0 else output(q, changed, v)
            sign = (seq[p] >> bit & 1) * 2 - 1
            ref = (-sign * ((yy - y) * dy).sum() / 8).to(dtype)
            torch.testing.assert_close(gs[side][0, p, 0, bit].cpu(), ref, atol=0, rtol=0)


def test_credit_linearity():
    _, _, _, dy, xs = inputs(577, 8, 1024, torch.float32)
    a = dy.to("cuda").float()[None, :, None]
    b = a.flip(1).contiguous()
    gs = [torch.autograd.grad(rosa_bitflip(*xs, rows=128), xs, grad) for grad in (a, b, a + b)]
    for x, y, z in zip(*gs):
        torch.testing.assert_close(x + y, z, atol=0, rtol=0)
