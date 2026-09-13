from copy import deepcopy
from functools import partial

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from examples.rosa_4bit import Rosa4Bit
from rosa_soft import rosa_bitflip, rosa_soft
from tests.oracle import bitflip_vjp, carrier, hard

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
OPS = {"soft": rosa_soft, "bitflip": rosa_bitflip}


def dp(q, k, v):
    """Literal row DP, vectorized only over independent candidates and heads."""
    q, k, v = (x.detach().cpu().float().numpy() for x in (q, k, v))
    b, t, h, _ = q.shape
    state = np.zeros((b, t, h), dtype=np.int64)
    out = np.zeros_like(v)
    for i in range(1, t):
        current = np.zeros_like(state)
        previous = np.concatenate((np.zeros((b, 1, h), dtype=np.int64), state[:, :i-1]), axis=1)
        equal = ((q[:, i:i+1] > 0) == (k[:, :i] > 0)).all(axis=-1)
        current[:, :i] = equal * (1 + previous)
        length = current[:, :i].max(axis=1)
        end = i - 1 - current[:, :i][:, ::-1].argmax(axis=1)
        value = v[np.arange(b)[:, None], end + 1, np.arange(h)[None, :]]
        out[:, i] = np.where(length[..., None] > 0, np.where(value > 0, 1., -1.), 0.)
        state = current
    return torch.from_numpy(out)


class Reference(torch.autograd.Function):
    @staticmethod
    def forward(ctx, name, q, k, v):
        ctx.name = name
        ctx.save_for_backward(q, k, v)
        return hard(*(x.detach().cpu() for x in (q, k, v)))[0].to(v)

    @staticmethod
    def backward(ctx, dy):
        xs = ctx.saved_tensors
        if ctx.name == "bitflip":
            grads = bitflip_vjp(*xs, dy)
        else:
            with torch.enable_grad():
                leaves = [x.detach().cpu().double().requires_grad_() for x in xs]
                grads = torch.autograd.grad(carrier(*leaves), leaves, dy.cpu().double())
        return None, *(g.to(x) for g, x in zip(grads, xs))


def test_adapter_shape_checks():
    for width in (0, 3, 4.0, True):
        with pytest.raises(ValueError, match="width"):
            Rosa4Bit(width)
    layer = Rosa4Bit(8)
    for shape in ((1, 4, 4), (1, 4, 2, 4)):
        x = torch.ones(shape)
        with pytest.raises(ValueError):
            layer(x, x, x)
    x = torch.ones(1, 5, 8)
    with pytest.raises(ValueError, match="matching"):
        layer(x, x[:, :-1], x)


@CUDA
@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_amplitude_and_projection_gradients(name, dtype):
    torch.manual_seed(431)
    xs = [torch.randn(1, 7, 8, device="cuda", dtype=dtype).requires_grad_() for _ in range(3)]
    with torch.no_grad():
        xs[0][:, 1] = xs[1][:, 0]
        xs[0][:, 2] = 0
    layer = Rosa4Bit(8, OPS[name]).cuda()
    with torch.no_grad():
        layer.emb.copy_(torch.tensor([-2., -.25, 0., 1.5, 2., 0., -1., .75], device="cuda"))
    ref = Rosa4Bit(8, partial(Reference.apply, name)).cuda()
    ref.load_state_dict(layer.state_dict(), strict=True)
    y, expected = layer(*xs), ref(*xs)
    assert y.dtype == dtype and torch.equal(y, expected)
    grouped = [x.reshape(1, 7, 2, 4) for x in xs]
    raw = dp(*grouped).to(y).flatten(-2)
    assert torch.equal(raw, hard(*grouped)[0].flatten(-2))
    assert torch.equal(y, (raw * layer.emb).to(dtype))
    dy = torch.randn_like(y)
    actual = torch.autograd.grad(y, (*xs, layer.emb), dy)
    want = torch.autograd.grad(expected, (*xs, ref.emb), dy)
    tol = {torch.float32: 3e-5, torch.float16: .002, torch.bfloat16: .015}[dtype]
    for a, e in zip(actual, want):
        torch.testing.assert_close(a, e, atol=tol, rtol=tol)
    torch.testing.assert_close(actual[-1], (raw.float() * dy.float()).sum((0, 1), keepdim=True))
    # Even a zero-amplitude channel must still be able to learn its amplitude.
    assert actual[-1][..., [2, 5]].abs().sum() > 0
    assert actual[0].abs().sum() > 0 and actual[1].abs().sum() > 0


@CUDA
@pytest.mark.parametrize("name", OPS)
def test_amplitude_only_training(name):
    torch.manual_seed(912)
    xs = [torch.randn(2, 9, 8, device="cuda") for _ in range(3)]
    xs[0][:, 1] = xs[1][:, 0]
    layer = Rosa4Bit(8, OPS[name]).cuda()
    y = layer(*xs)
    y.sum().backward()
    assert layer.emb.grad is not None and layer.emb.grad.abs().sum() > 0
    assert all(x.grad is None for x in xs)


@CUDA
def test_value_gradient_matches_literal_edits():
    torch.manual_seed(524)
    q = torch.ones(1, 5, 4, device="cuda")
    k = -torch.ones_like(q)
    k[:, 0] = 1
    v = torch.randn_like(q).requires_grad_()
    layer = Rosa4Bit(4, rosa_bitflip).cuda()
    with torch.no_grad():
        layer.emb.copy_(torch.tensor([-2., 0., .5, 3.], device="cuda"))
    dy = torch.randn_like(v)
    actual = torch.autograd.grad(layer(q, k, v), v, dy)[0].cpu().double()
    q, k, v = (x.detach().cpu().double().reshape(1, 5, 1, 4) for x in (q, k, v))
    amplitude = layer.emb.detach().cpu().double().reshape(1, 1, 1, 4)
    base = dp(q, k, v).double() * amplitude
    expected = torch.zeros_like(v)
    for p in range(v.numel()):
        edited = v.clone()
        sign = 1 if v.flatten()[p] > 0 else -1
        edited.flatten()[p] = -sign
        delta = ((dp(q, k, edited).double() * amplitude - base) * dy.cpu().double().reshape_as(v)).sum()
        expected.flatten()[p] = -.5 * sign * delta / (1 + v.flatten()[p].abs()).square()
    torch.testing.assert_close(actual, expected.reshape_as(actual), atol=3e-5, rtol=3e-5)


class Block(nn.Module):
    """Two pre-normalized residual branches with independent time-mixed Q/K/V."""
    def __init__(self, width, op):
        super().__init__()
        self.ln = nn.LayerNorm(width)
        self.mix = nn.Parameter(torch.full((3, 1, 1, width), .25))
        self.proj = nn.ModuleList(nn.Linear(width, width) for _ in range(3))
        self.rosa_qkv = Rosa4Bit(width, op)
        self.out = nn.Linear(width, width)
        self.ln2 = nn.LayerNorm(width)
        self.x_k = nn.Parameter(torch.full((1, 1, width), .25))
        self.key = nn.Linear(width, width * 4, bias=False)
        self.value = nn.Linear(width * 4, width, bias=False)

    def forward(self, x):
        z = self.ln(x)
        previous = nn.functional.pad(z[:, :-1], (0, 0, 1, 0))
        q, k, v = [p(z + a * (previous - z)) for p, a in zip(self.proj, self.mix)]
        x = x + self.out(self.rosa_qkv(q, k, v))
        z = self.ln2(x)
        previous = nn.functional.pad(z[:, :-1], (0, 0, 1, 0))
        hidden = self.key(z + self.x_k * (previous - z)).relu().square()
        return x + self.value(hidden)


@CUDA
@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("compiled", [False, True])
def test_residual_checkpoint_training(name, dtype, compiled, monkeypatch):
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("BF16 model requires Ampere")
    torch._dynamo.reset()
    torch.manual_seed(318)
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    model = nn.Sequential(Block(8, OPS[name]), Block(8, OPS[name]), nn.Linear(8, 11)).cuda()
    ref = deepcopy(model)
    for block in ref[:2]:
        block.rosa_qkv.op = partial(Reference.apply, name)

    def forward(x):
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            for block in model[:2]:
                x = checkpoint(block, x, use_reentrant=False)
            return model[-1](x)

    call = torch.compile(forward, fullgraph=True) if compiled else forward
    opt = torch.optim.AdamW(model.parameters(), lr=.001)
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16, init_scale=16)
    x = torch.randn(1, 7, 8, device="cuda")
    target = torch.randint(0, 11, (1, 7), device="cuda")
    for _ in range(2):
        ref.load_state_dict(model.state_dict(), strict=True)
        opt.zero_grad(set_to_none=True)
        ref.zero_grad(set_to_none=True)
        y = call(x)
        with torch.autocast("cuda", dtype=dtype, enabled=dtype != torch.float32):
            expected = ref(x)
        assert y.dtype == dtype
        tol = {torch.float32: 8e-5, torch.float16: .004, torch.bfloat16: .03}[dtype]
        torch.testing.assert_close(y, expected, atol=tol, rtol=tol)
        loss = nn.functional.cross_entropy(y.float().flatten(0, 1), target.flatten())
        ref_loss = nn.functional.cross_entropy(expected.float().flatten(0, 1), target.flatten())
        scaler.scale(loss).backward()
        ref_loss.backward()
        scaler.unscale_(opt)
        for a, e in zip(model.parameters(), ref.parameters()):
            assert a.grad is not None and torch.isfinite(a.grad).all()
            torch.testing.assert_close(a.grad, e.grad, atol=tol, rtol=tol)
        scaler.step(opt)
        scaler.update()


@CUDA
@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_192_heads_at_512_tokens(name, dtype):
    torch.manual_seed(809)
    xs = [torch.randn(1, 512, 768, device="cuda", dtype=dtype).requires_grad_() for _ in range(3)]
    layer = Rosa4Bit(768, OPS[name]).cuda()
    with torch.no_grad():
        layer.emb.uniform_(-2, 2)
    y = layer(*xs)
    expected = dp(*(x.reshape(1, 512, 192, 4) for x in xs)).to(y).flatten(-2)
    assert torch.equal(y, (expected * layer.emb).to(dtype))
    dy = torch.randn_like(y)
    grads = torch.autograd.grad(y, (*xs, layer.emb), dy)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
    # Check head isolation without making the full-size test enumerate every bit.
    channels = torch.tensor([0, 1, 2, 3, 764, 765, 766, 767], device="cuda")
    small = [x.detach().index_select(-1, channels).requires_grad_() for x in xs]
    part = Rosa4Bit(8, OPS[name]).cuda()
    with torch.no_grad():
        part.emb.copy_(layer.emb.index_select(-1, channels))
    other = torch.autograd.grad(part(*small), small, dy.index_select(-1, channels))
    tol = {torch.float32: 3e-4, torch.float16: .01, torch.bfloat16: .06}[dtype]
    for a, e in zip(grads, other):
        torch.testing.assert_close(a.index_select(-1, channels), e, atol=tol, rtol=tol)
