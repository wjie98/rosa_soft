from functools import partial

import pytest
import torch

from rosa_soft import rosa_bitflip
from tests.oracle import bitflip_vjp, hard
from tests.test_joint import reference

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def inputs(dtype=torch.float32, t=5, d=4, dv=5):
    torch.manual_seed(725)
    xs = [torch.randn(2, t, h, width, device="cuda", dtype=dtype)
          for h, width in ((4, d), (4, d), (2, dv))]
    if t > 1:
        xs[0][:, 1] = xs[1][:, 0]
        xs[0][:, 0, :, 0] = 0
    return [x.transpose(1, 2).contiguous().transpose(1, 2) for x in xs]


def check(y, expected, xs, dy, grads):
    assert torch.equal(y.cpu().double(), expected.cpu().double())
    actual = torch.autograd.grad(y, [x for x in xs if x.requires_grad], dy)
    tol = {torch.float32: 3e-5, torch.float16: .002, torch.bfloat16: .015}[y.dtype]
    for a, g in zip(actual, [g for x, g in zip(xs, grads) if x.requires_grad]):
        torch.testing.assert_close(a.cpu().float(), g.to(a.dtype).cpu().float(), atol=tol, rtol=tol)


@pytest.mark.parametrize("mask", range(1, 8))
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_independent_oracle(mask, dtype):
    xs = [x.requires_grad_(bool(mask & (1 << i))) for i, x in enumerate(inputs(dtype))]
    dy = torch.randn(2, 5, 4, 10, device="cuda", dtype=dtype)[..., ::2]
    check(rosa_bitflip(*xs, rows=3, chunks=2), hard(*xs)[0], xs, dy, bitflip_vjp(*xs, dy))


@pytest.mark.parametrize("tied,mask", [("qk", 1), ("qk", 2), ("qk", 3), ("qkv", 1)])
@pytest.mark.parametrize("dtype,compiled", [(torch.float16, False), (torch.bfloat16, False),
                                           (torch.float32, False), (torch.float32, True)])
def test_joint_oracle(tied, mask, dtype, compiled):
    x, _, v = inputs(dtype)
    x.requires_grad_(bool(mask & 1))
    v = x if tied == "qkv" else v.requires_grad_(bool(mask & 2))
    dy = torch.randn(2, 5, 4, v.size(-1)*2, device="cuda", dtype=dtype)[..., ::2]
    expected, grads = reference(x, v, dy, tied)
    if compiled:
        torch._dynamo.reset()
    fn = torch.compile(rosa_bitflip, fullgraph=True) if compiled else rosa_bitflip
    check(fn(x, x, v, tied=tied, chunks=2, rows=3), expected,
          (x,) if tied == "qkv" else (x, v), dy, grads)


@pytest.mark.parametrize("t", [0, 1, 129])
@pytest.mark.parametrize("tied", [None, "qk", "qkv"])
def test_empty_and_unlimited(t, tied):
    q, k, v = inputs(t=t, d=1)
    q[:, :65] = k[:, :65] = 1
    q.requires_grad_()
    k = q if tied else k.requires_grad_()
    v = q if tied == "qkv" else v.requires_grad_()
    xs = (q,) if tied == "qkv" else (q, v) if tied else (q, k, v)
    dy = torch.randn(2, t, 4, v.size(-1), device="cuda")
    y = rosa_bitflip(q, k, v, chunks=2, rows=7, tied=tied)
    expected = rosa_bitflip(q, k, v, rows=31, tied=tied)
    assert torch.equal(y, hard(q, k, v)[0])
    for a, b in zip(torch.autograd.grad(y, xs, dy), torch.autograd.grad(expected, xs, dy)):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("d,dv", [(8, 64), (32, 33)])
def test_widths(d, dv):
    xs = [x.requires_grad_() for x in inputs(t=3, d=d, dv=dv)]
    dy = torch.randn(2, 3, 4, dv, device="cuda")
    check(rosa_bitflip(*xs, chunks=2, rows=1), hard(*xs)[0], xs, dy, bitflip_vjp(*xs, dy))


@pytest.mark.parametrize("tied", [None, "qk", "qkv"])
def test_compile_alias_and_parameter_changes(tied):
    torch._dynamo.reset()
    torch.manual_seed(956)
    fn = torch.compile(rosa_bitflip, fullgraph=True, dynamic=True)
    for t, chunks in ((7, 2), (5, 1), (7, 4), (0, 2)):
        x = torch.randn(1, t, 4, 4, device="cuda", requires_grad=True)
        a, b = torch.randn_like(x), torch.randn_like(x)
        expected = rosa_bitflip(x, x, x, tied=tied, rows=3)
        gs = []
        for dy in (a, b, 2*a-3*b):
            y = fn(x, x, x, chunks=chunks, tied=tied, rows=3)
            assert torch.equal(y, expected)
            gs.append(torch.autograd.grad(y, x, dy)[0])
        want = torch.autograd.grad(expected, x, a)[0]
        torch.testing.assert_close(gs[0], want, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(gs[2], 2*gs[0]-3*gs[1], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize("mask", range(1, 8))
def test_compiled_masks(mask):
    torch._dynamo.reset()
    xs = [x.requires_grad_(bool(mask & (1 << i))) for i, x in enumerate(inputs())]
    dy = torch.randn(2, 5, 4, 5, device="cuda")
    fn = torch.compile(partial(rosa_bitflip, chunks=2, rows=3), fullgraph=True)
    check(fn(*xs), hard(*xs)[0], xs, dy, bitflip_vjp(*xs, dy))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("compiled", [False, True])
def test_checkpoint_training(dtype, compiled, monkeypatch):
    from tests import test_4bit_training as training
    monkeypatch.setitem(training.OPS, "bitflip", partial(rosa_bitflip, chunks=2))
    training.test_residual_checkpoint_training("bitflip", dtype, compiled, monkeypatch)


def test_validation_and_opcheck():
    q, k, v = [x.requires_grad_() for x in inputs()]
    for chunks in (0, -1, 3, 8, 1.5, True, None):
        with pytest.raises(ValueError, match="chunks"):
            rosa_bitflip(q, k, v, chunks=chunks)
    with pytest.raises(ValueError, match="GQA"):
        rosa_bitflip(q, k, v, chunks=4)
    for chunks in (0, -1, 3, 4):
        with pytest.raises(RuntimeError, match="chunks"):
            torch.ops.rosa_soft.bitflip_forward(q, k, v, 3, chunks)
        with pytest.raises(RuntimeError, match="chunks"):
            torch.ops.rosa_soft.joint_forward(q, v, 3, False, chunks)
    torch.library.opcheck(torch.ops.rosa_soft.bitflip_forward.default, (q, k, v, 3, 2))
    torch.library.opcheck(torch.ops.rosa_soft.joint_forward.default, (q, v, 3, False, 2))


@pytest.mark.parametrize("tied", [None, "qk", "qkv"])
def test_graph_on_nondefault_stream(tied):
    q, k, v = [x.requires_grad_() for x in inputs()]
    k = q if tied else k
    v = q if tied == "qkv" else v
    xs = (q,) if tied == "qkv" else (q, v) if tied else (q, k, v)
    dy = torch.randn(2, 5, 4, v.size(-1), device="cuda")

    def step():
        y = rosa_bitflip(q, k, v, tied=tied, chunks=2, rows=3)
        return y, torch.autograd.grad(y, xs, dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y, grads = step()
        with torch.no_grad():
            for x in xs:
                x.copy_(torch.randn_like(x))
            dy.copy_(torch.randn_like(dy))
        graph.replay()
        expected, want = reference(q, v, dy, tied) if tied else (hard(q, k, v)[0], bitflip_vjp(q, k, v, dy))
        assert torch.equal(y.cpu().double(), expected.cpu().double())
        for a, b in zip(grads, want):
            torch.testing.assert_close(a.cpu().double(), b.cpu().double(), atol=3e-5, rtol=3e-5)
    torch.cuda.current_stream().wait_stream(stream)
