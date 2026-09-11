import pytest
import torch

from rosa_soft import rosa_bitflip, rosa_hard, rosa_soft
from tests.oracle import bitflip_vjp, hard

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def inputs(dtype=torch.float32, t=7, d=3, mask=7):
    torch.manual_seed(149)
    q, k = [torch.randn(1, t, 2, d, device="cuda", dtype=dtype) for _ in range(2)]
    v = torch.randn(1, t, 1, 5, device="cuda", dtype=dtype)
    if t:
        q[:, 0, :, 0] = k[:, 0, :, 0] = v[:, 0, :, 0] = 0
    return [x.requires_grad_(bool(mask & (1 << i))) for i, x in enumerate((q, k, v))]


def check(xs, dy, rows=256, fn=rosa_bitflip):
    y = fn(*xs, rows=rows)
    expected, _ = hard(*(x.detach().cpu() for x in xs))
    assert torch.equal(y.cpu(), expected)
    grads = bitflip_vjp(*xs, dy)
    y.backward(dy)
    tol = {torch.float32: 3e-5, torch.float16: .002, torch.bfloat16: .01}[xs[0].dtype]
    for x, g in zip(xs, grads):
        if x.requires_grad:
            torch.testing.assert_close(x.grad.cpu().float(), g.to(x.dtype).float(), atol=tol, rtol=tol)
        else:
            assert x.grad is None


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("mask", range(1, 8))
def test_gradients(dtype, mask):
    xs = inputs(dtype, mask=mask)
    check(xs, torch.randn(1, 7, 2, 5, device="cuda", dtype=dtype), rows=3)


@pytest.mark.parametrize("d", [1, 8, 9, 16, 32])
@pytest.mark.parametrize("pattern", ["constant", "periodic", "one_bit", "mixed"])
def test_structured_edits(d, pattern):
    xs = inputs(t=9, d=d)
    q, k, v = xs
    with torch.no_grad():
        q.fill_(-1)
        k.fill_(-1)
        if pattern == "periodic":
            q[:, ::2, :, -1] = 1
            k[:, 1::2, :, -1] = 1
        if pattern == "one_bit":
            q[..., -1] = 1
        if pattern == "mixed":
            q[:, ::3, :, -1] = 1
            k[:, ::5, :, -1] = 1
    check(xs, torch.randn(1, 9, 2, 5, device="cuda"), rows=4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
@pytest.mark.parametrize("t", [0, 1])
@pytest.mark.parametrize("mask", range(1, 8))
def test_empty_and_single(dtype, t, mask):
    xs = inputs(dtype, t=t, mask=mask)
    check(xs, torch.ones(1, t, 2, 5, device="cuda", dtype=dtype))


@pytest.mark.parametrize("dv", [2, 31, 64, 128, 129, 257, 513])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_cancellation(dtype, dv):
    q = torch.full((1, 3, 1, 1), -1., device="cuda", dtype=dtype, requires_grad=True)
    k = q.detach().clone()
    k[:, 1] = 1
    k.requires_grad_()
    v = torch.ones(1, 3, 1, dv, device="cuda", dtype=dtype)
    v[:, 1, :, 1] = -1
    v.requires_grad_()
    dy = torch.zeros_like(v)
    small = 2.**-14 if dtype == torch.float16 else 1.
    dy[:, 2, :, 0] = 2.**14 if dtype == torch.float16 else 2.**80
    dy[:, 2, :, 1] = small
    actual = torch.autograd.grad(rosa_bitflip(q, k, v, rows=1), (q, k, v), dy)
    for a, e in zip(actual, bitflip_vjp(q, k, v, dy)):
        torch.testing.assert_close(a.cpu(), e.to(dtype), atol=0, rtol=0)
    assert actual[0][0, 2, 0, 0] == small / 4
    assert actual[1][0, 1, 0, 0] == -small / 4


@pytest.mark.parametrize("rows", [1, 31, 256])
def test_unlimited_hard_and_band_invariance(rows):
    torch.manual_seed(210)
    t = 129
    q = torch.full((1, t, 2, 8), -1., device="cuda", requires_grad=True)
    k = q.detach().clone().requires_grad_()
    v = torch.randn(1, t, 1, 3, device="cuda", requires_grad=True)
    dy = torch.randn(1, t, 2, 3, device="cuda")
    y = rosa_bitflip(q, k, v, rows=rows)
    assert torch.equal(y, rosa_soft(q, k, v))
    assert torch.equal(y, rosa_hard(q, k, v)[0])
    a = torch.autograd.grad(y, (q, k, v), dy)
    b = torch.autograd.grad(rosa_bitflip(q, k, v, rows=7), (q, k, v), dy)
    for x, z in zip(a, b):
        torch.testing.assert_close(x, z, atol=3e-5, rtol=3e-5)


def test_noncontiguous_linearity_and_aliases():
    xs = [x.transpose(1, 2).contiguous().transpose(1, 2).detach().requires_grad_() for x in inputs()]
    a, b = [torch.randn(1, 7, 2, 5, device="cuda") for _ in range(2)]
    grads = [torch.autograd.grad(rosa_bitflip(*xs), xs, g) for g in (a, b, 2*a - 3*b)]
    for x, y, z in zip(*grads):
        torch.testing.assert_close(z, 2*x - 3*y, atol=8e-5, rtol=8e-5)
    q = inputs(t=5)[0]
    dy = torch.randn_like(q)
    expected = sum(bitflip_vjp(q, q, q, dy))
    rosa_bitflip(q, q, q).backward(dy)
    torch.testing.assert_close(q.grad.cpu().double(), expected, atol=3e-5, rtol=3e-5)


def test_batch_and_grouped_values():
    torch.manual_seed(871)
    q, k = [torch.randn(2, 5, 4, 3, device="cuda") for _ in range(2)]
    v = torch.randn(2, 5, 2, 5, device="cuda")
    xs = [x.transpose(1, 2).contiguous().transpose(1, 2).requires_grad_()
          for x in (q, k, v)]
    dy = torch.randn(2, 5, 4, 5, device="cuda")
    dy = dy.transpose(1, 2).contiguous().transpose(1, 2)
    check(xs, dy, rows=2)
    contiguous = [x.detach().contiguous().requires_grad_() for x in xs]
    check(contiguous, dy.contiguous(), rows=2)
    for a, b in zip(xs, contiguous):
        torch.testing.assert_close(a.grad, b.grad, atol=3e-5, rtol=3e-5)


def test_changed_graph_on_nondefault_stream():
    xs = inputs(t=11)
    dy = torch.randn(1, 11, 2, 5, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        def step():
            for x in xs:
                x.grad = None
            y = rosa_bitflip(*xs, rows=3)
            y.backward(dy)
            return y
        step()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            y = step()
        grads = tuple(x.grad for x in xs)
        for _ in range(3):
            with torch.no_grad():
                for x in xs:
                    x.copy_(torch.randn_like(x))
                dy.copy_(torch.randn_like(dy))
            graph.replay()
            assert torch.equal(y.cpu(), hard(*(x.detach().cpu() for x in xs))[0])
            for a, e in zip(grads, bitflip_vjp(*xs, dy)):
                torch.testing.assert_close(a.cpu().double(), e, atol=4e-5, rtol=4e-5)
    torch.cuda.current_stream().wait_stream(stream)


@pytest.mark.parametrize("mask", range(1, 8))
def test_opcheck_and_compile(mask):
    torch._dynamo.reset()
    xs = inputs(mask=mask)
    torch.library.opcheck(torch.ops.rosa_soft.bitflip_forward.default, (*xs, 3))
    fn = torch.compile(rosa_bitflip, fullgraph=True, dynamic=True)
    for t in (7, 1, 0):
        xs = inputs(t=t, mask=mask)
        check(xs, torch.ones(1, t, 2, 5, device="cuda"), rows=3, fn=fn)


def test_saved_inputs():
    for mask in range(1, 8):
        saved = []
        with torch.autograd.graph.saved_tensors_hooks(
            lambda x: (saved.append(x), x)[1], lambda x: x
        ):
            rosa_bitflip(*inputs(mask=mask))
        assert saved[0].numel() == (42 if mask & 1 else 0)
        assert saved[1].numel() == (42 if mask & 2 else 0)
        assert saved[3].numel() == (14 if mask & 3 else 0)
        assert saved[4].numel() == (14 if mask & 3 else 0)


def test_errors_and_determinism():
    xs = inputs()
    for rows in (0, 257, 1.5, True):
        with pytest.raises(ValueError, match="rows"):
            rosa_bitflip(*xs, rows=rows)
    with pytest.raises(ValueError, match="dense"):
        rosa_bitflip(*(x[0] for x in xs))
    with pytest.raises(RuntimeError, match="shape"):
        rosa_bitflip(xs[0], xs[1][:, :-1], xs[2])
    with pytest.raises(RuntimeError, match="matching"):
        rosa_bitflip(xs[0].half(), xs[1], xs[2])
    before = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="deterministic"):
            rosa_bitflip(*xs).sum().backward()
    finally:
        torch.use_deterministic_algorithms(before, warn_only=warn)


@pytest.mark.parametrize("mask", [1, 2, 4, 7])
def test_native_backward_validation(mask):
    q, k, v = inputs(t=3, mask=mask)
    y, pq, pk, route = torch.ops.rosa_soft.bitflip_forward(q, k, v, 3)
    args = [q, k, v, torch.ones_like(y), pq, pk, route, q.size(-1), 3, mask]
    bad = [(3, args[3][:, :-1]), (6, route[:, :-1])]
    if mask & 3:
        bad += [(4, pq[:, :-1]), (5, pk.to(torch.int64))]
    if mask & 1:
        bad += [(0, q[:, :-1].contiguous())]
    if mask & 2:
        bad += [(1, k[:, :-1].contiguous())]
    for pos, value in bad:
        changed = args.copy()
        changed[pos] = value
        with pytest.raises(RuntimeError, match="invalid"):
            torch.ops.rosa_soft.bitflip_backward(*changed)


def test_double_is_not_a_supported_dispatch_type():
    xs = [x.double() for x in inputs()]
    with pytest.raises(RuntimeError, match="FP16/BF16/FP32"):
        rosa_bitflip(*xs)


@pytest.mark.parametrize("q,k,end", [
    ([1,1,0,1,0,0,0,0,1], [0,1,0,0,1,1,0,1,1], 4),
    ([0,1,0,0,0,1,1,0,0], [1,0,0,1,1,1,0,1,1], 2),
    ([0,1,0,1,1,0,1,0,0], [1,0,0,1,1,1,1,1,0], 2),
])
def test_latest_terminal_gap(q, k, end):
    # Cover later terminal matches at the last key, inside the gap, or absent.
    torch.manual_seed(816)
    xs = [(2 * torch.tensor(x, device="cuda", dtype=torch.float32) - 1)
          [None, :, None, None].requires_grad_() for x in (q, k)]
    xs.append(torch.randn(1, 9, 1, 5, device="cuda", requires_grad=True))
    assert hard(*(x.detach().cpu() for x in xs))[1][0, -1, 0] == end
    check(xs, torch.randn_like(xs[-1]), rows=3)


def constant_reference(d, t):
    # Q[p] changes only output p. K[p] routes to V[p] for p<i<=2p.
    torch.manual_seed(661)
    q = torch.full((1, t, 1, d), -1., device="cuda", requires_grad=True)
    k = q.detach().clone().requires_grad_()
    v = (torch.randint(0, 2, (1, t, 1, 1), device="cuda") * 2 - 1).float().requires_grad_()
    dy = (torch.randint(0, 2, v.shape, device="cuda") * 2 - 1).float()
    y = rosa_bitflip(q, k, v, rows=128)
    assert torch.equal(y[:, 1:], v[:, 1:])
    assert y[:, 0].count_nonzero() == 0
    actual = torch.autograd.grad(y, (q, k, v), dy)
    value, g = v.detach().cpu().flatten().double(), dy.cpu().flatten().double()
    u = value * g
    p = torch.arange(t)
    stop = torch.minimum(2*p, torch.tensor(t-1)) + 1
    pg = torch.cat((torch.zeros(1, dtype=torch.float64), g.cumsum(0)))
    pu = torch.cat((torch.zeros(1, dtype=torch.float64), u.cumsum(0)))
    gk = (value * (pg[stop] - pg[p+1]) - (pu[stop] - pu[p+1])) / 8
    gk[0] = -u[1] / 8
    gq = -u / 8
    gq[0] = 0
    gv = g / 4
    gv[0] = 0
    for a, e in zip(actual, (gq, gk, gv)):
        torch.testing.assert_close(a.cpu(), e[None, :, None, None].expand(a.shape).float(), atol=0, rtol=0)


@pytest.mark.parametrize("d", [8, 32])
def test_long_constant(d):
    constant_reference(d, 545)


def test_long_one_bit():
    torch.manual_seed(91)
    t, d = 545, 8
    q = torch.full((1, t, 1, d), -1., device="cuda")
    k = q.clone().requires_grad_()
    q[..., -1] = 1
    q.requires_grad_()
    v = (torch.randint(0, 2, (1, t, 1, 3), device="cuda") * 2 - 1).float().requires_grad_()
    dy = (torch.randint(0, 2, v.shape, device="cuda") * 2 - 1).float()
    y = rosa_bitflip(q, k, v, rows=31)
    assert y.count_nonzero() == 0
    actual = torch.autograd.grad(y, (q, k, v), dy)
    value, g = v.detach().cpu(), dy.cpu()
    qgrad, kgrad = torch.zeros_like(q, device="cpu"), torch.zeros_like(k, device="cpu")
    qgrad[:, 1:, :, -1] = -(value[:, 1:] * g[:, 1:]).sum(-1) / 8
    tail = g.flip(1).cumsum(1).flip(1)
    kgrad[:, :-1, :, -1] = (value[:, 1:] * tail[:, 1:]).sum(-1) / 8
    for a, e in zip(actual, (qgrad, kgrad, torch.zeros_like(value))):
        torch.testing.assert_close(a.cpu(), e, atol=0, rtol=0)
