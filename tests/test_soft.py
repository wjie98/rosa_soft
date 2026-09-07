import itertools

import pytest
import torch

import rosa_soft
from tests.oracle import carrier, hard

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _inputs(layout, dtype, mask):
    if layout == "dense":
        q = torch.randn(1, 11, 2, 4, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn(1, 11, 1, 6, device="cuda", dtype=dtype)
        cu = torch.empty(0, device="cuda", dtype=torch.int32)
    else:
        q = torch.randn(17, 2, 4, device="cuda", dtype=dtype)
        k = torch.randn_like(q)
        v = torch.randn(17, 1, 6, device="cuda", dtype=dtype)
        cu = torch.tensor([0, 5, 5, 12, 17], device="cuda", dtype=torch.int32)
    for i, x in enumerate((q, k, v)):
        x.requires_grad_(bool(mask & (1 << i)))
    return q, k, v, cu


@CUDA
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_gpu_hard_matches_sam_and_naive(dtype):
    torch.manual_seed(21)
    q = torch.randn(2, 65, 3, 5, device="cuda", dtype=dtype)
    k = torch.randn_like(q)
    v = torch.randn(2, 65, 1, 7, device="cuda", dtype=dtype)
    expected, _ = hard(q, k, v)
    actual = rosa_soft.rosa_soft(q, k, v)
    sam, _ = rosa_soft.rosa_hard(q, k, v)
    assert torch.equal(actual.cpu(), expected.cpu())
    assert torch.equal(actual, sam)


@CUDA
def test_gpu_packed_hard_isolates_empty_segments():
    torch.manual_seed(22)
    cu = torch.tensor([0, 6, 6, 15, 20], device="cuda", dtype=torch.int32)
    q = torch.randn(20, 3, 5, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn(20, 1, 4, device="cuda")
    actual = rosa_soft.rosa_soft(q, k, v, cu)
    expected = []
    offsets = cu.cpu().tolist()
    for start, end in zip(offsets[:-1], offsets[1:]):
        if start != end:
            expected.append(hard(q[start:end][None], k[start:end][None], v[start:end][None])[0][0])
    assert torch.equal(actual, torch.cat(expected))


@CUDA
def test_gpu_hard_suffix_is_not_truncated_at_32():
    t, bits = 71, 8
    code = torch.arange(40, device="cuda")
    motif = torch.where(
        ((code[:, None] >> torch.arange(bits, device="cuda")) & 1).bool(),
        1.0,
        -1.0,
    )
    q = -torch.ones(1, t, 1, bits, device="cuda")
    k = torch.ones_like(q)
    k[0, :40, 0] = motif
    k[0, 69, 0] = motif[-1]
    q[0, 31:71, 0] = motif
    v = -torch.ones(1, t, 1, 1, device="cuda")
    v[0, 40] = 1
    assert rosa_soft.rosa_soft(q, k, v)[0, 70, 0, 0].item() == 1


@CUDA
@pytest.mark.parametrize("layout,dtype,mask", list(itertools.product(
    ["dense", "packed"],
    [torch.float16, torch.bfloat16, torch.float32],
    [1, 2, 4, 3, 5, 6, 7],
)))
def test_native_vjp_matches_definition(layout, dtype, mask):
    torch.manual_seed(30 + mask)
    q, k, v, cu = _inputs(layout, dtype, mask)
    y, pq, pk = torch.ops.rosa_soft.forward(q, k, v, cu)
    dy = torch.randn_like(y)
    seed = torch.tensor(123456789, device="cuda", dtype=torch.int64)
    actual = torch.ops.rosa_soft.backward(
        q, k, v, dy, pq, pk, seed, cu, 0.8, 0.25, 2.2, mask
    )

    leaves = [x.detach().float().requires_grad_(bool(mask & (1 << i))) for i, x in enumerate((q, k, v))]
    if layout == "dense":
        ref = carrier(*leaves, 0.8, 0.25, 2.2, 123456789)
    else:
        parts = []
        offsets = cu.cpu().tolist()
        for b, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
            if start != end:
                parts.append(carrier(
                    leaves[0][start:end][None], leaves[1][start:end][None],
                    leaves[2][start:end][None], 0.8, 0.25, 2.2, 123456789 + b,
                )[0])
        ref = torch.cat(parts)
    expected = torch.autograd.grad(ref, [x for x in leaves if x.requires_grad], dy.float())
    expected = iter(expected)
    atol = {torch.float16: 3e-4, torch.bfloat16: 3e-4, torch.float32: 3e-6}[dtype]
    for i, grad in enumerate(actual):
        if mask & (1 << i):
            torch.testing.assert_close(grad, next(expected), rtol=2e-4, atol=atol)
        else:
            assert grad.numel() == 0


@CUDA
def test_public_autograd_and_packed_empty_segment():
    torch.manual_seed(8)
    q, k, v, cu = _inputs("packed", torch.float32, 7)
    y = rosa_soft.rosa_soft(q, k, v, cu, scale=0.9, dropout_p=0.1)
    y.square().mean().backward()
    assert all(x.grad is not None and torch.isfinite(x.grad).all() for x in (q, k, v))


@CUDA
@pytest.mark.parametrize("layout", ["dense", "packed"])
@pytest.mark.parametrize("mask", range(1, 8))
@pytest.mark.parametrize("bits", [1, 8, 16, 32])
def test_fp16_long_dispatch_matches_generic_value_width(layout, mask, bits):
    torch.manual_seed(99)
    t = 2049
    # H*T >= 8192 reaches the FP16 tiles when Q or K gradients are requested.
    q = torch.randn(1, t, 4, bits, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, t, 1, 65, device="cuda", dtype=torch.float16)
    dy = torch.randn(1, t, 4, 65, device="cuda", dtype=torch.float16)
    dy[..., -1] = 0
    cu = torch.empty(0, device="cuda", dtype=torch.int32)
    if layout == "packed":
        q, k, v, dy = (x[0] for x in (q, k, v, dy))
        cu = torch.tensor([0, 0, t, t], device="cuda", dtype=torch.int32)
    seed = torch.tensor(123456789, device="cuda", dtype=torch.int64)
    _, pq, pk = torch.ops.rosa_soft.forward(q, k, v[..., :64], cu)
    fast = torch.ops.rosa_soft.backward(
        q, k, v[..., :64], dy[..., :64], pq, pk, seed, cu, 0.8, 0.25, 2.2, mask
    )
    generic = torch.ops.rosa_soft.backward(
        q, k, v, dy, pq, pk, seed, cu, 0.8, 0.25, 2.2, mask
    )
    for i, (actual, expected) in enumerate(zip(fast, generic)):
        if mask & (1 << i):
            if i == 2:
                expected = expected[..., :64]
            torch.testing.assert_close(actual, expected, rtol=4e-2, atol=2e-3)
            relative = (actual - expected).norm() / expected.norm().clamp_min(1e-20)
            assert relative < 2e-4
        else:
            assert actual.numel() == 0


@CUDA
@pytest.mark.parametrize("bits", [1, 8, 32])
def test_fp16_long_dispatch_matches_definition(bits):
    torch.manual_seed(105 + bits)
    t = 2051
    q = torch.randn(1, t, 4, bits, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    if bits == 1:
        q.fill_(1.)
        k.fill_(1.)
    v = torch.randn(1, t, 1, 64, device="cuda", dtype=torch.float16)
    dy = torch.randn(1, t, 4, 64, device="cuda", dtype=torch.float16)
    cu = torch.empty(0, device="cuda", dtype=torch.int32)
    seed = torch.tensor(123456789, device="cuda", dtype=torch.int64)
    _, pq, pk = torch.ops.rosa_soft.forward(q, k, v, cu)
    actual = torch.ops.rosa_soft.backward(q, k, v, dy, pq, pk, seed, cu, .8, .25, 2.2, 7)
    leaves = [x.float().requires_grad_() for x in (q, k, v)]
    expected = torch.autograd.grad(carrier(*leaves, .8, .25, 2.2, 123456789), leaves, dy.float())
    for a, e in zip(actual, expected):
        torch.testing.assert_close(a, e, rtol=5e-4, atol=5e-4)
        assert (a - e).norm() / e.norm().clamp_min(1e-20) < 1e-4


@CUDA
def test_torch_compile_dense_and_packed():
    @torch.compile(fullgraph=True)
    def loss(q, k, v, cu):
        return rosa_soft.rosa_soft(q, k, v, cu).square().mean()

    for layout in ("dense", "packed"):
        q, k, v, cu = _inputs(layout, torch.float32, 7)
        value = loss(q, k, v, None if layout == "dense" else cu)
        value.backward()
        assert all(x.grad is not None for x in (q, k, v))


@CUDA
@pytest.mark.parametrize("layout", ["dense", "packed"])
def test_torch_compile_fp16_long(layout):
    torch.manual_seed(119)
    q = torch.randn(1, 2049, 4, 8, device="cuda", dtype=torch.float16)
    k = torch.randn_like(q)
    v = torch.randn(1, 2049, 1, 64, device="cuda", dtype=torch.float16)
    cu = None
    if layout == "packed":
        q, k, v = (x[0] for x in (q, k, v))
        cu = torch.tensor([0, 0, 2049, 2049], device="cuda", dtype=torch.int32)
    leaves = [x.requires_grad_() for x in (q, k, v)]
    compiled = torch.compile(rosa_soft.rosa_soft, fullgraph=True)
    expected = rosa_soft.rosa_soft(*leaves, cu)
    actual = compiled(*leaves, cu)
    assert torch.equal(actual, expected)
    dy = torch.randn_like(expected)
    for a, e in zip(torch.autograd.grad(actual, leaves, dy),
                    torch.autograd.grad(expected, leaves, dy)):
        torch.testing.assert_close(a, e, rtol=2e-3, atol=2e-4)


@CUDA
def test_argument_errors_are_clear():
    q = torch.ones(1, 3, 1, 2, device="cuda")
    with pytest.raises(ValueError, match="dropout_p"):
        rosa_soft.rosa_soft(q, q, q, dropout_p=1.0)
    with pytest.raises(ValueError, match="dense input"):
        rosa_soft.rosa_soft(q, q, q, torch.tensor([0, 3], device="cuda", dtype=torch.int32))
    with pytest.raises(ValueError, match="packed input"):
        rosa_soft.rosa_soft(q[0], q[0], q[0])
