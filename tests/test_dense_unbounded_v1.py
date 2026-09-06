import inspect

import pytest
import torch

from benchmarks.dense_unbounded_v1 import rosa_soft_dense_unbounded_v1
from rosa_soft.soft_reference import rosa_soft_reference


def _inputs(seed=17, length=7, *, requires_grad=True):
    generator = torch.Generator().manual_seed(seed)
    tensors = (
        torch.randn(2, length, 4, 3, generator=generator, dtype=torch.float64),
        torch.randn(2, length, 4, 3, generator=generator, dtype=torch.float64),
        torch.randn(2, length, 2, 5, generator=generator, dtype=torch.float64),
    )
    return tuple(tensor.requires_grad_(requires_grad) for tensor in tensors)


def test_public_surface_has_no_estimator_controls():
    assert tuple(inspect.signature(rosa_soft_dense_unbounded_v1).parameters) == (
        "query",
        "key",
        "value",
    )


def test_matches_full_horizon_production_output_and_vjp():
    inputs = _inputs()
    upstream = torch.randn(
        2,
        7,
        4,
        5,
        generator=torch.Generator().manual_seed(19),
        dtype=torch.float64,
    )
    frozen = rosa_soft_dense_unbounded_v1(*inputs)
    frozen_gradients = torch.autograd.grad(frozen, inputs, upstream)

    production_inputs = tuple(x.detach().requires_grad_() for x in inputs)
    production = rosa_soft_reference(
        *production_inputs,
        max_suffix_length=production_inputs[0].size(1),
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
    )
    production_gradients = torch.autograd.grad(
        production, production_inputs, upstream
    )

    torch.testing.assert_close(frozen, production, rtol=0.0, atol=0.0)
    for actual, expected in zip(frozen_gradients, production_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


def test_hard_forward_is_unlimited_beyond_32():
    length = 70
    query = torch.ones(1, length, 1, 1)
    key = torch.ones_like(query)
    value = torch.where(
        torch.arange(length).view(1, length, 1, 1).remainder(2) == 0,
        1.0,
        -1.0,
    )
    output = rosa_soft_dense_unbounded_v1(query, key, value)

    assert output[0, -1, 0, 0].item() == value[0, -1, 0, 0].item()


def test_forward_cannot_leak_sign_preserving_amplitude():
    inputs = _inputs(requires_grad=False)
    scales = tuple(torch.rand_like(x).add(0.01) for x in inputs)
    expected = rosa_soft_dense_unbounded_v1(*inputs)
    actual = rosa_soft_dense_unbounded_v1(
        *(tensor * scale for tensor, scale in zip(inputs, scales))
    )
    assert torch.equal(actual, expected)


@pytest.mark.parametrize("required", range(3))
def test_partial_gradients_are_finite(required):
    inputs = tuple(
        tensor.detach().requires_grad_(index == required)
        for index, tensor in enumerate(_inputs(length=5))
    )
    rosa_soft_dense_unbounded_v1(*inputs).square().sum().backward()
    for index, tensor in enumerate(inputs):
        assert (tensor.grad is not None) == (index == required)
        if tensor.grad is not None:
            assert bool(torch.isfinite(tensor.grad).all())
