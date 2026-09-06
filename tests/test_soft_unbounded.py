import pytest
import torch

import rosa_soft


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda,
    reason="RosaSoft CUDA extension is unavailable",
)


def _dense_case(*, tokens=37, dtype=torch.float32, seed=8100):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    query = torch.randn(
        1, tokens, 2, 8, device="cuda", dtype=dtype, generator=generator
    )
    key = torch.randn(query.shape, device="cuda", dtype=dtype, generator=generator)
    value = torch.randn(
        1, tokens, 1, 7, device="cuda", dtype=dtype, generator=generator
    )
    grad_output = torch.randn(
        1, tokens, 2, 7, device="cuda", dtype=dtype, generator=generator
    )
    return query, key, value, grad_output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("dropout_p", [0.0, 0.2])
def test_unbounded_public_operator_matches_full_horizon(dtype, dropout_p):
    inputs = _dense_case(dtype=dtype)

    def run(operator):
        torch.cuda.manual_seed(991)
        leaves = tuple(tensor.detach().clone().requires_grad_() for tensor in inputs[:3])
        output = operator(*leaves)
        gradients = torch.autograd.grad(output, leaves, inputs[3])
        return output.detach(), gradients

    actual_output, actual_gradients = run(
        lambda q, k, v: rosa_soft.rosa_soft_unbounded(
            q, k, v, dropout_p=dropout_p
        )
    )
    expected_output, expected_gradients = run(
        lambda q, k, v: rosa_soft.rosa_soft(
            q,
            k,
            v,
            max_suffix_length=q.size(1),
            dropout_p=dropout_p,
        )
    )

    assert torch.equal(actual_output, expected_output)
    tolerance = 5e-4 if dtype == torch.float32 else 2e-3
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(
            actual.float(), expected.float(), rtol=tolerance, atol=tolerance
        )


def test_unbounded_varlen_matches_full_horizon_and_empty_segments():
    query, key, value, grad_output = _dense_case(tokens=19)
    packed = tuple(tensor.squeeze(0) for tensor in (query, key, value))
    packed_grad = grad_output.squeeze(0)
    cu_seqlens = torch.tensor(
        [0, 5, 5, 12, 19], dtype=torch.int32, device="cuda"
    )

    def run(operator):
        leaves = tuple(tensor.detach().clone().requires_grad_() for tensor in packed)
        output = operator(*leaves)
        gradients = torch.autograd.grad(output, leaves, packed_grad)
        return output.detach(), gradients

    actual_output, actual_gradients = run(
        lambda q, k, v: rosa_soft.rosa_soft_unbounded_varlen(
            q, k, v, cu_seqlens
        )
    )
    expected_output, expected_gradients = run(
        lambda q, k, v: rosa_soft.rosa_soft_varlen(
            q,
            k,
            v,
            cu_seqlens,
            max_suffix_length=q.size(0),
        )
    )

    assert torch.equal(actual_output, expected_output)
    for actual, expected in zip(actual_gradients, expected_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-6, atol=1e-8)


def test_unbounded_operator_compiles_fullgraph():
    query, key, value, _ = _dense_case(tokens=17)
    compiled = torch.compile(
        rosa_soft.rosa_soft_unbounded,
        backend="aot_eager",
        fullgraph=True,
    )
    inputs = tuple(tensor.detach().requires_grad_() for tensor in (query, key, value))
    output = compiled(*inputs)
    gradients = torch.autograd.grad(output.sum(), inputs)

    assert output.shape == (1, 17, 2, 7)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
