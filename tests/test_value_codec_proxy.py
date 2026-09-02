import torch

from benchmarks.fast_weight_proxy import rosa_fast_weight_proxy
from benchmarks.value_codec_proxy import (
    _gather_encoded_values,
    encode_values,
    rosa_value_codec_state_proxy,
)
from rosa_soft.soft_reference import _hard_route_forward


def test_uniform_and_exponential_codecs_use_the_requested_level_counts():
    value = torch.linspace(-1.0, 1.0, 1001)

    uniform2 = encode_values(value, codec="uniform", quant_bits=2)
    exponential2 = encode_values(
        value,
        codec="exponential",
        quant_bits=2,
        exponent_min=-4.0,
    )
    exponential4 = encode_values(
        value,
        codec="exponential",
        quant_bits=4,
        exponent_min=-4.0,
    )

    assert uniform2.unique().numel() == 4
    assert exponential2.unique().numel() == 4
    assert exponential4.unique().numel() == 16
    torch.testing.assert_close(
        exponential2.abs().unique().sort().values,
        torch.tensor([1.0 / 16.0, 1.0]),
    )


def test_rms_codec_fixes_vector_rms_and_float_is_identity():
    value = torch.tensor(
        [[[[-3.0, -1.0, 2.0, 4.0], [0.5, 1.5, -2.0, 3.0]]]]
    )

    represented = encode_values(value, codec="rms")

    torch.testing.assert_close(
        represented.square().mean(dim=-1),
        torch.ones_like(represented[..., 0]),
    )
    torch.testing.assert_close(
        encode_values(value, codec="float"),
        value,
    )


def test_binary_proxy_matches_existing_quadratic_state_proxy():
    generator = torch.Generator().manual_seed(17)
    tensors = [
        torch.randn(2, 7, 2, 8, generator=generator),
        torch.randn(2, 7, 2, 8, generator=generator),
        torch.randn(2, 7, 1, 4, generator=generator),
    ]
    upstream = torch.randn(2, 7, 2, 4, generator=generator)

    expected_inputs = [tensor.clone().requires_grad_() for tensor in tensors]
    expected = rosa_fast_weight_proxy(
        *expected_inputs,
        proxy="state_quadratic_attention",
        max_suffix_length=1,
        fingerprint_length=1,
        mismatch_scale=3.0,
    )
    expected_grads = torch.autograd.grad(
        expected,
        expected_inputs,
        upstream,
    )

    actual_inputs = [tensor.clone().requires_grad_() for tensor in tensors]
    actual = rosa_value_codec_state_proxy(
        *actual_inputs,
        codec="binary",
        quant_bits=1,
        value_gradient="proxy",
        max_suffix_length=1,
        mismatch_scale=3.0,
    )
    actual_grads = torch.autograd.grad(actual, actual_inputs, upstream)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            rtol=1e-6,
            atol=1e-7,
        )


def test_selected_float_value_gradient_matches_direct_hard_gather():
    generator = torch.Generator().manual_seed(23)
    query = torch.randn(2, 6, 2, 4, generator=generator, requires_grad=True)
    key = torch.randn(2, 6, 2, 4, generator=generator, requires_grad=True)
    value = torch.randn(2, 6, 1, 3, generator=generator, requires_grad=True)
    upstream = torch.randn(2, 6, 2, 3, generator=generator)

    actual = rosa_value_codec_state_proxy(
        query,
        key,
        value,
        codec="float",
        value_gradient="selected",
        max_suffix_length=1,
        mismatch_scale=3.0,
    )
    actual_value_grad = torch.autograd.grad(
        actual,
        value,
        upstream,
        retain_graph=True,
    )[0]

    with torch.no_grad():
        _, _, selected, _ = _hard_route_forward(
            query.detach(),
            key.detach(),
            value.detach(),
        )
    direct_value = value.detach().clone().requires_grad_()
    direct = _gather_encoded_values(direct_value, selected, query.size(2))
    expected_value_grad = torch.autograd.grad(
        direct,
        direct_value,
        upstream,
    )[0]

    torch.testing.assert_close(actual, direct, rtol=0, atol=0)
    torch.testing.assert_close(
        actual_value_grad,
        expected_value_grad,
        rtol=0,
        atol=0,
    )


def test_value_codec_changes_values_without_changing_hard_routes():
    generator = torch.Generator().manual_seed(31)
    query = torch.randn(1, 8, 2, 4, generator=generator)
    key = torch.randn(1, 8, 2, 4, generator=generator)
    value = torch.randn(1, 8, 1, 4, generator=generator)
    with torch.no_grad():
        _, _, selected, _ = _hard_route_forward(query, key, value)

    for codec, bits in (("uniform", 4), ("exponential", 4), ("rms", 4)):
        actual = rosa_value_codec_state_proxy(
            query,
            key,
            value,
            codec=codec,
            quant_bits=bits,
            value_gradient="selected",
            max_suffix_length=1,
        )
        represented = encode_values(value, codec=codec, quant_bits=bits)
        expected = _gather_encoded_values(represented, selected, query.size(2))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
