import itertools
import json

import pytest
import torch

from benchmarks import fast_weight_proxy as fast_weight
from benchmarks import fast_weight_proxy_ablation as ablation
from rosa_soft.soft_reference import (
    _causal_route_mask,
    _hard_route_forward,
    _pairwise_soft_match_gates,
    _suffix_prefix_product_scores,
)


def _inputs(*, bits=2, sequence_length=6, dtype=torch.float64):
    generator = torch.Generator().manual_seed(1200 + bits + sequence_length)
    shapes = (
        (1, sequence_length, 2, bits),
        (1, sequence_length, 2, bits),
        (1, sequence_length, 1, 3),
    )
    return tuple(
        torch.randn(shape, generator=generator, dtype=dtype).requires_grad_()
        for shape in shapes
    )


def test_symbol_features_are_exact_exponential_hamming_kernel():
    bits = 3
    mismatch_scale = 2.7
    patterns = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=bits)),
        dtype=torch.float64,
    ).view(1, -1, 1, bits)
    features = fast_weight._symbol_kernel_features(
        patterns,
        mismatch_scale,
    )[0, 0]
    actual = features @ features.mT
    mismatch_count = (
        patterns[0, :, 0, None] != patterns[0, None, :, 0]
    ).sum(dim=-1)
    expected = torch.exp(
        -mismatch_scale * mismatch_count.to(torch.float64) / bits
    )

    torch.testing.assert_close(actual, expected, rtol=1e-14, atol=1e-14)


@pytest.mark.parametrize(
    ("bits", "degree", "expected_dim"),
    ((8, 1, 9), (8, 2, 37), (8, 3, 93), (8, 8, 256)),
)
def test_symbol_interaction_feature_dimensions(bits, degree, expected_dim):
    logits = torch.ones(1, 1, 1, bits, dtype=torch.float64)
    features = fast_weight._symbol_interaction_features(logits, 3.0, degree)

    assert features.shape == (1, 1, 1, expected_dim)


def test_full_symbol_interaction_features_equal_exact_kernel():
    bits = 4
    patterns = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=bits)),
        dtype=torch.float64,
    ).view(1, -1, 1, bits)
    exact = fast_weight._symbol_kernel_features(patterns, 2.5)[0, 0]
    interactions = fast_weight._symbol_interaction_features(
        patterns,
        2.5,
        bits,
    )[0, 0]

    torch.testing.assert_close(
        interactions @ interactions.mT,
        exact @ exact.mT,
        rtol=2e-14,
        atol=2e-14,
    )


def test_exact_suffix_fingerprint_inner_product_is_dense_suffix_score():
    query, key, _ = _inputs(bits=2, sequence_length=5)
    query = query[:, :, :1].detach()
    key = key[:, :, :1].detach()
    mismatch_scale = 1.8
    window = 3
    query_levels = fast_weight._exact_suffix_levels(
        fast_weight._symbol_kernel_features(query, mismatch_scale),
        window,
    )
    key_levels = fast_weight._exact_suffix_levels(
        fast_weight._symbol_kernel_features(key, mismatch_scale),
        window,
    )
    query_fingerprint = fast_weight._join_levels(query_levels)
    key_fingerprint = fast_weight._join_levels(key_levels)
    feature_scores = torch.einsum(
        "bhtf,bhaf->bhta",
        query_fingerprint,
        key_fingerprint[:, :, :-1],
    )
    feature_scores = torch.nn.functional.pad(feature_scores, (1, 0))

    causal = _causal_route_mask(query.size(1), query.device)
    gates = _pairwise_soft_match_gates(
        query,
        key,
        causal,
        mismatch_scale,
    )
    expected = _suffix_prefix_product_scores(gates, window)
    valid = causal.view(1, 1, query.size(1), query.size(1)).clone()
    valid[..., 0] = False

    torch.testing.assert_close(
        feature_scores[valid],
        expected[valid],
        rtol=2e-14,
        atol=2e-14,
    )


@pytest.mark.parametrize("proxy", fast_weight.PROXIES)
def test_every_proxy_preserves_hard_forward_and_has_finite_vjp(proxy):
    inputs = _inputs()
    expected, _, _, _ = _hard_route_forward(
        *(tensor.detach() for tensor in inputs),
    )
    output = fast_weight.rosa_fast_weight_proxy(
        *inputs,
        proxy=proxy,
        max_suffix_length=3,
        fingerprint_length=3,
        mismatch_scale=2.0,
        sketch_dim=16,
        sketch_seed=7,
    )
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad((output * upstream).sum(), inputs)

    assert torch.equal(output, expected)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("proxy", fast_weight.PROXIES)
def test_fast_weight_carrier_has_no_sign_preserving_amplitude_leak(proxy):
    inputs = _inputs()
    generator = torch.Generator().manual_seed(83)
    scaled = tuple(
        tensor.detach()
        * (
            0.1
            + torch.rand(
                tensor.shape,
                dtype=tensor.dtype,
                generator=generator,
            )
        )
        for tensor in inputs
    )
    common = {
        "proxy": proxy,
        "fingerprint_length": 3,
        "mismatch_scale": 2.0,
        "sketch_dim": 16,
        "sketch_seed": 11,
    }
    original = fast_weight._fast_weight_carrier(*inputs, **common)
    rescaled = fast_weight._fast_weight_carrier(*scaled, **common)

    torch.testing.assert_close(original, rescaled, rtol=0.0, atol=0.0)


def test_delta_rule_replaces_repeated_fingerprint_with_latest_value():
    query_features = torch.ones(1, 1, 3, 1, dtype=torch.float64)
    key_features = torch.ones_like(query_features)
    values = torch.tensor(
        [[[[1.0], [-1.0], [1.0]]]],
        dtype=torch.float64,
    )
    additive = fast_weight._linear_attention_carrier(
        query_features,
        key_features,
        values,
    )
    delta = fast_weight._delta_rule_carrier(
        query_features,
        key_features,
        values,
    )

    assert additive[0, 2, 0, 0].item() == pytest.approx(0.0)
    assert delta[0, 2, 0, 0].item() == pytest.approx(1.0)


def test_zero_history_normalization_has_finite_zero_vjp():
    features = torch.zeros(2, 3, requires_grad=True)
    normalized = fast_weight._normalize(features)
    normalized.sum().backward()

    assert torch.equal(normalized, torch.zeros_like(normalized))
    assert torch.equal(features.grad, torch.zeros_like(features))


def test_tensor_sketch_suffix_inner_product_is_unbiased_in_mean():
    symbols = torch.tensor(
        [[[[1.0, -1.0]], [[-1.0, -1.0]], [[1.0, 1.0]]]],
        dtype=torch.float64,
    )
    local = fast_weight._symbol_kernel_features(symbols, 3.0)
    estimates = []
    for seed in range(256):
        level = fast_weight._tensor_sketch_suffix_levels(
            local,
            max_suffix_length=3,
            sketch_dim=32,
            sketch_seed=seed,
        )[-1]
        estimates.append((level[0, 0, -1] ** 2).sum())
    mean_estimate = torch.stack(estimates).mean()

    assert mean_estimate.item() == pytest.approx(1.0, abs=0.08)


@pytest.mark.parametrize("proxy", fast_weight.PROXIES)
def test_custom_vjp_matches_materialized_carrier(proxy):
    inputs = _inputs()
    common = {
        "proxy": proxy,
        "max_suffix_length": 3,
        "fingerprint_length": 3,
        "mismatch_scale": 2.2,
        "sketch_dim": 16,
        "sketch_seed": 5,
    }
    output = fast_weight.rosa_fast_weight_proxy(*inputs, **common)
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad((output * upstream).sum(), inputs)

    direct_inputs = tuple(
        tensor.detach().requires_grad_() for tensor in inputs
    )
    carrier = fast_weight._fast_weight_carrier(
        *direct_inputs,
        proxy=proxy,
        fingerprint_length=3,
        mismatch_scale=2.2,
        sketch_dim=16,
        sketch_seed=5,
    )
    expected = torch.autograd.grad(
        (carrier * upstream).sum(),
        direct_inputs,
    )

    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_exact_expansion_rejects_intractable_level():
    inputs = _inputs(bits=8, sequence_length=4)
    with pytest.raises(ValueError, match="use sketch_suffix_delta"):
        output = fast_weight.rosa_fast_weight_proxy(
            *inputs,
            proxy="exact_suffix_delta",
            max_suffix_length=3,
        )
        output.sum().backward()


def test_prototype_rejects_more_than_eight_symbol_bits():
    inputs = _inputs(bits=9)
    with pytest.raises(ValueError, match="at most 8"):
        fast_weight.rosa_fast_weight_proxy(
            *inputs,
            proxy="sketch_suffix_delta",
            max_suffix_length=2,
        )


def test_gradient_ablation_compares_only_declared_production_and_bitflip():
    args = ablation.build_parser().parse_args(
        [
            "--gradient-only",
            "--gradient-seeds",
            "0",
            "--gradient-sequence-length",
            "4",
            "--gradient-max-suffix-length",
            "2",
            "--gradient-fingerprint-length",
            "2",
            "--sketch-dim",
            "8",
        ]
    )
    report = ablation.run_matrix(args)

    assert report["baselines"] == ["production", "bitflip"]
    assert set(report["gradient_summaries"]) == set(ablation.ESTIMATORS)
    assert all(
        summary["hard_forward_equal_cases"] == 1
        for summary in report["gradient_summaries"].values()
    )
    assert report["proxy_has_trainable_parameters"] is False
    assert report["proxy_has_auxiliary_loss"] is False
    assert report["proxy_uses_attention_scale"] is False


def test_reported_state_dimensions_expose_exact_growth_and_sketch_bound():
    args = ablation.build_parser().parse_args([])
    dimensions = ablation._state_dimensions(args)

    assert dimensions["local_feature"] == 4
    assert dimensions["single_suffix"] == 64
    assert dimensions["single_suffix_sketch"] == 32
    assert dimensions["exact_multi_suffix"] == 4 + 16 + 64
    assert dimensions["sketch_multi_suffix"] == 3 * 32
    assert dimensions["exact_single_delta_memory_floats_per_head"] == 128
    assert dimensions["single_sketch_delta_memory_floats_per_head"] == 64
    assert dimensions["state_linear"] == 3
    assert dimensions["state_quadratic"] == 4
    assert dimensions["state_cubic"] == 4
    assert dimensions["state_full_delta_memory_floats_per_head"] == 8
    assert dimensions["state_quadratic_attention_total_floats_per_head"] == 12


def test_zero_bitflip_gradient_metrics_remain_json_serializable():
    zero = (torch.zeros(2), torch.zeros(2), torch.zeros(2))
    nonzero = (torch.ones(2), torch.zeros(2), torch.zeros(2))

    both_zero = ablation._gradient_metrics(zero, zero)
    undefined_ratio = ablation._gradient_metrics(nonzero, zero)

    assert both_zero["combined"]["relative_l2_error_to_bitflip"] == 0.0
    assert both_zero["combined"]["norm_ratio_to_bitflip"] == 1.0
    assert undefined_ratio["combined"]["relative_l2_error_to_bitflip"] is None
    assert undefined_ratio["combined"]["norm_ratio_to_bitflip"] is None
    json.dumps({"zero": both_zero, "nonzero": undefined_ratio}, allow_nan=False)
