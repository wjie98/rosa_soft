import itertools
import json

import pytest
import torch
import torch.nn.functional as F

from benchmarks import suffix_kernel_proxy as suffix_kernel
from benchmarks import suffix_kernel_ablation as ablation
from rosa_soft.soft_reference import (
    _causal_route_mask,
    _hard_route_forward,
    _pairwise_soft_match_gates,
    _suffix_prefix_product_scores,
)


def _inputs(*, bits=2, sequence_length=6, dtype=torch.float64):
    generator = torch.Generator().manual_seed(9100 + bits + sequence_length)
    shapes = (
        (1, sequence_length, 2, bits),
        (1, sequence_length, 2, bits),
        (1, sequence_length, 1, 3),
    )
    return tuple(
        torch.randn(shape, generator=generator, dtype=dtype).requires_grad_()
        for shape in shapes
    )


def _feature_route_scores(features_q, features_k):
    scores = torch.einsum(
        "bhtf,bhaf->bhta",
        features_q,
        features_k[:, :, :-1],
    )
    return F.pad(scores, (1, 0))


def _exact_features(logits, *, route_kernel="raw", window=3, scale=2.3):
    return suffix_kernel._suffix_kernel_features(
        logits,
        representation="exact",
        route_kernel=route_kernel,
        fingerprint_length=window,
        mismatch_scale=scale,
        sketch_dim=16,
        sketch_count=1,
        sketch_seed=0,
    )


def test_exact_raw_features_equal_dense_suffix_scores_and_vjp():
    query, key, _ = _inputs(bits=2, sequence_length=5)
    query = query[:, :, :1]
    key = key[:, :, :1]
    window = 3
    mismatch_scale = 1.9

    feature_scores = _feature_route_scores(
        _exact_features(
            query,
            window=window,
            scale=mismatch_scale,
        ),
        _exact_features(
            key,
            window=window,
            scale=mismatch_scale,
        ),
    )
    causal = _causal_route_mask(query.size(1), query.device)
    gates = _pairwise_soft_match_gates(
        query,
        key,
        causal,
        mismatch_scale,
    )
    dense_scores = _suffix_prefix_product_scores(gates, window)
    nonnull = causal.clone()
    nonnull[:, 0] = False
    mask = nonnull.view(1, 1, query.size(1), query.size(1))

    torch.testing.assert_close(
        feature_scores[mask],
        dense_scores[mask],
        rtol=2e-14,
        atol=2e-14,
    )

    generator = torch.Generator().manual_seed(31)
    upstream = torch.randn(
        dense_scores.shape,
        generator=generator,
        dtype=dense_scores.dtype,
    ) * mask
    feature_vjp = torch.autograd.grad(
        (feature_scores * upstream).sum(),
        (query, key),
        retain_graph=True,
    )
    dense_vjp = torch.autograd.grad(
        (dense_scores * upstream).sum(),
        (query, key),
    )
    for actual, expected in zip(feature_vjp, dense_vjp):
        torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_global_quadratic_kernel_is_square_of_raw_suffix_score():
    query, key, _ = _inputs(bits=2, sequence_length=5)
    query = query[:, :, :1]
    key = key[:, :, :1]
    raw = _feature_route_scores(
        _exact_features(query),
        _exact_features(key),
    )
    quadratic = _feature_route_scores(
        _exact_features(query, route_kernel="quadratic"),
        _exact_features(key, route_kernel="quadratic"),
    )

    torch.testing.assert_close(quadratic, raw.square(), rtol=2e-13, atol=2e-13)


def test_level_quadratic_kernel_squares_each_suffix_level_separately():
    query, key, _ = _inputs(bits=2, sequence_length=5)
    query = query[:, :, :1]
    key = key[:, :, :1]
    query_branches = suffix_kernel._suffix_feature_branches(
        query,
        representation="exact",
        fingerprint_length=3,
        mismatch_scale=2.3,
        sketch_dim=16,
        sketch_count=1,
        sketch_seed=0,
    )
    key_branches = suffix_kernel._suffix_feature_branches(
        key,
        representation="exact",
        fingerprint_length=3,
        mismatch_scale=2.3,
        sketch_dim=16,
        sketch_count=1,
        sketch_seed=0,
    )
    expected = sum(
        _feature_route_scores(query_level, key_level).square()
        for query_level, key_level in zip(
            query_branches[0],
            key_branches[0],
        )
    )
    actual = _feature_route_scores(
        _exact_features(query, route_kernel="level_quadratic"),
        _exact_features(key, route_kernel="level_quadratic"),
    )

    torch.testing.assert_close(actual, expected, rtol=2e-13, atol=2e-13)


def test_nested_tensor_sketch_is_unbiased_and_multiple_branches_reduce_error():
    symbols = torch.tensor(
        list(itertools.product((-1.0, 1.0), repeat=2))[:3],
        dtype=torch.float64,
    ).view(1, 3, 1, 2)
    local = suffix_kernel._symbol_kernel_features(symbols, 3.0)
    exact = suffix_kernel._exact_suffix_levels(local, 3)[-1][0, 0, -1]
    target = exact.square().sum()
    single_errors = []
    four_branch_errors = []
    estimates = []
    for seed in range(128):
        single = suffix_kernel._nested_tensor_sketch_suffix_levels(
            local,
            max_suffix_length=3,
            sketch_dim=32,
            sketch_seed=seed,
        )[-1][0, 0, -1]
        estimate = single.square().sum()
        estimates.append(estimate)
        single_errors.append((estimate - target).square())

        branch_estimates = []
        for branch in range(4):
            sketched = suffix_kernel._nested_tensor_sketch_suffix_levels(
                local,
                max_suffix_length=3,
                sketch_dim=32,
                sketch_seed=seed + 10_000_019 * branch,
            )[-1][0, 0, -1]
            branch_estimates.append(sketched.square().sum())
        averaged = torch.stack(branch_estimates).mean()
        four_branch_errors.append((averaged - target).square())

    assert torch.stack(estimates).mean().item() == pytest.approx(
        target.item(),
        abs=0.08,
    )
    assert torch.stack(four_branch_errors).mean() < torch.stack(single_errors).mean()


def test_sketch_raw_keeps_the_local_kernel_exact_and_seed_independent():
    query, key, _ = _inputs(bits=3, sequence_length=5)
    query = query[:, :, :1]
    key = key[:, :, :1]
    exact = _feature_route_scores(
        suffix_kernel._suffix_kernel_features(
            query,
            representation="exact",
            route_kernel="raw",
            fingerprint_length=1,
            mismatch_scale=2.1,
            sketch_dim=8,
            sketch_count=1,
            sketch_seed=0,
        ),
        suffix_kernel._suffix_kernel_features(
            key,
            representation="exact",
            route_kernel="raw",
            fingerprint_length=1,
            mismatch_scale=2.1,
            sketch_dim=8,
            sketch_count=1,
            sketch_seed=0,
        ),
    )
    for seed in (0, 1, 99):
        sketched = _feature_route_scores(
            suffix_kernel._suffix_kernel_features(
                query,
                representation="sketch",
                route_kernel="raw",
                fingerprint_length=1,
                mismatch_scale=2.1,
                sketch_dim=8,
                sketch_count=4,
                sketch_seed=seed,
            ),
            suffix_kernel._suffix_kernel_features(
                key,
                representation="sketch",
                route_kernel="raw",
                fingerprint_length=1,
                mismatch_scale=2.1,
                sketch_dim=8,
                sketch_count=4,
                sketch_seed=seed,
            ),
        )
        torch.testing.assert_close(sketched, exact, rtol=0.0, atol=0.0)


def test_sketch_global_quadratic_state_dimension_is_independent_of_window():
    logits = torch.ones(1, 6, 1, 2, dtype=torch.float64)
    dimensions = []
    for window in (1, 2, 4, 6):
        features = suffix_kernel._suffix_kernel_features(
            logits,
            representation="sketch",
            route_kernel="quadratic",
            fingerprint_length=window,
            mismatch_scale=2.0,
            sketch_dim=16,
            sketch_count=3,
            sketch_seed=7,
        )
        dimensions.append(features.size(-1))

    assert dimensions == [3 * (16 * 17 // 2)] * 4


def test_every_declared_ablation_estimator_parses_without_ambiguity():
    for estimator in ablation.PROXY_ESTIMATORS:
        config = ablation._proxy_config(estimator)
        assert estimator == (
            f"{config.representation}_{config.route_kernel}_{config.read_rule}"
        )


def test_small_ablation_report_is_strict_json_and_exposes_diagnostics():
    args = ablation.build_parser().parse_args(
        [
            "--device",
            "cpu",
            "--diagnostic-estimators",
            "production",
            "exact_raw_additive",
            "--diagnostic-seeds",
            "0",
            "--diagnostic-probes",
            "2",
            "--diagnostic-context-length",
            "24",
            "--sketch-route-kernels",
            "raw",
            "--sketch-dims",
            "8",
            "--sketch-counts",
            "1",
            "--sketch-data-seeds",
            "0",
            "--sketch-hash-seeds",
            "0",
            "--sketch-batch",
            "2",
            "--sketch-sequence-length",
            "8",
            "--skip-fit",
            "--skip-gradient",
        ]
    )
    report = ablation.run_matrix(args)

    assert report["proxy_has_trainable_parameters"] is False
    assert report["proxy_deletes_candidates"] is False
    assert set(report["diagnostic_summaries"]) == {
        "production",
        "exact_raw_additive",
    }
    assert report["sketch_summaries"]["raw:R8:C1"]["cases"] == 1
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("representation", suffix_kernel.REPRESENTATIONS)
@pytest.mark.parametrize("read_rule", suffix_kernel.READ_RULES)
@pytest.mark.parametrize("route_kernel", suffix_kernel.ROUTE_KERNELS)
def test_proxy_preserves_hard_forward_and_has_finite_vjp(
    representation,
    read_rule,
    route_kernel,
):
    inputs = _inputs()
    expected, _, _, _ = _hard_route_forward(
        *(tensor.detach() for tensor in inputs),
    )
    output = suffix_kernel.rosa_suffix_kernel_proxy(
        *inputs,
        representation=representation,
        read_rule=read_rule,
        route_kernel=route_kernel,
        max_suffix_length=3,
        fingerprint_length=3,
        mismatch_scale=2.0,
        sketch_dim=16,
        sketch_count=2,
        sketch_seed=7,
    )
    upstream = torch.randn_like(output)
    gradients = torch.autograd.grad((output * upstream).sum(), inputs)

    assert torch.equal(output, expected)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("representation", suffix_kernel.REPRESENTATIONS)
@pytest.mark.parametrize("read_rule", suffix_kernel.READ_RULES)
@pytest.mark.parametrize("route_kernel", suffix_kernel.ROUTE_KERNELS)
def test_carrier_has_no_sign_preserving_amplitude_leak(
    representation,
    read_rule,
    route_kernel,
):
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
        "representation": representation,
        "read_rule": read_rule,
        "route_kernel": route_kernel,
        "fingerprint_length": 3,
        "mismatch_scale": 2.0,
        "sketch_dim": 16,
        "sketch_count": 2,
        "sketch_seed": 11,
    }
    original = suffix_kernel._suffix_kernel_carrier(*inputs, **common)
    rescaled = suffix_kernel._suffix_kernel_carrier(*scaled, **common)

    torch.testing.assert_close(original, rescaled, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("representation", suffix_kernel.REPRESENTATIONS)
@pytest.mark.parametrize("read_rule", suffix_kernel.READ_RULES)
@pytest.mark.parametrize("route_kernel", suffix_kernel.ROUTE_KERNELS)
def test_custom_vjp_matches_materialized_carrier(
    representation,
    read_rule,
    route_kernel,
):
    inputs = _inputs()
    common = {
        "representation": representation,
        "read_rule": read_rule,
        "route_kernel": route_kernel,
        "max_suffix_length": 3,
        "fingerprint_length": 3,
        "mismatch_scale": 2.2,
        "sketch_dim": 16,
        "sketch_count": 2,
        "sketch_seed": 5,
    }
    output = suffix_kernel.rosa_suffix_kernel_proxy(*inputs, **common)
    upstream = torch.randn_like(output)
    actual = torch.autograd.grad((output * upstream).sum(), inputs)

    direct_inputs = tuple(tensor.detach().requires_grad_() for tensor in inputs)
    carrier = suffix_kernel._suffix_kernel_carrier(
        *direct_inputs,
        representation=representation,
        read_rule=read_rule,
        route_kernel=route_kernel,
        fingerprint_length=3,
        mismatch_scale=2.2,
        sketch_dim=16,
        sketch_count=2,
        sketch_seed=5,
    )
    expected = torch.autograd.grad(
        (carrier * upstream).sum(),
        direct_inputs,
    )

    for actual_gradient, expected_gradient in zip(actual, expected):
        torch.testing.assert_close(actual_gradient, expected_gradient)


def test_final_query_credit_reaches_exactly_its_suffix_fingerprint():
    query, key, value = _inputs(bits=2, sequence_length=7)
    carrier = suffix_kernel._suffix_kernel_carrier(
        query,
        key,
        value,
        representation="exact",
        read_rule="additive",
        route_kernel="raw",
        fingerprint_length=3,
        mismatch_scale=2.0,
        sketch_dim=16,
        sketch_count=1,
        sketch_seed=0,
    )
    upstream = torch.zeros_like(carrier)
    upstream[:, -1] = 1.0
    (query_gradient,) = torch.autograd.grad(
        (carrier * upstream).sum(),
        (query,),
    )
    active_positions = query_gradient.abs().sum(dim=(0, 2, 3)) > 0

    assert torch.equal(
        active_positions,
        torch.tensor([False, False, False, False, True, True, True]),
    )


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    (
        ("representation", "unknown", "representation"),
        ("read_rule", "unknown", "read_rule"),
        ("route_kernel", "unknown", "route_kernel"),
        ("fingerprint_length", 0, "fingerprint_length"),
        ("sketch_dim", 12, "power of two"),
        ("sketch_count", 0, "positive"),
    ),
)
def test_invalid_configuration_is_rejected(keyword, value, message):
    inputs = _inputs()
    kwargs = {
        "representation": "sketch",
        "read_rule": "additive",
        "route_kernel": "level_quadratic",
        "max_suffix_length": 3,
        "fingerprint_length": 3,
        "sketch_dim": 16,
        "sketch_count": 2,
    }
    kwargs[keyword] = value
    with pytest.raises((TypeError, ValueError), match=message):
        suffix_kernel.rosa_suffix_kernel_proxy(*inputs, **kwargs)
