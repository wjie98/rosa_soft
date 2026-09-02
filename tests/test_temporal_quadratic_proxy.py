import math

import pytest
import torch

from benchmarks import fast_weight_proxy_ablation as ablation
from benchmarks import temporal_quadratic_proxy as temporal
from rosa_soft.soft_reference import _hard_route_forward, _hard_sign


def _inputs(*, sequence_length=7, bits=4, dtype=torch.float64):
    generator = torch.Generator().manual_seed(41 + sequence_length + bits)
    query = torch.randn(
        1,
        sequence_length,
        2,
        bits,
        generator=generator,
        dtype=dtype,
    )
    key = torch.randn(
        1,
        sequence_length,
        2,
        bits,
        generator=generator,
        dtype=dtype,
    )
    value = torch.randn(
        1,
        sequence_length,
        1,
        3,
        generator=generator,
        dtype=dtype,
    )
    return query, key, value


def _explicit_suffix_states(logits, window, state_dim):
    symbols = _hard_sign(logits).permute(0, 2, 1, 3)
    projection = temporal._fixed_projection(
        symbols.size(-1),
        state_dim,
        dtype=symbols.dtype,
        device=symbols.device,
    )
    projected = torch.einsum("bhtd,rd->bhtr", symbols, projection)
    angles = temporal._rotation_angles(
        state_dim,
        window,
        dtype=symbols.dtype,
        device=symbols.device,
    )
    outputs = []
    for position in range(symbols.size(2)):
        state = torch.zeros_like(projected[:, :, 0])
        for offset in range(min(window, position + 1)):
            state = state + temporal._rotate_pairs(
                projected[:, :, position - offset],
                angles * offset,
            )
        outputs.append(state / math.sqrt(window * symbols.size(-1)))
    return torch.stack(outputs, dim=2)


@pytest.mark.parametrize("window", [1, 2, 4, 7])
def test_orthogonal_rolling_scan_matches_explicit_suffix_sum(window):
    query, _, _ = _inputs(sequence_length=9)

    actual = temporal._temporal_suffix_states(query, window, state_dim=8)
    expected = _explicit_suffix_states(query, window, state_dim=8)

    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


def test_fixed_projection_has_orthonormal_symbol_columns():
    projection = temporal._fixed_projection(
        8,
        16,
        dtype=torch.float64,
        device=torch.device("cpu"),
    )

    torch.testing.assert_close(
        projection.mT @ projection,
        torch.eye(8, dtype=torch.float64),
        rtol=0.0,
        atol=2e-14,
    )


def test_pair_rotation_is_norm_preserving():
    generator = torch.Generator().manual_seed(92)
    state = torch.randn(3, 5, 16, generator=generator, dtype=torch.float64)
    angles = temporal._rotation_angles(
        16,
        32,
        dtype=state.dtype,
        device=state.device,
    )

    rotated = temporal._rotate_pairs(state, angles)

    torch.testing.assert_close(
        rotated.square().sum(dim=-1),
        state.square().sum(dim=-1),
        rtol=2e-14,
        atol=2e-14,
    )


def test_quadratic_features_are_exact_degree_two_polynomial_kernel():
    generator = torch.Generator().manual_seed(101)
    left = torch.randn(2, 3, 8, generator=generator, dtype=torch.float64)
    right = torch.randn(2, 4, 8, generator=generator, dtype=torch.float64)
    left_features = temporal._quadratic_polynomial_features(left)
    right_features = temporal._quadratic_polynomial_features(right)

    actual = torch.einsum("bif,bjf->bij", left_features, right_features)
    expected = torch.einsum("bir,bjr->bij", left, right).square()

    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


def test_identical_suffix_is_a_stationary_query_match():
    query, _, _ = _inputs(sequence_length=6)
    query.requires_grad_()
    query_features = temporal._temporal_quadratic_fingerprints(
        query,
        max_suffix_length=4,
        state_dim=8,
    )
    key_features = temporal._temporal_quadratic_fingerprints(
        query.detach(),
        max_suffix_length=4,
        state_dim=8,
    )
    similarity = (
        query_features[:, :, -1] * key_features[:, :, -1]
    ).sum()
    gradient = torch.autograd.grad(similarity, query)[0]

    torch.testing.assert_close(
        gradient,
        torch.zeros_like(gradient),
        rtol=0.0,
        atol=2e-14,
    )


def test_temporal_carrier_matches_explicit_causal_candidate_sum():
    query, key, value = _inputs(sequence_length=6)
    window = 3
    query_features = temporal._temporal_quadratic_fingerprints(
        query,
        window,
        state_dim=8,
    )
    key_features = temporal._temporal_quadratic_fingerprints(
        key,
        window,
        state_dim=8,
    )
    hard_values = _hard_sign(value).permute(0, 2, 1, 3).expand(-1, 2, -1, -1)
    expected = torch.zeros(1, 2, 6, 3, dtype=torch.float64)
    for position in range(1, 6):
        weights = torch.einsum(
            "bhf,bhaf->bha",
            query_features[:, :, position],
            key_features[:, :, :position],
        )
        numerator = torch.einsum(
            "bha,bhav->bhv",
            weights,
            hard_values[:, :, 1 : position + 1],
        )
        expected[:, :, position] = numerator / weights.sum(
            dim=-1,
            keepdim=True,
        ).clamp_min(1e-6)

    actual = temporal._temporal_quadratic_carrier(
        query,
        key,
        value,
        max_suffix_length=window,
        state_dim=8,
    ).permute(0, 2, 1, 3)

    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("window", [1, 3, 6])
def test_public_proxy_preserves_exact_hard_forward(window):
    inputs = tuple(tensor.requires_grad_() for tensor in _inputs())
    expected, _, _, _ = _hard_route_forward(
        *(tensor.detach() for tensor in inputs),
    )
    actual = temporal.rosa_temporal_quadratic_proxy(
        *inputs,
        max_suffix_length=window,
        state_dim=8,
    )

    assert torch.equal(actual, expected)


def test_custom_vjp_matches_materialized_temporal_carrier():
    direct_inputs = tuple(
        tensor.requires_grad_() for tensor in _inputs(sequence_length=6)
    )
    wrapped_inputs = tuple(
        tensor.detach().clone().requires_grad_() for tensor in direct_inputs
    )
    generator = torch.Generator().manual_seed(18)
    upstream = torch.randn(
        1,
        6,
        2,
        3,
        generator=generator,
        dtype=torch.float64,
    )
    direct = temporal._temporal_quadratic_carrier(
        *direct_inputs,
        max_suffix_length=4,
        state_dim=8,
    )
    wrapped = temporal.rosa_temporal_quadratic_proxy(
        *wrapped_inputs,
        max_suffix_length=4,
        state_dim=8,
    )
    direct_gradients = torch.autograd.grad(
        (direct * upstream).sum(),
        direct_inputs,
    )
    wrapped_gradients = torch.autograd.grad(
        (wrapped * upstream).sum(),
        wrapped_inputs,
    )

    for actual, expected in zip(wrapped_gradients, direct_gradients):
        torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


def test_final_query_reaches_every_and_only_active_suffix_query_position():
    window = 4
    query, key, value = _inputs(sequence_length=8)
    query.requires_grad_()
    carrier = temporal._temporal_quadratic_carrier(
        query,
        key,
        value,
        max_suffix_length=window,
        state_dim=8,
    )
    gradient = torch.autograd.grad(carrier[:, -1].sum(), query)[0]
    position_norms = gradient.square().sum(dim=(0, 2, 3)).sqrt()

    assert torch.equal(position_norms[:-window], torch.zeros(8 - window))
    assert bool(position_norms[-window:].gt(0).all())


def test_carrier_has_no_sign_preserving_amplitude_leak():
    inputs = _inputs()
    generator = torch.Generator().manual_seed(72)
    scaled = tuple(
        tensor
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
    common = {"max_suffix_length": 4, "state_dim": 8}

    original = temporal._temporal_quadratic_carrier(*inputs, **common)
    rescaled = temporal._temporal_quadratic_carrier(*scaled, **common)

    assert torch.equal(original, rescaled)


@pytest.mark.parametrize("state_dim", [True, 7, 4, 256])
def test_invalid_state_dimensions_are_rejected(state_dim):
    inputs = _inputs(bits=8)
    expected_error = TypeError if state_dim is True else ValueError

    with pytest.raises(expected_error):
        temporal.rosa_temporal_quadratic_proxy(
            *inputs,
            max_suffix_length=3,
            state_dim=state_dim,
        )


def test_matched_ablation_can_compare_temporal_proxy_to_bitflip():
    args = ablation.build_parser().parse_args(
        [
            "--gradient-only",
            "--estimators",
            "production",
            "bitflip",
            ablation.TEMPORAL_ESTIMATOR,
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
            "--temporal-state-dim",
            "8",
        ]
    )

    report = ablation.run_matrix(args)

    assert set(report["gradient_summaries"]) == {
        "production",
        "bitflip",
        ablation.TEMPORAL_ESTIMATOR,
    }
    assert report["gradient_summaries"][ablation.TEMPORAL_ESTIMATOR][
        "hard_forward_equal_cases"
    ] == 1
    assert report["state_dimensions"]["temporal_state"] == 8
    assert report["state_dimensions"]["temporal_quadratic"] == 36
