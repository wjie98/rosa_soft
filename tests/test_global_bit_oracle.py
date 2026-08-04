import math

import pytest
import torch

from benchmarks.global_bit_oracle import (
    arm_disarm_samples,
    bitflip_residual_samples,
    exact_bitflip_vjp,
    exact_margin_edit_oracle,
    exact_shared_bit_oracle,
    mean_field_winner_oracle,
    sampled_bitflip_residual_vjp,
)
from rosa_soft.soft_reference import _hard_route_forward


def _base_inputs(sequence_length=3, qk_bits=1, value_bits=2):
    generator = torch.Generator().manual_seed(41)
    query = torch.randn(
        1,
        sequence_length,
        1,
        qk_bits,
        generator=generator,
        dtype=torch.float64,
    )
    key = torch.randn(
        1,
        sequence_length,
        1,
        qk_bits,
        generator=generator,
        dtype=torch.float64,
    )
    value = torch.randn(
        1,
        sequence_length,
        1,
        value_bits,
        generator=generator,
        dtype=torch.float64,
    )
    grad_output = torch.randn_like(value, generator=generator)
    return query, key, value, grad_output


def test_exact_shared_bit_oracle_normalizes_and_matches_local_expectation():
    query, key, value, grad_output = _base_inputs(
        sequence_length=4,
        qk_bits=2,
    )
    result = exact_shared_bit_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=3,
        bit_temperature=0.8,
    )

    assert result.layout.bit_count == 12
    assert result.state_count == 4096
    assert result.probability_sum == pytest.approx(1.0, abs=1e-12)
    torch.testing.assert_close(
        result.bit_gradient,
        result.local_expectation_bit_gradient,
        rtol=1e-11,
        atol=1e-12,
    )
    torch.testing.assert_close(
        result.route_probabilities.sum(dim=-1),
        torch.ones(1, 4, 1, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


def test_exact_shared_bit_oracle_matches_finite_differences():
    query, key, value, grad_output = _base_inputs(
        sequence_length=3,
        qk_bits=1,
    )
    temperature = 0.7
    epsilon = 1e-5
    exact = exact_shared_bit_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=2,
        bit_temperature=temperature,
    )

    for tensor_name, indices, exact_gradient in (
        ("query", exact.layout.query_indices, exact.query_gradient),
        ("key", exact.layout.key_indices, exact.key_gradient),
    ):
        for flat_index in indices:
            positive_query = query.clone()
            negative_query = query.clone()
            positive_key = key.clone()
            negative_key = key.clone()
            positive = positive_query if tensor_name == "query" else positive_key
            negative = negative_query if tensor_name == "query" else negative_key
            positive.reshape(-1)[flat_index] += epsilon
            negative.reshape(-1)[flat_index] -= epsilon
            positive_scalar = exact_shared_bit_oracle(
                positive_query,
                positive_key,
                value,
                grad_output,
                max_suffix_length=2,
                bit_temperature=temperature,
            ).expected_scalar
            negative_scalar = exact_shared_bit_oracle(
                negative_query,
                negative_key,
                value,
                grad_output,
                max_suffix_length=2,
                bit_temperature=temperature,
            ).expected_scalar
            finite_difference = (positive_scalar - negative_scalar) / (
                2.0 * epsilon
            )
            assert finite_difference.item() == pytest.approx(
                exact_gradient.reshape(-1)[flat_index].item(),
                rel=2e-9,
                abs=2e-10,
            )


def test_shared_query_bit_exposes_mean_field_impossible_route():
    query = torch.tensor(
        [[[[ -1.0]], [[-1.0]], [[0.0]]]],
        dtype=torch.float64,
    )
    key = torch.tensor(
        [[[[1.0]], [[1.0]], [[-1.0]]]],
        dtype=torch.float64,
    )
    value = torch.tensor(
        [[[[0.0]], [[1.0]], [[-1.0]]]],
        dtype=torch.float64,
    )
    grad_output = torch.zeros_like(value)
    grad_output[:, 2] = 1.0
    query_mask = torch.zeros_like(query, dtype=torch.bool)
    query_mask[:, 2] = True
    key_mask = torch.zeros_like(key, dtype=torch.bool)

    exact = exact_shared_bit_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=1,
        query_stochastic_mask=query_mask,
        key_stochastic_mask=key_mask,
    )
    mean_field = mean_field_winner_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=1,
        query_stochastic_mask=query_mask,
        key_stochastic_mask=key_mask,
    )

    torch.testing.assert_close(
        exact.route_probabilities[0, 2, 0],
        torch.tensor([0.5, 0.0, 0.5], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        mean_field.route_probabilities[0, 2, 0],
        torch.tensor([0.25, 0.25, 0.5], dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )
    torch.testing.assert_close(
        mean_field.route_probabilities.sum(dim=-1),
        torch.ones(1, 3, 1, dtype=torch.float64),
        rtol=0,
        atol=1e-12,
    )


def test_fixed_bit_oracles_reduce_to_one_hard_state():
    query, key, value, grad_output = _base_inputs()
    query_mask = torch.zeros_like(query, dtype=torch.bool)
    key_mask = torch.zeros_like(key, dtype=torch.bool)
    hard_output, _, hard_routes, _ = _hard_route_forward(query, key, value, 2)
    exact = exact_shared_bit_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=2,
        query_stochastic_mask=query_mask,
        key_stochastic_mask=key_mask,
    )
    mean_field = mean_field_winner_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=2,
        query_stochastic_mask=query_mask,
        key_stochastic_mask=key_mask,
    )

    assert exact.state_count == 1
    assert exact.bit_gradient.numel() == 0
    torch.testing.assert_close(exact.expected_output, hard_output)
    torch.testing.assert_close(mean_field.expected_output, hard_output)
    torch.testing.assert_close(
        exact.route_probabilities.argmax(dim=-1).permute(0, 2, 1),
        hard_routes,
    )
    assert torch.count_nonzero(mean_field.query_gradient) == 0
    assert torch.count_nonzero(mean_field.key_gradient) == 0


def test_arm_and_disarm_match_one_bit_exact_gradient():
    probability = 0.3
    query = torch.tensor(
        [[[[ -1.0]], [[math.log(probability / (1.0 - probability))]]]],
        dtype=torch.float64,
    )
    key = torch.tensor(
        [[[[1.0]], [[-1.0]]]],
        dtype=torch.float64,
    )
    value = torch.tensor(
        [[[[0.0]], [[1.0]]]],
        dtype=torch.float64,
    )
    grad_output = torch.zeros_like(value)
    grad_output[:, 1] = 1.0
    query_mask = torch.zeros_like(query, dtype=torch.bool)
    query_mask[:, 1] = True
    key_mask = torch.zeros_like(key, dtype=torch.bool)
    sample_count = 10_000
    uniforms = (
        torch.arange(sample_count, dtype=torch.float64).unsqueeze(1) + 0.5
    ) / sample_count

    exact = exact_shared_bit_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=1,
        query_stochastic_mask=query_mask,
        key_stochastic_mask=key_mask,
    )
    samples = arm_disarm_samples(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=1,
        sample_count=sample_count,
        uniforms=uniforms,
        query_stochastic_mask=query_mask,
        key_stochastic_mask=key_mask,
    )

    assert exact.expected_scalar == pytest.approx(probability, abs=1e-12)
    assert exact.bit_gradient.item() == pytest.approx(
        probability * (1.0 - probability),
        abs=1e-12,
    )
    assert samples.arm_bit_gradients.mean().item() == pytest.approx(
        exact.bit_gradient.item(),
        abs=1e-12,
    )
    assert samples.disarm_bit_gradients.mean().item() == pytest.approx(
        exact.bit_gradient.item(),
        abs=1e-12,
    )


def test_arm_and_disarm_match_two_bit_stratified_expectation():
    query = torch.tensor(
        [[[[1.0]], [[-1.0]], [[1.0]]]],
        dtype=torch.float64,
    )
    key = torch.tensor(
        [[[[math.log(0.25 / 0.75)]], [[0.0]], [[1.0]]]],
        dtype=torch.float64,
    )
    value = torch.tensor(
        [[[[1.0]], [[-1.0]], [[1.0]]]],
        dtype=torch.float64,
    )
    grad_output = torch.tensor(
        [[[[0.0]], [[0.3]], [[-0.7]]]],
        dtype=torch.float64,
    )
    query_mask = torch.zeros_like(query, dtype=torch.bool)
    key_mask = torch.zeros_like(key, dtype=torch.bool)
    key_mask[:, :2] = True
    grid = (torch.arange(4, dtype=torch.float64) + 0.5) / 4.0
    uniforms = torch.cartesian_prod(grid, grid)
    exact = exact_shared_bit_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=2,
        query_stochastic_mask=query_mask,
        key_stochastic_mask=key_mask,
    )
    samples = arm_disarm_samples(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=2,
        sample_count=uniforms.size(0),
        uniforms=uniforms,
        query_stochastic_mask=query_mask,
        key_stochastic_mask=key_mask,
    )

    torch.testing.assert_close(
        samples.arm_bit_gradients.mean(dim=0),
        exact.bit_gradient,
        rtol=0,
        atol=1e-14,
    )
    torch.testing.assert_close(
        samples.disarm_bit_gradients.mean(dim=0),
        exact.bit_gradient,
        rtol=0,
        atol=1e-14,
    )


def test_bitflip_residual_is_exact_when_averaged_over_coordinates():
    surrogate = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)
    bitflip = torch.tensor([-1.0, 0.5, 0.1], dtype=torch.float64)
    indices = torch.arange(3).view(3, 1)

    samples = bitflip_residual_samples(
        surrogate,
        bitflip,
        sample_count=3,
        flips_per_sample=1,
        sampled_indices=indices,
    )

    torch.testing.assert_close(
        samples.mean(dim=0),
        bitflip,
        rtol=0,
        atol=1e-15,
    )
    assert torch.count_nonzero(samples - surrogate.unsqueeze(0)) == 3


def test_bitflip_residual_is_exact_when_every_bit_is_sampled():
    surrogate = torch.tensor([0.2, -0.4, 0.7], dtype=torch.float64)
    bitflip = torch.tensor([-1.0, 0.5, 0.1], dtype=torch.float64)

    samples = bitflip_residual_samples(
        surrogate,
        bitflip,
        sample_count=4,
        flips_per_sample=3,
        seed=7,
    )

    torch.testing.assert_close(
        samples,
        bitflip.expand_as(samples),
        rtol=0,
        atol=1e-15,
    )


def test_sampled_bitflip_vjp_evaluates_only_selected_residuals():
    query, key, value, grad_output = _base_inputs(
        sequence_length=3,
        qk_bits=1,
    )
    generator = torch.Generator().manual_seed(99)
    surrogate_query = torch.randn(
        query.shape,
        generator=generator,
        dtype=query.dtype,
    )
    surrogate_key = torch.randn(
        key.shape,
        generator=generator,
        dtype=key.dtype,
    )
    result = sampled_bitflip_residual_vjp(
        query,
        key,
        value,
        grad_output,
        surrogate_query,
        surrogate_key,
        max_suffix_length=2,
        flips_per_sample=2,
        seed=13,
    )
    _, _, exact_bitflip, layout = exact_bitflip_vjp(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=2,
    )
    surrogate_bits = layout.gather(surrogate_query, surrogate_key)
    expected = bitflip_residual_samples(
        surrogate_bits,
        exact_bitflip,
        sample_count=1,
        flips_per_sample=2,
        sampled_indices=result.sampled_indices.unsqueeze(0),
    )[0]

    torch.testing.assert_close(result.bit_gradient, expected)
    torch.testing.assert_close(
        layout.gather(result.query_gradient, result.key_gradient),
        expected,
    )


def test_margin_edit_finds_coordinated_winner_change():
    query = torch.tensor(
        [[[[ -1.0]], [[-1.0]], [[2.0]]]],
        dtype=torch.float64,
    )
    key = torch.tensor(
        [[[[ -0.1]], [[0.1]], [[-1.0]]]],
        dtype=torch.float64,
    )
    value = torch.tensor(
        [[[[0.0]], [[1.0]], [[-1.0]]]],
        dtype=torch.float64,
    )
    grad_output = torch.zeros_like(value)
    grad_output[:, 2] = -1.0

    base_output, _, base_routes, _ = _hard_route_forward(
        query,
        key,
        value,
        2,
    )
    result = exact_margin_edit_oracle(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=2,
        eta=1.0,
    )
    verified_output, _, verified_routes, _ = _hard_route_forward(
        result.target_query_symbols,
        result.target_key_symbols,
        value,
        2,
    )

    assert base_routes[0, 0, 2].item() == 2
    assert base_output[0, 2, 0, 0].item() == -1.0
    assert result.flipped_bits == 2
    assert result.target_routes[0, 0, 2].item() == 1
    assert result.linearized_cost == pytest.approx(-2.0)
    assert result.margin_penalty == pytest.approx(0.4)
    assert result.objective_gain == pytest.approx(1.6)
    assert torch.equal(result.target_output, verified_output)
    assert torch.equal(result.target_routes, verified_routes)


def test_margin_edit_keeps_base_when_upstream_is_zero():
    query, key, value, _ = _base_inputs()
    result = exact_margin_edit_oracle(
        query,
        key,
        value,
        torch.zeros_like(value),
        max_suffix_length=2,
    )

    assert result.flipped_bits == 0
    assert result.objective_gain == pytest.approx(0.0)
    assert torch.count_nonzero(result.query_gradient) == 0
    assert torch.count_nonzero(result.key_gradient) == 0


def test_margin_edit_prefers_base_across_zero_logit_ties():
    query = torch.zeros(1, 2, 1, 1, dtype=torch.float64)
    key = torch.zeros_like(query)
    value = torch.tensor(
        [[[[0.0]], [[1.0]]]],
        dtype=torch.float64,
    )
    result = exact_margin_edit_oracle(
        query,
        key,
        value,
        torch.zeros_like(value),
        max_suffix_length=1,
    )

    assert result.flipped_bits == 0
    assert result.objective_gain == pytest.approx(0.0)
    assert torch.count_nonzero(result.query_gradient) == 0
    assert torch.count_nonzero(result.key_gradient) == 0


def test_oracle_rejects_intractable_enumeration():
    query, key, value, grad_output = _base_inputs(
        sequence_length=4,
        qk_bits=2,
    )
    with pytest.raises(ValueError, match="refusing to enumerate"):
        exact_shared_bit_oracle(
            query,
            key,
            value,
            grad_output,
            max_bits=8,
        )
