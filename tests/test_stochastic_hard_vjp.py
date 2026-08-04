import math

import pytest
import torch

from benchmarks.global_bit_oracle import (
    exact_shared_bit_oracle,
    mean_field_winner_oracle,
)
from benchmarks.stochastic_hard_vjp import (
    _ARM,
    _DISARM,
    _REFERENCE,
    _mean_field_latest_probabilities,
    _sampled_qk_vjp,
    mean_field_hard_rosa,
    stochastic_hard_rosa,
)
from rosa_soft import rosa_soft_reference


def _random_inputs():
    generator = torch.Generator().manual_seed(73)
    query = 0.25 * torch.randn(
        2, 4, 2, 2, generator=generator, dtype=torch.float64
    )
    key = 0.25 * torch.randn(
        2, 4, 2, 2, generator=generator, dtype=torch.float64
    )
    value = torch.randn(
        2, 4, 1, 3, generator=generator, dtype=torch.float64
    )
    upstream = torch.randn(
        2, 4, 2, 3, generator=generator, dtype=torch.float64
    )
    return query, key, value, upstream


@pytest.mark.parametrize("estimator", ["arm", "disarm"])
def test_stochastic_vjp_preserves_hard_forward_and_production_value_vjp(
    estimator,
):
    query, key, value, upstream = _random_inputs()
    production_inputs = [
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    ]
    stochastic_inputs = [
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    ]

    production_output = rosa_soft_reference(
        *production_inputs,
        max_suffix_length=3,
        scale=0.7,
        dropout_p=0.0,
        mismatch_scale=2.5,
    )
    production_output.backward(upstream)
    torch.manual_seed(101)
    stochastic_output = stochastic_hard_rosa(
        *stochastic_inputs,
        estimator=estimator,
        bit_temperature=0.6,
        pairs=32,
        max_suffix_length=3,
        scale=0.7,
        mismatch_scale=2.5,
    )
    stochastic_output.backward(upstream)

    torch.testing.assert_close(
        stochastic_output, production_output, rtol=0, atol=0
    )
    torch.testing.assert_close(
        stochastic_inputs[2].grad,
        production_inputs[2].grad,
        rtol=0,
        atol=0,
    )
    assert torch.isfinite(stochastic_inputs[0].grad).all()
    assert torch.isfinite(stochastic_inputs[1].grad).all()
    assert torch.count_nonzero(stochastic_inputs[0].grad[:, 0]) == 0
    assert torch.count_nonzero(stochastic_inputs[1].grad[:, -1]) == 0


@pytest.mark.parametrize("estimator", ["arm", "disarm"])
def test_stochastic_vjp_is_reproducible_from_torch_rng(estimator):
    query, key, value, upstream = _random_inputs()

    def run(seed):
        inputs = [
            tensor.detach().clone().requires_grad_()
            for tensor in (query, key, value)
        ]
        torch.manual_seed(seed)
        output = stochastic_hard_rosa(
            *inputs,
            estimator=estimator,
            bit_temperature=0.5,
            pairs=16,
            max_suffix_length=2,
        )
        output.backward(upstream)
        return output.detach(), tuple(tensor.grad for tensor in inputs)

    first_output, first_gradients = run(307)
    repeated_output, repeated_gradients = run(307)
    other_output, other_gradients = run(311)

    torch.testing.assert_close(first_output, repeated_output, rtol=0, atol=0)
    torch.testing.assert_close(first_output, other_output, rtol=0, atol=0)
    for first, repeated in zip(first_gradients, repeated_gradients):
        torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    assert not (
        torch.equal(first_gradients[0], other_gradients[0])
        and torch.equal(first_gradients[1], other_gradients[1])
    )


@pytest.mark.parametrize(
    "estimator_code", [_ARM, _DISARM], ids=["arm", "disarm"]
)
def test_grouped_stochastic_vjp_matches_one_bit_exact_oracle(estimator_code):
    batch = 2
    heads = 2
    samples = 400
    probability = 0.25
    relevant_logit = math.log(probability / (1.0 - probability))
    query = torch.full(
        (batch, 2, heads, 1), -1.0, dtype=torch.float64
    )
    query[:, 1] = relevant_logit
    key = torch.full((batch, 2, heads, 1), 50.0, dtype=torch.float64)
    value = torch.tensor(
        [
            [[[0.0], [0.0]], [[1.0], [-1.0]]],
            [[[0.0], [0.0]], [[-1.0], [1.0]]],
        ],
        dtype=torch.float64,
    )
    upstream = torch.zeros(batch, 2, heads, 1, dtype=torch.float64)
    upstream[:, 1, :, 0] = torch.tensor([[1.0, 2.0], [-3.0, 4.0]])
    grid = (
        torch.arange(samples, dtype=torch.float64) + 0.5
    ) / samples
    query_uniforms = torch.full(
        (samples, batch, 2, heads, 1), 0.5, dtype=torch.float64
    )
    query_uniforms[:, :, 1, :, 0] = grid.view(-1, 1, 1)
    key_uniforms = torch.full_like(query_uniforms, 0.5)

    query_gradient, key_gradient = _sampled_qk_vjp(
        query,
        key,
        value,
        upstream,
        query_uniforms,
        key_uniforms,
        max_suffix_length=1,
        bit_temperature=1.0,
        estimator_code=estimator_code,
        backend_code=_REFERENCE,
    )

    for batch_index in range(batch):
        for head_index in range(heads):
            query_mask = torch.zeros(1, 2, 1, 1, dtype=torch.bool)
            query_mask[:, 1] = True
            key_mask = torch.zeros_like(query_mask)
            exact = exact_shared_bit_oracle(
                query[batch_index : batch_index + 1, :, head_index : head_index + 1],
                key[batch_index : batch_index + 1, :, head_index : head_index + 1],
                value[batch_index : batch_index + 1, :, head_index : head_index + 1],
                upstream[
                    batch_index : batch_index + 1,
                    :,
                    head_index : head_index + 1,
                ],
                max_suffix_length=1,
                bit_temperature=1.0,
                query_stochastic_mask=query_mask,
                key_stochastic_mask=key_mask,
            )
            torch.testing.assert_close(
                query_gradient[
                    batch_index : batch_index + 1,
                    :,
                    head_index : head_index + 1,
                ],
                exact.query_gradient,
                rtol=0,
                atol=2e-15,
            )
    assert torch.count_nonzero(key_gradient) == 0


@pytest.mark.parametrize("estimator", ["arm", "disarm"])
def test_group_reward_does_not_leak_across_batch_or_heads(estimator):
    query, key, value, _ = _random_inputs()
    value = value.expand(-1, -1, 2, -1).contiguous()
    upstream = torch.zeros(2, 4, 2, 3, dtype=torch.float64)
    upstream[0, :, 1] = torch.tensor(
        [[1.0, -2.0, 0.5]] * 4, dtype=torch.float64
    )
    inputs = [
        tensor.detach().clone().requires_grad_()
        for tensor in (query, key, value)
    ]

    torch.manual_seed(419)
    output = stochastic_hard_rosa(
        *inputs,
        estimator=estimator,
        pairs=64,
        max_suffix_length=2,
    )
    output.backward(upstream)

    assert torch.count_nonzero(inputs[0].grad[1]) == 0
    assert torch.count_nonzero(inputs[1].grad[1]) == 0
    assert torch.count_nonzero(inputs[0].grad[0, :, 0]) == 0
    assert torch.count_nonzero(inputs[1].grad[0, :, 0]) == 0


def test_stochastic_vjp_rejects_nested_attention_dropout():
    query, key, value, _ = _random_inputs()
    with pytest.raises(ValueError, match="dropout_p=0"):
        stochastic_hard_rosa(
            query.requires_grad_(),
            key,
            value,
            estimator="arm",
            dropout_p=0.1,
        )


def test_batched_mean_field_probabilities_are_normalized_and_causal():
    query, key, _, _ = _random_inputs()
    probabilities = _mean_field_latest_probabilities(query, key, 0.7)

    torch.testing.assert_close(
        probabilities.sum(dim=-1),
        torch.ones(2, 2, 4, dtype=torch.float64),
        rtol=0,
        atol=2e-15,
    )
    for query_index in range(4):
        assert torch.count_nonzero(
            probabilities[:, :, query_index, query_index + 1 :]
        ) == 0


def test_mean_field_hard_vjp_matches_single_group_oracle():
    query, key, value, upstream = _random_inputs()
    query = query[:1, :, :1].detach().requires_grad_()
    key = key[:1, :, :1].detach().requires_grad_()
    value = value[:1].detach().requires_grad_()
    upstream = upstream[:1, :, :1]
    oracle = mean_field_winner_oracle(
        query.detach(),
        key.detach(),
        value.detach(),
        upstream,
        max_suffix_length=1,
        bit_temperature=0.7,
    )
    production_value = value.detach().clone().requires_grad_()
    production_output = rosa_soft_reference(
        query.detach(),
        key.detach(),
        production_value,
        max_suffix_length=1,
    )
    production_output.backward(upstream)

    output = mean_field_hard_rosa(
        query,
        key,
        value,
        bit_temperature=0.7,
        max_suffix_length=1,
    )
    output.backward(upstream)

    torch.testing.assert_close(output, production_output, rtol=0, atol=0)
    torch.testing.assert_close(query.grad, oracle.query_gradient)
    torch.testing.assert_close(key.grad, oracle.key_gradient)
    torch.testing.assert_close(value.grad, production_value.grad, rtol=0, atol=0)


def test_mean_field_hard_vjp_rejects_suffix_windows_above_one():
    query, key, value, _ = _random_inputs()
    with pytest.raises(ValueError, match="only W=1"):
        mean_field_hard_rosa(
            query.requires_grad_(),
            key,
            value,
            max_suffix_length=2,
        )
