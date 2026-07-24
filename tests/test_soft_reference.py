import math

import pytest
import torch

from rosa_soft import rosa_soft_reference
from rosa_soft.soft_reference import (
    ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    _action_mask,
    _pairwise_proxy_local_match,
    _sign_with_softsign_jacobian,
)
from rosa_soft.testing import inspect_rosa_soft


def _nonzero_randn(shape, *, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(shape, dtype=dtype, generator=generator)
    signs = torch.where(values >= 0, torch.ones_like(values), -torch.ones_like(values))
    return signs * (values.abs() + 0.2)


def _inspect(query, key, value, **kwargs):
    return inspect_rosa_soft(
        query,
        key,
        value,
        mismatch_noise=kwargs.pop(
            "mismatch_noise",
            torch.full(
                (
                    query.size(0),
                    query.size(2),
                    query.size(1),
                    max(query.size(1) - 1, 0),
                    query.size(3),
                ),
                0.37,
                dtype=query.dtype,
                device=query.device,
            ),
        ),
        **kwargs,
    )


def _naive_hard_lengths(query, key, max_suffix_length):
    batch, seq_len, heads, _ = query.shape
    q_bits = query > 0
    k_bits = key > 0
    lengths = torch.zeros(batch, heads, seq_len, seq_len, dtype=query.dtype)
    for b in range(batch):
        for h in range(heads):
            for query_pos in range(seq_len):
                for action in range(1, query_pos + 1):
                    limit = min(max_suffix_length, query_pos + 1, action)
                    for offset in range(limit):
                        key_pos = action - 1 - offset
                        if torch.equal(
                            q_bits[b, query_pos - offset, h],
                            k_bits[b, key_pos, h],
                        ):
                            lengths[b, h, query_pos, action] += 1
                        else:
                            break
    return lengths


def test_softsign_jacobian_is_exact_at_zero_and_nonfinite_values():
    logits = torch.tensor(
        [-math.inf, -1.0, 0.0, 1.0, math.inf, math.nan],
        dtype=torch.float64,
        requires_grad=True,
    )

    hard, straight_through = _sign_with_softsign_jacobian(logits)
    gradient = torch.autograd.grad(straight_through.sum(), logits)[0]

    torch.testing.assert_close(
        straight_through,
        hard,
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        gradient,
        torch.tensor(
            [0.0, 0.25, 1.0, 0.25, 0.0, math.nan],
            dtype=torch.float64,
        ),
        rtol=0,
        atol=0,
        equal_nan=True,
    )


@pytest.mark.parametrize("max_suffix_length", [1, 2, 4, 20])
def test_hard_suffix_dynamic_program_matches_naive(max_suffix_length):
    query = _nonzero_randn((2, 7, 2, 4), seed=1)
    key = _nonzero_randn((2, 7, 2, 4), seed=2)
    value = _nonzero_randn((2, 7, 1, 3), seed=3)

    _, inspection = _inspect(query, key, value, max_suffix_length=max_suffix_length)

    expected = _naive_hard_lengths(query, key, max_suffix_length)
    torch.testing.assert_close(inspection.hard_lengths, expected, rtol=0, atol=0)


def test_forward_is_hard_latest_tie_and_null_is_positive_zero():
    seq_len = 6
    query = torch.ones(1, seq_len, 1, 3, requires_grad=True)
    key = torch.ones_like(query, requires_grad=True)
    value = _nonzero_randn((1, seq_len, 1, 2), dtype=torch.float32, seed=4)
    value.requires_grad_()

    output, inspection = _inspect(query, key, value, max_suffix_length=3)

    expected = torch.zeros_like(output)
    expected[:, 1:] = torch.where(value[:, 1:] > 0, 1.0, -1.0)
    assert torch.equal(output, expected)
    assert torch.equal(inspection.selected_actions[0, 0], torch.arange(seq_len))
    assert set(output.unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert not torch.signbit(output[:, 0]).any()

    null_output = rosa_soft_reference(query, -key, value)
    assert torch.equal(null_output, torch.zeros_like(null_output))
    assert not torch.signbit(null_output).any()


def test_hard_forward_is_independent_of_soft_controls_and_random_sample():
    query = _nonzero_randn((1, 8, 2, 4), seed=5)
    key = _nonzero_randn((1, 8, 2, 4), seed=6)
    value = _nonzero_randn((1, 8, 1, 3), seed=7)
    shape = (1, 2, 8, 7, 4)

    output_a, _ = _inspect(
        query,
        key,
        value,
        max_suffix_length=5,
        route_temperature=0.2,
        mismatch_penalty=0.3,
        mismatch_noise=torch.zeros(shape, dtype=query.dtype),
    )
    output_b, _ = _inspect(
        query,
        key,
        value,
        max_suffix_length=5,
        route_temperature=5.0,
        mismatch_penalty=12.0,
        mismatch_noise=torch.ones(shape, dtype=query.dtype),
    )

    assert torch.equal(output_a, output_b)


def test_allocation_is_plain_route_temperature_softmax_without_winner_margin():
    query = _nonzero_randn((1, 8, 2, 3), seed=8)
    key = _nonzero_randn((1, 8, 2, 3), seed=9)
    value = _nonzero_randn((1, 8, 1, 2), seed=10)
    route_temperature = 1.3

    _, inspection = _inspect(
        query,
        key,
        value,
        route_temperature=route_temperature,
        mismatch_penalty=2.7,
    )

    valid_nonnull = inspection.valid_actions.clone()
    valid_nonnull[:, 0] = False
    valid_nonnull = valid_nonnull.view(1, 1, 8, 8)
    torch.testing.assert_close(
        inspection.route_scores.masked_select(valid_nonnull),
        inspection.proxy_scores.masked_select(valid_nonnull),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        inspection.route_scores[..., 0],
        torch.full_like(inspection.route_scores[..., 0], 0.5),
        rtol=0,
        atol=0,
    )

    centered = inspection.route_scores - inspection.route_scores.amax(
        dim=-1,
        keepdim=True,
    )
    expected = torch.softmax(centered / route_temperature, dim=-1)
    torch.testing.assert_close(inspection.route_probabilities, expected, rtol=0, atol=0)
    invalid = ~inspection.valid_actions.view(1, 1, 8, 8)
    assert torch.equal(
        inspection.route_probabilities.masked_select(invalid),
        torch.zeros_like(inspection.route_probabilities.masked_select(invalid)),
    )
    valid = inspection.valid_actions.view(1, 1, 8, 8)
    assert torch.all(inspection.route_probabilities.masked_select(valid) > 0)


def test_proxy_depends_on_signs_not_qk_magnitude():
    query = _nonzero_randn((1, 7, 2, 4), seed=11)
    key = _nonzero_randn((1, 7, 2, 4), seed=12)
    value = _nonzero_randn((1, 7, 1, 3), seed=13)
    uniform = torch.rand((1, 2, 7, 6, 4), dtype=query.dtype)

    _, base = _inspect(
        query,
        key,
        value,
        mismatch_noise=uniform,
    )
    _, scaled = _inspect(
        query * 0.01,
        key * 100.0,
        value,
        mismatch_noise=uniform,
    )

    torch.testing.assert_close(
        base.proxy_scores,
        scaled.proxy_scores,
        rtol=1e-13,
        atol=1e-15,
    )
    torch.testing.assert_close(
        base.route_probabilities,
        scaled.route_probabilities,
        rtol=1e-13,
        atol=1e-15,
    )


@pytest.mark.parametrize("uniform", [0.0, 0.25, 0.5, 0.9, 1.0])
def test_cubic_mismatch_gate_matches_formula(uniform):
    query = torch.ones(1, 2, 1, 1, dtype=torch.float64)
    key = -torch.ones_like(query)
    value = torch.ones_like(query)
    samples = torch.full((1, 1, 2, 1, 1), uniform, dtype=query.dtype)
    mismatch_penalty = 2.3

    _, inspection = _inspect(
        query,
        key,
        value,
        mismatch_penalty=mismatch_penalty,
        mismatch_noise=samples,
    )

    alpha = 1.0 - 0.5 * uniform**3
    expected = math.exp(-mismatch_penalty * alpha)
    torch.testing.assert_close(
        inspection.proxy_scores[0, 0, 1, 1],
        torch.tensor(expected, dtype=query.dtype),
    )
    assert 0.0 < expected < 1.0


def test_jacobian_anchor_removes_random_local_gradient_scale():
    action_mask = _action_mask(2, torch.device("cpu"))

    def gradients(uniform_value):
        query = torch.tensor([[[[0.7]], [[0.4]]]], dtype=torch.float64, requires_grad=True)
        key = torch.tensor([[[[-0.6]], [[-0.8]]]], dtype=torch.float64, requires_grad=True)
        uniform = torch.full((1, 1, 2, 1, 1), uniform_value, dtype=query.dtype)
        local = _pairwise_proxy_local_match(
            query,
            key,
            action_mask,
            mismatch_penalty=3.0,
            mismatch_noise=uniform,
        )
        grads = torch.autograd.grad(local[0, 0, 1, 1], (query, key))
        return local[0, 0, 1, 1].detach(), grads

    value_a, grads_a = gradients(0.0)
    value_b, grads_b = gradients(0.5)

    assert value_a != value_b
    for grad_a, grad_b in zip(grads_a, grads_b):
        torch.testing.assert_close(grad_a, grad_b, rtol=1e-12, atol=1e-12)


def test_mismatch_penalty_controls_mismatch_leak_but_not_exact_matches():
    query = torch.ones(1, 3, 1, 1, dtype=torch.float64)
    key = -torch.ones_like(query)
    key[:, 1] = 1
    value = torch.ones_like(query)
    uniform = torch.full((1, 1, 3, 2, 1), 0.5, dtype=query.dtype)

    _, low = _inspect(
        query,
        key,
        value,
        max_suffix_length=1,
        mismatch_penalty=0.2,
        mismatch_noise=uniform,
    )
    _, high = _inspect(
        query,
        key,
        value,
        max_suffix_length=1,
        mismatch_penalty=8.0,
        mismatch_noise=uniform,
    )

    assert high.proxy_scores[0, 0, 1, 1] < low.proxy_scores[0, 0, 1, 1]
    torch.testing.assert_close(
        high.proxy_scores[0, 0, 2, 2],
        low.proxy_scores[0, 0, 2, 2],
        rtol=0,
        atol=0,
    )


def test_route_temperature_only_changes_allocation_distribution():
    query = _nonzero_randn((1, 9, 1, 4), seed=14)
    key = _nonzero_randn((1, 9, 1, 4), seed=15)
    value = _nonzero_randn((1, 9, 1, 2), seed=16)
    uniform = torch.rand((1, 1, 9, 8, 4), dtype=query.dtype)

    output_cold, cold = _inspect(
        query,
        key,
        value,
        route_temperature=0.5,
        mismatch_noise=uniform,
    )
    output_hot, hot = _inspect(
        query,
        key,
        value,
        route_temperature=3.0,
        mismatch_noise=uniform,
    )

    assert torch.equal(output_cold, output_hot)
    assert torch.equal(cold.proxy_scores, hot.proxy_scores)
    assert not torch.equal(cold.route_probabilities, hot.route_probabilities)
    cold_entropy = -(cold.route_probabilities * cold.route_probabilities.clamp_min(1e-30).log()).sum(-1)
    hot_entropy = -(hot.route_probabilities * hot.route_probabilities.clamp_min(1e-30).log()).sum(-1)
    assert cold_entropy.mean() < hot_entropy.mean()


def test_backward_is_finite_and_soft_value_credit_reaches_hard_null_rows():
    query = torch.ones(1, 6, 2, 3, dtype=torch.float64, requires_grad=True)
    key = torch.full_like(query, -1.0, requires_grad=True)
    value = _nonzero_randn((1, 6, 1, 4), seed=17).requires_grad_()

    output = rosa_soft_reference(
        query,
        key,
        value,
        max_suffix_length=4,
        route_temperature=1.2,
        mismatch_penalty=3.0,
    )
    assert torch.equal(output, torch.zeros_like(output))
    output.backward(torch.randn_like(output))

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert value.grad[:, 1:].abs().sum() > 0


def test_grouped_value_heads_and_noncontiguous_inputs():
    query_base = _nonzero_randn((2, 4, 7, 3), dtype=torch.float32, seed=18)
    key_base = _nonzero_randn((2, 4, 7, 3), dtype=torch.float32, seed=19)
    value_base = _nonzero_randn((2, 2, 7, 5), dtype=torch.float32, seed=20)
    query = query_base.transpose(1, 2).requires_grad_()
    key = key_base.transpose(1, 2).requires_grad_()
    value = value_base.transpose(1, 2).requires_grad_()

    output = rosa_soft_reference(query, key, value, max_suffix_length=5)

    assert output.shape == (2, 7, 4, 5)
    output.float().square().mean().backward()
    assert all(x.grad is not None for x in (query, key, value))


def test_inference_path_does_not_consume_random_numbers():
    query = _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=21)
    key = _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=22)
    value = _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=23)
    state_before = torch.random.get_rng_state()

    with torch.no_grad():
        rosa_soft_reference(query, key, value)

    assert torch.equal(state_before, torch.random.get_rng_state())


def test_window_is_integral_and_normalized_to_sequence_length():
    query = _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=28)
    key = _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=29)
    value = _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=30)

    output_full, state_full = _inspect(query, key, value, max_suffix_length=5)
    output_large, state_large = _inspect(query, key, value, max_suffix_length=10**9)

    assert torch.equal(output_full, output_large)
    assert torch.equal(state_full.hard_lengths, state_large.hard_lengths)
    assert torch.equal(state_full.proxy_scores, state_large.proxy_scores)
    for invalid in (True, 1.5):
        with pytest.raises(TypeError, match="integer"):
            rosa_soft_reference(query, key, value, max_suffix_length=invalid)


def test_inspection_state_is_detached_and_uses_static_defaults():
    query = _nonzero_randn((2, 5, 2, 3), dtype=torch.float32, seed=24)
    key = _nonzero_randn((2, 5, 2, 3), dtype=torch.float32, seed=25)
    value = _nonzero_randn((2, 5, 1, 2), dtype=torch.float32, seed=26)

    _, inspection = inspect_rosa_soft(
        query,
        key,
        value,
        generator=torch.Generator().manual_seed(27),
    )

    assert inspection.route_temperature == ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE
    assert inspection.mismatch_penalty == ROSA_SOFT_DEFAULT_MISMATCH_PENALTY
    assert inspection.selected_actions.shape == (2, 2, 5)
    assert inspection.mismatch_noise.shape == (2, 2, 5, 4, 3)
    assert all(
        not tensor.requires_grad
        for tensor in (
            inspection.hard_lengths,
            inspection.proxy_scores,
            inspection.route_scores,
            inspection.route_probabilities,
            inspection.mismatch_noise,
        )
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_suffix_length": 0}, "max_suffix_length"),
        ({"route_temperature": 0.0}, "route_temperature"),
        ({"route_temperature": math.inf}, "route_temperature"),
        ({"mismatch_penalty": 0.0}, "mismatch_penalty"),
        ({"mismatch_penalty": math.nan}, "mismatch_penalty"),
    ],
)
def test_static_parameter_validation(kwargs, message):
    query = torch.ones(1, 2, 1, 1)
    with pytest.raises(ValueError, match=message):
        rosa_soft_reference(query, query, query, **kwargs)


@pytest.mark.parametrize(
    "removed",
    [
        "mismatch_relaxation",
        "value_backward",
        "return_state",
        "generator",
        "return_telemetry",
        "logit_epsilon",
        "qk_damper_strength",
    ],
)
def test_removed_training_controls_are_not_public(removed):
    query = torch.ones(1, 2, 1, 1)
    with pytest.raises(TypeError):
        rosa_soft_reference(query, query, query, **{removed: 1})


def test_shape_head_bit_dtype_and_uniform_validation():
    query = torch.ones(1, 3, 2, 4)
    key = torch.ones_like(query)
    value = torch.ones(1, 3, 1, 2)

    with pytest.raises(ValueError, match="sequence"):
        rosa_soft_reference(query, key[:, :-1], value)
    with pytest.raises(ValueError, match="divisible"):
        rosa_soft_reference(query[:, :, :1], key[:, :, :1], value.repeat(1, 1, 2, 1))
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        rosa_soft_reference(
            torch.ones(1, 3, 1, 33),
            torch.ones(1, 3, 1, 33),
            value,
        )
    with pytest.raises(ValueError, match="same dtype"):
        rosa_soft_reference(query.double(), key, value)
    with pytest.raises(ValueError, match="wrong shape"):
        inspect_rosa_soft(
            query,
            key,
            value,
            mismatch_noise=torch.zeros(1),
        )
