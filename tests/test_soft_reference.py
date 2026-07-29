import math

import pytest
import torch

from rosa_soft import rosa_soft_reference
from rosa_soft.soft_reference import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
    _apply_attention_dropout,
    _build_vjp_carrier,
    _causal_route_mask,
    _hard_sign_with_softsign_vjp,
    _pairwise_soft_match_gates,
    _suffix_prefix_product_scores,
)
from rosa_soft.testing import inspect_rosa_soft


def _nonzero_randn(shape, *, dtype=torch.float64, seed=0):
    generator = torch.Generator().manual_seed(seed)
    values = torch.randn(shape, dtype=dtype, generator=generator)
    return values.sign().masked_fill(values == 0, 1) * (values.abs() + 0.2)


def test_public_static_defaults_are_frozen():
    assert ROSA_SOFT_DEFAULT_SCALE == 1.0
    assert ROSA_SOFT_DEFAULT_DROPOUT_P == 0.0
    assert ROSA_SOFT_DEFAULT_MISMATCH_SCALE == 3.0


def _naive_exact_suffix_lengths(query, key, max_suffix_length):
    batch, seq_len, heads, _ = query.shape
    q_bits = query > 0
    k_bits = key > 0
    lengths = torch.zeros(batch, heads, seq_len, seq_len, dtype=query.dtype)
    for b in range(batch):
        for h in range(heads):
            for query_pos in range(seq_len):
                for route in range(1, query_pos + 1):
                    limit = min(max_suffix_length, query_pos + 1, route)
                    for offset in range(limit):
                        if not torch.equal(
                            q_bits[b, query_pos - offset, h],
                            k_bits[b, route - 1 - offset, h],
                        ):
                            break
                        lengths[b, h, query_pos, route] += 1
    return lengths


@pytest.mark.parametrize("gradient_mask", range(1, 8))
def test_reference_gradient_masks_match_full_vjp(gradient_mask):
    base = (
        _nonzero_randn((1, 7, 2, 3), seed=100),
        _nonzero_randn((1, 7, 2, 3), seed=101),
        _nonzero_randn((1, 7, 1, 2), seed=102),
    )
    grad_output = _nonzero_randn((1, 7, 2, 2), seed=103)
    full_inputs = tuple(value.clone().requires_grad_() for value in base)
    full_output = rosa_soft_reference(*full_inputs, max_suffix_length=5)
    full_gradients = torch.autograd.grad(
        full_output,
        full_inputs,
        grad_output,
    )

    masked_inputs = tuple(
        value.clone().requires_grad_(bool(gradient_mask & bit))
        for bit, value in zip((1, 2, 4), base)
    )
    masked_output = rosa_soft_reference(*masked_inputs, max_suffix_length=5)
    active = tuple(value for value in masked_inputs if value.requires_grad)
    masked_gradients = torch.autograd.grad(
        masked_output,
        active,
        grad_output,
    )

    assert torch.equal(masked_output, full_output)
    observed = iter(masked_gradients)
    for bit, expected in zip((1, 2, 4), full_gradients):
        if gradient_mask & bit:
            torch.testing.assert_close(next(observed), expected)


def test_softsign_vjp_is_the_declared_symbol_jacobian():
    logits = torch.tensor(
        [-math.inf, -1.0, 0.0, 1.0, math.inf],
        dtype=torch.float64,
        requires_grad=True,
    )
    symbols = _hard_sign_with_softsign_vjp(logits)
    gradient = torch.autograd.grad(symbols.sum(), logits)[0]

    assert torch.equal(
        symbols,
        torch.tensor([-1.0, -1.0, -1.0, 1.0, 1.0]),
    )
    torch.testing.assert_close(
        gradient,
        torch.tensor(
            [0.0, 0.25, 1.0, 0.25, 0.0],
            dtype=torch.float64,
        ),
    )


@pytest.mark.parametrize("max_suffix_length", [1, 2, 5, 64])
def test_exact_suffix_dynamic_program_matches_naive(max_suffix_length):
    query = _nonzero_randn((2, 6, 2, 3), seed=11)
    key = _nonzero_randn((2, 6, 2, 3), seed=12)
    value = _nonzero_randn((2, 6, 1, 2), seed=13)
    _, inspection = inspect_rosa_soft(
        query,
        key,
        value,
        max_suffix_length=max_suffix_length,
    )

    torch.testing.assert_close(
        inspection.exact_suffix_lengths,
        _naive_exact_suffix_lengths(
            query,
            key,
            min(max_suffix_length, query.size(1)),
        ),
        rtol=0,
        atol=0,
    )


def test_forward_is_exact_latest_longest_and_null():
    query = torch.ones(1, 5, 1, 2)
    key = torch.ones_like(query)
    value = torch.tensor(
        [[[[1.0]], [[-1.0]], [[1.0]], [[-1.0]], [[1.0]]]]
    )
    output, inspection = inspect_rosa_soft(
        query,
        key,
        value,
        max_suffix_length=1,
    )

    assert torch.equal(
        inspection.selected_route_indices,
        torch.tensor([[[0, 1, 2, 3, 4]]]),
    )
    assert torch.equal(
        output.flatten(),
        torch.tensor([0.0, -1.0, 1.0, -1.0, 1.0]),
    )

    null_output, null_state = inspect_rosa_soft(
        query,
        -key,
        value,
    )
    assert torch.equal(null_output, torch.zeros_like(null_output))
    assert torch.equal(
        null_state.selected_route_indices,
        torch.zeros_like(null_state.selected_route_indices),
    )


def test_hard_forward_is_independent_of_soft_controls():
    query = _nonzero_randn((1, 8, 1, 4), seed=20)
    key = _nonzero_randn((1, 8, 1, 4), seed=21)
    value = _nonzero_randn((1, 8, 1, 3), seed=22)
    cold = rosa_soft_reference(
        query,
        key,
        value,
        scale=0.1,
        mismatch_scale=0.2,
    )
    hot = rosa_soft_reference(
        query,
        key,
        value,
        scale=10.0,
        mismatch_scale=100.0,
    )
    assert torch.equal(cold, hot)


def test_local_match_gate_value_and_coherent_vjp_match_formula():
    query = torch.tensor([[[[2.0, -3.0]]]], requires_grad=True)
    key = torch.tensor([[[[1.0, 4.0]]]], requires_grad=True)
    mask = _causal_route_mask(2, query.device)
    query = torch.cat((query, query), dim=1)
    key = torch.cat((key, key), dim=1)
    mismatch_scale = 3.0
    gates = _pairwise_soft_match_gates(query, key, mask, mismatch_scale)
    gate = gates[0, 0, 1, 1]
    expected = math.exp(-mismatch_scale / 2)

    assert float(gate.detach()) == pytest.approx(expected)
    grad_query, grad_key = torch.autograd.grad(gate, (query, key))
    assert grad_query[0, 1, 0, 0] == pytest.approx(
        expected * mismatch_scale * 0.5 * 1.0 / 2 / 9
    )
    assert grad_query[0, 1, 0, 1] == pytest.approx(
        expected * mismatch_scale * 0.5 * 1.0 / 2 / 16
    )
    assert grad_key[0, 0, 0, 0] == pytest.approx(
        expected * mismatch_scale * 0.5 * 1.0 / 2 / 4
    )
    assert grad_key[0, 0, 0, 1] == pytest.approx(
        expected * mismatch_scale * 0.5 * -1.0 / 2 / 25
    )


def test_suffix_score_is_sum_of_prefix_products():
    local = torch.zeros(1, 1, 4, 4, dtype=torch.float64)
    local[0, 0, 3, 3] = 0.8
    local[0, 0, 2, 2] = 0.5
    local[0, 0, 1, 1] = 0.25
    scores = _suffix_prefix_product_scores(local, 3)
    assert scores[0, 0, 3, 3] == pytest.approx(
        0.8 + 0.8 * 0.5 + 0.8 * 0.5 * 0.25
    )


def test_candidate_count_correction_keeps_flat_null_mass_invariant():
    query = torch.ones(1, 9, 1, 4)
    key = -torch.ones_like(query)
    value = torch.ones(1, 9, 1, 1)
    _, inspection = inspect_rosa_soft(
        query,
        key,
        value,
        max_suffix_length=1,
        scale=1.3,
        mismatch_scale=2.0,
    )
    null = inspection.route_probabilities[0, 0, 1:, 0]
    torch.testing.assert_close(
        null,
        null[0].expand_as(null),
        rtol=1e-6,
        atol=1e-7,
    )
    valid = inspection.causal_route_mask.view(1, 1, 9, 9)
    assert torch.all(
        inspection.route_probabilities.masked_select(valid) > 0
    )


def test_scale_changes_allocation_not_scores():
    values = (
        _nonzero_randn((1, 7, 1, 4), seed=30),
        _nonzero_randn((1, 7, 1, 4), seed=31),
        _nonzero_randn((1, 7, 1, 2), seed=32),
    )
    _, low_scale = inspect_rosa_soft(
        *values,
        scale=0.25,
        mismatch_scale=3.0,
    )
    _, high_scale = inspect_rosa_soft(
        *values,
        scale=4.0,
        mismatch_scale=3.0,
    )
    assert torch.equal(
        low_scale.soft_suffix_scores,
        high_scale.soft_suffix_scores,
    )
    assert not torch.equal(
        low_scale.route_probabilities,
        high_scale.route_probabilities,
    )


def test_mismatch_scale_changes_fuzzy_scores_not_exact_scores():
    query = torch.ones(1, 3, 1, 4)
    key = query.clone()
    key[:, 0, :, 0] = -1
    value = torch.ones(1, 3, 1, 1)
    _, low = inspect_rosa_soft(
        query,
        key,
        value,
        mismatch_scale=1.0,
    )
    _, high = inspect_rosa_soft(
        query,
        key,
        value,
        mismatch_scale=9.0,
    )
    mismatch = low.exact_suffix_lengths == 0
    assert torch.all(
        high.soft_suffix_scores.masked_select(mismatch)
        <= low.soft_suffix_scores.masked_select(mismatch)
    )
    mask = _causal_route_mask(query.size(1), query.device)
    low_local = _pairwise_soft_match_gates(query, key, mask, 1.0)
    high_local = _pairwise_soft_match_gates(query, key, mask, 9.0)
    query_symbols = query.permute(0, 2, 1, 3)
    key_symbols = key.permute(0, 2, 1, 3)[..., :-1, :]
    exact_local = (
        query_symbols.unsqueeze(-2) == key_symbols.unsqueeze(-3)
    ).all(-1)
    exact_local = torch.nn.functional.pad(exact_local, (1, 0))
    torch.testing.assert_close(
        high_local.masked_select(exact_local),
        low_local.masked_select(exact_local),
    )


def test_raw_expected_prefix_keeps_long_near_match_visible():
    width = 8
    window = 16
    length = 2 * window + 4
    query_pos = length - 1
    target_route = window + 1
    decoy_route = 2 * window + 2
    generator = torch.Generator().manual_seed(41)
    query = torch.where(
        torch.rand(1, length, 1, width, generator=generator) > 0.5,
        0.1,
        -0.1,
    )
    key = (-query[:, query_pos]).repeat(1, length, 1, 1)
    for offset in range(window):
        key[:, target_route - 1 - offset] = query[:, query_pos - offset]
        key[:, decoy_route - 1 - offset] = query[:, query_pos - offset]
    key[:, decoy_route - window, :, 0] *= -1
    key[:, target_route - 1, :, 0] *= -1
    value = -torch.ones(1, length, 1, 1)
    value[:, target_route] = 1

    key = key.requires_grad_()
    output, inspection = inspect_rosa_soft(
        query,
        key,
        value,
        max_suffix_length=window,
        mismatch_scale=1.0,
    )
    assert inspection.selected_route_indices[0, 0, -1] == decoy_route
    assert inspection.route_probabilities[0, 0, -1, target_route] > 0.05

    train_output = rosa_soft_reference(
        query,
        key,
        value,
        max_suffix_length=window,
        mismatch_scale=1.0,
    )
    gradient = torch.autograd.grad(-train_output[0, -1, 0, 0], key)[0]
    assert gradient[0, target_route - 1, 0, 0].abs() > 0.05
    assert output[0, -1, 0, 0] == -1


def test_value_carrier_distributes_credit_to_all_causal_routes():
    query = torch.ones(1, 6, 2, 3, dtype=torch.float64, requires_grad=True)
    key = torch.full_like(query, -1.0, requires_grad=True)
    value = _nonzero_randn((1, 6, 1, 4), seed=50).requires_grad_()
    output = rosa_soft_reference(
        query,
        key,
        value,
        max_suffix_length=4,
        mismatch_scale=3.0,
    )
    assert torch.equal(output, torch.zeros_like(output))
    output.backward(torch.ones_like(output))

    for tensor in (query, key, value):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    assert torch.all(value.grad[:, 1:].abs().sum(dim=(0, 2, 3)) > 0)


def test_carrier_value_vjp_matches_closed_form():
    value = torch.tensor(
        [[[[0.5]], [[-1.0]], [[2.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    probabilities = torch.tensor(
        [[[[1.0, 0.0, 0.0], [0.2, 0.8, 0.0], [0.1, 0.3, 0.6]]]],
        dtype=torch.float64,
    )
    carrier = _build_vjp_carrier(value, probabilities, 1)
    gradient = torch.autograd.grad(carrier.sum(), value)[0].flatten()
    expected = torch.tensor(
        [
            0.0,
            (0.8 + 0.3) / (1 + 1.0) ** 2,
            0.6 / (1 + 2.0) ** 2,
        ],
        dtype=torch.float64,
    )
    torch.testing.assert_close(gradient, expected)


def test_singleton_backward_returns_connected_zero_gradients():
    inputs = tuple(
        torch.ones(1, 1, 1, 1, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    )
    output = rosa_soft_reference(*inputs)
    gradients = torch.autograd.grad(output.sum(), inputs)
    assert torch.equal(output, torch.zeros_like(output))
    assert all(torch.equal(gradient, torch.zeros_like(gradient)) for gradient in gradients)


def test_grouped_value_heads_and_noncontiguous_inputs():
    query = _nonzero_randn(
        (2, 4, 7, 3), dtype=torch.float32, seed=60
    ).transpose(1, 2).requires_grad_()
    key = _nonzero_randn(
        (2, 4, 7, 3), dtype=torch.float32, seed=61
    ).transpose(1, 2).requires_grad_()
    value = _nonzero_randn(
        (2, 2, 7, 5), dtype=torch.float32, seed=62
    ).transpose(1, 2).requires_grad_()
    output = rosa_soft_reference(
        query,
        key,
        value,
        max_suffix_length=5,
    )
    assert output.shape == (2, 7, 4, 5)
    output.square().mean().backward()
    assert all(value.grad is not None for value in (query, key, value))


def test_reference_is_deterministic_and_does_not_consume_rng():
    inputs = (
        _nonzero_randn((1, 6, 1, 3), dtype=torch.float32, seed=70),
        _nonzero_randn((1, 6, 1, 3), dtype=torch.float32, seed=71),
        _nonzero_randn((1, 6, 1, 2), dtype=torch.float32, seed=72),
    )
    state = torch.random.get_rng_state()

    def run():
        leaves = tuple(value.clone().requires_grad_() for value in inputs)
        output = rosa_soft_reference(*leaves, max_suffix_length=4)
        return output, torch.autograd.grad(output.sum(), leaves)

    first = run()
    torch.randn(100)
    second = run()
    assert torch.equal(first[0], second[0])
    for left, right in zip(first[1], second[1]):
        assert torch.equal(left, right)
    torch.random.set_rng_state(state)
    with torch.no_grad():
        rosa_soft_reference(*inputs)
    assert torch.equal(state, torch.random.get_rng_state())


def test_attention_dropout_is_seeded_backward_only():
    base = (
        _nonzero_randn((1, 6, 2, 3), dtype=torch.float32, seed=73),
        _nonzero_randn((1, 6, 2, 3), dtype=torch.float32, seed=74),
        _nonzero_randn((1, 6, 1, 2), dtype=torch.float32, seed=75),
    )
    grad_output = _nonzero_randn(
        (1, 6, 2, 2),
        dtype=torch.float32,
        seed=76,
    )

    def run(seed):
        leaves = tuple(tensor.clone().requires_grad_() for tensor in base)
        torch.manual_seed(seed)
        output = rosa_soft_reference(
            *leaves,
            max_suffix_length=4,
            dropout_p=0.5,
        )
        gradients = torch.autograd.grad(output, leaves, grad_output)
        return output, gradients

    first = run(77)
    replay = run(77)
    changed = run(78)
    assert torch.equal(first[0], replay[0])
    assert torch.equal(first[0], changed[0])
    for expected, actual in zip(first[1], replay[1]):
        assert torch.equal(expected, actual)
    assert any(
        not torch.equal(expected, actual)
        for expected, actual in zip(first[1], changed[1])
    )


def test_attention_dropout_hashes_routes_and_full_seed():
    weights = torch.ones(1, 1, 1, 4096)

    def mask(seed):
        dropped = _apply_attention_dropout(
            weights,
            0.5,
            torch.tensor(seed, dtype=torch.int64),
            0,
        )
        return dropped.ne(0).flatten()

    baseline = mask(7)
    high_seed_bits = mask((1 << 32) + 7)
    kept = int(baseline.sum())
    transitions = int((baseline[1:] != baseline[:-1]).sum())

    assert 1800 <= kept <= 2300
    assert transitions >= 1500
    assert not torch.equal(baseline, high_seed_bits)


def test_attention_dropout_no_grad_does_not_consume_rng():
    inputs = (
        _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=79),
        _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=80),
        _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=81),
    )
    state = torch.random.get_rng_state()
    with torch.no_grad():
        dropout_output = rosa_soft_reference(*inputs, dropout_p=0.5)
    assert torch.equal(state, torch.random.get_rng_state())
    deterministic_output = rosa_soft_reference(*inputs, dropout_p=0.0)
    assert torch.equal(dropout_output, deterministic_output)


def test_window_is_integral_and_clamped_to_sequence_length():
    values = (
        _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=80),
        _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=81),
        _nonzero_randn((1, 5, 1, 2), dtype=torch.float32, seed=82),
    )
    full_output, full = inspect_rosa_soft(*values, max_suffix_length=5)
    large_output, large = inspect_rosa_soft(
        *values,
        max_suffix_length=10**9,
    )
    assert torch.equal(full_output, large_output)
    assert torch.equal(full.exact_suffix_lengths, large.exact_suffix_lengths)
    assert torch.equal(full.soft_suffix_scores, large.soft_suffix_scores)
    assert full.effective_max_suffix_length == 5
    assert large.effective_max_suffix_length == 5
    for invalid in (True, 1.5):
        with pytest.raises(TypeError, match="integer"):
            rosa_soft_reference(*values, max_suffix_length=invalid)


def test_inspection_is_detached_and_uses_defaults():
    values = (
        _nonzero_randn((2, 5, 2, 3), dtype=torch.float32, seed=90),
        _nonzero_randn((2, 5, 2, 3), dtype=torch.float32, seed=91),
        _nonzero_randn((2, 5, 1, 2), dtype=torch.float32, seed=92),
    )
    _, inspection = inspect_rosa_soft(*values)
    assert inspection.scale == ROSA_SOFT_DEFAULT_SCALE
    assert inspection.dropout_p == ROSA_SOFT_DEFAULT_DROPOUT_P
    assert inspection.mismatch_scale == ROSA_SOFT_DEFAULT_MISMATCH_SCALE
    assert inspection.effective_max_suffix_length == 5
    assert inspection.selected_route_indices.shape == (2, 2, 5)
    assert all(
        not tensor.requires_grad
        for tensor in (
            inspection.exact_suffix_lengths,
            inspection.soft_suffix_scores,
            inspection.route_scores,
            inspection.route_probabilities,
        )
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_suffix_length": 0}, "max_suffix_length"),
        ({"scale": True}, "scale"),
        ({"scale": 0.0}, "scale"),
        ({"scale": math.inf}, "scale"),
        ({"scale": math.nan}, "scale"),
        ({"scale": 1e-300}, "scale"),
        ({"scale": 1e300}, "scale"),
        ({"dropout_p": True}, "dropout_p"),
        ({"dropout_p": -0.1}, "dropout_p"),
        ({"dropout_p": 1.0}, "dropout_p"),
        ({"dropout_p": math.nan}, "dropout_p"),
        ({"dropout_p": math.inf}, "dropout_p"),
        ({"mismatch_scale": 0.0}, "mismatch_scale"),
        ({"mismatch_scale": True}, "mismatch_scale"),
        ({"mismatch_scale": math.nan}, "mismatch_scale"),
        ({"mismatch_scale": math.inf}, "mismatch_scale"),
        ({"mismatch_scale": 1e-300}, "mismatch_scale"),
        ({"mismatch_scale": 1e300}, "mismatch_scale"),
    ],
)
def test_static_parameter_validation(kwargs, message):
    tensor = torch.ones(1, 5, 1, 1)
    with pytest.raises(ValueError, match=message):
        rosa_soft_reference(tensor, tensor, tensor, **kwargs)


def test_largest_supported_dropout_probability_is_accepted():
    tensor = torch.ones(1, 3, 1, 1, requires_grad=True)
    output = rosa_soft_reference(
        tensor,
        tensor,
        tensor,
        dropout_p=1.0 - 2.0**-24,
    )
    assert torch.isfinite(
        torch.autograd.grad(output.sum(), tensor)[0]
    ).all()


def test_shape_head_bit_and_dtype_validation():
    query = torch.ones(1, 3, 2, 4)
    key = torch.ones_like(query)
    value = torch.ones(1, 3, 1, 2)
    with pytest.raises(ValueError, match="sequence"):
        rosa_soft_reference(query, key[:, :-1], value)
    with pytest.raises(ValueError, match="divisible"):
        rosa_soft_reference(
            query[:, :, :1],
            key[:, :, :1],
            value.repeat(1, 1, 2, 1),
        )
    with pytest.raises(ValueError, match=r"\[1, 32\]"):
        rosa_soft_reference(
            torch.ones(1, 3, 1, 33),
            torch.ones(1, 3, 1, 33),
            value,
        )
    with pytest.raises(ValueError, match="same dtype"):
        rosa_soft_reference(query.double(), key, value)


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_reference_accepts_supported_dtypes(dtype):
    tensor = torch.ones(1, 3, 1, 1, dtype=dtype)
    assert rosa_soft_reference(tensor, tensor, tensor).dtype == dtype


@pytest.mark.skipif(
    not hasattr(torch, "float8_e4m3fn"),
    reason="PyTorch has no float8 dtype",
)
def test_reference_rejects_float8():
    tensor = torch.ones(1, 3, 1, 1, dtype=torch.float8_e4m3fn)
    with pytest.raises(ValueError, match="float16.*float64"):
        rosa_soft_reference(tensor, tensor, tensor)
