import itertools
import math

import pytest
import torch

from benchmarks import dual_score_reference as reference
from rosa_soft.soft_reference import (
    _hard_route_forward,
    rosa_soft_reference,
)


def _inputs(
    *,
    seed: int = 17,
    seq_len: int = 6,
    heads: int = 2,
    bits: int = 4,
    value_heads: int = 1,
    value_bits: int = 3,
    dtype: torch.dtype = torch.float64,
):
    generator = torch.Generator().manual_seed(seed)
    return tuple(
        tensor.requires_grad_(True)
        for tensor in (
            torch.randn(
                1, seq_len, heads, bits, generator=generator, dtype=dtype
            ),
            torch.randn(
                1, seq_len, heads, bits, generator=generator, dtype=dtype
            ),
            torch.randn(
                1,
                seq_len,
                value_heads,
                value_bits,
                generator=generator,
                dtype=dtype,
            ),
        )
    )


def _scalar_suffix_scores(local_match: torch.Tensor) -> torch.Tensor:
    expected = torch.zeros_like(local_match)
    seq_len = local_match.size(-1)
    for row in range(1, seq_len):
        for route in range(1, row + 1):
            product = local_match.new_ones(local_match.shape[:-2])
            score = local_match.new_zeros(local_match.shape[:-2])
            for offset in range(route):
                product = product * local_match[..., row - offset, route - offset]
                score = score + product
            expected[..., row, route] = score
    return expected


def _scalar_information_scores(
    local_log_gate: torch.Tensor,
    *,
    log_z: float,
    beta: float,
) -> torch.Tensor:
    expected = torch.full_like(local_log_gate, -torch.inf)
    seq_len = local_log_gate.size(-1)
    for row in range(1, seq_len):
        for route in range(1, row + 1):
            terms = []
            running = local_log_gate.new_zeros(local_log_gate.shape[:-2])
            for offset in range(route):
                running = running + local_log_gate[
                    ..., row - offset, route - offset
                ]
                terms.append(running - beta * log_z * (offset + 1))
            numerator = torch.logsumexp(torch.stack(terms), dim=0)
            lengths = torch.arange(
                1,
                route + 1,
                dtype=local_log_gate.dtype,
                device=local_log_gate.device,
            )
            normalizer = torch.logsumexp((1.0 - beta) * log_z * lengths, dim=0)
            expected[..., row, route] = numerator - normalizer
    return expected


def test_unbounded_suffix_recurrence_matches_scalar_oracle():
    generator = torch.Generator().manual_seed(101)
    local = torch.rand(2, 3, 7, 7, generator=generator, dtype=torch.float64)
    mask = torch.tril(torch.ones(7, 7, dtype=torch.bool), diagonal=-1)
    local = local * mask.roll(shifts=1, dims=1).view(1, 1, 7, 7)

    actual = reference._unbounded_suffix_scores(local)
    expected = _scalar_suffix_scores(local)

    torch.testing.assert_close(actual, expected, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("beta", [0.25, 0.5, 1.0])
def test_log_information_recurrence_matches_scalar_oracle(beta):
    generator = torch.Generator().manual_seed(103)
    mismatch = torch.randint(0, 5, (2, 1, 7, 7), generator=generator)
    local_log_gate = -1.7 * mismatch.to(torch.float64)
    log_z = reference.null_log_gate_mean(4, 1.7)

    actual = reference._unbounded_information_log_evidence(
        local_log_gate,
        log_null_gate_mean=log_z,
        evidence_power=beta,
    )
    expected = _scalar_information_scores(
        local_log_gate,
        log_z=log_z,
        beta=beta,
    )

    finite = torch.isfinite(expected)
    assert torch.equal(torch.isfinite(actual), finite)
    torch.testing.assert_close(
        actual[finite], expected[finite], rtol=2e-14, atol=2e-14
    )


@pytest.mark.parametrize(
    ("bits", "length"),
    [(1, 1), (1, 4), (2, 1), (2, 4), (4, 1), (4, 2), (8, 1)],
)
def test_information_weight_has_exact_unit_random_background_mean(bits, length):
    mismatch_patterns = itertools.product((0, 1), repeat=bits * length)
    counts = torch.tensor(
        [
            [
                sum(pattern[offset : offset + bits])
                for offset in range(0, bits * length, bits)
            ]
            for pattern in mismatch_patterns
        ],
        dtype=torch.float64,
    )
    local_log_gates = -3.0 * counts
    scores = reference.tempered_diagonal_log_evidence(
        local_log_gates,
        symbol_dim=bits,
        mismatch_scale=3.0,
        evidence_power=0.5,
    )

    assert scores.exp().mean().item() == pytest.approx(
        1.0, rel=3e-13, abs=3e-13
    )


def test_exact_information_evidence_is_finite_and_grows_with_bits_and_length():
    lengths = (1, 8, 64, 2048)
    scores = {}
    for bits in (1, 4, 8, 32):
        rows = []
        for length in lengths:
            rows.append(
                reference.tempered_diagonal_log_evidence(
                    torch.zeros(length, dtype=torch.float64),
                    symbol_dim=bits,
                )
            )
        scores[bits] = torch.stack(rows)
        assert bool(torch.isfinite(scores[bits]).all())
        assert bool((scores[bits][1:] > scores[bits][:-1]).all())
    for left, right in zip((1, 4, 8), (4, 8, 32)):
        assert bool((scores[right] > scores[left]).all())


def test_discovery_mode_matches_full_horizon_production_reference():
    query, key, value = _inputs(seq_len=7)
    upstream = torch.randn_like(value.expand(-1, -1, 2, -1))

    production = rosa_soft_reference(
        query,
        key,
        value,
        max_suffix_length=query.size(1),
        scale=1.3,
        mismatch_scale=2.1,
    )
    expected = torch.autograd.grad(
        (production * upstream).sum(), (query, key, value), retain_graph=True
    )
    candidate = reference.rosa_dual_score_reference(
        query,
        key,
        value,
        score_mode="discovery",
        scale=1.3,
        mismatch_scale=2.1,
    )
    actual = torch.autograd.grad(
        (candidate * upstream).sum(), (query, key, value)
    )

    torch.testing.assert_close(candidate, production, rtol=0.0, atol=0.0)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=2e-13, atol=2e-13)


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_all_modes_share_exact_unlimited_hard_forward(score_mode):
    query, key, value = _inputs(seq_len=8)
    expected, _, _, _ = _hard_route_forward(query, key, value)
    actual = reference.rosa_dual_score_reference(
        query,
        key,
        value,
        score_mode=score_mode,
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_hard_forward_is_not_truncated_at_32(score_mode):
    sequence_length = 70
    query = torch.ones(1, sequence_length, 1, 1, dtype=torch.float32)
    key = torch.ones_like(query)
    value = torch.where(
        torch.arange(sequence_length).remainder(2).view(1, -1, 1, 1) == 0,
        1.0,
        -1.0,
    )
    expected, lengths, routes, _ = reference._hard_route_forward_unbounded(
        query, key, value
    )

    assert lengths[0, 0, -1, -1].item() == sequence_length - 1
    assert routes[0, 0, -1].item() == sequence_length - 1
    actual = reference.rosa_dual_score_reference(
        query, key, value, score_mode=score_mode
    )
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_scores_do_not_leak_sign_preserving_amplitude(score_mode):
    query, key, value = _inputs(seq_len=7)
    query_scaled = query.detach() * torch.rand_like(query).add(0.01)
    key_scaled = key.detach() * torch.rand_like(key).add(0.01)
    value_scaled = value.detach() * torch.rand_like(value).add(0.01)

    original = reference.rosa_dual_score_reference(
        query, key, value, score_mode=score_mode
    )
    scaled = reference.rosa_dual_score_reference(
        query_scaled, key_scaled, value_scaled, score_mode=score_mode
    )
    torch.testing.assert_close(original, scaled, rtol=0.0, atol=0.0)


def test_dual_distribution_is_an_exact_convex_probability_mixture():
    query, key, _ = _inputs(seq_len=8)
    state = reference.build_dual_score_state(
        query,
        key,
        information_weight=0.37,
    )
    expected = (
        0.63 * state.discovery_probabilities
        + 0.37 * state.information_probabilities
    )

    torch.testing.assert_close(state.route_probabilities, expected)
    torch.testing.assert_close(
        state.route_probabilities.sum(dim=-1),
        torch.ones_like(state.route_probabilities[..., 0]),
    )
    valid = state.causal_route_mask.view(1, 1, 8, 8)
    assert bool((state.route_probabilities[valid.expand_as(expected)] > 0).all())
    assert bool((state.route_probabilities[~valid.expand_as(expected)] == 0).all())


def test_scale_changes_discovery_but_not_calibrated_information_units():
    query, key, _ = _inputs(seq_len=7)
    low = reference.build_dual_score_state(query, key, scale=0.5)
    high = reference.build_dual_score_state(query, key, scale=2.0)

    assert not torch.equal(
        low.discovery_probabilities, high.discovery_probabilities
    )
    assert torch.equal(
        low.information_probabilities, high.information_probabilities
    )


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_final_query_specialization_matches_full_dense_last_row(score_mode):
    generator = torch.Generator().manual_seed(181)
    sequence_length = 7
    query = torch.randn(
        2, sequence_length, 1, 4, generator=generator, dtype=torch.float64
    )
    key = torch.randn_like(query)
    value = torch.randn(
        2, sequence_length, 1, 3, generator=generator, dtype=torch.float64
    )
    dense_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in (query, key, value)
    )
    dense_carrier = reference.dual_score_carrier(
        *dense_inputs,
        score_mode=score_mode,
        scale=1.3,
        mismatch_scale=2.2,
        evidence_power=0.5,
        information_weight=0.4,
    )[:, -1, 0]

    final_inputs = (
        query[:, 1:, 0].detach().requires_grad_(True),
        key[:, :-1, 0].detach().requires_grad_(True),
        value[:, 1:, 0].detach().requires_grad_(True),
    )
    final_carrier, final_probabilities = reference.final_query_score_carrier(
        *final_inputs,
        score_mode=score_mode,
        scale=1.3,
        mismatch_scale=2.2,
        evidence_power=0.5,
        information_weight=0.4,
    )

    torch.testing.assert_close(final_carrier, dense_carrier, rtol=2e-14, atol=2e-14)
    dense_state = reference.build_dual_score_state(
        dense_inputs[0],
        dense_inputs[1],
        scale=1.3,
        mismatch_scale=2.2,
        evidence_power=0.5,
        information_weight=0.4,
    )
    dense_probabilities = reference._selected_probabilities(
        dense_state, score_mode
    )[:, 0, -1, 1:]
    torch.testing.assert_close(
        final_probabilities, dense_probabilities, rtol=2e-14, atol=2e-14
    )

    upstream = torch.randn_like(final_carrier)
    dense_gradients = torch.autograd.grad(
        (dense_carrier * upstream).sum(), dense_inputs
    )
    final_gradients = torch.autograd.grad(
        (final_carrier * upstream).sum(), final_inputs
    )
    torch.testing.assert_close(
        final_gradients[0], dense_gradients[0][:, 1:, 0], rtol=2e-13, atol=2e-13
    )
    torch.testing.assert_close(
        final_gradients[1], dense_gradients[1][:, :-1, 0], rtol=2e-13, atol=2e-13
    )
    torch.testing.assert_close(
        final_gradients[2], dense_gradients[2][:, 1:, 0], rtol=2e-13, atol=2e-13
    )


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_custom_backward_matches_materialized_carrier(score_mode):
    query, key, value = _inputs(seq_len=6)
    upstream = torch.randn_like(value.expand(-1, -1, 2, -1))
    output = reference.rosa_dual_score_reference(
        query,
        key,
        value,
        score_mode=score_mode,
        scale=1.2,
        mismatch_scale=1.9,
        evidence_power=0.5,
        information_weight=0.4,
    )
    actual = torch.autograd.grad(
        (output * upstream).sum(), (query, key, value)
    )

    direct_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in (query, key, value)
    )
    carrier = reference.dual_score_carrier(
        *direct_inputs,
        score_mode=score_mode,
        scale=1.2,
        mismatch_scale=1.9,
        evidence_power=0.5,
        information_weight=0.4,
    )
    expected = torch.autograd.grad(
        (carrier * upstream).sum(), direct_inputs
    )
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_attention_dropout_is_seeded_backward_only(score_mode):
    base = tuple(tensor.detach() for tensor in _inputs(seed=203, seq_len=7))
    upstream = torch.randn(
        1, 7, 2, 3, generator=torch.Generator().manual_seed(204), dtype=torch.float64
    )

    def run(seed):
        leaves = tuple(tensor.clone().requires_grad_() for tensor in base)
        torch.manual_seed(seed)
        output = reference.rosa_dual_score_reference(
            *leaves,
            score_mode=score_mode,
            dropout_p=0.5,
        )
        return output, torch.autograd.grad(output, leaves, upstream)

    first = run(205)
    replay = run(205)
    changed = run(206)
    assert torch.equal(first[0], replay[0])
    assert torch.equal(first[0], changed[0])
    for expected, actual in zip(first[1], replay[1]):
        assert torch.equal(expected, actual)
    assert any(
        not torch.equal(expected, actual)
        for expected, actual in zip(first[1], changed[1])
    )


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_dropout_custom_backward_matches_fixed_seed_carrier(score_mode):
    inputs = _inputs(seed=206, seq_len=6)
    seed = torch.tensor(0x123456789, dtype=torch.int64)
    upstream = torch.randn_like(inputs[2].expand(-1, -1, 2, -1))
    output = reference._dual_score_reference_with_seed(
        *inputs,
        seed,
        score_mode=score_mode,
        scale=1.2,
        dropout_p=0.4,
        mismatch_scale=2.1,
        evidence_power=0.25,
        information_weight=0.4,
        dropout_batch_offset=3,
    )
    actual = torch.autograd.grad(
        (output * upstream).sum(), inputs, retain_graph=True
    )

    direct_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in inputs
    )
    carrier = reference.dual_score_carrier(
        *direct_inputs,
        score_mode=score_mode,
        scale=1.2,
        dropout_p=0.4,
        dropout_seed=seed,
        mismatch_scale=2.1,
        evidence_power=0.25,
        information_weight=0.4,
        dropout_batch_offset=3,
    )
    expected = torch.autograd.grad((carrier * upstream).sum(), direct_inputs)

    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right, rtol=2e-14, atol=2e-14)


def test_no_grad_dropout_does_not_consume_rng():
    inputs = tuple(tensor.detach() for tensor in _inputs(seed=207, seq_len=5))
    state = torch.random.get_rng_state()
    with torch.no_grad():
        dropped = reference.rosa_dual_score_reference(*inputs, dropout_p=0.7)
    assert torch.equal(state, torch.random.get_rng_state())
    deterministic = reference.rosa_dual_score_reference(*inputs, dropout_p=0.0)
    assert torch.equal(dropped, deterministic)


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_singleton_returns_connected_zero_gradients(score_mode):
    inputs = tuple(
        torch.ones(1, 1, 1, 1, dtype=torch.float64, requires_grad=True)
        for _ in range(3)
    )
    output = reference.rosa_dual_score_reference(
        *inputs, score_mode=score_mode
    )
    gradients = torch.autograd.grad(output.sum(), inputs)

    assert torch.equal(output, torch.zeros_like(output))
    assert all(
        torch.equal(gradient, torch.zeros_like(gradient))
        for gradient in gradients
    )


@pytest.mark.parametrize("required_input", range(3))
def test_custom_backward_supports_partial_gradients(required_input):
    tensors = tuple(tensor.detach() for tensor in _inputs(seed=209, seq_len=5))
    inputs = tuple(
        tensor.requires_grad_(index == required_input)
        for index, tensor in enumerate(tensors)
    )
    output = reference.rosa_dual_score_reference(*inputs)
    output.square().sum().backward()

    for index, tensor in enumerate(inputs):
        if index == required_input:
            assert tensor.grad is not None
            assert bool(torch.isfinite(tensor.grad).all())
        else:
            assert tensor.grad is None


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_grouped_value_heads_and_noncontiguous_inputs(score_mode):
    generator = torch.Generator().manual_seed(213)
    query = torch.randn(2, 4, 7, 3, generator=generator).transpose(1, 2)
    key = torch.randn(2, 4, 7, 3, generator=generator).transpose(1, 2)
    value = torch.randn(2, 2, 7, 5, generator=generator).transpose(1, 2)
    inputs = tuple(tensor.requires_grad_() for tensor in (query, key, value))

    output = reference.rosa_dual_score_reference(
        *inputs, score_mode=score_mode
    )
    output.square().mean().backward()

    assert output.shape == (2, 7, 4, 5)
    assert all(not tensor.is_contiguous() for tensor in inputs)
    assert all(
        tensor.grad is not None and bool(torch.isfinite(tensor.grad).all())
        for tensor in inputs
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_cuda_low_precision_matches_fp32_reference(score_mode, dtype):
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip("bfloat16 is unavailable on this GPU")
    generator = torch.Generator(device="cuda").manual_seed(223)
    low_inputs = (
        torch.randn(1, 8, 2, 4, generator=generator, device="cuda", dtype=dtype),
        torch.randn(1, 8, 2, 4, generator=generator, device="cuda", dtype=dtype),
        torch.randn(1, 8, 1, 3, generator=generator, device="cuda", dtype=dtype),
    )
    low_inputs = tuple(tensor.requires_grad_() for tensor in low_inputs)
    fp32_inputs = tuple(
        tensor.detach().float().requires_grad_() for tensor in low_inputs
    )
    upstream = torch.randn(
        1, 8, 2, 3, generator=generator, device="cuda", dtype=dtype
    )

    low_output = reference.rosa_dual_score_reference(
        *low_inputs, score_mode=score_mode
    )
    fp32_output = reference.rosa_dual_score_reference(
        *fp32_inputs, score_mode=score_mode
    )
    low_gradients = torch.autograd.grad(low_output, low_inputs, upstream)
    fp32_gradients = torch.autograd.grad(
        fp32_output, fp32_inputs, upstream.float()
    )

    assert torch.equal(low_output.float(), fp32_output)
    for low, fp32 in zip(low_gradients, fp32_gradients):
        assert bool(torch.isfinite(low).all())
        torch.testing.assert_close(
            low.float(), fp32, rtol=3e-2, atol=5e-3
        )


def test_information_and_dual_keep_dense_finite_qk_credit():
    query, key, value = _inputs(seed=211, seq_len=9, heads=1, value_heads=1)
    upstream = torch.randn_like(value)
    for mode in ("information", "dual"):
        output = reference.rosa_dual_score_reference(
            query, key, value, score_mode=mode
        )
        grad_query, grad_key = torch.autograd.grad(
            (output * upstream).sum(), (query, key), retain_graph=True
        )
        assert bool(torch.isfinite(grad_query).all())
        assert bool(torch.isfinite(grad_key).all())
        assert int(torch.count_nonzero(grad_query[:, 1:])) > 0
        assert int(torch.count_nonzero(grad_key[:, :-1])) > 0


def test_null_and_latest_tie_hard_semantics_are_unchanged():
    query = torch.ones(1, 4, 1, 1, dtype=torch.float64, requires_grad=True)
    key = -torch.ones_like(query, requires_grad=True)
    value = torch.tensor(
        [[[[1.0]], [[-1.0]], [[1.0]], [[-1.0]]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    null_output = reference.rosa_dual_score_reference(query, key, value)
    assert torch.equal(null_output, torch.zeros_like(null_output))

    tie_query = torch.tensor(
        [[[-1.0], [-1.0], [-1.0], [1.0]]], dtype=torch.float64
    ).unsqueeze(2)
    tie_key = torch.tensor(
        [[[1.0], [1.0], [1.0], [-1.0]]], dtype=torch.float64
    ).unsqueeze(2)
    tie_value = value.detach().clone()
    _, lengths, routes, _ = reference._hard_route_forward_unbounded(
        tie_query, tie_key, tie_value
    )
    assert torch.equal(lengths[0, 0, 3, 1:4], torch.ones(3, dtype=torch.float64))
    assert routes[0, 0, 3].item() == 3
    assert reference.rosa_dual_score_reference(
        tie_query, tie_key, tie_value
    )[0, 3, 0, 0].item() == -1.0


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_packed_varlen_matches_independent_dense_segments(score_mode):
    generator = torch.Generator().manual_seed(307)
    query = torch.randn(7, 2, 4, generator=generator, dtype=torch.float64)
    key = torch.randn_like(query)
    value = torch.randn(7, 1, 3, generator=generator, dtype=torch.float64)
    cu_seqlens = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
    packed_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in (query, key, value)
    )
    packed = reference.rosa_dual_score_varlen_reference(
        *packed_inputs,
        cu_seqlens,
        score_mode=score_mode,
    )

    dense_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in (query, key, value)
    )
    dense = torch.cat(
        [
            reference.rosa_dual_score_reference(
                *(tensor[start:end].unsqueeze(0) for tensor in dense_inputs),
                score_mode=score_mode,
            ).squeeze(0)
            for start, end in ((0, 3), (3, 7))
        ],
        dim=0,
    )
    upstream = torch.randn_like(packed)
    packed_grads = torch.autograd.grad(
        (packed * upstream).sum(), packed_inputs
    )
    dense_grads = torch.autograd.grad((dense * upstream).sum(), dense_inputs)

    torch.testing.assert_close(packed, dense, rtol=0.0, atol=0.0)
    for left, right in zip(packed_grads, dense_grads):
        torch.testing.assert_close(left, right, rtol=2e-14, atol=2e-14)


@pytest.mark.parametrize("score_mode", reference.SCORE_MODES)
def test_packed_varlen_dropout_replays_global_seed_and_sequence_offsets(score_mode):
    generator = torch.Generator().manual_seed(311)
    base = (
        torch.randn(7, 2, 4, generator=generator, dtype=torch.float64),
        torch.randn(7, 2, 4, generator=generator, dtype=torch.float64),
        torch.randn(7, 1, 3, generator=generator, dtype=torch.float64),
    )
    cu_seqlens = torch.tensor([0, 3, 3, 7], dtype=torch.int32)
    packed_inputs = tuple(tensor.clone().requires_grad_() for tensor in base)
    torch.manual_seed(313)
    packed = reference.rosa_dual_score_varlen_reference(
        *packed_inputs,
        cu_seqlens,
        score_mode=score_mode,
        dropout_p=0.4,
    )

    dense_inputs = tuple(tensor.clone().requires_grad_() for tensor in base)
    torch.manual_seed(313)
    seed = reference.make_dropout_seed(dense_inputs[0], 0.4, True)
    dense = torch.cat(
        [
            reference._dual_score_reference_with_seed(
                *(tensor[start:end].unsqueeze(0) for tensor in dense_inputs),
                seed,
                score_mode=score_mode,
                scale=1.0,
                dropout_p=0.4,
                mismatch_scale=3.0,
                evidence_power=reference.DEFAULT_EVIDENCE_POWER,
                information_weight=reference.DEFAULT_INFORMATION_WEIGHT,
                dropout_batch_offset=sequence,
            ).squeeze(0)
            for sequence, (start, end) in enumerate(((0, 3), (3, 3), (3, 7)))
            if start != end
        ],
        dim=0,
    )
    upstream = torch.randn_like(packed)
    packed_gradients = torch.autograd.grad(
        (packed * upstream).sum(), packed_inputs
    )
    dense_gradients = torch.autograd.grad((dense * upstream).sum(), dense_inputs)

    assert torch.equal(packed, dense)
    for left, right in zip(packed_gradients, dense_gradients):
        assert torch.equal(left, right)


@pytest.mark.parametrize(
    ("keyword", "value", "message"),
    [
        ("score_mode", "missing", "score_mode"),
        ("evidence_power", 0.0, "evidence_power"),
        ("evidence_power", 1.1, "evidence_power"),
        ("information_weight", -0.1, "information_weight"),
        ("information_weight", 1.1, "information_weight"),
    ],
)
def test_invalid_score_controls_are_rejected(keyword, value, message):
    query, key, dense_value = _inputs(seq_len=3)
    arguments = {keyword: value}
    with pytest.raises(ValueError, match=message):
        reference.rosa_dual_score_reference(
            query, key, dense_value, **arguments
        )
