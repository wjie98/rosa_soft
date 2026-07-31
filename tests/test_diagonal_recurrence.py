import pytest
import torch

from benchmarks.diagonal_recurrence import (
    diagonal_suffix_log_gate_vjp,
    diagonal_suffix_scores,
    diagonal_symbol_vjp,
    direct_finite_suffix_scores,
    finite_suffix_log_gate_vjp_scan,
    finite_suffix_scores_scan,
)
from rosa_soft.soft_reference import _suffix_prefix_product_scores


@pytest.mark.parametrize("length", [1, 2, 7, 32, 65])
@pytest.mark.parametrize("max_suffix_length", [1, 2, 8, 32, 128])
def test_diagonal_scan_matches_direct_scores_and_log_gate_vjp(
    length,
    max_suffix_length,
):
    generator = torch.Generator().manual_seed(
        1000 + length + max_suffix_length
    )
    log_gates = (
        -6.0
        * torch.rand(
            3,
            length,
            dtype=torch.float64,
            generator=generator,
        )
    ).requires_grad_()
    route_score_vjp = torch.randn(
        log_gates.shape,
        dtype=torch.float64,
        generator=generator,
    )
    expected_scores = direct_finite_suffix_scores(
        log_gates,
        max_suffix_length,
    )
    expected_vjp = torch.autograd.grad(
        (expected_scores * route_score_vjp).sum(),
        log_gates,
    )[0]

    actual_scores = finite_suffix_scores_scan(
        log_gates.detach(),
        max_suffix_length,
    )
    actual_vjp = finite_suffix_log_gate_vjp_scan(
        log_gates.detach(),
        route_score_vjp,
        max_suffix_length,
    )

    torch.testing.assert_close(
        actual_scores,
        expected_scores.detach(),
        rtol=1e-12,
        atol=1e-12,
    )
    torch.testing.assert_close(
        actual_vjp,
        expected_vjp,
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("seq_len", [2, 7, 33])
@pytest.mark.parametrize("max_suffix_length", [1, 4, 32, 128])
def test_diagonal_matrix_mapping_matches_reference(
    seq_len,
    max_suffix_length,
):
    generator = torch.Generator().manual_seed(
        2000 + seq_len + max_suffix_length
    )
    gates = torch.exp(
        -6.0
        * torch.rand(
            2,
            3,
            seq_len,
            seq_len,
            dtype=torch.float64,
            generator=generator,
        )
    )
    row = torch.arange(seq_len).view(seq_len, 1)
    route = torch.arange(seq_len).view(1, seq_len)
    causal_nonnull = (route >= 1) & (route <= row)
    gates = gates * causal_nonnull

    expected = _suffix_prefix_product_scores(
        gates,
        max_suffix_length,
    )
    actual = diagonal_suffix_scores(
        gates,
        max_suffix_length,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("seq_len", [2, 7, 33])
@pytest.mark.parametrize("max_suffix_length", [1, 4, 32, 128])
def test_diagonal_matrix_log_gate_vjp_matches_autograd(
    seq_len,
    max_suffix_length,
):
    generator = torch.Generator().manual_seed(
        2500 + seq_len + max_suffix_length
    )
    log_gates = (
        -6.0
        * torch.rand(
            2,
            3,
            seq_len,
            seq_len,
            dtype=torch.float64,
            generator=generator,
        )
    ).requires_grad_()
    row = torch.arange(seq_len).view(seq_len, 1)
    route = torch.arange(seq_len).view(1, seq_len)
    causal_nonnull = (route >= 1) & (route <= row)
    gates = log_gates.exp() * causal_nonnull
    route_score_vjp = torch.randn(
        gates.shape,
        dtype=torch.float64,
        generator=generator,
    )
    scores = _suffix_prefix_product_scores(
        gates,
        max_suffix_length,
    )
    expected = torch.autograd.grad(
        (scores * route_score_vjp).sum(),
        log_gates,
    )[0]

    actual = diagonal_suffix_log_gate_vjp(
        gates.detach(),
        route_score_vjp,
        max_suffix_length,
    )

    torch.testing.assert_close(
        actual,
        expected,
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("seq_len", [2, 7, 33])
@pytest.mark.parametrize("max_suffix_length", [1, 4, 32, 128])
def test_diagonal_symbol_vjp_matches_full_matrix_autograd(
    dtype,
    seq_len,
    max_suffix_length,
):
    generator = torch.Generator().manual_seed(
        2700 + seq_len + max_suffix_length
    )
    symbol_dim = 16
    query_symbols = (
        torch.randn(
            2,
            3,
            seq_len,
            symbol_dim,
            dtype=dtype,
            generator=generator,
        ).sign()
    ).requires_grad_()
    key_symbols = (
        torch.randn(
            2,
            3,
            seq_len,
            symbol_dim,
            dtype=dtype,
            generator=generator,
        ).sign()
    ).requires_grad_()
    mismatch_scale = 3.0
    mismatch_rate = 0.5 * (
        1.0
        - query_symbols.unsqueeze(-2)
        * key_symbols[..., :-1, :].unsqueeze(-3)
    ).mean(dim=-1)
    local_match_gates = torch.nn.functional.pad(
        torch.exp(-mismatch_scale * mismatch_rate),
        (1, 0),
    )
    row = torch.arange(seq_len).view(seq_len, 1)
    route = torch.arange(seq_len).view(1, seq_len)
    causal_nonnull = (route >= 1) & (route <= row)
    local_match_gates = local_match_gates * causal_nonnull
    route_score_vjp = torch.randn(
        local_match_gates.shape,
        dtype=dtype,
        generator=generator,
    )
    scores = _suffix_prefix_product_scores(
        local_match_gates,
        max_suffix_length,
    )
    expected_query, expected_key = torch.autograd.grad(
        (scores * route_score_vjp).sum(),
        (query_symbols, key_symbols),
    )

    actual_query, actual_key = diagonal_symbol_vjp(
        query_symbols.detach(),
        key_symbols.detach(),
        route_score_vjp,
        max_suffix_length,
        mismatch_scale,
    )

    tolerance = 2e-5 if dtype == torch.float32 else 1e-11
    torch.testing.assert_close(
        actual_query,
        expected_query,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        actual_key,
        expected_key,
        rtol=tolerance,
        atol=tolerance,
    )
    assert torch.count_nonzero(actual_key[..., -1, :]) == 0


@pytest.mark.parametrize(
    ("seq_len", "max_suffix_length"),
    [(129, 32), (129, 128), (257, 32), (257, 128)],
)
def test_diagonal_symbol_vjp_fp32_exact_match_with_softmax_adjoint(
    seq_len,
    max_suffix_length,
):
    generator = torch.Generator().manual_seed(
        2900 + seq_len + max_suffix_length
    )
    symbol_dim = 32
    query_symbols = torch.ones(
        1,
        seq_len,
        symbol_dim,
        requires_grad=True,
    )
    key_symbols = torch.ones_like(
        query_symbols,
        requires_grad=True,
    )
    mismatch_scale = 3.0
    scale = 2.0
    mismatch_rate = 0.5 * (
        1.0
        - query_symbols.unsqueeze(-2)
        * key_symbols[..., :-1, :].unsqueeze(-3)
    ).mean(dim=-1)
    local_match_gates = torch.nn.functional.pad(
        torch.exp(-mismatch_scale * mismatch_rate),
        (1, 0),
    )
    row = torch.arange(seq_len).view(seq_len, 1)
    route = torch.arange(seq_len).view(1, seq_len)
    causal_nonnull = (route >= 1) & (route <= row)
    local_match_gates = local_match_gates * causal_nonnull
    scores = _suffix_prefix_product_scores(
        local_match_gates,
        max_suffix_length,
    )
    utility = torch.randn(
        scores.shape,
        generator=generator,
    ) * causal_nonnull
    candidate_count_log = (
        torch.arange(seq_len)
        .clamp_min(1)
        .float()
        .log()
        .view(1, seq_len, 1)
    )
    nonnull_logits = (
        scale * scores - candidate_count_log
    ).masked_fill(~causal_nonnull, -torch.inf)
    all_probabilities = torch.softmax(
        torch.cat(
            (
                torch.full((1, seq_len, 1), 0.5 * scale),
                nonnull_logits[..., 1:],
            ),
            dim=-1,
        ),
        dim=-1,
    )
    probabilities = torch.nn.functional.pad(
        all_probabilities[..., 1:],
        (1, 0),
    )
    expected_utility = (
        probabilities * utility
    ).sum(dim=-1, keepdim=True)
    route_score_vjp = (
        scale
        * probabilities
        * (utility - expected_utility)
    ).detach()
    expected_query, expected_key = torch.autograd.grad(
        (scores * route_score_vjp).sum(),
        (query_symbols, key_symbols),
    )

    actual_query, actual_key = diagonal_symbol_vjp(
        query_symbols.detach(),
        key_symbols.detach(),
        route_score_vjp,
        max_suffix_length,
        mismatch_scale,
    )

    torch.testing.assert_close(
        actual_query,
        expected_query,
        rtol=5e-5,
        atol=5e-5,
    )
    torch.testing.assert_close(
        actual_key,
        expected_key,
        rtol=5e-5,
        atol=5e-5,
    )


@pytest.mark.parametrize(
    "pattern",
    ["exact", "one_mismatch", "alternating", "random_hamming"],
)
@pytest.mark.parametrize("max_suffix_length", [1, 32, 128])
def test_diagonal_scan_fp32_stays_stable_on_discrete_gate_patterns(
    pattern,
    max_suffix_length,
):
    length = 1024
    mismatch_scale = 3.0
    symbol_dim = 32
    generator = torch.Generator().manual_seed(
        3000 + max_suffix_length
    )
    if pattern == "exact":
        mismatch_count = torch.zeros(2, length)
    elif pattern == "one_mismatch":
        mismatch_count = torch.ones(2, length)
    elif pattern == "alternating":
        mismatch_count = (
            torch.arange(length).remainder(2).expand(2, -1)
        )
    else:
        mismatch_count = torch.randint(
            0,
            symbol_dim + 1,
            (2, length),
            generator=generator,
        )
    log_gates = (
        -mismatch_scale
        * mismatch_count.to(torch.float32)
        / symbol_dim
    ).requires_grad_()
    route_score_vjp = torch.randn(
        log_gates.shape,
        generator=generator,
    )
    expected_scores = direct_finite_suffix_scores(
        log_gates,
        max_suffix_length,
    )
    expected_vjp = torch.autograd.grad(
        (expected_scores * route_score_vjp).sum(),
        log_gates,
    )[0]

    actual_scores = finite_suffix_scores_scan(
        log_gates.detach(),
        max_suffix_length,
    )
    actual_vjp = finite_suffix_log_gate_vjp_scan(
        log_gates.detach(),
        route_score_vjp,
        max_suffix_length,
    )

    torch.testing.assert_close(
        actual_scores,
        expected_scores.detach(),
        rtol=2e-5,
        atol=2e-5,
    )
    torch.testing.assert_close(
        actual_vjp,
        expected_vjp,
        rtol=1e-3 if pattern == "exact" else 3e-5,
        atol=1e-3 if pattern == "exact" else 3e-5,
    )
