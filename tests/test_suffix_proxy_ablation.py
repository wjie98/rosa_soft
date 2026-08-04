import itertools
import math
from argparse import Namespace

import pytest
import torch

from benchmarks import suffix_proxy_ablation as ablation
from rosa_soft.soft_reference import _hard_route_forward


def _inputs(bits: int = 4):
    generator = torch.Generator().manual_seed(1234 + bits)
    return tuple(
        tensor.requires_grad_(True)
        for tensor in (
            torch.randn(1, 6, 2, bits, generator=generator),
            torch.randn(1, 6, 2, bits, generator=generator),
            torch.randn(1, 6, 1, 3, generator=generator),
        )
    )


def test_normalized_sqrt_suffix_utility_contract():
    scores = torch.tensor([0.0, 1.0, 2.0, 8.0, 32.0], requires_grad=True)
    utility = ablation._suffix_score_utility(scores, "sqrt")

    assert utility[0].item() == pytest.approx(0.0)
    assert utility[1].item() == pytest.approx(1.0)
    assert bool((utility[1:] > utility[:-1]).all())
    secant_slopes = (utility[1:] - utility[:-1]) / (
        scores[1:] - scores[:-1]
    )
    assert bool((secant_slopes[1:] < secant_slopes[:-1]).all())

    utility.sum().backward()
    expected = ablation._SQRT_SUFFIX_SCALE / (2.0 * torch.sqrt(1.0 + scores))
    torch.testing.assert_close(scores.grad, expected)


@pytest.mark.parametrize(
    "mode",
    ["power075", "sqrt", "power025", "log"],
)
def test_concave_suffix_utilities_share_zero_one_calibration(mode):
    scores = torch.arange(0.0, 33.0, dtype=torch.float64).requires_grad_()
    utility = ablation._suffix_score_utility(scores, mode)

    assert utility[0].item() == pytest.approx(0.0)
    assert utility[1].item() == pytest.approx(1.0)
    increments = utility[1:] - utility[:-1]
    assert bool((increments > 0).all())
    assert bool((increments[1:] < increments[:-1]).all())

    utility.sum().backward()
    assert bool(torch.isfinite(scores.grad).all())
    assert bool((scores.grad > 0).all())


def test_power_family_converges_toward_logarithmic_limit():
    scores = torch.tensor([0.0, 1.0, 2.0, 8.0, 32.0], dtype=torch.float64)
    alpha = 1e-6
    power = torch.expm1(alpha * torch.log1p(scores)) / math.expm1(
        alpha * math.log(2.0)
    )
    logarithmic = ablation._suffix_score_utility(scores, "log")

    torch.testing.assert_close(power, logarithmic, rtol=2e-6, atol=1e-12)


def test_collision_likelihood_ratio_has_unit_random_null_mean():
    bits = 2
    suffix_length = 3
    bit_patterns = list(
        itertools.product(
            (0, 1),
            repeat=bits * suffix_length,
        )
    )
    mismatch_counts = torch.tensor(
        [
            [
                sum(pattern[offset : offset + bits])
                for offset in range(0, len(pattern), bits)
            ]
            for pattern in bit_patterns
        ],
        dtype=torch.float64,
    )
    diagonal_gates = torch.exp(-3.0 * mismatch_counts / bits)
    local_gates = torch.zeros(
        len(bit_patterns),
        1,
        suffix_length,
        suffix_length,
        dtype=torch.float64,
    )
    diagonal = torch.arange(suffix_length)
    local_gates[:, 0, diagonal, diagonal] = diagonal_gates

    route_log_ratios = ablation._collision_log_likelihood_ratio_scores(
        local_gates,
        symbol_dim=bits,
        max_suffix_length=suffix_length,
        mismatch_scale=3.0,
    )

    assert route_log_ratios[:, 0, -1, -1].exp().mean().item() == pytest.approx(
        1.0,
        rel=1e-12,
        abs=1e-12,
    )


@pytest.mark.parametrize("bits", [1, 2, 4, 8, 16, 32])
def test_one_mismatch_gate_has_expected_dimension_scaling(bits):
    query = torch.ones(1, 2, 1, bits)
    key = torch.ones(1, 2, 1, bits)
    key[0, 0, 0, 0] = -1.0
    mask = torch.tensor([[True, False], [True, True]])

    mean_gate = ablation._pairwise_proxy_match_gates(
        query,
        key,
        mask,
        3.0,
        "mean",
    )[..., 1, 1]
    sqrt_gate = ablation._pairwise_proxy_match_gates(
        query,
        key,
        mask,
        3.0,
        "sqrt",
    )[..., 1, 1]

    assert mean_gate.item() == pytest.approx(math.exp(-3.0 / bits))
    assert sqrt_gate.item() == pytest.approx(
        math.exp(-3.0 / math.sqrt(bits))
    )


def test_proxy_scores_do_not_leak_sign_preserving_amplitude():
    query, key, _ = _inputs()
    query_scaled = query.detach() * torch.rand_like(query).add_(0.1)
    key_scaled = key.detach() * torch.rand_like(key).add_(0.1)

    for dimension_mode, suffix_mode in ablation.PROXY_CONFIGS.values():
        original = ablation._proxy_state(
            query,
            key,
            max_suffix_length=4,
            scale=1.0,
            mismatch_scale=2.0,
            dimension_mode=dimension_mode,
            suffix_mode=suffix_mode,
        )
        scaled = ablation._proxy_state(
            query_scaled,
            key_scaled,
            max_suffix_length=4,
            scale=1.0,
            mismatch_scale=2.0,
            dimension_mode=dimension_mode,
            suffix_mode=suffix_mode,
        )
        for left, right in zip(original, scaled):
            torch.testing.assert_close(left, right, rtol=0.0, atol=0.0)


def test_all_proxy_variants_share_exact_hard_forward():
    query, key, value = _inputs()
    expected, _, _, _ = _hard_route_forward(query, key, value, 4)

    outputs = [
        ablation.rosa_soft_suffix_proxy(
            query,
            key,
            value,
            proxy=proxy,
            max_suffix_length=4,
            scale=1.3,
            mismatch_scale=2.0,
        )
        for proxy in ablation.PROXY_CONFIGS
    ]
    for output in outputs:
        torch.testing.assert_close(output, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("proxy", tuple(ablation.PROXY_CONFIGS))
def test_custom_backward_matches_materialized_carrier(proxy):
    query, key, value = _inputs()
    generator = torch.Generator().manual_seed(987)
    upstream = torch.randn(1, 6, 2, 3, generator=generator)
    output = ablation.rosa_soft_suffix_proxy(
        query,
        key,
        value,
        proxy=proxy,
        max_suffix_length=4,
        scale=1.2,
        mismatch_scale=1.7,
    )
    actual = torch.autograd.grad(
        (output * upstream).sum(),
        (query, key, value),
    )

    direct_inputs = tuple(
        tensor.detach().requires_grad_(True) for tensor in (query, key, value)
    )
    dimension_mode, suffix_mode = ablation.PROXY_CONFIGS[proxy]
    carrier = ablation._surrogate_carrier(
        *direct_inputs,
        max_suffix_length=4,
        scale=1.2,
        mismatch_scale=1.7,
        dimension_mode=dimension_mode,
        suffix_mode=suffix_mode,
    )
    expected = torch.autograd.grad(
        (carrier * upstream).sum(),
        direct_inputs,
    )
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)


def test_small_matrix_reports_every_declared_proxy_cell():
    args = Namespace(
        proxies=list(ablation.PROXY_CONFIGS),
        qk_bits=[2],
        windows=[2],
        mismatch_scales=[1.5],
        model_seeds=[0],
        shell_seeds=[0],
        device="cpu",
        steps=1,
        success_threshold=1e-3,
        sequence_length=8,
        shell_sequence_lengths=[8],
        vocab_size=4,
        motif_min=2,
        motif_max=3,
        target_mode="strict-longest-latest",
        hidden_size=8,
        heads=2,
        value_heads=1,
        value_bits=2,
        scale=1.0,
        learning_rate=0.01,
        weight_decay=0.01,
        grad_clip=1.0,
        fit_only=False,
        shell_only=False,
        run_competition=False,
        competition_only=False,
        competition_bits=[2],
        competition_distractor_lengths=[2],
        competition_short_length=1,
        competition_max_length=4,
        competition_initial_margin=0.25,
        competition_learning_rate=1.0,
        competition_steps=1,
    )

    report = ablation.run_matrix(args)

    proxy_count = len(ablation.PROXY_CONFIGS)
    assert len(report["fits"]) == proxy_count
    assert len(report["gradient_shell"]) == proxy_count
    assert len(report["fit_summaries"]) == proxy_count
    assert {row["proxy"] for row in report["fits"]} == set(
        ablation.PROXY_CONFIGS
    )


def test_sqrt_suffix_preserves_more_short_target_credit():
    args = Namespace(
        device="cpu",
        competition_max_length=8,
        competition_short_length=1,
        competition_initial_margin=0.25,
        competition_learning_rate=1.0,
        competition_steps=1,
        scale=1.0,
    )
    baseline = ablation.run_length_competition(
        proxy="baseline",
        qk_bits=4,
        distractor_length=4,
        mismatch_scale=3.0,
        args=args,
    )
    compressed = ablation.run_length_competition(
        proxy="sqrt_suffix",
        qk_bits=4,
        distractor_length=4,
        mismatch_scale=3.0,
        args=args,
    )

    assert compressed["initial_target_probability"] > baseline[
        "initial_target_probability"
    ]
    assert abs(compressed["initial_oldest_bit_gradient"]) > abs(
        baseline["initial_oldest_bit_gradient"]
    )
