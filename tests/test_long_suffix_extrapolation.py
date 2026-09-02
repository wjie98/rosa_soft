import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks import fast_weight_proxy
from benchmarks import long_suffix_extrapolation as extrapolation
from benchmarks import temporal_quadratic_proxy
from benchmarks import suffix_kernel_proxy
from rosa_soft.soft_reference import _hard_sign, rosa_soft_reference


def _spec(window=4, *, seed=3, probes=3, context_length=96):
    return extrapolation.make_gate_spec(
        seed=seed,
        probes=probes,
        window=window,
        bits=8,
        train_context_length=context_length,
    )


def _brute_suffix_lengths(query, keys, window):
    query = extrapolation._pack_signs(_hard_sign(query))
    keys = extrapolation._pack_signs(_hard_sign(keys))
    lengths = torch.zeros(keys.shape, dtype=torch.int64)
    for probe in range(keys.size(0)):
        for endpoint in range(keys.size(1)):
            length = 0
            for offset in range(window):
                query_position = window - 1 - offset
                key_position = endpoint - offset
                if key_position < 0:
                    break
                if keys[probe, key_position] != query[probe, query_position]:
                    break
                length += 1
            lengths[probe, endpoint] = length
    return lengths


def test_dense_suffix_scan_matches_independent_scalar_oracle():
    spec = _spec(window=4, context_length=48)
    context = extrapolation.materialize_context(
        spec,
        48,
        device=torch.device("cpu"),
    )
    query = extrapolation.make_query_logits(
        spec,
        torch.tensor(-0.25),
        logit_margin=1.0,
    )

    actual = extrapolation._suffix_lengths_from_codes(
        extrapolation._pack_signs(_hard_sign(query)),
        extrapolation._pack_signs(context.key_signs),
        spec.window,
    )
    expected = _brute_suffix_lengths(query, context.key_signs, spec.window)

    assert torch.equal(actual, expected)


@pytest.mark.parametrize("window", [1, 2, 4, 8, 32])
def test_solved_gate_has_one_exact_target_and_strict_aligned_distractors(window):
    context_length = max(96, 4 * window)
    context_length = (context_length // window) * window
    spec = _spec(window=window, context_length=context_length)
    context = extrapolation.materialize_context(
        spec,
        context_length,
        device=torch.device("cpu"),
    )
    solved = extrapolation.make_query_logits(
        spec,
        torch.tensor(0.25),
        logit_margin=1.0,
    )
    route = extrapolation.dense_hard_route(solved, context, window)

    assert torch.equal(route.selected_positions, context.target_positions)
    assert torch.equal(
        route.target_suffix_lengths,
        torch.full_like(route.target_suffix_lengths, window),
    )
    assert torch.equal(
        route.max_distractor_suffix_lengths,
        torch.full_like(route.max_distractor_suffix_lengths, max(window - 1, 0)),
    )
    assert not route.exact_distractor_counts.any()
    assert torch.equal(
        route.strict_distractor_counts,
        torch.full_like(
            route.strict_distractor_counts,
            context_length // window - 1,
        ),
    )


@pytest.mark.parametrize("window", [1, 2, 4, 8, 32])
def test_initial_fault_installs_later_exact_distractor_traps(window):
    context_length = max(192, 8 * window)
    context_length = (context_length // window) * window
    spec = _spec(window=window, context_length=context_length)
    context = extrapolation.materialize_context(
        spec,
        context_length,
        device=torch.device("cpu"),
    )
    faulty = extrapolation.make_query_logits(
        spec,
        torch.tensor(-0.25),
        logit_margin=1.0,
    )
    route = extrapolation.dense_hard_route(faulty, context, window)

    assert torch.equal(
        route.target_suffix_lengths,
        torch.full_like(route.target_suffix_lengths, max(window - 1, 0)),
    )
    assert bool(route.exact_distractor_counts.gt(0).all())
    assert not bool(route.selected_positions.eq(context.target_positions).any())


@pytest.mark.parametrize("window", [1, 2, 4, 8])
@pytest.mark.parametrize("fault_logit", [-0.25, 0.25])
def test_streaming_scan_is_exactly_equal_to_dense_scan(window, fault_logit):
    spec = _spec(window=window, context_length=96)
    context = extrapolation.materialize_context(
        spec,
        96,
        device=torch.device("cpu"),
    )
    query = extrapolation.make_query_logits(
        spec,
        torch.tensor(fault_logit),
        logit_margin=1.0,
    )
    dense = extrapolation.dense_hard_route(query, context, window)
    streamed = extrapolation.streaming_hard_evaluate(
        spec,
        fault_logit,
        context_length=96,
        chunk_size=13,
        logit_margin=1.0,
        device=torch.device("cpu"),
    )

    assert streamed["route_accuracy"] == pytest.approx(
        float(dense.selected_positions.eq(context.target_positions).float().mean())
    )
    assert streamed["target_suffix_mean"] == pytest.approx(
        float(dense.target_suffix_lengths.float().mean())
    )
    assert streamed["max_suffix_mean"] == pytest.approx(
        float(dense.max_suffix_lengths.float().mean())
    )
    assert streamed["exact_distractor_mean"] == pytest.approx(
        float(dense.exact_distractor_counts.float().mean())
    )
    assert streamed["strict_distractor_mean"] == pytest.approx(
        float(dense.strict_distractor_counts.float().mean())
    )


def test_all_estimators_have_identical_hard_forward_but_different_oldest_vjp():
    spec = _spec(window=4, context_length=96)
    context = extrapolation.materialize_context(
        spec,
        96,
        device=torch.device("cpu"),
    )
    outputs = {}
    gradients = {}
    for estimator in extrapolation.ESTIMATORS:
        fault = torch.tensor(-0.25, requires_grad=True)
        output, _, _ = extrapolation.hard_forward_with_proxy(
            estimator,
            spec,
            context,
            fault,
            logit_margin=1.0,
            scale=1.0,
            mismatch_scale=3.0,
        )
        loss = (output - spec.target_values).square().mean()
        gradients[estimator] = float(torch.autograd.grad(loss, fault)[0])
        outputs[estimator] = output.detach()

    for output in outputs.values():
        assert torch.equal(output, outputs["production"])
    assert gradients["production"] < 0.0
    expected_bitflip = -2.0 * float(
        (outputs["production"] - spec.target_values).square().mean()
    )
    assert gradients["exact_relevant_bitflip"] == pytest.approx(expected_bitflip)
    assert gradients["state_quadratic_attention"] == 0.0


def _full_sequence_inputs(spec, context, query_trajectory):
    probes, context_length, bits = context.key_signs.shape
    sequence_length = context_length + 1
    query = torch.full(
        (probes, sequence_length, 1, bits),
        -1.0,
        dtype=query_trajectory.dtype,
    )
    query[:, -spec.window :, 0] = query_trajectory
    key = torch.full_like(query, -1.0)
    key[:, :context_length, 0] = context.key_signs
    value_bits = context.values.size(-1)
    value = torch.zeros(
        probes,
        sequence_length,
        1,
        value_bits,
        dtype=query_trajectory.dtype,
    )
    value[:, 1:, 0] = context.values
    return query, key, value


def test_final_row_production_carrier_vjp_matches_full_reference_exactly():
    spec = _spec(window=4, probes=2, context_length=12)
    context = extrapolation.materialize_context(
        spec,
        12,
        device=torch.device("cpu"),
    )
    query_trajectory = extrapolation.make_query_logits(
        spec,
        torch.tensor(-0.25),
        logit_margin=1.0,
    ).to(torch.float64)
    context = extrapolation.DenseContext(
        key_signs=context.key_signs.to(torch.float64),
        values=context.values.to(torch.float64),
        target_positions=context.target_positions,
    )
    full_query, full_key, full_value = _full_sequence_inputs(
        spec,
        context,
        query_trajectory,
    )
    full_query.requires_grad_()
    reference = rosa_soft_reference(
        full_query,
        full_key,
        full_value,
        max_suffix_length=spec.window,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
    )[:, -1, 0]
    reference_gradient = torch.autograd.grad(reference.sum(), full_query)[0][
        :, -spec.window :, 0
    ]

    direct_query = query_trajectory.detach().requires_grad_()
    carrier, _ = extrapolation._production_suffix_carrier(
        direct_query,
        context,
        window=spec.window,
        scale=1.0,
        mismatch_scale=3.0,
    )
    direct_gradient = torch.autograd.grad(carrier.sum(), direct_query)[0]
    direct_hard = extrapolation.dense_hard_route(
        direct_query,
        context,
        spec.window,
    ).output

    assert torch.equal(reference, direct_hard)
    torch.testing.assert_close(
        direct_gradient,
        reference_gradient,
        rtol=2e-14,
        atol=2e-14,
    )


def test_final_row_quadratic_carrier_matches_full_carrier_exactly():
    spec = _spec(window=4, probes=2, context_length=12)
    context = extrapolation.materialize_context(
        spec,
        12,
        device=torch.device("cpu"),
    )
    query_trajectory = extrapolation.make_query_logits(
        spec,
        torch.tensor(-0.25),
        logit_margin=1.0,
    ).to(torch.float64)
    context = extrapolation.DenseContext(
        key_signs=context.key_signs.to(torch.float64),
        values=context.values.to(torch.float64),
        target_positions=context.target_positions,
    )
    full_query, full_key, full_value = _full_sequence_inputs(
        spec,
        context,
        query_trajectory,
    )
    full_query.requires_grad_()
    reference = fast_weight_proxy._fast_weight_carrier(
        full_query,
        full_key,
        full_value,
        proxy="state_quadratic_attention",
        fingerprint_length=1,
        mismatch_scale=3.0,
        sketch_dim=8,
        sketch_seed=0,
    )[:, -1, 0]
    reference_gradient = torch.autograd.grad(reference.sum(), full_query)[0][
        :, -spec.window :, 0
    ]

    direct_query = query_trajectory.detach().requires_grad_()
    direct, _ = extrapolation._quadratic_attention_carrier(
        direct_query,
        context,
        mismatch_scale=3.0,
    )
    direct_gradient = torch.autograd.grad(direct.sum(), direct_query)[0]

    torch.testing.assert_close(direct, reference, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(
        direct_gradient,
        reference_gradient,
        rtol=2e-14,
        atol=2e-14,
    )


def test_final_row_temporal_carrier_matches_full_carrier_exactly():
    spec = _spec(window=4, probes=2, context_length=12)
    context = extrapolation.materialize_context(
        spec,
        12,
        device=torch.device("cpu"),
    )
    query_trajectory = extrapolation.make_query_logits(
        spec,
        torch.tensor(-0.25),
        logit_margin=1.0,
    ).to(torch.float64)
    context = extrapolation.DenseContext(
        key_signs=context.key_signs.to(torch.float64),
        values=context.values.to(torch.float64),
        target_positions=context.target_positions,
    )
    full_query, full_key, full_value = _full_sequence_inputs(
        spec,
        context,
        query_trajectory,
    )
    full_query.requires_grad_()
    reference = temporal_quadratic_proxy._temporal_quadratic_carrier(
        full_query,
        full_key,
        full_value,
        max_suffix_length=spec.window,
        state_dim=8,
    )[:, -1, 0]
    reference_gradient = torch.autograd.grad(reference.sum(), full_query)[0][
        :, -spec.window :, 0
    ]

    direct_query = query_trajectory.detach().requires_grad_()
    temporal_context = extrapolation._make_temporal_quadratic_context(
        context,
        window=spec.window,
        state_dim=8,
    )
    direct, _ = extrapolation._temporal_quadratic_attention_carrier(
        direct_query,
        temporal_context,
        window=spec.window,
        state_dim=8,
    )
    direct_gradient = torch.autograd.grad(direct.sum(), direct_query)[0]

    torch.testing.assert_close(direct, reference, rtol=2e-14, atol=2e-14)
    torch.testing.assert_close(
        direct_gradient,
        reference_gradient,
        rtol=2e-14,
        atol=2e-14,
    )


@pytest.mark.parametrize("route_kernel", suffix_kernel_proxy.ROUTE_KERNELS)
def test_streamed_suffix_kernel_context_matches_full_carrier_exactly(route_kernel):
    spec = _spec(window=3, probes=2, context_length=12)
    context = extrapolation.materialize_context(
        spec,
        12,
        device=torch.device("cpu"),
    )
    query_trajectory = extrapolation.make_query_logits(
        spec,
        torch.tensor(-0.25),
        logit_margin=1.0,
    ).to(torch.float64)
    context = extrapolation.DenseContext(
        key_signs=context.key_signs.to(torch.float64),
        values=context.values.to(torch.float64),
        target_positions=context.target_positions,
    )
    full_query, full_key, full_value = _full_sequence_inputs(
        spec,
        context,
        query_trajectory,
    )
    full_query.requires_grad_()
    reference = suffix_kernel_proxy._suffix_kernel_carrier(
        full_query,
        full_key,
        full_value,
        representation="sketch",
        read_rule="additive",
        route_kernel=route_kernel,
        fingerprint_length=spec.window,
        mismatch_scale=3.0,
        sketch_dim=16,
        sketch_count=2,
        sketch_seed=7,
    )[:, -1, 0]
    reference_gradient = torch.autograd.grad(reference.sum(), full_query)[0][
        :, -spec.window :, 0
    ]

    suffix_context = extrapolation._make_suffix_kernel_context(
        context,
        route_kernel=route_kernel,
        window=spec.window,
        mismatch_scale=3.0,
        sketch_dim=16,
        sketch_count=2,
        sketch_seed=7,
    )
    direct_query = query_trajectory.detach().requires_grad_()
    direct, _ = extrapolation._suffix_kernel_attention_carrier(
        direct_query,
        suffix_context,
    )
    direct_gradient = torch.autograd.grad(direct.sum(), direct_query)[0]

    torch.testing.assert_close(direct, reference, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        direct_gradient,
        reference_gradient,
        rtol=2e-12,
        atol=2e-12,
    )


@pytest.mark.parametrize("route_kernel", suffix_kernel_proxy.ROUTE_KERNELS)
def test_exact_candidate_suffix_kernel_matches_full_feature_carrier(route_kernel):
    spec = extrapolation.make_gate_spec(
        seed=4,
        probes=2,
        window=2,
        bits=3,
        train_context_length=12,
    )
    context = extrapolation.materialize_context(
        spec,
        12,
        device=torch.device("cpu"),
    )
    query_trajectory = extrapolation.make_query_logits(
        spec,
        torch.tensor(-0.25),
        logit_margin=1.0,
    ).to(torch.float64)
    context = extrapolation.DenseContext(
        key_signs=context.key_signs.to(torch.float64),
        values=context.values.to(torch.float64),
        target_positions=context.target_positions,
    )
    full_query, full_key, full_value = _full_sequence_inputs(
        spec,
        context,
        query_trajectory,
    )
    full_query.requires_grad_()
    reference = suffix_kernel_proxy._suffix_kernel_carrier(
        full_query,
        full_key,
        full_value,
        representation="exact",
        read_rule="additive",
        route_kernel=route_kernel,
        fingerprint_length=spec.window,
        mismatch_scale=3.0,
        sketch_dim=16,
        sketch_count=1,
        sketch_seed=0,
    )[:, -1, 0]
    reference_gradient = torch.autograd.grad(reference.sum(), full_query)[0][
        :, -spec.window :, 0
    ]

    direct_query = query_trajectory.detach().requires_grad_()
    direct, _ = extrapolation._exact_suffix_kernel_carrier(
        direct_query,
        context,
        route_kernel=route_kernel,
        window=spec.window,
        mismatch_scale=3.0,
    )
    direct_gradient = torch.autograd.grad(direct.sum(), direct_query)[0]

    torch.testing.assert_close(direct, reference, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        direct_gradient,
        reference_gradient,
        rtol=2e-12,
        atol=2e-12,
    )


def test_temporal_proxy_reaches_the_oldest_fault_beyond_window_one():
    spec = _spec(window=4, context_length=96)
    context = extrapolation.materialize_context(
        spec,
        96,
        device=torch.device("cpu"),
    )
    fault = torch.tensor(-0.25, requires_grad=True)
    output, _, _ = extrapolation.hard_forward_with_proxy(
        extrapolation.TEMPORAL_ESTIMATOR,
        spec,
        context,
        fault,
        logit_margin=1.0,
        scale=1.0,
        mismatch_scale=3.0,
        temporal_state_dim=8,
    )
    gradient = torch.autograd.grad(
        (output - spec.target_values).square().mean(),
        fault,
    )[0]

    assert torch.isfinite(gradient)
    assert gradient != 0.0


def test_quadratic_proxy_can_reach_the_fault_only_at_window_one():
    spec = _spec(window=1, context_length=96)
    context = extrapolation.materialize_context(
        spec,
        96,
        device=torch.device("cpu"),
    )
    fault = torch.tensor(-0.25, requires_grad=True)
    output, _, _ = extrapolation.hard_forward_with_proxy(
        "state_quadratic_attention",
        spec,
        context,
        fault,
        logit_margin=1.0,
        scale=1.0,
        mismatch_scale=3.0,
    )
    gradient = torch.autograd.grad(
        (output - spec.target_values).square().mean(),
        fault,
    )[0]

    assert torch.isfinite(gradient)
    assert gradient != 0.0


def test_small_matrix_report_is_strict_json_and_exposes_extrapolation():
    args = SimpleNamespace(
        device="cpu",
        estimators=list(extrapolation.ESTIMATORS),
        windows=[1, 2],
        seeds=[0],
        probes=2,
        bits=8,
        value_bits=8,
        value_mode="balanced_binary",
        train_context_length=32,
        eval_context_lengths=[32, 64],
        steps=8,
        learning_rate=0.05,
        initial_fault_logit=-0.25,
        logit_margin=1.0,
        scale=1.0,
        mismatch_scale=3.0,
        gradient_stall_tolerance=1e-14,
        chunk_size=17,
    )

    report = extrapolation.run_matrix(args)

    assert report["train_context_length"] == 32
    assert report["eval_context_lengths"] == [32, 64]
    assert len(report["records"]) == 6
    assert (
        report["summaries"]["state_quadratic_attention"]["2"][
            "train_success_fraction"
        ]
        == 0.0
    )
    json.dumps(report, allow_nan=False)
