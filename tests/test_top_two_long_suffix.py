import json

import pytest
import torch

from benchmarks import long_suffix_extrapolation
from benchmarks import top_two_long_suffix


def _case(window=4, context_length=48):
    spec = long_suffix_extrapolation.make_gate_spec(
        seed=3,
        probes=3,
        window=window,
        bits=8,
        train_context_length=context_length,
    )
    context = long_suffix_extrapolation.materialize_context(
        spec, context_length, device=torch.device("cpu")
    )
    return spec, context


@pytest.mark.parametrize(
    "estimator", top_two_long_suffix.ESTIMATORS
)
def test_all_estimators_preserve_the_same_exact_hard_forward(estimator):
    spec, context = _case()
    fault = torch.tensor(-0.25, requires_grad=True)
    output, selected, _ = top_two_long_suffix.hard_forward_with_proxy(
        estimator,
        spec,
        context,
        fault,
        logit_margin=1.0,
        scale=1.0,
        mismatch_scale=3.0,
    )
    expected = long_suffix_extrapolation.dense_hard_route(
        long_suffix_extrapolation.make_query_logits(
            spec, fault.detach(), logit_margin=1.0
        ),
        context,
        spec.window,
    )

    assert torch.equal(output.detach(), expected.output)
    assert torch.equal(selected, expected.selected_positions)
    gradient = torch.autograd.grad(output.square().mean(), fault)[0]
    assert torch.isfinite(gradient)


def test_final_row_production_matches_existing_reference_carrier_and_vjp():
    spec, context = _case()
    fault = torch.tensor(-0.25, dtype=torch.float64, requires_grad=True)
    context = long_suffix_extrapolation.DenseContext(
        key_signs=context.key_signs.to(torch.float64),
        values=context.values.to(torch.float64),
        target_positions=context.target_positions,
    )
    query = long_suffix_extrapolation.make_query_logits(
        spec, fault, logit_margin=1.0
    )
    expected, _ = long_suffix_extrapolation._production_suffix_carrier(
        query,
        context,
        window=spec.window,
        scale=1.0,
        mismatch_scale=3.0,
    )
    actual, _ = top_two_long_suffix._competition_carrier(
        "production_full",
        query,
        context,
        torch.zeros(spec.probes, dtype=torch.int64),
        window=spec.window,
        scale=1.0,
        mismatch_scale=3.0,
    )
    grad_output = torch.randn_like(expected)
    expected_gradient = torch.autograd.grad(
        expected, fault, grad_output, retain_graph=True
    )[0]
    actual_gradient = torch.autograd.grad(actual, fault, grad_output)[0]

    torch.testing.assert_close(actual, expected, rtol=2e-15, atol=2e-15)
    torch.testing.assert_close(
        actual_gradient, expected_gradient, rtol=2e-14, atol=2e-14
    )


def test_tiny_matrix_is_json_serializable():
    args = top_two_long_suffix.build_parser().parse_args(
        [
            "--estimators",
            "production_full",
            "top2_qk",
            "--windows",
            "2",
            "--seeds",
            "0",
            "--probes",
            "2",
            "--train-context-length",
            "12",
            "--eval-context-lengths",
            "12",
            "24",
            "--steps",
            "2",
            "--chunk-size",
            "7",
        ]
    )
    report = top_two_long_suffix.run_matrix(args)

    assert set(report["summary"]) == {"production_full", "top2_qk"}
    json.dumps(report, allow_nan=False)
