import json

import torch

from benchmarks.contextual_estimator_recall import (
    EstimatorResetRnnRosaLM,
    build_parser,
    run_benchmark,
)


def _model(estimator):
    return EstimatorResetRnnRosaLM(
        associations=2,
        hidden_size=8,
        num_heads=1,
        qk_bits=2,
        value_heads=1,
        value_bits=2,
        context_scale=0.25,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
        estimator=estimator,
        bit_temperature=0.5,
        antithetic_pairs=2,
    )


def test_research_estimators_preserve_one_hard_forward():
    generator = torch.Generator().manual_seed(29)
    query = torch.randn(2, 5, 1, 2, generator=generator)
    key = torch.randn(2, 5, 1, 2, generator=generator)
    value = torch.randn(2, 5, 1, 2, generator=generator)
    expected = _model("production")._routed_values(
        query, key, value, "rosa"
    )

    for estimator in ("mean_field", "arm", "disarm"):
        inputs = [tensor.detach().clone().requires_grad_() for tensor in (
            query,
            key,
            value,
        )]
        torch.manual_seed(31)
        actual = _model(estimator)._routed_values(*inputs, "rosa")
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.sum().backward()
        assert all(tensor.grad is not None for tensor in inputs)


def test_contextual_estimator_benchmark_smoke_schema():
    args = build_parser().parse_args(
        [
            "--estimators",
            "production",
            "mean_field",
            "arm",
            "disarm",
            "--seeds",
            "0",
            "--train-pairs",
            "2",
            "--validation-pairs",
            "2",
            "--associations",
            "2",
            "--hidden-size",
            "8",
            "--heads",
            "1",
            "--qk-bits",
            "2",
            "--value-heads",
            "1",
            "--value-bits",
            "2",
            "--steps",
            "1",
            "--baseline-steps",
            "0",
            "--antithetic-pairs",
            "2",
        ]
    )

    report = run_benchmark(args)

    assert report["schema_version"] == 1
    json.dumps(report, allow_nan=False)
    assert set(report["summary"]) == set(args.estimators)
    assert report["shortcut_checks_passed"] is True
    run = report["runs"][0]
    assert run["residual_only_baseline"]["training_loss"] is None
    assert set(run["candidates"]) == set(args.estimators)
    assert all(
        candidate["validation"]["query_residual_max_difference"] == 0.0
        for candidate in run["candidates"].values()
    )
