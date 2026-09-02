import json

from benchmarks.state_attention_estimator_vjp import (
    build_parser,
    run_benchmark,
)


def test_state_attention_estimator_vjp_smoke():
    args = build_parser().parse_args(
        [
            "--cases",
            "2:5",
            "--heads",
            "1",
            "--qk-bits",
            "2",
            "--value-heads",
            "1",
            "--value-bits",
            "2",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ]
    )

    report = run_benchmark(args)

    assert report["schema_version"] == 1
    assert len(report["cases"]) == 1
    measurements = report["cases"][0]["measurements"]
    assert {row["estimator"] for row in measurements} == set(args.estimators)
    assert all(row["hard_forward_equal"] for row in measurements)
    assert all(row["vjp_ms"] > 0.0 for row in measurements)
    json.dumps(report, allow_nan=False)
