import json

import pytest

from benchmarks.top_two_value_ablation import build_parser, run_benchmark


def _smoke_args():
    return build_parser().parse_args(
        [
            "--sections",
            "gradient",
            "fit",
            "context",
            "--gradient-estimators",
            "production_full",
            "split_dense",
            "selected_value",
            "winner_rest",
            "hard_winner_rest",
            "no_prior",
            "hard_null_gate",
            "top2_qk",
            "--gradient-seeds",
            "0",
            "--gradient-lengths",
            "4",
            "--gradient-bits",
            "2",
            "--fit-estimators",
            "production_full",
            "split_dense",
            "selected_value",
            "winner_rest",
            "hard_winner_rest",
            "no_prior",
            "hard_null_gate",
            "top2_qk",
            "--fit-seeds",
            "0",
            "--fit-steps",
            "1",
            "--context-estimators",
            "production_full",
            "split_dense",
            "selected_value",
            "winner_rest",
            "hard_winner_rest",
            "no_prior",
            "hard_null_gate",
            "top2_qk",
            "--context-seeds",
            "0",
            "--context-steps",
            "1",
            "--context-train-pairs",
            "2",
            "--context-validation-pairs",
            "2",
        ]
    )


def test_ablation_smoke_schema_and_qk_equivalent_controls():
    report = run_benchmark(_smoke_args())

    assert report["schema_version"] == 1
    assert set(report["sections"]) == {"gradient", "fit", "context"}
    json.dumps(report, allow_nan=False)

    gradient = report["sections"]["gradient"]["summary"]
    for estimator in ("split_dense", "selected_value"):
        assert gradient[estimator]["mean_cosine"] == pytest.approx(
            gradient["production_full"]["mean_cosine"], abs=1e-7
        )
        for metric in gradient["production_full"]:
            if metric != "mean_cosine":
                assert gradient[estimator][metric] == gradient["production_full"][
                    metric
                ]
    assert (
        gradient["top2_qk"]["missed_oracle_useful_fraction"]
        >= gradient["split_dense"]["missed_oracle_useful_fraction"]
    )


def test_every_training_section_contains_each_requested_estimator():
    args = _smoke_args()
    report = run_benchmark(args)

    for section, estimators in (
        ("fit", args.fit_estimators),
        ("context", args.context_estimators),
    ):
        assert set(report["sections"][section]["summary"]) == set(estimators)
