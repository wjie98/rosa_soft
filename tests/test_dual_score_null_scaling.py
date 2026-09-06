import json

import pytest
import torch

from benchmarks import dual_score_null_scaling as null_scaling


def test_small_null_scaling_benchmark_is_complete_and_strict_json():
    args = null_scaling.build_parser().parse_args(
        [
            "--candidate-counts",
            "16",
            "--symbol-dims",
            "2",
            "--suffix-length",
            "4",
            "--probes",
            "2",
            "--seeds",
            "3",
        ]
    )
    report = null_scaling.run_benchmark(args)

    assert len(report["records"]) == len(null_scaling.SCORE_MODES)
    assert len(report["summary"]) == len(null_scaling.SCORE_MODES)
    assert all(
        0.0 < row["mean_nonnull_mass"] < 1.0
        for row in report["records"]
    )
    json.dumps(report, allow_nan=False)


def test_dual_nonnull_mass_is_probability_mixture():
    common = {
        "candidate_count": 32,
        "symbol_dim": 4,
        "suffix_length": 5,
        "probes": 3,
        "seed": 7,
        "evidence_power": 0.25,
        "information_weight": 0.3,
        "device": torch.device("cpu"),
    }
    masses = {
        mode: null_scaling.measure_condition(score_mode=mode, **common)[
            "mean_nonnull_mass"
        ]
        for mode in null_scaling.SCORE_MODES
    }

    assert masses["dual"] == pytest.approx(
        0.7 * masses["discovery"] + 0.3 * masses["information"],
        rel=2e-6,
        abs=2e-6,
    )
