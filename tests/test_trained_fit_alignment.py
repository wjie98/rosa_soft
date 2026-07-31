import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    name = "_trained_fit_alignment_benchmark"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "benchmarks/trained_fit_alignment.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


trained_alignment = _load_benchmark()


def test_trained_alignment_reports_hard_and_parameter_diagnostics():
    args = trained_alignment.build_parser().parse_args(
        [
            "--model-seeds",
            "1",
            "--steps",
            "1",
            "--sequence-length",
            "8",
            "--heads",
            "1",
            "--qk-bits",
            "2",
            "--value-heads",
            "1",
            "--value-bits",
            "2",
            "--max-suffix-length",
            "4",
            "--top-k",
            "2",
        ]
    )

    report = trained_alignment.run_experiment(args)
    run = report["runs"][0]

    assert report["schema_version"] == 1
    assert report["target_mode"] == "any-candidate"
    assert run["fit_target_count"] > 0
    assert run["combined_alignment"]["candidate_count"] == 36
    assert run["parameter_alignment"]["production_l2_norm"] >= 0
    assert run["parameter_alignment"]["bitflip_l2_norm"] >= 0
    assert run["diagnostic_dropout_p"] == 0.0
    assert json.dumps(report, allow_nan=False)


def test_trained_alignment_accepts_strict_target_mode():
    args = trained_alignment.build_parser().parse_args(
        [
            "--model-seeds",
            "0",
            "--steps",
            "0",
            "--sequence-length",
            "8",
            "--target-mode",
            "strict-longest-latest",
            "--top-k",
            "2",
        ]
    )

    report = trained_alignment.run_experiment(args)

    assert report["target_mode"] == "strict-longest-latest"
