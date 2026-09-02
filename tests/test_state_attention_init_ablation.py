import json

from benchmarks.state_attention_init_ablation import (
    build_parser,
    run_ablation,
)


def test_initialization_ablation_smoke_schema():
    args = build_parser().parse_args(
        [
            "--experiments",
            "E0_default_dv4",
            "E3_shared_dv8",
            "--estimators",
            "state_quadratic_attention",
            "--model-seeds",
            "0",
            "--data-seeds",
            "0",
            "--train-pairs",
            "1",
            "--validation-pairs",
            "1",
            "--associations",
            "2",
            "--payload-bits",
            "2",
            "--hidden-size",
            "16",
            "--context-depth",
            "1",
            "--heads",
            "1",
            "--qk-bits",
            "8",
            "--value-heads",
            "1",
            "--qk-shared-bits",
            "4",
            "--steps",
            "1",
        ]
    )

    report = run_ablation(args)

    assert report["schema_version"] == 1
    assert [record["experiment"] for record in report["experiments"]] == [
        "E0_default_dv4",
        "E3_shared_dv8",
    ]
    assert report["experiments"][0]["report"]["payload_bits"] == 2
    assert report["experiments"][1]["report"]["value_dim"] == 8
    assert all(
        record["report"]["initialization"]["controlled_module_reset"]
        for record in report["experiments"]
    )
    json.dumps(report, allow_nan=False)


def test_control_experiments_isolate_value_geometry_and_qk_correlation():
    args = build_parser().parse_args(
        [
            "--experiments",
            "E6_shared_dv4_orthogonal",
            "E7_independent_qk_dv4",
            "--estimators",
            "state_quadratic_attention",
            "--model-seeds",
            "0",
            "--data-seeds",
            "0",
            "--train-pairs",
            "1",
            "--validation-pairs",
            "1",
            "--associations",
            "2",
            "--payload-bits",
            "2",
            "--hidden-size",
            "16",
            "--context-depth",
            "1",
            "--heads",
            "1",
            "--qk-bits",
            "8",
            "--value-heads",
            "1",
            "--qk-shared-bits",
            "4",
            "--steps",
            "1",
        ]
    )

    report = run_ablation(args)
    value_control, qk_control = report["experiments"]

    assert value_control["config"]["value_init"] == "orthogonal"
    assert value_control["report"]["value_dim"] == 4
    assert qk_control["config"]["qk_correlation"] == 0.0
    assert qk_control["report"]["initialization"]["qk_correlation"] == 0.0
