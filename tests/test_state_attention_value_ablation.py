import json

from benchmarks.state_attention_value_ablation import (
    build_parser,
    run_ablation,
)


def test_state_attention_value_ablation_smoke():
    args = build_parser().parse_args(
        [
            "--experiments",
            "binary_v4_proxy",
            "exp4_v4_selected",
            "rms_v4_selected",
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
    assert len(report["experiments"]) == 3
    for record in report["experiments"]:
        run = record["report"]["runs"][0]
        validation = run["candidates"]["state_quadratic_attention"][
            "validation"
        ]
        assert "value_representation_rms" in validation
        assert "value_storage_unique_code_fraction" in validation
    json.dumps(report, allow_nan=False)
