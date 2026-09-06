import json

from benchmarks import dual_score_ablation as ablation


def test_small_ablation_matrix_is_complete_and_strict_json():
    args = ablation.build_parser().parse_args(
        [
            "--seeds",
            "3",
            "--sequence-lengths",
            "4",
            "--symbol-dims",
            "2",
            "--capacity-symbol-dims",
            "2",
            "--candidate-counts",
            "8192",
            "--suffix-lengths",
            "1",
            "4",
        ]
    )
    report = ablation.run_ablation(args)

    assert len(report["records"]) == len(ablation.ESTIMATORS)
    assert set(report["summary"]) == set(ablation.ESTIMATORS)
    assert len(report["capacity"]) == 2
    assert all(row["hard_loss_difference"] == 0.0 for row in report["records"])
    json.dumps(report, allow_nan=False)


def test_information_capacity_uses_symbol_width_while_sqrt_does_not():
    rows = ablation.exact_score_capacity(
        symbol_dims=[2, 8],
        candidate_counts=[1_000_000],
        suffix_lengths=[8],
        mismatch_scale=3.0,
        scale=1.0,
    )
    low, high = rows

    assert low["sqrt_logit_after_prior"] == high["sqrt_logit_after_prior"]
    assert (
        high["information_logit_after_prior"]
        > low["information_logit_after_prior"]
    )


def test_information_capacity_uses_requested_evidence_power():
    common = {
        "symbol_dims": [8],
        "candidate_counts": [1_000_000],
        "suffix_lengths": [8],
        "mismatch_scale": 3.0,
        "scale": 1.0,
    }
    low = ablation.exact_score_capacity(**common, evidence_power=0.25)[0]
    high = ablation.exact_score_capacity(**common, evidence_power=1.0)[0]

    assert low["evidence_power"] == 0.25
    assert high["evidence_power"] == 1.0
    assert high["information_logit_after_prior"] > low[
        "information_logit_after_prior"
    ]
