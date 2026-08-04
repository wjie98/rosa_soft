import itertools
import math
from argparse import Namespace

import pytest
import torch

from benchmarks import null_calibration_ablation as ablation


@pytest.mark.parametrize("bits", [1, 2, 4, 8])
def test_null_local_gate_mean_matches_exhaustive_bits(bits):
    mismatch_scale = 3.0
    gates = []
    for symbols in itertools.product((0, 1), repeat=bits):
        mismatch_count = sum(symbols)
        gates.append(math.exp(-mismatch_scale * mismatch_count / bits))

    assert ablation.null_local_gate_mean(bits, mismatch_scale) == pytest.approx(
        sum(gates) / len(gates),
        rel=1e-14,
    )


def test_planted_one_token_likelihood_ratio_is_exactly_normalized_gate():
    gate_mean = ablation.null_local_gate_mean(4, 3.0)
    planted = ablation.planted_route_state(
        qk_bits=4,
        max_suffix_length=1,
        planted_suffix_length=1,
        mismatch_scale=3.0,
        scale=1.0,
    )

    assert planted["raw_suffix_score"] == pytest.approx(1.0)
    assert planted["route_score"] == pytest.approx(1.0)
    assert planted["route_log_likelihood_ratio"] == pytest.approx(
        -math.log(gate_mean)
    )


def test_candidate_correction_keeps_random_null_mass_independent_of_count():
    partition = {"mean_route_weight": 2.0}
    small = ablation.calibration_record(
        qk_bits=4,
        max_suffix_length=8,
        planted_suffix_length=4,
        candidate_count=1_000,
        mismatch_scale=3.0,
        scale=1.0,
        null_partition=partition,
    )
    large = ablation.calibration_record(
        qk_bits=4,
        max_suffix_length=8,
        planted_suffix_length=4,
        candidate_count=1_000_000,
        mismatch_scale=3.0,
        scale=1.0,
        null_partition=partition,
    )

    assert small["current"][
        "background_recall_probability"
    ] == pytest.approx(
        large["current"]["background_recall_probability"]
    )
    assert small["moment_calibrated_sqrt"][
        "background_recall_probability"
    ] == pytest.approx(0.5)
    assert large["no_candidate_correction"][
        "background_recall_probability"
    ] > small["no_candidate_correction"][
        "background_recall_probability"
    ]


def test_collision_likelihood_ratio_uses_bit_capacity():
    partition = {"mean_route_weight": 2.0}
    low_bits = ablation.calibration_record(
        qk_bits=1,
        max_suffix_length=8,
        planted_suffix_length=8,
        candidate_count=1_000,
        mismatch_scale=3.0,
        scale=1.0,
        null_partition=partition,
    )
    high_bits = ablation.calibration_record(
        qk_bits=8,
        max_suffix_length=8,
        planted_suffix_length=8,
        candidate_count=1_000,
        mismatch_scale=3.0,
        scale=1.0,
        null_partition=partition,
    )

    assert high_bits["collision_lr_joint"][
        "effective_candidate_capacity"
    ] > low_bits["collision_lr_joint"]["effective_candidate_capacity"]
    assert high_bits["current"]["effective_candidate_capacity"] == pytest.approx(
        low_bits["current"]["effective_candidate_capacity"]
    )


def test_small_report_has_every_requested_cell():
    args = Namespace(
        device="cpu",
        qk_bits=[2],
        windows=[2],
        planted_lengths=[1, 2, 4],
        candidate_counts=[10, 100],
        mismatch_scale=3.0,
        scale=1.0,
        sample_count=1024,
        chunk_size=256,
        seed=17,
    )

    report = ablation.run_ablation(args)

    assert len(report["partitions"]) == 1
    assert len(report["records"]) == 4
    partition = report["partitions"]["D2:W2"]
    assert partition["mean_route_weight"] > 0
    assert partition["route_weight_standard_error"] >= 0
