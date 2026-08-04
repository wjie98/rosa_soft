import pytest
import torch

from benchmarks.pretraining_codebook import (
    evaluate_codebook,
    synthetic_snapshot,
)


def test_aligned_unique_codebook_routes_every_concept():
    query, key, labels = synthetic_snapshot(
        seed=1,
        concepts=16,
        trajectory_length=4,
        bits=4,
        mode="aligned",
        corruption_p=0.0,
        active_bits=1,
        labels=16,
    )
    metrics = evaluate_codebook(query, key, labels)
    full = metrics.horizons[-1]

    assert metrics.bit_agreement == 1.0
    assert full.query_key_trajectory_alignment == 1.0
    assert full.route_accuracy == 1.0
    assert full.strict_route_accuracy == 1.0
    assert full.conflicting_states == 0
    assert full.continuation_conditional_entropy_bits == 0.0


def test_collapsed_codebook_reports_incompatible_continuations():
    query, key, labels = synthetic_snapshot(
        seed=2,
        concepts=32,
        trajectory_length=4,
        bits=4,
        mode="collapsed",
        corruption_p=0.0,
        active_bits=1,
        labels=32,
    )
    metrics = evaluate_codebook(query, key, labels)
    full = metrics.horizons[-1]

    assert metrics.dead_bit_fraction >= 0.75
    assert full.key_states < 32
    assert full.conflicting_states > 0
    assert full.conflicting_sample_fraction > 0.0
    assert full.continuation_conditional_entropy_bits > 0.0
    assert full.strict_route_accuracy < 1.0


def test_role_drift_breaks_qk_alignment_without_changing_key_entropy():
    aligned = synthetic_snapshot(
        seed=3,
        concepts=16,
        trajectory_length=4,
        bits=4,
        mode="aligned",
        corruption_p=0.0,
        active_bits=1,
        labels=16,
    )
    drifted = synthetic_snapshot(
        seed=3,
        concepts=16,
        trajectory_length=4,
        bits=4,
        mode="role_drift",
        corruption_p=0.5,
        active_bits=1,
        labels=16,
    )
    aligned_metrics = evaluate_codebook(*aligned)
    drifted_metrics = evaluate_codebook(*drifted)

    assert drifted_metrics.bit_agreement < aligned_metrics.bit_agreement
    assert drifted_metrics.mean_bit_entropy == pytest.approx(
        aligned_metrics.mean_bit_entropy
    )
    assert drifted_metrics.horizons[-1].route_accuracy < 1.0


def test_corruption_exposes_horizon_specific_failure():
    query, key, labels = synthetic_snapshot(
        seed=4,
        concepts=32,
        trajectory_length=8,
        bits=2,
        mode="corrupted",
        corruption_p=0.1,
        active_bits=1,
        labels=32,
    )
    metrics = evaluate_codebook(query, key, labels)

    assert len(metrics.horizons) == 8
    assert metrics.horizons[-1].query_key_trajectory_alignment <= (
        metrics.horizons[0].query_key_trajectory_alignment
    )
    assert all(
        0.0 <= horizon.route_accuracy <= 1.0
        for horizon in metrics.horizons
    )


def test_codebook_input_validation():
    with pytest.raises(ValueError, match="share shape"):
        evaluate_codebook(
            torch.ones(2, 3, 4),
            torch.ones(2, 3, 5),
            torch.arange(2),
        )
