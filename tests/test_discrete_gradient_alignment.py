import importlib.util
import json
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_benchmark():
    name = "_discrete_gradient_alignment_benchmark"
    spec = importlib.util.spec_from_file_location(
        name,
        ROOT / "benchmarks/discrete_gradient_alignment.py",
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


alignment = _load_benchmark()


def test_bit_flip_oracle_matches_hand_computed_hard_differences():
    query = torch.ones(1, 3, 1, 1)
    key = torch.ones(1, 3, 1, 1)
    value = torch.tensor([[[[1.0]], [[-1.0]], [[1.0]]]])
    loss_weights = torch.ones(1, 3, 1, 1)
    original_query = query.clone()
    original_key = key.clone()

    oracle = alignment.discrete_bit_flip_oracle(
        query,
        key,
        value,
        loss_weights,
        max_suffix_length=1,
    )

    assert oracle.base_loss == 0
    torch.testing.assert_close(
        oracle.query_loss_deltas,
        torch.tensor([[[[0.0]], [[1.0]], [[-1.0]]]]),
        rtol=0,
        atol=0,
    )
    torch.testing.assert_close(
        oracle.key_loss_deltas,
        torch.tensor([[[[1.0]], [[-2.0]], [[0.0]]]]),
        rtol=0,
        atol=0,
    )
    assert torch.equal(query, original_query)
    assert torch.equal(key, original_key)


def test_measurement_is_deterministic_and_ignores_global_rng_state():
    kwargs = {
        "seed": 23,
        "seq_len": 4,
        "symbol_dim": 2,
        "max_suffix_length": 2,
        "scale": 0.5,
        "mismatch_scale": 3.0,
        "value_dim": 2,
        "top_k": 3,
    }

    torch.manual_seed(1)
    first = alignment.measure_configuration(**kwargs)
    torch.manual_seed(999)
    torch.randn(100)
    second = alignment.measure_configuration(**kwargs)

    assert first == second
    assert json.dumps(first, allow_nan=False)


def _assert_optional_fraction(value):
    assert value is None or 0.0 <= value <= 1.0


def _assert_alignment_bounds(summary, expected_candidates, requested_top_k):
    assert summary["candidate_count"] == expected_candidates
    assert 0 <= summary["surrogate_nonzero_count"] <= expected_candidates
    assert 0 <= summary["oracle_nonzero_count"] <= expected_candidates
    assert (
        0
        <= summary["oracle_useful_count"]
        <= summary["oracle_nonzero_count"]
    )
    assert (
        0
        <= summary["predicted_useful_count"]
        <= summary["surrogate_nonzero_count"]
    )
    assert (
        0
        <= summary["sign_agreement_count"]
        <= summary["sign_comparable_count"]
        <= min(
            summary["surrogate_nonzero_count"],
            summary["oracle_nonzero_count"],
        )
    )
    _assert_optional_fraction(summary["sign_agreement"])
    assert (
        summary["cosine_similarity"] is None
        or -1.0 <= summary["cosine_similarity"] <= 1.0
    )

    top_k = summary["top_k"]
    assert top_k["requested"] == requested_top_k
    assert (
        0
        <= top_k["evaluated_recommendations"]
        <= min(requested_top_k, summary["predicted_useful_count"])
    )
    assert (
        0
        <= top_k["oracle_useful_count"]
        <= top_k["evaluated_recommendations"]
    )
    _assert_optional_fraction(top_k["oracle_useful_fraction"])
    assert (
        top_k["mean_oracle_loss_delta"] is None
    ) == (top_k["evaluated_recommendations"] == 0)

    zero_support = summary["zero_support"]
    assert (
        zero_support["surrogate_zero_count"]
        == expected_candidates - summary["surrogate_nonzero_count"]
    )
    assert (
        0
        <= zero_support["missed_oracle_nonzero_count"]
        <= summary["oracle_nonzero_count"]
    )
    assert (
        0
        <= zero_support["missed_oracle_useful_count"]
        <= summary["oracle_useful_count"]
    )
    _assert_optional_fraction(zero_support["surrogate_zero_fraction"])
    _assert_optional_fraction(
        zero_support["missed_oracle_nonzero_fraction"]
    )
    _assert_optional_fraction(
        zero_support["missed_oracle_useful_fraction"]
    )


def test_alignment_metrics_have_basic_bounds_and_dense_candidate_counts():
    seq_len = 4
    symbol_dim = 2
    top_k = 3
    result = alignment.measure_configuration(
        seed=5,
        seq_len=seq_len,
        symbol_dim=symbol_dim,
        max_suffix_length=3,
        scale=2.0,
        mismatch_scale=9.0,
        value_dim=3,
        top_k=top_k,
    )

    role_candidates = seq_len * symbol_dim
    _assert_alignment_bounds(
        result["query"],
        role_candidates,
        top_k,
    )
    _assert_alignment_bounds(
        result["key"],
        role_candidates,
        top_k,
    )
    _assert_alignment_bounds(
        result["combined"],
        2 * role_candidates,
        top_k,
    )
    assert result["hard_forward_loss_difference"] == 0.0


def test_default_matrix_varies_every_requested_control():
    defaults = (
        alignment.DEFAULT_SEEDS,
        alignment.DEFAULT_SEQUENCE_LENGTHS,
        alignment.DEFAULT_SYMBOL_DIMS,
        alignment.DEFAULT_MAX_SUFFIX_LENGTHS,
        alignment.DEFAULT_SCALES,
        alignment.DEFAULT_MISMATCH_SCALES,
    )

    assert all(len(set(values)) > 1 for values in defaults)


def test_report_schema_uses_current_control_names():
    report = alignment.run_experiment(
        seeds=[7],
        sequence_lengths=[4],
        symbol_dims=[2],
        max_suffix_lengths=[1],
        scales=[1.0],
        mismatch_scales=[3.0],
        value_dim=2,
        top_k=2,
    )

    assert report["schema_version"] == 2
    assert report["matrix"]["scales"] == [1.0]
    assert report["matrix"]["mismatch_scales"] == [3.0]
    assert "mismatch_penalties" not in report["matrix"]
