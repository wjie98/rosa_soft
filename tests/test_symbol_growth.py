import pytest
import torch

from benchmarks.symbol_growth import (
    DormantBitProjection,
    GrowthRouter,
    collect_conflict_stats,
    growth_loss,
    make_growth_batch,
)


def test_conflict_stats_measure_only_states_with_different_labels():
    logits = torch.tensor(
        [
            [-1.0, -1.0],
            [-2.0, -3.0],
            [1.0, -1.0],
            [2.0, -4.0],
        ]
    )
    labels = torch.tensor([0, 1, 2, 2])
    stats = collect_conflict_stats(logits, labels)

    assert stats.states == 2
    assert stats.conflicting_states == 1
    assert stats.conflicting_samples == 2
    assert stats.conflicting_pairs == 1
    assert stats.max_state_size == 2
    assert stats.conditional_entropy_bits == 0.5


def test_conflict_stats_reject_an_empty_codebook():
    with pytest.raises(ValueError, match="at least one"):
        collect_conflict_stats(torch.empty(0, 2), torch.empty(0, dtype=torch.long))


def test_dormant_split_activates_one_bit_and_reduces_conflicts():
    torch.manual_seed(3)
    features = torch.eye(8)
    labels = torch.arange(8)
    projection = DormantBitProjection(
        feature_size=8,
        max_bits=4,
        active_bits=1,
    )
    _, initial_key = projection(features)
    before = collect_conflict_stats(initial_key[:, 0], labels)
    event = projection.activate_split(features, labels)
    _, final_key = projection(features)
    after = collect_conflict_stats(final_key[:, 0], labels)

    assert projection.active_bits == 2
    assert event["activated_bit"] == 1
    assert event["split_fit_accuracy"] == 1.0
    assert after.conflicting_pairs < before.conflicting_pairs


def test_growth_batch_has_only_strict_historical_payload_routes():
    batch = make_growth_batch(8)

    assert torch.all(batch.expected_routes < batch.query_positions)
    assert torch.equal(
        batch.value[0, batch.expected_routes, 0],
        batch.target[0],
    )
    assert torch.all(batch.type_symbol[0, 0::2][:8] == 1)


def test_inactive_projection_rows_receive_no_gradient():
    batch = make_growth_batch(6)
    torch.manual_seed(5)
    model = GrowthRouter(
        candidate_count=6,
        max_bits=5,
        active_bits=2,
        aligned_head_init=False,
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        operator="reference",
    )
    loss = growth_loss(model(batch), batch.target)
    loss.backward()

    assert torch.count_nonzero(model.projection.query_weight.grad[:2]) > 0
    assert torch.count_nonzero(model.projection.key_weight.grad[:2]) > 0
    assert torch.count_nonzero(model.projection.query_weight.grad[2:]) == 0
    assert torch.count_nonzero(model.projection.key_weight.grad[2:]) == 0
