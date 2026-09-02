import pytest
import torch

from benchmarks.training_hard_fusion import (
    ambiguity_stats,
    dual_accumulator_softmax_vjp,
    suffix_length_matrix,
)


def test_suffix_length_matrix_matches_hand_computed_latest_ties():
    query = torch.tensor([1, 2, 1, 2, 1])
    key = torch.tensor([1, 2, 1, 2, 1])
    lengths = suffix_length_matrix(query, key)
    assert lengths[4].tolist() == [1, 0, 3, 0, 0]


def test_bounded_ambiguity_identifies_when_capped_route_is_not_exact():
    query = torch.tensor([0, 0, 0, 1])
    key = torch.tensor([0, 1, 1, 0])
    narrow = ambiguity_stats(query, key, window=1)
    wide = ambiguity_stats(query, key, window=4)
    assert narrow.fallback_rows > 0
    assert narrow.continuation_candidates > 0
    assert narrow.capped_route_errors > 0
    assert wide.fallback_rows == 0
    assert wide.capped_route_errors == 0


@pytest.mark.parametrize("routes", [2, 5, 17])
@pytest.mark.parametrize("parameters", [1, 4, 11])
def test_dual_accumulator_vjp_matches_autograd(routes, parameters):
    generator = torch.Generator().manual_seed(routes * 31 + parameters)
    features = torch.randn(routes, parameters, generator=generator)
    utility = torch.randn(routes, generator=generator)
    parameter = torch.randn(parameters, generator=generator, requires_grad=True)
    logits = features @ parameter
    expected = (logits.softmax(dim=0) * utility).sum()
    expected.backward()

    actual, accumulator_a, accumulator_b, expected_utility = (
        dual_accumulator_softmax_vjp(
            logits.detach(),
            utility,
            features,
        )
    )
    assert torch.allclose(actual, parameter.grad, atol=1e-6, rtol=1e-6)
    assert torch.allclose(
        actual,
        accumulator_a - expected_utility * accumulator_b,
    )
