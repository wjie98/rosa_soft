import math

import pytest
import torch

from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from rosa_soft.diagnostics import (
    summarize_rosa_soft,
    summarize_rosa_soft_gradients,
)
from rosa_soft.testing import inspect_rosa_soft


def test_diagnostics_summarize_detached_inspection():
    query = torch.ones(2, 6, 2, 3)
    key = torch.ones_like(query)
    value = torch.tensor(
        [[[[1.0, -1.0]] for _ in range(6)]] * 2,
    )
    _, inspection = inspect_rosa_soft(
        query,
        key,
        value,
        max_suffix_length=3,
        scale=1.25,
        mismatch_scale=4.5,
    )

    diagnostics = summarize_rosa_soft(
        inspection,
        top_k=3,
        quantile=0.9,
    )
    values = diagnostics.as_float_dict()

    assert diagnostics.scale == 1.25
    assert diagnostics.dropout_p == 0.0
    assert diagnostics.mismatch_scale == 4.5
    assert diagnostics.top_k == 3
    assert diagnostics.quantile == 0.9
    assert diagnostics.route_rows == 24
    assert diagnostics.competitive_route_count > 0
    assert diagnostics.hard_nonnull_route_fraction == torch.tensor(5 / 6)
    assert diagnostics.max_exact_suffix_length == torch.tensor(3.0)
    assert diagnostics.soft_hard_score_gap_mean == 0
    assert diagnostics.soft_hard_score_gap_quantile == 0
    assert 0 < diagnostics.hard_selected_route_probability_mean <= 1
    assert diagnostics.effective_route_count_mean >= 1
    assert diagnostics.soft_hard_route_agreement == 1
    assert diagnostics.hard_route_lag_mean == 0
    assert diagnostics.numerically_nonzero_route_fraction == 1
    assert diagnostics.query_positive_fraction == 1
    assert diagnostics.key_positive_fraction == 1
    assert diagnostics.query_bit_imbalance == 1
    assert diagnostics.key_bit_imbalance == 1
    assert diagnostics.query_sign_margin_mean == 1
    assert diagnostics.key_sign_margin_mean == 1
    assert diagnostics.query_softsign_derivative_mean == 0.25
    assert diagnostics.key_softsign_derivative_mean == 0.25
    assert set(values) == set(diagnostics.__dataclass_fields__)
    assert all(isinstance(value, float) for value in values.values())


def test_inspection_detaches_every_retained_tensor():
    query = torch.randn(1, 4, 1, 2, requires_grad=True)
    key = torch.randn(1, 4, 1, 2, requires_grad=True)
    value = torch.randn(1, 4, 1, 2, requires_grad=True)

    hard_output, inspection = inspect_rosa_soft(
        query,
        key,
        value,
    )
    retained_tensors = {
        "hard_output": hard_output,
        **{
            name: value
            for name, value in vars(inspection).items()
            if isinstance(value, torch.Tensor)
        },
    }

    assert set(retained_tensors) == {
        "hard_output",
        "query",
        "key",
        "exact_suffix_lengths",
        "soft_suffix_scores",
        "route_probabilities",
        "causal_route_mask",
        "selected_route_indices",
    }
    for tensor in retained_tensors.values():
        assert not tensor.requires_grad
        assert tensor.grad_fn is None


def test_diagnostics_report_approximation_error_for_mismatches():
    generator = torch.Generator().manual_seed(2)
    query = torch.randn(2, 9, 2, 4, generator=generator)
    key = torch.randn(2, 9, 2, 4, generator=generator)
    value = torch.randn(2, 9, 1, 3, generator=generator)
    _, inspection = inspect_rosa_soft(
        query,
        key,
        value,
        max_suffix_length=5,
    )

    diagnostics = summarize_rosa_soft(inspection, top_k=8)

    assert diagnostics.competitive_route_count > 0
    assert diagnostics.soft_hard_score_gap_mean > 0
    assert (
        diagnostics.soft_hard_score_gap_quantile
        >= diagnostics.soft_hard_score_gap_mean
    )
    assert torch.isfinite(
        diagnostics.hard_selected_route_probability_mean
    )
    assert torch.isfinite(diagnostics.effective_route_count_mean)
    assert 0 <= diagnostics.null_route_probability_mean <= 1
    assert 0 <= diagnostics.soft_route_entropy_mean
    assert 0 <= diagnostics.query_bit_imbalance <= 1
    assert 0 <= diagnostics.key_bit_imbalance <= 1


def test_singleton_has_no_competitive_nonnull_candidates():
    tensor = torch.ones(1, 1, 1, 1)
    _, inspection = inspect_rosa_soft(tensor, tensor, tensor)

    diagnostics = summarize_rosa_soft(inspection)

    assert diagnostics.route_rows == 1
    assert diagnostics.competitive_route_count == 0
    assert math.isnan(float(diagnostics.soft_hard_score_gap_mean))
    assert math.isnan(float(diagnostics.soft_hard_score_gap_quantile))
    assert diagnostics.hard_selected_route_probability_mean == 1
    assert diagnostics.effective_route_count_mean == 1
    assert diagnostics.hard_nonnull_route_fraction == 0
    assert math.isnan(float(diagnostics.hard_route_lag_mean))
    assert math.isnan(
        float(diagnostics.selected_suffix_length_candidate_count_mean)
    )
    assert math.isnan(
        float(diagnostics.selected_suffix_length_probability_mass_mean)
    )


def test_diagnostics_preserve_static_defaults():
    tensor = torch.ones(1, 3, 1, 1)
    _, inspection = inspect_rosa_soft(tensor, tensor, tensor)

    diagnostics = summarize_rosa_soft(inspection)

    assert diagnostics.scale == ROSA_SOFT_DEFAULT_SCALE
    assert diagnostics.mismatch_scale == ROSA_SOFT_DEFAULT_MISMATCH_SCALE
    assert diagnostics.top_k == 4
    assert diagnostics.quantile == 0.95


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"quantile": -0.1}, "quantile"),
        ({"quantile": 1.1}, "quantile"),
        ({"quantile": math.nan}, "quantile"),
        ({"quantile": math.inf}, "quantile"),
    ],
)
def test_diagnostics_validate_reduction_controls(kwargs, message):
    tensor = torch.ones(1, 3, 1, 1)
    _, inspection = inspect_rosa_soft(tensor, tensor, tensor)

    with pytest.raises(ValueError, match=message):
        summarize_rosa_soft(inspection, **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": True}, "top_k.*integer"),
        ({"top_k": 1.5}, "top_k.*integer"),
        ({"top_k": "2"}, "top_k.*integer"),
        ({"quantile": True}, "quantile.*real"),
        ({"quantile": "0.5"}, "quantile.*real"),
        ({"quantile": 0.5 + 0j}, "quantile.*real"),
    ],
)
def test_diagnostics_reject_non_integer_and_non_real_controls(
    kwargs,
    message,
):
    tensor = torch.ones(1, 3, 1, 1)
    _, inspection = inspect_rosa_soft(tensor, tensor, tensor)

    with pytest.raises(TypeError, match=message):
        summarize_rosa_soft(inspection, **kwargs)


def test_gradient_diagnostics_report_scale_support_and_feature_balance():
    grad_query = torch.tensor(
        [[[[1.0, 0.0], [3.0, 0.0]]]],
    )
    grad_key = torch.tensor(
        [[[[0.0, 2.0], [0.0, 4.0]]]],
    )
    grad_value = torch.ones(1, 1, 1, 2)

    diagnostics = summarize_rosa_soft_gradients(
        grad_query,
        grad_key,
        grad_value,
    )
    values = diagnostics.as_float_dict()

    assert diagnostics.query.l2_norm == torch.sqrt(torch.tensor(10.0))
    assert diagnostics.key.l2_norm == torch.sqrt(torch.tensor(20.0))
    assert diagnostics.value.l2_norm == torch.sqrt(torch.tensor(2.0))
    assert diagnostics.query.finite_fraction == 1
    assert diagnostics.key.finite_fraction == 1
    assert diagnostics.value.finite_fraction == 1
    assert diagnostics.query.nonzero_fraction == 0.5
    assert diagnostics.key.nonzero_fraction == 0.5
    assert diagnostics.value.nonzero_fraction == 1
    assert diagnostics.query.feature_rms_cv == 1
    assert diagnostics.key.feature_rms_cv == 1
    assert diagnostics.qk_to_value_norm_ratio == torch.sqrt(
        torch.tensor(15.0)
    )
    assert set(values) == {
        f"{role}_{metric}"
        for role in ("query", "key", "value")
        for metric in diagnostics.query.__dataclass_fields__
    } | {"qk_to_value_norm_ratio"}


def test_gradient_diagnostics_surface_nonfinite_and_zero_value():
    diagnostics = summarize_rosa_soft_gradients(
        torch.tensor([1.0, float("nan")]),
        torch.tensor([float("inf"), 0.0]),
        torch.zeros(2),
    )

    assert diagnostics.query.finite_fraction == 0.5
    assert diagnostics.key.finite_fraction == 0.5
    assert diagnostics.value.l2_norm == 0
    assert math.isinf(float(diagnostics.qk_to_value_norm_ratio))
