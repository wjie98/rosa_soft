import math

import pytest
import torch

from rosa_soft import (
    ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
)
from rosa_soft.diagnostics import summarize_rosa_soft
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
        route_temperature=1.25,
        mismatch_penalty=4.5,
        generator=torch.Generator().manual_seed(1),
    )

    diagnostics = summarize_rosa_soft(
        inspection,
        top_k=3,
        quantile=0.9,
    )
    values = diagnostics.as_float_dict()

    assert diagnostics.route_temperature == 1.25
    assert diagnostics.mismatch_penalty == 4.5
    assert diagnostics.route_rows == 24
    assert diagnostics.competitive_route_count > 0
    assert diagnostics.hard_nonnull_route_fraction == torch.tensor(5 / 6)
    assert diagnostics.max_exact_suffix_length == torch.tensor(3.0)
    assert diagnostics.proxy_exact_length_error_mean == 0
    assert diagnostics.proxy_exact_length_error_quantile == 0
    assert 0 < diagnostics.selected_route_probability <= 1
    assert diagnostics.effective_route_count >= 1
    assert diagnostics.proxy_hard_route_agreement == 1
    assert set(values) == {
        "route_temperature",
        "mismatch_penalty",
        "route_rows",
        "competitive_route_count",
        "selected_route_probability",
        "effective_route_count",
        "proxy_exact_length_error_mean",
        "proxy_exact_length_error_quantile",
        "proxy_hard_route_agreement",
        "hard_nonnull_route_fraction",
        "max_exact_suffix_length",
    }
    assert all(isinstance(value, float) for value in values.values())


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
        generator=torch.Generator().manual_seed(3),
    )

    diagnostics = summarize_rosa_soft(inspection, top_k=8)

    assert diagnostics.competitive_route_count > 0
    assert diagnostics.proxy_exact_length_error_mean > 0
    assert diagnostics.proxy_exact_length_error_quantile >= diagnostics.proxy_exact_length_error_mean
    assert torch.isfinite(diagnostics.selected_route_probability)
    assert torch.isfinite(diagnostics.effective_route_count)


def test_singleton_has_no_competitive_nonnull_candidates():
    tensor = torch.ones(1, 1, 1, 1)
    _, inspection = inspect_rosa_soft(tensor, tensor, tensor)

    diagnostics = summarize_rosa_soft(inspection)

    assert diagnostics.route_rows == 1
    assert diagnostics.competitive_route_count == 0
    assert math.isnan(float(diagnostics.proxy_exact_length_error_mean))
    assert math.isnan(float(diagnostics.proxy_exact_length_error_quantile))
    assert diagnostics.selected_route_probability == 1
    assert diagnostics.effective_route_count == 1
    assert diagnostics.hard_nonnull_route_fraction == 0


def test_diagnostics_preserve_static_defaults():
    tensor = torch.ones(1, 3, 1, 1)
    _, inspection = inspect_rosa_soft(tensor, tensor, tensor)

    diagnostics = summarize_rosa_soft(inspection)

    assert diagnostics.route_temperature == ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE
    assert diagnostics.mismatch_penalty == ROSA_SOFT_DEFAULT_MISMATCH_PENALTY


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"top_k": 0}, "top_k"),
        ({"quantile": -0.1}, "quantile"),
        ({"quantile": 1.1}, "quantile"),
    ],
)
def test_diagnostics_validate_reduction_controls(kwargs, message):
    tensor = torch.ones(1, 3, 1, 1)
    _, inspection = inspect_rosa_soft(tensor, tensor, tensor)

    with pytest.raises(ValueError, match=message):
        summarize_rosa_soft(inspection, **kwargs)
