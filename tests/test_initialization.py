import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from rosa_soft.initialization import (
    RosaProjectionInit,
    initialize_orthogonal_value_,
    initialize_partial_shared_qk_,
    initialize_rosa_projections_,
)


def _projections(*, bias=True):
    return (
        nn.Linear(32, 16, bias=bias),
        nn.Linear(32, 16, bias=bias),
        nn.Linear(32, 8, bias=bias),
    )


def test_combined_initializer_is_deterministic_and_does_not_tie_parameters():
    first = _projections()
    second = _projections()
    config = RosaProjectionInit(
        num_heads=2,
        qk_bits=8,
        num_value_heads=1,
        value_bits=8,
        shared_qk_bits=4,
        seed=19,
    )

    initialize_rosa_projections_(*first, config)
    initialize_rosa_projections_(*second, config)

    for left, right in zip(first, second):
        torch.testing.assert_close(left.weight, right.weight, rtol=0, atol=0)
        assert left.weight.data_ptr() != right.weight.data_ptr()
        assert torch.count_nonzero(left.bias) == 0
    assert first[0].weight.data_ptr() != first[1].weight.data_ptr()


def test_partial_shared_qk_has_requested_head_local_geometry():
    query, key, _ = _projections(bias=False)
    initialize_partial_shared_qk_(
        query,
        key,
        num_heads=2,
        qk_bits=8,
        shared_qk_bits=4,
        correlation=0.83,
        logit_median=0.6,
        seed=23,
    )
    query_heads = query.weight.view(2, 8, 32)
    key_heads = key.weight.view(2, 8, 32)

    shared_cosine = F.cosine_similarity(
        query_heads[:, :4],
        key_heads[:, :4],
        dim=-1,
    )
    private_cosine = F.cosine_similarity(
        query_heads[:, 4:],
        key_heads[:, 4:],
        dim=-1,
    )
    expected_norm = 0.6 / 0.6744897501960817

    torch.testing.assert_close(
        shared_cosine,
        torch.full_like(shared_cosine, 0.83),
        rtol=1e-6,
        atol=1e-6,
    )
    torch.testing.assert_close(
        private_cosine,
        torch.zeros_like(private_cosine),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        query_heads.norm(dim=-1),
        torch.full((2, 8), expected_norm),
        rtol=1e-6,
        atol=1e-6,
    )
    assert not torch.equal(query_heads[0], query_heads[1])


def test_single_qk_bit_uses_correlated_independent_parameters():
    query = nn.Linear(8, 2, bias=False)
    key = nn.Linear(8, 2, bias=False)
    initialize_partial_shared_qk_(
        query,
        key,
        num_heads=2,
        qk_bits=1,
        correlation=0.71,
        seed=27,
    )

    cosine = F.cosine_similarity(
        query.weight.view(2, 1, 8),
        key.weight.view(2, 1, 8),
        dim=-1,
    )
    torch.testing.assert_close(
        cosine,
        torch.full_like(cosine, 0.71),
        rtol=1e-6,
        atol=1e-6,
    )
    assert query.weight.data_ptr() != key.weight.data_ptr()


def test_zero_shared_bits_produces_orthogonal_qk_rows():
    query = nn.Linear(8, 4, bias=False)
    key = nn.Linear(8, 4, bias=False)
    initialize_partial_shared_qk_(
        query,
        key,
        num_heads=1,
        qk_bits=4,
        shared_qk_bits=0,
        seed=31,
    )

    cosine = F.cosine_similarity(query.weight, key.weight, dim=-1)
    torch.testing.assert_close(
        cosine,
        torch.zeros_like(cosine),
        rtol=0,
        atol=1e-6,
    )


def test_orthogonal_value_is_independent_per_head():
    value = nn.Linear(24, 12, bias=False)
    initialize_orthogonal_value_(
        value,
        num_value_heads=3,
        value_bits=4,
        seed=29,
    )
    heads = value.weight.view(3, 4, 24)
    gram = heads @ heads.transpose(-1, -2)
    expected_diagonal = (0.6 / 0.6744897501960817) ** 2

    torch.testing.assert_close(
        gram,
        torch.eye(4).expand(3, -1, -1) * expected_diagonal,
        rtol=1e-5,
        atol=1e-6,
    )
    assert not torch.equal(heads[0], heads[1])


@pytest.mark.parametrize(
    ("operation", "message"),
    [
        (
            lambda: initialize_partial_shared_qk_(
                nn.Linear(8, 8),
                nn.Linear(8, 8),
                num_heads=1,
                qk_bits=8,
                shared_qk_bits=4,
            ),
            "2 \\* qk_bits",
        ),
        (
            lambda: initialize_partial_shared_qk_(
                nn.Linear(32, 8),
                nn.Linear(32, 8),
                num_heads=1,
                qk_bits=8,
                shared_qk_bits=9,
            ),
            "shared_qk_bits",
        ),
        (
            lambda: initialize_orthogonal_value_(
                nn.Linear(4, 8),
                num_value_heads=1,
                value_bits=8,
            ),
            "value_bits <= input width",
        ),
    ],
)
def test_invalid_projection_geometry_is_rejected(operation, message):
    with pytest.raises(ValueError, match=message):
        operation()


def test_seed_change_changes_basis_without_changing_static_scale():
    first = nn.Linear(32, 8, bias=False)
    second = nn.Linear(32, 8, bias=False)
    initialize_orthogonal_value_(
        first,
        num_value_heads=1,
        value_bits=8,
        seed=1,
    )
    initialize_orthogonal_value_(
        second,
        num_value_heads=1,
        value_bits=8,
        seed=2,
    )

    assert not torch.equal(first.weight, second.weight)
    assert math.isclose(
        float(first.weight.detach().norm()),
        float(second.weight.detach().norm()),
        rel_tol=1e-6,
    )
