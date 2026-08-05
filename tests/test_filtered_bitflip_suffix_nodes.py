import itertools

import pytest
import torch

from benchmarks.filtered_bitflip import (
    brute_force_bitflip,
    unlimited_hard_routes_diagonal,
)
from benchmarks.filtered_bitflip_suffix_nodes import (
    SuffixNodeReplacementIndex,
    suffix_node_filtered_bitflip,
)


def _random_codes(sequence_length: int, bit_width: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    alphabet = 1 << bit_width
    return (
        torch.randint(
            alphabet,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        ),
        torch.randint(
            alphabet,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        ),
    )


def _assert_matches_brute(query: torch.Tensor, key: torch.Tensor, bit_width: int):
    expected = brute_force_bitflip(query, key, bit_width)
    for backend in ("flat", "periodic"):
        actual = suffix_node_filtered_bitflip(
            query,
            key,
            bit_width,
            posting_backend=backend,
        )
        assert torch.equal(actual.base_routes, expected.base.routes)
        assert torch.equal(actual.base_lengths, expected.base.lengths)
        assert torch.equal(actual.flipped_routes, expected.flipped_routes)
        assert torch.equal(actual.flipped_lengths, expected.flipped_lengths)


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [5, 23])
def test_suffix_node_bitflip_matches_brute_random(bit_width: int, seed: int):
    query, key = _random_codes(9, bit_width, seed)
    _assert_matches_brute(query, key, bit_width)


@pytest.mark.parametrize(
    "query,key,bit_width",
    [
        ([0] * 10, [0] * 10, 4),
        ([0, 1] * 5, [1, 0] * 5, 1),
        ([0, 3, 1, 2] * 3, [2, 0, 3, 1] * 3, 2),
    ],
)
def test_suffix_node_bitflip_matches_brute_structured(query, key, bit_width):
    _assert_matches_brute(
        torch.tensor(query, dtype=torch.uint8),
        torch.tensor(key, dtype=torch.uint8),
        bit_width,
    )


def test_suffix_node_base_queries_match_hard_routes():
    query, key = _random_codes(24, 5, 31)
    expected = unlimited_hard_routes_diagonal(query, key)
    index = SuffixNodeReplacementIndex(query, key)
    for query_index in range(query.numel()):
        length, route, _ = index.best_excluding(query_index, set())
        assert route == int(expected.routes[query_index])
        assert length == int(expected.lengths[query_index])


def test_suffix_node_bitflip_matches_exhaustive_d1_t4():
    states = tuple(itertools.product(range(2), repeat=4))
    for query_values in states:
        query = torch.tensor(query_values, dtype=torch.uint8)
        for key_values in states:
            key = torch.tensor(key_values, dtype=torch.uint8)
            _assert_matches_brute(query, key, 1)


def test_periodic_postings_skip_collapsed_route_blocks():
    query = torch.zeros(24, dtype=torch.uint8)
    flat = suffix_node_filtered_bitflip(query, query.clone(), 4)
    periodic = suffix_node_filtered_bitflip(
        query,
        query.clone(),
        4,
        posting_backend="periodic",
    )
    assert torch.equal(periodic.flipped_routes, flat.flipped_routes)
    assert periodic.route_runs < periodic.route_postings / 4
    assert periodic.route_candidates_probed < flat.route_candidates_probed / 2
    assert periodic.periodic_jumps > 0


def test_suffix_node_bit_gradient_matches_brute():
    query, key = _random_codes(8, 3, 107)
    generator = torch.Generator().manual_seed(109)
    value = torch.randn(8, 5, generator=generator)
    grad_output = torch.randn(8, 5, generator=generator)
    expected = brute_force_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    actual = suffix_node_filtered_bitflip(
        query,
        key,
        3,
        posting_backend="periodic",
        value=value,
        grad_output=grad_output,
    )
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.bit_gradient, expected.bit_gradient)
