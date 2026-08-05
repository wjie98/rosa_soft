import itertools

import pytest
import torch

from benchmarks.filtered_bitflip import (
    brute_force_bitflip,
    unlimited_hard_routes_diagonal,
)
from benchmarks.filtered_bitflip_certificates import (
    PeriodicEventFamily,
    PeriodicEventFamilyIndex,
    build_candidate_lines,
    certificate_filtered_bitflip,
)


def _random_codes(sequence_length: int, bit_width: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    alphabet = 1 << bit_width
    query = torch.randint(
        alphabet,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    key = torch.randint(
        alphabet,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    return query, key


def _assert_matches_brute(query: torch.Tensor, key: torch.Tensor, bit_width: int):
    expected = brute_force_bitflip(query, key, bit_width)
    for backend in ("canonical", "periodic"):
        actual = certificate_filtered_bitflip(
            query,
            key,
            bit_width,
            overlay_backend=backend,
        )
        assert torch.equal(actual.base_routes, expected.base.routes)
        assert torch.equal(actual.base_lengths, expected.base.lengths)
        assert torch.equal(actual.flipped_routes, expected.flipped_routes)
        assert torch.equal(actual.flipped_lengths, expected.flipped_lengths)


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [3, 19, 71])
def test_certificate_bitflip_matches_brute_random(bit_width: int, seed: int):
    query, key = _random_codes(9, bit_width, seed)
    _assert_matches_brute(query, key, bit_width)


@pytest.mark.parametrize(
    "query,key,bit_width",
    [
        ([0] * 10, [0] * 10, 4),
        ([0, 1] * 5, [1, 0] * 5, 1),
        ([0, 3, 1, 2] * 3, [2, 0, 3, 1] * 3, 2),
        (list(range(12)), [0] + list(range(11)), 4),
    ],
)
def test_certificate_bitflip_matches_brute_structured(query, key, bit_width):
    _assert_matches_brute(
        torch.tensor(query, dtype=torch.uint8),
        torch.tensor(key, dtype=torch.uint8),
        bit_width,
    )


def test_candidate_lines_reconstruct_dense_candidate_matrix():
    query, key = _random_codes(14, 3, 29)
    expected = unlimited_hard_routes_diagonal(
        query,
        key,
        return_candidate_lengths=True,
    ).candidate_lengths
    assert expected is not None
    actual = torch.zeros_like(expected)
    for line in build_candidate_lines(query, key):
        for query_index in range(line.start, line.stop):
            route = query_index + line.route_offset
            assert actual[query_index, route] == 0
            actual[query_index, route] = query_index + line.normalized_length
    assert torch.equal(actual, expected)


def test_certificate_bitflip_matches_exhaustive_d1_t4():
    states = tuple(itertools.product(range(2), repeat=4))
    for query_values in states:
        query = torch.tensor(query_values, dtype=torch.uint8)
        for key_values in states:
            key = torch.tensor(key_values, dtype=torch.uint8)
            _assert_matches_brute(query, key, 1)


def test_certificate_bit_gradient_matches_brute():
    query, key = _random_codes(8, 3, 101)
    generator = torch.Generator().manual_seed(103)
    value = torch.randn(8, 5, generator=generator)
    grad_output = torch.randn(8, 5, generator=generator)
    expected = brute_force_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    actual = certificate_filtered_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.bit_gradient, expected.bit_gradient)


def test_periodic_backend_compresses_collapsed_event_families():
    query = torch.zeros(24, dtype=torch.uint8)
    result = certificate_filtered_bitflip(
        query,
        query.clone(),
        4,
        overlay_backend="periodic",
    )
    assert result.event_families < result.raw_events / 4
    assert result.periodic_compressed_events > 0.9 * result.raw_events


def test_periodic_family_index_matches_full_family_scan():
    families = tuple(
        PeriodicEventFamily(
            creates_match=index % 2 == 0,
            count=4,
            first_start=3 * index,
            first_stop=3 * index + 5,
            first_long_normalized_length=-index,
            first_short_normalized_length=-3 * index,
            first_route_offset=-index,
            delta_start=2,
            delta_stop=2,
            delta_long_normalized_length=-1,
            delta_short_normalized_length=-2,
            delta_route_offset=1,
        )
        for index in range(12)
    )
    index = PeriodicEventFamilyIndex(families)
    methods = (
        ("event_covering", "event_cover_range"),
        ("event_intersecting", "event_intersect_range"),
        ("add_covering", "add_cover_range"),
        ("add_intersecting", "add_intersect_range"),
    )
    for start in range(0, 40, 3):
        for stop in range(start + 1, min(start + 9, 48)):
            for indexed_name, family_name in methods:
                actual, _ = getattr(index, indexed_name)(start, stop)
                expected = tuple(
                    (family, phase_range)
                    for family in families
                    if (
                        phase_range := getattr(family, family_name)(start, stop)
                    )[0]
                    < phase_range[1]
                )
                assert set(actual) == set(expected)


def test_periodic_family_index_rejects_irrelevant_envelopes_early():
    families = tuple(
        PeriodicEventFamily(True, 1, row, row + 1, 0, 0, -row)
        for row in range(0, 512, 4)
    )
    index = PeriodicEventFamilyIndex(families)
    matches, examined = index.event_intersecting(250, 251)
    assert not matches
    assert examined < len(families) // 8
