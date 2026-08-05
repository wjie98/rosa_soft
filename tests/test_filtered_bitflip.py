import itertools

import pytest
import torch

from benchmarks.global_bit_oracle import exact_bitflip_vjp
from benchmarks.filtered_bitflip import (
    BitFlip,
    RouteChangeRange,
    apply_influence_events,
    brute_force_bitflip,
    filtered_bitflip,
    generate_influence_events,
    hard_output_from_routes,
    pack_sign_codes,
    profile_bitflip_structure,
    route_change_ranges,
    semantic_bit_flips,
    unlimited_hard_routes_diagonal,
    unlimited_hard_routes_direct,
)
from rosa_soft.soft_reference import _hard_route_forward


def _float_codes(codes, bit_width):
    rows = []
    for code in codes:
        rows.append(
            [1.0 if code & (1 << bit) else -1.0 for bit in range(bit_width)]
        )
    return torch.tensor(rows, dtype=torch.float64).view(1, len(codes), 1, bit_width)


def _random_inputs(sequence_length=6, bit_width=3, value_dim=4, seed=7):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn(
        1,
        sequence_length,
        1,
        bit_width,
        generator=generator,
        dtype=torch.float64,
    )
    key = torch.randn_like(query, generator=generator)
    value = torch.randn(
        1,
        sequence_length,
        1,
        value_dim,
        generator=generator,
        dtype=torch.float64,
    )
    grad_output = torch.randn_like(value, generator=generator)
    return query, key, value, grad_output


def test_pack_sign_codes_uses_positive_as_one_and_zero_as_minus_one():
    logits = torch.tensor(
        [[-1.0, 0.0, 2.0], [3.0, -4.0, 5.0]],
        dtype=torch.float64,
    )
    assert pack_sign_codes(logits).tolist() == [4, 5]


@pytest.mark.parametrize("sequence_length", range(1, 9))
@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_unlimited_diagonal_matches_frozen_hard_forward(
    sequence_length,
    bit_width,
):
    query, key, value, _ = _random_inputs(
        sequence_length,
        bit_width,
        seed=31 + sequence_length + bit_width,
    )
    packed_query = pack_sign_codes(query)
    packed_key = pack_sign_codes(key)
    result = unlimited_hard_routes_diagonal(
        packed_query,
        packed_key,
        return_candidate_lengths=True,
    )
    hard_output, exact_lengths, routes, _ = _hard_route_forward(
        query,
        key,
        value,
        sequence_length,
    )

    torch.testing.assert_close(result.routes, routes[0, 0])
    torch.testing.assert_close(
        result.lengths,
        exact_lengths[0, 0].amax(dim=-1).to(torch.int64),
    )
    torch.testing.assert_close(
        hard_output_from_routes(result.routes, value[0, :, 0]),
        hard_output[0, :, 0],
    )


@pytest.mark.parametrize(
    "query_codes,key_codes",
    [
        ([0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]),
        ([0, 1, 0, 1, 0, 1], [1, 0, 1, 0, 1, 0]),
        ([0, 1, 2, 3, 0, 1], [0, 1, 2, 3, 0, 1]),
        ([3, 2, 1, 0, 3, 2], [0, 1, 2, 3, 0, 1]),
    ],
)
def test_direct_scan_matches_diagonal_recurrence(query_codes, key_codes):
    query = torch.tensor(query_codes, dtype=torch.uint8)
    key = torch.tensor(key_codes, dtype=torch.uint8)
    direct = unlimited_hard_routes_direct(
        query,
        key,
        return_candidate_lengths=True,
    )
    diagonal = unlimited_hard_routes_diagonal(
        query,
        key,
        return_candidate_lengths=True,
    )
    torch.testing.assert_close(direct.routes, diagonal.routes)
    torch.testing.assert_close(direct.lengths, diagonal.lengths)
    torch.testing.assert_close(
        direct.candidate_lengths,
        diagonal.candidate_lengths,
    )


def test_semantic_flips_exclude_unused_query_and_key_boundaries():
    flips = semantic_bit_flips(4, 2)
    assert len(flips) == 12
    assert BitFlip("query", 0, 0) not in flips
    assert BitFlip("key", 3, 0) not in flips
    assert flips[:2] == (BitFlip("query", 1, 0), BitFlip("query", 1, 1))


@pytest.mark.parametrize("sequence_length", [2, 3, 5, 7])
@pytest.mark.parametrize("bit_width", [1, 2, 4])
def test_brute_unlimited_bitflip_matches_existing_full_horizon_oracle(
    sequence_length,
    bit_width,
):
    query, key, value, grad_output = _random_inputs(
        sequence_length,
        bit_width,
        seed=113 + sequence_length + bit_width,
    )
    result = brute_force_bitflip(
        pack_sign_codes(query),
        pack_sign_codes(key),
        bit_width,
        value=value[0, :, 0],
        grad_output=grad_output[0, :, 0],
    )
    _, _, expected, layout = exact_bitflip_vjp(
        query,
        key,
        value,
        grad_output,
        max_suffix_length=sequence_length,
    )
    assert layout.bit_count == len(result.flips)
    torch.testing.assert_close(result.bit_gradient, expected, rtol=0, atol=0)


@pytest.mark.parametrize("sequence_length", [2, 3, 5, 8])
@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_influence_events_match_complete_bitflip_reruns(
    sequence_length,
    bit_width,
):
    query, key, value, grad_output = _random_inputs(
        sequence_length,
        bit_width,
        seed=211 + sequence_length + bit_width,
    )
    packed_query = pack_sign_codes(query)
    packed_key = pack_sign_codes(key)
    brute = brute_force_bitflip(
        packed_query,
        packed_key,
        bit_width,
        value=value[0, :, 0],
        grad_output=grad_output[0, :, 0],
    )
    events = filtered_bitflip(
        packed_query,
        packed_key,
        bit_width,
        value=value[0, :, 0],
        grad_output=grad_output[0, :, 0],
    )
    assert events.flips == brute.flips
    torch.testing.assert_close(events.flipped_routes, brute.flipped_routes)
    torch.testing.assert_close(events.flipped_lengths, brute.flipped_lengths)
    torch.testing.assert_close(events.bit_gradient, brute.bit_gradient)


def test_event_old_and_new_lengths_cover_break_and_create():
    query = torch.tensor([0, 0, 1, 0], dtype=torch.uint8)
    key = torch.tensor([0, 1, 1, 0], dtype=torch.uint8)
    base = unlimited_hard_routes_diagonal(
        query,
        key,
        return_candidate_lengths=True,
    )
    assert base.candidate_lengths is not None
    for flip in semantic_bit_flips(4, 1):
        events = generate_influence_events(query, key, flip)
        routes, lengths, _ = apply_influence_events(
            base.candidate_lengths,
            events,
        )
        edited_query = query.clone()
        edited_key = key.clone()
        target = edited_query if flip.source == "query" else edited_key
        target[flip.position] ^= 1 << flip.bit
        expected = unlimited_hard_routes_diagonal(edited_query, edited_key)
        torch.testing.assert_close(routes, expected.routes)
        torch.testing.assert_close(lengths, expected.lengths)


def test_exhaustive_one_bit_event_formula():
    sequence_length = 4
    for query_state, key_state in itertools.product(
        range(1 << sequence_length),
        repeat=2,
    ):
        query = torch.tensor(
            [(query_state >> index) & 1 for index in range(sequence_length)],
            dtype=torch.uint8,
        )
        key = torch.tensor(
            [(key_state >> index) & 1 for index in range(sequence_length)],
            dtype=torch.uint8,
        )
        brute = brute_force_bitflip(query, key, 1)
        events = filtered_bitflip(query, key, 1)
        torch.testing.assert_close(events.flipped_routes, brute.flipped_routes)
        torch.testing.assert_close(events.flipped_lengths, brute.flipped_lengths)


def test_route_change_ranges_split_on_gaps_and_offsets():
    base = torch.tensor([0, 0, 1, 2, 1, 4], dtype=torch.int64)
    flipped = torch.tensor([0, 1, 2, 2, 3, 4], dtype=torch.int64)
    assert route_change_ranges(base, flipped) == (
        RouteChangeRange(1, 3, 0),
        RouteChangeRange(4, 5, -1),
    )


def test_structural_profile_counts_final_ranges():
    query = torch.tensor([0, 0, 1, 0, 1], dtype=torch.uint8)
    key = torch.tensor([1, 0, 1, 0, 0], dtype=torch.uint8)
    result = filtered_bitflip(query, key, 1)
    profile = profile_bitflip_structure(result)
    assert profile["flip_count"] == 8
    assert profile["raw_local_pair_events"] > 0
    assert profile["event_cell_updates"] >= profile["raw_local_pair_events"]
    assert profile["merged_route_change_ranges"] <= profile["changed_query_rows"]
