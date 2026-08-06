import itertools

import pytest
import torch

from benchmarks.filtered_bitflip import (
    brute_force_bitflip,
    unlimited_hard_routes_diagonal,
)
from benchmarks.sam_bitflip import (
    ArithmeticEndPositionIndex,
    ExplicitEndPositionIndex,
    ImplicitEndPositionIndex,
    SamBitflipSolver,
    SuffixAutomaton,
    materialize_route_changes,
    sam_bitflip,
)


def _codes(state, sequence_length):
    return torch.tensor(
        [(state >> position) & 1 for position in range(sequence_length)],
        dtype=torch.uint8,
    )


def _random_codes(sequence_length, bit_width, seed):
    generator = torch.Generator().manual_seed(seed)
    query = torch.randint(
        0,
        1 << bit_width,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    key = torch.randint(
        0,
        1 << bit_width,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    return query, key


def _direct_left(query, key, query_position, key_position):
    length = 0
    while (
        query_position - length - 1 >= 0
        and key_position - length - 1 >= 0
        and query[query_position - length - 1]
        == key[key_position - length - 1]
    ):
        length += 1
    return length


def _direct_right(query, key, query_position, key_position):
    length = 0
    while (
        query_position + length + 1 < len(query)
        and key_position + length + 1 < len(key) - 1
        and query[query_position + length + 1]
        == key[key_position + length + 1]
    ):
        length += 1
    return length


@pytest.mark.parametrize("sequence_length", range(0, 17))
@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_sam_base_trace_matches_unlimited_diagonal(
    sequence_length,
    bit_width,
):
    query, key = _random_codes(
        sequence_length,
        bit_width,
        101 + 17 * sequence_length + bit_width,
    )
    result = SamBitflipSolver(query, key, bit_width).base
    expected = unlimited_hard_routes_diagonal(query, key)
    torch.testing.assert_close(result.routes, expected.routes)
    torch.testing.assert_close(result.lengths, expected.lengths)


def test_explicit_and_implicit_endpos_match_every_state_and_bound():
    for sequence_length in range(1, 24):
        _, key = _random_codes(sequence_length, 2, 311 + sequence_length)
        automaton = SuffixAutomaton(key[:-1].tolist())
        explicit = ExplicitEndPositionIndex(automaton)
        implicit = ImplicitEndPositionIndex(automaton)
        for state in range(automaton.state_count):
            for bound in range(-1, sequence_length + 2):
                assert explicit.predecessor(state, bound) == implicit.predecessor(
                    state,
                    bound,
                )


def test_arithmetic_certificates_match_explicit_predecessors():
    for sequence_length in range(1, 32):
        _, key = _random_codes(sequence_length, 3, 419 + sequence_length)
        automaton = SuffixAutomaton(key[:-1].tolist())
        explicit = ExplicitEndPositionIndex(automaton)
        certified = ArithmeticEndPositionIndex(automaton, explicit)
        for state in range(automaton.state_count):
            for bound in range(-1, sequence_length + 2):
                assert certified.predecessor(
                    state,
                    bound,
                ) == explicit.predecessor(state, bound)


def test_suffix_link_lca_matches_direct_left_and_right_contexts():
    for sequence_length in range(2, 14):
        for seed in range(5):
            query, key = _random_codes(
                sequence_length,
                2,
                601 + 29 * sequence_length + seed,
            )
            solver = SamBitflipSolver(query, key, 2)
            query_list = query.tolist()
            key_list = key.tolist()
            for key_position in range(sequence_length - 1):
                for query_position in range(
                    key_position + 1,
                    sequence_length,
                ):
                    assert solver.left_context(
                        query_position,
                        key_position,
                    ) == _direct_left(
                        query_list,
                        key_list,
                        query_position,
                        key_position,
                    )
                    assert solver.right_context(
                        query_position,
                        key_position,
                    ) == _direct_right(
                        query_list,
                        key_list,
                        query_position,
                        key_position,
                    )


@pytest.mark.parametrize("sequence_length", [4, 5])
def test_exhaustive_binary_sam_bitflip_matches_full_rerun(sequence_length):
    for query_state, key_state in itertools.product(
        range(1 << sequence_length),
        repeat=2,
    ):
        query = _codes(query_state, sequence_length)
        key = _codes(key_state, sequence_length)
        result = sam_bitflip(query, key, 1, materialize_routes=True)
        expected = brute_force_bitflip(query, key, 1)
        torch.testing.assert_close(result.base.routes, expected.base.routes)
        torch.testing.assert_close(result.base.lengths, expected.base.lengths)
        torch.testing.assert_close(
            result.flipped_routes,
            expected.flipped_routes,
        )
        torch.testing.assert_close(
            result.flipped_lengths,
            expected.flipped_lengths,
        )


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
@pytest.mark.parametrize("sequence_length", [2, 3, 6, 11, 17])
def test_random_sam_bitflip_matches_full_rerun(sequence_length, bit_width):
    for seed in range(4):
        query, key = _random_codes(
            sequence_length,
            bit_width,
            1201 + 97 * sequence_length + 11 * bit_width + seed,
        )
        result = sam_bitflip(
            query,
            key,
            bit_width,
            materialize_routes=True,
        )
        expected = brute_force_bitflip(query, key, bit_width)
        assert result.flips == expected.flips
        torch.testing.assert_close(
            result.flipped_routes,
            expected.flipped_routes,
        )
        torch.testing.assert_close(
            result.flipped_lengths,
            expected.flipped_lengths,
        )


@pytest.mark.parametrize(
    "query,key,bit_width",
    [
        ([0] * 16, [0] * 16, 1),
        ([0, 1] * 8, [1, 0] * 8, 1),
        ([0, 1, 2, 3] * 4, [3, 0, 1, 2] * 4, 2),
        ([0, 7, 0, 7] * 4, [7, 0, 7, 0] * 4, 3),
        (list(range(16)), list(reversed(range(16))), 4),
    ],
)
def test_periodic_collapse_and_latest_ties_match_full_rerun(
    query,
    key,
    bit_width,
):
    query_codes = torch.tensor(query, dtype=torch.uint8)
    key_codes = torch.tensor(key, dtype=torch.uint8)
    result = sam_bitflip(
        query_codes,
        key_codes,
        bit_width,
        materialize_routes=True,
    )
    expected = brute_force_bitflip(query_codes, key_codes, bit_width)
    torch.testing.assert_close(result.flipped_routes, expected.flipped_routes)
    torch.testing.assert_close(result.flipped_lengths, expected.flipped_lengths)


@pytest.mark.parametrize("endpos_backend", ["explicit", "implicit"])
@pytest.mark.parametrize("center_backend", ["lists", "bitset"])
@pytest.mark.parametrize("hot_cache_size", [0, 1, 4])
def test_execution_backends_preserve_exact_result(
    endpos_backend,
    center_backend,
    hot_cache_size,
):
    query, key = _random_codes(18, 3, 1907)
    result = sam_bitflip(
        query,
        key,
        3,
        endpos_backend=endpos_backend,
        center_backend=center_backend,
        hot_cache_size=hot_cache_size,
        materialize_routes=True,
    )
    expected = brute_force_bitflip(query, key, 3)
    torch.testing.assert_close(result.flipped_routes, expected.flipped_routes)
    torch.testing.assert_close(result.flipped_lengths, expected.flipped_lengths)


def test_last_m_cache_uses_exact_cold_fallback():
    query = torch.tensor([0, 1] * 16, dtype=torch.uint8)
    key = torch.tensor([1, 0] * 16, dtype=torch.uint8)
    result = sam_bitflip(
        query,
        key,
        1,
        hot_cache_size=1,
        arithmetic_endpos=False,
        materialize_routes=True,
    )
    expected = brute_force_bitflip(query, key, 1)
    torch.testing.assert_close(result.flipped_routes, expected.flipped_routes)
    assert result.profile.cache_hits > 0
    assert result.profile.cache_cold_fallbacks > 0


def test_query_replay_performs_no_more_work_than_full_suffix_replay():
    query, key = _random_codes(48, 4, 2309)
    result = sam_bitflip(query, key, 4)
    assert result.profile.query_replay_steps <= (
        result.profile.query_replay_possible_steps
    )
    assert result.profile.query_replay_coalescences > 0


def test_virtual_run_batch_and_heap_counters_are_consistent():
    query, key = _random_codes(32, 3, 2609)
    result = sam_bitflip(query, key, 3)
    profile = result.profile
    assert profile.virtual_runs == profile.virtual_center_candidates
    assert profile.virtual_heap_pushes == profile.virtual_runs
    assert profile.virtual_heap_pops <= profile.virtual_heap_pushes
    assert profile.virtual_active_rows >= profile.virtual_runs


def test_sam_bitflip_vjp_matches_full_rerun():
    query, key = _random_codes(13, 4, 2903)
    generator = torch.Generator().manual_seed(2909)
    value = torch.randn(13, 7, dtype=torch.float64, generator=generator)
    grad_output = torch.randn(13, 7, dtype=torch.float64, generator=generator)
    result = sam_bitflip(
        query,
        key,
        4,
        value=value,
        grad_output=grad_output,
        hot_cache_size=2,
    )
    expected = brute_force_bitflip(
        query,
        key,
        4,
        value=value,
        grad_output=grad_output,
    )
    torch.testing.assert_close(result.bit_gradient, expected.bit_gradient)


def test_compressed_route_changes_reconstruct_full_counterfactuals():
    query, key = _random_codes(23, 4, 3203)
    result = sam_bitflip(query, key, 4)
    assert result.flipped_routes is None
    assert result.flipped_lengths is None
    routes, lengths = materialize_route_changes(
        result.base,
        result.route_changes,
    )
    expected = brute_force_bitflip(query, key, 4)
    torch.testing.assert_close(routes, expected.flipped_routes)
    torch.testing.assert_close(lengths, expected.flipped_lengths)


@pytest.mark.parametrize(
    "query,key,bit_width",
    [
        ([0] * 24, [0] * 24, 8),
        ([0, 1] * 12, [1, 0] * 12, 1),
        ([0, 1, 2, 3] * 6, [3, 0, 1, 2] * 6, 2),
    ],
)
def test_arithmetic_endpos_certificates_preserve_result(
    query,
    key,
    bit_width,
):
    query_codes = torch.tensor(query, dtype=torch.uint8)
    key_codes = torch.tensor(key, dtype=torch.uint8)
    certified = sam_bitflip(
        query_codes,
        key_codes,
        bit_width,
        arithmetic_endpos=True,
        materialize_routes=True,
    )
    fallback = sam_bitflip(
        query_codes,
        key_codes,
        bit_width,
        arithmetic_endpos=False,
        materialize_routes=True,
    )
    torch.testing.assert_close(certified.flipped_routes, fallback.flipped_routes)
    torch.testing.assert_close(certified.flipped_lengths, fallback.flipped_lengths)
    assert certified.profile.arithmetic_hits > 0


@pytest.mark.parametrize(
    "query,key,bit_width,error",
    [
        (torch.zeros(2, 2, dtype=torch.uint8), torch.zeros(4, dtype=torch.uint8), 1, "vectors"),
        (torch.zeros(3, dtype=torch.uint8), torch.zeros(4, dtype=torch.uint8), 1, "shape"),
        (torch.zeros(3), torch.zeros(3), 1, "uint8"),
        (torch.tensor([0, 2], dtype=torch.uint8), torch.zeros(2, dtype=torch.uint8), 1, "outside"),
    ],
)
def test_sam_bitflip_rejects_invalid_inputs(query, key, bit_width, error):
    with pytest.raises(ValueError, match=error):
        sam_bitflip(query, key, bit_width)
