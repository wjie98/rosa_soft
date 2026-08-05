import itertools
import random

import pytest
import torch

from benchmarks.filtered_bitflip import (
    CodeOccurrenceIndex,
    brute_force_bitflip,
    generate_influence_events,
    unlimited_hard_routes_diagonal,
)
from benchmarks.filtered_bitflip_compact import (
    AffineRouteRange,
    CompactReplacementCertificateTree,
    CompactSuffixNodeReplacementIndex,
    ExcludedRouteIndex,
    PackedRankBits,
    PeriodicRouteRange,
    WaveletMatrix,
    compact_filtered_bitflip,
)
from benchmarks.filtered_bitflip_indexes import DirectMatchIndex
from benchmarks.filtered_bitflip_periodic import (
    materialize_affine_winner_segments,
)
from benchmarks.filtered_bitflip_suffix_nodes import SuffixNodeReplacementIndex
from benchmarks.filtered_bitflip_suffix_nodes import ArithmeticRouteRun


def _random_codes(sequence_length: int, bit_width: int, seed: int):
    generator = torch.Generator().manual_seed(seed)
    return (
        torch.randint(
            1 << bit_width,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        ),
        torch.randint(
            1 << bit_width,
            (sequence_length,),
            dtype=torch.uint8,
            generator=generator,
        ),
    )


def test_packed_rank_bits_exhaustive():
    for size in range(10):
        for values in itertools.product((False, True), repeat=size):
            bits = PackedRankBits(values)
            for stop in range(size + 1):
                assert bits.rank1(stop) == sum(values[:stop])


@pytest.mark.parametrize("seed", range(8))
def test_wavelet_matrix_matches_sorted_range_queries(seed):
    generator = random.Random(seed)
    values = [generator.randrange(37) for _ in range(80)]
    matrix = WaveletMatrix(values, maximum_value=63)
    for _ in range(200):
        start = generator.randrange(len(values) + 1)
        stop = generator.randrange(start, len(values) + 1)
        limit = generator.randrange(-2, 66)
        expected = [value for value in values[start:stop] if value <= limit]
        assert matrix.range_predecessor(start, stop, limit) == (
            max(expected) if expected else None
        )
        assert matrix.range_count_less(
            start, stop, limit
        ) == sum(value < limit for value in values[start:stop])
        ordered = sorted(values[start:stop])
        for rank, expected_value in enumerate(ordered):
            assert matrix.kth_smallest(start, stop, rank) == expected_value


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [13, 67])
def test_compact_suffix_index_matches_posting_reference(bit_width, seed):
    query, key = _random_codes(24, bit_width, seed)
    expected = SuffixNodeReplacementIndex(query, key)
    actual = CompactSuffixNodeReplacementIndex(query, key)
    randomizer = random.Random(seed)
    for query_index in range(query.numel()):
        exclusions = [set(), set(range(query_index + 1))]
        exclusions.extend(
            {
                route
                for route in range(1, query_index + 1)
                if randomizer.random() < 0.4
            }
            for _ in range(4)
        )
        for excluded in exclusions:
            reference = expected.best_excluding(
                query_index, excluded, backend="periodic"
            )
            for backend in ("wavelet", "auto"):
                compact = actual.best_excluding(
                    query_index, excluded, backend=backend
                )
                assert compact[:2] == reference[:2]


@pytest.mark.parametrize(
    "query,key",
    [
        ([0] * 32, [0] * 32),
        ([0, 1] * 16, [1, 0] * 16),
        ([0, 1, 2, 3] * 8, [2, 3, 0, 1] * 8),
        (list(range(32)), list(reversed(range(32)))),
    ],
)
def test_compact_suffix_index_structured_cases(query, key):
    query_codes = torch.tensor(query, dtype=torch.uint8)
    key_codes = torch.tensor(key, dtype=torch.uint8)
    expected = unlimited_hard_routes_diagonal(query_codes, key_codes)
    index = CompactSuffixNodeReplacementIndex(query_codes, key_codes)
    for query_index in range(len(query)):
        length, route, _ = index.best_excluding(query_index, set())
        assert (length, route) == (
            int(expected.lengths[query_index]),
            int(expected.routes[query_index]),
        )


def test_compact_suffix_storage_avoids_quadratic_collapse_postings():
    codes = torch.zeros(128, dtype=torch.uint8)
    posting = SuffixNodeReplacementIndex(codes, codes.clone())
    compact = CompactSuffixNodeReplacementIndex(codes, codes.clone())
    assert posting.route_postings > 8_000
    assert len(compact.key_ranks) == 127
    assert compact.periodic_nodes >= 120
    assert compact.logical_bytes < posting.logical_bytes / 2


def test_compact_suffix_accepts_symbolic_arithmetic_exclusions():
    codes = torch.zeros(32, dtype=torch.uint8)
    index = CompactSuffixNodeReplacementIndex(codes, codes.clone())
    excluded_routes = set(range(2, 25, 2))
    symbolic = ExcludedRouteIndex(
        runs=(ArithmeticRouteRun(2, 24, 2),)
    )
    assert index.best_excluding(24, symbolic)[:2] == index.best_excluding(
        24, excluded_routes
    )[:2]


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_bounded_certificate_matches_brute_on_collapse(top_k):
    query = torch.zeros(16, dtype=torch.uint8)
    expected = brute_force_bitflip(query, query.clone(), 4)
    actual = compact_filtered_bitflip(
        query,
        query.clone(),
        4,
        top_k=top_k,
        materialize_routes=True,
    )
    assert actual.periodic_interval_flips > 0
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.flipped_lengths, expected.flipped_lengths)


def test_bounded_certificate_is_exact_for_every_d1_t4_edit():
    states = tuple(itertools.product(range(2), repeat=4))
    for query_values in states:
        query = torch.tensor(query_values, dtype=torch.uint8)
        for key_values in states:
            key = torch.tensor(key_values, dtype=torch.uint8)
            expected = brute_force_bitflip(query, key, 1)
            suffix = CompactSuffixNodeReplacementIndex(query, key)
            tree = CompactReplacementCertificateTree(
                query, key, suffix, top_k=1
            )
            matches = DirectMatchIndex(query, key)
            occurrences = CodeOccurrenceIndex.build(query, key)
            for flip_index, flip in enumerate(expected.flips):
                events = generate_influence_events(
                    query,
                    key,
                    flip,
                    match_index=matches,
                    occurrence_index=occurrences,
                )
                segments, _ = tree.solve(events)
                routes, lengths = materialize_affine_winner_segments(4, segments)
                assert torch.equal(routes, expected.flipped_routes[flip_index])
                assert torch.equal(lengths, expected.flipped_lengths[flip_index])


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_compact_hybrid_matches_brute_random(bit_width):
    query, key = _random_codes(10, bit_width, 1700 + bit_width)
    expected = brute_force_bitflip(query, key, bit_width)
    actual = compact_filtered_bitflip(
        query, key, bit_width, top_k=1, materialize_routes=True
    )
    assert torch.equal(actual.base_routes, expected.base.routes)
    assert torch.equal(actual.base_lengths, expected.base.lengths)
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.flipped_lengths, expected.flipped_lengths)


def test_compact_dispatch_partitions_every_semantic_flip_once():
    generator = torch.Generator().manual_seed(1761)
    random_key = torch.randint(
        16, (16,), dtype=torch.uint8, generator=generator
    )
    shifted_query = torch.cat((random_key[:1], random_key[:-1]))
    cases = (
        (
            torch.randint(16, (16,), dtype=torch.uint8, generator=generator),
            torch.randint(16, (16,), dtype=torch.uint8, generator=generator),
        ),
        (shifted_query, random_key),
        (
            torch.tensor([0, 1, 2, 3] * 4, dtype=torch.uint8),
            torch.tensor([1, 2, 3, 0] * 4, dtype=torch.uint8),
        ),
        (torch.zeros(16, dtype=torch.uint8), torch.zeros(16, dtype=torch.uint8)),
    )
    for query, key in cases:
        result = compact_filtered_bitflip(query, key, 4)
        dispatched = (
            result.empty_flips
            + result.sparse_flips
            + result.certificate_flips
            + result.periodic_interval_flips
        )
        assert dispatched == len(result.flips)
        assert len(result.route_changes) == len(result.flips)
        assert result.raw_events >= result.materialized_events


def test_compact_solver_validates_dispatch_controls_eagerly():
    codes = torch.zeros(4, dtype=torch.uint8)
    with pytest.raises(ValueError, match="top_k"):
        compact_filtered_bitflip(codes, codes, 1, top_k=0)
    with pytest.raises(ValueError, match="interval_compression"):
        compact_filtered_bitflip(codes, codes, 1, interval_compression=0)


def test_compact_hybrid_vjp_does_not_materialize_route_matrix():
    query = torch.zeros(16, dtype=torch.uint8)
    key = query.clone()
    generator = torch.Generator().manual_seed(1801)
    value = torch.randn(16, 7, generator=generator)
    grad_output = torch.randn(16, 7, generator=generator)
    expected = brute_force_bitflip(
        query,
        key,
        4,
        value=value,
        grad_output=grad_output,
    )
    actual = compact_filtered_bitflip(
        query,
        key,
        4,
        top_k=1,
        value=value,
        grad_output=grad_output,
    )
    assert actual.flipped_routes is None
    assert actual.flipped_lengths is None
    assert actual.changed_route_rows < expected.flipped_routes.numel()
    torch.testing.assert_close(
        actual.bit_gradient,
        expected.bit_gradient,
        rtol=1e-6,
        atol=1e-6,
    )


def test_periodic_interval_hybrid_vjp_matches_brute():
    motif = torch.tensor([0, 3, 1, 2], dtype=torch.uint8)
    key = motif.repeat(4)
    query = torch.empty_like(key)
    query[0] = key[0]
    query[1:] = key[:-1]
    generator = torch.Generator().manual_seed(1811)
    value = torch.randn(16, 5, generator=generator)
    grad_output = torch.randn(16, 5, generator=generator)
    expected = brute_force_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    actual = compact_filtered_bitflip(
        query,
        key,
        3,
        top_k=1,
        value=value,
        grad_output=grad_output,
    )
    assert actual.periodic_interval_flips > 0
    torch.testing.assert_close(
        actual.bit_gradient,
        expected.bit_gradient,
        rtol=2e-6,
        atol=2e-6,
    )


def test_uniform_key_interval_emits_one_constant_route_range():
    codes = torch.zeros(16, dtype=torch.uint8)
    result = compact_filtered_bitflip(codes, codes.clone(), 1)
    flip_index = result.flips.index(
        next(
            flip
            for flip in result.flips
            if flip.source == "key" and flip.position == 7
        )
    )
    assert result.route_changes[flip_index] == (
        AffineRouteRange(8, 15, 7, 0),
    )


@pytest.mark.parametrize("case", ["uniform", "periodic"])
def test_compressed_route_ranges_reconstruct_materialized_routes(case):
    if case == "uniform":
        query = torch.zeros(16, dtype=torch.uint8)
        key = query.clone()
        bit_width = 2
    else:
        motif = torch.tensor([0, 3, 1, 2], dtype=torch.uint8)
        key = motif.repeat(4)
        query = torch.empty_like(key)
        query[0] = key[0]
        query[1:] = key[:-1]
        bit_width = 3
    result = compact_filtered_bitflip(
        query,
        key,
        bit_width,
        top_k=1,
        materialize_routes=True,
    )
    for flip_index, changes in enumerate(result.route_changes):
        reconstructed = result.base_routes.clone()
        for change in changes:
            reconstructed[change.start : change.stop] = torch.tensor(
                [
                    change.route_at(query_index)
                    for query_index in range(change.start, change.stop)
                ],
                dtype=torch.int64,
            )
        assert torch.equal(reconstructed, result.flipped_routes[flip_index])


def test_periodic_key_vjp_keeps_one_staircase_route_range():
    motif = torch.tensor([0, 3, 1, 2], dtype=torch.uint8)
    key = motif.repeat(16)
    query = torch.empty_like(key)
    query[0] = 7
    query[1:] = key[:-1]
    result = compact_filtered_bitflip(query, key, 3)
    flip_index = result.flips.index(
        next(
            flip
            for flip in result.flips
            if flip.source == "key"
            and flip.position == 31
            and flip.bit == 0
        )
    )
    assert result.route_changes[flip_index] == (
        PeriodicRouteRange(32, 60, 32, -4, 4, -4),
    )
    assert result.route_change_descriptors < (
        result.affine_equivalent_route_ranges
    )


def test_periodic_interval_sweep_keeps_winning_create_additions():
    query = torch.tensor(
        [1, 0, 4, 4, 5, 0, 4, 4, 5, 0, 4, 4, 5],
        dtype=torch.uint8,
    )
    key = torch.tensor(
        [0, 4, 4, 5, 0, 4, 4, 5, 0, 4, 4, 5, 0],
        dtype=torch.uint8,
    )
    expected = brute_force_bitflip(query, key, 3)
    actual = compact_filtered_bitflip(
        query, key, 3, top_k=1, materialize_routes=True
    )
    assert actual.periodic_creation_intervals_expanded > 0
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.flipped_lengths, expected.flipped_lengths)


def test_collapse_builds_direct_families_without_materializing_events():
    codes = torch.zeros(64, dtype=torch.uint8)
    result = compact_filtered_bitflip(codes, codes.clone(), 8)
    assert result.raw_events == 32_256
    assert result.materialized_events == 0
    assert result.direct_event_family_flips == len(result.flips)
    assert result.direct_event_families == len(result.flips)


def test_periodic_early_key_flips_keep_the_general_exact_event_path():
    motif = torch.tensor([0, 3, 1, 2], dtype=torch.uint8)
    key = motif.repeat(4)
    query = torch.empty_like(key)
    query[0] = 7
    query[1:] = key[:-1]
    result = compact_filtered_bitflip(query, key, 3)
    early_key_flips = (2 * motif.numel() - 1) * 3
    assert result.sparse_flips + result.certificate_flips == early_key_flips
    assert result.materialized_events > 0
    assert result.direct_event_family_flips == (
        len(result.flips) - early_key_flips
    )


def test_nonperiodic_shift_uses_exact_monotone_query_create_cells():
    motif = torch.tensor([0, 3, 1, 6, 2, 5, 7, 4], dtype=torch.uint8)
    key = motif.repeat(3)
    key[5] ^= 1
    key[16] ^= 3
    query = torch.empty_like(key)
    query[0] = 7
    query[1:] = key[:-1]
    expected = brute_force_bitflip(query, key, 3)
    actual = compact_filtered_bitflip(
        query, key, 3, materialize_routes=True
    )
    assert actual.direct_event_family_flips == 0
    assert actual.monotone_lce_cell_flips == (query.numel() - 1) * 3
    assert 0 < actual.monotone_lce_cells < actual.monotone_create_occurrences
    assert actual.monotone_create_occurrences < actual.raw_events
    assert actual.materialized_events < actual.raw_events
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.flipped_lengths, expected.flipped_lengths)


def test_monotone_query_path_keeps_break_only_baseline_edits():
    key = torch.tensor(
        [0, 2, 4, 6, 0, 4, 2, 6, 4, 0, 6, 2], dtype=torch.uint8
    )
    query = torch.empty_like(key)
    query[0] = 6
    query[1:] = key[:-1]
    expected = brute_force_bitflip(query, key, 3)
    actual = compact_filtered_bitflip(
        query, key, 3, materialize_routes=True
    )
    assert actual.monotone_lce_cell_flips > 0
    assert torch.equal(actual.flipped_routes, expected.flipped_routes)
    assert torch.equal(actual.flipped_lengths, expected.flipped_lengths)


def test_nonperiodic_monotone_query_cell_vjp_matches_brute():
    key = torch.tensor(
        [0, 3, 1, 6, 2, 4, 7, 4, 0, 3, 5, 6, 2, 5, 1, 4],
        dtype=torch.uint8,
    )
    query = torch.empty_like(key)
    query[0] = 7
    query[1:] = key[:-1]
    generator = torch.Generator().manual_seed(1961)
    value = torch.randn(16, 5, generator=generator)
    grad_output = torch.randn(16, 5, generator=generator)
    expected = brute_force_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    actual = compact_filtered_bitflip(
        query,
        key,
        3,
        value=value,
        grad_output=grad_output,
    )
    assert actual.monotone_lce_cell_flips > 0
    torch.testing.assert_close(
        actual.bit_gradient,
        expected.bit_gradient,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.parametrize("sequence_length", [0, 1])
def test_compact_hybrid_handles_no_semantic_flips(sequence_length):
    codes = torch.zeros(sequence_length, dtype=torch.uint8)
    result = compact_filtered_bitflip(
        codes, codes.clone(), 1, materialize_routes=True
    )
    assert result.flips == ()
    assert result.flipped_routes.shape == (0, sequence_length)
    assert result.flipped_lengths.shape == (0, sequence_length)
    assert result.route_changes == ()


def test_top_certificate_storage_is_bounded_per_tree_node():
    query, key = _random_codes(96, 1, 1907)
    suffix = CompactSuffixNodeReplacementIndex(query, key)
    tree = CompactReplacementCertificateTree(query, key, suffix, top_k=2)
    assert tree.top_postings <= 2 * (2 * tree.size)
    assert all(len(candidates) <= 2 for candidates in tree.top)
    assert tree.retained_intervals <= 2 * query.numel()
    assert any(priority is not None for priority in tree.omitted_upper)
