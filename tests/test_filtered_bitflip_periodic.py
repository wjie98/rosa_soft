import pytest
import torch

from benchmarks.filtered_bitflip import (
    CodeOccurrenceIndex,
    brute_force_bitflip,
    generate_influence_events,
    semantic_bit_flips,
)
from benchmarks.filtered_bitflip_certificates import (
    build_periodic_event_families,
    materialize_winner_segments,
)
from benchmarks.filtered_bitflip_indexes import DirectMatchIndex
from benchmarks.filtered_bitflip_periodic import (
    AffineWinnerSegment,
    ShiftAlignedPeriodicKeyFallback,
    ShiftAlignedQuerySuffixFallback,
    ShiftPeriodicEventFamilyBuilder,
    UniformPeriodicSuffixFallback,
    materialize_affine_winner_segments,
)


def _event_fields(events):
    return sorted(
        (
            event.creates_match,
            event.start,
            event.stop,
            event.long_normalized_length,
            event.short_normalized_length,
            event.route_offset,
        )
        for event in events
    )


def _family_fields(families):
    fields = []
    for family in families:
        for index in range(family.count):
            fields.append(
                (
                    family.creates_match,
                    family.start(index),
                    family.stop(index),
                    family.old_priority(index)[0],
                    family.first_short_normalized_length
                    + family.delta_short_normalized_length * index,
                    family.old_priority(index)[1],
                )
            )
    return sorted(fields)


@pytest.mark.parametrize(
    "motif",
    [
        [0],
        [0, 1],
        [0, 1, 3],
        [0, 3, 1, 2],
    ],
)
def test_direct_periodic_event_families_match_every_materialized_event(motif):
    period = len(motif)
    sequence_length = 3 * period + 2
    key = torch.tensor(motif, dtype=torch.uint8).repeat(
        (sequence_length + period - 1) // period
    )[:sequence_length]
    query = torch.empty_like(key)
    query[0] = 7
    query[1:] = key[:-1]
    builder = ShiftPeriodicEventFamilyBuilder(query, key, period)
    matches = DirectMatchIndex(query, key)
    occurrences = CodeOccurrenceIndex.build(query, key)
    for flip in semantic_bit_flips(sequence_length, 3):
        events = generate_influence_events(
            query,
            key,
            flip,
            match_index=matches,
            occurrence_index=occurrences,
        )
        direct = builder.build(flip)
        assert _family_fields(direct.families) == _event_fields(events)
        assert direct.event_count == len(events)
        assert direct.event_cells == sum(
            event.stop - event.start for event in events
        )


@pytest.mark.parametrize(
    "sequence_length,symbol,bit_width",
    [
        (2, 0, 1),
        (5, 0, 4),
        (9, 3, 4),
        (7, 255, 8),
    ],
)
def test_uniform_periodic_interval_matches_every_brute_flip(
    sequence_length,
    symbol,
    bit_width,
):
    query = torch.full((sequence_length,), symbol, dtype=torch.uint8)
    key = query.clone()
    fallback = UniformPeriodicSuffixFallback.build_if_supported(query, key)
    assert fallback is not None
    expected = brute_force_bitflip(query, key, bit_width)
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
        segments, stats = fallback.solve(
            0,
            sequence_length,
            build_periodic_event_families(events),
        )
        routes, lengths = materialize_affine_winner_segments(
            sequence_length, segments
        )
        assert torch.equal(routes, expected.flipped_routes[flip_index])
        assert torch.equal(lengths, expected.flipped_lengths[flip_index])
        assert stats.rows_solved <= sequence_length


def test_uniform_periodic_interval_rejects_nonuniform_or_unaligned_codes():
    alternating = torch.tensor([0, 1, 0, 1], dtype=torch.uint8)
    assert (
        UniformPeriodicSuffixFallback.build_if_supported(
            alternating, alternating.clone()
        )
        is None
    )
    zeros = torch.zeros(4, dtype=torch.uint8)
    ones = torch.ones(4, dtype=torch.uint8)
    assert UniformPeriodicSuffixFallback.build_if_supported(zeros, ones) is None


def test_uniform_periodic_interval_without_events_returns_base_routes():
    codes = torch.full((8,), 7, dtype=torch.uint8)
    fallback = UniformPeriodicSuffixFallback.build_if_supported(codes, codes)
    assert fallback is not None
    segments, stats = fallback.solve(0, 8, ())
    routes, lengths = materialize_winner_segments(8, segments)
    torch.testing.assert_close(routes, torch.arange(8))
    torch.testing.assert_close(lengths, torch.arange(8))
    assert stats.rows_solved == 0
    assert stats.predecessor_probes == 0


@pytest.mark.parametrize("sequence_length", [2, 3, 8, 17])
def test_uniform_key_affine_interval_matches_every_key_flip(sequence_length):
    query = torch.zeros(sequence_length, dtype=torch.uint8)
    key = query.clone()
    fallback = UniformPeriodicSuffixFallback.build_if_supported(query, key)
    assert fallback is not None
    expected = brute_force_bitflip(query, key, 1)
    matches = DirectMatchIndex(query, key)
    occurrences = CodeOccurrenceIndex.build(query, key)
    for flip_index, flip in enumerate(expected.flips):
        if flip.source != "key":
            continue
        events = generate_influence_events(
            query,
            key,
            flip,
            match_index=matches,
            occurrence_index=occurrences,
        )
        segments, _ = fallback.solve_key_flip(
            flip.position,
            build_periodic_event_families(events),
        )
        routes, lengths = materialize_affine_winner_segments(
            sequence_length, segments
        )
        assert torch.equal(routes, expected.flipped_routes[flip_index])
        assert torch.equal(lengths, expected.flipped_lengths[flip_index])


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
@pytest.mark.parametrize("seed", [41, 107])
def test_shift_aligned_query_interval_matches_every_query_flip(bit_width, seed):
    generator = torch.Generator().manual_seed(seed + bit_width)
    key = torch.randint(
        1 << bit_width,
        (13,),
        dtype=torch.uint8,
        generator=generator,
    )
    query = torch.randint(
        1 << bit_width,
        (13,),
        dtype=torch.uint8,
        generator=generator,
    )
    query[1:] = key[:-1]
    fallback = ShiftAlignedQuerySuffixFallback.build_if_supported(query, key)
    assert fallback is not None
    expected = brute_force_bitflip(query, key, bit_width)
    matches = DirectMatchIndex(query, key)
    occurrences = CodeOccurrenceIndex.build(query, key)
    flips = semantic_bit_flips(query.numel(), bit_width)
    for flip_index, flip in enumerate(flips):
        if flip.source != "query":
            continue
        events = generate_influence_events(
            query,
            key,
            flip,
            match_index=matches,
            occurrence_index=occurrences,
        )
        segments, _ = fallback.solve_query_flip(
            flip.position,
            build_periodic_event_families(events),
        )
        routes, lengths = materialize_winner_segments(13, segments)
        assert torch.equal(routes, expected.flipped_routes[flip_index])
        assert torch.equal(lengths, expected.flipped_lengths[flip_index])


def test_shift_aligned_query_interval_rejects_one_broken_alignment():
    key = torch.tensor([1, 2, 3, 4, 5], dtype=torch.uint8)
    query = torch.tensor([9, 1, 2, 0, 4], dtype=torch.uint8)
    assert ShiftAlignedQuerySuffixFallback.build_if_supported(query, key) is None


@pytest.mark.parametrize("period", [2, 3, 4])
def test_shift_aligned_periodic_key_interval_matches_supported_key_flips(period):
    sequence_length = 25
    motif = torch.arange(period, dtype=torch.uint8)
    key = motif.repeat((sequence_length + period - 1) // period)[
        :sequence_length
    ].clone()
    query = torch.empty_like(key)
    query[0] = 7
    query[1:] = key[:-1]
    fallback = ShiftAlignedPeriodicKeyFallback.build_if_supported(query, key)
    assert fallback is not None
    assert fallback.period == period
    bit_width = 3
    expected = brute_force_bitflip(query, key, bit_width)
    matches = DirectMatchIndex(query, key)
    occurrences = CodeOccurrenceIndex.build(query, key)
    flips = semantic_bit_flips(sequence_length, bit_width)
    for flip_index, flip in enumerate(flips):
        if flip.source != "key" or not fallback.supports_position(flip.position):
            continue
        events = generate_influence_events(
            query,
            key,
            flip,
            match_index=matches,
            occurrence_index=occurrences,
        )
        segments, _ = fallback.solve_key_flip(
            flip.position,
            build_periodic_event_families(events),
        )
        routes, lengths = materialize_affine_winner_segments(
            sequence_length, segments
        )
        assert torch.equal(routes, expected.flipped_routes[flip_index])
        assert torch.equal(lengths, expected.flipped_lengths[flip_index])


def test_shift_aligned_periodic_key_requires_enough_complete_periods():
    key = torch.tensor([0, 1, 2, 3, 0, 1], dtype=torch.uint8)
    query = torch.tensor([9, 0, 1, 2, 3, 0], dtype=torch.uint8)
    assert ShiftAlignedPeriodicKeyFallback.build_if_supported(query, key) is None


def test_early_periodic_key_has_an_unchanged_partial_suffix_counterexample():
    motif = torch.tensor([0, 1, 0], dtype=torch.uint8)
    key = motif.repeat(4)[:11]
    query = torch.empty_like(key)
    query[0] = 3
    query[1:] = key[:-1]
    fallback = ShiftAlignedPeriodicKeyFallback.build_if_supported(query, key)
    assert fallback is not None
    assert fallback.period == 3
    assert not fallback.supports_position(2)
    expected = brute_force_bitflip(query, key, 2)
    flip_index = expected.flips.index(
        next(
            flip
            for flip in expected.flips
            if flip.source == "key"
            and flip.position == 2
            and flip.bit == 0
        )
    )
    assert int(expected.flipped_routes[flip_index, 3]) == 1
    phase_route = 2 - ((2 - 3) % fallback.period)
    assert phase_route == 0


def test_periodic_key_baseline_is_one_staircase_segment():
    motif = torch.tensor([0, 3, 1, 2], dtype=torch.uint8)
    key = motif.repeat(16)
    query = torch.empty_like(key)
    query[0] = 7
    query[1:] = key[:-1]
    fallback = ShiftAlignedPeriodicKeyFallback.build_if_supported(query, key)
    assert fallback is not None
    flip = next(
        flip
        for flip in semantic_bit_flips(64, 3)
        if flip.source == "key" and flip.position == 31 and flip.bit == 0
    )
    events = generate_influence_events(
        query,
        key,
        flip,
        match_index=DirectMatchIndex(query, key),
        occurrence_index=CodeOccurrenceIndex.build(query, key),
    )
    segments, _ = fallback.solve_key_flip(
        flip.position, build_periodic_event_families(events)
    )
    staircase = next(
        segment
        for segment in segments
        if isinstance(segment, AffineWinnerSegment)
    )
    assert (staircase.start, staircase.stop) == (32, 60)
    assert staircase.rows_per_step == 4
    assert staircase.first_priority == (-4, -4)
    assert staircase.priority_at(59) == (-28, -28)


@pytest.mark.parametrize(
    "values,expected",
    [
        ([0, 0, 0, 0, 0], 1),
        ([0, 1, 0, 1, 0], 2),
        ([0, 1, 2, 0, 1, 2, 0], 3),
        ([0, 1, 2, 3], 4),
    ],
)
def test_period_detection_uses_exact_finite_string_period(values, expected):
    assert ShiftAlignedPeriodicKeyFallback._minimum_period(values) == expected
