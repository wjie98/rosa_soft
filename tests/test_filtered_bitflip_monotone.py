import itertools

import pytest
import torch

from benchmarks.filtered_bitflip import (
    CodeOccurrenceIndex,
    generate_influence_events,
    semantic_bit_flips,
)
from benchmarks.filtered_bitflip_indexes import DirectMatchIndex, _SuffixArrayPair
from benchmarks.filtered_bitflip_monotone import (
    SemanticLceGeometry,
    build_influence_lce_cells,
)


def _random_codes(sequence_length, bit_width, seed):
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


@pytest.mark.parametrize("sequence_length", range(2, 18))
@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_semantic_suffix_geometry_matches_direct_lce(
    sequence_length, bit_width
):
    query, key = _random_codes(
        sequence_length,
        bit_width,
        2100 + sequence_length + bit_width,
    )
    expected = DirectMatchIndex(query, key)
    actual = SemanticLceGeometry(query, key)
    for query_position in range(1, sequence_length):
        for key_position in range(query_position):
            assert actual.left_matches(
                query_position, key_position
            ) == expected.left_matches(query_position, key_position)
            assert actual.right_matches(
                query_position, key_position
            ) == expected.right_matches(query_position, key_position)


def test_rank_runs_are_exact_maximal_and_monotone():
    query, key = _random_codes(24, 3, 2201)
    geometry = SemanticLceGeometry(query, key)
    for index in (geometry.forward_runs, geometry.reverse_runs):
        for target_rank in range(index.rank_count):
            runs = index.runs(target_rank)
            covered = {}
            by_side = {"lower": [], "upper": []}
            for run in runs:
                by_side[run.side].append(run)
                for rank in range(run.rank_start, run.rank_stop):
                    assert rank != target_rank
                    assert rank not in covered
                    covered[rank] = run.lce
                    assert run.lce == index.lce_by_rank(target_rank, rank)
            assert set(covered) == set(range(index.rank_count)) - {target_rank}

            lower = sorted(by_side["lower"], key=lambda run: run.rank_start)
            upper = sorted(by_side["upper"], key=lambda run: run.rank_start)
            assert all(
                left.lce < right.lce
                for left, right in zip(lower, lower[1:])
            )
            assert all(
                left.lce > right.lce
                for left, right in zip(upper, upper[1:])
            )


def test_semantic_geometry_rejects_a_full_key_forward_pair():
    query, key = _random_codes(8, 3, 2251)
    full_key_pair = _SuffixArrayPair(
        tuple(int(value) for value in query.tolist()),
        tuple(int(value) for value in key.tolist()),
    )
    with pytest.raises(ValueError, match="semantic key boundary"):
        SemanticLceGeometry(query, key, forward_pair=full_key_pair)


def test_lce_cells_reconstruct_every_d1_t4_influence_event():
    states = tuple(itertools.product(range(2), repeat=4))
    event_count = 0
    for query_values in states:
        query = torch.tensor(query_values, dtype=torch.uint8)
        for key_values in states:
            key = torch.tensor(key_values, dtype=torch.uint8)
            geometry = SemanticLceGeometry(query, key)
            occurrences = CodeOccurrenceIndex.build(query, key)
            matches = DirectMatchIndex(query, key)
            for flip in semantic_bit_flips(4, 1):
                expected = generate_influence_events(
                    query,
                    key,
                    flip,
                    match_index=matches,
                    occurrence_index=occurrences,
                )
                batch = build_influence_lce_cells(
                    query,
                    key,
                    flip,
                    geometry=geometry,
                    occurrence_index=occurrences,
                )
                assert batch.occurrence_count == len(expected)
                assert batch.expand_events() == expected
                event_count += len(expected)
    assert event_count == 3072


def _create_envelope(events, sequence_length):
    envelope = []
    for query_index in range(sequence_length):
        priorities = [
            (
                event.new_length(query_index),
                query_index + event.route_offset,
            )
            for event in events
            if event.start <= query_index < event.stop
        ]
        envelope.append(max(priorities, default=None))
    return tuple(envelope)


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_one_query_create_representative_per_cell_preserves_envelope(bit_width):
    query, key = _random_codes(20, bit_width, 2300 + bit_width)
    geometry = SemanticLceGeometry(query, key)
    occurrences = CodeOccurrenceIndex.build(query, key)
    for flip in semantic_bit_flips(20, bit_width):
        if flip.source != "query":
            continue
        batch = build_influence_lce_cells(
            query,
            key,
            flip,
            geometry=geometry,
            occurrence_index=occurrences,
        )
        all_creates = tuple(
            event for event in batch.expand_events() if event.creates_match
        )
        representatives = batch.dominant_query_create_events()
        assert _create_envelope(
            representatives, 20
        ) == _create_envelope(all_creates, 20)


def test_key_cell_rejects_one_representative_shortcut():
    query = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.uint8)
    key = torch.tensor([1, 0, 1, 0, 1, 0], dtype=torch.uint8)
    flip = next(
        flip
        for flip in semantic_bit_flips(6, 1)
        if flip.source == "key" and flip.position == 1
    )
    batch = build_influence_lce_cells(query, key, flip)
    with pytest.raises(ValueError, match="query flip"):
        batch.dominant_query_create_events()


@pytest.mark.parametrize("bit_width", [1, 2, 4])
def test_every_cell_local_winner_has_a_closed_form(bit_width):
    query, key = _random_codes(18, bit_width, 2400 + bit_width)
    geometry = SemanticLceGeometry(query, key)
    occurrences = CodeOccurrenceIndex.build(query, key)
    for flip in semantic_bit_flips(18, bit_width):
        batch = build_influence_lce_cells(
            query,
            key,
            flip,
            geometry=geometry,
            occurrence_index=occurrences,
        )
        for cell in batch.cells:
            events = tuple(
                cell.event_at(position)
                for position in cell.varying_positions
            )
            for query_index in range(18):
                active = tuple(
                    event
                    for event in events
                    if event.start <= query_index < event.stop
                )
                actual = cell.dominant_event_at(query_index)
                if not active:
                    assert actual is None
                    continue
                expected = max(
                    active,
                    key=lambda event: (
                        event.new_length(query_index),
                        query_index + event.route_offset,
                    ),
                )
                assert actual == expected
