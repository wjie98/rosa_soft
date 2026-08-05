import itertools
import random

import pytest
import torch

from benchmarks.filtered_bitflip import (
    CodeOccurrenceIndex,
    semantic_bit_flips,
)
from benchmarks.filtered_bitflip_monotone import (
    SemanticLceGeometry,
    build_query_create_lce_cells,
)
from benchmarks.filtered_bitflip_orthogonal import (
    CausalQueryCreateOrthogonalBuilder,
    CodePartitionedOrthogonalIndex,
    OrthogonalAggregate,
    OrthogonalPoint,
)
from benchmarks.filtered_bitflip_orthogonal_kd import (
    CodePartitionedKdIndex,
    StaticQueryCreateKdBuilder,
)


def _expected_aggregate(points):
    if not points:
        return OrthogonalAggregate()
    positions = [point.position for point in points]
    return OrthogonalAggregate(
        len(points), min(positions), max(positions)
    )


def test_code_partitioned_rectangle_aggregates_match_active_points():
    randomizer = random.Random(3101)
    size = 31
    forward = list(range(size))
    reverse = list(range(size))
    randomizer.shuffle(forward)
    randomizer.shuffle(reverse)
    points = tuple(
        OrthogonalPoint(
            code=randomizer.randrange(4),
            forward_rank=forward[position],
            reverse_rank=reverse[position],
            position=position,
        )
        for position in range(size)
    )
    index = CodePartitionedOrthogonalIndex(points)
    active = []
    for position, point in enumerate(points):
        index.activate(position)
        active.append(point)
        for _ in range(12):
            forward_bounds = sorted(
                (randomizer.randrange(size + 1), randomizer.randrange(size + 1))
            )
            reverse_bounds = sorted(
                (randomizer.randrange(size + 1), randomizer.randrange(size + 1))
            )
            code = randomizer.randrange(5)
            expected = _expected_aggregate(
                [
                    candidate
                    for candidate in active
                    if candidate.code == code
                    and forward_bounds[0]
                    <= candidate.forward_rank
                    < forward_bounds[1]
                    and reverse_bounds[0]
                    <= candidate.reverse_rank
                    < reverse_bounds[1]
                ]
            )
            assert index.rectangle(
                code,
                forward_bounds[0],
                forward_bounds[1],
                reverse_bounds[0],
                reverse_bounds[1],
            ) == expected


@pytest.mark.parametrize("leaf_size", [1, 4, 8])
def test_code_partitioned_kd_boxes_match_static_points(leaf_size):
    randomizer = random.Random(3150 + leaf_size)
    size = 31
    forward = list(range(size))
    reverse = list(range(size))
    randomizer.shuffle(forward)
    randomizer.shuffle(reverse)
    points = tuple(
        OrthogonalPoint(
            code=randomizer.randrange(4),
            forward_rank=forward[position],
            reverse_rank=reverse[position],
            position=position,
        )
        for position in range(size)
    )
    index = CodePartitionedKdIndex(points, leaf_size=leaf_size)
    for _ in range(256):
        forward_bounds = sorted(
            (randomizer.randrange(size + 1), randomizer.randrange(size + 1))
        )
        reverse_bounds = sorted(
            (randomizer.randrange(size + 1), randomizer.randrange(size + 1))
        )
        position_bounds = sorted(
            (randomizer.randrange(size + 1), randomizer.randrange(size + 1))
        )
        code = randomizer.randrange(5)
        expected = _expected_aggregate(
            [
                point
                for point in points
                if point.code == code
                and forward_bounds[0]
                <= point.forward_rank
                < forward_bounds[1]
                and reverse_bounds[0]
                <= point.reverse_rank
                < reverse_bounds[1]
                and position_bounds[0]
                <= point.position
                < position_bounds[1]
            ]
        )
        actual, nodes_visited, point_probes = index.box(
            code,
            forward_bounds[0],
            forward_bounds[1],
            reverse_bounds[0],
            reverse_bounds[1],
            position_bounds[0],
            position_bounds[1],
        )
        assert actual == expected
        assert nodes_visited >= 0
        assert point_probes >= 0


def test_static_kd_validates_boxes_before_empty_code_lookup():
    index = CodePartitionedKdIndex(
        (OrthogonalPoint(0, 0, 0, 0),)
    )
    with pytest.raises(ValueError, match="code"):
        index.box(256, 0, 1, 0, 1, 0, 1)
    with pytest.raises(ValueError, match="nonnegative"):
        index.box(1, -1, 1, 0, 1, 0, 1)
    with pytest.raises(ValueError, match="ordered"):
        index.box(1, 1, 0, 0, 1, 0, 1)
    assert index.box(1, 0, 1, 0, 1, 0, 1) == (
        OrthogonalAggregate(),
        0,
        0,
    )


def _cell_summary(batch):
    return {
        (cell.left_matches, cell.right_matches): (
            len(cell.varying_positions),
            min(cell.varying_positions),
            max(cell.varying_positions),
        )
        for cell in batch.cells
    }


def _orthogonal_cell_summary(batch):
    return {
        (cell.left_matches, cell.right_matches): (
            cell.aggregate.count,
            cell.aggregate.min_position,
            cell.aggregate.max_position,
        )
        for cell in batch.cells
    }


def _assert_query_builders_equal(query, key, bit_width):
    geometry = SemanticLceGeometry(query, key)
    occurrences = CodeOccurrenceIndex.build(query, key)
    orthogonal = CausalQueryCreateOrthogonalBuilder(
        query,
        key,
        geometry=geometry,
        occurrence_index=occurrences,
    )
    static_kd = StaticQueryCreateKdBuilder(
        query,
        key,
        geometry=geometry,
        occurrence_index=occurrences,
    )
    for flip in semantic_bit_flips(query.numel(), bit_width):
        if flip.source != "query":
            continue
        expected = build_query_create_lce_cells(
            query,
            key,
            flip,
            geometry=geometry,
            occurrence_index=occurrences,
        )
        actual = orthogonal.build(flip)
        kd_actual = static_kd.build(flip)
        assert actual.occurrence_count == expected.occurrence_count
        assert _orthogonal_cell_summary(actual) == _cell_summary(expected)
        assert _orthogonal_cell_summary(kd_actual) == _cell_summary(expected)
        expected_events = tuple(
            sorted(
                expected.dominant_query_create_events(),
                key=lambda event: event.key_position,
            )
        )
        assert actual.dominant_query_create_events() == expected_events
        assert kd_actual.dominant_query_create_events() == expected_events
    assert orthogonal.index.point_activations == max(query.numel() - 1, 0)


def test_orthogonal_cells_match_every_d1_t4_query_flip():
    states = tuple(itertools.product(range(2), repeat=4))
    for query_values in states:
        query = torch.tensor(query_values, dtype=torch.uint8)
        for key_values in states:
            key = torch.tensor(key_values, dtype=torch.uint8)
            _assert_query_builders_equal(query, key, 1)


@pytest.mark.parametrize("bit_width", [1, 2, 4, 8])
def test_orthogonal_cells_match_random_and_boundary_occurrences(bit_width):
    generator = torch.Generator().manual_seed(3200 + bit_width)
    query = torch.randint(
        1 << bit_width,
        (24,),
        dtype=torch.uint8,
        generator=generator,
    )
    key = torch.randint(
        1 << bit_width,
        (24,),
        dtype=torch.uint8,
        generator=generator,
    )
    key[0] = query[-1] ^ 1
    key[-2] = query[-1] ^ 1
    _assert_query_builders_equal(query, key, bit_width)


def test_orthogonal_query_sweep_rejects_position_rewind():
    query = torch.tensor([0, 1, 0, 1, 0], dtype=torch.uint8)
    key = torch.tensor([1, 0, 1, 0, 1], dtype=torch.uint8)
    builder = CausalQueryCreateOrthogonalBuilder(query, key)
    flips = tuple(
        flip
        for flip in semantic_bit_flips(5, 1)
        if flip.source == "query"
    )
    builder.build(flips[-1])
    with pytest.raises(ValueError, match="position order"):
        builder.build(flips[0])


@pytest.mark.parametrize("leaf_size", [1, 4, 8, 16])
def test_static_kd_leaf_blocking_preserves_exact_cells(leaf_size):
    generator = torch.Generator().manual_seed(3250 + leaf_size)
    query = torch.randint(
        4, (32,), dtype=torch.uint8, generator=generator
    )
    key = torch.randint(
        4, (32,), dtype=torch.uint8, generator=generator
    )
    geometry = SemanticLceGeometry(query, key)
    occurrences = CodeOccurrenceIndex.build(query, key)
    expected_builder = CausalQueryCreateOrthogonalBuilder(
        query,
        key,
        geometry=geometry,
        occurrence_index=occurrences,
    )
    actual_builder = StaticQueryCreateKdBuilder(
        query,
        key,
        geometry=geometry,
        occurrence_index=occurrences,
        leaf_size=leaf_size,
    )
    for flip in semantic_bit_flips(32, 2):
        if flip.source != "query":
            continue
        expected = expected_builder.build(flip)
        actual = actual_builder.build(flip)
        assert _orthogonal_cell_summary(actual) == _orthogonal_cell_summary(
            expected
        )
