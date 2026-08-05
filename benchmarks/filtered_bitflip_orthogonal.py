"""Code-partitioned orthogonal indexes for exact monotone-LCE cells.

The static point set has three coordinates: forward suffix rank, reversed
suffix rank, and original position.  Query flips are processed in increasing
position order, so activating key positions once turns the causal 3D prefix
constraint into a dynamic 2D rectangle aggregate.

This post-v1 experiment is deliberately not dispatched by the frozen compact
solver.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Sequence

from torch import Tensor

from benchmarks.filtered_bitflip import (
    BitFlip,
    CodeOccurrenceIndex,
    InfluenceEvent,
    _validate_codes,
)
from benchmarks.filtered_bitflip_monotone import (
    LceRankRun,
    SemanticLceGeometry,
)


__all__ = [
    "CausalQueryCreateOrthogonalBuilder",
    "CodePartitionedOrthogonalIndex",
    "LceAxisBand",
    "OrthogonalAggregate",
    "OrthogonalCellBatch",
    "OrthogonalLceCell",
    "OrthogonalPoint",
]


_EMPTY_MIN = 1 << 62


@dataclass(frozen=True)
class OrthogonalAggregate:
    """Count and original-position extrema for one exact box query."""

    count: int = 0
    min_position: int | None = None
    max_position: int | None = None

    def __post_init__(self) -> None:
        if self.count < 0:
            raise ValueError("orthogonal count must be nonnegative")
        if self.count == 0:
            if self.min_position is not None or self.max_position is not None:
                raise ValueError("an empty aggregate cannot have extrema")
        elif (
            self.min_position is None
            or self.max_position is None
            or self.min_position > self.max_position
        ):
            raise ValueError("a nonempty aggregate requires ordered extrema")

    def combine(self, other: "OrthogonalAggregate") -> "OrthogonalAggregate":
        if self.count == 0:
            return other
        if other.count == 0:
            return self
        return OrthogonalAggregate(
            self.count + other.count,
            min(self.min_position, other.min_position),
            max(self.max_position, other.max_position),
        )


@dataclass(frozen=True)
class OrthogonalPoint:
    code: int
    forward_rank: int
    reverse_rank: int
    position: int

    def __post_init__(self) -> None:
        if not 0 <= self.code < 256:
            raise ValueError("point code must fit in one byte")
        if min(self.forward_rank, self.reverse_rank, self.position) < 0:
            raise ValueError("orthogonal coordinates must be nonnegative")


class _InnerAggregateTree:
    """Dynamic aggregate over one sorted reverse-rank coordinate set."""

    def __init__(self, reverse_ranks: Sequence[int]) -> None:
        ranks = tuple(sorted(reverse_ranks))
        if len(set(ranks)) != len(ranks):
            raise ValueError("reverse ranks must be unique inside an outer node")
        size = 1
        while size < max(len(ranks), 1):
            size *= 2
        self.reverse_ranks = ranks
        self.size = size
        self.counts = [0] * (2 * size)
        self.minimums = [_EMPTY_MIN] * (2 * size)
        self.maximums = [-1] * (2 * size)

    def activate(self, reverse_rank: int, position: int) -> None:
        offset = bisect_left(self.reverse_ranks, reverse_rank)
        if (
            offset == len(self.reverse_ranks)
            or self.reverse_ranks[offset] != reverse_rank
        ):
            raise RuntimeError("activation rank is absent from inner topology")
        node = self.size + offset
        if self.counts[node]:
            raise RuntimeError("one orthogonal point was activated twice")
        self.counts[node] = 1
        self.minimums[node] = position
        self.maximums[node] = position
        node //= 2
        while node:
            left = 2 * node
            right = left + 1
            self.counts[node] = self.counts[left] + self.counts[right]
            self.minimums[node] = min(
                self.minimums[left], self.minimums[right]
            )
            self.maximums[node] = max(
                self.maximums[left], self.maximums[right]
            )
            node //= 2

    def query(self, rank_start: int, rank_stop: int) -> OrthogonalAggregate:
        start = bisect_left(self.reverse_ranks, rank_start) + self.size
        stop = bisect_left(self.reverse_ranks, rank_stop) + self.size
        count = 0
        minimum = _EMPTY_MIN
        maximum = -1
        while start < stop:
            if start & 1:
                count += self.counts[start]
                minimum = min(minimum, self.minimums[start])
                maximum = max(maximum, self.maximums[start])
                start += 1
            if stop & 1:
                stop -= 1
                count += self.counts[stop]
                minimum = min(minimum, self.minimums[stop])
                maximum = max(maximum, self.maximums[stop])
            start //= 2
            stop //= 2
        if count == 0:
            return OrthogonalAggregate()
        return OrthogonalAggregate(count, minimum, maximum)

    @property
    def aggregate(self) -> OrthogonalAggregate:
        if self.counts[1] == 0:
            return OrthogonalAggregate()
        return OrthogonalAggregate(
            self.counts[1], self.minimums[1], self.maximums[1]
        )

    def group_by_bands(
        self,
        bands: tuple["LceAxisBand", ...],
    ) -> tuple[dict[int, OrthogonalAggregate], int]:
        """Aggregate active ranks by bands through one synchronized traversal."""

        starts = tuple(band.rank_start for band in bands)
        groups: dict[int, OrthogonalAggregate] = {}
        nodes_visited = 0

        def band_index(rank: int) -> int:
            index = bisect_right(starts, rank) - 1
            if index < 0 or not bands[index].rank_start <= rank < bands[index].rank_stop:
                raise RuntimeError("active reverse rank is outside LCE bands")
            return index

        def visit(node: int, start: int, stop: int) -> None:
            nonlocal nodes_visited
            nodes_visited += 1
            if self.counts[node] == 0 or start >= len(self.reverse_ranks):
                return
            actual_stop = min(stop, len(self.reverse_ranks))
            first_band = band_index(self.reverse_ranks[start])
            last_band = band_index(self.reverse_ranks[actual_stop - 1])
            if first_band == last_band:
                aggregate = OrthogonalAggregate(
                    self.counts[node],
                    self.minimums[node],
                    self.maximums[node],
                )
                groups[first_band] = groups.get(
                    first_band, OrthogonalAggregate()
                ).combine(aggregate)
                return
            middle = (start + stop) // 2
            visit(2 * node, start, middle)
            visit(2 * node + 1, middle, stop)

        if self.counts[1]:
            visit(1, 0, self.size)
        return groups, nodes_visited

    @property
    def logical_bytes(self) -> int:
        return 4 * len(self.reverse_ranks) + 12 * len(self.counts)


class _CodeOrthogonalIndex:
    """Outer forward-rank tree with inner reverse-rank aggregate trees."""

    def __init__(self, points: Sequence[OrthogonalPoint]) -> None:
        ordered = tuple(sorted(points, key=lambda point: point.forward_rank))
        forward_ranks = tuple(point.forward_rank for point in ordered)
        if len(set(forward_ranks)) != len(forward_ranks):
            raise ValueError("forward ranks must be unique inside a code bucket")
        size = 1
        while size < max(len(ordered), 1):
            size *= 2
        rank_lists: list[list[int]] = [[] for _ in range(2 * size)]
        for offset, point in enumerate(ordered):
            node = size + offset
            while node:
                rank_lists[node].append(point.reverse_rank)
                node //= 2
        self.forward_ranks = forward_ranks
        self.points = ordered
        self.size = size
        self.inner = tuple(
            _InnerAggregateTree(ranks) if ranks else None
            for ranks in rank_lists
        )

    def activate(self, point: OrthogonalPoint) -> None:
        offset = bisect_left(self.forward_ranks, point.forward_rank)
        if (
            offset == len(self.forward_ranks)
            or self.points[offset] != point
        ):
            raise RuntimeError("activation point is absent from code topology")
        node = self.size + offset
        while node:
            inner = self.inner[node]
            if inner is None:
                raise RuntimeError("missing inner aggregate topology")
            inner.activate(point.reverse_rank, point.position)
            node //= 2

    def query(
        self,
        forward_start: int,
        forward_stop: int,
        reverse_start: int,
        reverse_stop: int,
    ) -> OrthogonalAggregate:
        start = bisect_left(self.forward_ranks, forward_start) + self.size
        stop = bisect_left(self.forward_ranks, forward_stop) + self.size
        aggregate = OrthogonalAggregate()
        while start < stop:
            if start & 1:
                inner = self.inner[start]
                if inner is not None:
                    aggregate = aggregate.combine(
                        inner.query(reverse_start, reverse_stop)
                    )
                start += 1
            if stop & 1:
                stop -= 1
                inner = self.inner[stop]
                if inner is not None:
                    aggregate = aggregate.combine(
                        inner.query(reverse_start, reverse_stop)
                    )
            start //= 2
            stop //= 2
        return aggregate

    @property
    def aggregate(self) -> OrthogonalAggregate:
        root = self.inner[1]
        return root.aggregate if root is not None else OrthogonalAggregate()

    def group_by_bands(
        self,
        forward_bands: tuple["LceAxisBand", ...],
        reverse_bands: tuple["LceAxisBand", ...],
    ) -> tuple[dict[tuple[int, int], OrthogonalAggregate], int]:
        """Group active points without issuing independent box queries."""

        starts = tuple(band.rank_start for band in forward_bands)
        groups: dict[tuple[int, int], OrthogonalAggregate] = {}
        nodes_visited = 0

        def band_index(rank: int) -> int:
            index = bisect_right(starts, rank) - 1
            if index < 0 or not (
                forward_bands[index].rank_start
                <= rank
                < forward_bands[index].rank_stop
            ):
                raise RuntimeError("active forward rank is outside LCE bands")
            return index

        def visit(node: int, start: int, stop: int) -> None:
            nonlocal nodes_visited
            nodes_visited += 1
            inner = self.inner[node]
            if inner is None or inner.aggregate.count == 0:
                return
            if start >= len(self.forward_ranks):
                return
            actual_stop = min(stop, len(self.forward_ranks))
            first_band = band_index(self.forward_ranks[start])
            last_band = band_index(self.forward_ranks[actual_stop - 1])
            if first_band == last_band:
                reverse_groups, inner_visits = inner.group_by_bands(
                    reverse_bands
                )
                nodes_visited += inner_visits
                for reverse_band, aggregate in reverse_groups.items():
                    key = (first_band, reverse_band)
                    groups[key] = groups.get(
                        key, OrthogonalAggregate()
                    ).combine(aggregate)
                return
            middle = (start + stop) // 2
            visit(2 * node, start, middle)
            visit(2 * node + 1, middle, stop)

        root = self.inner[1]
        if root is not None and root.aggregate.count:
            visit(1, 0, self.size)
        return groups, nodes_visited

    @property
    def logical_bytes(self) -> int:
        return 4 * len(self.forward_ranks) + sum(
            tree.logical_bytes for tree in self.inner if tree is not None
        )


class CodePartitionedOrthogonalIndex:
    """Exact dynamic 2D realization of a code-filtered causal 3D index."""

    def __init__(self, points: Sequence[OrthogonalPoint]) -> None:
        by_code: list[list[OrthogonalPoint]] = [[] for _ in range(256)]
        points_by_position = {}
        for point in points:
            if point.position in points_by_position:
                raise ValueError("orthogonal positions must be unique")
            points_by_position[point.position] = point
            by_code[point.code].append(point)
        self.by_code = {
            code: _CodeOrthogonalIndex(code_points)
            for code, code_points in enumerate(by_code)
            if code_points
        }
        self.points_by_position = points_by_position
        self.active_positions: set[int] = set()
        self.point_activations = 0
        self.rectangle_queries = 0
        self.logical_bytes = 16 * len(points) + sum(
            index.logical_bytes for index in self.by_code.values()
        )

    def activate(self, position: int) -> None:
        try:
            point = self.points_by_position[position]
        except KeyError as error:
            raise IndexError("activation position is outside the point set") from error
        if position in self.active_positions:
            raise RuntimeError("one orthogonal position was activated twice")
        self.by_code[point.code].activate(point)
        self.active_positions.add(position)
        self.point_activations += 1

    def aggregate(self, code: int) -> OrthogonalAggregate:
        index = self.by_code.get(code)
        return index.aggregate if index is not None else OrthogonalAggregate()

    def rectangle(
        self,
        code: int,
        forward_start: int,
        forward_stop: int,
        reverse_start: int,
        reverse_stop: int,
    ) -> OrthogonalAggregate:
        if not (
            0 <= code < 256
            and 0 <= forward_start <= forward_stop
            and 0 <= reverse_start <= reverse_stop
        ):
            raise ValueError("invalid code or rectangle bounds")
        self.rectangle_queries += 1
        index = self.by_code.get(code)
        if index is None:
            return OrthogonalAggregate()
        return index.query(
            forward_start,
            forward_stop,
            reverse_start,
            reverse_stop,
        )

    def group_by_bands(
        self,
        code: int,
        forward_bands: tuple["LceAxisBand", ...],
        reverse_bands: tuple["LceAxisBand", ...],
    ) -> tuple[dict[tuple[int, int], OrthogonalAggregate], int]:
        index = self.by_code.get(code)
        if index is None:
            return {}, 0
        return index.group_by_bands(forward_bands, reverse_bands)


@dataclass(frozen=True)
class LceAxisBand:
    rank_start: int
    rank_stop: int
    lce: int

    def __post_init__(self) -> None:
        if not 0 <= self.rank_start < self.rank_stop:
            raise ValueError("LCE axis band must be nonempty")
        if self.lce < 0:
            raise ValueError("LCE axis depth must be nonnegative")


@dataclass(frozen=True)
class OrthogonalLceCell:
    left_matches: int
    right_matches: int
    aggregate: OrthogonalAggregate
    rectangle_count: int

    def __post_init__(self) -> None:
        if self.aggregate.count <= 0 or self.rectangle_count <= 0:
            raise ValueError("an orthogonal LCE cell must be nonempty")

    def query_create_event(self, flip: BitFlip) -> InfluenceEvent:
        if flip.source != "query" or self.aggregate.max_position is None:
            raise ValueError("query create event requires a query-side cell")
        return InfluenceEvent(
            flip=flip,
            query_position=flip.position,
            key_position=self.aggregate.max_position,
            left_matches=self.left_matches,
            right_matches=self.right_matches,
            creates_match=True,
        )


@dataclass(frozen=True)
class OrthogonalCellBatch:
    flip: BitFlip
    cells: tuple[OrthogonalLceCell, ...]
    occurrence_count: int
    rectangle_queries: int
    tree_nodes_visited: int
    point_probes: int = 0

    def dominant_query_create_events(self) -> tuple[InfluenceEvent, ...]:
        return tuple(
            sorted(
                (cell.query_create_event(self.flip) for cell in self.cells),
                key=lambda event: event.key_position,
            )
        )


def _convert_runs(runs: Sequence[LceRankRun]) -> list[LceAxisBand]:
    return [
        LceAxisBand(run.rank_start, run.rank_stop, run.lce) for run in runs
    ]


def _axis_bands(
    geometry: SemanticLceGeometry,
    *,
    context: str,
    fixed_position: int,
) -> tuple[LceAxisBand, ...]:
    run_index = (
        geometry.forward_runs if context == "right" else geometry.reverse_runs
    )
    target_rank = geometry.context_rank(context, "query", fixed_position)
    virtual_rank = run_index.rank_count
    if target_rank is None:
        return (LceAxisBand(0, virtual_rank + 1, 0),)
    bands = _convert_runs(run_index.runs(target_rank))
    boundary = LceAxisBand(virtual_rank, virtual_rank + 1, 0)
    if bands and bands[-1].rank_stop == boundary.rank_start and (
        bands[-1].lce == 0
    ):
        previous = bands[-1]
        bands[-1] = LceAxisBand(
            previous.rank_start, boundary.rank_stop, 0
        )
    else:
        bands.append(boundary)
    return tuple(bands)


def _enumerate_nonempty_cells_boxes(
    index: CodePartitionedOrthogonalIndex,
    code: int,
    forward_bands: tuple[LceAxisBand, ...],
    reverse_bands: tuple[LceAxisBand, ...],
) -> tuple[tuple[OrthogonalLceCell, ...], int]:
    cells: dict[tuple[int, int], tuple[OrthogonalAggregate, int]] = {}
    query_start = index.rectangle_queries

    def visit(
        forward_first: int,
        forward_last: int,
        reverse_first: int,
        reverse_last: int,
    ) -> None:
        aggregate = index.rectangle(
            code,
            forward_bands[forward_first].rank_start,
            forward_bands[forward_last - 1].rank_stop,
            reverse_bands[reverse_first].rank_start,
            reverse_bands[reverse_last - 1].rank_stop,
        )
        if aggregate.count == 0:
            return
        if forward_last - forward_first == 1 and (
            reverse_last - reverse_first == 1
        ):
            key = (
                reverse_bands[reverse_first].lce,
                forward_bands[forward_first].lce,
            )
            previous, rectangles = cells.get(
                key, (OrthogonalAggregate(), 0)
            )
            cells[key] = (previous.combine(aggregate), rectangles + 1)
            return
        if forward_last - forward_first >= reverse_last - reverse_first and (
            forward_last - forward_first > 1
        ):
            middle = (forward_first + forward_last) // 2
            visit(forward_first, middle, reverse_first, reverse_last)
            visit(middle, forward_last, reverse_first, reverse_last)
        else:
            middle = (reverse_first + reverse_last) // 2
            visit(forward_first, forward_last, reverse_first, middle)
            visit(forward_first, forward_last, middle, reverse_last)

    if forward_bands and reverse_bands:
        visit(0, len(forward_bands), 0, len(reverse_bands))
    result = tuple(
        OrthogonalLceCell(left, right, aggregate, rectangles)
        for (left, right), (aggregate, rectangles) in sorted(cells.items())
    )
    return result, index.rectangle_queries - query_start


def _enumerate_nonempty_cells_tree(
    index: CodePartitionedOrthogonalIndex,
    code: int,
    forward_bands: tuple[LceAxisBand, ...],
    reverse_bands: tuple[LceAxisBand, ...],
) -> tuple[tuple[OrthogonalLceCell, ...], int]:
    groups, nodes_visited = index.group_by_bands(
        code, forward_bands, reverse_bands
    )
    cells: dict[tuple[int, int], tuple[OrthogonalAggregate, int]] = {}
    for (forward_band, reverse_band), aggregate in groups.items():
        key = (
            reverse_bands[reverse_band].lce,
            forward_bands[forward_band].lce,
        )
        previous, rectangles = cells.get(key, (OrthogonalAggregate(), 0))
        cells[key] = (previous.combine(aggregate), rectangles + 1)
    result = tuple(
        OrthogonalLceCell(left, right, aggregate, rectangles)
        for (left, right), (aggregate, rectangles) in sorted(cells.items())
    )
    return result, nodes_visited


class CausalQueryCreateOrthogonalBuilder:
    """Sweep key positions once and emit exact query-create LCE cells."""

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        *,
        geometry: SemanticLceGeometry | None = None,
        occurrence_index: CodeOccurrenceIndex | None = None,
        enumeration_backend: str = "tree",
    ) -> None:
        _validate_codes(query_codes, key_codes)
        self.query_codes = query_codes
        self.key_codes = key_codes
        self.geometry = geometry or SemanticLceGeometry(query_codes, key_codes)
        self.occurrence_index = occurrence_index or CodeOccurrenceIndex.build(
            query_codes, key_codes
        )
        self.sequence_length = query_codes.numel()
        if enumeration_backend not in {"tree", "boxes"}:
            raise ValueError("unknown orthogonal enumeration backend")
        self.enumeration_backend = enumeration_backend
        forward_virtual = self.geometry.forward_runs.rank_count
        reverse_virtual = self.geometry.reverse_runs.rank_count
        points = []
        for position in range(max(self.sequence_length - 1, 0)):
            forward_rank = self.geometry.context_rank(
                "right", "key", position
            )
            reverse_rank = self.geometry.context_rank(
                "left", "key", position
            )
            points.append(
                OrthogonalPoint(
                    self.occurrence_index.key_codes[position],
                    forward_virtual if forward_rank is None else forward_rank,
                    reverse_virtual if reverse_rank is None else reverse_rank,
                    position,
                )
            )
        self.index = CodePartitionedOrthogonalIndex(points)
        self.next_key_position = 0
        self.last_query_position = 0

    @property
    def logical_bytes(self) -> int:
        return self.index.logical_bytes

    def _advance(self, query_position: int) -> None:
        if query_position < self.last_query_position:
            raise ValueError("query flips must be processed in position order")
        while self.next_key_position < query_position:
            self.index.activate(self.next_key_position)
            self.next_key_position += 1
        self.last_query_position = query_position

    def build(self, flip: BitFlip) -> OrthogonalCellBatch:
        if flip.source != "query":
            raise ValueError("orthogonal query builder requires a query flip")
        if not 1 <= flip.position < self.sequence_length:
            raise ValueError("query flip is not semantically active")
        if not 0 <= flip.bit < 8:
            raise ValueError("flip bit is outside the packed code")
        self._advance(flip.position)
        code = self.occurrence_index.query_codes[flip.position] ^ (1 << flip.bit)
        active = self.index.aggregate(code)
        if active.count == 0:
            return OrthogonalCellBatch(flip, (), 0, 0, 0)
        if active.count == 1:
            key_position = active.max_position
            if key_position is None:
                raise RuntimeError("nonempty aggregate lost its position")
            cell = OrthogonalLceCell(
                self.geometry.left_matches(flip.position, key_position),
                self.geometry.right_matches(flip.position, key_position),
                active,
                1,
            )
            return OrthogonalCellBatch(flip, (cell,), 1, 0, 0)

        forward_bands = _axis_bands(
            self.geometry,
            context="right",
            fixed_position=flip.position,
        )
        reverse_bands = _axis_bands(
            self.geometry,
            context="left",
            fixed_position=flip.position,
        )
        if self.enumeration_backend == "tree":
            cells, tree_nodes_visited = _enumerate_nonempty_cells_tree(
                self.index, code, forward_bands, reverse_bands
            )
            rectangle_queries = 0
        else:
            cells, rectangle_queries = _enumerate_nonempty_cells_boxes(
                self.index, code, forward_bands, reverse_bands
            )
            tree_nodes_visited = 0
        occurrence_count = sum(cell.aggregate.count for cell in cells)
        if occurrence_count != active.count:
            raise RuntimeError("orthogonal cells do not partition active points")
        return OrthogonalCellBatch(
            flip,
            cells,
            occurrence_count,
            rectangle_queries,
            tree_nodes_visited,
        )
