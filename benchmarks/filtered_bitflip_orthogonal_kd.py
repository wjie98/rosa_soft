"""Post-v1 linear-space 3D reference for monotone-LCE cells."""

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
from benchmarks.filtered_bitflip_monotone import SemanticLceGeometry
from benchmarks.filtered_bitflip_orthogonal import (
    LceAxisBand,
    OrthogonalAggregate,
    OrthogonalCellBatch,
    OrthogonalLceCell,
    OrthogonalPoint,
    _axis_bands,
)


__all__ = [
    "CodePartitionedKdIndex",
    "StaticQueryCreateKdBuilder",
]


@dataclass(frozen=True)
class _KdNode:
    forward_start: int
    forward_stop: int
    reverse_start: int
    reverse_stop: int
    position_start: int
    position_stop: int
    count: int
    left: int
    right: int
    points: tuple[OrthogonalPoint, ...]

    @property
    def leaf(self) -> bool:
        return self.left < 0


class _CodeKdIndex:
    def __init__(
        self,
        points: Sequence[OrthogonalPoint],
        *,
        leaf_size: int,
        axis_order: tuple[int, int, int],
    ) -> None:
        if leaf_size <= 0:
            raise ValueError("kd leaf size must be positive")
        if sorted(axis_order) != [0, 1, 2]:
            raise ValueError("kd axis order must be a permutation of 0,1,2")
        self.axis_order = axis_order
        self.nodes: list[_KdNode | None] = []
        self.root = self._build(tuple(points), 0, leaf_size)
        if any(node is None for node in self.nodes):
            raise RuntimeError("kd construction left an uninitialized node")

    @staticmethod
    def _coordinate(point: OrthogonalPoint, axis: int) -> int:
        if axis == 0:
            return point.forward_rank
        if axis == 1:
            return point.reverse_rank
        return point.position

    def _build(
        self,
        points: tuple[OrthogonalPoint, ...],
        depth: int,
        leaf_size: int,
    ) -> int:
        if not points:
            raise ValueError("a kd node cannot be empty")
        node_index = len(self.nodes)
        self.nodes.append(None)
        forward_values = [point.forward_rank for point in points]
        reverse_values = [point.reverse_rank for point in points]
        position_values = [point.position for point in points]
        bounds = (
            min(forward_values),
            max(forward_values) + 1,
            min(reverse_values),
            max(reverse_values) + 1,
            min(position_values),
            max(position_values) + 1,
        )
        if len(points) <= leaf_size:
            self.nodes[node_index] = _KdNode(
                *bounds,
                len(points),
                -1,
                -1,
                points,
            )
            return node_index

        axis = self.axis_order[depth % 3]
        ordered = tuple(
            sorted(points, key=lambda point: self._coordinate(point, axis))
        )
        middle = len(ordered) // 2
        left = self._build(ordered[:middle], depth + 1, leaf_size)
        right = self._build(ordered[middle:], depth + 1, leaf_size)
        self.nodes[node_index] = _KdNode(
            *bounds,
            len(points),
            left,
            right,
            (),
        )
        return node_index

    @staticmethod
    def _band_index(
        bands: tuple[LceAxisBand, ...],
        starts: tuple[int, ...],
        rank: int,
    ) -> int:
        index = bisect_right(starts, rank) - 1
        if index < 0 or not (
            bands[index].rank_start <= rank < bands[index].rank_stop
        ):
            raise RuntimeError("kd point rank is outside LCE bands")
        return index

    def box(
        self,
        forward_start: int,
        forward_stop: int,
        reverse_start: int,
        reverse_stop: int,
        position_start: int,
        position_stop: int,
    ) -> tuple[OrthogonalAggregate, int, int]:
        bounds = (
            forward_start,
            forward_stop,
            reverse_start,
            reverse_stop,
            position_start,
            position_stop,
        )
        if min(bounds) < 0:
            raise ValueError("orthogonal box bounds must be nonnegative")
        if (
            forward_stop < forward_start
            or reverse_stop < reverse_start
            or position_stop < position_start
        ):
            raise ValueError("orthogonal box bounds must be ordered")
        if (
            forward_start == forward_stop
            or reverse_start == reverse_stop
            or position_start == position_stop
        ):
            return OrthogonalAggregate(), 0, 0

        nodes_visited = 0
        point_probes = 0

        def visit(node_index: int) -> OrthogonalAggregate:
            nonlocal nodes_visited, point_probes
            nodes_visited += 1
            node = self.nodes[node_index]
            if node is None:
                raise RuntimeError("missing kd node")
            if (
                node.forward_stop <= forward_start
                or node.forward_start >= forward_stop
                or node.reverse_stop <= reverse_start
                or node.reverse_start >= reverse_stop
                or node.position_stop <= position_start
                or node.position_start >= position_stop
            ):
                return OrthogonalAggregate()
            if (
                forward_start <= node.forward_start
                and node.forward_stop <= forward_stop
                and reverse_start <= node.reverse_start
                and node.reverse_stop <= reverse_stop
                and position_start <= node.position_start
                and node.position_stop <= position_stop
            ):
                return OrthogonalAggregate(
                    node.count,
                    node.position_start,
                    node.position_stop - 1,
                )
            if node.leaf:
                aggregate = OrthogonalAggregate()
                for point in node.points:
                    point_probes += 1
                    if (
                        forward_start <= point.forward_rank < forward_stop
                        and reverse_start <= point.reverse_rank < reverse_stop
                        and position_start <= point.position < position_stop
                    ):
                        aggregate = aggregate.combine(
                            OrthogonalAggregate(
                                1, point.position, point.position
                            )
                        )
                return aggregate
            return visit(node.left).combine(visit(node.right))

        return visit(self.root), nodes_visited, point_probes

    def group_position_range(
        self,
        position_start: int,
        position_stop: int,
        forward_bands: tuple[LceAxisBand, ...],
        reverse_bands: tuple[LceAxisBand, ...],
    ) -> tuple[
        dict[tuple[int, int], tuple[OrthogonalAggregate, int]],
        int,
        int,
    ]:
        if position_start < 0 or position_stop < position_start:
            raise ValueError("position range must be nonnegative and ordered")
        if position_start == position_stop:
            return {}, 0, 0
        forward_starts = tuple(band.rank_start for band in forward_bands)
        reverse_starts = tuple(band.rank_start for band in reverse_bands)
        groups: dict[
            tuple[int, int], tuple[OrthogonalAggregate, int]
        ] = {}
        nodes_visited = 0
        point_probes = 0

        def append(
            forward_band: int,
            reverse_band: int,
            aggregate: OrthogonalAggregate,
        ) -> None:
            key = (forward_band, reverse_band)
            previous, rectangles = groups.get(
                key, (OrthogonalAggregate(), 0)
            )
            groups[key] = (previous.combine(aggregate), rectangles + 1)

        def visit(node_index: int) -> None:
            nonlocal nodes_visited, point_probes
            nodes_visited += 1
            node = self.nodes[node_index]
            if node is None:
                raise RuntimeError("missing kd node")
            if (
                node.position_stop <= position_start
                or node.position_start >= position_stop
            ):
                return
            forward_first = self._band_index(
                forward_bands, forward_starts, node.forward_start
            )
            forward_last = self._band_index(
                forward_bands, forward_starts, node.forward_stop - 1
            )
            reverse_first = self._band_index(
                reverse_bands, reverse_starts, node.reverse_start
            )
            reverse_last = self._band_index(
                reverse_bands, reverse_starts, node.reverse_stop - 1
            )
            if (
                position_start <= node.position_start
                and node.position_stop <= position_stop
                and forward_first == forward_last
                and reverse_first == reverse_last
            ):
                append(
                    forward_first,
                    reverse_first,
                    OrthogonalAggregate(
                        node.count,
                        node.position_start,
                        node.position_stop - 1,
                    ),
                )
                return
            if node.leaf:
                for point in node.points:
                    point_probes += 1
                    if not position_start <= point.position < position_stop:
                        continue
                    append(
                        self._band_index(
                            forward_bands,
                            forward_starts,
                            point.forward_rank,
                        ),
                        self._band_index(
                            reverse_bands,
                            reverse_starts,
                            point.reverse_rank,
                        ),
                        OrthogonalAggregate(
                            1, point.position, point.position
                        ),
                    )
                return
            visit(node.left)
            visit(node.right)

        visit(self.root)
        return groups, nodes_visited, point_probes

    @property
    def logical_bytes(self) -> int:
        nodes = len(self.nodes)
        leaf_points = sum(
            len(node.points) for node in self.nodes if node is not None
        )
        return 40 * nodes + 16 * leaf_points


class CodePartitionedKdIndex:
    """Exact static 3D kd index with one tree per packed center code."""

    def __init__(
        self,
        points: Sequence[OrthogonalPoint],
        *,
        leaf_size: int = 4,
        axis_order: tuple[int, int, int] = (2, 1, 0),
    ) -> None:
        by_code: list[list[OrthogonalPoint]] = [[] for _ in range(256)]
        for point in points:
            by_code[point.code].append(point)
        self.by_code = {
            code: _CodeKdIndex(
                code_points,
                leaf_size=leaf_size,
                axis_order=axis_order,
            )
            for code, code_points in enumerate(by_code)
            if code_points
        }
        self.point_count = len(points)
        self.node_count = sum(
            len(index.nodes) for index in self.by_code.values()
        )
        self.logical_bytes = sum(
            index.logical_bytes for index in self.by_code.values()
        )

    def group_prefix(
        self,
        code: int,
        position_stop: int,
        forward_bands: tuple[LceAxisBand, ...],
        reverse_bands: tuple[LceAxisBand, ...],
    ) -> tuple[
        dict[tuple[int, int], tuple[OrthogonalAggregate, int]],
        int,
        int,
    ]:
        return self.group_position_range(
            code,
            0,
            position_stop,
            forward_bands,
            reverse_bands,
        )

    def box(
        self,
        code: int,
        forward_start: int,
        forward_stop: int,
        reverse_start: int,
        reverse_stop: int,
        position_start: int,
        position_stop: int,
    ) -> tuple[OrthogonalAggregate, int, int]:
        """Aggregate one half-open code-filtered 3D box exactly."""

        if not 0 <= code < 256:
            raise ValueError("orthogonal code must fit in one byte")
        bounds = (
            forward_start,
            forward_stop,
            reverse_start,
            reverse_stop,
            position_start,
            position_stop,
        )
        if min(bounds) < 0:
            raise ValueError("orthogonal box bounds must be nonnegative")
        if (
            forward_stop < forward_start
            or reverse_stop < reverse_start
            or position_stop < position_start
        ):
            raise ValueError("orthogonal box bounds must be ordered")
        if (
            forward_start == forward_stop
            or reverse_start == reverse_stop
            or position_start == position_stop
        ):
            return OrthogonalAggregate(), 0, 0
        index = self.by_code.get(code)
        if index is None:
            return OrthogonalAggregate(), 0, 0
        return index.box(
            forward_start,
            forward_stop,
            reverse_start,
            reverse_stop,
            position_start,
            position_stop,
        )

    def group_position_range(
        self,
        code: int,
        position_start: int,
        position_stop: int,
        forward_bands: tuple[LceAxisBand, ...],
        reverse_bands: tuple[LceAxisBand, ...],
    ) -> tuple[
        dict[tuple[int, int], tuple[OrthogonalAggregate, int]],
        int,
        int,
    ]:
        """Group an arbitrary position interval by two LCE partitions."""

        if not 0 <= code < 256:
            raise ValueError("orthogonal code must fit in one byte")
        if position_start < 0 or position_stop < position_start:
            raise ValueError("position range must be nonnegative and ordered")
        if position_start == position_stop:
            return {}, 0, 0
        index = self.by_code.get(code)
        if index is None:
            return {}, 0, 0
        return index.group_position_range(
            position_start,
            position_stop,
            forward_bands,
            reverse_bands,
        )


class StaticQueryCreateKdBuilder:
    """Emit exact query-create cells from a linear-space static 3D index."""

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        *,
        geometry: SemanticLceGeometry | None = None,
        occurrence_index: CodeOccurrenceIndex | None = None,
        leaf_size: int = 4,
        axis_order: tuple[int, int, int] = (2, 1, 0),
    ) -> None:
        _validate_codes(query_codes, key_codes)
        self.query_codes = query_codes
        self.key_codes = key_codes
        self.geometry = geometry or SemanticLceGeometry(query_codes, key_codes)
        self.occurrence_index = occurrence_index or CodeOccurrenceIndex.build(
            query_codes, key_codes
        )
        self.sequence_length = query_codes.numel()
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
        self.index = CodePartitionedKdIndex(
            points,
            leaf_size=leaf_size,
            axis_order=axis_order,
        )

    @property
    def logical_bytes(self) -> int:
        return self.index.logical_bytes

    def build(self, flip: BitFlip) -> OrthogonalCellBatch:
        if flip.source != "query":
            raise ValueError("static kd builder requires a query flip")
        if not 1 <= flip.position < self.sequence_length:
            raise ValueError("query flip is not semantically active")
        if not 0 <= flip.bit < 8:
            raise ValueError("flip bit is outside the packed code")
        code = self.occurrence_index.query_codes[flip.position] ^ (1 << flip.bit)
        positions = self.occurrence_index.key_positions[code]
        active_count = bisect_left(positions, flip.position)
        if active_count == 0:
            return OrthogonalCellBatch(flip, (), 0, 0, 0)
        if active_count == 1:
            key_position = positions[0]
            cell = OrthogonalLceCell(
                self.geometry.left_matches(flip.position, key_position),
                self.geometry.right_matches(flip.position, key_position),
                OrthogonalAggregate(1, key_position, key_position),
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
        groups, nodes_visited, point_probes = self.index.group_prefix(
            code,
            flip.position,
            forward_bands,
            reverse_bands,
        )
        cells: dict[
            tuple[int, int], tuple[OrthogonalAggregate, int]
        ] = {}
        for (forward_band, reverse_band), value in groups.items():
            aggregate, rectangles = value
            key = (
                reverse_bands[reverse_band].lce,
                forward_bands[forward_band].lce,
            )
            previous, previous_rectangles = cells.get(
                key, (OrthogonalAggregate(), 0)
            )
            cells[key] = (
                previous.combine(aggregate),
                previous_rectangles + rectangles,
            )
        result = tuple(
            OrthogonalLceCell(left, right, aggregate, rectangles)
            for (left, right), (aggregate, rectangles) in sorted(cells.items())
        )
        occurrence_count = sum(cell.aggregate.count for cell in result)
        if occurrence_count != active_count:
            raise RuntimeError("kd cells do not partition the causal prefix")
        return OrthogonalCellBatch(
            flip,
            result,
            occurrence_count,
            rectangle_queries=0,
            tree_nodes_visited=nodes_visited,
            point_probes=point_probes,
        )
