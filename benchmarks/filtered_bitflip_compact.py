"""Frozen v1 exact solver structures for filtered bitflip research.

This remains a CPU research implementation.  It avoids the quadratic route
postings and precomputed query-ancestor paths used by the first suffix-node
prototype while preserving exact latest-longest ROSA semantics.  New index
routes should use a separate module instead of extending this dispatcher.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import torch
from torch import Tensor

from benchmarks.filtered_bitflip import (
    BitFlip,
    CodeOccurrenceIndex,
    InfluenceEvent,
    MatchLengthIndex,
    _validate_codes,
    generate_influence_events,
    semantic_bit_flips,
)
from benchmarks.filtered_bitflip_certificates import (
    PeriodicEventFamily,
    PeriodicEventFamilyIndex,
    Priority,
    WinnerSegment,
    build_periodic_event_families,
)
from benchmarks.filtered_bitflip_indexes import (
    LibsaisSuffixArrayMatchIndex,
    SuffixArrayMatchIndex,
    _SuffixArrayPair,
)
from benchmarks.filtered_bitflip_monotone import (
    SemanticLceGeometry,
    build_query_create_lce_cells,
)
from benchmarks.filtered_bitflip_periodic import (
    AffineWinnerSegment,
    ShiftAlignedPeriodicKeyFallback,
    ShiftAlignedQuerySuffixFallback,
    ShiftPeriodicEventFamilyBuilder,
    UniformPeriodicSuffixFallback,
    materialize_affine_winner_segments,
)
from benchmarks.filtered_bitflip_suffix_nodes import (
    ArithmeticRouteRun,
    SuffixQueryStats,
    compress_route_runs,
)


__all__ = [
    "AffineRouteRange",
    "CompactBitflipResult",
    "CompactCertificateStats",
    "CompactReplacementCertificateTree",
    "CompactSuffixNode",
    "CompactSuffixNodeReplacementIndex",
    "ExcludedRouteIndex",
    "FILTERED_BITFLIP_INDEX_VERSION",
    "PackedRankBits",
    "PeriodicRouteRange",
    "WaveletMatrix",
    "affine_route_vjp",
    "compact_filtered_bitflip",
]


_WORD_BITS = 64
FILTERED_BITFLIP_INDEX_VERSION = "filtered-bitflip-index-v1"


class PackedRankBits:
    """Packed immutable bit vector with constant-time rank-one queries."""

    def __init__(self, bits: Iterable[bool]) -> None:
        words: list[int] = []
        word = 0
        size = 0
        for bit in bits:
            offset = size % _WORD_BITS
            if offset == 0 and size:
                words.append(word)
                word = 0
            if bit:
                word |= 1 << offset
            size += 1
        if size:
            words.append(word)
        prefix = [0]
        for packed in words:
            prefix.append(prefix[-1] + packed.bit_count())
        self.size = size
        self.words = tuple(words)
        self.word_prefix_ones = tuple(prefix)

    def rank1(self, stop: int) -> int:
        """Count set bits in ``[0, stop)``."""

        if not 0 <= stop <= self.size:
            raise IndexError("bit-vector rank boundary is out of range")
        word_index, offset = divmod(stop, _WORD_BITS)
        result = self.word_prefix_ones[word_index]
        if offset:
            result += (
                self.words[word_index] & ((1 << offset) - 1)
            ).bit_count()
        return result

    @property
    def logical_bytes(self) -> int:
        return 8 * len(self.words) + 4 * len(self.word_prefix_ones)


@dataclass(frozen=True)
class _WaveletLevel:
    bits: PackedRankBits
    zero_count: int


class WaveletMatrix:
    """Static integer sequence supporting exact range predecessor queries."""

    def __init__(
        self,
        values: Sequence[int],
        *,
        maximum_value: int | None = None,
    ) -> None:
        if any(value < 0 for value in values):
            raise ValueError("wavelet values must be nonnegative")
        observed = max(values, default=0)
        if maximum_value is None:
            maximum_value = observed
        if maximum_value < observed:
            raise ValueError("maximum_value is smaller than an input value")
        self.size = len(values)
        self.bit_width = max(1, maximum_value.bit_length())
        ordered = list(values)
        levels = []
        for shift in range(self.bit_width - 1, -1, -1):
            bits = PackedRankBits((value >> shift) & 1 for value in ordered)
            zeros = [value for value in ordered if not (value >> shift) & 1]
            ones = [value for value in ordered if (value >> shift) & 1]
            levels.append(_WaveletLevel(bits, len(zeros)))
            ordered = zeros + ones
        self.levels = tuple(levels)
        self.logical_bytes = sum(
            level.bits.logical_bytes + 4 for level in self.levels
        )

    def _validate_range(self, start: int, stop: int) -> None:
        if not 0 <= start <= stop <= self.size:
            raise IndexError("wavelet range is out of bounds")

    @staticmethod
    def _zero_rank(bits: PackedRankBits, index: int) -> int:
        return index - bits.rank1(index)

    def range_count_less(self, start: int, stop: int, upper: int) -> int:
        """Count values in ``[start, stop)`` strictly below ``upper``."""

        self._validate_range(start, stop)
        if start == stop or upper <= 0:
            return 0
        if upper >= 1 << self.bit_width:
            return stop - start
        count = 0
        for level_index, level in enumerate(self.levels):
            shift = self.bit_width - 1 - level_index
            left_zero = self._zero_rank(level.bits, start)
            right_zero = self._zero_rank(level.bits, stop)
            if (upper >> shift) & 1:
                count += right_zero - left_zero
                start = level.zero_count + level.bits.rank1(start)
                stop = level.zero_count + level.bits.rank1(stop)
            else:
                start = left_zero
                stop = right_zero
        return count

    def kth_smallest(self, start: int, stop: int, rank: int) -> int:
        """Return the zero-based ``rank``-th value in one sequence range."""

        self._validate_range(start, stop)
        if not 0 <= rank < stop - start:
            raise IndexError("wavelet order statistic is out of range")
        value = 0
        for level_index, level in enumerate(self.levels):
            shift = self.bit_width - 1 - level_index
            left_zero = self._zero_rank(level.bits, start)
            right_zero = self._zero_rank(level.bits, stop)
            zero_count = right_zero - left_zero
            if rank < zero_count:
                start = left_zero
                stop = right_zero
            else:
                rank -= zero_count
                value |= 1 << shift
                start = level.zero_count + level.bits.rank1(start)
                stop = level.zero_count + level.bits.rank1(stop)
        return value

    def range_predecessor(
        self,
        start: int,
        stop: int,
        limit: int,
    ) -> int | None:
        """Return the largest value no greater than ``limit`` in the range."""

        count = self.range_count_less(start, stop, limit + 1)
        if count == 0:
            return None
        return self.kth_smallest(start, stop, count - 1)


@dataclass(frozen=True)
class CompactSuffixNode:
    """One LCP interval with parent topology and global-wavelet bounds."""

    depth: int
    rank_start: int
    rank_stop: int
    parent: int
    key_start: int
    key_stop: int
    route_run: ArithmeticRouteRun | None


class ExcludedRouteIndex:
    """Exact excluded routes represented by points and arithmetic runs."""

    def __init__(
        self,
        *,
        routes: Iterable[int] = (),
        runs: Iterable[ArithmeticRouteRun] = (),
    ) -> None:
        self.routes = frozenset(routes)
        route_runs = list(runs)
        if self.routes:
            route_runs.extend(compress_route_runs(tuple(sorted(self.routes))))
        self.runs = tuple(sorted(route_runs, key=lambda run: (run.first, run.last)))

    @classmethod
    def from_routes(cls, routes: set[int]) -> "ExcludedRouteIndex":
        return cls(routes=routes)

    def contains(self, route: int) -> bool:
        return route in self.routes or any(
            run.contains(route) for run in self.runs
        )

    def containing_run(self, route: int) -> ArithmeticRouteRun | None:
        matches = [run for run in self.runs if run.contains(route)]
        if not matches:
            return None
        contiguous = [run for run in matches if run.step == 1]
        return min(contiguous or matches, key=lambda run: run.first)


class CompactSuffixNodeReplacementIndex:
    """Linear-topology suffix index with one global route wavelet matrix."""

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        *,
        suffix_backend: str = "python",
        library_path: Path | str | None = None,
        periodic_min_count: int = 4,
    ) -> None:
        _validate_codes(query_codes, key_codes)
        if periodic_min_count < 2:
            raise ValueError("periodic_min_count must be at least two")
        query = tuple(int(value) for value in query_codes.tolist())
        key = tuple(int(value) for value in key_codes.tolist())
        self.sequence_length = len(query)
        pair = _SuffixArrayPair(
            query[::-1],
            key[::-1],
            backend=suffix_backend,
            library_path=library_path,
        )
        raw_nodes = sorted(
            self._lcp_intervals(pair.lcp),
            key=lambda item: (item[1], -item[2], item[0]),
        )
        topology: list[tuple[int, int, int, int]] = []
        stack: list[int] = []
        for depth, rank_start, rank_stop in raw_nodes:
            while stack and not (
                topology[stack[-1]][1] <= rank_start
                and rank_stop <= topology[stack[-1]][2]
            ):
                stack.pop()
            parent = stack[-1] if stack else -1
            topology.append((depth, rank_start, rank_stop, parent))
            stack.append(len(topology) - 1)

        key_rank_routes = sorted(
            (
                pair.inverse[
                    self.sequence_length
                    + 1
                    + self.sequence_length
                    - route
                ],
                route,
            )
            for route in range(1, self.sequence_length)
        )
        self.key_ranks = tuple(rank for rank, _ in key_rank_routes)
        route_sequence = tuple(route for _, route in key_rank_routes)
        self.routes = WaveletMatrix(
            route_sequence,
            maximum_value=max(self.sequence_length - 1, 0),
        )

        nodes = []
        periodic_nodes = 0
        for depth, rank_start, rank_stop, parent in topology:
            key_start = bisect_left(self.key_ranks, rank_start)
            key_stop = bisect_left(self.key_ranks, rank_stop)
            count = key_stop - key_start
            route_run = None
            if count >= periodic_min_count:
                first = self.routes.kth_smallest(key_start, key_stop, 0)
                last = self.routes.kth_smallest(
                    key_start, key_stop, count - 1
                )
                if last - first + 1 == count:
                    route_run = ArithmeticRouteRun(first, last, 1)
                    periodic_nodes += 1
            nodes.append(
                CompactSuffixNode(
                    depth,
                    rank_start,
                    rank_stop,
                    parent,
                    key_start,
                    key_stop,
                    route_run,
                )
            )
        self.nodes = tuple(nodes)
        self.periodic_nodes = periodic_nodes

        starts: list[list[int]] = [[] for _ in range(pair.text_length)]
        for node_index, node in enumerate(self.nodes):
            starts[node.rank_start].append(node_index)
        active: list[int] = []
        deepest_by_rank = [-1] * pair.text_length
        for rank in range(pair.text_length):
            while active and self.nodes[active[-1]].rank_stop <= rank:
                active.pop()
            active.extend(starts[rank])
            if active:
                deepest_by_rank[rank] = active[-1]
        self.query_leaf_nodes = tuple(
            deepest_by_rank[pair.inverse[self.sequence_length - 1 - index]]
            for index in range(self.sequence_length)
        )
        self.logical_bytes = (
            28 * len(self.nodes)
            + 4 * len(self.key_ranks)
            + self.routes.logical_bytes
            + 4 * len(self.query_leaf_nodes)
        )

    @staticmethod
    def _lcp_intervals(lcp: Sequence[int]) -> tuple[tuple[int, int, int], ...]:
        stack: list[tuple[int, int]] = []
        intervals = []
        for rank in range(1, len(lcp)):
            depth = lcp[rank]
            left = rank - 1
            while stack and stack[-1][0] > depth:
                previous_depth, previous_left = stack.pop()
                intervals.append((previous_depth, previous_left, rank))
                left = previous_left
            if depth > 0 and (not stack or stack[-1][0] < depth):
                stack.append((depth, left))
        while stack:
            depth, left = stack.pop()
            intervals.append((depth, left, len(lcp)))
        return tuple(intervals)

    def _wavelet_predecessor(
        self,
        node: CompactSuffixNode,
        limit: int,
        excluded: ExcludedRouteIndex,
    ) -> tuple[int | None, int]:
        probes = 0
        route = self.routes.range_predecessor(
            node.key_start, node.key_stop, limit
        )
        while route is not None:
            probes += 1
            if not excluded.contains(route):
                return route, probes
            excluded_run = excluded.containing_run(route)
            next_limit = (
                excluded_run.first - 1
                if excluded_run is not None and excluded_run.step == 1
                else route - 1
            )
            route = self.routes.range_predecessor(
                node.key_start, node.key_stop, next_limit
            )
        return None, probes

    @staticmethod
    def _run_predecessor(
        run: ArithmeticRouteRun,
        limit: int,
        excluded: ExcludedRouteIndex,
    ) -> tuple[int | None, int, int]:
        route = run.predecessor(limit)
        probes = 0
        jumps = 0
        while route is not None and route >= run.first:
            probes += 1
            excluded_run = excluded.containing_run(route)
            if excluded_run is None:
                return route, probes, jumps
            if excluded_run.step == 1:
                route = run.predecessor(excluded_run.first - 1)
                jumps += 1
            else:
                route -= 1
        return None, probes, jumps

    def best_excluding(
        self,
        query_index: int,
        excluded_routes: set[int] | ExcludedRouteIndex,
        *,
        backend: str = "auto",
    ) -> tuple[int, int, SuffixQueryStats]:
        """Return exact best unchanged ``(length, route)`` for one row."""

        if not 0 <= query_index < self.sequence_length:
            raise ValueError("query index is outside the suffix index")
        if backend not in {"auto", "wavelet"}:
            raise ValueError(f"unknown compact predecessor backend: {backend}")
        excluded = (
            excluded_routes
            if isinstance(excluded_routes, ExcludedRouteIndex)
            else ExcludedRouteIndex.from_routes(excluded_routes)
        )
        nodes_visited = 0
        probes = 0
        jumps = 0
        node_index = self.query_leaf_nodes[query_index]
        while node_index >= 0:
            node = self.nodes[node_index]
            nodes_visited += 1
            if backend == "auto" and node.route_run is not None:
                route, local_probes, local_jumps = self._run_predecessor(
                    node.route_run, query_index, excluded
                )
                jumps += local_jumps
            else:
                route, local_probes = self._wavelet_predecessor(
                    node, query_index, excluded
                )
            probes += local_probes
            if route is not None:
                return node.depth, route, SuffixQueryStats(
                    nodes_visited, probes, jumps
                )
            node_index = node.parent
        return 0, 0, SuffixQueryStats(nodes_visited, probes, jumps)


def _maximum(
    left: Priority | None,
    right: Priority | None,
) -> Priority | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _merge_segments(
    segments: list[WinnerSegment],
) -> tuple[WinnerSegment, ...]:
    if not segments:
        return ()
    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if previous.stop == segment.start and previous.priority == segment.priority:
            merged[-1] = WinnerSegment(
                previous.start, segment.stop, previous.priority
            )
        else:
            merged.append(segment)
    return tuple(merged)


@dataclass(frozen=True)
class CompactCertificateStats:
    tree_nodes_visited: int
    top_candidates_probed: int
    family_candidates_examined: int
    suffix_fallback_rows: int
    periodic_fallback_intervals: int
    periodic_fallback_rows: int
    periodic_predecessor_probes: int
    periodic_contiguous_jumps: int
    emitted_segments: int
    merged_segments: int


class CompactReplacementCertificateTree:
    """Bounded top-certificate tree with exact suffix fallback at leaves."""

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        suffix_index: CompactSuffixNodeReplacementIndex,
        *,
        top_k: int = 4,
    ) -> None:
        _validate_codes(query_codes, key_codes)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        sequence_length = query_codes.numel()
        size = 1
        while size < max(sequence_length, 1):
            size *= 2
        self.sequence_length = sequence_length
        self.size = size
        self.top_k = top_k
        self.suffix_index = suffix_index
        self.periodic_fallback = UniformPeriodicSuffixFallback.build_if_supported(
            query_codes, key_codes
        )
        self.shift_query_fallback = (
            ShiftAlignedQuerySuffixFallback.build_if_supported(
                query_codes, key_codes
            )
        )
        self.periodic_key_fallback = (
            ShiftAlignedPeriodicKeyFallback.build_if_supported(
                query_codes, key_codes
            )
        )
        top: list[list[Priority]] = [[] for _ in range(2 * size)]
        tail_upper: list[Priority | None] = [None] * (2 * size)
        omitted_upper: list[Priority | None] = [None] * (2 * size)
        if self.periodic_fallback is not None:
            self.top = tuple(() for _ in range(2 * size))
            self.tail_upper = tuple(None for _ in range(2 * size))
            self.subtree_upper = tuple(None for _ in range(2 * size))
            self.omitted_upper = tuple(None for _ in range(2 * size))
            self.retained_intervals = 0
            self.top_postings = 0
            self.truncated_nodes = 0
            self.logical_bytes = 0
            return
        retained_intervals = 0

        def add_interval(
            start: int,
            stop: int,
            priority: Priority,
        ) -> None:
            nonlocal retained_intervals
            retained_intervals += 1
            for node in self._canonical_nodes(start, stop):
                candidates = top[node]
                candidates.append(priority)
                candidates.sort(reverse=True)
                if len(candidates) > top_k:
                    removed = candidates.pop()
                    tail_upper[node] = _maximum(tail_upper[node], removed)

        active: dict[Priority, int] = {}
        for query_index in range(sequence_length):
            excluded_routes: set[int] = set()
            row_top: list[Priority] = []
            row_tail: Priority | None = None
            for candidate_rank in range(top_k + 1):
                length, route, _ = suffix_index.best_excluding(
                    query_index, excluded_routes
                )
                if length <= 0:
                    break
                priority = (
                    length - query_index,
                    route - query_index,
                )
                excluded_routes.add(route)
                if candidate_rank < top_k:
                    row_top.append(priority)
                else:
                    row_tail = priority
            current = set(row_top)
            for priority in tuple(active):
                if priority not in current:
                    add_interval(active.pop(priority), query_index, priority)
            for priority in row_top:
                active.setdefault(priority, query_index)
            omitted_upper[size + query_index] = row_tail
        for priority, start in active.items():
            add_interval(start, sequence_length, priority)

        self.top = tuple(tuple(candidates) for candidates in top)
        self.tail_upper = tuple(tail_upper)
        subtree_upper: list[Priority | None] = [None] * (2 * size)
        for node in range(2 * size - 1, 0, -1):
            local = self.top[node][0] if self.top[node] else tail_upper[node]
            if node < size:
                local = _maximum(local, subtree_upper[2 * node])
                local = _maximum(local, subtree_upper[2 * node + 1])
                omitted_upper[node] = _maximum(
                    omitted_upper[2 * node], omitted_upper[2 * node + 1]
                )
            subtree_upper[node] = local
        self.subtree_upper = tuple(subtree_upper)
        self.omitted_upper = tuple(omitted_upper)
        self.retained_intervals = retained_intervals
        self.top_postings = sum(len(candidates) for candidates in self.top)
        self.truncated_nodes = sum(value is not None for value in self.tail_upper)
        self.logical_bytes = (
            8 * self.top_postings
            + 8 * self.truncated_nodes
            + 8 * sum(value is not None for value in self.subtree_upper)
            + 8 * sum(value is not None for value in self.omitted_upper)
        )

    def _canonical_nodes(self, start: int, stop: int) -> tuple[int, ...]:
        if not 0 <= start <= stop <= self.sequence_length:
            raise ValueError("certificate interval is invalid")
        result = []
        left = start + self.size
        right = stop + self.size
        while left < right:
            if left & 1:
                result.append(left)
                left += 1
            if right & 1:
                right -= 1
                result.append(right)
            left //= 2
            right //= 2
        return tuple(result)

    @staticmethod
    def _contains_priority(
        matches: Sequence[tuple[PeriodicEventFamily, tuple[int, int]]],
        priority: Priority,
    ) -> bool:
        return any(
            family.contains_old_priority(priority, index_range)
            for family, index_range in matches
        )

    def solve(
        self,
        events: tuple[InfluenceEvent, ...],
    ) -> tuple[
        tuple[WinnerSegment | AffineWinnerSegment, ...],
        CompactCertificateStats,
    ]:
        families = build_periodic_event_families(events)
        if (
            self.shift_query_fallback is not None
            and events
            and events[0].flip.source == "query"
        ):
            segments, periodic_stats = (
                self.shift_query_fallback.solve_query_flip(
                    events[0].flip.position, families
                )
            )
            return segments, CompactCertificateStats(
                tree_nodes_visited=1,
                top_candidates_probed=0,
                family_candidates_examined=0,
                suffix_fallback_rows=0,
                periodic_fallback_intervals=1,
                periodic_fallback_rows=periodic_stats.rows_solved,
                periodic_predecessor_probes=0,
                periodic_contiguous_jumps=0,
                emitted_segments=periodic_stats.emitted_segments,
                merged_segments=periodic_stats.emitted_segments,
            )
        if self.periodic_fallback is not None:
            if events and events[0].flip.source == "key":
                segments, periodic_stats = (
                    self.periodic_fallback.solve_key_flip(
                        events[0].flip.position, families
                    )
                )
            else:
                segments, periodic_stats = self.periodic_fallback.solve(
                    0, self.sequence_length, families
                )
            return segments, CompactCertificateStats(
                tree_nodes_visited=1,
                top_candidates_probed=0,
                family_candidates_examined=0,
                suffix_fallback_rows=0,
                periodic_fallback_intervals=1,
                periodic_fallback_rows=periodic_stats.rows_solved,
                periodic_predecessor_probes=(
                    periodic_stats.predecessor_probes
                ),
                periodic_contiguous_jumps=periodic_stats.contiguous_jumps,
                emitted_segments=periodic_stats.emitted_segments,
                merged_segments=periodic_stats.emitted_segments,
            )
        if (
            self.periodic_key_fallback is not None
            and events
            and events[0].flip.source == "key"
            and self.periodic_key_fallback.supports_position(
                events[0].flip.position
            )
        ):
            segments, periodic_stats = (
                self.periodic_key_fallback.solve_key_flip(
                    events[0].flip.position, families
                )
            )
            return segments, CompactCertificateStats(
                tree_nodes_visited=1,
                top_candidates_probed=0,
                family_candidates_examined=0,
                suffix_fallback_rows=0,
                periodic_fallback_intervals=1,
                periodic_fallback_rows=periodic_stats.rows_solved,
                periodic_predecessor_probes=0,
                periodic_contiguous_jumps=0,
                emitted_segments=periodic_stats.emitted_segments,
                merged_segments=periodic_stats.emitted_segments,
            )
        event_index = PeriodicEventFamilyIndex(families)
        active_sources: list[tuple[Priority, ...]] = []
        active_tails: list[Priority] = []
        segments: list[WinnerSegment] = []
        nodes_visited = 0
        candidates_probed = 0
        families_examined = 0
        fallback_rows = 0

        def family_queries(start: int, stop: int, *, leaf: bool):
            nonlocal families_examined
            break_cover, examined = event_index.event_covering(
                start, stop, creates_match=False
            )
            families_examined += examined
            add_cover, examined = event_index.add_covering(start, stop)
            families_examined += examined
            if leaf:
                break_intersect = break_cover
                add_intersect = add_cover
            else:
                break_intersect, examined = event_index.event_intersecting(
                    start, stop, creates_match=False
                )
                families_examined += examined
                add_intersect, examined = event_index.add_intersecting(
                    start, stop
                )
                families_examined += examined
            return break_cover, break_intersect, add_cover, add_intersect

        def exact_leaf(
            query_index: int,
            break_cover,
            add_cover,
        ) -> Priority | None:
            nonlocal fallback_rows
            fallback_rows += 1
            excluded_runs = []
            for family, index_range in break_cover:
                first_phase, stop_phase = index_range
                first_route = (
                    query_index + family.old_priority(first_phase)[1]
                )
                last_route = (
                    query_index + family.old_priority(stop_phase - 1)[1]
                )
                step = abs(family.delta_route_offset)
                excluded_runs.append(
                    ArithmeticRouteRun(
                        min(first_route, last_route),
                        max(first_route, last_route),
                        step if step else 1,
                    )
                )
            length, route, _ = self.suffix_index.best_excluding(
                query_index, ExcludedRouteIndex(runs=excluded_runs)
            )
            best = (
                (length - query_index, route - query_index)
                if length > 0
                else None
            )
            for family, index_range in add_cover:
                best = _maximum(
                    best, family.maximum_added_priority(index_range)
                )
            return best

        def visit(node: int, start: int, stop: int) -> None:
            nonlocal nodes_visited, candidates_probed
            if start >= self.sequence_length:
                return
            nodes_visited += 1
            clipped_stop = min(stop, self.sequence_length)
            source_added = bool(self.top[node])
            if source_added:
                active_sources.append(self.top[node])
            tail = self.tail_upper[node]
            if tail is not None:
                active_tails.append(tail)

            break_cover, break_intersect, add_cover, add_intersect = (
                family_queries(
                    start,
                    clipped_stop,
                    leaf=node >= self.size,
                )
            )
            best: Priority | None = None
            selected_base: Priority | None = None
            for family, index_range in add_cover:
                best = _maximum(
                    best, family.maximum_added_priority(index_range)
                )
            for source in active_sources:
                for priority in source:
                    candidates_probed += 1
                    if self._contains_priority(break_cover, priority):
                        continue
                    if best is None or priority > best:
                        best = priority
                        selected_base = priority
                    break

            upper = self.subtree_upper[node]
            if active_tails:
                upper = _maximum(upper, max(active_tails))
            upper = _maximum(upper, self.omitted_upper[node])
            for family, index_range in add_intersect:
                upper = _maximum(
                    upper, family.maximum_added_priority(index_range)
                )
            stable = best is not None and (upper is None or best >= upper)
            if selected_base is not None and self._contains_priority(
                break_intersect, selected_base
            ):
                stable = False
            if best is None and upper is None:
                stable = True

            if not stable and node >= self.size:
                leaf_upper = self.omitted_upper[node]
                if active_tails:
                    leaf_upper = _maximum(leaf_upper, max(active_tails))
                stable = best is not None and (
                    leaf_upper is None or best >= leaf_upper
                )
                if best is None and leaf_upper is None:
                    stable = True

            if stable:
                segments.append(WinnerSegment(start, clipped_stop, best))
            elif node >= self.size:
                segments.append(
                    WinnerSegment(
                        start,
                        clipped_stop,
                        exact_leaf(start, break_cover, add_cover),
                    )
                )
            else:
                middle = (start + stop) // 2
                visit(2 * node, start, middle)
                visit(2 * node + 1, middle, stop)

            if tail is not None:
                active_tails.pop()
            if source_added:
                active_sources.pop()

        visit(1, 0, self.size)
        merged = _merge_segments(segments)
        return merged, CompactCertificateStats(
            tree_nodes_visited=nodes_visited,
            top_candidates_probed=candidates_probed,
            family_candidates_examined=families_examined,
            suffix_fallback_rows=fallback_rows,
            periodic_fallback_intervals=0,
            periodic_fallback_rows=0,
            periodic_predecessor_probes=0,
            periodic_contiguous_jumps=0,
            emitted_segments=len(segments),
            merged_segments=len(merged),
        )


@dataclass(frozen=True)
class AffineRouteRange:
    """Changed rows whose replacement route is one integer affine function."""

    start: int
    stop: int
    first_route: int
    route_step: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.start >= self.stop:
            raise ValueError("affine route range must be nonempty")

    def route_at(self, query_index: int) -> int:
        if not self.start <= query_index < self.stop:
            raise IndexError("query row is outside the affine route range")
        return self.first_route + self.route_step * (query_index - self.start)


@dataclass(frozen=True)
class PeriodicRouteRange:
    """Changed rows with a staircase-periodic replacement route offset."""

    start: int
    stop: int
    block_origin: int
    origin_route_offset: int
    rows_per_step: int
    route_offset_step: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.start >= self.stop:
            raise ValueError("periodic route range must be nonempty")
        if self.block_origin > self.start:
            raise ValueError("block origin cannot follow the range")
        if self.rows_per_step < 1:
            raise ValueError("rows_per_step must be positive")

    def route_at(self, query_index: int) -> int:
        if not self.start <= query_index < self.stop:
            raise IndexError("query row is outside the periodic route range")
        block = (query_index - self.block_origin) // self.rows_per_step
        return (
            query_index
            + self.origin_route_offset
            + self.route_offset_step * block
        )


RouteRange = AffineRouteRange | PeriodicRouteRange


def _base_winner_segments(
    routes: Tensor,
    lengths: Tensor,
) -> tuple[WinnerSegment, ...]:
    segments = []
    for query_index, (route, length) in enumerate(
        zip(routes.tolist(), lengths.tolist())
    ):
        priority = (
            (int(length) - query_index, int(route) - query_index)
            if length > 0
            else None
        )
        segment = WinnerSegment(query_index, query_index + 1, priority)
        if segments and segments[-1].priority == priority:
            previous = segments[-1]
            segments[-1] = WinnerSegment(
                previous.start, query_index + 1, priority
            )
        else:
            segments.append(segment)
    return tuple(segments)


def _segments_to_route_changes(
    base: tuple[WinnerSegment, ...],
    counterfactual: tuple[WinnerSegment | AffineWinnerSegment, ...],
) -> tuple[RouteRange, ...]:
    def can_keep_periodic(segment: AffineWinnerSegment) -> bool:
        if segment.rows_per_step == 1 or segment.first_priority is None:
            return False
        first_offset = segment.first_priority[1]
        last_priority = segment.priority_at(segment.stop - 1)
        if last_priority is None:
            return False
        last_offset = last_priority[1]
        if (
            first_offset == 0
            or last_offset == 0
            or (first_offset < 0) != (last_offset < 0)
        ):
            return False
        return all(
            base_segment.priority == (0, 0)
            for base_segment in base
            if (
                base_segment.start < segment.stop
                and segment.start < base_segment.stop
            )
        )

    expanded_counterfactual: list[WinnerSegment | AffineWinnerSegment] = []
    for segment in counterfactual:
        if not isinstance(segment, AffineWinnerSegment) or (
            segment.rows_per_step == 1
        ) or can_keep_periodic(segment):
            expanded_counterfactual.append(segment)
            continue
        block_start = segment.start
        while block_start < segment.stop:
            block_stop = min(
                block_start + segment.rows_per_step, segment.stop
            )
            expanded_counterfactual.append(
                WinnerSegment(
                    block_start,
                    block_stop,
                    segment.priority_at(block_start),
                )
            )
            block_start = block_stop
    counterfactual = tuple(expanded_counterfactual)

    def route_function(segment, start: int) -> tuple[int, int]:
        if isinstance(segment, WinnerSegment):
            if segment.priority is None:
                return 0, 0
            return start + segment.priority[1], 1
        priority = segment.priority_at(start)
        if priority is None:
            return 0, 0
        return start + priority[1], 1 + segment.priority_step[1]

    def append_change(start: int, stop: int, route: int, step: int) -> None:
        if start >= stop:
            return
        candidate = AffineRouteRange(start, stop, route, step)
        if (
            changes
            and isinstance(changes[-1], AffineRouteRange)
            and changes[-1].stop == candidate.start
            and changes[-1].route_step == candidate.route_step
            and changes[-1].route_at(changes[-1].stop - 1)
            + candidate.route_step
            == candidate.first_route
        ):
            previous = changes[-1]
            changes[-1] = AffineRouteRange(
                previous.start,
                candidate.stop,
                previous.first_route,
                previous.route_step,
            )
        else:
            changes.append(candidate)

    def append_periodic(
        start: int,
        stop: int,
        segment: AffineWinnerSegment,
    ) -> None:
        if start >= stop or segment.first_priority is None:
            return
        candidate = PeriodicRouteRange(
            start=start,
            stop=stop,
            block_origin=segment.start,
            origin_route_offset=segment.first_priority[1],
            rows_per_step=segment.rows_per_step,
            route_offset_step=segment.priority_step[1],
        )
        if (
            changes
            and isinstance(changes[-1], PeriodicRouteRange)
            and changes[-1].stop == candidate.start
            and changes[-1].block_origin == candidate.block_origin
            and (
                changes[-1].origin_route_offset
                == candidate.origin_route_offset
            )
            and changes[-1].rows_per_step == candidate.rows_per_step
            and (
                changes[-1].route_offset_step
                == candidate.route_offset_step
            )
        ):
            previous = changes[-1]
            changes[-1] = PeriodicRouteRange(
                previous.start,
                candidate.stop,
                previous.block_origin,
                previous.origin_route_offset,
                previous.rows_per_step,
                previous.route_offset_step,
            )
        else:
            changes.append(candidate)

    changes: list[RouteRange] = []
    base_index = 0
    changed_index = 0
    while base_index < len(base) and changed_index < len(counterfactual):
        base_segment = base[base_index]
        changed_segment = counterfactual[changed_index]
        start = max(base_segment.start, changed_segment.start)
        stop = min(base_segment.stop, changed_segment.stop)
        if start < stop:
            if (
                isinstance(changed_segment, AffineWinnerSegment)
                and changed_segment.rows_per_step > 1
            ):
                append_periodic(start, stop, changed_segment)
            else:
                base_route, base_step = route_function(base_segment, start)
                changed_route, changed_step = route_function(
                    changed_segment, start
                )
                route_delta = changed_route - base_route
                step_delta = changed_step - base_step
                equal_row: int | None = None
                if step_delta == 0:
                    if route_delta == 0:
                        equal_row = -1
                elif (-route_delta) % step_delta == 0:
                    relative = (-route_delta) // step_delta
                    if 0 <= relative < stop - start:
                        equal_row = start + relative
                if equal_row != -1:
                    if equal_row is None:
                        append_change(
                            start, stop, changed_route, changed_step
                        )
                    else:
                        append_change(
                            start,
                            equal_row,
                            changed_route,
                            changed_step,
                        )
                        after = equal_row + 1
                        append_change(
                            after,
                            stop,
                            changed_route + changed_step * (after - start),
                            changed_step,
                        )
        if base_segment.stop == stop:
            base_index += 1
        if changed_segment.stop == stop:
            changed_index += 1
    return tuple(changes)


def _updates_to_route_changes(
    base_routes: Tensor,
    updates: dict[int, tuple[int, int]],
) -> tuple[AffineRouteRange, ...]:
    points = [
        (query_index, route)
        for query_index, (_, route) in sorted(updates.items())
        if route != int(base_routes[query_index])
    ]
    if not points:
        return ()
    changes = []
    run_start = points[0][0]
    first_route = points[0][1]
    previous_index, previous_route = points[0]
    step: int | None = None
    for query_index, route in points[1:]:
        candidate_step = route - previous_route
        if (
            query_index == previous_index + 1
            and (step is None or candidate_step == step)
        ):
            step = candidate_step
        else:
            changes.append(
                AffineRouteRange(
                    run_start,
                    previous_index + 1,
                    first_route,
                    0 if step is None else step,
                )
            )
            run_start = query_index
            first_route = route
            step = None
        previous_index = query_index
        previous_route = route
    changes.append(
        AffineRouteRange(
            run_start,
            previous_index + 1,
            first_route,
            0 if step is None else step,
        )
    )
    return tuple(changes)


def affine_route_vjp(
    flips: Sequence[BitFlip],
    query_codes: Tensor,
    key_codes: Tensor,
    base_routes: Tensor,
    route_changes: Sequence[Sequence[RouteRange]],
    value: Tensor | None,
    grad_output: Tensor | None,
) -> Tensor | None:
    """Accumulate exact hard bitflip VJPs from compressed route ranges."""

    if value is None and grad_output is None:
        return None
    if value is None or grad_output is None:
        raise ValueError("value and grad_output must be supplied together")
    if value.shape != grad_output.shape or value.ndim != 2:
        raise ValueError("value and grad_output must share shape [T, Dv]")
    if not value.dtype.is_floating_point or not grad_output.dtype.is_floating_point:
        raise ValueError("value and grad_output must be floating point")
    if value.device != grad_output.device:
        raise ValueError("value and grad_output must be on one device")
    if len(flips) != len(route_changes):
        raise ValueError("one route-change sequence is required per flip")
    if value.size(0) != base_routes.numel():
        raise ValueError("value length must match base routes")
    hard_value = torch.where(value > 0, torch.ones_like(value), -torch.ones_like(value))
    hard_value = hard_value.clone()
    if hard_value.size(0):
        hard_value[0] = 0
    device = value.device
    base_on_device = base_routes.to(device=device)
    gradients = []
    for flip, ranges in zip(flips, route_changes):
        row_parts = []
        route_parts = []
        previous_stop = 0
        for change in ranges:
            if change.start < previous_stop or change.stop > base_routes.numel():
                raise ValueError("route-change ranges must be ordered and disjoint")
            rows = torch.arange(change.start, change.stop, device=device)
            if isinstance(change, AffineRouteRange):
                routes = change.first_route + change.route_step * (
                    rows - change.start
                )
            else:
                blocks = torch.div(
                    rows - change.block_origin,
                    change.rows_per_step,
                    rounding_mode="floor",
                )
                routes = (
                    rows
                    + change.origin_route_offset
                    + change.route_offset_step * blocks
                )
            if bool(((routes < 0) | (routes >= base_routes.numel())).any()):
                raise ValueError("an affine replacement route is out of range")
            row_parts.append(rows)
            route_parts.append(routes)
            previous_stop = change.stop
        if row_parts:
            rows = torch.cat(row_parts)
            routes = torch.cat(route_parts).to(dtype=torch.int64)
            delta = (
                (hard_value[routes] - hard_value[base_on_device[rows]])
                * grad_output[rows]
            ).sum()
        else:
            delta = value.new_zeros(())
        codes = query_codes if flip.source == "query" else key_codes
        base_symbol = (
            1.0
            if int(codes[flip.position]) & (1 << flip.bit)
            else -1.0
        )
        gradients.append(-base_symbol * delta)
    return torch.stack(gradients) if gradients else value.new_empty(0)


def _canonical_interval_count(
    sequence_length: int,
    start: int,
    stop: int,
) -> int:
    size = 1
    while size < max(sequence_length, 1):
        size *= 2
    count = 0
    left = start + size
    right = stop + size
    while left < right:
        if left & 1:
            count += 1
            left += 1
        if right & 1:
            right -= 1
            count += 1
        left //= 2
        right //= 2
    return count


def _overlay_entry_count(
    sequence_length: int,
    events: Sequence[InfluenceEvent],
) -> int:
    count = 0
    for event in events:
        if not event.creates_match:
            count += _canonical_interval_count(
                sequence_length, event.start, event.stop
            )
            add_start = event.start + 1
        else:
            add_start = event.start
        if add_start < event.stop:
            count += _canonical_interval_count(
                sequence_length, add_start, event.stop
            )
    return count


@dataclass(frozen=True)
class CompactBitflipResult:
    flips: tuple[BitFlip, ...]
    base_routes: Tensor
    base_lengths: Tensor
    route_changes: tuple[tuple[RouteRange, ...], ...]
    bit_gradient: Tensor | None
    flipped_routes: Tensor | None
    flipped_lengths: Tensor | None
    sparse_flips: int
    certificate_flips: int
    periodic_interval_flips: int
    empty_flips: int
    raw_events: int
    materialized_events: int
    processed_event_cells: int
    direct_event_family_flips: int
    direct_event_families: int
    monotone_lce_cell_flips: int
    monotone_lce_cells: int
    monotone_create_occurrences: int
    top_certificate_postings: int
    certificate_retained_intervals: int
    certificate_nodes_visited: int
    certificate_suffix_fallback_rows: int
    periodic_interval_rows: int
    periodic_winner_segments: int
    periodic_creation_intervals_expanded: int
    periodic_predecessor_probes: int
    periodic_contiguous_jumps: int
    sparse_replacement_rows: int
    changed_route_rows: int
    route_change_descriptors: int
    affine_equivalent_route_ranges: int
    periodic_route_ranges: int


@dataclass
class _CompactCounters:
    sparse_flips: int = 0
    certificate_flips: int = 0
    periodic_interval_flips: int = 0
    empty_flips: int = 0
    raw_events: int = 0
    materialized_events: int = 0
    processed_event_cells: int = 0
    direct_event_family_flips: int = 0
    direct_event_families: int = 0
    monotone_lce_cell_flips: int = 0
    monotone_lce_cells: int = 0
    monotone_create_occurrences: int = 0
    certificate_nodes_visited: int = 0
    certificate_suffix_fallback_rows: int = 0
    periodic_interval_rows: int = 0
    periodic_winner_segments: int = 0
    periodic_creation_intervals_expanded: int = 0
    periodic_predecessor_probes: int = 0
    periodic_contiguous_jumps: int = 0
    sparse_replacement_rows: int = 0
    changed_route_rows: int = 0
    periodic_route_ranges: int = 0


@dataclass(frozen=True)
class _PreparedFlip:
    flip: BitFlip
    events: tuple[InfluenceEvent, ...]
    families: tuple[PeriodicEventFamily, ...] | None
    event_count: int
    event_cells: int
    periodic_supported: bool


class _CompactBitflipSolver:
    """Shared indexes and explicit state transitions for one exact solve."""

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        bit_width: int,
        *,
        top_k: int,
        interval_compression: int,
        suffix_backend: str,
        library_path: Path | str | None,
        match_index: MatchLengthIndex | None,
        value: Tensor | None,
        grad_output: Tensor | None,
        materialize_routes: bool,
    ) -> None:
        _validate_codes(query_codes, key_codes)
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if interval_compression < 1:
            raise ValueError("interval_compression must be positive")
        self.query_codes = query_codes
        self.key_codes = key_codes
        self.sequence_length = query_codes.numel()
        self.top_k = top_k
        self.interval_compression = interval_compression
        self.value = value
        self.grad_output = grad_output
        self.materialize_routes = materialize_routes

        self.suffix_index = CompactSuffixNodeReplacementIndex(
            query_codes,
            key_codes,
            suffix_backend=suffix_backend,
            library_path=library_path,
        )
        if match_index is None:
            if suffix_backend == "libsais":
                match_index = LibsaisSuffixArrayMatchIndex(
                    query_codes, key_codes, library_path=library_path
                )
            else:
                match_index = SuffixArrayMatchIndex(query_codes, key_codes)
        self.match_index = match_index
        self.occurrence_index = CodeOccurrenceIndex.build(
            query_codes, key_codes
        )
        self.uniform_fallback = (
            UniformPeriodicSuffixFallback.build_if_supported(
                query_codes, key_codes
            )
        )
        self.shift_query_fallback = (
            ShiftAlignedQuerySuffixFallback.build_if_supported(
                query_codes, key_codes
            )
        )
        self.periodic_key_fallback = (
            ShiftAlignedPeriodicKeyFallback.build_if_supported(
                query_codes, key_codes
            )
        )
        self.periodic_family_builder = self._build_periodic_family_builder()
        self.monotone_geometry = self._build_monotone_geometry()

        base_pairs = [
            self.suffix_index.best_excluding(query_index, set())[:2]
            for query_index in range(self.sequence_length)
        ]
        self.base_lengths = torch.tensor(
            [length for length, _ in base_pairs], dtype=torch.int64
        )
        self.base_routes = torch.tensor(
            [route for _, route in base_pairs], dtype=torch.int64
        )
        self.base_segments = _base_winner_segments(
            self.base_routes, self.base_lengths
        )
        self.flips = semantic_bit_flips(self.sequence_length, bit_width)
        self.certificate_tree: CompactReplacementCertificateTree | None = None
        self.counters = _CompactCounters()
        self.route_changes: list[tuple[RouteRange, ...]] = []
        self.route_rows: list[Tensor] = []
        self.length_rows: list[Tensor] = []

    def _build_periodic_family_builder(
        self,
    ) -> ShiftPeriodicEventFamilyBuilder | None:
        if self.sequence_length <= 1:
            return None
        if self.uniform_fallback is not None:
            return ShiftPeriodicEventFamilyBuilder(
                self.query_codes, self.key_codes, 1
            )
        if self.periodic_key_fallback is not None:
            return ShiftPeriodicEventFamilyBuilder(
                self.query_codes,
                self.key_codes,
                self.periodic_key_fallback.period,
            )
        return None

    def _build_monotone_geometry(self) -> SemanticLceGeometry | None:
        if (
            self.shift_query_fallback is None
            or self.periodic_family_builder is not None
        ):
            return None
        shared_pairs = isinstance(self.match_index, SuffixArrayMatchIndex)
        return SemanticLceGeometry(
            self.query_codes,
            self.key_codes,
            forward_pair=(self.match_index.forward if shared_pairs else None),
            reverse_pair=(self.match_index.reverse if shared_pairs else None),
        )

    def _supports_periodic_interval(self, flip: BitFlip) -> bool:
        return (
            self.uniform_fallback is not None
            or (
                self.shift_query_fallback is not None
                and flip.source == "query"
            )
            or (
                self.periodic_key_fallback is not None
                and flip.source == "key"
                and self.periodic_key_fallback.supports_position(flip.position)
            )
        )

    def _prepare_flip(self, flip: BitFlip) -> _PreparedFlip:
        periodic_supported = self._supports_periodic_interval(flip)
        if self.periodic_family_builder is not None and periodic_supported:
            batch = self.periodic_family_builder.build(flip)
            events: tuple[InfluenceEvent, ...] = ()
            families = batch.families
            event_count = batch.event_count
            event_cells = batch.event_cells
            self.counters.direct_event_family_flips += 1
            self.counters.direct_event_families += len(families)
        elif self.monotone_geometry is not None and flip.source == "query":
            batch = build_query_create_lce_cells(
                self.query_codes,
                self.key_codes,
                flip,
                geometry=self.monotone_geometry,
                occurrence_index=self.occurrence_index,
            )
            events = batch.dominant_query_create_events()
            families = build_periodic_event_families(events)
            old_code = self.occurrence_index.query_codes[flip.position]
            old_positions = self.occurrence_index.key_positions[old_code]
            event_count = (
                bisect_left(old_positions, flip.position)
                + batch.occurrence_count
            )
            event_cells = sum(event.stop - event.start for event in events)
            self.counters.materialized_events += len(events)
            self.counters.monotone_lce_cell_flips += 1
            self.counters.monotone_lce_cells += len(batch.cells)
            self.counters.monotone_create_occurrences += batch.occurrence_count
        else:
            events = generate_influence_events(
                self.query_codes,
                self.key_codes,
                flip,
                match_index=self.match_index,
                occurrence_index=self.occurrence_index,
            )
            families = None
            event_count = len(events)
            event_cells = sum(event.stop - event.start for event in events)
            self.counters.materialized_events += event_count

        self.counters.raw_events += event_count
        self.counters.processed_event_cells += event_cells
        return _PreparedFlip(
            flip,
            events,
            families,
            event_count,
            event_cells,
            periodic_supported,
        )

    def _solve_periodic(
        self,
        prepared: _PreparedFlip,
    ) -> tuple[WinnerSegment | AffineWinnerSegment, ...]:
        families = prepared.families
        if families is None:
            families = build_periodic_event_families(prepared.events)
        flip = prepared.flip
        if flip.source == "query" and self.shift_query_fallback is not None:
            segments, stats = self.shift_query_fallback.solve_query_flip(
                flip.position, families
            )
        elif self.uniform_fallback is not None:
            if flip.source == "key":
                segments, stats = self.uniform_fallback.solve_key_flip(
                    flip.position, families
                )
            else:
                segments, stats = self.uniform_fallback.solve(
                    0, self.sequence_length, families
                )
        elif flip.source == "query":
            if self.shift_query_fallback is None:
                raise RuntimeError("missing shifted query interval solver")
            segments, stats = self.shift_query_fallback.solve_query_flip(
                flip.position, families
            )
        else:
            if self.periodic_key_fallback is None:
                raise RuntimeError("missing periodic key interval solver")
            segments, stats = self.periodic_key_fallback.solve_key_flip(
                flip.position, families
            )
        self.counters.periodic_interval_flips += 1
        self.counters.periodic_interval_rows += stats.rows_solved
        self.counters.periodic_winner_segments += stats.emitted_segments
        self.counters.periodic_creation_intervals_expanded += (
            stats.creation_intervals_expanded
        )
        self.counters.periodic_predecessor_probes += stats.predecessor_probes
        self.counters.periodic_contiguous_jumps += stats.contiguous_jumps
        return segments

    def _use_certificate(self, prepared: _PreparedFlip) -> bool:
        overlay_entries = _overlay_entry_count(
            self.sequence_length, prepared.events
        )
        return prepared.event_cells >= self.interval_compression * max(
            overlay_entries, 1
        )

    def _solve_certificate(
        self,
        events: tuple[InfluenceEvent, ...],
    ) -> tuple[WinnerSegment | AffineWinnerSegment, ...]:
        if self.certificate_tree is None:
            self.certificate_tree = CompactReplacementCertificateTree(
                self.query_codes,
                self.key_codes,
                self.suffix_index,
                top_k=self.top_k,
            )
        segments, stats = self.certificate_tree.solve(events)
        self.counters.certificate_flips += 1
        self.counters.certificate_nodes_visited += stats.tree_nodes_visited
        self.counters.certificate_suffix_fallback_rows += (
            stats.suffix_fallback_rows
        )
        self.counters.periodic_interval_rows += stats.periodic_fallback_rows
        self.counters.periodic_predecessor_probes += (
            stats.periodic_predecessor_probes
        )
        self.counters.periodic_contiguous_jumps += (
            stats.periodic_contiguous_jumps
        )
        return segments

    def _solve_sparse(
        self,
        events: tuple[InfluenceEvent, ...],
    ) -> tuple[tuple[RouteRange, ...], dict[int, tuple[int, int]]]:
        changed: dict[int, dict[int, int]] = {}
        for event in events:
            for query_index in range(event.start, event.stop):
                route = query_index + event.route_offset
                row = changed.setdefault(query_index, {})
                if route in row:
                    raise RuntimeError(
                        "influence events overlap one candidate cell"
                    )
                row[route] = event.new_length(query_index)

        updates: dict[int, tuple[int, int]] = {}
        for query_index, row in changed.items():
            self.counters.sparse_replacement_rows += 1
            best_length, best_route, _ = self.suffix_index.best_excluding(
                query_index, set(row)
            )
            for route, length in row.items():
                if length > best_length or (
                    length == best_length
                    and length > 0
                    and route > best_route
                ):
                    best_length = length
                    best_route = route
            updates[query_index] = (best_length, best_route)
        self.counters.sparse_flips += 1
        return _updates_to_route_changes(self.base_routes, updates), updates

    def _materialize_updates(
        self,
        updates: dict[int, tuple[int, int]],
    ) -> tuple[Tensor, Tensor]:
        routes = self.base_routes.clone()
        lengths = self.base_lengths.clone()
        for query_index, (length, route) in updates.items():
            routes[query_index] = route
            lengths[query_index] = length
        return routes, lengths

    def _solve_flip(
        self,
        prepared: _PreparedFlip,
    ) -> tuple[tuple[RouteRange, ...], Tensor | None, Tensor | None]:
        if prepared.event_count == 0:
            self.counters.empty_flips += 1
            routes = self.base_routes.clone() if self.materialize_routes else None
            lengths = (
                self.base_lengths.clone() if self.materialize_routes else None
            )
            return (), routes, lengths

        if prepared.periodic_supported:
            segments = self._solve_periodic(prepared)
            changes = _segments_to_route_changes(self.base_segments, segments)
            if self.materialize_routes:
                routes, lengths = materialize_affine_winner_segments(
                    self.sequence_length, segments
                )
            else:
                routes = lengths = None
            return changes, routes, lengths

        if self._use_certificate(prepared):
            segments = self._solve_certificate(prepared.events)
            changes = _segments_to_route_changes(self.base_segments, segments)
            if self.materialize_routes:
                routes, lengths = materialize_affine_winner_segments(
                    self.sequence_length, segments
                )
            else:
                routes = lengths = None
            return changes, routes, lengths

        changes, updates = self._solve_sparse(prepared.events)
        if self.materialize_routes:
            routes, lengths = self._materialize_updates(updates)
        else:
            routes = lengths = None
        return changes, routes, lengths

    def _record_flip(
        self,
        changes: tuple[RouteRange, ...],
        routes: Tensor | None,
        lengths: Tensor | None,
    ) -> None:
        self.route_changes.append(changes)
        self.counters.changed_route_rows += sum(
            change.stop - change.start for change in changes
        )
        self.counters.periodic_route_ranges += sum(
            isinstance(change, PeriodicRouteRange) for change in changes
        )
        if self.materialize_routes:
            if routes is None or lengths is None:
                raise RuntimeError("materialized solve did not return routes")
            self.route_rows.append(routes)
            self.length_rows.append(lengths)

    @staticmethod
    def _affine_equivalent_count(change: RouteRange) -> int:
        if not isinstance(change, PeriodicRouteRange):
            return 1
        return (
            (change.stop - 1 - change.block_origin) // change.rows_per_step
            - (change.start - change.block_origin) // change.rows_per_step
            + 1
        )

    def _stack_materialized(self, rows: list[Tensor]) -> Tensor | None:
        if not self.materialize_routes:
            return None
        if rows:
            return torch.stack(rows)
        return torch.empty(0, self.sequence_length, dtype=torch.int64)

    def _build_result(self) -> CompactBitflipResult:
        route_changes = tuple(self.route_changes)
        route_change_descriptors = sum(
            len(changes) for changes in route_changes
        )
        affine_equivalent_route_ranges = sum(
            self._affine_equivalent_count(change)
            for changes in route_changes
            for change in changes
        )
        counters = self.counters
        certificate = self.certificate_tree
        return CompactBitflipResult(
            flips=self.flips,
            base_routes=self.base_routes,
            base_lengths=self.base_lengths,
            route_changes=route_changes,
            bit_gradient=affine_route_vjp(
                self.flips,
                self.query_codes,
                self.key_codes,
                self.base_routes,
                route_changes,
                self.value,
                self.grad_output,
            ),
            flipped_routes=self._stack_materialized(self.route_rows),
            flipped_lengths=self._stack_materialized(self.length_rows),
            sparse_flips=counters.sparse_flips,
            certificate_flips=counters.certificate_flips,
            periodic_interval_flips=counters.periodic_interval_flips,
            empty_flips=counters.empty_flips,
            raw_events=counters.raw_events,
            materialized_events=counters.materialized_events,
            processed_event_cells=counters.processed_event_cells,
            direct_event_family_flips=counters.direct_event_family_flips,
            direct_event_families=counters.direct_event_families,
            monotone_lce_cell_flips=counters.monotone_lce_cell_flips,
            monotone_lce_cells=counters.monotone_lce_cells,
            monotone_create_occurrences=(
                counters.monotone_create_occurrences
            ),
            top_certificate_postings=(
                certificate.top_postings if certificate is not None else 0
            ),
            certificate_retained_intervals=(
                certificate.retained_intervals
                if certificate is not None
                else 0
            ),
            certificate_nodes_visited=(
                counters.certificate_nodes_visited
            ),
            certificate_suffix_fallback_rows=(
                counters.certificate_suffix_fallback_rows
            ),
            periodic_interval_rows=counters.periodic_interval_rows,
            periodic_winner_segments=counters.periodic_winner_segments,
            periodic_creation_intervals_expanded=(
                counters.periodic_creation_intervals_expanded
            ),
            periodic_predecessor_probes=(
                counters.periodic_predecessor_probes
            ),
            periodic_contiguous_jumps=counters.periodic_contiguous_jumps,
            sparse_replacement_rows=counters.sparse_replacement_rows,
            changed_route_rows=counters.changed_route_rows,
            route_change_descriptors=route_change_descriptors,
            affine_equivalent_route_ranges=affine_equivalent_route_ranges,
            periodic_route_ranges=counters.periodic_route_ranges,
        )

    def solve(self) -> CompactBitflipResult:
        for flip in self.flips:
            self._record_flip(*self._solve_flip(self._prepare_flip(flip)))
        return self._build_result()


def compact_filtered_bitflip(
    query_codes: Tensor,
    key_codes: Tensor,
    bit_width: int,
    *,
    top_k: int = 4,
    interval_compression: int = 2,
    suffix_backend: str = "python",
    library_path: Path | str | None = None,
    match_index: MatchLengthIndex | None = None,
    value: Tensor | None = None,
    grad_output: Tensor | None = None,
    materialize_routes: bool = False,
) -> CompactBitflipResult:
    """Run the exact v1 sparse/interval solver.

    ``top_k`` and ``interval_compression`` select exact execution paths only;
    neither changes the one-bit counterfactual. Route matrices are omitted
    unless ``materialize_routes`` is requested for validation.
    """

    return _CompactBitflipSolver(
        query_codes,
        key_codes,
        bit_width,
        top_k=top_k,
        interval_compression=interval_compression,
        suffix_backend=suffix_backend,
        library_path=library_path,
        match_index=match_index,
        value=value,
        grad_output=grad_output,
        materialize_routes=materialize_routes,
    ).solve()
