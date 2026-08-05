"""Full-certificate baseline and shared periodic-family algebra.

The frozen v1 solver reuses the event-family types but has its own bounded
certificate implementation. The full solver here remains an exact baseline.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

from benchmarks.filtered_bitflip import (
    BitFlip,
    CodeOccurrenceIndex,
    InfluenceEvent,
    _bit_gradient,
    _route_change_ranges_values,
    _validate_codes,
    generate_influence_events,
    semantic_bit_flips,
    unlimited_hard_routes_diagonal,
)
from benchmarks.filtered_bitflip_indexes import DirectMatchIndex


__all__ = [
    "CandidateLine",
    "CertificateBitflipResult",
    "CertificateSolveStats",
    "PeriodicCertificateSolveStats",
    "PeriodicEventFamily",
    "PeriodicEventFamilyIndex",
    "ReplacementCertificateTree",
    "WinnerSegment",
    "build_candidate_lines",
    "build_periodic_event_families",
    "certificate_filtered_bitflip",
    "materialize_winner_segments",
]


Priority = tuple[int, int]


@dataclass(frozen=True)
class CandidateLine:
    """One positive diagonal match run with constant normalized priority."""

    line_id: int
    start: int
    stop: int
    normalized_length: int
    route_offset: int

    @property
    def priority(self) -> Priority:
        return self.normalized_length, self.route_offset


@dataclass(frozen=True)
class PeriodicEventFamily:
    """A sequence of exact events whose integer fields change affinely."""

    creates_match: bool
    count: int
    first_start: int
    first_stop: int
    first_long_normalized_length: int
    first_short_normalized_length: int
    first_route_offset: int
    delta_start: int = 0
    delta_stop: int = 0
    delta_long_normalized_length: int = 0
    delta_short_normalized_length: int = 0
    delta_route_offset: int = 0

    def _field(self, first: int, delta: int, index: int) -> int:
        if not 0 <= index < self.count:
            raise IndexError("periodic event index is outside the family")
        return first + delta * index

    def start(self, index: int) -> int:
        return self._field(self.first_start, self.delta_start, index)

    def stop(self, index: int) -> int:
        return self._field(self.first_stop, self.delta_stop, index)

    def old_priority(self, index: int) -> Priority:
        return (
            self._field(
                self.first_long_normalized_length,
                self.delta_long_normalized_length,
                index,
            ),
            self._field(
                self.first_route_offset,
                self.delta_route_offset,
                index,
            ),
        )

    def added_priority(self, index: int) -> Priority:
        if self.creates_match:
            first = self.first_long_normalized_length
            delta = self.delta_long_normalized_length
        else:
            first = self.first_short_normalized_length
            delta = self.delta_short_normalized_length
        return (
            self._field(first, delta, index),
            self._field(
                self.first_route_offset,
                self.delta_route_offset,
                index,
            ),
        )

    def _index_interval(
        self,
        first: int,
        delta: int,
        predicate,
    ) -> tuple[int, int]:
        if self.count == 0:
            return 0, 0
        first_matches = predicate(first)
        last_matches = predicate(first + delta * (self.count - 1))
        if first_matches and last_matches:
            return 0, self.count
        if not first_matches and not last_matches:
            return 0, 0
        low = 0
        high = self.count
        if first_matches:
            while low < high:
                middle = (low + high) // 2
                if predicate(first + delta * middle):
                    low = middle + 1
                else:
                    high = middle
            return 0, low
        while low < high:
            middle = (low + high) // 2
            if predicate(first + delta * middle):
                high = middle
            else:
                low = middle + 1
        return low, self.count

    @staticmethod
    def _intersect_ranges(*ranges: tuple[int, int]) -> tuple[int, int]:
        start = max(item[0] for item in ranges)
        stop = min(item[1] for item in ranges)
        return (start, stop) if start < stop else (0, 0)

    def event_cover_range(self, start: int, stop: int) -> tuple[int, int]:
        return self._intersect_ranges(
            self._index_interval(
                self.first_start,
                self.delta_start,
                lambda value: value <= start,
            ),
            self._index_interval(
                self.first_stop,
                self.delta_stop,
                lambda value: value >= stop,
            ),
        )

    def event_intersect_range(self, start: int, stop: int) -> tuple[int, int]:
        return self._intersect_ranges(
            self._index_interval(
                self.first_start,
                self.delta_start,
                lambda value: value < stop,
            ),
            self._index_interval(
                self.first_stop,
                self.delta_stop,
                lambda value: value > start,
            ),
        )

    def add_cover_range(self, start: int, stop: int) -> tuple[int, int]:
        shift = 0 if self.creates_match else 1
        return self._intersect_ranges(
            self._index_interval(
                self.first_start + shift,
                self.delta_start,
                lambda value: value <= start,
            ),
            self._index_interval(
                self.first_stop,
                self.delta_stop,
                lambda value: value >= stop,
            ),
        )

    def add_intersect_range(self, start: int, stop: int) -> tuple[int, int]:
        shift = 0 if self.creates_match else 1
        return self._intersect_ranges(
            self._index_interval(
                self.first_start + shift,
                self.delta_start,
                lambda value: value < stop,
            ),
            self._index_interval(
                self.first_stop,
                self.delta_stop,
                lambda value: value > start,
            ),
        )

    def contains_old_priority(
        self,
        priority: Priority,
        index_range: tuple[int, int],
    ) -> bool:
        if self.creates_match or index_range[0] >= index_range[1]:
            return False
        candidate_index: int | None = None
        fields = (
            (
                self.first_long_normalized_length,
                self.delta_long_normalized_length,
                priority[0],
            ),
            (
                self.first_route_offset,
                self.delta_route_offset,
                priority[1],
            ),
        )
        for first, delta, target in fields:
            if delta == 0:
                if first != target:
                    return False
                continue
            difference = target - first
            if difference % delta:
                return False
            index = difference // delta
            if candidate_index is not None and candidate_index != index:
                return False
            candidate_index = index
        if candidate_index is None:
            return True
        return index_range[0] <= candidate_index < index_range[1]

    def maximum_added_priority(
        self,
        index_range: tuple[int, int],
    ) -> Priority | None:
        start, stop = index_range
        if start >= stop:
            return None
        if self.creates_match:
            length_delta = self.delta_long_normalized_length
        else:
            length_delta = self.delta_short_normalized_length
        if length_delta > 0:
            index = stop - 1
        elif length_delta < 0:
            index = start
        elif self.delta_route_offset > 0:
            index = stop - 1
        else:
            index = start
        return self.added_priority(index)


@dataclass(frozen=True)
class _FamilyEnvelopeNode:
    center: int
    by_start: tuple[tuple[int, int], ...]
    by_stop: tuple[tuple[int, int], ...]
    left: "_FamilyEnvelopeNode | None"
    right: "_FamilyEnvelopeNode | None"


class _FamilyEnvelopeIndex:
    """Static interval tree over conservative periodic-family envelopes."""

    def __init__(self, envelopes: tuple[tuple[int, int, int], ...]) -> None:
        self.root = self._build(envelopes)

    @classmethod
    def _build(
        cls,
        envelopes: tuple[tuple[int, int, int], ...],
    ) -> _FamilyEnvelopeNode | None:
        if not envelopes:
            return None
        centers = sorted((start + stop) // 2 for start, stop, _ in envelopes)
        center = centers[len(centers) // 2]
        left = []
        right = []
        crossing = []
        for start, stop, family_index in envelopes:
            if stop <= center:
                left.append((start, stop, family_index))
            elif start > center:
                right.append((start, stop, family_index))
            else:
                crossing.append((start, stop, family_index))
        return _FamilyEnvelopeNode(
            center=center,
            by_start=tuple(
                sorted((start, family_index) for start, _, family_index in crossing)
            ),
            by_stop=tuple(
                sorted((stop, family_index) for _, stop, family_index in crossing)
            ),
            left=cls._build(tuple(left)),
            right=cls._build(tuple(right)),
        )

    def intersecting(self, start: int, stop: int) -> tuple[int, ...]:
        if start >= stop:
            return ()
        result = []

        def visit(node: _FamilyEnvelopeNode | None) -> None:
            if node is None:
                return
            if stop <= node.center:
                count = bisect_left(node.by_start, (stop, -1))
                result.extend(index for _, index in node.by_start[:count])
                visit(node.left)
            elif start > node.center:
                first = bisect_right(node.by_stop, (start, 1 << 62))
                result.extend(index for _, index in node.by_stop[first:])
                visit(node.right)
            else:
                result.extend(index for _, index in node.by_start)
                visit(node.left)
                visit(node.right)

        visit(self.root)
        return tuple(result)


class PeriodicEventFamilyIndex:
    """Index periodic event envelopes and return exact active phase ranges."""

    def __init__(self, families: tuple[PeriodicEventFamily, ...]) -> None:
        self.families = families
        event_envelopes = []
        add_envelopes = []
        for family_index, family in enumerate(families):
            last = family.count - 1
            event_start = min(family.start(0), family.start(last))
            event_stop = max(family.stop(0), family.stop(last))
            if event_start < event_stop:
                event_envelopes.append(
                    (event_start, event_stop, family_index)
                )
            shift = 0 if family.creates_match else 1
            add_start = min(family.start(0), family.start(last)) + shift
            if add_start < event_stop:
                add_envelopes.append((add_start, event_stop, family_index))
        self.event_envelopes = _FamilyEnvelopeIndex(tuple(event_envelopes))
        self.add_envelopes = _FamilyEnvelopeIndex(tuple(add_envelopes))

    def _query(
        self,
        envelope_index: _FamilyEnvelopeIndex,
        start: int,
        stop: int,
        range_method: str,
        *,
        creates_match: bool | None = None,
    ) -> tuple[tuple[tuple[PeriodicEventFamily, tuple[int, int]], ...], int]:
        candidates = envelope_index.intersecting(start, stop)
        matches = []
        for family_index in candidates:
            family = self.families[family_index]
            if creates_match is not None and family.creates_match != creates_match:
                continue
            index_range = getattr(family, range_method)(start, stop)
            if index_range[0] < index_range[1]:
                matches.append((family, index_range))
        return tuple(matches), len(candidates)

    def event_covering(
        self,
        start: int,
        stop: int,
        *,
        creates_match: bool | None = None,
    ):
        return self._query(
            self.event_envelopes,
            start,
            stop,
            "event_cover_range",
            creates_match=creates_match,
        )

    def event_intersecting(
        self,
        start: int,
        stop: int,
        *,
        creates_match: bool | None = None,
    ):
        return self._query(
            self.event_envelopes,
            start,
            stop,
            "event_intersect_range",
            creates_match=creates_match,
        )

    def add_covering(self, start: int, stop: int):
        return self._query(
            self.add_envelopes, start, stop, "add_cover_range"
        )

    def add_intersecting(self, start: int, stop: int):
        return self._query(
            self.add_envelopes, start, stop, "add_intersect_range"
        )


def _event_fields(event: InfluenceEvent) -> tuple[int, int, int, int, int]:
    return (
        event.start,
        event.stop,
        event.long_normalized_length,
        event.short_normalized_length,
        event.route_offset,
    )


def build_periodic_event_families(
    events: tuple[InfluenceEvent, ...],
) -> tuple[PeriodicEventFamily, ...]:
    """Compress runs of at least three events with one exact field delta."""

    families = []
    index = 0
    while index < len(events):
        stop = index + 1
        delta = (0, 0, 0, 0, 0)
        if index + 2 < len(events):
            first = _event_fields(events[index])
            second = _event_fields(events[index + 1])
            third = _event_fields(events[index + 2])
            candidate_delta = tuple(b - a for a, b in zip(first, second))
            if (
                events[index].creates_match
                == events[index + 1].creates_match
                == events[index + 2].creates_match
                and candidate_delta
                == tuple(c - b for b, c in zip(second, third))
            ):
                delta = candidate_delta
                stop = index + 3
                previous = third
                while stop < len(events):
                    current = _event_fields(events[stop])
                    if (
                        events[stop].creates_match
                        != events[index].creates_match
                        or tuple(c - p for p, c in zip(previous, current))
                        != delta
                    ):
                        break
                    previous = current
                    stop += 1
        event = events[index]
        fields = _event_fields(event)
        families.append(
            PeriodicEventFamily(
                event.creates_match,
                stop - index,
                *fields,
                *delta,
            )
        )
        index = stop
    return tuple(families)


@dataclass(frozen=True)
class WinnerSegment:
    """A row interval whose exact winner is one affine candidate or null."""

    start: int
    stop: int
    priority: Optional[Priority]


@dataclass(frozen=True)
class CertificateSolveStats:
    tree_nodes_visited: int
    base_candidates_probed: int
    canonical_forbid_entries: int
    canonical_add_entries: int
    emitted_segments: int
    merged_segments: int


@dataclass(frozen=True)
class PeriodicCertificateSolveStats:
    event_families: int
    compressed_events: int
    tree_nodes_visited: int
    base_candidates_probed: int
    emitted_segments: int
    merged_segments: int
    family_candidates_examined: int


@dataclass(frozen=True)
class CertificateBitflipResult:
    flips: tuple[BitFlip, ...]
    base_routes: Tensor
    base_lengths: Tensor
    flipped_routes: Tensor
    flipped_lengths: Tensor
    bit_gradient: Tensor | None
    candidate_lines: int
    certificate_postings: int
    raw_events: int
    event_cells: int
    event_families: int
    periodic_compressed_events: int
    tree_nodes_visited: int
    base_candidates_probed: int
    canonical_forbid_entries: int
    canonical_add_entries: int
    emitted_segments: int
    merged_segments: int
    final_changed_rows: int
    final_route_change_ranges: int


def build_candidate_lines(
    query_codes: Tensor,
    key_codes: Tensor,
) -> tuple[CandidateLine, ...]:
    """Decompose every positive candidate into maximal diagonal match runs."""

    _validate_codes(query_codes, key_codes)
    query = tuple(int(value) for value in query_codes.tolist())
    key = tuple(int(value) for value in key_codes.tolist())
    sequence_length = len(query)
    lines: list[CandidateLine] = []
    for route_offset in range(1 - sequence_length, 1):
        first_row = max(1, 1 - route_offset)
        run_start: int | None = None
        for query_index in range(first_row, sequence_length):
            route = query_index + route_offset
            matches = query[query_index] == key[route - 1]
            if matches and run_start is None:
                run_start = query_index
            elif not matches and run_start is not None:
                lines.append(
                    CandidateLine(
                        len(lines),
                        run_start,
                        query_index,
                        1 - run_start,
                        route_offset,
                    )
                )
                run_start = None
        if run_start is not None:
            lines.append(
                CandidateLine(
                    len(lines),
                    run_start,
                    sequence_length,
                    1 - run_start,
                    route_offset,
                )
            )
    return tuple(lines)


def _maximum(
    left: Priority | None,
    right: Priority | None,
) -> Priority | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _merge_winner_segments(
    segments: list[WinnerSegment],
) -> tuple[WinnerSegment, ...]:
    if not segments:
        return ()
    merged = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        if previous.stop == segment.start and previous.priority == segment.priority:
            merged[-1] = WinnerSegment(
                previous.start,
                segment.stop,
                previous.priority,
            )
        else:
            merged.append(segment)
    return tuple(merged)


class ReplacementCertificateTree:
    """Segment tree of sorted candidate certificates and temporary overlays."""

    def __init__(self, sequence_length: int, lines: tuple[CandidateLine, ...]):
        if sequence_length < 0:
            raise ValueError("sequence_length must be nonnegative")
        size = 1
        while size < max(sequence_length, 1):
            size *= 2
        self.sequence_length = sequence_length
        self.size = size
        self.lines = lines
        self.cover: list[list[int]] = [[] for _ in range(2 * size)]
        self.by_priority: dict[Priority, int] = {}
        for line in lines:
            if line.line_id >= len(lines) or lines[line.line_id] != line:
                raise ValueError("candidate line ids must be dense and ordered")
            if line.priority in self.by_priority:
                raise ValueError("candidate line priority must identify one run")
            self.by_priority[line.priority] = line.line_id
            for node in self._canonical_nodes(line.start, line.stop):
                self.cover[node].append(line.line_id)
        for candidates in self.cover:
            candidates.sort(
                key=lambda line_id: self.lines[line_id].priority,
                reverse=True,
            )
        self.subtree_upper: list[Priority | None] = [None] * (2 * size)
        for node in range(2 * size - 1, 0, -1):
            local = (
                self.lines[self.cover[node][0]].priority
                if self.cover[node]
                else None
            )
            if node < size:
                local = _maximum(local, self.subtree_upper[2 * node])
                local = _maximum(local, self.subtree_upper[2 * node + 1])
            self.subtree_upper[node] = local
        self.certificate_postings = sum(len(items) for items in self.cover)
        self.logical_bytes = 20 * len(lines) + 4 * self.certificate_postings

    @classmethod
    def build(
        cls,
        query_codes: Tensor,
        key_codes: Tensor,
    ) -> "ReplacementCertificateTree":
        _validate_codes(query_codes, key_codes)
        return cls(query_codes.numel(), build_candidate_lines(query_codes, key_codes))

    def _canonical_nodes(self, start: int, stop: int) -> tuple[int, ...]:
        if not 0 <= start <= stop <= self.sequence_length:
            raise ValueError("certificate interval is invalid")
        nodes = []
        left = start + self.size
        right = stop + self.size
        while left < right:
            if left & 1:
                nodes.append(left)
                left += 1
            if right & 1:
                right -= 1
                nodes.append(right)
            left //= 2
            right //= 2
        return tuple(nodes)

    def _event_overlays(
        self,
        events: tuple[InfluenceEvent, ...],
    ) -> tuple[list[list[int]], list[list[Priority]], int, int]:
        forbidden: list[list[int]] = [[] for _ in range(2 * self.size)]
        additions: list[list[Priority]] = [[] for _ in range(2 * self.size)]
        forbid_entries = 0
        add_entries = 0
        for event in events:
            if event.creates_match:
                add_start = event.start
                added = (
                    event.long_normalized_length,
                    event.route_offset,
                )
            else:
                old_priority = (
                    event.long_normalized_length,
                    event.route_offset,
                )
                try:
                    line_id = self.by_priority[old_priority]
                except KeyError as error:
                    raise RuntimeError(
                        "break event has no base candidate certificate"
                    ) from error
                line = self.lines[line_id]
                if not line.start <= event.start < event.stop <= line.stop:
                    raise RuntimeError("break interval escapes its base match run")
                nodes = self._canonical_nodes(event.start, event.stop)
                for node in nodes:
                    forbidden[node].append(line_id)
                forbid_entries += len(nodes)
                add_start = event.start + 1
                added = (
                    event.short_normalized_length,
                    event.route_offset,
                )
            if add_start < event.stop:
                nodes = self._canonical_nodes(add_start, event.stop)
                for node in nodes:
                    additions[node].append(added)
                add_entries += len(nodes)
        return forbidden, additions, forbid_entries, add_entries

    def overlay_entry_counts(
        self,
        events: tuple[InfluenceEvent, ...],
    ) -> tuple[int, int]:
        """Count canonical break/add entries without allocating overlay trees."""

        forbid_entries = 0
        add_entries = 0
        for event in events:
            if not event.creates_match:
                forbid_entries += len(
                    self._canonical_nodes(event.start, event.stop)
                )
                add_start = event.start + 1
            else:
                add_start = event.start
            if add_start < event.stop:
                add_entries += len(
                    self._canonical_nodes(add_start, event.stop)
                )
        return forbid_entries, add_entries

    def solve(
        self,
        events: tuple[InfluenceEvent, ...],
    ) -> tuple[tuple[WinnerSegment, ...], CertificateSolveStats]:
        """Return the exact winner envelope without expanding event cells."""

        forbidden, additions, forbid_entries, add_entries = self._event_overlays(
            events
        )
        addition_upper: list[Priority | None] = [None] * (2 * self.size)
        forbidden_subtree: list[frozenset[int]] = [
            frozenset() for _ in range(2 * self.size)
        ]
        for node in range(2 * self.size - 1, 0, -1):
            local_add = max(additions[node]) if additions[node] else None
            local_forbidden = set(forbidden[node])
            if node < self.size:
                local_add = _maximum(local_add, addition_upper[2 * node])
                local_add = _maximum(local_add, addition_upper[2 * node + 1])
                local_forbidden.update(forbidden_subtree[2 * node])
                local_forbidden.update(forbidden_subtree[2 * node + 1])
            addition_upper[node] = local_add
            forbidden_subtree[node] = frozenset(local_forbidden)

        active_forbidden: dict[int, int] = {}
        active_additions: list[Priority] = []
        cover_sources: list[list[int]] = []
        segments: list[WinnerSegment] = []
        nodes_visited = 0
        candidates_probed = 0

        def current_winner() -> tuple[Priority | None, int | None]:
            nonlocal candidates_probed
            best_priority = max(active_additions) if active_additions else None
            best_line: int | None = None
            for source in cover_sources:
                for line_id in source:
                    candidates_probed += 1
                    if line_id in active_forbidden:
                        continue
                    priority = self.lines[line_id].priority
                    if best_priority is None or priority > best_priority:
                        best_priority = priority
                        best_line = line_id
                    break
            return best_priority, best_line

        def visit(node: int, start: int, stop: int) -> None:
            nonlocal nodes_visited
            if start >= self.sequence_length:
                return
            nodes_visited += 1
            source_added = bool(self.cover[node])
            if source_added:
                cover_sources.append(self.cover[node])
            for line_id in forbidden[node]:
                active_forbidden[line_id] = active_forbidden.get(line_id, 0) + 1
            addition_count = len(additions[node])
            active_additions.extend(additions[node])

            priority, base_line = current_winner()
            upper = _maximum(self.subtree_upper[node], addition_upper[node])
            stable = priority is not None and (upper is None or priority >= upper)
            if base_line is not None and base_line in forbidden_subtree[node]:
                stable = False
            if priority is None and upper is None:
                stable = True
            clipped_stop = min(stop, self.sequence_length)
            if stable or node >= self.size:
                segments.append(WinnerSegment(start, clipped_stop, priority))
            else:
                middle = (start + stop) // 2
                visit(2 * node, start, middle)
                visit(2 * node + 1, middle, stop)

            if addition_count:
                del active_additions[-addition_count:]
            for line_id in forbidden[node]:
                count = active_forbidden[line_id] - 1
                if count:
                    active_forbidden[line_id] = count
                else:
                    del active_forbidden[line_id]
            if source_added:
                cover_sources.pop()

        visit(1, 0, self.size)
        merged = _merge_winner_segments(segments)
        return merged, CertificateSolveStats(
            tree_nodes_visited=nodes_visited,
            base_candidates_probed=candidates_probed,
            canonical_forbid_entries=forbid_entries,
            canonical_add_entries=add_entries,
            emitted_segments=len(segments),
            merged_segments=len(merged),
        )

    def solve_periodic(
        self,
        events: tuple[InfluenceEvent, ...],
    ) -> tuple[tuple[WinnerSegment, ...], PeriodicCertificateSolveStats]:
        """Solve with affine event families instead of canonical event entries."""

        families = build_periodic_event_families(events)
        event_index = PeriodicEventFamilyIndex(families)
        cover_sources: list[list[int]] = []
        segments: list[WinnerSegment] = []
        nodes_visited = 0
        candidates_probed = 0
        family_candidates_examined = 0

        def visit(node: int, start: int, stop: int) -> None:
            nonlocal nodes_visited, candidates_probed
            nonlocal family_candidates_examined
            if start >= self.sequence_length:
                return
            nodes_visited += 1
            source_added = bool(self.cover[node])
            if source_added:
                cover_sources.append(self.cover[node])
            clipped_stop = min(stop, self.sequence_length)

            break_cover, examined = event_index.event_covering(
                start,
                clipped_stop,
                creates_match=False,
            )
            family_candidates_examined += examined
            best_priority: Priority | None = None
            best_line: int | None = None
            add_cover, examined = event_index.add_covering(
                start, clipped_stop
            )
            family_candidates_examined += examined
            for family, index_range in add_cover:
                priority = family.maximum_added_priority(index_range)
                best_priority = _maximum(best_priority, priority)
            for source in cover_sources:
                for line_id in source:
                    candidates_probed += 1
                    priority = self.lines[line_id].priority
                    if any(
                        family.contains_old_priority(priority, index_range)
                        for family, index_range in break_cover
                    ):
                        continue
                    if best_priority is None or priority > best_priority:
                        best_priority = priority
                        best_line = line_id
                    break

            upper = self.subtree_upper[node]
            add_intersect, examined = event_index.add_intersecting(
                start, clipped_stop
            )
            family_candidates_examined += examined
            for family, index_range in add_intersect:
                upper = _maximum(
                    upper, family.maximum_added_priority(index_range)
                )
            stable = best_priority is not None and (
                upper is None or best_priority >= upper
            )
            if best_line is not None:
                priority = self.lines[best_line].priority
                break_intersect, examined = event_index.event_intersecting(
                    start,
                    clipped_stop,
                    creates_match=False,
                )
                family_candidates_examined += examined
                if any(
                    family.contains_old_priority(priority, index_range)
                    for family, index_range in break_intersect
                ):
                    stable = False
            if best_priority is None and upper is None:
                stable = True
            if stable or node >= self.size:
                segments.append(
                    WinnerSegment(start, clipped_stop, best_priority)
                )
            else:
                middle = (start + stop) // 2
                visit(2 * node, start, middle)
                visit(2 * node + 1, middle, stop)

            if source_added:
                cover_sources.pop()

        visit(1, 0, self.size)
        merged = _merge_winner_segments(segments)
        return merged, PeriodicCertificateSolveStats(
            event_families=len(families),
            compressed_events=sum(
                family.count for family in families if family.count > 1
            ),
            tree_nodes_visited=nodes_visited,
            base_candidates_probed=candidates_probed,
            emitted_segments=len(segments),
            merged_segments=len(merged),
            family_candidates_examined=family_candidates_examined,
        )


def materialize_winner_segments(
    sequence_length: int,
    segments: tuple[WinnerSegment, ...],
) -> tuple[Tensor, Tensor]:
    routes = torch.zeros(sequence_length, dtype=torch.int64)
    lengths = torch.zeros(sequence_length, dtype=torch.int64)
    covered = 0
    for segment in segments:
        if segment.start != covered or not segment.start < segment.stop:
            raise RuntimeError("winner segments must be contiguous and nonempty")
        covered = segment.stop
        if segment.priority is None:
            continue
        normalized_length, route_offset = segment.priority
        rows = torch.arange(segment.start, segment.stop, dtype=torch.int64)
        candidate_lengths = rows + normalized_length
        candidate_routes = rows + route_offset
        if bool((candidate_lengths <= 0).any()) or bool((candidate_routes <= 0).any()):
            raise RuntimeError("a winner certificate produced a null candidate")
        lengths[segment.start : segment.stop] = candidate_lengths
        routes[segment.start : segment.stop] = candidate_routes
    if covered != sequence_length:
        raise RuntimeError("winner segments do not cover the sequence")
    return routes, lengths


def certificate_filtered_bitflip(
    query_codes: Tensor,
    key_codes: Tensor,
    bit_width: int,
    *,
    match_index=None,
    overlay_backend: str = "canonical",
    value: Tensor | None = None,
    grad_output: Tensor | None = None,
) -> CertificateBitflipResult:
    """Evaluate exact bit flips through a compressed replacement envelope."""

    _validate_codes(query_codes, key_codes)
    if overlay_backend not in {"canonical", "periodic"}:
        raise ValueError(f"unknown overlay backend: {overlay_backend}")
    flips = semantic_bit_flips(query_codes.numel(), bit_width)
    base = unlimited_hard_routes_diagonal(query_codes, key_codes)
    tree = ReplacementCertificateTree.build(query_codes, key_codes)
    if match_index is None:
        match_index = DirectMatchIndex(query_codes, key_codes)
    occurrence = CodeOccurrenceIndex.build(query_codes, key_codes)
    route_rows = []
    length_rows = []
    raw_events = 0
    event_cells = 0
    event_families = 0
    periodic_compressed_events = 0
    nodes_visited = 0
    candidates_probed = 0
    forbid_entries = 0
    add_entries = 0
    emitted_segments = 0
    merged_segments = 0
    for flip in flips:
        events = generate_influence_events(
            query_codes,
            key_codes,
            flip,
            match_index=match_index,
            occurrence_index=occurrence,
        )
        raw_events += len(events)
        event_cells += sum(event.stop - event.start for event in events)
        if events:
            if overlay_backend == "periodic":
                segments, stats = tree.solve_periodic(events)
                event_families += stats.event_families
                periodic_compressed_events += stats.compressed_events
            else:
                segments, stats = tree.solve(events)
                event_families += len(events)
            routes, lengths = materialize_winner_segments(
                query_codes.numel(),
                segments,
            )
            nodes_visited += stats.tree_nodes_visited
            candidates_probed += stats.base_candidates_probed
            if isinstance(stats, CertificateSolveStats):
                forbid_entries += stats.canonical_forbid_entries
                add_entries += stats.canonical_add_entries
            emitted_segments += stats.emitted_segments
            merged_segments += stats.merged_segments
        else:
            routes = base.routes.clone()
            lengths = base.lengths.clone()
        route_rows.append(routes)
        length_rows.append(lengths)
    sequence_length = query_codes.numel()
    flipped_routes = (
        torch.stack(route_rows)
        if route_rows
        else torch.empty(0, sequence_length, dtype=torch.int64)
    )
    flipped_lengths = (
        torch.stack(length_rows)
        if length_rows
        else torch.empty(0, sequence_length, dtype=torch.int64)
    )
    final_changed_rows = int((flipped_routes != base.routes.unsqueeze(0)).sum())
    base_route_values = [int(route) for route in base.routes.tolist()]
    final_ranges = sum(
        len(_route_change_ranges_values(base_route_values, routes))
        for routes in flipped_routes.tolist()
    )
    return CertificateBitflipResult(
        flips=flips,
        base_routes=base.routes,
        base_lengths=base.lengths,
        flipped_routes=flipped_routes,
        flipped_lengths=flipped_lengths,
        bit_gradient=_bit_gradient(
            flips,
            query_codes,
            key_codes,
            base.routes,
            flipped_routes,
            value,
            grad_output,
        ),
        candidate_lines=len(tree.lines),
        certificate_postings=tree.certificate_postings,
        raw_events=raw_events,
        event_cells=event_cells,
        event_families=event_families,
        periodic_compressed_events=periodic_compressed_events,
        tree_nodes_visited=nodes_visited,
        base_candidates_probed=candidates_probed,
        canonical_forbid_entries=forbid_entries,
        canonical_add_entries=add_entries,
        emitted_segments=emitted_segments,
        merged_segments=merged_segments,
        final_changed_rows=final_changed_rows,
        final_route_change_ranges=final_ranges,
    )
