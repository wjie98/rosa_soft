"""First-generation exact winner-filter baselines, outside the v1 solver."""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from torch import Tensor

from benchmarks.filtered_bitflip import (
    BitFlip,
    CodeOccurrenceIndex,
    InfluenceEvent,
    _apply_flip,
    _bit_gradient,
    _code_list,
    _route_change_ranges_values,
    _validate_codes,
    generate_influence_events,
    semantic_bit_flips,
    unlimited_hard_routes_diagonal,
)
from benchmarks.filtered_bitflip_indexes import DirectMatchIndex


__all__ = [
    "FilterQuery",
    "FilteredWinnerResult",
    "RleWinnerIndex",
    "SegmentWinnerIndex",
    "WinnerGeometry",
    "winner_filtered_bitflip",
]


@dataclass(frozen=True)
class FilterQuery:
    ranges: tuple[tuple[int, int], ...]
    node_visits: int

    @property
    def row_count(self) -> int:
        return sum(stop - start for start, stop in self.ranges)


@dataclass(frozen=True)
class WinnerGeometry:
    sequence_length: int
    stride: int
    codes: tuple[int, ...]

    @classmethod
    def build(cls, routes: Tensor, lengths: Tensor) -> "WinnerGeometry":
        if routes.ndim != 1 or lengths.shape != routes.shape:
            raise ValueError("routes and lengths must share shape [T]")
        sequence_length = routes.numel()
        stride = 2 * sequence_length + 1
        codes = tuple(
            cls._encode(
                int(lengths[index]) - index,
                int(routes[index]) - index,
                sequence_length,
                stride,
            )
            for index in range(sequence_length)
        )
        return cls(sequence_length, stride, codes)

    @staticmethod
    def _encode(
        normalized_length: int,
        route_offset: int,
        sequence_length: int,
        stride: int,
    ) -> int:
        if not -sequence_length <= normalized_length <= sequence_length:
            raise ValueError("normalized length is outside the encoding")
        if not -sequence_length <= route_offset <= sequence_length:
            raise ValueError("route offset is outside the encoding")
        return (
            (normalized_length + sequence_length) * stride
            + route_offset
            + sequence_length
        )

    def encode(self, normalized_length: int, route_offset: int) -> int:
        return self._encode(
            normalized_length,
            route_offset,
            self.sequence_length,
            self.stride,
        )

    def event_long_code(self, event: InfluenceEvent) -> int:
        return self.encode(
            event.long_normalized_length,
            event.route_offset,
        )


def _merge_ranges(ranges: Iterable[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    ordered = sorted((start, stop) for start, stop in ranges if start < stop)
    if not ordered:
        return ()
    merged = [ordered[0]]
    for start, stop in ordered[1:]:
        previous_start, previous_stop = merged[-1]
        if start <= previous_stop:
            merged[-1] = (previous_start, max(previous_stop, stop))
        else:
            merged.append((start, stop))
    return tuple(merged)


class RleWinnerIndex:
    """Run-level baseline over normalized base winners."""

    name = "rle"

    def __init__(self, geometry: WinnerGeometry) -> None:
        self.geometry = geometry
        runs = []
        for position, code in enumerate(geometry.codes):
            if runs and runs[-1][2] == code:
                start, _, previous = runs[-1]
                runs[-1] = (start, position + 1, previous)
            else:
                runs.append((position, position + 1, code))
        self.runs = tuple(runs)
        self.starts = tuple(start for start, _, _ in runs)
        self.logical_bytes = len(runs) * 12

    def _query(
        self,
        start: int,
        stop: int,
        target: int,
        mode: str,
    ) -> FilterQuery:
        if not 0 <= start <= stop <= self.geometry.sequence_length:
            raise ValueError("winner query range is invalid")
        if start == stop or not self.runs:
            return FilterQuery((), 0)
        run_index = max(bisect_right(self.starts, start) - 1, 0)
        selected = []
        visits = 0
        while run_index < len(self.runs):
            run_start, run_stop, code = self.runs[run_index]
            if run_start >= stop:
                break
            visits += 1
            keep = target > code if mode == "greater" else code == target
            if keep:
                selected.append((max(start, run_start), min(stop, run_stop)))
            run_index += 1
        return FilterQuery(_merge_ranges(selected), visits)

    def greater(self, start: int, stop: int, target: int) -> FilterQuery:
        return self._query(start, stop, target, "greater")

    def equal(self, start: int, stop: int, target: int) -> FilterQuery:
        return self._query(start, stop, target, "equal")


class SegmentWinnerIndex:
    """Flat min/max tree that accepts or rejects whole winner ranges."""

    name = "segment"

    def __init__(self, geometry: WinnerGeometry) -> None:
        size = 1
        while size < geometry.sequence_length:
            size *= 2
        minimum = [1 << 62] * (2 * size)
        maximum = [-(1 << 62)] * (2 * size)
        for position, code in enumerate(geometry.codes):
            minimum[size + position] = code
            maximum[size + position] = code
        for node in range(size - 1, 0, -1):
            minimum[node] = min(minimum[2 * node], minimum[2 * node + 1])
            maximum[node] = max(maximum[2 * node], maximum[2 * node + 1])
        self.geometry = geometry
        self.size = size
        self.minimum = tuple(minimum)
        self.maximum = tuple(maximum)
        self.logical_bytes = 16 * len(minimum)

    def _query(
        self,
        start: int,
        stop: int,
        target: int,
        mode: str,
    ) -> FilterQuery:
        if not 0 <= start <= stop <= self.geometry.sequence_length:
            raise ValueError("winner query range is invalid")
        if start == stop:
            return FilterQuery((), 0)
        selected = []
        visits = 0
        stack = [(1, 0, self.size)]
        while stack:
            node, node_start, node_stop = stack.pop()
            if node_stop <= start or stop <= node_start:
                continue
            visits += 1
            fully_covered = start <= node_start and node_stop <= stop
            node_min = self.minimum[node]
            node_max = self.maximum[node]
            if mode == "greater":
                if target <= node_min:
                    continue
                if fully_covered and target > node_max:
                    selected.append((node_start, node_stop))
                    continue
            else:
                if target < node_min or target > node_max:
                    continue
                if fully_covered and node_min == node_max == target:
                    selected.append((node_start, node_stop))
                    continue
            if node >= self.size:
                if node_start < self.geometry.sequence_length:
                    selected.append((node_start, node_start + 1))
                continue
            middle = (node_start + node_stop) // 2
            stack.append((2 * node + 1, middle, node_stop))
            stack.append((2 * node, node_start, middle))
        clipped = (
            (range_start, min(range_stop, self.geometry.sequence_length))
            for range_start, range_stop in selected
        )
        return FilterQuery(_merge_ranges(clipped), visits)

    def greater(self, start: int, stop: int, target: int) -> FilterQuery:
        return self._query(start, stop, target, "greater")

    def equal(self, start: int, stop: int, target: int) -> FilterQuery:
        return self._query(start, stop, target, "equal")


@dataclass(frozen=True)
class FilteredWinnerResult:
    flips: tuple[BitFlip, ...]
    base_routes: Tensor
    base_lengths: Tensor
    flipped_routes: Tensor
    flipped_lengths: Tensor
    bit_gradient: Tensor | None
    raw_events: int
    admitted_event_ranges: int
    admitted_rows: int
    exact_recompute_rows: int
    winner_node_visits: int
    final_changed_rows: int
    final_route_change_ranges: int


def _event_query(
    event: InfluenceEvent,
    geometry: WinnerGeometry,
    winner_index,
) -> FilterQuery:
    code = geometry.event_long_code(event)
    if event.creates_match:
        return winner_index.greater(event.start, event.stop, code)
    return winner_index.equal(event.start, event.stop, code)


def _hard_route_at_row(
    query: Sequence[int],
    key: Sequence[int],
    query_index: int,
) -> tuple[int, int]:
    best_route = 0
    best_length = 0
    for route in range(1, query_index + 1):
        length = 0
        while (
            length < route
            and query[query_index - length] == key[route - 1 - length]
        ):
            length += 1
        if length > best_length or (length == best_length and length > 0):
            best_route = route
            best_length = length
    return best_route, best_length


def winner_filtered_bitflip(
    query_codes: Tensor,
    key_codes: Tensor,
    bit_width: int,
    *,
    match_index=None,
    winner_backend: str = "segment",
    value: Tensor | None = None,
    grad_output: Tensor | None = None,
) -> FilteredWinnerResult:
    """Recompute only rows that an exact event can move past the base winner."""

    _validate_codes(query_codes, key_codes)
    flips = semantic_bit_flips(query_codes.numel(), bit_width)
    base = unlimited_hard_routes_diagonal(query_codes, key_codes)
    geometry = WinnerGeometry.build(base.routes, base.lengths)
    constructors = {
        "rle": RleWinnerIndex,
        "segment": SegmentWinnerIndex,
    }
    try:
        winner_index = constructors[winner_backend](geometry)
    except KeyError as error:
        raise ValueError(f"unknown winner backend: {winner_backend}") from error
    if match_index is None:
        match_index = DirectMatchIndex(query_codes, key_codes)
    occurrence_index = CodeOccurrenceIndex.build(query_codes, key_codes)
    route_rows = []
    length_rows = []
    raw_events = 0
    admitted_event_ranges = 0
    admitted_rows = 0
    exact_recompute_rows = 0
    winner_node_visits = 0
    for flip in flips:
        events = generate_influence_events(
            query_codes,
            key_codes,
            flip,
            match_index=match_index,
            occurrence_index=occurrence_index,
        )
        raw_events += len(events)
        queries = [
            _event_query(event, geometry, winner_index) for event in events
        ]
        admitted_event_ranges += sum(len(query.ranges) for query in queries)
        winner_node_visits += sum(query.node_visits for query in queries)
        affected = _merge_ranges(
            item for query in queries for item in query.ranges
        )
        admitted_rows += sum(stop - start for start, stop in affected)
        routes = base.routes.clone()
        lengths = base.lengths.clone()
        if flip.source == "query":
            query = _code_list(_apply_flip(query_codes, flip))
            key = _code_list(key_codes)
        else:
            query = _code_list(query_codes)
            key = _code_list(_apply_flip(key_codes, flip))
        for start, stop in affected:
            for query_index in range(start, stop):
                route, length = _hard_route_at_row(query, key, query_index)
                routes[query_index] = route
                lengths[query_index] = length
                exact_recompute_rows += 1
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
    final_changed_rows = int(
        (flipped_routes != base.routes.unsqueeze(0)).sum()
    )
    base_route_values = [int(value) for value in base.routes.tolist()]
    flipped_route_values = flipped_routes.tolist()
    final_route_ranges = sum(
        len(
            _route_change_ranges_values(
                base_route_values,
                [int(value) for value in routes],
            )
        )
        for routes in flipped_route_values
    )
    return FilteredWinnerResult(
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
        raw_events=raw_events,
        admitted_event_ranges=admitted_event_ranges,
        admitted_rows=admitted_rows,
        exact_recompute_rows=exact_recompute_rows,
        winner_node_visits=winner_node_visits,
        final_changed_rows=final_changed_rows,
        final_route_change_ranges=final_route_ranges,
    )
