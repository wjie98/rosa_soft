"""First-generation exact suffix-node replacement baseline.

The frozen v1 solver reuses arithmetic route-run descriptors but replaces the
posting-heavy index with its compact suffix-node implementation.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor

from benchmarks.filtered_bitflip import (
    BitFlip,
    CodeOccurrenceIndex,
    _bit_gradient,
    _route_change_ranges_values,
    _validate_codes,
    generate_influence_events,
    semantic_bit_flips,
    unlimited_hard_routes_diagonal,
)
from benchmarks.filtered_bitflip_indexes import DirectMatchIndex, _SuffixArrayPair


__all__ = [
    "ArithmeticRouteRun",
    "SuffixNode",
    "SuffixNodeBitflipResult",
    "SuffixNodeReplacementIndex",
    "compress_route_runs",
    "suffix_node_filtered_bitflip",
]


@dataclass(frozen=True)
class ArithmeticRouteRun:
    """A finite increasing arithmetic progression of route ids."""

    first: int
    last: int
    step: int

    def __post_init__(self) -> None:
        if self.first > self.last or self.step <= 0:
            raise ValueError("route run must be nonempty and increasing")
        if (self.last - self.first) % self.step:
            raise ValueError("route run endpoint is off progression")

    def contains(self, route: int) -> bool:
        return (
            self.first <= route <= self.last
            and (route - self.first) % self.step == 0
        )

    def predecessor(self, limit: int) -> int | None:
        if limit < self.first:
            return None
        clipped = min(limit, self.last)
        return clipped - (clipped - self.first) % self.step


def compress_route_runs(routes: tuple[int, ...]) -> tuple[ArithmeticRouteRun, ...]:
    """Greedily partition sorted routes into exact constant-step runs."""

    if any(left >= right for left, right in zip(routes, routes[1:])):
        raise ValueError("routes must be strictly increasing")
    runs = []
    index = 0
    while index < len(routes):
        if index + 1 == len(routes):
            runs.append(ArithmeticRouteRun(routes[index], routes[index], 1))
            break
        step = routes[index + 1] - routes[index]
        stop = index + 2
        while stop < len(routes) and routes[stop] - routes[stop - 1] == step:
            stop += 1
        runs.append(
            ArithmeticRouteRun(routes[index], routes[stop - 1], step)
        )
        index = stop
    return tuple(runs)


@dataclass(frozen=True)
class SuffixNode:
    """One positive LCP interval and its key-route postings."""

    depth: int
    rank_start: int
    rank_stop: int
    routes: tuple[int, ...]
    route_runs: tuple[ArithmeticRouteRun, ...]


@dataclass(frozen=True)
class SuffixQueryStats:
    nodes_visited: int
    route_candidates_probed: int
    periodic_jumps: int


class _ExcludedRoutes:
    def __init__(self, routes: set[int]) -> None:
        self.routes = routes
        self.sorted_routes = tuple(sorted(routes))
        self.runs = compress_route_runs(self.sorted_routes)
        self.starts = tuple(run.first for run in self.runs)

    def contains(self, route: int) -> bool:
        return route in self.routes

    def containing_run(self, route: int) -> ArithmeticRouteRun | None:
        index = bisect_right(self.starts, route) - 1
        if index < 0:
            return None
        run = self.runs[index]
        return run if run.contains(route) else None


class SuffixNodeReplacementIndex:
    """LCP interval nodes for exact latest-route replacement queries."""

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        *,
        suffix_backend: str = "python",
        library_path: Path | str | None = None,
    ) -> None:
        _validate_codes(query_codes, key_codes)
        query = tuple(int(value) for value in query_codes.tolist())
        key = tuple(int(value) for value in key_codes.tolist())
        self.sequence_length = len(query)
        pair = _SuffixArrayPair(
            query[::-1],
            key[::-1],
            backend=suffix_backend,
            library_path=library_path,
        )
        raw_nodes = self._lcp_intervals(pair.lcp)
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
        key_ranks = tuple(rank for rank, _ in key_rank_routes)
        nodes = []
        for depth, rank_start, rank_stop in raw_nodes:
            first = bisect_right(key_ranks, rank_start - 1)
            last = bisect_right(key_ranks, rank_stop - 1)
            routes = tuple(
                sorted(route for _, route in key_rank_routes[first:last])
            )
            if routes:
                nodes.append(
                    SuffixNode(
                        depth,
                        rank_start,
                        rank_stop,
                        routes,
                        compress_route_runs(routes),
                    )
                )
        self.nodes = tuple(nodes)
        query_nodes = []
        for query_index in range(self.sequence_length):
            text_position = self.sequence_length - 1 - query_index
            rank = pair.inverse[text_position]
            ancestors = tuple(
                sorted(
                    (
                        node_index
                        for node_index, node in enumerate(self.nodes)
                        if node.rank_start <= rank < node.rank_stop
                    ),
                    key=lambda node_index: self.nodes[node_index].depth,
                    reverse=True,
                )
            )
            query_nodes.append(ancestors)
        self.query_nodes = tuple(query_nodes)
        self.route_postings = sum(len(node.routes) for node in self.nodes)
        self.route_runs = sum(len(node.route_runs) for node in self.nodes)
        self.logical_bytes = (
            16 * len(self.nodes)
            + 4 * self.route_postings
            + 12 * self.route_runs
            + 4 * sum(len(path) for path in self.query_nodes)
        )

    @staticmethod
    def _lcp_intervals(lcp: tuple[int, ...]) -> tuple[tuple[int, int, int], ...]:
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

    @staticmethod
    def _latest_flat(
        routes: tuple[int, ...],
        limit: int,
        excluded: _ExcludedRoutes,
    ) -> tuple[int | None, int]:
        index = bisect_right(routes, limit) - 1
        probes = 0
        while index >= 0:
            probes += 1
            route = routes[index]
            if not excluded.contains(route):
                return route, probes
            index -= 1
        return None, probes

    @staticmethod
    def _latest_periodic(
        runs: tuple[ArithmeticRouteRun, ...],
        limit: int,
        excluded: _ExcludedRoutes,
    ) -> tuple[int | None, int, int]:
        probes = 0
        jumps = 0
        for candidate_run in reversed(runs):
            route = candidate_run.predecessor(limit)
            while route is not None and route >= candidate_run.first:
                probes += 1
                excluded_run = excluded.containing_run(route)
                if excluded_run is None:
                    return route, probes, jumps
                if (
                    candidate_run.step % excluded_run.step == 0
                    and (route - excluded_run.first) % excluded_run.step == 0
                ):
                    route = candidate_run.predecessor(excluded_run.first - 1)
                    jumps += 1
                else:
                    route -= candidate_run.step
        return None, probes, jumps

    def best_excluding(
        self,
        query_index: int,
        excluded_routes: set[int],
        *,
        backend: str = "flat",
    ) -> tuple[int, int, SuffixQueryStats]:
        if not 0 <= query_index < self.sequence_length:
            raise ValueError("query index is outside the suffix index")
        if backend not in {"flat", "periodic"}:
            raise ValueError(f"unknown suffix posting backend: {backend}")
        excluded = _ExcludedRoutes(excluded_routes)
        nodes_visited = 0
        probes = 0
        jumps = 0
        for node_index in self.query_nodes[query_index]:
            node = self.nodes[node_index]
            nodes_visited += 1
            if backend == "periodic":
                route, local_probes, local_jumps = self._latest_periodic(
                    node.route_runs,
                    query_index,
                    excluded,
                )
                jumps += local_jumps
            else:
                route, local_probes = self._latest_flat(
                    node.routes,
                    query_index,
                    excluded,
                )
            probes += local_probes
            if route is not None:
                return node.depth, route, SuffixQueryStats(
                    nodes_visited,
                    probes,
                    jumps,
                )
        return 0, 0, SuffixQueryStats(nodes_visited, probes, jumps)


@dataclass(frozen=True)
class SuffixNodeBitflipResult:
    flips: tuple[BitFlip, ...]
    base_routes: Tensor
    base_lengths: Tensor
    flipped_routes: Tensor
    flipped_lengths: Tensor
    bit_gradient: Tensor | None
    suffix_nodes: int
    route_postings: int
    route_runs: int
    logical_bytes: int
    raw_events: int
    event_cells: int
    replacement_rows: int
    suffix_nodes_visited: int
    route_candidates_probed: int
    periodic_jumps: int
    final_changed_rows: int
    final_route_change_ranges: int


def suffix_node_filtered_bitflip(
    query_codes: Tensor,
    key_codes: Tensor,
    bit_width: int,
    *,
    posting_backend: str = "flat",
    suffix_backend: str = "python",
    library_path: Path | str | None = None,
    match_index=None,
    value: Tensor | None = None,
    grad_output: Tensor | None = None,
) -> SuffixNodeBitflipResult:
    """Use suffix nodes for exact replacement after expanding changed rows."""

    _validate_codes(query_codes, key_codes)
    suffix_index = SuffixNodeReplacementIndex(
        query_codes,
        key_codes,
        suffix_backend=suffix_backend,
        library_path=library_path,
    )
    base = unlimited_hard_routes_diagonal(query_codes, key_codes)
    flips = semantic_bit_flips(query_codes.numel(), bit_width)
    if match_index is None:
        match_index = DirectMatchIndex(query_codes, key_codes)
    occurrence = CodeOccurrenceIndex.build(query_codes, key_codes)
    route_rows = []
    length_rows = []
    raw_events = 0
    event_cells = 0
    replacement_rows = 0
    nodes_visited = 0
    candidates_probed = 0
    periodic_jumps = 0
    for flip in flips:
        events = generate_influence_events(
            query_codes,
            key_codes,
            flip,
            match_index=match_index,
            occurrence_index=occurrence,
        )
        raw_events += len(events)
        changed: dict[int, dict[int, int]] = {}
        for event in events:
            event_cells += event.stop - event.start
            for query_index in range(event.start, event.stop):
                route = query_index + event.route_offset
                row = changed.setdefault(query_index, {})
                if route in row:
                    raise RuntimeError("influence events overlap one candidate cell")
                row[route] = event.new_length(query_index)
        routes = base.routes.clone()
        lengths = base.lengths.clone()
        for query_index, row in changed.items():
            replacement_rows += 1
            unchanged_length, unchanged_route, stats = suffix_index.best_excluding(
                query_index,
                set(row),
                backend=posting_backend,
            )
            nodes_visited += stats.nodes_visited
            candidates_probed += stats.route_candidates_probed
            periodic_jumps += stats.periodic_jumps
            best_length = unchanged_length
            best_route = unchanged_route
            for route, length in row.items():
                if length > best_length or (
                    length == best_length and length > 0 and route > best_route
                ):
                    best_length = length
                    best_route = route
            routes[query_index] = best_route
            lengths[query_index] = best_length
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
    base_values = [int(route) for route in base.routes.tolist()]
    final_ranges = sum(
        len(_route_change_ranges_values(base_values, routes))
        for routes in flipped_routes.tolist()
    )
    return SuffixNodeBitflipResult(
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
        suffix_nodes=len(suffix_index.nodes),
        route_postings=suffix_index.route_postings,
        route_runs=suffix_index.route_runs,
        logical_bytes=suffix_index.logical_bytes,
        raw_events=raw_events,
        event_cells=event_cells,
        replacement_rows=replacement_rows,
        suffix_nodes_visited=nodes_visited,
        route_candidates_probed=candidates_probed,
        periodic_jumps=periodic_jumps,
        final_changed_rows=final_changed_rows,
        final_route_change_ranges=final_ranges,
    )
