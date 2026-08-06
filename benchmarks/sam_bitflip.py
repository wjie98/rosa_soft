"""Exact suffix-automaton route for unlimited ROSA bitflip counterfactuals.

This is an independent CPU research path.  It deliberately depends only on
the original unlimited-suffix semantics in ``filtered_bitflip`` and does not
dispatch into the frozen filtered-bitflip v1 index.

The implementation separates a key edit into two exact languages:

* substrings of the original key that do not contain the edited position;
* new substrings crossing the edited position, represented as anchored runs.

Query edits instead replay the deterministic SAM trajectory from the previous
checkpoint and stop as soon as it coalesces with the original trajectory.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass, field
from heapq import heappop, heappush
from math import gcd
from typing import Literal, Optional, Protocol, Sequence

import torch
from torch import Tensor

from benchmarks.filtered_bitflip import (
    BitFlip,
    HardRouteResult,
    hard_output_from_routes,
    semantic_bit_flips,
)


__all__ = [
    "ExplicitEndPositionIndex",
    "AffineRouteChange",
    "ArithmeticEndPositionIndex",
    "HybridEndPositionIndex",
    "ImplicitEndPositionIndex",
    "MatchTrace",
    "SamBitflipProfile",
    "SamBitflipResult",
    "SamBitflipSolver",
    "SuffixAutomaton",
    "VirtualRun",
    "materialize_route_changes",
    "sam_bitflip",
]


@dataclass
class _SamState:
    max_length: int = 0
    suffix_link: int = -1
    transitions: dict[int, int] = field(default_factory=dict)


class SuffixAutomaton:
    """Static SAM with prefix terminals and suffix-link tree support."""

    def __init__(self, symbols: Sequence[int]) -> None:
        self.symbols = tuple(int(symbol) for symbol in symbols)
        self.states = [_SamState()]
        self.prefix_states: list[int] = []
        last = 0
        for symbol in self.symbols:
            last = self._extend(last, symbol)
            self.prefix_states.append(last)

        self.link_children: list[list[int]] = [
            [] for _ in range(len(self.states))
        ]
        for state in range(1, len(self.states)):
            self.link_children[self.states[state].suffix_link].append(state)
        self.tin, self.tout, self.depth = self._build_link_tree_order()
        self._ancestors = self._build_ancestor_table()
        self.latest_ends = self._build_latest_ends()

    def _extend(self, last: int, symbol: int) -> int:
        current = len(self.states)
        self.states.append(_SamState(max_length=self.states[last].max_length + 1))
        parent = last
        while parent >= 0 and symbol not in self.states[parent].transitions:
            self.states[parent].transitions[symbol] = current
            parent = self.states[parent].suffix_link

        if parent < 0:
            self.states[current].suffix_link = 0
            return current

        child = self.states[parent].transitions[symbol]
        if self.states[parent].max_length + 1 == self.states[child].max_length:
            self.states[current].suffix_link = child
            return current

        clone = len(self.states)
        self.states.append(
            _SamState(
                max_length=self.states[parent].max_length + 1,
                suffix_link=self.states[child].suffix_link,
                transitions=self.states[child].transitions.copy(),
            )
        )
        while (
            parent >= 0
            and self.states[parent].transitions.get(symbol) == child
        ):
            self.states[parent].transitions[symbol] = clone
            parent = self.states[parent].suffix_link
        self.states[child].suffix_link = clone
        self.states[current].suffix_link = clone
        return current

    def _build_link_tree_order(self) -> tuple[list[int], list[int], list[int]]:
        state_count = len(self.states)
        tin = [0] * state_count
        tout = [0] * state_count
        depth = [0] * state_count
        clock = 0
        stack: list[tuple[int, bool]] = [(0, False)]
        while stack:
            state, leaving = stack.pop()
            if leaving:
                tout[state] = clock
                continue
            tin[state] = clock
            clock += 1
            stack.append((state, True))
            for child in reversed(self.link_children[state]):
                depth[child] = depth[state] + 1
                stack.append((child, False))
        return tin, tout, depth

    def _build_ancestor_table(self) -> list[list[int]]:
        state_count = len(self.states)
        levels = max(1, state_count.bit_length())
        ancestors = [[0] * state_count for _ in range(levels)]
        for state in range(1, state_count):
            ancestors[0][state] = self.states[state].suffix_link
        for level in range(1, levels):
            previous = ancestors[level - 1]
            current = ancestors[level]
            for state in range(state_count):
                current[state] = previous[previous[state]]
        return ancestors

    def _build_latest_ends(self) -> tuple[int, ...]:
        latest = [-1] * len(self.states)
        for end, state in enumerate(self.prefix_states):
            latest[state] = end
        for state in sorted(
            range(1, len(self.states)),
            key=lambda state: self.states[state].max_length,
            reverse=True,
        ):
            parent = self.states[state].suffix_link
            latest[parent] = max(latest[parent], latest[state])
        return tuple(latest)

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def edge_count(self) -> int:
        return sum(len(state.transitions) for state in self.states)

    def min_length(self, state: int) -> int:
        if state == 0:
            return 0
        parent = self.states[state].suffix_link
        return self.states[parent].max_length + 1

    def normalize(self, state: int, length: int) -> tuple[int, int]:
        """Return the canonical endpos state for one represented length."""

        if length <= 0:
            return 0, 0
        while state != 0:
            parent = self.states[state].suffix_link
            if self.states[parent].max_length < length:
                break
            state = parent
        return state, length

    def state_for_length(self, state: int, length: int) -> int:
        """Return the weighted suffix-link ancestor representing ``length``."""

        if length <= 0:
            return 0
        for level in range(len(self._ancestors) - 1, -1, -1):
            ancestor = self._ancestors[level][state]
            if ancestor != 0 and self.states[ancestor].max_length >= length:
                state = ancestor
        return state

    def lca(self, left: int, right: int) -> int:
        """Lowest common ancestor in the suffix-link tree."""

        if self.depth[left] < self.depth[right]:
            left, right = right, left
        difference = self.depth[left] - self.depth[right]
        level = 0
        while difference:
            if difference & 1:
                left = self._ancestors[level][left]
            difference >>= 1
            level += 1
        if left == right:
            return left
        for level in range(len(self._ancestors) - 1, -1, -1):
            left_parent = self._ancestors[level][left]
            right_parent = self._ancestors[level][right]
            if left_parent != right_parent:
                left = left_parent
                right = right_parent
        return self._ancestors[0][left]

    def common_suffix_length(self, trace: "MatchTrace", key_end: int) -> int:
        """LCS between a matched query prefix and one exact key prefix."""

        if trace.length == 0 or key_end < 0:
            return 0
        ancestor = self.lca(trace.state, self.prefix_states[key_end])
        return min(
            trace.length,
            self.states[ancestor].max_length,
            key_end + 1,
        )


class EndPositionIndex(Protocol):
    logical_entries: int

    def predecessor(self, state: int, bound: int) -> int: ...


class ExplicitEndPositionIndex:
    """M=infinity oracle storing every end position in every SAM state."""

    def __init__(self, automaton: SuffixAutomaton) -> None:
        positions: list[list[int]] = [
            [] for _ in range(automaton.state_count)
        ]
        for end, state in enumerate(automaton.prefix_states):
            positions[state].append(end)
        order = sorted(
            range(1, automaton.state_count),
            key=lambda state: automaton.states[state].max_length,
            reverse=True,
        )
        for state in order:
            parent = automaton.states[state].suffix_link
            positions[parent].extend(positions[state])
        for state_positions in positions:
            state_positions.sort()
        self.positions = tuple(tuple(values) for values in positions)
        self.logical_entries = sum(len(values) for values in self.positions)

    def predecessor(self, state: int, bound: int) -> int:
        if bound < 0:
            return -1
        positions = self.positions[state]
        index = bisect_right(positions, bound) - 1
        return positions[index] if index >= 0 else -1


@dataclass
class _WaveletNode:
    low: int
    high: int
    prefix_ones: list[int]
    zero_positions: list[int]
    one_positions: list[int]
    left: int = -1
    right: int = -1


class _WaveletRangePredecessor:
    """Rightmost sequence position whose value lies in one value interval."""

    def __init__(self, values: Sequence[int], value_count: int) -> None:
        high = 1 << max(0, (max(value_count, 1) - 1).bit_length())
        self.nodes: list[_WaveletNode] = []
        self.root = self._build(list(values), 0, high)
        self.logical_entries = sum(
            len(node.prefix_ones)
            + len(node.zero_positions)
            + len(node.one_positions)
            for node in self.nodes
        )

    def _build(self, values: list[int], low: int, high: int) -> int:
        node_index = len(self.nodes)
        node = _WaveletNode(low, high, [0], [], [])
        self.nodes.append(node)
        if high - low == 1 or not values:
            node.prefix_ones.extend([0] * len(values))
            return node_index

        middle = (low + high) // 2
        zeros: list[int] = []
        ones: list[int] = []
        one_count = 0
        for position, value in enumerate(values):
            if value < middle:
                zeros.append(value)
                node.zero_positions.append(position)
            else:
                ones.append(value)
                node.one_positions.append(position)
                one_count += 1
            node.prefix_ones.append(one_count)
        if zeros:
            node.left = self._build(zeros, low, middle)
        if ones:
            node.right = self._build(ones, middle, high)
        return node_index

    def latest(self, prefix_size: int, value_low: int, value_high: int) -> int:
        if prefix_size <= 0 or value_low >= value_high:
            return -1
        return self._latest(self.root, prefix_size, value_low, value_high)

    def _latest(
        self,
        node_index: int,
        prefix_size: int,
        value_low: int,
        value_high: int,
    ) -> int:
        if node_index < 0 or prefix_size <= 0:
            return -1
        node = self.nodes[node_index]
        if value_high <= node.low or node.high <= value_low:
            return -1
        if value_low <= node.low and node.high <= value_high:
            return prefix_size - 1

        one_count = node.prefix_ones[prefix_size]
        zero_count = prefix_size - one_count
        left_position = self._latest(
            node.left,
            zero_count,
            value_low,
            value_high,
        )
        if left_position >= 0:
            left_position = node.zero_positions[left_position]
        right_position = self._latest(
            node.right,
            one_count,
            value_low,
            value_high,
        )
        if right_position >= 0:
            right_position = node.one_positions[right_position]
        return max(left_position, right_position)


class ImplicitEndPositionIndex:
    """Exact endpos predecessor through suffix-link subtrees.

    ``end in endpos(state)`` iff ``state`` is an ancestor of the prefix
    terminal for ``end``.  Euler numbering turns this into a 2D query over
    ``(end, terminal_tin)`` without materializing every ancestor occurrence.
    """

    def __init__(self, automaton: SuffixAutomaton) -> None:
        terminal_euler = [
            automaton.tin[state] for state in automaton.prefix_states
        ]
        self.automaton = automaton
        self.range_index = _WaveletRangePredecessor(
            terminal_euler,
            automaton.state_count,
        )
        self.logical_entries = self.range_index.logical_entries

    def predecessor(self, state: int, bound: int) -> int:
        if bound < 0 or not self.automaton.prefix_states:
            return -1
        prefix_size = min(bound + 1, len(self.automaton.prefix_states))
        return self.range_index.latest(
            prefix_size,
            self.automaton.tin[state],
            self.automaton.tout[state],
        )


def _merge_latest(left: list[int], right: list[int], limit: int) -> list[int]:
    merged = sorted(left + right)
    return merged[-limit:] if len(merged) > limit else merged


class HybridEndPositionIndex:
    """Small exact postings plus last-M cache and exact cold fallback."""

    def __init__(
        self,
        automaton: SuffixAutomaton,
        cold_index: EndPositionIndex,
        cache_size: int,
    ) -> None:
        if cache_size <= 0:
            raise ValueError("cache_size must be positive")
        hot: list[list[int]] = [[] for _ in range(automaton.state_count)]
        counts = [0] * automaton.state_count
        for end, state in enumerate(automaton.prefix_states):
            hot[state].append(end)
            counts[state] += 1
        order = sorted(
            range(1, automaton.state_count),
            key=lambda state: automaton.states[state].max_length,
            reverse=True,
        )
        for state in order:
            parent = automaton.states[state].suffix_link
            hot[parent] = _merge_latest(hot[parent], hot[state], cache_size)
            counts[parent] += counts[state]
        self.hot = tuple(tuple(values) for values in hot)
        self.counts = tuple(counts)
        self.cold_index = cold_index
        self.cache_size = cache_size
        self.logical_entries = cold_index.logical_entries + sum(
            len(values) for values in self.hot
        )
        self.hits = 0
        self.negative_hits = 0
        self.cold_fallbacks = 0

    def predecessor(self, state: int, bound: int) -> int:
        if bound < 0:
            return -1
        positions = self.hot[state]
        index = bisect_right(positions, bound) - 1
        if index >= 0:
            self.hits += 1
            return positions[index]
        if self.counts[state] <= self.cache_size:
            self.negative_hits += 1
            return -1
        self.cold_fallbacks += 1
        return self.cold_index.predecessor(state, bound)


class ArithmeticEndPositionIndex:
    """Exact arithmetic-progression certificates over endpos subtrees."""

    def __init__(
        self,
        automaton: SuffixAutomaton,
        fallback: EndPositionIndex,
    ) -> None:
        state_count = automaton.state_count
        counts = [0] * state_count
        minimums = [-1] * state_count
        maximums = [-1] * state_count
        gaps = [0] * state_count
        for end, state in enumerate(automaton.prefix_states):
            counts[state] = 1
            minimums[state] = end
            maximums[state] = end
        order = sorted(
            range(1, state_count),
            key=lambda state: automaton.states[state].max_length,
            reverse=True,
        )
        for state in order:
            if counts[state] == 0:
                continue
            parent = automaton.states[state].suffix_link
            if counts[parent] == 0:
                counts[parent] = counts[state]
                minimums[parent] = minimums[state]
                maximums[parent] = maximums[state]
                gaps[parent] = gaps[state]
                continue
            gaps[parent] = gcd(
                gaps[parent],
                gaps[state],
                abs(minimums[parent] - minimums[state]),
            )
            counts[parent] += counts[state]
            minimums[parent] = min(minimums[parent], minimums[state])
            maximums[parent] = max(maximums[parent], maximums[state])

        steps = [-1] * state_count
        certified = 0
        for state in range(state_count):
            if counts[state] == 1:
                steps[state] = 0
                certified += 1
            elif counts[state] > 1 and gaps[state] > 0:
                span = maximums[state] - minimums[state]
                if span // gaps[state] + 1 == counts[state]:
                    steps[state] = gaps[state]
                    certified += 1
        self.counts = tuple(counts)
        self.minimums = tuple(minimums)
        self.maximums = tuple(maximums)
        self.steps = tuple(steps)
        self.fallback = fallback
        self.certified_states = certified
        self.logical_entries = fallback.logical_entries + 4 * state_count
        self.hits = 0
        self.fallbacks = 0

    def predecessor(self, state: int, bound: int) -> int:
        if bound < 0 or self.counts[state] == 0:
            return -1
        step = self.steps[state]
        if step < 0:
            self.fallbacks += 1
            return self.fallback.predecessor(state, bound)
        self.hits += 1
        minimum = self.minimums[state]
        if bound < minimum:
            return -1
        maximum = self.maximums[state]
        if step == 0 or bound >= maximum:
            return maximum
        return minimum + ((bound - minimum) // step) * step


@dataclass(frozen=True)
class MatchTrace:
    state: int
    length: int
    latest_end: int

    @property
    def route(self) -> int:
        return self.latest_end + 1 if self.length > 0 else 0


@dataclass(frozen=True)
class VirtualRun:
    """One edited-key path crossing the anchored pair ``K[j] <-> Q[p]``."""

    key_position: int
    query_position: int
    stop: int
    length_offset: int
    route_offset: int

    def priority(self, row: int) -> tuple[int, int]:
        return self.length_offset + row, self.route_offset + row


@dataclass(frozen=True)
class AffineRouteChange:
    """One contiguous changed range with arithmetic route and length values."""

    start: int
    stop: int
    route_start: int
    route_step: int
    length_start: int
    length_step: int

    def value(self, row: int) -> tuple[int, int]:
        if not self.start <= row < self.stop:
            raise IndexError("row is outside the route-change range")
        offset = row - self.start
        return (
            self.length_start + offset * self.length_step,
            self.route_start + offset * self.route_step,
        )


def _compress_updates(
    updates: dict[int, tuple[int, int]],
) -> tuple[AffineRouteChange, ...]:
    """Compress ``row -> (length, route)`` updates without approximation."""

    items = sorted(updates.items())
    changes: list[AffineRouteChange] = []
    index = 0
    while index < len(items):
        start_row, (start_length, start_route) = items[index]
        stop_index = index + 1
        route_step = 0
        length_step = 0
        if stop_index < len(items) and items[stop_index][0] == start_row + 1:
            next_row, (next_length, next_route) = items[stop_index]
            route_step = next_route - start_route
            length_step = next_length - start_length
            stop_index += 1
            previous_row = next_row
            previous_length = next_length
            previous_route = next_route
            while stop_index < len(items):
                row, (length, route) = items[stop_index]
                if (
                    row != previous_row + 1
                    or route - previous_route != route_step
                    or length - previous_length != length_step
                ):
                    break
                previous_row = row
                previous_length = length
                previous_route = route
                stop_index += 1
        changes.append(
            AffineRouteChange(
                start=start_row,
                stop=items[stop_index - 1][0] + 1,
                route_start=start_route,
                route_step=route_step,
                length_start=start_length,
                length_step=length_step,
            )
        )
        index = stop_index
    return tuple(changes)


def materialize_route_changes(
    base: HardRouteResult,
    route_changes: Sequence[Sequence[AffineRouteChange]],
) -> tuple[Tensor, Tensor]:
    """Expand validation matrices from the compressed exact result."""

    route_rows = []
    length_rows = []
    for changes in route_changes:
        routes = base.routes.clone()
        lengths = base.lengths.clone()
        for change in changes:
            for row in range(change.start, change.stop):
                length, route = change.value(row)
                lengths[row] = length
                routes[row] = route
        route_rows.append(routes)
        length_rows.append(lengths)
    sequence_length = base.routes.numel()
    return (
        torch.stack(route_rows)
        if route_rows
        else torch.empty(0, sequence_length, dtype=torch.int64),
        torch.stack(length_rows)
        if length_rows
        else torch.empty(0, sequence_length, dtype=torch.int64),
    )


class _WinnerIntervalIndex:
    """Static segment tree for rows whose base winner contains one key point."""

    def __init__(self, routes: Tensor, lengths: Tensor, key_length: int) -> None:
        size = 1
        while size < max(1, key_length):
            size <<= 1
        self.size = size
        self.buckets: list[list[int]] = [[] for _ in range(2 * size)]
        for row in range(1, routes.numel()):
            route = int(routes[row])
            length = int(lengths[row])
            if route == 0 or length == 0:
                continue
            left = route - length
            right = route - 1
            self._add(left, right, row)
        self.logical_entries = sum(len(bucket) for bucket in self.buckets)

    def _add(self, left: int, right: int, row: int) -> None:
        left += self.size
        right += self.size
        while left <= right:
            if left & 1:
                self.buckets[left].append(row)
                left += 1
            if not right & 1:
                self.buckets[right].append(row)
                right -= 1
            left >>= 1
            right >>= 1

    def rows(self, key_position: int) -> tuple[int, ...]:
        node = key_position + self.size
        result: list[int] = []
        while node:
            result.extend(self.buckets[node])
            node >>= 1
        result.sort()
        return tuple(result)


class _CodePositionIndex:
    """Both occurrence lists and Python-int bitsets for query centers."""

    def __init__(self, query: Sequence[int]) -> None:
        positions: list[list[int]] = [[] for _ in range(256)]
        masks = [0] * 256
        for position in range(1, len(query)):
            code = int(query[position])
            positions[code].append(position)
            masks[code] |= 1 << position
        self.positions = tuple(tuple(values) for values in positions)
        self.masks = tuple(masks)

    def after(
        self,
        code: int,
        position: int,
        backend: Literal["lists", "bitset"],
    ) -> tuple[int, ...]:
        if backend == "lists":
            values = self.positions[code]
            return values[bisect_right(values, position) :]
        bits = self.masks[code] >> (position + 1)
        result: list[int] = []
        while bits:
            least = bits & -bits
            result.append(position + 1 + least.bit_length() - 1)
            bits ^= least
        return tuple(result)


@dataclass
class _WorkCounters:
    predecessor_queries: int = 0
    suffix_link_fallbacks: int = 0
    query_replay_steps: int = 0
    query_replay_possible_steps: int = 0
    query_replay_coalescences: int = 0
    query_branches_merged: int = 0
    replacement_rows: int = 0
    replacement_suffix_states: int = 0
    virtual_center_candidates: int = 0
    virtual_runs: int = 0
    virtual_active_rows: int = 0
    virtual_heap_pushes: int = 0
    virtual_heap_pops: int = 0
    changed_rows: int = 0
    route_change_descriptors: int = 0


@dataclass(frozen=True)
class SamBitflipProfile:
    endpos_backend: str
    center_backend: str
    hot_cache_size: int
    arithmetic_endpos: bool
    forward_states: int
    forward_edges: int
    reverse_states: int
    reverse_edges: int
    endpos_logical_entries: int
    winner_interval_entries: int
    predecessor_queries: int
    suffix_link_fallbacks: int
    cache_hits: int
    cache_negative_hits: int
    cache_cold_fallbacks: int
    arithmetic_certified_states: int
    arithmetic_hits: int
    arithmetic_fallbacks: int
    query_replay_steps: int
    query_replay_possible_steps: int
    query_replay_coalescences: int
    query_branches_merged: int
    replacement_rows: int
    replacement_suffix_states: int
    virtual_center_candidates: int
    virtual_runs: int
    virtual_active_rows: int
    virtual_heap_pushes: int
    virtual_heap_pops: int
    changed_rows: int
    route_change_descriptors: int


@dataclass(frozen=True)
class SamBitflipResult:
    flips: tuple[BitFlip, ...]
    base: HardRouteResult
    route_changes: tuple[tuple[AffineRouteChange, ...], ...]
    flipped_routes: Optional[Tensor]
    flipped_lengths: Optional[Tensor]
    bit_gradient: Optional[Tensor]
    profile: SamBitflipProfile


def _validate_inputs(query_codes: Tensor, key_codes: Tensor, bit_width: int) -> None:
    if query_codes.ndim != 1 or key_codes.ndim != 1:
        raise ValueError("query_codes and key_codes must be vectors")
    if query_codes.shape != key_codes.shape:
        raise ValueError("query_codes and key_codes must have one shape")
    if query_codes.dtype != torch.uint8 or key_codes.dtype != torch.uint8:
        raise ValueError("packed codes must use torch.uint8")
    if query_codes.device.type != "cpu" or key_codes.device.type != "cpu":
        raise ValueError("SAM bitflip is CPU-only")
    if not 1 <= bit_width <= 8:
        raise ValueError("bit_width must be in 1..8")
    if query_codes.numel() and (
        int(query_codes.max()) >= 1 << bit_width
        or int(key_codes.max()) >= 1 << bit_width
    ):
        raise ValueError("packed code uses bits outside bit_width")


class SamBitflipSolver:
    """Shared exact index and all one-bit ROSA counterfactuals."""

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        bit_width: int,
        *,
        endpos_backend: Literal["explicit", "implicit"] = "implicit",
        hot_cache_size: int = 0,
        center_backend: Literal["lists", "bitset"] = "bitset",
        arithmetic_endpos: bool = True,
    ) -> None:
        _validate_inputs(query_codes, key_codes, bit_width)
        if hot_cache_size < 0:
            raise ValueError("hot_cache_size must be nonnegative")
        if endpos_backend not in {"explicit", "implicit"}:
            raise ValueError("endpos_backend must be 'explicit' or 'implicit'")
        if center_backend not in {"lists", "bitset"}:
            raise ValueError("center_backend must be 'lists' or 'bitset'")

        self.query_codes = query_codes
        self.key_codes = key_codes
        self.query = tuple(int(code) for code in query_codes.tolist())
        self.key = tuple(int(code) for code in key_codes.tolist())
        self.bit_width = bit_width
        self.sequence_length = len(self.query)
        self.key_length = max(self.sequence_length - 1, 0)
        self.endpos_backend = endpos_backend
        self.hot_cache_size = hot_cache_size
        self.center_backend = center_backend
        self.arithmetic_endpos_enabled = arithmetic_endpos
        self.counters = _WorkCounters()

        usable_key = self.key[: self.key_length]
        self.automaton = SuffixAutomaton(usable_key)
        if endpos_backend == "explicit":
            cold_index: EndPositionIndex = ExplicitEndPositionIndex(self.automaton)
        else:
            cold_index = ImplicitEndPositionIndex(self.automaton)
        endpos: EndPositionIndex = cold_index
        if hot_cache_size:
            endpos = HybridEndPositionIndex(
                self.automaton,
                cold_index,
                hot_cache_size,
            )
        if arithmetic_endpos:
            endpos = ArithmeticEndPositionIndex(self.automaton, endpos)
        self.endpos = endpos

        self.base_traces = self._build_base_traces()
        self.base = self._hard_result(self.base_traces)
        self.full_query_traces = self._build_full_query_traces()

        reverse_key = tuple(reversed(usable_key))
        self.reverse_automaton = SuffixAutomaton(reverse_key)
        reverse_query = tuple(reversed(self.query[1:]))
        self.reverse_query_traces = self._trace_symbols_unrestricted(
            self.reverse_automaton,
            reverse_query,
        )

        self.winner_intervals = _WinnerIntervalIndex(
            self.base.routes,
            self.base.lengths,
            self.key_length,
        )
        self.query_positions = _CodePositionIndex(self.query)
        self._replacement_cache: dict[int, dict[int, tuple[int, int]]] = {}
        self._run_cache: dict[int, dict[int, tuple[VirtualRun, ...]]] = {}

    def _predecessor(
        self,
        index: EndPositionIndex,
        state: int,
        bound: int,
        *,
        count_work: bool,
    ) -> int:
        if count_work:
            self.counters.predecessor_queries += 1
        return index.predecessor(state, bound)

    def _advance(
        self,
        automaton: SuffixAutomaton,
        index: EndPositionIndex,
        previous: MatchTrace,
        symbol: int,
        bound: int,
        *,
        count_work: bool = True,
    ) -> MatchTrace:
        state = previous.state
        length = previous.length
        while True:
            next_state = automaton.states[state].transitions.get(symbol, -1)
            if next_state >= 0:
                next_length = length + 1
                next_state, next_length = automaton.normalize(
                    next_state,
                    next_length,
                )
                latest_end = self._predecessor(
                    index,
                    next_state,
                    bound,
                    count_work=count_work,
                )
                if latest_end >= 0:
                    return MatchTrace(next_state, next_length, latest_end)
            if state == 0:
                return MatchTrace(0, 0, -1)
            state = automaton.states[state].suffix_link
            length = min(length, automaton.states[state].max_length)
            if count_work:
                self.counters.suffix_link_fallbacks += 1

    def _advance_unrestricted(
        self,
        automaton: SuffixAutomaton,
        previous: MatchTrace,
        symbol: int,
    ) -> MatchTrace:
        state = previous.state
        length = previous.length
        next_state = automaton.states[state].transitions.get(symbol, -1)
        while state != 0 and next_state < 0:
            state = automaton.states[state].suffix_link
            length = min(length, automaton.states[state].max_length)
            next_state = automaton.states[state].transitions.get(symbol, -1)
        if next_state < 0:
            return MatchTrace(0, 0, -1)
        next_state, next_length = automaton.normalize(next_state, length + 1)
        return MatchTrace(
            next_state,
            next_length,
            automaton.latest_ends[next_state],
        )

    def _trace_symbols_unrestricted(
        self,
        automaton: SuffixAutomaton,
        symbols: Sequence[int],
    ) -> tuple[MatchTrace, ...]:
        traces = []
        previous = MatchTrace(0, 0, -1)
        for symbol in symbols:
            previous = self._advance_unrestricted(
                automaton,
                previous,
                symbol,
            )
            traces.append(previous)
        return tuple(traces)

    def _build_base_traces(self) -> tuple[MatchTrace, ...]:
        if not self.sequence_length:
            return ()
        traces = [MatchTrace(0, 0, -1)]
        for row in range(1, self.sequence_length):
            traces.append(
                self._advance(
                    self.automaton,
                    self.endpos,
                    traces[-1],
                    self.query[row],
                    row - 1,
                )
            )
        return tuple(traces)

    def _build_full_query_traces(self) -> tuple[MatchTrace, ...]:
        if not self.sequence_length:
            return ()
        traces = [MatchTrace(0, 0, -1)]
        for row in range(1, self.sequence_length):
            traces.append(
                self._advance_unrestricted(
                    self.automaton,
                    traces[-1],
                    self.query[row],
                )
            )
        return tuple(traces)

    def _hard_result(self, traces: Sequence[MatchTrace]) -> HardRouteResult:
        routes = torch.tensor(
            [trace.route for trace in traces],
            dtype=torch.int64,
        )
        lengths = torch.tensor(
            [trace.length for trace in traces],
            dtype=torch.int64,
        )
        return HardRouteResult(routes, lengths, None, 0)

    def left_context(self, query_position: int, key_position: int) -> int:
        """Matches immediately left of an anchored ``(Q[p], K[j])`` pair."""

        if query_position <= 0 or key_position <= 0:
            return 0
        return self.automaton.common_suffix_length(
            self.full_query_traces[query_position - 1],
            key_position - 1,
        )

    def right_context(self, query_position: int, key_position: int) -> int:
        """Matches immediately right of an anchored ``(Q[p], K[j])`` pair."""

        if (
            query_position + 1 >= self.sequence_length
            or key_position + 1 >= self.key_length
        ):
            return 0
        reverse_query_index = self.key_length - query_position - 1
        reverse_key_end = self.key_length - key_position - 2
        return self.reverse_automaton.common_suffix_length(
            self.reverse_query_traces[reverse_query_index],
            reverse_key_end,
        )

    def _record_query_update(
        self,
        updates: dict[int, tuple[int, int]],
        row: int,
        trace: MatchTrace,
    ) -> None:
        candidate = trace.length, trace.route
        base = int(self.base.lengths[row]), int(self.base.routes[row])
        if candidate != base:
            updates[row] = candidate

    def _query_flip_updates(self) -> list[dict[int, tuple[int, int]]]:
        """Replay all D alternatives per position with exact trace merging."""

        all_updates: list[dict[int, tuple[int, int]]] = []
        for position in range(1, self.sequence_length):
            updates = [dict() for _ in range(self.bit_width)]
            previous = self.base_traces[position - 1]
            groups: dict[MatchTrace, list[int]] = {}
            self.counters.query_replay_possible_steps += self.bit_width * (
                self.sequence_length - position
            )
            for bit in range(self.bit_width):
                trace = self._advance(
                    self.automaton,
                    self.endpos,
                    previous,
                    self.query[position] ^ (1 << bit),
                    position - 1,
                )
                self.counters.query_replay_steps += 1
                self._record_query_update(updates[bit], position, trace)
                if trace == self.base_traces[position]:
                    self.counters.query_replay_coalescences += 1
                else:
                    groups.setdefault(trace, []).append(bit)

            for row in range(position + 1, self.sequence_length):
                if not groups:
                    break
                self.counters.query_branches_merged += sum(
                    len(bits) - 1 for bits in groups.values()
                )
                next_groups: dict[MatchTrace, list[int]] = {}
                for previous_trace, bits in groups.items():
                    trace = self._advance(
                        self.automaton,
                        self.endpos,
                        previous_trace,
                        self.query[row],
                        row - 1,
                    )
                    self.counters.query_replay_steps += 1
                    for bit in bits:
                        self._record_query_update(updates[bit], row, trace)
                    if trace == self.base_traces[row]:
                        self.counters.query_replay_coalescences += len(bits)
                    else:
                        next_groups.setdefault(trace, []).extend(bits)
                groups = next_groups
            all_updates.extend(updates)
        return all_updates

    def _best_avoiding_key_position(
        self,
        trace: MatchTrace,
        bound: int,
        key_position: int,
    ) -> tuple[int, int]:
        """Longest/latest match whose key interval excludes one position."""

        def feasible(length: int) -> bool:
            self.counters.replacement_suffix_states += 1
            state = self.automaton.state_for_length(trace.state, length)
            left_end = self._predecessor(
                self.endpos,
                state,
                min(bound, key_position - 1),
                count_work=True,
            )
            if left_end >= 0:
                return True
            latest_end = self._predecessor(
                self.endpos,
                state,
                bound,
                count_work=True,
            )
            return latest_end >= key_position + length

        low = 0
        high = trace.length
        while low < high:
            middle = (low + high + 1) // 2
            if feasible(middle):
                low = middle
            else:
                high = middle - 1
        if low == 0:
            return 0, 0

        state = self.automaton.state_for_length(trace.state, low)
        left_end = self._predecessor(
            self.endpos,
            state,
            min(bound, key_position - 1),
            count_work=True,
        )
        latest_end = self._predecessor(
            self.endpos,
            state,
            bound,
            count_work=True,
        )
        end = latest_end if latest_end >= key_position + low else left_end
        return low, end + 1

    def _replacement_rows(self, key_position: int) -> dict[int, tuple[int, int]]:
        cached = self._replacement_cache.get(key_position)
        if cached is not None:
            return cached
        replacements: dict[int, tuple[int, int]] = {}
        for row in self.winner_intervals.rows(key_position):
            self.counters.replacement_rows += 1
            replacements[row] = self._best_avoiding_key_position(
                self.base_traces[row],
                row - 1,
                key_position,
            )
        self._replacement_cache[key_position] = replacements
        return replacements

    def _virtual_run_batches(
        self,
        key_position: int,
    ) -> dict[int, tuple[VirtualRun, ...]]:
        cached = self._run_cache.get(key_position)
        if cached is not None:
            return cached
        batches: dict[int, tuple[VirtualRun, ...]] = {}
        source = self.key[key_position]
        for bit in range(self.bit_width):
            target = source ^ (1 << bit)
            centers = self.query_positions.after(
                target,
                key_position,
                self.center_backend,
            )
            self.counters.virtual_center_candidates += len(centers)
            runs = []
            for query_position in centers:
                left = self.left_context(query_position, key_position)
                right = self.right_context(query_position, key_position)
                runs.append(
                    VirtualRun(
                        key_position=key_position,
                        query_position=query_position,
                        stop=query_position + right,
                        length_offset=left + 1 - query_position,
                        route_offset=key_position + 1 - query_position,
                    )
                )
            batches[bit] = tuple(runs)
            self.counters.virtual_runs += len(runs)
        self._run_cache[key_position] = batches
        return batches

    def _apply_virtual_runs(
        self,
        updates: dict[int, tuple[int, int]],
        runs: Sequence[VirtualRun],
    ) -> None:
        if not runs:
            return
        heap: list[tuple[int, int, int, int]] = []
        run_index = 0
        row = runs[0].query_position
        while run_index < len(runs) or heap:
            if not heap and run_index < len(runs):
                row = max(row, runs[run_index].query_position)
            while (
                run_index < len(runs)
                and runs[run_index].query_position <= row
            ):
                run = runs[run_index]
                heappush(
                    heap,
                    (
                        -run.length_offset,
                        -run.route_offset,
                        run.stop,
                        run.query_position,
                    ),
                )
                self.counters.virtual_heap_pushes += 1
                run_index += 1
            while heap and heap[0][2] < row:
                heappop(heap)
                self.counters.virtual_heap_pops += 1
            if not heap:
                continue

            self.counters.virtual_active_rows += 1
            length = -heap[0][0] + row
            route = -heap[0][1] + row
            base = int(self.base.lengths[row]), int(self.base.routes[row])
            current = updates.get(row, base)
            if (length, route) > current:
                if (length, route) == base:
                    updates.pop(row, None)
                else:
                    updates[row] = length, route
            row += 1

    def _solve_key_flip(self, flip: BitFlip) -> dict[int, tuple[int, int]]:
        updates: dict[int, tuple[int, int]] = {}
        for row, (length, route) in self._replacement_rows(
            flip.position
        ).items():
            base = int(self.base.lengths[row]), int(self.base.routes[row])
            if (length, route) != base:
                updates[row] = length, route
        runs = self._virtual_run_batches(flip.position)[flip.bit]
        self._apply_virtual_runs(updates, runs)
        return updates

    def solve(
        self,
        *,
        value: Optional[Tensor] = None,
        grad_output: Optional[Tensor] = None,
        materialize_routes: bool = False,
    ) -> SamBitflipResult:
        flips = semantic_bit_flips(self.sequence_length, self.bit_width)
        update_rows = self._query_flip_updates()
        update_rows.extend(
            self._solve_key_flip(flip)
            for flip in flips
            if flip.source == "key"
        )
        if len(update_rows) != len(flips):
            raise RuntimeError("counterfactual rows do not match semantic flips")
        route_changes = tuple(
            _compress_updates(updates) for updates in update_rows
        )
        self.counters.changed_rows = sum(
            len(updates) for updates in update_rows
        )
        self.counters.route_change_descriptors = sum(
            len(changes) for changes in route_changes
        )
        if materialize_routes:
            flipped_routes, flipped_lengths = materialize_route_changes(
                self.base,
                route_changes,
            )
        else:
            flipped_routes = flipped_lengths = None
        bit_gradient = self._bit_gradient(
            flips,
            route_changes,
            value,
            grad_output,
        )
        return SamBitflipResult(
            flips=flips,
            base=self.base,
            route_changes=route_changes,
            flipped_routes=flipped_routes,
            flipped_lengths=flipped_lengths,
            bit_gradient=bit_gradient,
            profile=self.profile(),
        )

    def _bit_gradient(
        self,
        flips: Sequence[BitFlip],
        route_changes: Sequence[Sequence[AffineRouteChange]],
        value: Optional[Tensor],
        grad_output: Optional[Tensor],
    ) -> Optional[Tensor]:
        if value is None and grad_output is None:
            return None
        if value is None or grad_output is None:
            raise ValueError("value and grad_output must be supplied together")
        if value.ndim != 2 or value.shape != grad_output.shape:
            raise ValueError("value and grad_output must share shape [T, Dv]")
        if value.size(0) != self.sequence_length:
            raise ValueError("value length must match the code sequence")
        hard_value = torch.where(
            value > 0,
            torch.ones_like(value),
            -torch.ones_like(value),
        ).clone()
        if hard_value.size(0):
            hard_value[0] = 0
        base_output = hard_output_from_routes(self.base.routes, value)
        gradients = []
        for flip, changes in zip(flips, route_changes):
            delta = value.new_zeros(())
            for change in changes:
                for row in range(change.start, change.stop):
                    _, route = change.value(row)
                    delta = delta + (
                        (hard_value[route] - base_output[row])
                        * grad_output[row]
                    ).sum()
            codes = self.query if flip.source == "query" else self.key
            sign = 1.0 if codes[flip.position] & (1 << flip.bit) else -1.0
            gradients.append(-sign * delta)
        return torch.stack(gradients) if gradients else value.new_empty(0)

    def profile(self) -> SamBitflipProfile:
        arithmetic = (
            self.endpos
            if isinstance(self.endpos, ArithmeticEndPositionIndex)
            else None
        )
        fallback = arithmetic.fallback if arithmetic is not None else self.endpos
        cache = fallback if isinstance(fallback, HybridEndPositionIndex) else None
        return SamBitflipProfile(
            endpos_backend=self.endpos_backend,
            center_backend=self.center_backend,
            hot_cache_size=self.hot_cache_size,
            arithmetic_endpos=self.arithmetic_endpos_enabled,
            forward_states=self.automaton.state_count,
            forward_edges=self.automaton.edge_count,
            reverse_states=self.reverse_automaton.state_count,
            reverse_edges=self.reverse_automaton.edge_count,
            endpos_logical_entries=self.endpos.logical_entries,
            winner_interval_entries=self.winner_intervals.logical_entries,
            predecessor_queries=self.counters.predecessor_queries,
            suffix_link_fallbacks=self.counters.suffix_link_fallbacks,
            cache_hits=cache.hits if cache is not None else 0,
            cache_negative_hits=cache.negative_hits if cache is not None else 0,
            cache_cold_fallbacks=cache.cold_fallbacks if cache is not None else 0,
            arithmetic_certified_states=(
                arithmetic.certified_states if arithmetic is not None else 0
            ),
            arithmetic_hits=arithmetic.hits if arithmetic is not None else 0,
            arithmetic_fallbacks=(
                arithmetic.fallbacks if arithmetic is not None else 0
            ),
            query_replay_steps=self.counters.query_replay_steps,
            query_replay_possible_steps=self.counters.query_replay_possible_steps,
            query_replay_coalescences=self.counters.query_replay_coalescences,
            query_branches_merged=self.counters.query_branches_merged,
            replacement_rows=self.counters.replacement_rows,
            replacement_suffix_states=self.counters.replacement_suffix_states,
            virtual_center_candidates=self.counters.virtual_center_candidates,
            virtual_runs=self.counters.virtual_runs,
            virtual_active_rows=self.counters.virtual_active_rows,
            virtual_heap_pushes=self.counters.virtual_heap_pushes,
            virtual_heap_pops=self.counters.virtual_heap_pops,
            changed_rows=self.counters.changed_rows,
            route_change_descriptors=self.counters.route_change_descriptors,
        )


def sam_bitflip(
    query_codes: Tensor,
    key_codes: Tensor,
    bit_width: int,
    *,
    value: Optional[Tensor] = None,
    grad_output: Optional[Tensor] = None,
    endpos_backend: Literal["explicit", "implicit"] = "implicit",
    hot_cache_size: int = 0,
    center_backend: Literal["lists", "bitset"] = "bitset",
    arithmetic_endpos: bool = True,
    materialize_routes: bool = False,
) -> SamBitflipResult:
    """Evaluate every semantically active bit with the exact SAM route."""

    return SamBitflipSolver(
        query_codes,
        key_codes,
        bit_width,
        endpos_backend=endpos_backend,
        hot_cache_size=hot_cache_size,
        center_backend=center_backend,
        arithmetic_endpos=arithmetic_endpos,
    ).solve(
        value=value,
        grad_output=grad_output,
        materialize_routes=materialize_routes,
    )
