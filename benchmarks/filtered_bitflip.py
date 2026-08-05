"""Exact research prototypes for filtered unlimited-suffix ROSA bit flips.

The frozen RosaSoft operator deliberately retains its finite-window dense
surrogate.  This module defines a separate, CPU-only semantic oracle for a
future estimator whose hard suffix has no configured horizon.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from heapq import merge
from typing import Iterable, Optional, Protocol, Sequence

import torch
from torch import Tensor


__all__ = [
    "BitFlip",
    "BitflipResult",
    "CodeOccurrenceIndex",
    "HardRouteResult",
    "InfluenceEvent",
    "RouteChangeRange",
    "apply_influence_events",
    "brute_force_bitflip",
    "filtered_bitflip",
    "generate_influence_events",
    "hard_output_from_routes",
    "pack_sign_codes",
    "profile_bitflip_structure",
    "route_change_ranges",
    "semantic_bit_flips",
    "unlimited_hard_routes_diagonal",
    "unlimited_hard_routes_direct",
]


class MatchLengthIndex(Protocol):
    """Optional exact or experimental LCE backend used by event discovery."""

    def left_matches(self, query_position: int, key_position: int) -> int: ...

    def right_matches(self, query_position: int, key_position: int) -> int: ...


@dataclass(frozen=True)
class CodeOccurrenceIndex:
    """Contiguous positions grouped by their packed byte code."""

    query_codes: tuple[int, ...]
    key_codes: tuple[int, ...]
    query_positions: tuple[tuple[int, ...], ...]
    key_positions: tuple[tuple[int, ...], ...]

    @classmethod
    def build(
        cls,
        query_codes: Tensor,
        key_codes: Tensor,
    ) -> "CodeOccurrenceIndex":
        _validate_codes(query_codes, key_codes)
        query_buckets: list[list[int]] = [[] for _ in range(256)]
        key_buckets: list[list[int]] = [[] for _ in range(256)]
        query_values = _code_list(query_codes)
        key_values = _code_list(key_codes)
        for position, code in enumerate(query_values):
            query_buckets[code].append(position)
        for position, code in enumerate(key_values):
            key_buckets[code].append(position)
        return cls(
            tuple(query_values),
            tuple(key_values),
            tuple(tuple(bucket) for bucket in query_buckets),
            tuple(tuple(bucket) for bucket in key_buckets),
        )

    def changed_pairs(
        self,
        flip: BitFlip,
    ) -> tuple[tuple[int, int], ...]:
        """Return only code-equal or one-target-bit-different local pairs."""

        mask = 1 << flip.bit
        if flip.source == "query":
            position = flip.position
            code = self.query_codes[position]
            selected = []
            for target in (code, code ^ mask):
                positions = self.key_positions[target]
                stop = bisect_left(positions, position)
                selected.append(positions[:stop])
            return tuple(
                (position, key_position)
                for key_position in merge(*selected)
            )
        else:
            position = flip.position
            code = self.key_codes[position]
            selected = []
            for target in (code, code ^ mask):
                positions = self.query_positions[target]
                start = bisect_right(positions, position)
                selected.append(positions[start:])
            return tuple(
                (query_position, position)
                for query_position in merge(*selected)
            )


@dataclass(frozen=True)
class HardRouteResult:
    """Latest-longest routes for one packed Q/K sequence."""

    routes: Tensor
    lengths: Tensor
    candidate_lengths: Optional[Tensor]
    symbol_comparisons: int


@dataclass(frozen=True, order=True)
class BitFlip:
    """One semantically active Q or K code bit."""

    source: str
    position: int
    bit: int

    def __post_init__(self) -> None:
        if self.source not in {"query", "key"}:
            raise ValueError("source must be 'query' or 'key'")
        if self.position < 0 or self.bit < 0:
            raise ValueError("position and bit must be nonnegative")


@dataclass(frozen=True)
class InfluenceEvent:
    """One changed local equality and its exact downstream diagonal range."""

    flip: BitFlip
    query_position: int
    key_position: int
    left_matches: int
    right_matches: int
    creates_match: bool

    @property
    def start(self) -> int:
        return self.query_position

    @property
    def stop(self) -> int:
        """Exclusive query-row bound."""

        return self.query_position + self.right_matches + 1

    @property
    def route_offset(self) -> int:
        return self.key_position + 1 - self.query_position

    @property
    def long_normalized_length(self) -> int:
        return self.left_matches + 1 - self.query_position

    @property
    def short_normalized_length(self) -> int:
        return -self.query_position

    def old_length(self, query_index: int) -> int:
        suffix = query_index - self.query_position
        if not 0 <= suffix <= self.right_matches:
            raise IndexError("query index is outside the influence event")
        if self.creates_match:
            return suffix
        return self.left_matches + 1 + suffix

    def new_length(self, query_index: int) -> int:
        suffix = query_index - self.query_position
        if not 0 <= suffix <= self.right_matches:
            raise IndexError("query index is outside the influence event")
        if self.creates_match:
            return self.left_matches + 1 + suffix
        return suffix


@dataclass(frozen=True)
class RouteChangeRange:
    """A contiguous changed-row range with one affine new route."""

    start: int
    stop: int
    route_offset: int


@dataclass(frozen=True)
class BitflipResult:
    """All one-bit hard counterfactuals for one sequence."""

    flips: tuple[BitFlip, ...]
    base: HardRouteResult
    flipped_routes: Tensor
    flipped_lengths: Tensor
    bit_gradient: Optional[Tensor]
    events: Optional[tuple[tuple[InfluenceEvent, ...], ...]] = None


def _validate_codes(query_codes: Tensor, key_codes: Tensor) -> None:
    if query_codes.ndim != 1 or key_codes.ndim != 1:
        raise ValueError("query_codes and key_codes must be vectors")
    if query_codes.shape != key_codes.shape:
        raise ValueError("query_codes and key_codes must have one shape")
    if query_codes.dtype != torch.uint8 or key_codes.dtype != torch.uint8:
        raise ValueError("packed codes must use torch.uint8")
    if query_codes.device.type != "cpu" or key_codes.device.type != "cpu":
        raise ValueError("unlimited bitflip prototypes are CPU-only")


def _code_list(codes: Tensor) -> list[int]:
    return [int(value) for value in codes.tolist()]


def pack_sign_codes(logits: Tensor) -> Tensor:
    """Pack up to eight sign bits into one byte per sequence position."""

    if logits.ndim == 4:
        if logits.size(0) != 1 or logits.size(2) != 1:
            raise ValueError("rank-4 logits must have shape [1, T, 1, D]")
        logits = logits[0, :, 0, :]
    if logits.ndim != 2:
        raise ValueError("logits must have shape [T, D] or [1, T, 1, D]")
    if not logits.dtype.is_floating_point:
        raise ValueError("logits must be floating point")
    if logits.device.type != "cpu":
        raise ValueError("pack_sign_codes currently requires CPU logits")
    bit_width = logits.size(1)
    if not 1 <= bit_width <= 8:
        raise ValueError("the packed prototype supports 1..8 bits")
    shifts = torch.arange(bit_width, dtype=torch.int64)
    positive = (logits > 0).to(torch.int64)
    return ((positive << shifts).sum(dim=-1)).to(torch.uint8).contiguous()


def _select_route_rows(candidate_lengths: Tensor) -> tuple[Tensor, Tensor]:
    sequence_length = candidate_lengths.size(0)
    routes = torch.zeros(sequence_length, dtype=torch.int64)
    lengths = torch.zeros(sequence_length, dtype=torch.int64)
    for query_index in range(1, sequence_length):
        row = candidate_lengths[query_index, 1 : query_index + 1]
        best_length = int(row.max()) if row.numel() else 0
        if best_length <= 0:
            continue
        winners = torch.nonzero(row == best_length, as_tuple=False).flatten()
        routes[query_index] = int(winners[-1]) + 1
        lengths[query_index] = best_length
    return routes, lengths


def unlimited_hard_routes_diagonal(
    query_codes: Tensor,
    key_codes: Tensor,
    *,
    return_candidate_lengths: bool = False,
) -> HardRouteResult:
    """Compute unlimited suffixes with the exact diagonal recurrence."""

    _validate_codes(query_codes, key_codes)
    query = _code_list(query_codes)
    key = _code_list(key_codes)
    sequence_length = len(query)
    previous = [0] * sequence_length
    routes = torch.zeros(sequence_length, dtype=torch.int64)
    lengths = torch.zeros(sequence_length, dtype=torch.int64)
    candidates = (
        torch.zeros(
            sequence_length,
            sequence_length,
            dtype=torch.int32,
        )
        if return_candidate_lengths
        else None
    )
    comparisons = 0
    for query_index in range(1, sequence_length):
        current = [0] * sequence_length
        best_length = 0
        best_route = 0
        for route in range(1, query_index + 1):
            comparisons += 1
            if query[query_index] == key[route - 1]:
                current[route] = previous[route - 1] + 1
            length = current[route]
            if candidates is not None:
                candidates[query_index, route] = length
            if length > best_length or (
                length == best_length and length > 0
            ):
                best_length = length
                best_route = route
        routes[query_index] = best_route
        lengths[query_index] = best_length
        previous = current
    return HardRouteResult(
        routes=routes,
        lengths=lengths,
        candidate_lengths=candidates,
        symbol_comparisons=comparisons,
    )


def unlimited_hard_routes_direct(
    query_codes: Tensor,
    key_codes: Tensor,
    *,
    return_candidate_lengths: bool = False,
) -> HardRouteResult:
    """Compute each candidate by a direct packed suffix scan."""

    _validate_codes(query_codes, key_codes)
    query = _code_list(query_codes)
    key = _code_list(key_codes)
    sequence_length = len(query)
    routes = torch.zeros(sequence_length, dtype=torch.int64)
    lengths = torch.zeros(sequence_length, dtype=torch.int64)
    candidates = (
        torch.zeros(
            sequence_length,
            sequence_length,
            dtype=torch.int32,
        )
        if return_candidate_lengths
        else None
    )
    comparisons = 0
    for query_index in range(1, sequence_length):
        best_length = 0
        best_route = 0
        for route in range(1, query_index + 1):
            length = 0
            while length < route:
                comparisons += 1
                if query[query_index - length] != key[route - 1 - length]:
                    break
                length += 1
            if candidates is not None:
                candidates[query_index, route] = length
            if length > best_length or (
                length == best_length and length > 0
            ):
                best_length = length
                best_route = route
        routes[query_index] = best_route
        lengths[query_index] = best_length
    return HardRouteResult(
        routes=routes,
        lengths=lengths,
        candidate_lengths=candidates,
        symbol_comparisons=comparisons,
    )


def hard_output_from_routes(routes: Tensor, value: Tensor) -> Tensor:
    """Gather hard-sign values, reserving value position zero for null."""

    if routes.ndim != 1 or value.ndim != 2:
        raise ValueError("routes and value must have shapes [T] and [T, Dv]")
    if value.size(0) != routes.numel() or not value.dtype.is_floating_point:
        raise ValueError("value must be floating point and match route length")
    hard_value = torch.where(
        value > 0,
        torch.ones_like(value),
        -torch.ones_like(value),
    )
    hard_value = hard_value.clone()
    if hard_value.size(0):
        hard_value[0] = 0
    return hard_value[routes]


def semantic_bit_flips(sequence_length: int, bit_width: int) -> tuple[BitFlip, ...]:
    """Return bits that can affect causal shifted ROSA routes."""

    if sequence_length < 0:
        raise ValueError("sequence_length must be nonnegative")
    if not 1 <= bit_width <= 8:
        raise ValueError("bit_width must be in 1..8")
    query = tuple(
        BitFlip("query", position, bit)
        for position in range(1, sequence_length)
        for bit in range(bit_width)
    )
    key = tuple(
        BitFlip("key", position, bit)
        for position in range(max(sequence_length - 1, 0))
        for bit in range(bit_width)
    )
    return query + key


def _apply_flip(codes: Tensor, flip: BitFlip) -> Tensor:
    result = codes.clone()
    result[flip.position] = result[flip.position] ^ (1 << flip.bit)
    return result


def _bit_gradient(
    flips: Sequence[BitFlip],
    query_codes: Tensor,
    key_codes: Tensor,
    base_routes: Tensor,
    flipped_routes: Tensor,
    value: Optional[Tensor],
    grad_output: Optional[Tensor],
) -> Optional[Tensor]:
    if value is None and grad_output is None:
        return None
    if value is None or grad_output is None:
        raise ValueError("value and grad_output must be supplied together")
    if value.shape != grad_output.shape or value.ndim != 2:
        raise ValueError("value and grad_output must share shape [T, Dv]")
    base_output = hard_output_from_routes(base_routes, value)
    gradients = []
    for flip, routes in zip(flips, flipped_routes):
        output = hard_output_from_routes(routes, value)
        delta = ((output - base_output) * grad_output).sum()
        code = query_codes if flip.source == "query" else key_codes
        base_symbol = 1.0 if int(code[flip.position]) & (1 << flip.bit) else -1.0
        gradients.append(-base_symbol * delta)
    if not gradients:
        return value.new_empty(0)
    return torch.stack(gradients)


def brute_force_bitflip(
    query_codes: Tensor,
    key_codes: Tensor,
    bit_width: int,
    *,
    value: Optional[Tensor] = None,
    grad_output: Optional[Tensor] = None,
) -> BitflipResult:
    """Rerun the complete unlimited hard route after every one-bit edit."""

    _validate_codes(query_codes, key_codes)
    flips = semantic_bit_flips(query_codes.numel(), bit_width)
    base = unlimited_hard_routes_diagonal(
        query_codes,
        key_codes,
        return_candidate_lengths=True,
    )
    route_rows = []
    length_rows = []
    for flip in flips:
        if flip.source == "query":
            query = _apply_flip(query_codes, flip)
            key = key_codes
        else:
            query = query_codes
            key = _apply_flip(key_codes, flip)
        result = unlimited_hard_routes_diagonal(query, key)
        route_rows.append(result.routes)
        length_rows.append(result.lengths)
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
    return BitflipResult(
        flips=flips,
        base=base,
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
    )


def _left_matches(
    query: Sequence[int],
    key: Sequence[int],
    query_position: int,
    key_position: int,
) -> int:
    length = 0
    while (
        query_position - length - 1 >= 0
        and key_position - length - 1 >= 0
        and query[query_position - length - 1]
        == key[key_position - length - 1]
    ):
        length += 1
    return length


def _right_matches(
    query: Sequence[int],
    key: Sequence[int],
    query_position: int,
    key_position: int,
) -> int:
    length = 0
    sequence_length = len(query)
    while (
        query_position + length + 1 < sequence_length
        and key_position + length + 1 < sequence_length - 1
        and query[query_position + length + 1]
        == key[key_position + length + 1]
    ):
        length += 1
    return length


def generate_influence_events(
    query_codes: Tensor,
    key_codes: Tensor,
    flip: BitFlip,
    *,
    match_index: Optional[MatchLengthIndex] = None,
    occurrence_index: Optional[CodeOccurrenceIndex] = None,
) -> tuple[InfluenceEvent, ...]:
    """Generate every exact local match transition caused by one bit flip."""

    _validate_codes(query_codes, key_codes)
    sequence_length = query_codes.numel()
    if flip.bit >= 8:
        raise ValueError("flip bit is outside the packed code")
    if flip.source == "query" and not 1 <= flip.position < sequence_length:
        raise ValueError("query flip is not semantically active")
    if flip.source == "key" and not 0 <= flip.position < sequence_length - 1:
        raise ValueError("key flip is not semantically active")
    query = _code_list(query_codes)
    key = _code_list(key_codes)
    mask = 1 << flip.bit
    if occurrence_index is not None:
        pairs: Iterable[tuple[int, int]] = occurrence_index.changed_pairs(flip)
    elif flip.source == "query":
        pairs: Iterable[tuple[int, int]] = (
            (flip.position, key_position)
            for key_position in range(flip.position)
        )
    else:
        pairs = (
            (query_position, flip.position)
            for query_position in range(flip.position + 1, sequence_length)
        )
    events = []
    for query_position, key_position in pairs:
        difference = query[query_position] ^ key[key_position]
        if difference == 0:
            creates_match = False
        elif difference == mask:
            creates_match = True
        else:
            continue
        events.append(
            InfluenceEvent(
                flip=flip,
                query_position=query_position,
                key_position=key_position,
                left_matches=(
                    match_index.left_matches(query_position, key_position)
                    if match_index is not None
                    else _left_matches(
                        query,
                        key,
                        query_position,
                        key_position,
                    )
                ),
                right_matches=(
                    match_index.right_matches(query_position, key_position)
                    if match_index is not None
                    else _right_matches(
                        query,
                        key,
                        query_position,
                        key_position,
                    )
                ),
                creates_match=creates_match,
            )
        )
    return tuple(events)


def apply_influence_events(
    base_candidate_lengths: Tensor,
    events: Sequence[InfluenceEvent],
) -> tuple[Tensor, Tensor, Tensor]:
    """Apply exact event edits to a base candidate matrix."""

    if (
        base_candidate_lengths.ndim != 2
        or base_candidate_lengths.size(0) != base_candidate_lengths.size(1)
    ):
        raise ValueError("base_candidate_lengths must be square")
    candidates = base_candidate_lengths.clone()
    touched: set[tuple[int, int]] = set()
    for event in events:
        for query_index in range(event.start, event.stop):
            route = query_index + event.route_offset
            cell = (query_index, route)
            if cell in touched:
                raise RuntimeError("one bit flip produced overlapping events")
            touched.add(cell)
            expected = event.old_length(query_index)
            actual = int(candidates[query_index, route])
            if actual != expected:
                raise RuntimeError(
                    "event formula disagrees with the base candidate: "
                    f"cell={cell}, expected={expected}, actual={actual}"
                )
            candidates[query_index, route] = event.new_length(query_index)
    routes, lengths = _select_route_rows(candidates)
    return routes, lengths, candidates


def filtered_bitflip(
    query_codes: Tensor,
    key_codes: Tensor,
    bit_width: int,
    *,
    value: Optional[Tensor] = None,
    grad_output: Optional[Tensor] = None,
) -> BitflipResult:
    """Evaluate every bit while filtering work through exact influence ranges."""

    _validate_codes(query_codes, key_codes)
    flips = semantic_bit_flips(query_codes.numel(), bit_width)
    base = unlimited_hard_routes_diagonal(
        query_codes,
        key_codes,
        return_candidate_lengths=True,
    )
    if base.candidate_lengths is None:
        raise RuntimeError("candidate matrix was not constructed")
    route_rows = []
    length_rows = []
    event_rows = []
    for flip in flips:
        events = generate_influence_events(query_codes, key_codes, flip)
        routes, lengths, _ = apply_influence_events(
            base.candidate_lengths,
            events,
        )
        route_rows.append(routes)
        length_rows.append(lengths)
        event_rows.append(events)
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
    return BitflipResult(
        flips=flips,
        base=base,
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
        events=tuple(event_rows),
    )


def route_change_ranges(
    base_routes: Tensor,
    flipped_routes: Tensor,
) -> tuple[RouteChangeRange, ...]:
    """Run-length encode changed rows by affine route offset."""

    if base_routes.ndim != 1 or flipped_routes.shape != base_routes.shape:
        raise ValueError("base_routes and flipped_routes must share shape [T]")
    return _route_change_ranges_values(
        [int(value) for value in base_routes.tolist()],
        [int(value) for value in flipped_routes.tolist()],
    )


def _route_change_ranges_values(
    base_routes: Sequence[int],
    flipped_routes: Sequence[int],
) -> tuple[RouteChangeRange, ...]:
    if len(base_routes) != len(flipped_routes):
        raise ValueError("base and flipped route lists must have one length")
    ranges = []
    start: Optional[int] = None
    offset: Optional[int] = None
    for query_index in range(len(base_routes) + 1):
        changed = (
            query_index < len(base_routes)
            and base_routes[query_index] != flipped_routes[query_index]
        )
        current_offset = (
            flipped_routes[query_index] - query_index if changed else None
        )
        if start is not None and (not changed or current_offset != offset):
            ranges.append(RouteChangeRange(start, query_index, int(offset)))
            start = None
            offset = None
        if changed and start is None:
            start = query_index
            offset = current_offset
    return tuple(ranges)


def profile_bitflip_structure(result: BitflipResult) -> dict[str, int | float]:
    """Summarize event expansion and final route-change compressibility."""

    if result.events is None:
        raise ValueError("structural profiling requires an event result")
    raw_events = sum(len(events) for events in result.events)
    event_cells = sum(
        event.stop - event.start
        for events in result.events
        for event in events
    )
    changed_rows = 0
    ranges = []
    offsets: set[int] = set()
    max_ranges_per_flip = 0
    for routes in result.flipped_routes:
        changed_rows += int((routes != result.base.routes).sum())
        flip_ranges = route_change_ranges(result.base.routes, routes)
        ranges.extend(flip_ranges)
        max_ranges_per_flip = max(max_ranges_per_flip, len(flip_ranges))
        offsets.update(item.route_offset for item in flip_ranges)
    flip_count = len(result.flips)
    return {
        "flip_count": flip_count,
        "raw_local_pair_events": raw_events,
        "event_cell_updates": event_cells,
        "changed_query_rows": changed_rows,
        "merged_route_change_ranges": len(ranges),
        "distinct_new_route_offsets": len(offsets),
        "max_ranges_per_flip": max_ranges_per_flip,
        "events_per_flip": raw_events / max(flip_count, 1),
        "ranges_per_flip": len(ranges) / max(flip_count, 1),
    }
