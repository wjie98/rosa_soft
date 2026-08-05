"""Exact periodic event families and whole-interval v1 fallbacks."""

from __future__ import annotations

from dataclasses import dataclass
from heapq import heappop, heappush

import torch
from torch import Tensor

from benchmarks.filtered_bitflip import BitFlip, _validate_codes
from benchmarks.filtered_bitflip_certificates import (
    PeriodicEventFamily,
    Priority,
    WinnerSegment,
)
from benchmarks.filtered_bitflip_suffix_nodes import ArithmeticRouteRun


__all__ = [
    "AffineWinnerSegment",
    "DirectPeriodicEventFamilies",
    "PeriodicIntervalStats",
    "ShiftPeriodicEventFamilyBuilder",
    "ShiftAlignedPeriodicKeyFallback",
    "ShiftAlignedQuerySuffixFallback",
    "UniformPeriodicSuffixFallback",
    "materialize_affine_winner_segments",
]


@dataclass(frozen=True)
class PeriodicIntervalStats:
    rows_solved: int
    excluded_runs: int
    predecessor_probes: int
    contiguous_jumps: int
    emitted_segments: int
    creation_intervals_expanded: int = 0


@dataclass(frozen=True)
class DirectPeriodicEventFamilies:
    """Compressed exact events built without materializing changed pairs."""

    families: tuple[PeriodicEventFamily, ...]
    event_count: int
    event_cells: int


def _minimum_period(values: list[int]) -> int:
    if not values:
        return 0
    prefix = [0] * len(values)
    for index in range(1, len(values)):
        matched = prefix[index - 1]
        while matched and values[index] != values[matched]:
            matched = prefix[matched - 1]
        if values[index] == values[matched]:
            matched += 1
        prefix[index] = matched
    return len(values) - prefix[-1]


def _family_event_cells(family: PeriodicEventFamily) -> int:
    first_width = family.first_stop - family.first_start
    width_step = family.delta_stop - family.delta_start
    return (
        family.count * first_width
        + width_step * family.count * (family.count - 1) // 2
    )


class ShiftPeriodicEventFamilyBuilder:
    """Build exact event families directly from shifted periodic phases."""

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        period: int,
    ) -> None:
        _validate_codes(query_codes, key_codes)
        key = [int(value) for value in key_codes.tolist()]
        if not 1 <= period <= len(key):
            raise ValueError("period must fit the sequence")
        if query_codes.numel() > 1 and not bool(
            (query_codes[1:] == key_codes[:-1]).all()
        ):
            raise ValueError("direct periodic families require shifted Q/K")
        if _minimum_period(key) != period:
            raise ValueError("period must be the exact finite minimum")
        self.sequence_length = len(key)
        self.period = period
        self.motif = tuple(key[:period])

    def _context_matches(
        self,
        query_phase: int,
        key_phase: int,
        direction: int,
    ) -> int:
        for matched in range(self.period):
            offset = matched + 1
            if (
                self.motif[(query_phase + direction * offset) % self.period]
                != self.motif[(key_phase + direction * offset) % self.period]
            ):
                return matched
        raise RuntimeError("distinct phases contradict the minimum period")

    @staticmethod
    def _append_family(
        families: list[PeriodicEventFamily],
        *,
        creates_match: bool,
        count: int,
        query_position: int,
        key_position: int,
        left_matches: int,
        right_matches: int,
        delta_query: int = 0,
        delta_key: int = 0,
        delta_left: int = 0,
        delta_right: int = 0,
    ) -> None:
        if count <= 0:
            return
        families.append(
            PeriodicEventFamily(
                creates_match=creates_match,
                count=count,
                first_start=query_position,
                first_stop=query_position + right_matches + 1,
                first_long_normalized_length=(
                    left_matches + 1 - query_position
                ),
                first_short_normalized_length=-query_position,
                first_route_offset=(
                    key_position + 1 - query_position
                ),
                delta_start=delta_query,
                delta_stop=delta_query + delta_right,
                delta_long_normalized_length=delta_left - delta_query,
                delta_short_normalized_length=-delta_query,
                delta_route_offset=delta_key - delta_query,
            )
        )

    def _query_families(self, flip: BitFlip) -> list[PeriodicEventFamily]:
        position = flip.position
        center_phase = (position - 1) % self.period
        center_code = self.motif[center_phase]
        mask = 1 << flip.bit
        families: list[PeriodicEventFamily] = []
        for key_phase, key_code in enumerate(self.motif):
            difference = center_code ^ key_code
            if difference not in (0, mask) or key_phase >= position:
                continue
            count = (position - 1 - key_phase) // self.period + 1
            creates_match = difference == mask
            right_matches = self.sequence_length - position - 1
            if key_phase == center_phase:
                self._append_family(
                    families,
                    creates_match=creates_match,
                    count=count,
                    query_position=position,
                    key_position=key_phase,
                    left_matches=key_phase,
                    right_matches=right_matches,
                    delta_key=self.period,
                    delta_left=self.period,
                )
                continue

            left_limit = self._context_matches(
                center_phase, key_phase, -1
            )
            right_limit = self._context_matches(
                center_phase, key_phase, 1
            )
            right_matches = min(right_matches, right_limit)
            first_key = key_phase
            if first_key < left_limit:
                self._append_family(
                    families,
                    creates_match=creates_match,
                    count=1,
                    query_position=position,
                    key_position=first_key,
                    left_matches=first_key,
                    right_matches=right_matches,
                )
                first_key += self.period
                count -= 1
            self._append_family(
                families,
                creates_match=creates_match,
                count=count,
                query_position=position,
                key_position=first_key,
                left_matches=left_limit,
                right_matches=right_matches,
                delta_key=self.period,
            )
        return families

    def _key_families(self, flip: BitFlip) -> list[PeriodicEventFamily]:
        position = flip.position
        key_phase = position % self.period
        key_code = self.motif[key_phase]
        mask = 1 << flip.bit
        families: list[PeriodicEventFamily] = []
        for query_phase, query_code in enumerate(self.motif):
            difference = query_code ^ key_code
            if difference not in (0, mask):
                continue
            first_mapped_key = position + (
                (query_phase - position) % self.period
            )
            first_query = first_mapped_key + 1
            if first_query >= self.sequence_length:
                continue
            count = (
                (self.sequence_length - 1 - first_query) // self.period + 1
            )
            creates_match = difference == mask
            if query_phase == key_phase:
                self._append_family(
                    families,
                    creates_match=creates_match,
                    count=count,
                    query_position=first_query,
                    key_position=position,
                    left_matches=position,
                    right_matches=self.sequence_length - first_query - 1,
                    delta_query=self.period,
                    delta_right=-self.period,
                )
                continue

            left_limit = self._context_matches(
                query_phase, key_phase, -1
            )
            right_limit = self._context_matches(
                query_phase, key_phase, 1
            )
            left_matches = min(position, left_limit)
            last_query = first_query + (count - 1) * self.period
            truncate_last = (
                self.sequence_length - last_query - 1 < right_limit
            )
            regular_count = count - int(truncate_last)
            self._append_family(
                families,
                creates_match=creates_match,
                count=regular_count,
                query_position=first_query,
                key_position=position,
                left_matches=left_matches,
                right_matches=right_limit,
                delta_query=self.period,
            )
            if truncate_last:
                self._append_family(
                    families,
                    creates_match=creates_match,
                    count=1,
                    query_position=last_query,
                    key_position=position,
                    left_matches=left_matches,
                    right_matches=self.sequence_length - last_query - 1,
                )
        return families

    def build(self, flip: BitFlip) -> DirectPeriodicEventFamilies:
        if not 0 <= flip.bit < 8:
            raise ValueError("flip bit is outside the packed code")
        if flip.source == "query":
            if not 1 <= flip.position < self.sequence_length:
                raise ValueError("query flip is not semantically active")
            families = self._query_families(flip)
        else:
            if not 0 <= flip.position < self.sequence_length - 1:
                raise ValueError("key flip is not semantically active")
            families = self._key_families(flip)
        return DirectPeriodicEventFamilies(
            families=tuple(families),
            event_count=sum(family.count for family in families),
            event_cells=sum(_family_event_cells(family) for family in families),
        )


@dataclass(frozen=True)
class AffineWinnerSegment:
    """A winner whose normalized priority changes every fixed row block."""

    start: int
    stop: int
    first_priority: Priority | None
    priority_step: Priority = (0, 0)
    rows_per_step: int = 1

    def __post_init__(self) -> None:
        if self.start < 0 or self.start >= self.stop:
            raise ValueError("affine winner segment must be nonempty")
        if self.first_priority is None and self.priority_step != (0, 0):
            raise ValueError("a null winner cannot have a priority step")
        if self.rows_per_step < 1:
            raise ValueError("rows_per_step must be positive")

    def priority_at(self, query_index: int) -> Priority | None:
        if not self.start <= query_index < self.stop:
            raise IndexError("query row is outside the affine winner segment")
        if self.first_priority is None:
            return None
        offset = (query_index - self.start) // self.rows_per_step
        return (
            self.first_priority[0] + self.priority_step[0] * offset,
            self.first_priority[1] + self.priority_step[1] * offset,
        )


def materialize_affine_winner_segments(
    sequence_length: int,
    segments: tuple[WinnerSegment | AffineWinnerSegment, ...],
) -> tuple[Tensor, Tensor]:
    routes = torch.zeros(sequence_length, dtype=torch.int64)
    lengths = torch.zeros(sequence_length, dtype=torch.int64)
    covered = 0
    for segment in segments:
        if segment.start != covered or not segment.start < segment.stop:
            raise RuntimeError("winner segments must be contiguous and nonempty")
        covered = segment.stop
        if isinstance(segment, WinnerSegment):
            first_priority = segment.priority
            priority_step = (0, 0)
            rows_per_step = 1
        else:
            first_priority = segment.first_priority
            priority_step = segment.priority_step
            rows_per_step = segment.rows_per_step
        if first_priority is None:
            continue
        rows = torch.arange(segment.start, segment.stop, dtype=torch.int64)
        relative = (rows - segment.start) // rows_per_step
        normalized_lengths = first_priority[0] + priority_step[0] * relative
        route_offsets = first_priority[1] + priority_step[1] * relative
        candidate_lengths = rows + normalized_lengths
        candidate_routes = rows + route_offsets
        if bool((candidate_lengths <= 0).any()) or bool((candidate_routes <= 0).any()):
            raise RuntimeError("an affine winner produced a null candidate")
        lengths[segment.start : segment.stop] = candidate_lengths
        routes[segment.start : segment.stop] = candidate_routes
    if covered != sequence_length:
        raise RuntimeError("winner segments do not cover the sequence")
    return routes, lengths


def _merge_segments(segments: list[WinnerSegment]) -> tuple[WinnerSegment, ...]:
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


def _maximum(
    left: Priority | None,
    right: Priority | None,
) -> Priority | None:
    if left is None:
        return right
    if right is None:
        return left
    return max(left, right)


def _creation_envelope(
    start: int,
    stop: int,
    families: tuple[PeriodicEventFamily, ...],
    *,
    minimum_priority: Priority | None = None,
) -> tuple[tuple[WinnerSegment, ...], int]:
    """Return the exact maximum create priority over every row interval."""

    if start == stop:
        return (), 0
    if not 0 <= start < stop:
        raise ValueError("creation envelope interval must be nonempty")
    if stop == start + 1:
        best: Priority | None = None
        for family in families:
            if not family.creates_match:
                continue
            index_range = family.add_cover_range(start, stop)
            priority = family.maximum_added_priority(index_range)
            if priority is not None:
                best = _maximum(best, priority)
        if minimum_priority is not None and best is not None:
            if best <= minimum_priority:
                best = None
        return (WinnerSegment(start, stop, best),), 0

    intervals: list[tuple[int, int, Priority]] = []
    for family in families:
        if not family.creates_match:
            continue
        index_range = family.add_intersect_range(start, stop)
        maximum = family.maximum_added_priority(index_range)
        if maximum is None or (
            minimum_priority is not None and maximum <= minimum_priority
        ):
            continue
        for index in range(*index_range):
            interval_start = max(family.start(index), start)
            interval_stop = min(family.stop(index), stop)
            priority = family.added_priority(index)
            if (
                interval_start < interval_stop
                and (
                    minimum_priority is None
                    or priority > minimum_priority
                )
            ):
                intervals.append((interval_start, interval_stop, priority))
    if not intervals:
        return (WinnerSegment(start, stop, None),), 0

    intervals.sort(key=lambda item: item[0])
    boundaries = sorted(
        {start, stop}
        | {interval_start for interval_start, _, _ in intervals}
        | {interval_stop for _, interval_stop, _ in intervals}
    )
    active: list[tuple[int, int, int, int]] = []
    next_interval = 0
    segments: list[WinnerSegment] = []
    for interval_start, interval_stop in zip(boundaries, boundaries[1:]):
        while (
            next_interval < len(intervals)
            and intervals[next_interval][0] <= interval_start
        ):
            _, active_stop, priority = intervals[next_interval]
            heappush(
                active,
                (-priority[0], -priority[1], active_stop, next_interval),
            )
            next_interval += 1
        while active and active[0][2] <= interval_start:
            heappop(active)
        priority = (-active[0][0], -active[0][1]) if active else None
        segments.append(
            WinnerSegment(interval_start, interval_stop, priority)
        )
    return _merge_segments(segments), len(intervals)


def _overlay_envelopes(
    base: tuple[WinnerSegment, ...],
    additions: tuple[WinnerSegment, ...],
) -> tuple[WinnerSegment, ...]:
    """Take the exact lexicographic maximum of two complete envelopes."""

    if not base:
        return additions
    if not additions:
        return base
    segments: list[WinnerSegment] = []
    base_index = 0
    addition_index = 0
    while base_index < len(base) and addition_index < len(additions):
        base_segment = base[base_index]
        addition_segment = additions[addition_index]
        start = max(base_segment.start, addition_segment.start)
        stop = min(base_segment.stop, addition_segment.stop)
        if start < stop:
            segments.append(
                WinnerSegment(
                    start,
                    stop,
                    _maximum(base_segment.priority, addition_segment.priority),
                )
            )
        if base_segment.stop == stop:
            base_index += 1
        if addition_segment.stop == stop:
            addition_index += 1
    merged = _merge_segments(segments)
    if (
        not merged
        or merged[0].start != base[0].start
        or merged[-1].stop != base[-1].stop
    ):
        raise RuntimeError("winner envelopes do not cover the same row range")
    return merged


def _append_base_segments(
    segments: list[WinnerSegment],
    start: int,
    stop: int,
) -> None:
    if start >= stop:
        return
    if start == 0:
        segments.append(WinnerSegment(0, 1, None))
        start = 1
    if start < stop:
        segments.append(WinnerSegment(start, stop, (0, 0)))


class ShiftAlignedQuerySuffixFallback:
    """Exact whole-suffix fallback for a Q edit on a shifted base route.

    If ``Q[1:] == K[:-1]``, route offset zero has base length ``t``.  After
    flipping ``Q[p]``, every unchanged candidate that does not cross ``p`` has
    length at most ``t - p``.  The shortened offset-zero route attains exactly
    that length and wins its tie by using the latest legal route.  A candidate
    longer than ``t - p`` must cross the edited equality and is therefore an
    explicit create event.  The exact winner is consequently the maximum event
    addition, with no suffix-index query.
    """

    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    @classmethod
    def build_if_supported(
        cls,
        query_codes: Tensor,
        key_codes: Tensor,
    ) -> "ShiftAlignedQuerySuffixFallback | None":
        _validate_codes(query_codes, key_codes)
        if query_codes.numel() <= 1 or bool(
            (query_codes[1:] == key_codes[:-1]).all()
        ):
            return cls(query_codes.numel())
        return None

    def solve_query_flip(
        self,
        position: int,
        families: tuple[PeriodicEventFamily, ...],
    ) -> tuple[tuple[WinnerSegment, ...], PeriodicIntervalStats]:
        if not 1 <= position < self.sequence_length:
            raise ValueError("query flip is outside the shifted suffix language")
        segments: list[WinnerSegment] = []
        _append_base_segments(segments, 0, position)
        row_additions, row_expanded = _creation_envelope(
            position, position + 1, families
        )
        segments.extend(row_additions)
        suffix_expanded = 0
        if position + 1 < self.sequence_length:
            baseline = (
                WinnerSegment(
                    position + 1, self.sequence_length, (-position, 0)
                ),
            )
            additions, suffix_expanded = _creation_envelope(
                position + 1,
                self.sequence_length,
                families,
                minimum_priority=(-position, 0),
            )
            segments.extend(_overlay_envelopes(baseline, additions))
        merged = _merge_segments(segments)
        return merged, PeriodicIntervalStats(
            rows_solved=self.sequence_length - position,
            excluded_runs=0,
            predecessor_probes=0,
            contiguous_jumps=0,
            emitted_segments=len(merged),
            creation_intervals_expanded=row_expanded + suffix_expanded,
        )


class ShiftAlignedPeriodicKeyFallback:
    """Exact key-edit fallback for a shifted, finite periodic language.

    For a key edit at ``j``, routes above ``j`` either use an explicit create
    event or are bounded by the shortened offset-zero route.  Routes at most
    ``j`` avoid the edit.  If K has exact period ``p`` and ``j >= 2p - 1``, the
    latest route no greater than ``j`` with the query row's phase has full
    length equal to that route and is at least ``p`` long.  Every other phase
    mismatches within one period and therefore cannot beat it.
    """

    def __init__(self, sequence_length: int, period: int) -> None:
        self.sequence_length = sequence_length
        self.period = period

    @staticmethod
    def _minimum_period(values: list[int]) -> int:
        return _minimum_period(values)

    @classmethod
    def build_if_supported(
        cls,
        query_codes: Tensor,
        key_codes: Tensor,
    ) -> "ShiftAlignedPeriodicKeyFallback | None":
        _validate_codes(query_codes, key_codes)
        sequence_length = query_codes.numel()
        if sequence_length <= 1 or not bool(
            (query_codes[1:] == key_codes[:-1]).all()
        ):
            return None
        period = cls._minimum_period(key_codes.tolist())
        if 2 * period - 1 >= sequence_length - 1:
            return None
        return cls(sequence_length, period)

    def supports_position(self, position: int) -> bool:
        return 2 * self.period - 1 <= position < self.sequence_length - 1

    def solve_key_flip(
        self,
        position: int,
        families: tuple[PeriodicEventFamily, ...],
    ) -> tuple[
        tuple[WinnerSegment | AffineWinnerSegment, ...],
        PeriodicIntervalStats,
    ]:
        if not self.supports_position(position):
            raise ValueError("key position lacks a complete periodic certificate")
        affected_start = position + 1
        prefix: list[WinnerSegment] = []
        _append_base_segments(prefix, 0, affected_start)
        baseline: list[WinnerSegment | AffineWinnerSegment] = []
        transition_block = (position + self.period) // self.period
        transition = affected_start + (transition_block - 1) * self.period
        periodic_stop = min(transition, self.sequence_length)
        if affected_start < periodic_stop:
            baseline.append(
                AffineWinnerSegment(
                    affected_start,
                    periodic_stop,
                    (-self.period, -self.period),
                    (-self.period, -self.period),
                    self.period,
                )
            )
        if transition < self.sequence_length:
            baseline.append(
                WinnerSegment(
                    transition,
                    self.sequence_length,
                    (-affected_start, 0),
                )
            )
        baseline_floor = min(
            (
                segment.priority
                if isinstance(segment, WinnerSegment)
                else min(
                    segment.priority_at(segment.start),
                    segment.priority_at(segment.stop - 1),
                )
            )
            for segment in baseline
        )
        additions, creation_intervals_expanded = _creation_envelope(
            affected_start,
            self.sequence_length,
            families,
            minimum_priority=baseline_floor,
        )
        if all(segment.priority is None for segment in additions):
            merged = tuple(prefix) + tuple(baseline)
        else:
            expanded_baseline: list[WinnerSegment] = []
            for segment in baseline:
                if isinstance(segment, WinnerSegment):
                    expanded_baseline.append(segment)
                    continue
                block_start = segment.start
                while block_start < segment.stop:
                    block_stop = min(
                        block_start + segment.rows_per_step, segment.stop
                    )
                    expanded_baseline.append(
                        WinnerSegment(
                            block_start,
                            block_stop,
                            segment.priority_at(block_start),
                        )
                    )
                    block_start = block_stop
            merged = _merge_segments(
                prefix
                + list(
                    _overlay_envelopes(
                        _merge_segments(expanded_baseline), additions
                    )
                )
            )
        return merged, PeriodicIntervalStats(
            rows_solved=self.sequence_length - affected_start,
            excluded_runs=0,
            predecessor_probes=0,
            contiguous_jumps=0,
            emitted_segments=len(merged),
            creation_intervals_expanded=creation_intervals_expanded,
        )


class UniformPeriodicSuffixFallback:
    """Solve complete suffix replacement when Q and K are one equal symbol.

    For row ``t`` and route ``r``, the base suffix length is exactly ``r``.
    Hence every base priority is ``(o, o)`` for ``o = r - t`` and the valid
    offsets are the integer interval ``[1 - t, 0]``.  Periodic break families
    remove arithmetic subsets of that interval, so the best unchanged route is
    an exact arithmetic predecessor query rather than an LCP-tree query.
    """

    def __init__(self, sequence_length: int) -> None:
        self.sequence_length = sequence_length

    @classmethod
    def build_if_supported(
        cls,
        query_codes: Tensor,
        key_codes: Tensor,
    ) -> "UniformPeriodicSuffixFallback | None":
        _validate_codes(query_codes, key_codes)
        if query_codes.numel() == 0:
            return cls(0)
        query = query_codes.tolist()
        key = key_codes.tolist()
        symbol = int(query[0])
        if all(int(value) == symbol for value in query) and all(
            int(value) == symbol for value in key
        ):
            return cls(query_codes.numel())
        return None

    @staticmethod
    def _excluded_offset_runs(
        query_index: int,
        families: tuple[PeriodicEventFamily, ...],
    ) -> tuple[ArithmeticRouteRun, ...]:
        runs = []
        for family in families:
            if family.creates_match:
                continue
            first_phase, stop_phase = family.event_cover_range(
                query_index, query_index + 1
            )
            if first_phase >= stop_phase:
                continue
            first_offset = family.old_priority(first_phase)[1]
            last_offset = family.old_priority(stop_phase - 1)[1]
            step = abs(family.delta_route_offset)
            runs.append(
                ArithmeticRouteRun(
                    min(first_offset, last_offset),
                    max(first_offset, last_offset),
                    step if step else 1,
                )
            )
        return tuple(runs)

    @staticmethod
    def _latest_offset(
        minimum: int,
        runs: tuple[ArithmeticRouteRun, ...],
    ) -> tuple[int | None, int, int]:
        candidate = 0
        probes = 0
        jumps = 0
        while candidate >= minimum:
            probes += 1
            containing = [run for run in runs if run.contains(candidate)]
            if not containing:
                return candidate, probes, jumps
            contiguous = [run for run in containing if run.step == 1]
            if contiguous:
                candidate = min(run.first for run in contiguous) - 1
                jumps += 1
            else:
                candidate -= 1
        return None, probes, jumps

    @staticmethod
    def _best_addition(
        query_index: int,
        families: tuple[PeriodicEventFamily, ...],
    ) -> Priority | None:
        best = None
        for family in families:
            phase_range = family.add_cover_range(
                query_index, query_index + 1
            )
            best = _maximum(
                best, family.maximum_added_priority(phase_range)
            )
        return best

    def solve(
        self,
        start: int,
        stop: int,
        families: tuple[PeriodicEventFamily, ...],
    ) -> tuple[tuple[WinnerSegment, ...], PeriodicIntervalStats]:
        if not 0 <= start <= stop <= self.sequence_length:
            raise ValueError("periodic fallback interval is out of range")
        segments = []
        if families:
            affected_start = max(
                start,
                min(
                    min(family.start(0), family.start(family.count - 1))
                    for family in families
                ),
            )
            affected_stop = min(
                stop,
                max(
                    max(family.stop(0), family.stop(family.count - 1))
                    for family in families
                ),
            )
        else:
            affected_start = affected_stop = start
        if affected_start >= affected_stop:
            affected_start = affected_stop = start

        _append_base_segments(segments, start, affected_start)
        excluded_run_count = 0
        probes = 0
        jumps = 0
        for query_index in range(affected_start, affected_stop):
            if query_index == 0:
                base_priority = None
            else:
                runs = self._excluded_offset_runs(query_index, families)
                excluded_run_count += len(runs)
                offset, local_probes, local_jumps = self._latest_offset(
                    1 - query_index, runs
                )
                probes += local_probes
                jumps += local_jumps
                base_priority = (offset, offset) if offset is not None else None
            priority = _maximum(
                base_priority,
                self._best_addition(query_index, families),
            )
            segments.append(
                WinnerSegment(query_index, query_index + 1, priority)
            )
        _append_base_segments(segments, affected_stop, stop)
        merged = _merge_segments(segments)
        return merged, PeriodicIntervalStats(
            rows_solved=max(affected_stop - affected_start, 0),
            excluded_runs=excluded_run_count,
            predecessor_probes=probes,
            contiguous_jumps=jumps,
            emitted_segments=len(merged),
        )

    def solve_key_flip(
        self,
        position: int,
        families: tuple[PeriodicEventFamily, ...],
    ) -> tuple[
        tuple[WinnerSegment | AffineWinnerSegment, ...],
        PeriodicIntervalStats,
    ]:
        if not 0 <= position < self.sequence_length - 1:
            raise ValueError("key flip is outside the uniform suffix language")
        if any(family.creates_match for family in families):
            return self.solve(0, self.sequence_length, families)
        affected_start = position + 1
        segments: list[WinnerSegment | AffineWinnerSegment] = []
        _append_base_segments(segments, 0, affected_start)
        if position == 0:
            if affected_start < min(affected_start + 1, self.sequence_length):
                segments.append(
                    WinnerSegment(affected_start, affected_start + 1, None)
                )
            prefix_stop = affected_start + 1
        else:
            prefix_stop = min(2 * position + 1, self.sequence_length)
            if affected_start < prefix_stop:
                segments.append(
                    AffineWinnerSegment(
                        affected_start,
                        prefix_stop,
                        (-1, -1),
                        (-1, -1),
                    )
                )
        main_start = max(affected_start + 1, 2 * position + 1)
        main_start = min(main_start, self.sequence_length)
        if prefix_stop < main_start:
            segments.append(WinnerSegment(prefix_stop, main_start, None))
        if main_start < self.sequence_length:
            segments.append(
                WinnerSegment(
                    main_start,
                    self.sequence_length,
                    (-(position + 1), 0),
                )
            )
        return tuple(segments), PeriodicIntervalStats(
            rows_solved=self.sequence_length - affected_start,
            excluded_runs=0,
            predecessor_probes=0,
            contiguous_jumps=0,
            emitted_segments=len(segments),
        )
