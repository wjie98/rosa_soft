"""Exact monotone-LCE decompositions for filtered-bitflip occurrences.

This module studies arbitrary occurrence lists. The frozen v1 solver uses its
query-create cells only under the shifted-query proof; the general and
key-side cell builders remain exact research references.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from typing import Literal, Sequence

from torch import Tensor

from benchmarks.filtered_bitflip import (
    BitFlip,
    CodeOccurrenceIndex,
    InfluenceEvent,
    _code_list,
    _validate_codes,
)
from benchmarks.filtered_bitflip_indexes import _SuffixArrayPair


__all__ = [
    "InfluenceLceCell",
    "InfluenceLceCellBatch",
    "LceRankRun",
    "MonotoneLceRunIndex",
    "OccurrenceLceRun",
    "SemanticLceGeometry",
    "build_influence_lce_cells",
    "build_query_create_lce_cells",
]


RankSide = Literal["lower", "upper"]
ContextSide = Literal["left", "right"]
SequenceSide = Literal["query", "key"]


@dataclass(frozen=True)
class LceRankRun:
    """One maximal suffix-rank interval with constant LCE to a fixed rank."""

    side: RankSide
    rank_start: int
    rank_stop: int
    lce: int

    def __post_init__(self) -> None:
        if self.side not in {"lower", "upper"}:
            raise ValueError("rank side must be 'lower' or 'upper'")
        if not 0 <= self.rank_start < self.rank_stop:
            raise ValueError("LCE rank run must be nonempty")
        if self.lce < 0:
            raise ValueError("LCE must be nonnegative")


class MonotoneLceRunIndex:
    """Enumerate a fixed suffix's LCE plateaus in output-linear time.

    For a target suffix at rank ``r``, an LCE to rank ``i < r`` is the
    minimum of ``lcp[i + 1:r + 1]``.  Shrinking that interval while moving
    toward ``r`` cannot decrease its minimum.  The upper-rank arm is
    symmetric.  Previous/next-strictly-smaller links skip a complete constant
    plateau at a time.
    """

    def __init__(self, pair: _SuffixArrayPair) -> None:
        self.pair = pair
        lcp = pair.lcp
        size = len(lcp)

        previous_smaller = [0] * size
        stack: list[int] = []
        for index in range(size):
            while stack and lcp[stack[-1]] >= lcp[index]:
                stack.pop()
            previous_smaller[index] = stack[-1] if stack else 0
            stack.append(index)

        next_smaller = [size] * size
        stack.clear()
        for index in range(size - 1, 0, -1):
            while stack and lcp[stack[-1]] >= lcp[index]:
                stack.pop()
            next_smaller[index] = stack[-1] if stack else size
            stack.append(index)

        self.previous_smaller = tuple(previous_smaller)
        self.next_smaller = tuple(next_smaller)

    @property
    def rank_count(self) -> int:
        return len(self.pair.suffix_array)

    def lce_by_rank(self, first_rank: int, second_rank: int) -> int:
        """Return the unbounded generalized-string LCE for two ranks."""

        if not 0 <= first_rank < self.rank_count:
            raise IndexError("first suffix rank is out of range")
        if not 0 <= second_rank < self.rank_count:
            raise IndexError("second suffix rank is out of range")
        if first_rank == second_rank:
            position = self.pair.suffix_array[first_rank]
            return self.pair.text_length - position
        start = min(first_rank, second_rank) + 1
        stop = max(first_rank, second_rank) + 1
        return self.pair.rmq.minimum(start, stop)

    def runs(self, target_rank: int) -> tuple[LceRankRun, ...]:
        """Return maximal constant-LCE runs covering every other rank."""

        if not 0 <= target_rank < self.rank_count:
            raise IndexError("target suffix rank is out of range")
        lcp = self.pair.lcp
        runs = []

        edge = target_rank
        while edge > 0:
            depth = lcp[edge]
            boundary = 0 if depth == 0 else self.previous_smaller[edge]
            runs.append(LceRankRun("lower", boundary, edge, depth))
            edge = boundary

        edge = target_rank + 1
        while edge < self.rank_count:
            depth = lcp[edge]
            boundary = (
                self.rank_count
                if depth == 0
                else self.next_smaller[edge]
            )
            runs.append(LceRankRun("upper", edge, boundary, depth))
            edge = boundary

        return tuple(sorted(runs, key=lambda run: run.rank_start))


class SemanticLceGeometry:
    """Boundary-exact forward/reverse generalized suffix geometry.

    ROSA never uses ``K[T - 1]`` as a route symbol.  The forward pair therefore
    indexes ``K[:-1]`` so its sentinel, rather than a candidate-dependent cap,
    ends every right LCE.  This is necessary for suffix-rank monotonicity to be
    a theorem rather than an empirical property of a capped LCP value.
    """

    def __init__(
        self,
        query_codes: Tensor,
        key_codes: Tensor,
        *,
        forward_pair: _SuffixArrayPair | None = None,
        reverse_pair: _SuffixArrayPair | None = None,
    ) -> None:
        _validate_codes(query_codes, key_codes)
        self.query = tuple(_code_list(query_codes))
        self.key = tuple(_code_list(key_codes))
        self.sequence_length = len(self.query)
        self.forward_pair = forward_pair or _SuffixArrayPair(
            self.query, self.key[:-1]
        )
        self.reverse_pair = reverse_pair or _SuffixArrayPair(
            self.query[::-1], self.key[::-1]
        )
        expected_forward_length = (
            len(self.query) + len(self.key[:-1]) + 2
        )
        expected_reverse_length = len(self.query) + len(self.key) + 2
        if (
            self.forward_pair.query_length != self.sequence_length
            or self.forward_pair.text_length != expected_forward_length
        ):
            raise ValueError(
                "forward pair does not use the semantic key boundary"
            )
        if (
            self.reverse_pair.query_length != self.sequence_length
            or self.reverse_pair.text_length != expected_reverse_length
        ):
            raise ValueError("reverse pair has incompatible sequence lengths")
        self.forward_runs = MonotoneLceRunIndex(self.forward_pair)
        self.reverse_runs = MonotoneLceRunIndex(self.reverse_pair)

    @staticmethod
    def _rank(pair: _SuffixArrayPair, side: SequenceSide, position: int) -> int:
        if side == "query":
            return pair.inverse[position]
        return pair.inverse[pair.query_length + 1 + position]

    def context_rank(
        self,
        context: ContextSide,
        side: SequenceSide,
        center_position: int,
    ) -> int | None:
        """Map the context adjacent to one edited center to a suffix rank."""

        if not 0 <= center_position < self.sequence_length:
            raise IndexError("center position is out of range")
        if context == "right":
            suffix_position = center_position + 1
            semantic_length = (
                self.sequence_length
                if side == "query"
                else self.sequence_length - 1
            )
            if suffix_position >= semantic_length:
                return None
            return self._rank(self.forward_pair, side, suffix_position)
        if context != "left":
            raise ValueError("context must be 'left' or 'right'")
        if center_position == 0:
            return None
        suffix_position = self.sequence_length - center_position
        return self._rank(self.reverse_pair, side, suffix_position)

    def context_lce(
        self,
        context: ContextSide,
        query_position: int,
        key_position: int,
    ) -> int:
        query_rank = self.context_rank(context, "query", query_position)
        key_rank = self.context_rank(context, "key", key_position)
        if query_rank is None or key_rank is None:
            return 0
        index = self.forward_runs if context == "right" else self.reverse_runs
        return index.lce_by_rank(query_rank, key_rank)

    def left_matches(self, query_position: int, key_position: int) -> int:
        return self.context_lce("left", query_position, key_position)

    def right_matches(self, query_position: int, key_position: int) -> int:
        return self.context_lce("right", query_position, key_position)


@dataclass(frozen=True)
class OccurrenceLceRun:
    """A nonempty code-filtered rank interval carrying one constant LCE."""

    context: ContextSide
    side: RankSide | Literal["boundary"]
    rank_start: int
    rank_stop: int
    lce: int
    positions: tuple[int, ...]


@dataclass(frozen=True)
class InfluenceLceCell:
    """Occurrences with identical exact left/right influence geometry."""

    flip: BitFlip
    creates_match: bool
    left_matches: int
    right_matches: int
    varying_positions: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.varying_positions:
            raise ValueError("an influence LCE cell must be nonempty")
        if any(
            left >= right
            for left, right in zip(
                self.varying_positions, self.varying_positions[1:]
            )
        ):
            raise ValueError("varying positions must be strictly increasing")

    def _event(self, varying_position: int) -> InfluenceEvent:
        if self.flip.source == "query":
            query_position = self.flip.position
            key_position = varying_position
        else:
            query_position = varying_position
            key_position = self.flip.position
        return InfluenceEvent(
            flip=self.flip,
            query_position=query_position,
            key_position=key_position,
            left_matches=self.left_matches,
            right_matches=self.right_matches,
            creates_match=self.creates_match,
        )

    def event_at(self, varying_position: int) -> InfluenceEvent:
        offset = bisect_left(self.varying_positions, varying_position)
        if (
            offset == len(self.varying_positions)
            or self.varying_positions[offset] != varying_position
        ):
            raise ValueError("position is not a member of this LCE cell")
        return self._event(varying_position)

    @property
    def dominant_query_event(self) -> InfluenceEvent:
        """Return the latest-route representative for a query-side cell."""

        if self.flip.source != "query":
            raise ValueError("key-side cells do not have one global dominator")
        return self.event_at(self.varying_positions[-1])

    def dominant_event_at(self, query_index: int) -> InfluenceEvent | None:
        """Return this cell's exact local winner at one output row.

        Query-side events share one interval and the latest key position wins.
        Key-side events have shifted equal-width intervals; the earliest still
        active query position wins because both normalized priority fields
        decrease with that position.
        """

        if self.flip.source == "query":
            event = self.dominant_query_event
            return event if event.start <= query_index < event.stop else None
        first_active = bisect_left(
            self.varying_positions,
            query_index - self.right_matches,
        )
        if first_active == len(self.varying_positions):
            return None
        query_position = self.varying_positions[first_active]
        if query_position > query_index:
            return None
        return self._event(query_position)


@dataclass(frozen=True)
class InfluenceLceCellBatch:
    """Exact run/cell decomposition for one semantic bit flip."""

    flip: BitFlip
    cells: tuple[InfluenceLceCell, ...]
    left_runs: tuple[OccurrenceLceRun, ...]
    right_runs: tuple[OccurrenceLceRun, ...]
    occurrence_count: int

    def expand_events(self) -> tuple[InfluenceEvent, ...]:
        events = [
            cell._event(position)
            for cell in self.cells
            for position in cell.varying_positions
        ]
        return tuple(
            sorted(
                events,
                key=lambda event: (
                    event.query_position,
                    event.key_position,
                ),
            )
        )

    def dominant_query_create_events(self) -> tuple[InfluenceEvent, ...]:
        """Collapse create cells when only their candidate envelope is needed."""

        if self.flip.source != "query":
            raise ValueError("dominant create events require a query flip")
        return tuple(
            cell.dominant_query_event
            for cell in self.cells
            if cell.creates_match
        )


def _candidate_positions(
    occurrence_index: CodeOccurrenceIndex,
    flip: BitFlip,
    code: int,
) -> tuple[int, ...]:
    if flip.source == "query":
        positions = occurrence_index.key_positions[code]
        return positions[:bisect_left(positions, flip.position)]
    positions = occurrence_index.query_positions[code]
    return positions[bisect_right(positions, flip.position) :]


def _occurrence_runs(
    geometry: SemanticLceGeometry,
    *,
    context: ContextSide,
    fixed_side: SequenceSide,
    candidate_positions: tuple[int, ...],
    target_rank: int | None,
    structural_runs: tuple[LceRankRun, ...],
) -> tuple[OccurrenceLceRun, ...]:
    if not candidate_positions:
        return ()
    candidate_side: SequenceSide = "key" if fixed_side == "query" else "query"
    if target_rank is None:
        return (
            OccurrenceLceRun(
                context, "boundary", -1, -1, 0, candidate_positions
            ),
        )

    starts = tuple(run.rank_start for run in structural_runs)
    grouped: list[list[int]] = [[] for _ in structural_runs]
    boundary_positions = []
    for position in candidate_positions:
        rank = geometry.context_rank(context, candidate_side, position)
        if rank is None:
            boundary_positions.append(position)
            continue
        run_offset = bisect_right(starts, rank) - 1
        if run_offset < 0:
            raise RuntimeError("candidate rank is outside LCE run coverage")
        run = structural_runs[run_offset]
        if not run.rank_start <= rank < run.rank_stop:
            raise RuntimeError("candidate rank is inside the target-rank gap")
        grouped[run_offset].append(position)

    result = [
        OccurrenceLceRun(
            context,
            run.side,
            run.rank_start,
            run.rank_stop,
            run.lce,
            tuple(positions),
        )
        for run, positions in zip(structural_runs, grouped)
        if positions
    ]
    if boundary_positions:
        result.append(
            OccurrenceLceRun(
                context,
                "boundary",
                -1,
                -1,
                0,
                tuple(boundary_positions),
            )
        )
    return tuple(result)


def _singleton_occurrence_run(
    geometry: SemanticLceGeometry,
    *,
    context: ContextSide,
    fixed_side: SequenceSide,
    fixed_position: int,
    candidate_position: int,
    lce: int,
) -> OccurrenceLceRun:
    candidate_side: SequenceSide = "key" if fixed_side == "query" else "query"
    target_rank = geometry.context_rank(context, fixed_side, fixed_position)
    candidate_rank = geometry.context_rank(
        context, candidate_side, candidate_position
    )
    if target_rank is None or candidate_rank is None:
        return OccurrenceLceRun(
            context, "boundary", -1, -1, lce, (candidate_position,)
        )
    side: RankSide = "lower" if candidate_rank < target_rank else "upper"
    return OccurrenceLceRun(
        context,
        side,
        candidate_rank,
        candidate_rank + 1,
        lce,
        (candidate_position,),
    )


def build_influence_lce_cells(
    query_codes: Tensor,
    key_codes: Tensor,
    flip: BitFlip,
    *,
    geometry: SemanticLceGeometry | None = None,
    occurrence_index: CodeOccurrenceIndex | None = None,
    transition_filter: Literal["both", "create", "break"] = "both",
) -> InfluenceLceCellBatch:
    """Partition one flip's exact occurrences by its two monotone LCE runs."""

    _validate_codes(query_codes, key_codes)
    sequence_length = query_codes.numel()
    if flip.bit >= 8:
        raise ValueError("flip bit is outside the packed code")
    if flip.source == "query" and not 1 <= flip.position < sequence_length:
        raise ValueError("query flip is not semantically active")
    if flip.source == "key" and not 0 <= flip.position < sequence_length - 1:
        raise ValueError("key flip is not semantically active")
    if geometry is None:
        geometry = SemanticLceGeometry(query_codes, key_codes)
    if occurrence_index is None:
        occurrence_index = CodeOccurrenceIndex.build(query_codes, key_codes)
    if transition_filter not in {"both", "create", "break"}:
        raise ValueError("unknown transition filter")

    fixed_codes = (
        occurrence_index.query_codes
        if flip.source == "query"
        else occurrence_index.key_codes
    )
    fixed_code = fixed_codes[flip.position]
    transitions = (
        (fixed_code, False),
        (fixed_code ^ (1 << flip.bit), True),
    )
    if transition_filter != "both":
        keep_create = transition_filter == "create"
        transitions = tuple(
            transition
            for transition in transitions
            if transition[1] == keep_create
        )
    fixed_side: SequenceSide = flip.source
    selected_transitions = tuple(
        (
            creates_match,
            _candidate_positions(occurrence_index, flip, code),
        )
        for code, creates_match in transitions
    )
    occurrence_count = sum(
        len(positions) for _, positions in selected_transitions
    )
    if occurrence_count <= 1:
        if occurrence_count == 0:
            return InfluenceLceCellBatch(flip, (), (), (), 0)
        creates_match, positions = next(
            transition
            for transition in selected_transitions
            if transition[1]
        )
        position = positions[0]
        if flip.source == "query":
            query_position, key_position = flip.position, position
        else:
            query_position, key_position = position, flip.position
        left_matches = geometry.left_matches(query_position, key_position)
        right_matches = geometry.right_matches(query_position, key_position)
        cell = InfluenceLceCell(
            flip,
            creates_match,
            left_matches,
            right_matches,
            positions,
        )
        left_run = _singleton_occurrence_run(
            geometry,
            context="left",
            fixed_side=fixed_side,
            fixed_position=flip.position,
            candidate_position=position,
            lce=left_matches,
        )
        right_run = _singleton_occurrence_run(
            geometry,
            context="right",
            fixed_side=fixed_side,
            fixed_position=flip.position,
            candidate_position=position,
            lce=right_matches,
        )
        return InfluenceLceCellBatch(
            flip, (cell,), (left_run,), (right_run,), 1
        )

    target_ranks = {
        context: geometry.context_rank(
            context, fixed_side, flip.position
        )
        for context in ("left", "right")
    }
    structural_runs = {
        context: (
            ()
            if target_ranks[context] is None
            else (
                geometry.reverse_runs
                if context == "left"
                else geometry.forward_runs
            ).runs(target_ranks[context])
        )
        for context in ("left", "right")
    }

    cells = []
    all_left_runs = []
    all_right_runs = []
    for creates_match, positions in selected_transitions:
        left_runs = _occurrence_runs(
            geometry,
            context="left",
            fixed_side=fixed_side,
            candidate_positions=positions,
            target_rank=target_ranks["left"],
            structural_runs=structural_runs["left"],
        )
        right_runs = _occurrence_runs(
            geometry,
            context="right",
            fixed_side=fixed_side,
            candidate_positions=positions,
            target_rank=target_ranks["right"],
            structural_runs=structural_runs["right"],
        )
        all_left_runs.extend(left_runs)
        all_right_runs.extend(right_runs)

        left_by_position = {
            position: run.lce
            for run in left_runs
            for position in run.positions
        }
        right_by_position = {
            position: run.lce
            for run in right_runs
            for position in run.positions
        }
        grouped: dict[tuple[int, int], list[int]] = {}
        for position in positions:
            cell_key = (
                left_by_position[position],
                right_by_position[position],
            )
            grouped.setdefault(cell_key, []).append(position)
        cells.extend(
            InfluenceLceCell(
                flip,
                creates_match,
                left_matches,
                right_matches,
                tuple(group_positions),
            )
            for (left_matches, right_matches), group_positions in grouped.items()
        )

    cells.sort(
        key=lambda cell: (
            cell.varying_positions[0],
            cell.creates_match,
            cell.left_matches,
            cell.right_matches,
        )
    )
    return InfluenceLceCellBatch(
        flip,
        tuple(cells),
        tuple(all_left_runs),
        tuple(all_right_runs),
        occurrence_count,
    )


def build_query_create_lce_cells(
    query_codes: Tensor,
    key_codes: Tensor,
    flip: BitFlip,
    *,
    geometry: SemanticLceGeometry | None = None,
    occurrence_index: CodeOccurrenceIndex | None = None,
) -> InfluenceLceCellBatch:
    """Build only exact create cells for a query-side bit flip."""

    if flip.source != "query":
        raise ValueError("query create cells require a query flip")
    return build_influence_lce_cells(
        query_codes,
        key_codes,
        flip,
        geometry=geometry,
        occurrence_index=occurrence_index,
        transition_filter="create",
    )
