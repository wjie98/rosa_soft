"""Pre-v1 exact structural dispatcher retained as a timing baseline."""

from __future__ import annotations

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
from benchmarks.filtered_bitflip_certificates import (
    ReplacementCertificateTree,
    materialize_winner_segments,
)
from benchmarks.filtered_bitflip_indexes import DirectMatchIndex
from benchmarks.filtered_bitflip_suffix_nodes import SuffixNodeReplacementIndex


__all__ = ["HybridBitflipResult", "hybrid_filtered_bitflip"]


_MIN_OVERLAY_COMPRESSION = 2


@dataclass(frozen=True)
class HybridBitflipResult:
    flips: tuple[BitFlip, ...]
    base_routes: Tensor
    base_lengths: Tensor
    flipped_routes: Tensor
    flipped_lengths: Tensor
    bit_gradient: Tensor | None
    certificate_flips: int
    suffix_node_flips: int
    empty_flips: int
    periodic_postings: bool
    raw_events: int
    event_cells: int
    estimated_overlay_entries: int
    certificate_tree_nodes_visited: int
    certificate_candidates_probed: int
    suffix_replacement_rows: int
    suffix_nodes_visited: int
    suffix_candidates_probed: int
    suffix_periodic_jumps: int
    final_changed_rows: int
    final_route_change_ranges: int


def hybrid_filtered_bitflip(
    query_codes: Tensor,
    key_codes: Tensor,
    bit_width: int,
    *,
    suffix_backend: str = "python",
    library_path: Path | str | None = None,
    match_index=None,
    value: Tensor | None = None,
    grad_output: Tensor | None = None,
) -> HybridBitflipResult:
    """Dispatch each exact flip from structural compression, never clarity."""

    _validate_codes(query_codes, key_codes)
    base = unlimited_hard_routes_diagonal(query_codes, key_codes)
    certificate_tree = ReplacementCertificateTree.build(query_codes, key_codes)
    suffix_index = SuffixNodeReplacementIndex(
        query_codes,
        key_codes,
        suffix_backend=suffix_backend,
        library_path=library_path,
    )
    use_periodic_postings = (
        suffix_index.route_runs * 2 < suffix_index.route_postings
    )
    posting_backend = "periodic" if use_periodic_postings else "flat"
    if match_index is None:
        match_index = DirectMatchIndex(query_codes, key_codes)
    occurrence = CodeOccurrenceIndex.build(query_codes, key_codes)
    flips = semantic_bit_flips(query_codes.numel(), bit_width)
    route_rows = []
    length_rows = []
    certificate_flips = 0
    suffix_flips = 0
    empty_flips = 0
    raw_events = 0
    event_cells = 0
    overlay_entries = 0
    certificate_nodes = 0
    certificate_probes = 0
    suffix_rows = 0
    suffix_nodes = 0
    suffix_probes = 0
    suffix_jumps = 0
    for flip in flips:
        events = generate_influence_events(
            query_codes,
            key_codes,
            flip,
            match_index=match_index,
            occurrence_index=occurrence,
        )
        raw_events += len(events)
        local_cells = sum(event.stop - event.start for event in events)
        event_cells += local_cells
        forbid_entries, add_entries = certificate_tree.overlay_entry_counts(events)
        local_entries = forbid_entries + add_entries
        overlay_entries += local_entries
        if not events:
            empty_flips += 1
            routes = base.routes.clone()
            lengths = base.lengths.clone()
        elif local_cells >= _MIN_OVERLAY_COMPRESSION * max(local_entries, 1):
            certificate_flips += 1
            segments, stats = certificate_tree.solve(events)
            routes, lengths = materialize_winner_segments(
                query_codes.numel(),
                segments,
            )
            certificate_nodes += stats.tree_nodes_visited
            certificate_probes += stats.base_candidates_probed
        else:
            suffix_flips += 1
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
            routes = base.routes.clone()
            lengths = base.lengths.clone()
            for query_index, row in changed.items():
                suffix_rows += 1
                best_length, best_route, stats = suffix_index.best_excluding(
                    query_index,
                    set(row),
                    backend=posting_backend,
                )
                suffix_nodes += stats.nodes_visited
                suffix_probes += stats.route_candidates_probed
                suffix_jumps += stats.periodic_jumps
                for route, length in row.items():
                    if length > best_length or (
                        length == best_length
                        and length > 0
                        and route > best_route
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
    return HybridBitflipResult(
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
        certificate_flips=certificate_flips,
        suffix_node_flips=suffix_flips,
        empty_flips=empty_flips,
        periodic_postings=use_periodic_postings,
        raw_events=raw_events,
        event_cells=event_cells,
        estimated_overlay_entries=overlay_entries,
        certificate_tree_nodes_visited=certificate_nodes,
        certificate_candidates_probed=certificate_probes,
        suffix_replacement_rows=suffix_rows,
        suffix_nodes_visited=suffix_nodes,
        suffix_candidates_probed=suffix_probes,
        suffix_periodic_jumps=suffix_jumps,
        final_changed_rows=final_changed_rows,
        final_route_change_ranges=final_ranges,
    )
