"""Profile exact monotone-LCE runs on nonperiodic occurrence lists."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from benchmarks.filtered_bitflip import (  # noqa: E402
    CodeOccurrenceIndex,
    generate_influence_events,
    semantic_bit_flips,
)
from benchmarks.filtered_bitflip_indexes import (  # noqa: E402
    SuffixArrayMatchIndex,
)
from benchmarks.filtered_bitflip_monotone import (  # noqa: E402
    SemanticLceGeometry,
    build_influence_lce_cells,
)


def _minimum_period(values: tuple[int, ...]) -> int:
    for period in range(1, len(values) + 1):
        if values[period:] == values[:-period]:
            return period
    raise RuntimeError("every finite sequence has its full length as a period")


def _shifted_query(key: torch.Tensor, first: int = 0) -> torch.Tensor:
    query = torch.empty_like(key)
    query[0] = first
    query[1:] = key[:-1]
    return query


def _thue_morse(sequence_length: int) -> torch.Tensor:
    return torch.tensor(
        [index.bit_count() & 1 for index in range(sequence_length)],
        dtype=torch.uint8,
    )


def _ruler_codes(sequence_length: int) -> torch.Tensor:
    values = []
    for index in range(sequence_length):
        value = index + 1
        trailing_zeros = (value & -value).bit_length() - 1
        values.append(trailing_zeros & 3)
    return torch.tensor(values, dtype=torch.uint8)


def make_cases(
    sequence_length: int, seed: int
) -> tuple[tuple[str, int, torch.Tensor, torch.Tensor], ...]:
    generator = torch.Generator().manual_seed(seed + sequence_length)

    independent_byte_query = torch.randint(
        256,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    independent_byte_key = torch.randint(
        256,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    independent_quad_query = torch.randint(
        4,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    independent_quad_key = torch.randint(
        4,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )
    shifted_random_key = torch.randint(
        256,
        (sequence_length,),
        dtype=torch.uint8,
        generator=generator,
    )

    motif = torch.tensor([0, 3, 1, 6, 2, 5, 7, 4], dtype=torch.uint8)
    motif_noise_key = motif.repeat(
        (sequence_length + len(motif) - 1) // len(motif)
    )[:sequence_length].clone()
    for position in range(5, sequence_length, 11):
        motif_noise_key[position] ^= 1 + ((position // 11) & 3)

    thue_morse_key = _thue_morse(sequence_length)
    if sequence_length:
        thue_morse_key[-1] ^= 1
    ruler_key = _ruler_codes(sequence_length)
    if sequence_length > 3:
        ruler_key[-3] ^= 1

    return (
        (
            "independent_byte",
            8,
            independent_byte_query,
            independent_byte_key,
        ),
        (
            "independent_quaternary",
            2,
            independent_quad_query,
            independent_quad_key,
        ),
        (
            "shifted_random_byte",
            8,
            _shifted_query(shifted_random_key, 17),
            shifted_random_key,
        ),
        (
            "shifted_motif_noise",
            3,
            _shifted_query(motif_noise_key, 7),
            motif_noise_key,
        ),
        (
            "shifted_thue_morse",
            1,
            _shifted_query(thue_morse_key, 0),
            thue_morse_key,
        ),
        (
            "shifted_ruler",
            2,
            _shifted_query(ruler_key, 3),
            ruler_key,
        ),
    )


def _median_ms(function, repeats: int):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - start) * 1e3)
    return result, statistics.median(samples)


def _empty_source_stats() -> dict[str, int]:
    return {
        "events": 0,
        "cells": 0,
        "left_runs": 0,
        "right_runs": 0,
        "create_events": 0,
        "create_cells": 0,
        "break_events": 0,
        "break_cells": 0,
        "singleton_cells": 0,
        "largest_cell": 0,
    }


def profile_case(
    name: str,
    bit_width: int,
    query: torch.Tensor,
    key: torch.Tensor,
    repeats: int,
) -> dict[str, object]:
    sequence_length = query.numel()
    flips = semantic_bit_flips(sequence_length, bit_width)
    occurrences = CodeOccurrenceIndex.build(query, key)

    match_index, match_build_ms = _median_ms(
        lambda: SuffixArrayMatchIndex(query, key), repeats
    )

    def discover_events() -> int:
        return sum(
            len(
                generate_influence_events(
                    query,
                    key,
                    flip,
                    match_index=match_index,
                    occurrence_index=occurrences,
                )
            )
            for flip in flips
        )

    event_count, event_discovery_ms = _median_ms(discover_events, repeats)
    geometry, geometry_build_ms = _median_ms(
        lambda: SemanticLceGeometry(query, key), repeats
    )

    def build_batches():
        return tuple(
            build_influence_lce_cells(
                query,
                key,
                flip,
                geometry=geometry,
                occurrence_index=occurrences,
            )
            for flip in flips
        )

    batches, cell_build_ms = _median_ms(build_batches, repeats)
    sources = {"query": _empty_source_stats(), "key": _empty_source_stats()}
    for batch in batches:
        stats = sources[batch.flip.source]
        stats["events"] += batch.occurrence_count
        stats["cells"] += len(batch.cells)
        stats["left_runs"] += len(batch.left_runs)
        stats["right_runs"] += len(batch.right_runs)
        for cell in batch.cells:
            cell_size = len(cell.varying_positions)
            transition = "create" if cell.creates_match else "break"
            stats[f"{transition}_events"] += cell_size
            stats[f"{transition}_cells"] += 1
            stats["singleton_cells"] += int(cell_size == 1)
            stats["largest_cell"] = max(stats["largest_cell"], cell_size)

    if sum(stats["events"] for stats in sources.values()) != event_count:
        raise RuntimeError("cell decomposition changed the event count")

    for stats in sources.values():
        stats["event_per_cell"] = stats["events"] / max(stats["cells"], 1)
        stats["event_per_left_run"] = stats["events"] / max(
            stats["left_runs"], 1
        )
        stats["event_per_right_run"] = stats["events"] / max(
            stats["right_runs"], 1
        )
        stats["create_representative_ratio"] = stats["create_cells"] / max(
            stats["create_events"], 1
        )
        stats["singleton_cell_fraction"] = stats["singleton_cells"] / max(
            stats["cells"], 1
        )

    key_values = tuple(int(value) for value in key.tolist())
    period = _minimum_period(key_values)
    return {
        "case": name,
        "sequence_length": sequence_length,
        "bit_width": bit_width,
        "key_minimum_finite_period": period,
        "key_has_half_length_period": period <= sequence_length // 2,
        "events": event_count,
        "suffix_match_build_ms": match_build_ms,
        "event_discovery_ms": event_discovery_ms,
        "semantic_geometry_build_ms": geometry_build_ms,
        "cell_build_ms": cell_build_ms,
        "sources": sources,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-lengths", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()

    results = [
        profile_case(name, bit_width, query, key, args.repeats)
        for sequence_length in args.sequence_lengths
        for name, bit_width, query, key in make_cases(
            sequence_length, args.seed
        )
    ]
    payload = {
        "seed": args.seed,
        "repeats": args.repeats,
        "results": results,
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n")


if __name__ == "__main__":
    main()
