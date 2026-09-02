"""Trace the production C++ ROSA runtime once and replay page caches offline.

The benchmark-only extension includes the same suffix-automaton source as the
production CPU runtime. Compile-time access hooks disappear from normal builds;
the study build records stable logical 4 KiB page IDs for state, edge SoA,
transition indexes, the root direct table, the latest-end link-cut tree, and
payload history. Larger page sizes and all cache policies are derived from the
same immutable trace, so policy experiments cannot perturb runtime behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Iterable, Sequence

import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path[:1]:
    sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "benchmarks" / "csrc" / "rosa_runtime_page_trace.cpp"
BUILD_ROOT = ROOT / "build" / "runtime_page_trace"

REGION_NAMES = {
    0: "state",
    1: "edge_symbol",
    2: "edge_target",
    3: "edge_next",
    4: "transition_meta",
    5: "transition_slots",
    6: "root_table",
    7: "latest_end_tree",
    8: "payload",
}
KIND_NAMES = {
    0: "transition",
    1: "query",
    2: "extension",
    3: "latest",
    4: "clone",
    5: "append",
    6: "index",
    7: "materialize",
    8: "payload",
    9: "rebuild",
}
COMPLEXITY_NAMES = (
    "updates",
    "transition_probes",
    "query_suffix_link_steps",
    "extension_suffix_link_steps",
    "latest_end_updates",
    "clones",
    "copied_edges",
    "max_query_suffix_link_steps",
    "max_latest_end_chain",
    "latest_path_updates",
    "latest_tree_rotations",
    "latest_tree_activations",
    "compressed_run_symbols",
    "compressed_materializations",
    "max_compressed_run_length",
    "root_table_lookups",
    "root_table_activations",
    "transition_index_lookups",
    "transition_index_activations",
)
SUMMARY_NAMES = (
    "states",
    "edges",
    "automaton_logical_bytes",
    "canonical_page_touches",
    "rle_records",
)

WRITE_FLAG = 1
APPEND_KIND = 5
DEFAULT_STRATEGIES = (
    "lru",
    "slru",
    "region_lru",
    "structural_pin_lru",
    "append_bypass_lru",
)

_MODULE: ModuleType | None = None


def load_runtime_page_trace(*, verbose: bool = False) -> ModuleType:
    """Build the trace-only CPU extension outside the package build graph."""

    global _MODULE
    if _MODULE is not None:
        return _MODULE
    try:
        import ninja

        os.environ["PATH"] = os.pathsep.join(
            (ninja.BIN_DIR, os.environ.get("PATH", ""))
        )
    except ImportError:
        pass

    from torch.utils import cpp_extension

    BUILD_ROOT.mkdir(parents=True, exist_ok=True)
    _MODULE = cpp_extension.load(
        name="rosa_runtime_page_trace_cpp",
        sources=[str(SOURCE)],
        build_directory=str(BUILD_ROOT),
        extra_cflags=["-O3", "-std=c++17"],
        with_cuda=False,
        verbose=verbose,
    )
    return _MODULE


@dataclass(frozen=True)
class RuntimePageTrace:
    records: torch.Tensor
    routes: torch.Tensor
    output: torch.Tensor
    summary: dict[str, int]
    complexity: dict[str, int]
    canonical_page_bytes: int


def _named_values(names: Sequence[str], values: torch.Tensor) -> dict[str, int]:
    entries = values.tolist()
    if len(entries) != len(names):
        raise RuntimeError("native trace metadata has an unexpected shape")
    return dict(zip(names, map(int, entries)))


def collect_runtime_page_trace(
    query: torch.Tensor,
    key: torch.Tensor,
    payload: torch.Tensor | None = None,
    *,
    qk_bits: int,
    payload_bits: int = 8,
    chunk_size: int | None = None,
    canonical_page_bytes: int = 4096,
    module: ModuleType | None = None,
) -> RuntimePageTrace:
    """Run one exact production-core trace; no cache participates in execution."""

    query = query.to(device="cpu", dtype=torch.uint8).flatten().contiguous()
    key = key.to(device="cpu", dtype=torch.uint8).flatten().contiguous()
    if payload is None:
        payload = torch.arange(query.numel(), dtype=torch.int64).to(torch.uint8)
    payload = payload.to(device="cpu", dtype=torch.uint8).flatten().contiguous()
    if query.numel() != key.numel() or query.numel() != payload.numel():
        raise ValueError("query, key, and payload lengths must match")
    if chunk_size is None:
        chunk_size = max(query.numel(), 1)
    if module is None:
        module = load_runtime_page_trace()
    result = module.trace_runtime(
        query,
        key,
        payload,
        int(qk_bits),
        int(payload_bits),
        int(chunk_size),
        int(canonical_page_bytes),
    )
    records = result["records"]
    if records.dtype != torch.int64 or records.dim() != 2 or records.size(1) != 7:
        raise RuntimeError("native trace records have an unexpected schema")
    return RuntimePageTrace(
        records=records,
        routes=result["routes"],
        output=result["output"],
        summary=_named_values(SUMMARY_NAMES, result["summary"]),
        complexity=_named_values(COMPLEXITY_NAMES, result["complexity"]),
        canonical_page_bytes=int(result["canonical_page_bytes"]),
    )


def validate_trace_against_runtime(
    trace: RuntimePageTrace,
    query: torch.Tensor,
    key: torch.Tensor,
    payload: torch.Tensor,
    *,
    qk_bits: int,
    payload_bits: int,
    chunk_size: int,
) -> None:
    """Check routes and payload successors against the packaged runtime."""

    from rosa_soft import RosaRuntime

    query = query.to(torch.uint8).flatten()
    key = key.to(torch.uint8).flatten()
    payload = payload.to(torch.uint8).flatten()
    outputs = []
    routes = []
    with RosaRuntime(1, 1, qk_bits, payload_bits) as runtime:
        for begin in range(0, query.numel(), chunk_size):
            end = min(query.numel(), begin + chunk_size)
            current_output, current_routes = runtime.update_packed(
                query[begin:end].view(1, -1, 1),
                key[begin:end].view(1, -1, 1),
                payload[begin:end].view(1, -1, 1),
            )
            outputs.append(current_output.flatten())
            routes.append(current_routes.flatten())
    expected_output = torch.cat(outputs) if outputs else torch.empty(0, dtype=torch.uint8)
    expected_routes = torch.cat(routes) if routes else torch.empty(0, dtype=torch.int64)
    if not torch.equal(trace.routes, expected_routes):
        mismatch = int(torch.nonzero(trace.routes != expected_routes)[0])
        raise RuntimeError(f"C++ trace route mismatch at token {mismatch}")
    if not torch.equal(trace.output, expected_output):
        mismatch = int(torch.nonzero(trace.output != expected_output)[0])
        raise RuntimeError(f"C++ trace payload mismatch at token {mismatch}")


PageKey = tuple[int, int, int]


@dataclass
class CacheStats:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    read_misses: int = 0
    write_misses: int = 0
    evictions: int = 0
    dirty_evictions: int = 0
    admission_bypasses: int = 0
    bypassed_pages: set[PageKey] = field(default_factory=set)
    hits_by_region: Counter[int] = field(default_factory=Counter)
    misses_by_region: Counter[int] = field(default_factory=Counter)
    misses_by_token: Counter[int] = field(default_factory=Counter)

    def hit(self, region: int, count: int) -> None:
        self.hits += count
        self.hits_by_region[region] += count

    def miss(
        self,
        token: int,
        region: int,
        *,
        write: bool,
        count: int = 1,
    ) -> None:
        self.misses += count
        self.misses_by_region[region] += count
        if token >= 0:
            self.misses_by_token[token] += count
        if write:
            self.write_misses += count
        else:
            self.read_misses += count


class _Cache:
    def __init__(self, capacity_pages: int) -> None:
        if capacity_pages < 1:
            raise ValueError("cache must hold at least one page")
        self.capacity_pages = capacity_pages
        self.stats = CacheStats()
        self.preloaded_pages = 0

    def touch(
        self,
        key: PageKey,
        *,
        token: int,
        write: bool,
        kind: int,
        repeats: int,
    ) -> None:
        raise NotImplementedError

    def resident_pages(self) -> int:
        raise NotImplementedError


class _LruCache(_Cache):
    def __init__(
        self,
        capacity_pages: int,
        *,
        pinned: Iterable[PageKey] = (),
        bypass_appends: bool = False,
    ) -> None:
        super().__init__(capacity_pages)
        ordered_pins = tuple(dict.fromkeys(pinned))[:capacity_pages]
        self.pinned = set(ordered_pins)
        self.pinned_dirty: set[PageKey] = set()
        self.preloaded_pages = len(self.pinned)
        self.dynamic_capacity = capacity_pages - self.preloaded_pages
        self.pages: OrderedDict[PageKey, bool] = OrderedDict()
        self.bypass_appends = bypass_appends

    def touch(
        self,
        key: PageKey,
        *,
        token: int,
        write: bool,
        kind: int,
        repeats: int,
    ) -> None:
        stats = self.stats
        stats.accesses += repeats
        region = key[0]
        if key in self.pinned:
            if write:
                self.pinned_dirty.add(key)
            stats.hit(region, repeats)
            return
        dirty = self.pages.pop(key, None)
        if dirty is not None:
            self.pages[key] = dirty or write
            stats.hit(region, repeats)
            return
        if self.bypass_appends and write and kind == APPEND_KIND:
            stats.miss(token, region, write=True, count=repeats)
            stats.admission_bypasses += repeats
            stats.bypassed_pages.add(key)
            return

        if self.dynamic_capacity == 0:
            stats.miss(token, region, write=write, count=repeats)
            return
        stats.miss(token, region, write=write)
        if repeats > 1:
            stats.hit(region, repeats - 1)
        if len(self.pages) == self.dynamic_capacity:
            _, evicted_dirty = self.pages.popitem(last=False)
            stats.evictions += 1
            stats.dirty_evictions += int(evicted_dirty)
        self.pages[key] = write

    def resident_pages(self) -> int:
        return len(self.pinned) + len(self.pages)


class _SlruCache(_Cache):
    """Scan-resistant segmented LRU: first touch probation, reuse protected."""

    def __init__(self, capacity_pages: int, protected_fraction: float = 0.75) -> None:
        super().__init__(capacity_pages)
        if capacity_pages == 1:
            self.protected_capacity = 0
        else:
            self.protected_capacity = min(
                capacity_pages - 1,
                max(1, round(capacity_pages * protected_fraction)),
            )
        self.probation_capacity = capacity_pages - self.protected_capacity
        self.probation: OrderedDict[PageKey, bool] = OrderedDict()
        self.protected: OrderedDict[PageKey, bool] = OrderedDict()

    def _evict_probation(self) -> None:
        if len(self.probation) < self.probation_capacity:
            return
        _, dirty = self.probation.popitem(last=False)
        self.stats.evictions += 1
        self.stats.dirty_evictions += int(dirty)

    def _promote(self, key: PageKey, dirty: bool) -> None:
        if self.protected_capacity == 0:
            self._evict_probation()
            self.probation[key] = dirty
            return
        if len(self.protected) == self.protected_capacity:
            demoted_key, demoted_dirty = self.protected.popitem(last=False)
            self._evict_probation()
            self.probation[demoted_key] = demoted_dirty
        self.protected[key] = dirty

    def touch(
        self,
        key: PageKey,
        *,
        token: int,
        write: bool,
        kind: int,
        repeats: int,
    ) -> None:
        del kind
        stats = self.stats
        stats.accesses += repeats
        region = key[0]
        dirty = self.protected.pop(key, None)
        if dirty is not None:
            self.protected[key] = dirty or write
            stats.hit(region, repeats)
            return
        dirty = self.probation.pop(key, None)
        if dirty is not None:
            self._promote(key, dirty or write)
            stats.hit(region, repeats)
            return

        stats.miss(token, region, write=write)
        if repeats > 1:
            stats.hit(region, repeats - 1)
            self._promote(key, write)
            return
        self._evict_probation()
        self.probation[key] = write

    def resident_pages(self) -> int:
        return len(self.probation) + len(self.protected)


class _RegionLruCache(_Cache):
    GROUPS = {
        0: "core",
        1: "core",
        2: "core",
        3: "core",
        4: "core",
        5: "core",
        6: "core",
        7: "latest",
        8: "payload",
    }
    WEIGHTS = {"core": 0.55, "latest": 0.30, "payload": 0.15}

    def __init__(self, capacity_pages: int, active_regions: set[int]) -> None:
        super().__init__(capacity_pages)
        active_groups = {self.GROUPS[region] for region in active_regions}
        capacities = _weighted_capacities(
            capacity_pages,
            {group: self.WEIGHTS[group] for group in active_groups},
        )
        self.capacities = capacities
        self.pages = {group: OrderedDict() for group in active_groups}

    def touch(
        self,
        key: PageKey,
        *,
        token: int,
        write: bool,
        kind: int,
        repeats: int,
    ) -> None:
        del kind
        stats = self.stats
        stats.accesses += repeats
        region = key[0]
        group = self.GROUPS[region]
        pages = self.pages[group]
        dirty = pages.pop(key, None)
        if dirty is not None:
            pages[key] = dirty or write
            stats.hit(region, repeats)
            return
        if self.capacities[group] == 0:
            stats.miss(token, region, write=write, count=repeats)
            return
        stats.miss(token, region, write=write)
        if repeats > 1:
            stats.hit(region, repeats - 1)
        if len(pages) == self.capacities[group]:
            _, evicted_dirty = pages.popitem(last=False)
            stats.evictions += 1
            stats.dirty_evictions += int(evicted_dirty)
        pages[key] = write

    def resident_pages(self) -> int:
        return sum(map(len, self.pages.values()))


def _weighted_capacities(total: int, weights: dict[str, float]) -> dict[str, int]:
    if total < len(weights):
        ordered = sorted(weights, key=weights.get, reverse=True)
        return {name: int(index < total) for index, name in enumerate(ordered)}
    weight_sum = sum(weights.values())
    capacities = {name: 1 for name in weights}
    remaining = total - len(weights)
    raw = {name: remaining * weight / weight_sum for name, weight in weights.items()}
    for name, value in raw.items():
        capacities[name] += math.floor(value)
    unassigned = total - sum(capacities.values())
    for name in sorted(raw, key=lambda item: raw[item] % 1, reverse=True)[:unassigned]:
        capacities[name] += 1
    return capacities


def _structural_pins(page_keys: set[PageKey]) -> list[PageKey]:
    priority = (6, 0, 4, 1, 2, 3)
    result = []
    for region in priority:
        key = (region, 0, 0)
        if key in page_keys:
            result.append(key)
    return result


def _make_cache(
    strategy: str,
    capacity_pages: int,
    *,
    active_regions: set[int],
    page_keys: set[PageKey],
) -> _Cache:
    if strategy == "lru":
        return _LruCache(capacity_pages)
    if strategy == "slru":
        return _SlruCache(capacity_pages)
    if strategy == "region_lru":
        return _RegionLruCache(capacity_pages, active_regions)
    if strategy == "structural_pin_lru":
        return _LruCache(
            capacity_pages,
            pinned=_structural_pins(page_keys),
        )
    if strategy == "append_bypass_lru":
        return _LruCache(capacity_pages, bypass_appends=True)
    raise ValueError(f"unknown cache strategy: {strategy}")


def _percentile(values: list[int], percentile: float) -> int:
    if not values:
        return 0
    values.sort()
    index = min(len(values) - 1, math.ceil(percentile * len(values)) - 1)
    return values[max(index, 0)]


def _cache_report(
    strategy: str,
    cache: _Cache,
    *,
    page_bytes: int,
    tokens: int,
) -> dict[str, object]:
    stats = cache.stats
    token_misses = [stats.misses_by_token.get(token, 0) for token in range(tokens)]
    admitted_misses = stats.misses - stats.admission_bypasses
    return {
        "strategy": strategy,
        "page_bytes": page_bytes,
        "capacity_pages": cache.capacity_pages,
        "cache_bytes": cache.capacity_pages * page_bytes,
        "resident_pages_at_end": cache.resident_pages(),
        "preloaded_pages": cache.preloaded_pages,
        "accesses": stats.accesses,
        "hits": stats.hits,
        "misses": stats.misses,
        "miss_rate": stats.misses / max(stats.accesses, 1),
        "misses_per_token": stats.misses / max(tokens, 1),
        "p50_misses_per_token": _percentile(token_misses, 0.50),
        "p99_misses_per_token": _percentile(token_misses, 0.99),
        "read_misses": stats.read_misses,
        "write_misses": stats.write_misses,
        "evictions": stats.evictions,
        "dirty_evictions": stats.dirty_evictions,
        "admission_bypasses": stats.admission_bypasses,
        "distinct_bypassed_pages": len(stats.bypassed_pages),
        "read_fill_bytes": stats.read_misses * page_bytes,
        "writeback_bytes": stats.dirty_evictions * page_bytes,
        "traffic_upper_bound_bytes": (
            admitted_misses
            + stats.dirty_evictions
            + len(stats.bypassed_pages)
            + cache.preloaded_pages
        )
        * page_bytes,
        "misses_by_region": {
            REGION_NAMES[region]: count
            for region, count in sorted(stats.misses_by_region.items())
        },
    }


def replay_cache_strategies(
    trace: RuntimePageTrace,
    *,
    page_bytes: int,
    cache_bytes: int,
    strategies: Iterable[str] = DEFAULT_STRATEGIES,
) -> list[dict[str, object]]:
    """Replay multiple policies without re-running or changing the automaton."""

    if page_bytes < trace.canonical_page_bytes:
        raise ValueError("page_bytes cannot be smaller than the canonical trace page")
    if page_bytes % trace.canonical_page_bytes:
        raise ValueError("page_bytes must be a multiple of the canonical page size")
    capacity_pages = cache_bytes // page_bytes
    if capacity_pages < 1:
        raise ValueError("cache must hold at least one page")
    factor = page_bytes // trace.canonical_page_bytes
    records = trace.records.numpy()
    mapped_keys = {
        (int(row[1]), int(row[2]), int(row[3]) // factor)
        for row in records
    }
    active_regions = {key[0] for key in mapped_keys}
    caches = {
        strategy: _make_cache(
            strategy,
            capacity_pages,
            active_regions=active_regions,
            page_keys=mapped_keys,
        )
        for strategy in strategies
    }
    for row in records:
        token, region, owner, base_page, flags, kind, repeats = map(int, row)
        key = (region, owner, base_page // factor)
        for cache in caches.values():
            cache.touch(
                key,
                token=token,
                write=bool(flags & WRITE_FLAG),
                kind=kind,
                repeats=repeats,
            )
    return [
        _cache_report(
            strategy,
            cache,
            page_bytes=page_bytes,
            tokens=trace.routes.numel(),
        )
        for strategy, cache in caches.items()
    ]


def summarize_trace(trace: RuntimePageTrace) -> dict[str, object]:
    records = trace.records.numpy()
    touches_by_region: Counter[int] = Counter()
    touches_by_kind: Counter[int] = Counter()
    unique_pages_by_region: dict[int, set[tuple[int, int]]] = {}
    for row in records:
        _, region, owner, page, _, kind, repeats = map(int, row)
        touches_by_region[region] += repeats
        touches_by_kind[kind] += repeats
        unique_pages_by_region.setdefault(region, set()).add((owner, page))
    return {
        **trace.summary,
        "canonical_page_bytes": trace.canonical_page_bytes,
        "trace_compression_ratio": (
            trace.summary["canonical_page_touches"]
            / max(trace.summary["rle_records"], 1)
        ),
        "touches_by_region": {
            REGION_NAMES[region]: count
            for region, count in sorted(touches_by_region.items())
        },
        "touches_by_kind": {
            KIND_NAMES[kind]: count
            for kind, count in sorted(touches_by_kind.items())
        },
        "unique_pages_by_region": {
            REGION_NAMES[region]: len(pages)
            for region, pages in sorted(unique_pages_by_region.items())
        },
        "complexity": trace.complexity,
        "route_checksum": int(trace.routes.sum()),
        "output_checksum": int(trace.output.to(torch.int64).sum()),
    }


def make_case(
    tokens: int,
    bits: int,
    pattern: str,
    seed: int,
    natural_path: Path,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    maximum = 1 << bits
    generator = torch.Generator().manual_seed(seed + tokens)
    if pattern == "random":
        query = torch.randint(maximum, (tokens,), generator=generator)
        key = torch.randint(maximum, (tokens,), generator=generator)
    elif pattern == "skewed":
        query = torch.randint(maximum, (tokens,), generator=generator)
        key = torch.randint(maximum, (tokens,), generator=generator)
        query.masked_fill_(torch.rand(tokens, generator=generator) < 0.9, 0)
        key.masked_fill_(torch.rand(tokens, generator=generator) < 0.9, 0)
    elif pattern == "all_match":
        query = torch.zeros(tokens, dtype=torch.int64)
        key = torch.zeros_like(query)
    elif pattern.startswith("periodic"):
        period = int(pattern.removeprefix("periodic"))
        motif = torch.randint(maximum, (period,), generator=generator)
        key = motif[torch.arange(tokens) % period].clone()
        query = key.clone()
    elif pattern.startswith("copied"):
        lag = int(pattern.removeprefix("copied"))
        key = torch.randint(maximum, (tokens,), generator=generator)
        query = torch.randint(maximum, (tokens,), generator=generator)
        if lag < tokens:
            query[lag:] = key[:-lag]
    elif pattern == "natural":
        raw = natural_path.read_bytes()
        if not raw:
            raise ValueError("natural input file is empty")
        values = torch.tensor(list(raw), dtype=torch.int64) & (maximum - 1)
        key = values[torch.arange(tokens) % values.numel()].clone()
        query = key.clone()
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    payload = torch.arange(tokens, dtype=torch.int64) & 255
    return query.to(torch.uint8), key.to(torch.uint8), payload.to(torch.uint8)


def _mib(values: Iterable[float]) -> list[int]:
    return [round(value * 1024 * 1024) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192])
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["random", "skewed", "periodic64", "copied1024", "natural"],
    )
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--payload-bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--chunk-size", type=int, default=1024)
    parser.add_argument("--canonical-page-kib", type=int, default=4)
    parser.add_argument("--page-kib", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument("--cache-mib", type=float, nargs="+", default=[0.25, 1.0])
    parser.add_argument("--strategies", nargs="+", default=list(DEFAULT_STRATEGIES))
    parser.add_argument(
        "--natural-path",
        type=Path,
        default=ROOT / "README.md",
    )
    parser.add_argument("--skip-native-validation", action="store_true")
    parser.add_argument("--verbose-build", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    module = load_runtime_page_trace(verbose=args.verbose_build)
    rows = []
    for tokens in args.tokens:
        for pattern in args.patterns:
            query, key, payload = make_case(
                tokens,
                args.bits,
                pattern,
                args.seed,
                args.natural_path,
            )
            trace = collect_runtime_page_trace(
                query,
                key,
                payload,
                qk_bits=args.bits,
                payload_bits=args.payload_bits,
                chunk_size=args.chunk_size,
                canonical_page_bytes=args.canonical_page_kib * 1024,
                module=module,
            )
            if not args.skip_native_validation:
                validate_trace_against_runtime(
                    trace,
                    query,
                    key,
                    payload,
                    qk_bits=args.bits,
                    payload_bits=args.payload_bits,
                    chunk_size=args.chunk_size,
                )
            cache_rows = []
            for page_kib in args.page_kib:
                for cache_bytes in _mib(args.cache_mib):
                    if cache_bytes < page_kib * 1024:
                        continue
                    cache_rows.extend(
                        replay_cache_strategies(
                            trace,
                            page_bytes=page_kib * 1024,
                            cache_bytes=cache_bytes,
                            strategies=args.strategies,
                        )
                    )
            row = {
                "tokens": tokens,
                "pattern": pattern,
                "trace": summarize_trace(trace),
                "cache_rows": cache_rows,
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "settings": {
            **vars(args),
            "natural_path": str(args.natural_path),
            "output": None if args.output is None else str(args.output),
        },
        "trace_model": (
            "same C++ automaton core as production; stable logical pages; "
            "allocator relocation and input/output tensor traffic excluded"
        ),
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
