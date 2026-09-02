"""Exact unlimited-SAM page-locality lower-bound experiment.

The prototype preserves the online query-before-key-update contract and latest
tie behavior. It uses ideal O(1) transition lookup but exact direct suffix-path
latest-end writes instead of the production link-cut tree. Transition accesses
are therefore a lower bound; aggregate page counts are a decomposition tool,
not a production-runtime prediction. No cache setting changes routes or limits
suffix length.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import torch

from rosa_soft import RosaRuntime


_RECORD_BYTES = {
    "state": 16,
    "edge": 9,
    "payload": 1,
    "control": 32,
}


@dataclass
class _CacheStats:
    accesses: int = 0
    hits: int = 0
    misses: int = 0
    read_misses: int = 0
    write_misses: int = 0
    dirty_evictions: int = 0


class _LruPages:
    def __init__(self, page_size: int, cache_bytes: int) -> None:
        if page_size <= 0 or cache_bytes < page_size:
            raise ValueError("cache must hold at least one positive-size page")
        self.page_size = page_size
        self.capacity = cache_bytes // page_size
        self.pages: OrderedDict[tuple[str, int], bool] = OrderedDict()
        self.stats = _CacheStats()

    def touch(self, page: tuple[str, int], *, write: bool) -> None:
        stats = self.stats
        stats.accesses += 1
        dirty = self.pages.pop(page, None)
        if dirty is not None:
            stats.hits += 1
            self.pages[page] = dirty or write
            return

        stats.misses += 1
        if write:
            stats.write_misses += 1
        else:
            stats.read_misses += 1
        if len(self.pages) == self.capacity:
            _, evicted_dirty = self.pages.popitem(last=False)
            stats.dirty_evictions += int(evicted_dirty)
        self.pages[page] = write

    def report(self) -> dict[str, int | float]:
        stats = self.stats
        return {
            "page_size": self.page_size,
            "capacity_pages": self.capacity,
            "resident_bytes": self.capacity * self.page_size,
            "accesses": stats.accesses,
            "hits": stats.hits,
            "misses": stats.misses,
            "miss_rate": stats.misses / max(stats.accesses, 1),
            "read_misses": stats.read_misses,
            "write_misses": stats.write_misses,
            "dirty_evictions": stats.dirty_evictions,
            "transfer_bytes_upper_bound": stats.misses * self.page_size,
        }


class PageAccessProfiler:
    """Replay logical record accesses through several exact LRU caches."""

    def __init__(
        self,
        configurations: Iterable[tuple[int, int]],
        *,
        hot_tail_bytes: int,
    ) -> None:
        self.caches = [_LruPages(*config) for config in configurations]
        self.hot_tail_bytes = hot_tail_bytes
        self.accesses_by_kind: Counter[str] = Counter()
        self.reads_by_region: Counter[str] = Counter()
        self.writes_by_region: Counter[str] = Counter()
        self.max_record = {region: -1 for region in _RECORD_BYTES}
        self.cold_writes: Counter[str] = Counter()
        self.cold_written_records: set[tuple[str, int]] = set()

    def touch(
        self,
        region: str,
        index: int,
        kind: str,
        *,
        write: bool = False,
    ) -> None:
        if region not in _RECORD_BYTES or index < 0:
            raise ValueError("invalid logical record")
        width = _RECORD_BYTES[region]
        self.max_record[region] = max(self.max_record[region], index)
        self.accesses_by_kind[kind] += 1
        target = self.writes_by_region if write else self.reads_by_region
        target[region] += 1

        frontier_bytes = (self.max_record[region] + 1) * width
        record_end = (index + 1) * width
        if (
            write
            and region in {"state", "edge"}
            and record_end <= frontier_bytes - self.hot_tail_bytes
        ):
            self.cold_writes[region] += 1
            self.cold_written_records.add((region, index))

        byte_offset = index * width
        for cache in self.caches:
            cache.touch(
                (region, byte_offset // cache.page_size),
                write=write,
            )

    def report(self) -> dict[str, object]:
        logical_bytes = {
            region: (maximum + 1) * _RECORD_BYTES[region]
            for region, maximum in self.max_record.items()
            if maximum >= 0
        }
        return {
            "logical_bytes": logical_bytes,
            "total_logical_bytes": sum(logical_bytes.values()),
            "accesses_by_kind": dict(sorted(self.accesses_by_kind.items())),
            "reads_by_region": dict(sorted(self.reads_by_region.items())),
            "writes_by_region": dict(sorted(self.writes_by_region.items())),
            "cold_writes": dict(sorted(self.cold_writes.items())),
            "distinct_cold_written_records": len(self.cold_written_records),
            "caches": [cache.report() for cache in self.caches],
        }


@dataclass
class _State:
    max_length: int = 0
    latest_end: int = -1
    suffix_link: int = -1
    transitions: dict[int, int] = field(default_factory=dict)


@dataclass
class _Edge:
    symbol: int
    next_state: int


class ExactPagedSamModel:
    """Exact online SAM with ideal indexed transitions and page tracing."""

    def __init__(self, profiler: PageAccessProfiler) -> None:
        self.profiler = profiler
        self.states = [_State()]
        self.edges: list[_Edge] = []
        self.query_state = 0
        self.last_key_state = 0
        self.key_count = 0
        self.payload_count = 0
        self.compressed_run_active = True
        self.compressed_run_initialized = False
        self.compressed_symbol = 0
        self.compressed_run_length = 0
        self.compressed_query_length = 0
        self.profiler.touch("state", 0, "state_create", write=True)
        self.profiler.touch("control", 0, "control_create", write=True)

    def _state(self, index: int, kind: str, *, write: bool = False) -> _State:
        self.profiler.touch("state", index, kind, write=write)
        return self.states[index]

    def _edge(self, index: int, kind: str, *, write: bool = False) -> _Edge:
        self.profiler.touch("edge", index, kind, write=write)
        return self.edges[index]

    def _add_state(self, state: _State | None = None) -> int:
        index = len(self.states)
        self.states.append(_State() if state is None else state)
        self.profiler.touch("state", index, "state_create", write=True)
        return index

    def _find_edge(self, state: int, symbol: int, kind: str) -> int:
        target = self._state(state, f"{kind}_transition_index")
        edge = target.transitions.get(symbol, -1)
        if edge >= 0:
            self._edge(edge, f"{kind}_transition")
        return edge

    def _add_transition(self, state: int, symbol: int, next_state: int) -> None:
        target = self._state(state, "extension_transition_add", write=True)
        if symbol in target.transitions:
            raise AssertionError("transition already exists")
        edge = len(self.edges)
        self.edges.append(_Edge(symbol, next_state))
        target.transitions[symbol] = edge
        self.profiler.touch("edge", edge, "edge_create", write=True)

    def _redirect_transition(self, edge: int, next_state: int) -> None:
        self._edge(edge, "extension_transition_redirect", write=True).next_state = (
            next_state
        )

    def _copy_transitions(self, source: int, target: int) -> None:
        source_state = self._state(source, "clone_source")
        target_state = self._state(target, "clone_target", write=True)
        for symbol, source_edge in source_state.transitions.items():
            next_state = self._edge(source_edge, "clone_edge_read").next_state
            edge = len(self.edges)
            self.edges.append(_Edge(symbol, next_state))
            target_state.transitions[symbol] = edge
            self.profiler.touch("edge", edge, "clone_edge_create", write=True)

    def _materialize_run(self) -> None:
        length = self.compressed_run_length
        root = self._state(0, "run_materialize_root", write=True)
        root.latest_end = length - 1
        for represented_length in range(1, length + 1):
            state = self._add_state(
                _State(
                    max_length=represented_length,
                    latest_end=length - 1,
                    suffix_link=represented_length - 1,
                )
            )
            self._add_transition(
                represented_length - 1,
                self.compressed_symbol,
                state,
            )
        self.query_state = self.compressed_query_length
        self.last_key_state = length
        self.key_count = length
        self.compressed_run_active = False

    def _match_query(self, symbol: int) -> int:
        state = self.query_state
        edge = self._find_edge(state, symbol, "query")
        while state != 0 and edge < 0:
            state = self._state(state, "query_suffix").suffix_link
            edge = self._find_edge(state, symbol, "query")
        if edge < 0:
            self.query_state = 0
            return -1
        self.query_state = self._edge(edge, "query_transition_target").next_state
        return self._state(self.query_state, "query_latest").latest_end

    def _extend_key(self, symbol: int) -> None:
        end_position = self.key_count
        self.key_count += 1
        previous_last = self._state(self.last_key_state, "extension_last")
        next_state = self._add_state(
            _State(max_length=previous_last.max_length + 1)
        )

        parent = self.last_key_state
        child = -1
        while parent >= 0:
            edge = self._find_edge(parent, symbol, "extension")
            if edge >= 0:
                child = self._edge(edge, "extension_transition_target").next_state
                break
            self._add_transition(parent, symbol, next_state)
            parent = self._state(parent, "extension_suffix").suffix_link

        if parent < 0:
            self._state(next_state, "extension_link", write=True).suffix_link = 0
        else:
            parent_state = self._state(parent, "extension_parent")
            child_state = self._state(child, "extension_child")
            if parent_state.max_length + 1 == child_state.max_length:
                self._state(
                    next_state, "extension_link", write=True
                ).suffix_link = child
            else:
                clone = self._add_state(
                    _State(
                        max_length=parent_state.max_length + 1,
                        latest_end=child_state.latest_end,
                        suffix_link=child_state.suffix_link,
                    )
                )
                self._copy_transitions(child, clone)
                self._state(child, "clone_reparent", write=True).suffix_link = clone
                self._state(
                    next_state, "extension_link", write=True
                ).suffix_link = clone
                while parent >= 0:
                    edge = self._find_edge(parent, symbol, "clone_redirect")
                    if edge < 0 or self._edge(
                        edge, "clone_redirect_target"
                    ).next_state != child:
                        break
                    self._redirect_transition(edge, clone)
                    parent = self._state(parent, "clone_suffix").suffix_link

        self.last_key_state = next_state
        state = next_state
        while state >= 0:
            current = self._state(state, "latest_write", write=True)
            current.latest_end = end_position
            state = current.suffix_link

    def _update_compressed_run(self, query: int, key: int) -> int:
        self.profiler.touch("control", 0, "run_control", write=True)
        if not self.compressed_run_initialized:
            self.compressed_run_initialized = True
            self.compressed_symbol = key
            self.compressed_run_length = 1
            return -1

        matched_end = -1
        if query == self.compressed_symbol:
            self.compressed_query_length = min(
                self.compressed_query_length + 1,
                self.compressed_run_length,
            )
            matched_end = self.compressed_run_length - 1
        else:
            self.compressed_query_length = 0

        if key == self.compressed_symbol:
            self.compressed_run_length += 1
            return matched_end
        self._materialize_run()
        self._extend_key(key)
        return matched_end

    def update(self, query: int, key: int) -> int:
        self.profiler.touch(
            "payload", self.payload_count, "payload_append", write=True
        )
        self.payload_count += 1
        if self.compressed_run_active:
            matched_end = self._update_compressed_run(query, key)
        else:
            matched_end = self._match_query(query)
            self._extend_key(key)
        if matched_end >= 0:
            self.profiler.touch(
                "payload", matched_end + 1, "payload_successor"
            )
        return matched_end

    def logical_counts(self) -> dict[str, int]:
        return {
            "states": len(self.states),
            "edges": len(self.edges),
            "payload_symbols": self.payload_count,
        }


def _make_case(
    tokens: int,
    bits: int,
    pattern: str,
    seed: int,
    natural_path: Path,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    return query.to(torch.uint8), key.to(torch.uint8)


def profile_case(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    page_sizes: Iterable[int],
    cache_bytes: Iterable[int],
    hot_tail_bytes: int,
    validate_native: bool,
) -> dict[str, object]:
    configurations = [
        (page_size, budget)
        for page_size in page_sizes
        for budget in cache_bytes
        if budget >= page_size
    ]
    profiler = PageAccessProfiler(
        configurations,
        hot_tail_bytes=hot_tail_bytes,
    )
    model = ExactPagedSamModel(profiler)
    start = time.perf_counter()
    routes = torch.tensor(
        [model.update(int(q), int(k)) for q, k in zip(query, key)],
        dtype=torch.int64,
    )
    elapsed = time.perf_counter() - start

    if validate_native:
        payload = torch.zeros(1, query.numel(), 1, dtype=torch.uint8)
        bits = max(1, max(int(query.max()), int(key.max())).bit_length())
        with RosaRuntime(1, 1, bits, 1) as runtime:
            _, native_routes = runtime.update_packed(
                query.view(1, -1, 1),
                key.view(1, -1, 1),
                payload,
            )
        native_routes = native_routes.flatten()
        if not torch.equal(routes, native_routes):
            mismatch = int(torch.nonzero(routes != native_routes)[0])
            raise RuntimeError(f"page model route mismatch at token {mismatch}")

    return {
        "tokens": query.numel(),
        "elapsed_ms": elapsed * 1000,
        "checksum": int(routes.sum()),
        "counts": model.logical_counts(),
        "page_profile": profiler.report(),
    }


def _parse_megabytes(values: Iterable[float]) -> list[int]:
    return [round(value * 1024 * 1024) for value in values]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 65536])
    parser.add_argument(
        "--patterns",
        nargs="+",
        default=["random", "skewed", "periodic64", "copied1024", "natural"],
    )
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--page-kib", type=int, nargs="+", default=[4, 16, 64])
    parser.add_argument(
        "--cache-mib", type=float, nargs="+", default=[0.25, 1.0, 4.0]
    )
    parser.add_argument("--hot-tail-mib", type=float, default=1.0)
    parser.add_argument(
        "--natural-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "README.md",
    )
    parser.add_argument("--skip-native-validation", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    page_sizes = [value * 1024 for value in args.page_kib]
    budgets = _parse_megabytes(args.cache_mib)
    hot_tail_bytes = round(args.hot_tail_mib * 1024 * 1024)
    rows = []
    for tokens in args.tokens:
        for pattern in args.patterns:
            query, key = _make_case(
                tokens,
                args.bits,
                pattern,
                args.seed,
                args.natural_path,
            )
            row = {
                "pattern": pattern,
                **profile_case(
                    query,
                    key,
                    page_sizes=page_sizes,
                    cache_bytes=budgets,
                    hot_tail_bytes=hot_tail_bytes,
                    validate_native=not args.skip_native_validation,
                ),
            }
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "settings": vars(args)
        | {
            "natural_path": str(args.natural_path),
            "output": str(args.output),
        },
        "model": (
            "exact routes; ideal indexed transitions; direct exact latest-end "
            "path writes rather than the production link-cut tree"
        ),
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
