"""Exact bounded-state prototype for a key stream made of one repeated block.

The repeated block is a macro symbol. Query matching remains token exact: the
tracker carries one suffix length per phase of the primitive block and derives
the latest finite-history end position for every active phase.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path

import torch

from rosa_soft import RosaRuntime


def _primitive_period(block: list[int]) -> list[int]:
    if not block:
        raise ValueError("block must not be empty")
    for period in range(1, len(block) + 1):
        if (
            len(block) % period == 0
            and all(
                block[index] == block[index % period]
                for index in range(len(block))
            )
        ):
            return block[:period]
    raise AssertionError("the full block is always a period")


@dataclass(frozen=True)
class PeriodicStateStats:
    tokens: int
    phase_updates: int
    period: int

    @property
    def logical_bytes(self) -> int:
        # Motif bytes, int32 phase lengths, and uint32 phase epochs.
        return self.period * (1 + 4 + 4)


class PeriodicSuffixState:
    """Exact latest-longest matcher for ``key[t] == motif[t % period]``."""

    def __init__(self, block: torch.Tensor | list[int]) -> None:
        raw = block.tolist() if isinstance(block, torch.Tensor) else list(block)
        self.motif = _primitive_period([int(symbol) for symbol in raw])
        self.period = len(self.motif)
        self._phases_by_symbol: dict[int, list[int]] = {}
        for phase, symbol in enumerate(self.motif):
            self._phases_by_symbol.setdefault(symbol, []).append(phase)
        self._lengths = [0] * self.period
        self._epochs = [-1] * self.period
        self._token_count = 0
        self._phase_updates = 0

    def update(self, query: int, key: int) -> int:
        token = self._token_count
        expected_key = self.motif[token % self.period]
        if key != expected_key:
            raise ValueError(
                f"key left periodic language at token {token}: "
                f"expected {expected_key}, got {key}"
            )

        previous_epoch = token - 1
        best_length = 0
        best_end = -1
        pending: list[tuple[int, int]] = []
        for phase in self._phases_by_symbol.get(query, ()):
            previous_phase = (phase - 1) % self.period
            previous_length = (
                self._lengths[previous_phase]
                if self._epochs[previous_phase] == previous_epoch
                else 0
            )
            length = previous_length + 1
            pending.append((phase, length))
            self._phase_updates += 1

            if phase >= token:
                continue
            latest_end = phase + ((token - 1 - phase) // self.period) * self.period
            finite_length = min(length, latest_end + 1)
            if finite_length > best_length or (
                finite_length == best_length
                and finite_length > 0
                and latest_end > best_end
            ):
                best_length = finite_length
                best_end = latest_end

        for phase, length in pending:
            self._lengths[phase] = length
            self._epochs[phase] = token
        self._token_count += 1
        return best_end

    def stats(self) -> PeriodicStateStats:
        return PeriodicStateStats(
            tokens=self._token_count,
            phase_updates=self._phase_updates,
            period=self.period,
        )


def periodic_routes(
    query: torch.Tensor,
    key: torch.Tensor,
    block: torch.Tensor,
) -> torch.Tensor:
    if query.ndim != 1 or key.shape != query.shape:
        raise ValueError("query and key must be equal-length vectors")
    state = PeriodicSuffixState(block)
    return torch.tensor(
        [state.update(int(q), int(k)) for q, k in zip(query, key)],
        dtype=torch.int64,
    )


def _time(function, repeats: int) -> tuple[float, object]:
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append(time.perf_counter() - start)
    return statistics.median(samples), result


def _run_case(
    tokens: int,
    period: int,
    bits: int,
    seed: int,
    repeats: int,
) -> dict[str, object]:
    generator = torch.Generator().manual_seed(seed + tokens + period)
    block = torch.randint(1 << bits, (period,), generator=generator, dtype=torch.uint8)
    motif = torch.tensor(_primitive_period(block.tolist()), dtype=torch.uint8)
    positions = torch.arange(tokens) % motif.numel()
    key = motif[positions]
    query = torch.randint(1 << bits, (tokens,), generator=generator, dtype=torch.uint8)

    tracker = PeriodicSuffixState(motif)
    periodic_seconds, periodic_result = _time(
        lambda: torch.tensor(
            [tracker.update(int(q), int(k)) for q, k in zip(query, key)],
            dtype=torch.int64,
        ),
        1,
    )
    tracker_stats = tracker.stats()

    def native_run():
        payload = torch.zeros(1, tokens, 1, dtype=torch.uint8)
        with RosaRuntime(1, 1, bits, 1) as runtime:
            _, ends = runtime.update_packed(
                query.view(1, tokens, 1),
                key.view(1, tokens, 1),
                payload,
            )
            return ends.flatten(), runtime.memory_stats(), runtime.complexity_stats()

    native_seconds, native_result = _time(native_run, repeats)
    native_ends, native_memory, native_complexity = native_result
    if not torch.equal(periodic_result, native_ends):
        mismatch = int(torch.nonzero(periodic_result != native_ends)[0])
        raise RuntimeError(f"periodic state disagrees with exact runtime at {mismatch}")

    return {
        "tokens": tokens,
        "requested_period": period,
        "primitive_period": motif.numel(),
        "bits": bits,
        "periodic_ms": periodic_seconds * 1000,
        "native_ms": native_seconds * 1000,
        "periodic_phase_updates": tracker_stats.phase_updates,
        "periodic_logical_bytes": tracker_stats.logical_bytes,
        "native_logical_bytes": native_memory["logical_bytes"],
        "native_automaton_bytes": (
            native_memory["logical_bytes"] - native_memory["payload_symbols"]
        ),
        "native_states": native_memory["states"],
        "native_complexity": native_complexity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[8192, 65536])
    parser.add_argument("--periods", type=int, nargs="+", default=[1, 2, 8, 64])
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for tokens in args.tokens:
        for period in args.periods:
            row = _run_case(tokens, period, args.bits, args.seed, args.repeats)
            rows.append(row)
            print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "settings": vars(args) | {"output": str(args.output)},
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
