"""Profile equivalent full-diagonal scheduling policies."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from benchmarks.persistent_diagonal_schedule import (
    SCHEDULES,
    diagonal_schedule,
    load_diagonal_schedule,
)


def _latency_ms(operation, *, warmup: int, repeats: int, rounds: int) -> float:
    samples = []
    for _ in range(rounds):
        for _ in range(warmup):
            operation()
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(repeats):
            operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    return statistics.median(samples)


def _case(tokens: int, heads: int, pattern: str, seed: int):
    generator = torch.Generator(device="cuda").manual_seed(seed + tokens)
    query = torch.randint(
        0,
        256,
        (1, heads, tokens),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    key = torch.randint(
        0,
        256,
        query.shape,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    if pattern == "equal":
        key.copy_(query)
    elif pattern == "periodic":
        periodic = (torch.arange(tokens, device="cuda") % 4).to(torch.int32)
        query.copy_(periodic)
        key.copy_(periodic)
    return query, key


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", nargs="+", type=int, default=[1024, 4096, 8192])
    parser.add_argument("--heads", nargs="+", type=int, default=[1, 4])
    parser.add_argument("--patterns", nargs="+", default=["random", "equal", "periodic"])
    parser.add_argument("--worker-multipliers", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2700)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    module = load_diagonal_schedule()
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    rows = []
    for tokens in args.lengths:
        for heads in args.heads:
            for pattern in args.patterns:
                query, key = _case(tokens, heads, pattern, args.seed)
                baseline = diagonal_schedule(
                    query, key, symbol_dim=8, module=module
                )
                baseline_ms = _latency_ms(
                    lambda: diagonal_schedule(
                        query, key, symbol_dim=8, module=module
                    ),
                    warmup=args.warmup,
                    repeats=args.repeats,
                    rounds=args.rounds,
                )
                methods = {"hardware_queue": {"latency_ms": baseline_ms}}
                paired = diagonal_schedule(
                    query,
                    key,
                    symbol_dim=8,
                    schedule="paired_launch",
                    module=module,
                )
                methods["paired_launch"] = {
                    "latency_ms": _latency_ms(
                        lambda: diagonal_schedule(
                            query,
                            key,
                            symbol_dim=8,
                            schedule="paired_launch",
                            module=module,
                        ),
                        warmup=args.warmup,
                        repeats=args.repeats,
                        rounds=args.rounds,
                    ),
                    "max_abs_error": float((paired - baseline).abs().max()),
                }
                for multiplier in args.worker_multipliers:
                    workers = sms * multiplier
                    for schedule in SCHEDULES[2:]:
                        operation = lambda schedule=schedule, workers=workers: diagonal_schedule(
                            query,
                            key,
                            symbol_dim=8,
                            schedule=schedule,
                            worker_blocks=workers,
                            module=module,
                        )
                        actual = operation()
                        methods[f"{schedule}_smx{multiplier}"] = {
                            "latency_ms": _latency_ms(
                                operation,
                                warmup=args.warmup,
                                repeats=args.repeats,
                                rounds=args.rounds,
                            ),
                            "max_abs_error": float(
                                (actual - baseline).abs().max()
                            ),
                        }
                row = {
                    "tokens": tokens,
                    "heads": heads,
                    "pattern": pattern,
                    "methods": methods,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "sm_count": sms,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "settings": vars(args) | {"output": str(args.output)},
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
