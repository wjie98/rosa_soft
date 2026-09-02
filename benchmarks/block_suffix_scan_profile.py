"""Profile exact suffix ownership strategies on one CUDA device."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import torch

from benchmarks.block_suffix_scan import (
    METHODS,
    block_suffix_scores,
    load_block_suffix_scan,
)


def _measure_ms(operation, *, warmup: int, repeats: int) -> float:
    for _ in range(warmup):
        operation()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lengths", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--windows", type=int, nargs="+", default=[1, 8, 16, 32, 64])
    parser.add_argument("--bits", type=int, nargs="+", default=[8, 32])
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=9)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    module = load_block_suffix_scan()
    rows = []
    for bits in args.bits:
        for seq_len in args.lengths:
            generator = torch.Generator(device="cuda").manual_seed(
                13000 + bits + seq_len
            )
            high = 2**31 - 1 if bits == 32 else 1 << bits
            low = -(2**31) if bits == 32 else 0
            query = torch.randint(
                low,
                high,
                (1, args.heads, seq_len),
                dtype=torch.int32,
                device="cuda",
                generator=generator,
            )
            key = torch.randint(
                low,
                high,
                query.shape,
                dtype=torch.int32,
                device="cuda",
                generator=generator,
            )
            for window in args.windows:
                effective_window = min(window, seq_len)
                timings = {}
                outputs = {}
                for method in args.methods:
                    operation = lambda method=method: block_suffix_scores(
                        query,
                        key,
                        symbol_dim=bits,
                        max_suffix_length=effective_window,
                        method=method,
                        module=module,
                    )
                    outputs[method] = operation()
                    timings[method] = _measure_ms(
                        operation,
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                reference = outputs["thread"]
                for method in args.methods:
                    error = float((outputs[method] - reference).abs().max())
                    row = {
                        "bits": bits,
                        "seq_len": seq_len,
                        "window": effective_window,
                        "method": method,
                        "latency_ms": timings[method],
                        "thread_ratio": timings[method] / timings["thread"],
                        "max_abs_error": error,
                    }
                    rows.append(row)
                    print(json.dumps(row, sort_keys=True), flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "heads": args.heads,
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
