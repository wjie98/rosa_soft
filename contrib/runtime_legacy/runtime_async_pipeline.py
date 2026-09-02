"""Measure packed GPU staging and the exact CPU runtime pipeline."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from rosa_soft import RosaRuntime


def _run(
    query: torch.Tensor,
    key: torch.Tensor,
    payload: torch.Tensor,
    bits: int,
    chunk_size: int,
    mode: str,
) -> tuple[float, int]:
    stream = torch.cuda.Stream(device=query.device) if mode != "blocking" else None
    checksum = 0
    pending = []
    torch.cuda.synchronize(query.device)
    start = time.perf_counter()
    with RosaRuntime(
        query.size(2),
        payload.size(2),
        bits,
        bits,
    ) as runtime:
        for begin in range(0, query.size(1), chunk_size):
            end = min(query.size(1), begin + chunk_size)
            if mode == "blocking":
                output, matched = runtime.update_packed(
                    query[:, begin:end],
                    key[:, begin:end],
                    payload[:, begin:end],
                )
                checksum += int(output.sum()) + 1_000_003 * int(matched.sum())
                continue

            work = runtime.update_packed(
                query[:, begin:end],
                key[:, begin:end],
                payload[:, begin:end],
                stream=stream,
                async_op=True,
            )
            pending.append(work)
            if mode == "stream_sync" or len(pending) == 2:
                output, matched = pending.pop(0).wait()
                checksum += int(output.sum()) + 1_000_003 * int(matched.sum())

        for work in pending:
            output, matched = work.wait()
            checksum += int(output.sum()) + 1_000_003 * int(matched.sum())
    torch.cuda.synchronize(query.device)
    return time.perf_counter() - start, checksum


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, nargs="+", default=[16384, 65536])
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[256, 1024, 4096])
    parser.add_argument("--modes", nargs="+", default=["blocking", "stream_sync", "stream_async"])
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--payload-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    rows = []
    for tokens in args.tokens:
        generator = torch.Generator().manual_seed(args.seed + tokens)
        shape = (args.batch, tokens, args.heads)
        query = torch.randint(1 << args.bits, shape, generator=generator, dtype=torch.uint8).to(args.device)
        key = torch.randint(1 << args.bits, shape, generator=generator, dtype=torch.uint8).to(args.device)
        payload = torch.randint(
            1 << args.bits,
            (args.batch, tokens, args.payload_heads),
            generator=generator,
            dtype=torch.uint8,
        ).to(args.device)
        expected_checksum = None
        for chunk_size in args.chunk_sizes:
            chunk_size = min(chunk_size, tokens)
            for mode in args.modes:
                samples = []
                checksums = set()
                for _ in range(args.repeats):
                    elapsed, checksum = _run(
                        query,
                        key,
                        payload,
                        args.bits,
                        chunk_size,
                        mode,
                    )
                    samples.append(elapsed)
                    checksums.add(checksum)
                if len(checksums) != 1:
                    raise RuntimeError("runtime pipeline is nondeterministic")
                checksum = next(iter(checksums))
                if expected_checksum is None:
                    expected_checksum = checksum
                elif checksum != expected_checksum:
                    raise RuntimeError("runtime mode or chunk size changed output")
                elapsed = statistics.median(samples)
                row = {
                    "tokens": tokens,
                    "chunk_size": chunk_size,
                    "mode": mode,
                    "latency_ms": elapsed * 1000,
                    "tokens_per_second": args.batch * tokens / elapsed,
                    "checksum": checksum,
                }
                rows.append(row)
                print(json.dumps(row, sort_keys=True), flush=True)

    report = {
        "settings": vars(args) | {"output": str(args.output)},
        "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "rows": rows,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
