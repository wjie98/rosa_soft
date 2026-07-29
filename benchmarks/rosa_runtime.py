"""Standalone RosaRuntime scaling benchmark."""

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rosa_soft import RosaRuntime


def benchmark(
    query,
    key,
    payload,
    qk_bits,
    payload_bits,
    max_suffix_length,
    repeats,
):
    timings = []
    stats = None
    for _ in range(repeats):
        rt = RosaRuntime(
            num_heads=query.size(-1),
            num_payload_heads=payload.size(-1),
            qk_bits=qk_bits,
            payload_bits=payload_bits,
            max_suffix_length=max_suffix_length,
        )
        t0 = time.perf_counter()
        output, matched_key_end_positions = rt.update_packed(
            query,
            key,
            payload,
        )
        elapsed = time.perf_counter() - t0
        stats = rt.stats()
        rt.close()
        timings.append(elapsed)
        del output, matched_key_end_positions
    timings.sort()
    return timings[len(timings) // 2], stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--T", type=int, default=256)
    parser.add_argument("--H", type=int, default=8)
    parser.add_argument("--Hv", type=int, default=2)
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--max-suffix-length", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(123)
    query = torch.randint(
        0,
        1 << args.bits,
        (args.B, args.T, args.H),
        dtype=torch.uint8,
    )
    key = torch.randint(
        0,
        1 << args.bits,
        (args.B, args.T, args.H),
        dtype=torch.uint8,
    )
    payload = torch.randint(
        0,
        1 << args.bits,
        (args.B, args.T, args.Hv),
        dtype=torch.uint8,
    )

    elapsed, stats = benchmark(
        query,
        key,
        payload,
        args.bits,
        args.bits,
        args.max_suffix_length,
        args.repeats,
    )

    print("max_suffix_length,time_ms,states,edges,payload_symbols")
    print(
        f"{args.max_suffix_length},{elapsed * 1000:.3f},"
        f"{stats[0]},{stats[1]},{stats[2]}"
    )


if __name__ == "__main__":
    main()
