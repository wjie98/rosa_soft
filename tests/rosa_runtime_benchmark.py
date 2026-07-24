import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rosa_soft import RosaRuntime


def benchmark(q, k, v, qk_bits, value_bits, max_suffix_length, repeats):
    timings = []
    stats = None
    for _ in range(repeats):
        rt = RosaRuntime(
            num_heads=q.size(-1),
            num_value_heads=v.size(-1),
            qk_bits=qk_bits,
            value_bits=value_bits,
            max_suffix_length=max_suffix_length,
        )
        t0 = time.perf_counter()
        out, endpos = rt.update_packed(q, k, v)
        elapsed = time.perf_counter() - t0
        stats = rt.stats()
        rt.close()
        timings.append(elapsed)
        del out, endpos
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
    q = torch.randint(0, 1 << args.bits, (args.B, args.T, args.H), dtype=torch.uint8)
    k = torch.randint(0, 1 << args.bits, (args.B, args.T, args.H), dtype=torch.uint8)
    v = torch.randint(0, 1 << args.bits, (args.B, args.T, args.Hv), dtype=torch.uint8)

    elapsed, stats = benchmark(
        q,
        k,
        v,
        args.bits,
        args.bits,
        args.max_suffix_length,
        args.repeats,
    )

    print("max_suffix_length,time_ms,states,edges,values")
    print(
        f"{args.max_suffix_length},{elapsed * 1000:.3f},"
        f"{stats[0]},{stats[1]},{stats[2]}"
    )


if __name__ == "__main__":
    main()
