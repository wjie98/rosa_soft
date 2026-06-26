import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rosa_soft import RosaRuntime


def bench_backend(backend, q, k, v, qk_bits, value_bits, repeats):
    timings = []
    stats = None
    for _ in range(repeats):
        rt = RosaRuntime(
            num_heads=q.size(-1),
            num_value_heads=v.size(-1),
            qk_bits=qk_bits,
            value_bits=value_bits,
            backend=backend,
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
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()

    torch.manual_seed(123)
    q = torch.randint(0, 1 << args.bits, (args.B, args.T, args.H), dtype=torch.uint8)
    k = torch.randint(0, 1 << args.bits, (args.B, args.T, args.H), dtype=torch.uint8)
    v = torch.randint(0, 1 << args.bits, (args.B, args.T, args.Hv), dtype=torch.uint8)

    map_time, map_stats = bench_backend("map", q, k, v, args.bits, args.bits, args.repeats)
    compact_time, compact_stats = bench_backend("compact", q, k, v, args.bits, args.bits, args.repeats)

    print("backend,time_ms,states,edges,values")
    print(f"map,{map_time * 1000:.3f},{map_stats[0]},{map_stats[1]},{map_stats[2]}")
    print(f"compact,{compact_time * 1000:.3f},{compact_stats[0]},{compact_stats[1]},{compact_stats[2]}")
    print(f"speedup,{map_time / compact_time:.3f}x,,,")


if __name__ == "__main__":
    main()
