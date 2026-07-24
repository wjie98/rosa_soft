"""Standalone time and memory probe for RosaSoft training operators."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path
from typing import Callable, Dict

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def _clear_gradients(*tensors: Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def benchmark(
    operator: Callable[..., Tensor],
    *,
    seq_len: int,
    args: argparse.Namespace,
) -> Dict[str, object]:
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed + seq_len)
    qk_shape = (args.batch, seq_len, args.heads, args.bits)
    if args.pattern == "all-match":
        query = torch.ones(
            qk_shape,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        key = torch.ones_like(query, requires_grad=True)
    else:
        query = torch.randn(
            qk_shape,
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
        key = torch.randn_like(query, requires_grad=True)
    payload = torch.randn(
        args.batch,
        seq_len,
        args.payload_heads,
        args.payload_dim,
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    grad_output = torch.randn(
        args.batch,
        seq_len,
        args.heads,
        args.payload_dim,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    def train_step() -> None:
        _clear_gradients(query, key, payload)
        output = operator(
            query,
            key,
            payload,
            max_suffix_length=args.max_suffix_length,
            route_temperature=args.route_temperature,
            mismatch_penalty=args.mismatch_penalty,
        )
        output.backward(grad_output)

    def forward_step() -> None:
        with torch.no_grad():
            operator(
                query,
                key,
                payload,
                max_suffix_length=args.max_suffix_length,
                route_temperature=args.route_temperature,
                mismatch_penalty=args.mismatch_penalty,
            )

    step = train_step if args.mode == "train" else forward_step
    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    _clear_gradients(query, key, payload)
    gc.collect()
    peak_mib = float("nan")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(args.repeats):
            step()
        end.record()
        end.synchronize()
        step_ms = start.elapsed_time(end) / args.repeats
        peak_mib = (
            torch.cuda.max_memory_allocated(device) - baseline
        ) / (1024 * 1024)
    else:
        import time

        started = time.perf_counter()
        for _ in range(args.repeats):
            step()
        step_ms = (time.perf_counter() - started) * 1000 / args.repeats

    return {
        "sequence_length": seq_len,
        "mode": args.mode,
        "pattern": args.pattern,
        "step_ms": step_ms,
        "peak_operator_mib": peak_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--operator",
        choices=["cuda", "reference", "both"],
        default="cuda",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "forward"],
        default="train",
    )
    parser.add_argument(
        "--pattern",
        choices=["random", "all-match"],
        default="random",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[64, 128, 256, 512],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--payload-heads", type=int, default=2)
    parser.add_argument("--payload-dim", type=int, default=8)
    parser.add_argument("--max-suffix-length", type=int, default=32)
    parser.add_argument(
        "--route-temperature",
        type=float,
        default=rosa_soft.ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    )
    parser.add_argument(
        "--mismatch-penalty",
        type=float,
        default=rosa_soft.ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.heads % args.payload_heads != 0:
        raise ValueError("--heads must be divisible by --payload-heads")
    if args.operator in {"cuda", "both"}:
        if torch.device(args.device).type != "cuda":
            raise ValueError("CUDA operator requires --device cuda")
        if not rosa_soft.HAS_ROSA_SOFT_CUDA:
            raise RuntimeError("RosaSoft CUDA extension is unavailable")

    operators = {}
    if args.operator in {"cuda", "both"}:
        operators["cuda"] = rosa_soft.rosa_soft
    if args.operator in {"reference", "both"}:
        operators["reference"] = rosa_soft.rosa_soft_reference

    results = {
        name: [
            benchmark(operator, seq_len=seq_len, args=args)
            for seq_len in args.sequence_lengths
        ]
        for name, operator in operators.items()
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
