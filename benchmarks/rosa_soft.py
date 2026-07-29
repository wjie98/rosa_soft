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
from rosa_soft.soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)


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
    layout = getattr(args, "layout", "dense")
    if layout == "varlen":
        total_tokens = args.batch * seq_len
        qk_shape = (total_tokens, args.heads, args.bits)
        segment_length = min(args.segment_length, seq_len)
        offsets = [0]
        for batch_index in range(args.batch):
            batch_start = batch_index * seq_len
            for end in range(
                segment_length,
                seq_len,
                segment_length,
            ):
                offsets.append(batch_start + end)
            offsets.append(batch_start + seq_len)
        cu_seqlens = torch.tensor(
            offsets,
            device=device,
            dtype=torch.int32,
        )
    else:
        qk_shape = (args.batch, seq_len, args.heads, args.bits)
        cu_seqlens = None
    gradients = getattr(args, "gradients", "qkv")
    needs_query = "q" in gradients
    needs_key = "k" in gradients
    needs_value = "v" in gradients
    if args.pattern == "all-match":
        query = torch.ones(
            qk_shape,
            device=device,
            dtype=dtype,
            requires_grad=needs_query,
        )
        key = torch.ones_like(query, requires_grad=needs_key)
    else:
        query = torch.randn(
            qk_shape,
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=needs_query,
        )
        key = torch.randn(
            qk_shape,
            device=device,
            dtype=dtype,
            generator=generator,
            requires_grad=needs_key,
        )
    value_shape = (
        (args.batch * seq_len, args.value_heads, args.value_dim)
        if layout == "varlen"
        else (
            args.batch,
            seq_len,
            args.value_heads,
            args.value_dim,
        )
    )
    value = torch.randn(
        value_shape,
        device=device,
        dtype=dtype,
        generator=generator,
        requires_grad=needs_value,
    )
    grad_output_shape = (
        (args.batch * seq_len, args.heads, args.value_dim)
        if layout == "varlen"
        else (args.batch, seq_len, args.heads, args.value_dim)
    )
    grad_output = torch.randn(
        grad_output_shape,
        device=device,
        dtype=dtype,
        generator=generator,
    )

    def run_operator() -> Tensor:
        positional = (
            (query, key, value, cu_seqlens)
            if cu_seqlens is not None
            else (query, key, value)
        )
        return operator(
            *positional,
            max_suffix_length=args.max_suffix_length,
            scale=args.scale,
            dropout_p=args.dropout_p,
            mismatch_scale=args.mismatch_scale,
        )

    def train_step() -> None:
        _clear_gradients(query, key, value)
        output = run_operator()
        output.backward(grad_output)

    def forward_step() -> None:
        with torch.no_grad():
            run_operator()

    step = train_step if args.mode == "train" else forward_step
    for _ in range(args.warmup):
        step()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    _clear_gradients(query, key, value)
    gc.collect()
    peak_mib = float("nan")
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        start.record(stream)
        for _ in range(args.repeats):
            step()
        end.record(stream)
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
        "layout": layout,
        "segments": (
            int(cu_seqlens.numel() - 1)
            if cu_seqlens is not None
            else args.batch
        ),
        "mode": args.mode,
        "pattern": args.pattern,
        "gradients": gradients,
        "dropout_p": args.dropout_p,
        "step_ms": step_ms,
        "peak_operator_mib": peak_mib,
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else str(device)
        ),
        "compute_capability": (
            ".".join(
                str(component)
                for component in torch.cuda.get_device_capability(device)
            )
            if device.type == "cuda"
            else None
        ),
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
    parser.add_argument(
        "--layout",
        choices=["dense", "varlen"],
        default="dense",
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=64,
        help="maximum packed segment length for --layout varlen",
    )
    parser.add_argument(
        "--gradients",
        choices=["q", "k", "v", "qk", "qv", "kv", "qkv"],
        default="qkv",
        help="inputs that require gradients in train mode; v is value",
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
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-dim", type=int, default=8)
    parser.add_argument("--max-suffix-length", type=int, default=32)
    parser.add_argument(
        "--scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_SCALE,
    )
    parser.add_argument(
        "--dropout-p",
        type=float,
        default=ROSA_SOFT_DEFAULT_DROPOUT_P,
    )
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.segment_length < 1:
        raise ValueError("--segment-length must be >= 1")
    if args.heads % args.value_heads != 0:
        raise ValueError("--heads must be divisible by --value-heads")
    if args.operator in {"cuda", "both"}:
        if torch.device(args.device).type != "cuda":
            raise ValueError("CUDA operator requires --device cuda")
        if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
            raise RuntimeError("RosaSoft CUDA extension is unavailable")

    operators = {}
    if args.operator in {"cuda", "both"}:
        operators["cuda"] = (
            rosa_soft.rosa_soft_varlen
            if args.layout == "varlen"
            else rosa_soft.rosa_soft
        )
    if args.operator in {"reference", "both"}:
        operators["reference"] = (
            rosa_soft.rosa_soft_varlen_reference
            if args.layout == "varlen"
            else rosa_soft.rosa_soft_reference
        )

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
