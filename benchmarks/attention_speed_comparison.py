"""Compare production RosaSoft, fixed SUFA, and PyTorch Flash SDPA."""

from __future__ import annotations

import argparse
import gc
import json
import statistics
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.attention import SDPBackend, sdpa_kernel

import rosa_soft
from benchmarks.historical_suffix_attention_speed import (
    _make_endpos,
    historical_rosa_sufa,
)


Operation = Callable[[], None]


def _measure_ms(operation: Operation, *, warmup: int, repeats: int) -> float:
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
    return start.elapsed_time(end) / repeats


def _measure_rotating(
    operations: dict[str, Operation],
    *,
    warmup: int,
    repeats: int,
    rounds: int,
) -> dict[str, list[float]]:
    names = list(operations)
    samples = {name: [] for name in names}
    for round_index in range(rounds):
        offset = round_index % len(names)
        order = names[offset:] + names[:offset]
        if round_index & 1:
            order.reverse()
        for name in order:
            samples[name].append(
                _measure_ms(
                    operations[name],
                    warmup=warmup,
                    repeats=repeats,
                )
            )
    return samples


def _peak_memory_mib(operation: Operation) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    operation()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated()
    return (peak - baseline) / (1024 * 1024)


def _clear_gradients(*tensors: Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def _random_tensor(
    shape: tuple[int, ...],
    *,
    dtype: torch.dtype,
    generator: torch.Generator,
    requires_grad: bool = False,
) -> Tensor:
    return torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
        generator=generator,
        requires_grad=requires_grad,
    )


def _case(sequence_length: int, args: argparse.Namespace) -> dict[str, object]:
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device="cuda").manual_seed(
        args.seed + sequence_length
    )
    query = _random_tensor(
        (args.batch, sequence_length, args.heads, args.bits),
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    key = _random_tensor(
        tuple(query.shape),
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    value = _random_tensor(
        (
            args.batch,
            sequence_length,
            args.value_heads,
            args.value_dim,
        ),
        dtype=dtype,
        generator=generator,
        requires_grad=True,
    )
    grad_output = _random_tensor(
        (args.batch, sequence_length, args.heads, args.value_dim),
        dtype=dtype,
        generator=generator,
    )
    endpos = _make_endpos(
        query.detach(),
        key.detach(),
        args.value_heads,
        args.suffix_window,
    )

    flash_inputs: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
    for head_dim in args.flash_head_dims:
        flash_query = _random_tensor(
            (args.batch, args.heads, sequence_length, head_dim),
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
        flash_key = _random_tensor(
            (args.batch, args.value_heads, sequence_length, head_dim),
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
        flash_value = _random_tensor(
            tuple(flash_key.shape),
            dtype=dtype,
            generator=generator,
            requires_grad=True,
        )
        flash_grad = _random_tensor(
            (args.batch, args.heads, sequence_length, head_dim),
            dtype=dtype,
            generator=generator,
        )
        flash_inputs[head_dim] = (
            flash_query,
            flash_key,
            flash_value,
            flash_grad,
        )

    def production_train() -> None:
        _clear_gradients(query, key, value)
        output = rosa_soft.rosa_soft(
            query,
            key,
            value,
            max_suffix_length=args.suffix_window,
            scale=1.0,
            dropout_p=0.0,
            mismatch_scale=3.0,
        )
        output.backward(grad_output)

    def suffix_train() -> None:
        _clear_gradients(query, key, value)
        output = historical_rosa_sufa(
            query,
            key,
            value,
            endpos,
            suffix_window=args.suffix_window,
            suffix_factor=0.5,
        )
        output.backward(grad_output)

    def production_forward() -> None:
        with torch.no_grad():
            torch.ops.rosa_soft.hard_forward(
                query,
                key,
                value,
            )

    def make_flash_train(head_dim: int) -> Operation:
        flash_query, flash_key, flash_value, flash_grad = flash_inputs[head_dim]

        def train() -> None:
            _clear_gradients(flash_query, flash_key, flash_value)
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                output = F.scaled_dot_product_attention(
                    flash_query,
                    flash_key,
                    flash_value,
                    is_causal=True,
                    enable_gqa=args.heads != args.value_heads,
                )
            output.backward(flash_grad)

        return train

    def make_flash_forward(head_dim: int) -> Operation:
        flash_query, flash_key, flash_value, _ = flash_inputs[head_dim]

        def forward() -> None:
            with torch.no_grad(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                F.scaled_dot_product_attention(
                    flash_query,
                    flash_key,
                    flash_value,
                    is_causal=True,
                    enable_gqa=args.heads != args.value_heads,
                )

        return forward

    train_operations = {
        "rosa_production": production_train,
        "fixed_suffix_sdpa": suffix_train,
    }
    forward_operations = {"rosa_hard": production_forward}
    all_inputs: dict[str, tuple[Tensor, ...]] = {
        "rosa_production": (query, key, value),
        "fixed_suffix_sdpa": (query, key, value),
    }
    for head_dim, tensors in flash_inputs.items():
        name = f"flash_gqa_d{head_dim}"
        train_operations[name] = make_flash_train(head_dim)
        forward_operations[name] = make_flash_forward(head_dim)
        all_inputs[name] = tensors[:3]

    train_samples = _measure_rotating(
        train_operations,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )
    forward_samples = _measure_rotating(
        forward_operations,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )
    operators = {}
    for name, operation in train_operations.items():
        train_ms = statistics.median(train_samples[name])
        forward_name = (
            "rosa_hard"
            if name in {"rosa_production", "fixed_suffix_sdpa"}
            else name
        )
        forward_ms = (
            statistics.median(forward_samples[forward_name])
            if forward_name in forward_samples
            else None
        )
        _clear_gradients(*all_inputs[name])
        peak_mib = _peak_memory_mib(operation)
        _clear_gradients(*all_inputs[name])
        operators[name] = {
            "train_ms": train_ms,
            "forward_ms": forward_ms,
            "peak_operator_mib": peak_mib,
            "train_samples_ms": train_samples[name],
            "forward_samples_ms": (
                forward_samples.get(forward_name, [])
            ),
        }
    production_ms = operators["rosa_production"]["train_ms"]
    for metrics in operators.values():
        metrics["production_train_ratio"] = production_ms / metrics["train_ms"]
    return {"sequence_length": sequence_length, "operators": operators}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[512, 1024, 2048, 4096, 8192],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--suffix-window", type=int, default=32)
    parser.add_argument(
        "--flash-head-dims",
        type=int,
        nargs="+",
        default=[8, 64],
    )
    parser.add_argument(
        "--dtype",
        choices=["float16", "bfloat16"],
        default="float16",
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.heads % args.value_heads:
        raise ValueError("--heads must be divisible by --value-heads")
    if args.bits * args.suffix_window > 256:
        raise ValueError("fixed SUFA requires bits * suffix_window <= 256")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
        raise RuntimeError("RosaSoft CUDA extension is unavailable")

    cases = []
    for sequence_length in args.sequence_lengths:
        case = _case(sequence_length, args)
        cases.append(case)
        summary = {
            name: round(metrics["train_ms"], 4)
            for name, metrics in case["operators"].items()
        }
        print(f"T={sequence_length}: {summary}", flush=True)
    report = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": torch.cuda.get_device_capability(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "settings": vars(args) | {"output": str(args.output)},
        "comparison_notes": [
            "RosaSoft and fixed SUFA share D-bit Q/K and Dv-wide grouped V.",
            "Fixed SUFA uses automatic SDPA with Q/K width D*W and V width Dv.",
            "Flash controls force PyTorch FLASH_ATTENTION and require equal Q/K/V head widths.",
            "Flash d=bits matches Q/K width but not output width; flash d=Dv matches output width but not ROSA symbol width.",
            "All training rows include forward and Q/K/V backward but exclude learned projections.",
        ],
        "cases": cases,
    }
    encoded = json.dumps(report, indent=2, default=str) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    else:
        print(encoded, end="")


if __name__ == "__main__":
    main()
