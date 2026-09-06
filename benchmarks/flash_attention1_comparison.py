"""Compare unlimited RosaSoft with the official FlashAttention 1.x kernel."""

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

import flash_attn
from flash_attn.flash_attn_interface import (
    flash_attn_unpadded_qkvpacked_func,
)

import rosa_soft


Operation = Callable[[], None]


def _measure_ms(
    operation: Operation,
    *,
    warmup: int,
    repeats: int,
) -> float:
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
) -> tuple[dict[str, float], dict[str, list[float]]]:
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
    medians = {
        name: statistics.median(values) for name, values in samples.items()
    }
    return medians, samples


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


def _numeric_check(
    *,
    heads: int,
    head_dim: int,
    sequence_length: int,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(seed)
    qkv = torch.randn(
        sequence_length,
        3,
        heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
        requires_grad=True,
    )
    cu_seqlens = torch.tensor(
        [0, sequence_length], device="cuda", dtype=torch.int32
    )
    grad_output = torch.randn(
        sequence_length,
        heads,
        head_dim,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    output = flash_attn_unpadded_qkvpacked_func(
        qkv,
        cu_seqlens,
        sequence_length,
        0.0,
        causal=True,
    )
    output.backward(grad_output)

    reference_qkv = qkv.detach().float().requires_grad_(True)
    query, key, value = [
        reference_qkv[:, index].permute(1, 0, 2).unsqueeze(0)
        for index in range(3)
    ]
    logits = torch.matmul(query, key.transpose(-2, -1)) * head_dim**-0.5
    causal_mask = torch.ones(
        sequence_length,
        sequence_length,
        device="cuda",
        dtype=torch.bool,
    ).tril_()
    logits = logits.masked_fill(~causal_mask, float("-inf"))
    probability = F.softmax(logits, dim=-1)
    reference_output = torch.matmul(probability, value)
    reference_output = reference_output.squeeze(0).permute(1, 0, 2)
    reference_output.backward(grad_output.float())

    output_error = (output.float() - reference_output.detach()).abs()
    gradient_error = (qkv.grad.float() - reference_qkv.grad).abs()
    return {
        "seq_len": sequence_length,
        "output_max_abs_error": output_error.max().item(),
        "output_mean_abs_error": output_error.mean().item(),
        "gradient_max_abs_error": gradient_error.max().item(),
        "gradient_mean_abs_error": gradient_error.mean().item(),
        "all_finite": bool(
            torch.isfinite(output).all() and torch.isfinite(qkv.grad).all()
        ),
    }


def _case(sequence_length: int, args: argparse.Namespace) -> dict[str, object]:
    generator = torch.Generator(device="cuda").manual_seed(
        args.seed + sequence_length
    )
    query = torch.randn(
        args.batch,
        sequence_length,
        args.heads,
        args.bits,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
        requires_grad=True,
    )
    key = torch.randn_like(query, generator=generator, requires_grad=True)
    value = torch.randn(
        args.batch,
        sequence_length,
        args.value_heads,
        args.head_dim,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
        requires_grad=True,
    )
    rosa_grad_output = torch.randn(
        args.batch,
        sequence_length,
        args.heads,
        args.head_dim,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )

    flash_qkv = torch.randn(
        args.batch * sequence_length,
        3,
        args.heads,
        args.head_dim,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
        requires_grad=True,
    )
    flash_grad_output = torch.randn(
        args.batch * sequence_length,
        args.heads,
        args.head_dim,
        device="cuda",
        dtype=torch.float16,
        generator=generator,
    )
    cu_seqlens = torch.arange(
        0,
        (args.batch + 1) * sequence_length,
        sequence_length,
        device="cuda",
        dtype=torch.int32,
    )

    def rosa_train() -> None:
        _clear_gradients(query, key, value)
        output = rosa_soft.rosa_soft_unbounded(
            query,
            key,
            value,
            scale=args.scale,
            dropout_p=0.0,
            mismatch_scale=args.mismatch_scale,
        )
        output.backward(rosa_grad_output)

    def flash_train() -> None:
        _clear_gradients(flash_qkv)
        output = flash_attn_unpadded_qkvpacked_func(
            flash_qkv,
            cu_seqlens,
            sequence_length,
            0.0,
            causal=True,
        )
        output.backward(flash_grad_output)

    def rosa_forward() -> None:
        with torch.no_grad():
            rosa_soft.rosa_soft_unbounded(query, key, value)

    def flash_forward() -> None:
        with torch.no_grad():
            flash_attn_unpadded_qkvpacked_func(
                flash_qkv,
                cu_seqlens,
                sequence_length,
                0.0,
                causal=True,
            )

    train_operations = {
        "rosa_unbounded": rosa_train,
        "flash_attention_1": flash_train,
    }
    forward_operations = {
        "rosa_hard": rosa_forward,
        "flash_attention_1": flash_forward,
    }
    train_medians, train_samples = _measure_rotating(
        train_operations,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )
    forward_medians, forward_samples = _measure_rotating(
        forward_operations,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.rounds,
    )

    train_memory = {}
    for name, operation in train_operations.items():
        train_memory[name] = _peak_memory_mib(operation)
    _clear_gradients(query, key, value, flash_qkv)

    return {
        "seq_len": sequence_length,
        "train": {
            name: {
                "latency_ms": train_medians[name],
                "latency_samples_ms": train_samples[name],
                "peak_operator_mib": train_memory[name],
            }
            for name in train_operations
        },
        "forward": {
            name: {
                "latency_ms": forward_medians[name],
                "latency_samples_ms": forward_samples[name],
            }
            for name in forward_operations
        },
        "train_latency_ratio_rosa_over_flash1": (
            train_medians["rosa_unbounded"]
            / train_medians["flash_attention_1"]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lengths", type=int, nargs="+", default=[2048, 4096, 8192]
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=73123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if args.head_dim not in {32, 64, 128}:
        parser.error("FlashAttention 1 comparison requires head_dim 32, 64, or 128")
    if args.heads % args.value_heads:
        parser.error("heads must be divisible by value_heads")

    rows = []
    for sequence_length in args.lengths:
        row = _case(sequence_length, args)
        rows.append(row)
        print(json.dumps(row, sort_keys=True))

    result = {
        "device": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "flash_attn": flash_attn.__version__,
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "comparison_contract": {
            "rosa": "B,T,Hq,D=bits Q/K; B,T,Hv,Dv=head_dim V",
            "flash_attention_1": "unpadded QKV-packed B*T,3,Hq,head_dim",
            "shared": "FP16, causal, dropout_p=0, same output shape",
            "caveat": (
                "The operators have different Q/K widths, value-head counts, "
                "forward semantics, and backward equations."
            ),
        },
        "flash1_numeric_check": _numeric_check(
            heads=args.heads,
            head_dim=args.head_dim,
            sequence_length=127,
            seed=args.seed + 1,
        ),
        "rows": rows,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
