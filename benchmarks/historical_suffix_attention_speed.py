"""Compare RosaSoft with the historical fixed-window SUFA surrogate.

The historical path reproduces ``suffix_attention_proxy`` from commit
``c776101``.  Both training operators use the current exact hard CUDA forward,
so the measurement isolates the old and current backward estimators instead of
mixing in changes to the historical CPU runtime.  Exact hard route end
positions are prepared before timing and are used only by the historical gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft


HISTORICAL_COMMIT = "c776101"


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "float32": torch.float32,
    }[name]


def _hard_softsign(tensor: Tensor) -> Tensor:
    soft = F.softsign(tensor)
    hard = torch.where(soft > 0, 1.0, -1.0)
    return (hard - soft).detach() + soft


def historical_suffix_attention_proxy(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    endpos: Tensor,
    *,
    suffix_window: int,
    suffix_factor: float = 0.5,
    scale: float | None = None,
) -> Tensor:
    """Reproduce the fixed-window proxy from historical commit c776101."""

    batch, seq_len, query_heads, query_bits = query.shape
    key_heads, key_bits = key.shape[2:]
    value_heads, value_dim = value.shape[2:]
    if query_heads != key_heads or query_bits != key_bits:
        raise ValueError("historical SUFA requires equal query/key shapes")
    if query_heads % value_heads:
        raise ValueError("query heads must be divisible by value heads")
    head_dim = query_bits * suffix_window
    if not 0 < head_dim <= 256:
        raise ValueError("historical SUFA requires 0 < bits * window <= 256")
    if not 0.0 < suffix_factor <= 1.0:
        raise ValueError("suffix_factor must be in (0, 1]")

    xq = _hard_softsign(query.permute(0, 2, 1, 3))
    xk = _hard_softsign(key.permute(0, 2, 1, 3))
    xv = _hard_softsign(value.permute(0, 2, 1, 3))

    repeats = query_heads // value_heads
    if repeats > 1:
        xv = xv[:, :, None, :, :].expand(
            batch,
            value_heads,
            repeats,
            seq_len,
            value_dim,
        )
        xv = xv.reshape(batch, query_heads, seq_len, value_dim)

    xq = F.pad(xq, (0, 0, suffix_window - 1, 0))
    xq = xq.unfold(-2, suffix_window, 1).transpose(-2, -1)
    xk = F.pad(xk, (0, 0, suffix_window, -1))
    xk = xk.unfold(-2, suffix_window, 1).transpose(-2, -1)

    offsets = suffix_window - 1 - torch.arange(
        suffix_window,
        device=query.device,
    )
    decay = suffix_factor ** offsets.float()
    decay = torch.sqrt(decay / decay.sum()).view(-1, 1)
    xq = (xq * decay.to(xq.dtype)).reshape(
        batch,
        query_heads,
        seq_len,
        head_dim,
    )
    xk = (xk * decay.to(xk.dtype)).reshape(
        batch,
        key_heads,
        seq_len,
        head_dim,
    )

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)
    output = F.scaled_dot_product_attention(
        xq,
        xk,
        xv,
        scale=float(scale),
        is_causal=True,
    )

    route_end = endpos.permute(0, 2, 1)
    rows = torch.arange(seq_len, device=endpos.device).view(1, 1, seq_len)
    gather_index = torch.where(route_end >= 0, route_end + 1, rows)
    gather_index = gather_index.unsqueeze(-1)
    gate_mask = (route_end >= suffix_window).unsqueeze(-1).to(xq.dtype)
    routed_key = torch.gather(xk, 2, gather_index.expand_as(xk))
    routed_value = torch.gather(xv, 2, gather_index.expand_as(xv))
    gate = torch.sigmoid(
        torch.sum(xq * routed_key, dim=-1, keepdim=True)
        * (6.0 / head_dim)
    )
    output = output * (1.0 - gate * gate_mask)
    output = output + routed_value * gate * gate_mask
    return output.permute(0, 2, 1, 3)


class _HistoricalSufaFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        endpos: Tensor,
        suffix_window: int,
        suffix_factor: float,
    ) -> Tensor:
        hard_output = torch.ops.rosa_soft.hard_forward(
            query,
            key,
            value,
        )[0]
        ctx.suffix_window = int(suffix_window)
        ctx.suffix_factor = float(suffix_factor)
        ctx.save_for_backward(
            query.detach(),
            key.detach(),
            value.detach(),
            endpos,
        )
        return hard_output

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        saved_query, saved_key, saved_value, endpos = ctx.saved_tensors
        query = saved_query.requires_grad_(True)
        key = saved_key.requires_grad_(True)
        value = saved_value.requires_grad_(True)
        with torch.enable_grad():
            soft_output = historical_suffix_attention_proxy(
                query,
                key,
                value,
                endpos,
                suffix_window=ctx.suffix_window,
                suffix_factor=ctx.suffix_factor,
            )
        gradients = torch.autograd.grad(
            soft_output,
            (query, key, value),
            grad_output,
            retain_graph=False,
            create_graph=False,
        )
        return *gradients, None, None, None


def historical_rosa_sufa(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    endpos: Tensor,
    *,
    suffix_window: int,
    suffix_factor: float,
) -> Tensor:
    return _HistoricalSufaFunction.apply(
        query,
        key,
        value,
        endpos,
        suffix_window,
        suffix_factor,
    )


def _make_endpos(
    query: Tensor,
    key: Tensor,
    value_heads: int,
    suffix_window: int,
) -> Tensor:
    del value_heads, suffix_window
    query_heads = query.size(2)
    bits = query.size(3)
    return rosa_soft.RosaSam(query_heads, bits).update(query, key)


def _measure_cuda(
    step: Callable[[], None],
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        step()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def _peak_memory_mib(step: Callable[[], None], device: torch.device) -> float:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    baseline = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    step()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return (peak - baseline) / (1024 * 1024)


def _case(
    *,
    seq_len: int,
    suffix_window: int,
    args: argparse.Namespace,
) -> dict[str, object]:
    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    generator = torch.Generator(device=device).manual_seed(
        args.seed + seq_len * 131 + suffix_window
    )
    qk_shape = (args.batch, seq_len, args.heads, args.bits)
    if args.pattern == "all-match":
        query = torch.ones(qk_shape, device=device, dtype=dtype)
        key = torch.ones_like(query)
    else:
        query = torch.randn(
            qk_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        key = torch.randn(
            qk_shape,
            device=device,
            dtype=dtype,
            generator=generator,
        )
    value = torch.randn(
        (args.batch, seq_len, args.value_heads, args.value_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    query.requires_grad_("q" in args.gradients)
    key.requires_grad_("k" in args.gradients)
    value.requires_grad_("v" in args.gradients)
    grad_output = torch.randn(
        (args.batch, seq_len, args.heads, args.value_dim),
        device=device,
        dtype=dtype,
        generator=generator,
    )
    endpos = _make_endpos(
        query.detach(),
        key.detach(),
        args.value_heads,
        suffix_window,
    )

    def clear_gradients() -> None:
        query.grad = None
        key.grad = None
        value.grad = None

    def current_train() -> None:
        clear_gradients()
        output = rosa_soft.rosa_soft(
            query,
            key,
            value,
            max_suffix_length=suffix_window,
            scale=args.current_scale,
            dropout_p=0.0,
            mismatch_scale=args.mismatch_scale,
        )
        output.backward(grad_output)

    def historical_train() -> None:
        clear_gradients()
        output = historical_rosa_sufa(
            query,
            key,
            value,
            endpos,
            suffix_window=suffix_window,
            suffix_factor=args.suffix_factor,
        )
        output.backward(grad_output)

    def hard_forward() -> None:
        with torch.no_grad():
            torch.ops.rosa_soft.hard_forward(
                query,
                key,
                value,
            )

    current_samples: list[float] = []
    historical_samples: list[float] = []
    operators = {
        "current": (current_train, current_samples),
        "historical": (historical_train, historical_samples),
    }
    for round_index in range(args.rounds):
        order = (
            ("current", "historical")
            if round_index % 2 == 0
            else ("historical", "current")
        )
        for name in order:
            function, samples = operators[name]
            samples.append(
                _measure_cuda(
                    function,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
            )

    hard_samples = [
        _measure_cuda(
            hard_forward,
            warmup=args.warmup,
            repeats=args.repeats,
        )
        for _ in range(args.rounds)
    ]
    current_ms = statistics.median(current_samples)
    historical_ms = statistics.median(historical_samples)
    hard_ms = statistics.median(hard_samples)
    clear_gradients()
    current_memory = _peak_memory_mib(current_train, device)
    clear_gradients()
    historical_memory = _peak_memory_mib(historical_train, device)
    clear_gradients()

    return {
        "sequence_length": seq_len,
        "suffix_window": suffix_window,
        "current_train_ms": current_ms,
        "historical_train_ms": historical_ms,
        "current_over_historical": current_ms / historical_ms,
        "hard_forward_ms": hard_ms,
        "current_backward_increment_ms": max(0.0, current_ms - hard_ms),
        "historical_backward_increment_ms": max(
            0.0,
            historical_ms - hard_ms,
        ),
        "current_peak_mib": current_memory,
        "historical_peak_mib": historical_memory,
        "current_samples_ms": current_samples,
        "historical_samples_ms": historical_samples,
        "hard_samples_ms": hard_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--dtype",
        choices=["float16", "float32"],
        default="float32",
    )
    parser.add_argument(
        "--sequence-lengths",
        type=int,
        nargs="+",
        default=[128, 256, 512, 1024, 2048],
    )
    parser.add_argument(
        "--suffix-windows",
        type=int,
        nargs="+",
        default=[32],
    )
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument(
        "--gradients",
        choices=["q", "k", "v", "qk", "qv", "kv", "qkv"],
        default="qkv",
    )
    parser.add_argument(
        "--pattern",
        choices=["random", "all-match"],
        default="random",
    )
    parser.add_argument("--current-scale", type=float, default=1.0)
    parser.add_argument("--mismatch-scale", type=float, default=3.0)
    parser.add_argument("--suffix-factor", type=float, default=0.5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if torch.device(args.device).type != "cuda":
        raise ValueError("this benchmark requires CUDA")
    if not rosa_soft.BUILD_CAPABILITIES.rosa_soft_cuda:
        raise RuntimeError("RosaSoft CUDA extension is unavailable")
    if args.heads % args.value_heads:
        raise ValueError("--heads must be divisible by --value-heads")
    if not 1 <= args.bits <= 8:
        raise ValueError("--bits must be in [1, 8]")
    for suffix_window in args.suffix_windows:
        if args.bits * suffix_window > 256:
            raise ValueError(
                "historical SUFA requires bits * suffix_window <= 256"
            )

    device = torch.device(args.device)
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
        args.device = str(device)
    torch.cuda.set_device(device)
    cases = []
    for suffix_window in args.suffix_windows:
        for seq_len in args.sequence_lengths:
            result = _case(
                seq_len=seq_len,
                suffix_window=suffix_window,
                args=args,
            )
            cases.append(result)
            print(
                f"T={seq_len:5d} W={suffix_window:3d} "
                f"current={result['current_train_ms']:.3f} ms "
                f"historical={result['historical_train_ms']:.3f} ms "
                f"ratio={result['current_over_historical']:.2f}x",
                file=sys.stderr,
                flush=True,
            )

    report = {
        "historical_source_commit": HISTORICAL_COMMIT,
        "comparison": (
            "shared current exact hard forward; historical fixed-window "
            "SUFA versus current RosaSoft backward"
        ),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": ".".join(
            str(value) for value in torch.cuda.get_device_capability(device)
        ),
        "settings": vars(args),
        "cases": cases,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
