"""Profile production, exact-bitflip, and quadratic-state ROSA VJPs."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

import torch
from torch import Tensor


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import rosa_soft  # noqa: E402
from benchmarks.estimator_fit_ablation import (  # noqa: E402
    rosa_soft_exact_bitflip,
)
from benchmarks.fast_weight_proxy import rosa_fast_weight_proxy  # noqa: E402
from rosa_soft.soft_contract import (  # noqa: E402
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
)
from rosa_soft.soft_reference import rosa_soft_reference  # noqa: E402


ESTIMATORS = ("production", "bitflip", "state_quadratic_attention")
Estimator = Callable[[Tensor, Tensor, Tensor], Tensor]


def _parse_case(encoded: str) -> tuple[int, int]:
    try:
        batch, sequence_length = (int(part) for part in encoded.split(":"))
    except (TypeError, ValueError) as error:
        raise argparse.ArgumentTypeError("cases must use B:T") from error
    if batch < 1 or sequence_length < 2:
        raise argparse.ArgumentTypeError("case batch must be >= 1 and T >= 2")
    return batch, sequence_length


def _operator(estimator: str, args: argparse.Namespace) -> Estimator:
    common = {
        "max_suffix_length": args.max_suffix_length,
        "scale": args.scale,
        "mismatch_scale": args.mismatch_scale,
    }
    if estimator == "production":
        production = (
            rosa_soft.rosa_soft
            if torch.device(args.device).type == "cuda"
            else rosa_soft_reference
        )

        def call(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
            return production(query, key, value, dropout_p=0.0, **common)

        return call
    if estimator == "bitflip":

        def call(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
            return rosa_soft_exact_bitflip(query, key, value, **common)

        return call

    def call(query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        return rosa_fast_weight_proxy(
            query,
            key,
            value,
            proxy="state_quadratic_attention",
            fingerprint_length=1,
            max_suffix_length=args.max_suffix_length,
            mismatch_scale=args.mismatch_scale,
        )

    return call


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _profile_estimator(
    estimator: str,
    tensors: tuple[Tensor, Tensor, Tensor],
    upstream: Tensor,
    args: argparse.Namespace,
) -> tuple[Tensor, dict[str, object]]:
    device = tensors[0].device
    call = _operator(estimator, args)

    def vjp() -> Tensor:
        output = call(*tensors)
        torch.autograd.grad(output, tensors, upstream)
        return output

    output = vjp().detach()
    for _ in range(args.warmup):
        vjp()
    _synchronize(device)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    baseline_allocated = (
        torch.cuda.memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    started = time.perf_counter()
    for _ in range(args.iterations):
        vjp()
    _synchronize(device)
    elapsed_ms = (time.perf_counter() - started) * 1000.0 / args.iterations
    peak_allocated = (
        torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None
    )
    return output, {
        "estimator": estimator,
        "vjp_ms": elapsed_ms,
        "baseline_allocated_mib": (
            baseline_allocated / 2**20
            if baseline_allocated is not None
            else None
        ),
        "peak_allocated_mib": (
            peak_allocated / 2**20
            if peak_allocated is not None
            else None
        ),
        "peak_incremental_mib": (
            (peak_allocated - baseline_allocated) / 2**20
            if peak_allocated is not None
            else None
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / 2**20
            if device.type == "cuda"
            else None
        ),
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    records = []
    for case_index, (batch, sequence_length) in enumerate(args.cases):
        generator = torch.Generator(device="cpu").manual_seed(
            args.seed + case_index
        )
        shapes = (
            (batch, sequence_length, args.heads, args.qk_bits),
            (batch, sequence_length, args.heads, args.qk_bits),
            (batch, sequence_length, args.value_heads, args.value_bits),
        )
        tensors = tuple(
            torch.randn(shape, generator=generator).to(device).requires_grad_()
            for shape in shapes
        )
        upstream = torch.randn(
            batch,
            sequence_length,
            args.heads,
            args.value_bits,
            generator=generator,
        ).to(device)
        outputs = {}
        measurements = []
        for estimator in args.estimators:
            output, measurement = _profile_estimator(
                estimator,
                tensors,
                upstream,
                args,
            )
            outputs[estimator] = output
            measurements.append(measurement)
        reference = outputs[args.estimators[0]]
        for measurement in measurements:
            measurement["hard_forward_equal"] = bool(
                torch.equal(outputs[measurement["estimator"]], reference)
            )
        records.append(
            {
                "batch": batch,
                "sequence_length": sequence_length,
                "measurements": measurements,
            }
        )
    return {
        "schema_version": 1,
        "objective": "matched hard-forward VJP latency and allocator peak",
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "estimators": list(args.estimators),
        "heads": args.heads,
        "qk_bits": args.qk_bits,
        "value_heads": args.value_heads,
        "value_bits": args.value_bits,
        "max_suffix_length": args.max_suffix_length,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "cases": records,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--estimators",
        nargs="+",
        choices=ESTIMATORS,
        default=list(ESTIMATORS),
    )
    parser.add_argument(
        "--cases",
        nargs="+",
        type=_parse_case,
        default=[(1, 20), (1, 40), (1, 80), (64, 20)],
        metavar="B:T",
    )
    parser.add_argument("--heads", type=int, default=2)
    parser.add_argument("--qk-bits", type=int, default=8)
    parser.add_argument("--value-heads", type=int, default=1)
    parser.add_argument("--value-bits", type=int, default=8)
    parser.add_argument("--max-suffix-length", type=int, default=1)
    parser.add_argument("--scale", type=float, default=ROSA_SOFT_DEFAULT_SCALE)
    parser.add_argument(
        "--mismatch-scale",
        type=float,
        default=ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    )
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json-out", default="")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.warmup < 0 or args.iterations < 1:
        raise ValueError("warmup must be >= 0 and iterations must be >= 1")
    report = run_benchmark(args)
    encoded = json.dumps(report, indent=2, allow_nan=False)
    print(encoded)
    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(encoded + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
