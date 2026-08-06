"""Profile factorized CPU indexing and exact CUDA descriptor VJP."""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


from benchmarks.filtered_bitflip_profile import make_training_codes
from benchmarks.sam_bitflip import sam_bitflip
from benchmarks.sam_bitflip_native import (
    NativeSamBitflip,
    build_native_sam_bitflip,
)
from benchmarks.sam_bitflip_vjp import descriptor_vjp, load_descriptor_vjp


CASES = (
    ("independent_d1", 1, "shift_random", 0.0),
    ("independent_d8", 8, "shift_random", 0.0),
    ("shift_d8", 8, "shift_random", 1.0),
    ("motif_d4", 4, "shift_motif", 1.0),
    ("collapse_d8", 8, "collapse", 1.0),
)


def _median_cpu(function, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        samples.append((time.perf_counter() - start) * 1e3)
    return result, statistics.median(samples)


def _median_cuda(function, repeats):
    samples = []
    result = None
    for _ in range(repeats):
        start = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return result, statistics.median(samples)


def _cached_arguments(result, query, key, value, grad_output):
    device = value.device
    return (
        result.base_routes.to(device),
        result.query_offsets.to(device),
        result.query_changes.to(device),
        result.key_delete_offsets.to(device),
        result.key_delete_changes.to(device),
        result.key_override_offsets.to(device),
        result.key_overrides.to(device),
        query.to(device),
        key.to(device),
        value,
        grad_output,
        result.bit_width,
    )


def profile_case(
    native,
    module,
    sequence_length,
    feature_size,
    case,
    repeats,
    seed,
    validate,
):
    name, bit_width, trajectory, clarity = case
    query, key = make_training_codes(
        sequence_length,
        bit_width,
        trajectory,
        clarity,
        seed,
    )
    result, cpu_ms = _median_cpu(
        lambda: native.solve_factorized(query, key, bit_width),
        repeats,
    )
    generator = torch.Generator().manual_seed(
        seed + 101 * sequence_length + bit_width
    )
    value_cpu = torch.randn(
        sequence_length,
        feature_size,
        dtype=torch.float32,
        generator=generator,
    )
    grad_cpu = torch.randn(
        sequence_length,
        feature_size,
        dtype=torch.float32,
        generator=generator,
    )
    value = value_cpu.cuda()
    grad_output = grad_cpu.cuda()
    actual, transfer_vjp_ms = _median_cuda(
        lambda: descriptor_vjp(
            result,
            query,
            key,
            value,
            grad_output,
            module=module,
        ),
        repeats,
    )
    max_abs_error = None
    if validate:
        expected = sam_bitflip(
            query,
            key,
            bit_width,
            value=value_cpu,
            grad_output=grad_cpu,
        ).bit_gradient
        actual_cpu = actual.cpu()
        difference = (actual_cpu - expected).abs()
        max_abs_error = (
            float(difference.max().item()) if difference.numel() else 0.0
        )
        # The CUDA kernel changes only the FP32 reduction order. Float64
        # oracle tests enforce descriptor exactness independently.
        torch.testing.assert_close(
            actual_cpu,
            expected,
            rtol=5e-4,
            atol=2e-4,
        )
    arguments = _cached_arguments(result, query, key, value, grad_output)
    module.descriptor_vjp(*arguments)
    torch.cuda.synchronize()
    _, cached_vjp_ms = _median_cuda(
        lambda: module.descriptor_vjp(*arguments),
        repeats,
    )
    return {
        "case": name,
        "sequence_length": sequence_length,
        "bit_width": bit_width,
        "cpu_factorized_ms": cpu_ms,
        "transfer_and_vjp_ms": transfer_vjp_ms,
        "cached_vjp_ms": cached_vjp_ms,
        "oracle_validated": validate,
        "max_abs_error": max_abs_error,
        "descriptor_bytes": result.descriptor_bytes,
        "profile": result.profile,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sequence-lengths", type=int, nargs="+", default=[64, 128, 256, 512]
    )
    parser.add_argument("--feature-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--validate-max-length", type=int, default=128)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--build-native", action="store_true")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    if args.build_native:
        build_native_sam_bitflip()
    native = NativeSamBitflip()
    module = load_descriptor_vjp()
    results = [
        profile_case(
            native,
            module,
            sequence_length,
            args.feature_size,
            case,
            args.repeats,
            args.seed,
            sequence_length <= args.validate_max_length,
        )
        for sequence_length in args.sequence_lengths
        for case in CASES
    ]
    report = {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "device": torch.cuda.get_device_name(),
        },
        "feature_size": args.feature_size,
        "repeats": args.repeats,
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)


if __name__ == "__main__":
    main()
