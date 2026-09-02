"""Small correctness workload for Compute Sanitizer execution."""

from __future__ import annotations

import torch

import rosa_soft  # noqa: F401 - registers torch.ops.rosa_soft
from benchmarks.flash_tc_vjp import PLANS, flash_tc_vjp, load_flash_tc_vjp


def _run_case(*, window: int, upstream_scale: float = 1.0) -> None:
    sequence_length = 65
    generator = torch.Generator(device="cuda").manual_seed(
        31000 + window
    )
    query = torch.randn(
        1,
        sequence_length,
        2,
        32,
        generator=generator,
        device="cuda",
    )
    key = torch.randn(
        query.shape,
        generator=generator,
        device="cuda",
    )
    value = torch.randn(
        1,
        sequence_length,
        1,
        33,
        generator=generator,
        device="cuda",
    )
    grad_output = upstream_scale * torch.randn(
        1,
        sequence_length,
        2,
        33,
        generator=generator,
        device="cuda",
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    dropout_seed = torch.tensor(123456789, dtype=torch.int64, device="cuda")
    arguments = (
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
    )
    module = load_flash_tc_vjp()
    baseline = flash_tc_vjp(
        *arguments,
        max_suffix_length=window,
        scale=2.0,
        dropout_p=0.2,
        mismatch_scale=3.0,
        gradient_mask=7,
        plan="baseline",
        module=module,
    )
    plans = PLANS[1:] if upstream_scale == 1.0 else ("tc_gate",)
    for plan in plans:
        actual = flash_tc_vjp(
            *arguments,
            max_suffix_length=window,
            scale=2.0,
            dropout_p=0.2,
            mismatch_scale=3.0,
            gradient_mask=7,
            plan=plan,
            module=module,
        )
        for candidate, reference in zip(actual, baseline):
            torch.testing.assert_close(
                candidate,
                reference,
                rtol=4e-4,
                atol=0.5 if upstream_scale > 1.0 else 3e-4,
            )
    torch.cuda.synchronize()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _run_case(window=32)
    _run_case(window=4)
    _run_case(window=32, upstream_scale=1e6)
    print("FlashROSA-TC sanitizer smoke passed")


if __name__ == "__main__":
    main()
