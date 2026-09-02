"""Small block-diagonal VJP workload for Compute Sanitizer."""

from __future__ import annotations

import argparse

import torch

import rosa_soft  # noqa: F401 - registers torch.ops.rosa_soft
from benchmarks.block_diagonal_vjp import (
    block_diagonal_vjp,
    load_block_diagonal_vjp,
)


def _run_case(
    *,
    sequence_length: int,
    dtype: torch.dtype,
    dropout_p: float,
    all_match: bool,
    gradient_masks: range | tuple[int, ...] = range(1, 8),
    plans: tuple[str, ...] = (
        "block_diagonal",
        "block_tf32",
        "block_tf32_pipeline",
    ),
) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        41000 + sequence_length
    )
    query = torch.randn(
        1,
        sequence_length,
        2,
        8,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        query.shape,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    if all_match:
        query.fill_(1)
        key.fill_(1)
    value = torch.randn(
        1,
        sequence_length,
        1,
        64,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        1,
        sequence_length,
        2,
        64,
        dtype=dtype,
        device="cuda",
        generator=generator,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    dropout_seed = (
        torch.tensor(123456789, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    arguments = (
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
    )
    module = load_block_diagonal_vjp()
    for gradient_mask in gradient_masks:
        baseline = block_diagonal_vjp(
            *arguments,
            max_suffix_length=32,
            scale=2.0,
            dropout_p=dropout_p,
            mismatch_scale=3.0,
            gradient_mask=gradient_mask,
            plan="baseline",
            module=module,
        )
        for plan in plans:
            tolerance = 5e-4 if plan == "block_diagonal" else 3e-3
            actual = block_diagonal_vjp(
                *arguments,
                max_suffix_length=32,
                scale=2.0,
                dropout_p=dropout_p,
                mismatch_scale=3.0,
                gradient_mask=gradient_mask,
                plan=plan,
                module=module,
            )
            for candidate, reference in zip(actual, baseline):
                torch.testing.assert_close(
                    candidate,
                    reference,
                    rtol=tolerance,
                    atol=tolerance,
                )
    torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--mask", type=int, default=7)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.minimal:
        _run_case(
            sequence_length=65,
            dtype=torch.float32,
            dropout_p=0.0,
            all_match=False,
            gradient_masks=(args.mask,),
            plans=("block_tf32_pipeline",),
        )
        print("Block-diagonal VJP sanitizer smoke passed")
        return
    _run_case(
        sequence_length=65,
        dtype=torch.float32,
        dropout_p=0.0,
        all_match=False,
    )
    _run_case(
        sequence_length=97,
        dtype=torch.float16,
        dropout_p=0.2,
        all_match=True,
    )
    _run_case(
        sequence_length=129,
        dtype=torch.bfloat16,
        dropout_p=0.2,
        all_match=False,
    )
    print("Block-diagonal VJP sanitizer smoke passed")


if __name__ == "__main__":
    main()
