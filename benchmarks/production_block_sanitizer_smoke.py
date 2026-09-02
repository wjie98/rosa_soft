"""Minimal production block-VJP workload for Compute Sanitizer."""

from __future__ import annotations

import torch

import rosa_soft  # noqa: F401 - registers torch.ops.rosa_soft


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    generator = torch.Generator(device="cuda").manual_seed(20260821)
    query = torch.randn(
        (1, 4096, 1, 8),
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        query.shape,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        (1, 4096, 1, 64),
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        (1, 4096, 1, 64),
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    dropout_seed = torch.tensor(
        123456789,
        dtype=torch.int64,
        device="cuda",
    )
    for gradient_mask in (4, 7):
        gradients = torch.ops.rosa_soft.surrogate_vjp_masked(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            dropout_seed,
            32,
            2.0,
            0.2,
            3.0,
            gradient_mask,
        )
        for gradient in gradients:
            if gradient.numel() and not bool(torch.isfinite(gradient).all()):
                raise RuntimeError("production block VJP returned nonfinite data")
    torch.cuda.synchronize()
    print("Production block VJP sanitizer smoke passed")


if __name__ == "__main__":
    main()
