"""Compute Sanitizer smoke for the production unbounded RosaSoft VJP."""

from __future__ import annotations

import torch

import rosa_soft  # noqa: F401 - registers production operators


def main() -> None:
    generator = torch.Generator(device="cuda").manual_seed(40691)
    seq_len = 2048
    num_heads = 8
    query = torch.randn(
        1,
        seq_len,
        num_heads,
        8,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn(
        query.shape,
        dtype=query.dtype,
        device="cuda",
        generator=generator,
    )
    value = torch.randn(
        1,
        seq_len,
        4,
        64,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        1,
        seq_len,
        num_heads,
        64,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    dropout_seed = torch.tensor(40691, dtype=torch.int64, device="cuda")
    gradients = torch.ops.rosa_soft.surrogate_vjp_unbounded_masked(
        query,
        key,
        value,
        grad_output,
        packed_query,
        packed_key,
        dropout_seed,
        1.0,
        0.1,
        3.0,
        7,
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    torch.cuda.synchronize()
    print("production unbounded VJP sanitizer smoke passed")


if __name__ == "__main__":
    main()
