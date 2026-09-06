"""Emit an Nsight-friendly trace of the production unbounded VJP."""

from __future__ import annotations

import argparse

import torch

import rosa_soft  # noqa: F401 - registers production operators


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--value-heads", type=int, default=2)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--value-dim", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--gradient-mask", type=int, default=7)
    parser.add_argument("--seed", type=int, default=58123)
    args = parser.parse_args()
    generator = torch.Generator(device="cuda").manual_seed(args.seed)
    query = torch.randn(
        args.batch,
        args.tokens,
        args.heads,
        args.bits,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn_like(query)
    value = torch.randn(
        args.batch,
        args.tokens,
        args.value_heads,
        args.value_dim,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        args.batch,
        args.tokens,
        args.heads,
        args.value_dim,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    dropout_seed = torch.empty(0, dtype=torch.int64, device="cuda")

    def operation():
        return torch.ops.rosa_soft.surrogate_vjp_unbounded_masked(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            dropout_seed,
            1.0,
            0.0,
            3.0,
            args.gradient_mask,
        )

    for _ in range(args.warmup):
        operation()
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push("unbounded_replay_measurement")
    for _ in range(args.iterations):
        operation()
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
