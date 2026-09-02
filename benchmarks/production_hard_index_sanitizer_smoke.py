"""Minimal production hard-index target for CUDA Compute Sanitizer."""

from __future__ import annotations

import argparse

import torch

import rosa_soft  # noqa: F401 - registers torch.ops.rosa_soft


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pattern",
        choices=[
            "random",
            "periodic4",
            "periodic64",
            "all_match",
            "all_mismatch",
        ],
        default="random",
    )
    parser.add_argument("--sequence-length", type=int, default=4096)
    args = parser.parse_args()
    if args.sequence_length < 512:
        raise ValueError("the D8 production hard index requires T >= 512")

    generator = torch.Generator(device="cuda").manual_seed(9400)
    shape = (1, args.sequence_length, 2, 8)
    query = torch.randn(
        shape,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    key = torch.randn(
        shape,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )
    if args.pattern.startswith("periodic"):
        period = int(args.pattern.removeprefix("periodic"))
        codes = torch.where(
            torch.randint(
                0,
                2,
                (1, period, 2, 8),
                generator=generator,
                device="cuda",
            ).bool(),
            1.0,
            -1.0,
        ).half()
        positions = torch.arange(args.sequence_length, device="cuda") % period
        query.copy_(codes[:, positions])
        key.copy_(codes[:, positions])
    elif args.pattern == "all_match":
        query.fill_(1)
        key.fill_(1)
    elif args.pattern == "all_mismatch":
        query.fill_(1)
        key.fill_(-1)
    value = torch.randn(
        1,
        args.sequence_length,
        1,
        8,
        generator=generator,
        device="cuda",
        dtype=torch.float16,
    )

    output, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query,
        key,
        value,
    )
    torch.cuda.synchronize()
    print(
        args.pattern,
        float(output.float().sum()),
        int(packed_query.long().sum()),
        int(packed_key.long().sum()),
    )


if __name__ == "__main__":
    main()
