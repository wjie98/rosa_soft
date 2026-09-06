"""Compute Sanitizer smoke for production unlimited hard diagonal DP."""

from __future__ import annotations

import torch

import rosa_soft  # noqa: F401 - registers production operators


def main() -> None:
    generator = torch.Generator(device="cuda").manual_seed(59017)
    tokens = 513
    query = torch.randn(1, tokens, 2, 8, device="cuda", generator=generator)
    key = torch.randn_like(query)
    value = torch.randn(1, tokens, 1, 5, device="cuda", generator=generator)
    torch.ops.rosa_soft.hard_forward(query, key, value)
    query.fill_(1)
    key.fill_(1)
    torch.ops.rosa_soft.hard_forward(query, key, value)

    lengths = (257, 0, 263)
    total_tokens = sum(lengths)
    packed_query = torch.randn(
        total_tokens, 2, 8, device="cuda", generator=generator
    )
    packed_key = torch.randn_like(packed_query)
    packed_value = torch.randn(
        total_tokens, 1, 5, device="cuda", generator=generator
    )
    offsets = torch.tensor(
        (0, 257, 257, total_tokens), dtype=torch.int32, device="cuda"
    )
    torch.ops.rosa_soft.hard_forward_varlen(
        packed_query, packed_key, packed_value, offsets
    )
    packed_query.fill_(1)
    packed_key.fill_(1)
    torch.ops.rosa_soft.hard_forward_varlen(
        packed_query, packed_key, packed_value, offsets
    )
    torch.cuda.synchronize()
    print("production hard DP sanitizer smoke passed")


if __name__ == "__main__":
    main()
