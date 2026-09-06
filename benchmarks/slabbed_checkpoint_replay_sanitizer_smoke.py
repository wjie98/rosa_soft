"""Compute Sanitizer smoke for fixed-width slabbed replay."""

from __future__ import annotations

import torch

from benchmarks.slabbed_checkpoint_replay import (
    load_slabbed_checkpoint_replay,
    slabbed_replay_vjp,
)


def main() -> None:
    module = load_slabbed_checkpoint_replay()
    generator = torch.Generator(device="cuda").manual_seed(91257)
    for seq_len, value_dim, dropout_p in ((35, 17, 0.0), (258, 65, 0.2)):
        query = torch.randn(
            1, seq_len, 2, 8, device="cuda", generator=generator
        )
        key = torch.randn(query.shape, device="cuda", generator=generator)
        value = torch.randn(
            1, seq_len, 1, value_dim, device="cuda", generator=generator
        )
        grad_output = torch.randn(
            1, seq_len, 2, value_dim, device="cuda", generator=generator
        )
        shifts = torch.arange(8, dtype=torch.int64, device="cuda")
        packed_query = (
            ((query > 0).to(torch.int64).permute(0, 2, 1, 3) << shifts)
            .sum(-1)
            .to(torch.int32)
        )
        packed_key = (
            ((key > 0).to(torch.int64).permute(0, 2, 1, 3) << shifts)
            .sum(-1)
            .to(torch.int32)
        )
        seed = (
            torch.tensor(1276543, dtype=torch.int64, device="cuda")
            if dropout_p
            else torch.empty(0, dtype=torch.int64, device="cuda")
        )
        outputs = slabbed_replay_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            scale=1.7,
            dropout_p=dropout_p,
            mismatch_scale=3.0,
            gradient_mask=7,
            module=module,
        )
        assert all(torch.isfinite(output).all() for output in outputs)
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
