"""Compute Sanitizer smoke for persistent block-wavefront kernels."""

from __future__ import annotations

import torch

import rosa_soft  # noqa: F401 - registers production operators
from benchmarks.persistent_wavefront_vjp import (
    load_persistent_wavefront_vjp,
    unbounded_replay_vjp,
    wavefront_vjp,
)


def main() -> None:
    generator = torch.Generator(device="cuda").manual_seed(47291)
    query = torch.randn(1, 33, 2, 8, device="cuda", generator=generator)
    key = torch.randn(query.shape, device="cuda", generator=generator)
    value = torch.randn(1, 33, 1, 17, device="cuda", generator=generator)
    grad_output = torch.randn(1, 33, 2, 17, device="cuda", generator=generator)
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query,
        key,
        value,
    )
    seed = torch.tensor(78123645, dtype=torch.int64, device="cuda")
    module = load_persistent_wavefront_vjp()
    for window in (1, 32):
        for plan in ("multilaunch", "persistent"):
            wavefront_vjp(
                query,
                key,
                value,
                grad_output,
                packed_query,
                packed_key,
                seed,
                max_suffix_length=window,
                scale=1.7,
                dropout_p=0.2,
                mismatch_scale=3.0,
                gradient_mask=7,
                plan=plan,
                module=module,
            )
    for group_size in (7, 32):
        unbounded_replay_vjp(
            query,
            key,
            value,
            grad_output,
            packed_query,
            packed_key,
            seed,
            group_size=group_size,
            scale=1.7,
            dropout_p=0.2,
            mismatch_scale=3.0,
            gradient_mask=7,
            module=module,
        )
    torch.cuda.synchronize()
    print("persistent wavefront sanitizer smoke passed")


if __name__ == "__main__":
    main()
