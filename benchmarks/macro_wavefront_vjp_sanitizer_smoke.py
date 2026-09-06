"""Compute Sanitizer smoke for exact macro-wavefront schedules."""

from __future__ import annotations

import torch

from benchmarks.macro_wavefront_vjp import (
    load_macro_wavefront_vjp,
    macro_wavefront_scores,
    macro_wavefront_stats,
    macro_wavefront_vjp,
)


def _pack_symbols(tensor: torch.Tensor) -> torch.Tensor:
    shifts = torch.arange(tensor.size(-1), dtype=torch.int64, device="cuda")
    return (
        ((tensor > 0).to(torch.int64).permute(0, 2, 1, 3) << shifts)
        .sum(-1)
        .to(torch.int32)
    )


def main() -> None:
    generator = torch.Generator(device="cuda").manual_seed(47291)
    query = torch.randn(1, 65, 2, 8, device="cuda", generator=generator)
    key = torch.randn_like(query)
    packed_query = _pack_symbols(query)
    packed_key = _pack_symbols(key)
    seed = torch.tensor(78123645, dtype=torch.int64, device="cuda")
    module = load_macro_wavefront_vjp()
    for value_dim in (17, 65):
        value = torch.randn(
            1, 65, 1, value_dim, device="cuda", generator=generator
        )
        grad_output = torch.randn(
            1, 65, 2, value_dim, device="cuda", generator=generator
        )
        for tile_size in (32, 64):
            for plan in (
                "multilaunch",
                "persistent",
                "persistent_barrier",
                "folded",
                "persistent_rows",
            ):
                macro_wavefront_scores(
                    packed_query,
                    packed_key,
                    symbol_dim=8,
                    tile_size=tile_size,
                    plan=plan,
                    module=module,
                )
                macro_wavefront_stats(
                    value,
                    grad_output,
                    packed_query,
                    packed_key,
                    seed,
                    symbol_dim=8,
                    tile_size=tile_size,
                    scale=1.7,
                    dropout_p=0.2,
                    plan=plan,
                    module=module,
                )
                macro_wavefront_vjp(
                    query,
                    key,
                    value,
                    grad_output,
                    packed_query,
                    packed_key,
                    seed,
                    tile_size=tile_size,
                    scale=1.7,
                    dropout_p=0.2,
                    gradient_mask=7,
                    plan=plan,
                    module=module,
                )
    torch.cuda.synchronize()
    print("macro wavefront sanitizer smoke passed")


if __name__ == "__main__":
    main()
