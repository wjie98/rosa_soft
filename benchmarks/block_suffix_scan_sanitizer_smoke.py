"""Small suffix-ownership workload for Compute Sanitizer."""

from __future__ import annotations

import torch

from benchmarks.block_suffix_scan import (
    HYBRID_METHODS,
    METHODS,
    TAIL_METHODS,
    block_suffix_hybrid_scores,
    block_suffix_scores,
    block_suffix_tail_scores,
    load_block_suffix_scan,
)


def _run_case(*, sequence_length: int, bits: int, all_match: bool) -> None:
    generator = torch.Generator(device="cuda").manual_seed(
        42000 + sequence_length + bits
    )
    high = 2**31 - 1 if bits == 32 else 1 << bits
    low = -(2**31) if bits == 32 else 0
    query = torch.randint(
        low,
        high,
        (1, 2, sequence_length),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    key = torch.randint(
        low,
        high,
        query.shape,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    if all_match:
        key.copy_(query)
    module = load_block_suffix_scan()
    for window in (32, 65):
        effective_window = min(window, sequence_length)
        reference = block_suffix_scores(
            query,
            key,
            symbol_dim=bits,
            max_suffix_length=effective_window,
            method="thread",
            module=module,
        )
        for method in METHODS[1:]:
            actual = block_suffix_scores(
                query,
                key,
                symbol_dim=bits,
                max_suffix_length=effective_window,
                method=method,
                module=module,
            )
            torch.testing.assert_close(
                actual,
                reference,
                rtol=2e-5,
                atol=2e-6,
            )
    torch.cuda.synchronize()


def _run_tail_case(*, bits: int, active_queries: int) -> None:
    sequence_length = 128
    route_start = 64
    generator = torch.Generator(device="cuda").manual_seed(
        43000 + bits + active_queries
    )
    high = 2**31 - 1 if bits == 32 else 1 << bits
    low = -(2**31) if bits == 32 else 0
    query = torch.randint(
        low,
        high,
        (1, 2, sequence_length),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    key = torch.randint(
        low,
        high,
        query.shape,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    module = load_block_suffix_scan()
    reference = block_suffix_tail_scores(
        query,
        key,
        symbol_dim=bits,
        max_suffix_length=32,
        route_start=route_start,
        active_queries=active_queries,
        method="physical_block64",
        module=module,
    )
    for method in TAIL_METHODS[1:]:
        actual = block_suffix_tail_scores(
            query,
            key,
            symbol_dim=bits,
            max_suffix_length=32,
            route_start=route_start,
            active_queries=active_queries,
            method=method,
            module=module,
        )
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    torch.cuda.synchronize()


def _run_hybrid_case(*, bits: int, tail_queries: int) -> None:
    sequence_length = 256
    generator = torch.Generator(device="cuda").manual_seed(
        44000 + bits + tail_queries
    )
    high = 2**31 - 1 if bits == 32 else 1 << bits
    low = -(2**31) if bits == 32 else 0
    query = torch.randint(
        low,
        high,
        (1, 2, sequence_length),
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    key = torch.randint(
        low,
        high,
        query.shape,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )
    module = load_block_suffix_scan()
    reference = block_suffix_hybrid_scores(
        query,
        key,
        symbol_dim=bits,
        max_suffix_length=32,
        tile_start=128,
        tail_queries=tail_queries,
        method="full_diagonal_thread",
        module=module,
    )
    for method in HYBRID_METHODS[1:]:
        actual = block_suffix_hybrid_scores(
            query,
            key,
            symbol_dim=bits,
            max_suffix_length=32,
            tile_start=128,
            tail_queries=tail_queries,
            method=method,
            module=module,
        )
        torch.testing.assert_close(actual, reference, rtol=2e-5, atol=2e-6)
    torch.cuda.synchronize()


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    _run_case(sequence_length=65, bits=8, all_match=False)
    _run_case(sequence_length=97, bits=32, all_match=True)
    for bits in (8, 32):
        for active_queries in (1, 4, 16, 32):
            _run_tail_case(bits=bits, active_queries=active_queries)
        for tail_queries in (1, 4, 16, 32):
            _run_hybrid_case(bits=bits, tail_queries=tail_queries)
    print("Block suffix-scan sanitizer smoke passed")


if __name__ == "__main__":
    main()
