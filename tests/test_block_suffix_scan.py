import pytest
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


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="the block suffix study requires CUDA",
)


@pytest.fixture(scope="session")
def block_suffix_module():
    try:
        return load_block_suffix_scan()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")


def _codes(shape, bits, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    if bits == 32:
        return torch.randint(
            -(2**31),
            2**31,
            shape,
            dtype=torch.int32,
            device="cuda",
            generator=generator,
        )
    return torch.randint(
        0,
        1 << bits,
        shape,
        dtype=torch.int32,
        device="cuda",
        generator=generator,
    )


@pytest.mark.parametrize("seq_len", [1, 2, 7, 33, 64, 65, 129])
@pytest.mark.parametrize("bits", [1, 8, 31, 32])
@pytest.mark.parametrize("window", [1, 4, 16, 32, 65])
def test_block_suffix_methods_match_thread_reference(
    seq_len,
    bits,
    window,
    block_suffix_module,
):
    query = _codes((2, 3, seq_len), bits, 11000 + seq_len + bits + window)
    key = _codes((2, 3, seq_len), bits, 12000 + seq_len + bits + window)
    expected = block_suffix_scores(
        query,
        key,
        symbol_dim=bits,
        max_suffix_length=window,
        method="thread",
        module=block_suffix_module,
    )
    for method in METHODS[1:]:
        actual = block_suffix_scores(
            query,
            key,
            symbol_dim=bits,
            max_suffix_length=window,
            method=method,
            module=block_suffix_module,
        )
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("seq_len", [32, 65, 130])
@pytest.mark.parametrize("window", [1, 31, 32, 33, 128])
def test_block_suffix_all_match_is_capped_run_length(
    seq_len,
    window,
    block_suffix_module,
):
    codes = torch.ones((1, 1, seq_len), dtype=torch.int32, device="cuda")
    row = torch.arange(seq_len, device="cuda").view(seq_len, 1)
    route = torch.arange(seq_len, device="cuda").view(1, seq_len)
    expected = torch.minimum(
        torch.minimum(row + 1, route),
        torch.tensor(min(window, seq_len), device="cuda"),
    ).float()
    expected.masked_fill_((route < 1) | (route > row), 0.0)
    for method in METHODS:
        actual = block_suffix_scores(
            codes,
            codes,
            symbol_dim=8,
            max_suffix_length=window,
            method=method,
            module=block_suffix_module,
        )[0, 0]
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("active_queries", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("bits", [8, 32])
def test_causal_tail_methods_match_candidate_thread_reference(
    active_queries,
    bits,
    block_suffix_module,
):
    query = _codes((2, 3, 256), bits, 43000 + active_queries + bits)
    key = _codes((2, 3, 256), bits, 44000 + active_queries + bits)
    arguments = dict(
        symbol_dim=bits,
        max_suffix_length=32,
        route_start=128,
        active_queries=active_queries,
        module=block_suffix_module,
    )
    expected = block_suffix_tail_scores(
        query,
        key,
        method="tail_thread",
        **arguments,
    )
    for method in TAIL_METHODS:
        actual = block_suffix_tail_scores(
            query,
            key,
            method=method,
            **arguments,
        )
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("active_queries", [1, 8, 32])
def test_causal_tail_all_match_scores_are_exact(
    active_queries,
    block_suffix_module,
):
    codes = torch.ones((1, 2, 256), dtype=torch.int32, device="cuda")
    expected = torch.tril(
        torch.full(
            (active_queries, active_queries),
            32.0,
            device="cuda",
        )
    )
    for method in TAIL_METHODS:
        actual = block_suffix_tail_scores(
            codes,
            codes,
            symbol_dim=8,
            max_suffix_length=32,
            route_start=128,
            active_queries=active_queries,
            method=method,
            module=block_suffix_module,
        )
        torch.testing.assert_close(actual[0, 0], expected, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("tail_queries", [1, 2, 4, 8, 16, 32])
@pytest.mark.parametrize("bits", [8, 32])
def test_hybrid_tile_methods_match_full_diagonal_reference(
    tail_queries,
    bits,
    block_suffix_module,
):
    query = _codes((2, 3, 256), bits, 45000 + tail_queries + bits)
    key = _codes((2, 3, 256), bits, 46000 + tail_queries + bits)
    arguments = dict(
        symbol_dim=bits,
        max_suffix_length=32,
        tile_start=128,
        tail_queries=tail_queries,
        module=block_suffix_module,
    )
    expected = block_suffix_hybrid_scores(
        query,
        key,
        method="full_diagonal_thread",
        **arguments,
    )
    for method in HYBRID_METHODS[1:]:
        actual = block_suffix_hybrid_scores(
            query,
            key,
            method=method,
            **arguments,
        )
        torch.testing.assert_close(actual, expected, rtol=3e-5, atol=3e-5)


@pytest.mark.parametrize("tail_queries", [1, 8, 32])
def test_hybrid_tile_all_match_scores_are_exact(
    tail_queries,
    block_suffix_module,
):
    codes = torch.ones((1, 2, 256), dtype=torch.int32, device="cuda")
    expected = torch.tril(torch.full((64, 64), 32.0, device="cuda"))
    for method in HYBRID_METHODS:
        actual = block_suffix_hybrid_scores(
            codes,
            codes,
            symbol_dim=8,
            max_suffix_length=32,
            tile_start=128,
            tail_queries=tail_queries,
            method=method,
            module=block_suffix_module,
        )
        torch.testing.assert_close(actual[0, 0], expected, rtol=0.0, atol=0.0)
