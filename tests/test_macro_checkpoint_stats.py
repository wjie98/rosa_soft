import pytest
import torch

import rosa_soft  # noqa: F401 - register production operators
from benchmarks.persistent_wavefront_vjp import (
    load_persistent_wavefront_vjp,
    macro_checkpoint_stats,
    unbounded_replay_stats,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="macro checkpoint statistics require CUDA",
)


@pytest.fixture(scope="session")
def checkpoint_module():
    try:
        return load_persistent_wavefront_vjp()
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")


def _case(seq_len: int, bits: int, dropout_p: float, pattern: str):
    generator = torch.Generator(device="cuda").manual_seed(
        91800 + seq_len * 17 + bits
    )
    query = torch.randn(
        1,
        seq_len,
        4,
        bits,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    key = torch.randn_like(query)
    if pattern == "all_match":
        query.fill_(1.0)
        key.fill_(1.0)
    elif pattern == "alternating":
        signs = torch.where(
            torch.arange(seq_len, device="cuda")[:, None, None] % 2 == 0,
            1.0,
            -1.0,
        )
        query.copy_(signs)
        key.copy_(signs)
    value = torch.randn(
        1,
        seq_len,
        2,
        64,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    grad_output = torch.randn(
        1,
        seq_len,
        4,
        64,
        dtype=torch.float16,
        device="cuda",
        generator=generator,
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    dropout_seed = (
        torch.tensor(0x31415926, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    return value, grad_output, packed_query, packed_key, dropout_seed


@pytest.mark.parametrize("macro_diagonals", [32, 64, 96])
@pytest.mark.parametrize("seq_len", [1, 2, 31, 32, 33, 65, 97, 193])
@pytest.mark.parametrize("bits", [1, 8, 32])
def test_macro_stats_match_existing_stats(
    macro_diagonals,
    seq_len,
    bits,
    checkpoint_module,
):
    arguments = _case(seq_len, bits, 0.0, "random")
    expected = unbounded_replay_stats(
        *arguments,
        symbol_dim=bits,
        group_size=min(128, seq_len),
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        module=checkpoint_module,
    )
    actual = macro_checkpoint_stats(
        *arguments,
        symbol_dim=bits,
        macro_diagonals=macro_diagonals,
        slab_size=min(128, seq_len),
        scale=1.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        module=checkpoint_module,
    )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-4)


@pytest.mark.parametrize("macro_diagonals", [32, 64, 96])
@pytest.mark.parametrize("pattern", ["random", "all_match", "alternating"])
def test_macro_stats_preserve_dropout_and_patterns(
    macro_diagonals,
    pattern,
    checkpoint_module,
):
    arguments = _case(131, 8, 0.2, pattern)
    expected = unbounded_replay_stats(
        *arguments,
        symbol_dim=8,
        group_size=131,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        module=checkpoint_module,
    )
    actual = macro_checkpoint_stats(
        *arguments,
        symbol_dim=8,
        macro_diagonals=macro_diagonals,
        slab_size=131,
        scale=1.7,
        dropout_p=0.2,
        mismatch_scale=3.0,
        module=checkpoint_module,
    )
    torch.testing.assert_close(actual, expected, rtol=3e-4, atol=3e-4)
