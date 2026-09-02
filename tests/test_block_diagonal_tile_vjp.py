import pytest
import torch

import rosa_soft  # noqa: F401 - registers the hard-forward torch operators
from benchmarks.block_diagonal_tile_vjp import (
    THREAD_VARIANTS,
    block_diagonal_tile32_vjp,
    load_block_diagonal_tile32,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability()[0] < 8,
    reason="the physical-tile VJP study requires Ampere CUDA",
)


def _case(bits, *, all_match=False, dropout_p=0.0):
    generator = torch.Generator(device="cuda").manual_seed(48000 + bits)
    query = torch.randn(
        1, 97, 2, bits, device="cuda", generator=generator
    )
    key = torch.randn(query.shape, device="cuda", generator=generator)
    if all_match:
        query.fill_(1)
        key.fill_(1)
    value = torch.randn(
        1, 97, 1, 32, device="cuda", generator=generator
    )
    grad_output = torch.randn(
        1, 97, 2, 32, device="cuda", generator=generator
    )
    _, packed_query, packed_key = torch.ops.rosa_soft.hard_forward(
        query, key, value
    )
    seed = (
        torch.tensor(87654321, dtype=torch.int64, device="cuda")
        if dropout_p
        else torch.empty(0, dtype=torch.int64, device="cuda")
    )
    return query, key, value, grad_output, packed_query, packed_key, seed


def _run(arguments, *, mask, threads, plan, module):
    return block_diagonal_tile32_vjp(
        *arguments,
        max_suffix_length=32,
        scale=2.0,
        dropout_p=0.0,
        mismatch_scale=3.0,
        gradient_mask=mask,
        threads=threads,
        plan=plan,
        module=module,
    )


@pytest.mark.parametrize("threads", THREAD_VARIANTS)
def test_tile32_thread_variants_match_streaming(threads):
    try:
        module = load_block_diagonal_tile32(threads)
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")
    arguments = _case(8)
    expected = _run(
        arguments,
        mask=7,
        threads=threads,
        plan="baseline",
        module=module,
    )
    actual = _run(
        arguments,
        mask=7,
        threads=threads,
        plan="block_tf32",
        module=module,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=3e-3, atol=3e-3)


@pytest.mark.parametrize("bits", [8, 32])
@pytest.mark.parametrize("mask", [1, 2, 3, 4, 5, 6, 7])
def test_tile32_192_thread_vjp_has_bounded_error(bits, mask):
    try:
        module = load_block_diagonal_tile32(192)
    except (OSError, RuntimeError) as error:
        pytest.skip(f"CUDA toolchain unavailable: {error}")
    arguments = _case(bits)
    expected = _run(
        arguments,
        mask=mask,
        threads=192,
        plan="baseline",
        module=module,
    )
    actual = _run(
        arguments,
        mask=mask,
        threads=192,
        plan="block_tf32",
        module=module,
    )
    for candidate, reference in zip(actual, expected):
        torch.testing.assert_close(candidate, reference, rtol=3e-3, atol=3e-3)
