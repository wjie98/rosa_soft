"""CUDA wrapper for the hard-forward RosaSoft operator."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .soft_contract import (
    ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
    ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    validate_cuda_scalars,
    validate_soft_inputs,
)


__all__ = [
    "ROSA_SOFT_DEFAULT_MISMATCH_PENALTY",
    "ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE",
    "rosa_soft",
]


def _validate_cuda_inputs(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
) -> int:
    normalized_length = validate_soft_inputs(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
    )
    if not query_logits.is_cuda:
        raise ValueError("rosa_soft requires CUDA tensors")
    if query_logits.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError("rosa_soft supports float32, float16, and bfloat16")
    validate_cuda_scalars(
        normalized_length,
        route_temperature,
        mismatch_penalty,
    )
    return normalized_length


@torch.library.register_fake("rosa_soft::soft_forward")
def _fake_soft_forward(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    max_suffix_length: int,
):
    del key_logits, max_suffix_length
    symbol_shape = query_logits.shape[:3]
    return [
        query_logits.new_empty(
            (*symbol_shape, payload_logits.shape[3])
        ),
        query_logits.new_empty(symbol_shape, dtype=torch.int32),
        query_logits.new_empty(symbol_shape, dtype=torch.int32),
    ]


@torch.library.register_fake("rosa_soft::soft_backward")
def _fake_soft_backward(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    grad_output: Tensor,
    query_symbols: Tensor,
    key_symbols: Tensor,
    rng_seed: Tensor,
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
):
    del (
        grad_output,
        query_symbols,
        key_symbols,
        rng_seed,
        max_suffix_length,
        route_temperature,
        mismatch_penalty,
    )
    return [
        query_logits.new_empty(
            query_logits.shape,
            dtype=torch.float32,
        ),
        key_logits.new_empty(
            key_logits.shape,
            dtype=torch.float32,
        ),
        payload_logits.new_empty(
            payload_logits.shape,
            dtype=torch.float32,
        ),
    ]


class _RosaSoftFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query_logits: Tensor,
        key_logits: Tensor,
        payload_logits: Tensor,
        max_suffix_length: int,
        route_temperature: float,
        mismatch_penalty: float,
        rng_seed: Tensor,
    ) -> Tensor:
        hard_output, query_symbols, key_symbols = (
            torch.ops.rosa_soft.soft_forward(
                query_logits,
                key_logits,
                payload_logits,
                int(max_suffix_length),
            )
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.route_temperature = float(route_temperature)
        ctx.mismatch_penalty = float(mismatch_penalty)
        ctx.save_for_backward(
            query_logits,
            key_logits,
            payload_logits,
            query_symbols,
            key_symbols,
            rng_seed,
        )
        return hard_output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        (
            query_logits,
            key_logits,
            payload_logits,
            query_symbols,
            key_symbols,
            rng_seed,
        ) = ctx.saved_tensors
        grad_query, grad_key, grad_payload = (
            torch.ops.rosa_soft.soft_backward(
                query_logits,
                key_logits,
                payload_logits,
                grad_output,
                query_symbols,
                key_symbols,
                rng_seed,
                ctx.max_suffix_length,
                ctx.route_temperature,
                ctx.mismatch_penalty,
            )
        )
        return (
            grad_query.to(query_logits.dtype),
            grad_key.to(key_logits.dtype),
            grad_payload.to(payload_logits.dtype),
            None,
            None,
            None,
            None,
        )


def _rosa_soft_with_seed(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    max_suffix_length: int,
    route_temperature: float,
    mismatch_penalty: float,
    rng_seed: Tensor,
) -> Tensor:
    """Private deterministic entry point used by CUDA parity tests."""

    return _RosaSoftFunction.apply(
        query_logits,
        key_logits,
        payload_logits,
        int(max_suffix_length),
        float(route_temperature),
        float(mismatch_penalty),
        rng_seed,
    )


def rosa_soft(
    query_logits: Tensor,
    key_logits: Tensor,
    payload_logits: Tensor,
    max_suffix_length: int = 32,
    route_temperature: float = ROSA_SOFT_DEFAULT_ROUTE_TEMPERATURE,
    mismatch_penalty: float = ROSA_SOFT_DEFAULT_MISMATCH_PENALTY,
) -> Tensor:
    """Run exact hard ROSA forward with the fixed stochastic CUDA VJP."""

    max_suffix_length = _validate_cuda_inputs(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        float(route_temperature),
        float(mismatch_penalty),
    )
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (query_logits, key_logits, payload_logits)
    )
    if not needs_backward:
        return torch.ops.rosa_soft.soft_forward(
            query_logits,
            key_logits,
            payload_logits,
            max_suffix_length,
        )[0]

    rng_seed = torch.empty(
        (),
        dtype=torch.int64,
        device=query_logits.device,
    ).random_()
    return _rosa_soft_with_seed(
        query_logits,
        key_logits,
        payload_logits,
        max_suffix_length,
        float(route_temperature),
        float(mismatch_penalty),
        rng_seed,
    )
