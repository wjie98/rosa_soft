"""CUDA wrapper for the hard-forward RosaSoft operator."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.autograd.function import once_differentiable

from .soft_contract import (
    ROSA_SOFT_DEFAULT_DROPOUT_P,
    ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
    ROSA_SOFT_DEFAULT_SCALE,
    make_dropout_seed,
    validate_fp32_surrogate_scalars,
    validate_rosa_soft_inputs,
    validate_rosa_soft_varlen_inputs,
)


__all__ = [
    "ROSA_SOFT_DEFAULT_DROPOUT_P",
    "ROSA_SOFT_DEFAULT_MISMATCH_SCALE",
    "ROSA_SOFT_DEFAULT_SCALE",
    "rosa_soft",
    "rosa_soft_varlen",
]


def _validate_cuda_call(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
) -> int:
    effective_max_suffix_length = validate_rosa_soft_inputs(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    if not query.is_cuda:
        raise ValueError("rosa_soft requires CUDA tensors")
    if query.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError("rosa_soft supports float32, float16, and bfloat16")
    validate_fp32_surrogate_scalars(
        effective_max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    return effective_max_suffix_length


def _validate_cuda_varlen_call(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
) -> int:
    effective_max_suffix_length = validate_rosa_soft_varlen_inputs(
        query,
        key,
        value,
        cu_seqlens,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    if not query.is_cuda:
        raise ValueError("rosa_soft_varlen requires CUDA tensors")
    if query.dtype not in (
        torch.float32,
        torch.float16,
        torch.bfloat16,
    ):
        raise ValueError(
            "rosa_soft_varlen supports float32, float16, and bfloat16"
        )
    validate_fp32_surrogate_scalars(
        effective_max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    return effective_max_suffix_length


def _gradient_mask(needs_input_grad: tuple[bool, ...]) -> int:
    needs_query, needs_key, needs_value = needs_input_grad[:3]
    return (
        int(needs_query)
        | (int(needs_key) << 1)
        | (int(needs_value) << 2)
    )


def _cast_requested_gradients(
    gradients: tuple[Tensor, Tensor, Tensor],
    inputs: tuple[Tensor, Tensor, Tensor],
    needs_input_grad: tuple[bool, ...],
) -> tuple[Tensor | None, Tensor | None, Tensor | None]:
    return tuple(
        gradient.to(input_tensor.dtype) if requested else None
        for gradient, input_tensor, requested in zip(
            gradients,
            inputs,
            needs_input_grad,
        )
    )


@torch.library.register_fake("rosa_soft::hard_forward")
def _fake_hard_forward(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    max_suffix_length: int,
):
    del key, max_suffix_length
    output_shape = (
        *query.shape[:3],
        value.shape[3],
    )
    packed_symbol_shape = (
        query.shape[0],
        query.shape[2],
        query.shape[1],
    )
    return (
        query.new_empty(output_shape),
        query.new_empty(
            packed_symbol_shape,
            dtype=torch.int32,
        ),
        query.new_empty(
            packed_symbol_shape,
            dtype=torch.int32,
        ),
    )


@torch.library.register_fake("rosa_soft::hard_forward_varlen")
def _fake_hard_forward_varlen(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
    max_suffix_length: int,
):
    del key, cu_seqlens, max_suffix_length
    token_head_shape = query.shape[:2]
    packed_symbol_shape = (
        query.shape[1],
        query.shape[0],
    )
    return (
        query.new_empty(
            (*token_head_shape, value.shape[2])
        ),
        query.new_empty(packed_symbol_shape, dtype=torch.int32),
        query.new_empty(packed_symbol_shape, dtype=torch.int32),
    )


@torch.library.register_fake("rosa_soft::surrogate_vjp_masked")
def _fake_surrogate_vjp_masked(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    gradient_mask: int,
):
    del (
        grad_output,
        packed_query_symbols,
        packed_key_symbols,
        dropout_seed,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    return tuple(
        input_tensor.new_empty(
            input_tensor.shape,
            dtype=torch.float32,
        )
        if gradient_mask & bit
        else input_tensor.new_empty((0,), dtype=torch.float32)
        for bit, input_tensor in zip(
            (1, 2, 4),
            (query, key, value),
        )
    )


@torch.library.register_fake("rosa_soft::surrogate_vjp_varlen_masked")
def _fake_surrogate_vjp_varlen_masked(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
    grad_output: Tensor,
    packed_query_symbols: Tensor,
    packed_key_symbols: Tensor,
    dropout_seed: Tensor,
    max_suffix_length: int,
    scale: float,
    dropout_p: float,
    mismatch_scale: float,
    gradient_mask: int,
):
    del (
        cu_seqlens,
        grad_output,
        packed_query_symbols,
        packed_key_symbols,
        dropout_seed,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    gradients = (
        query.new_empty(query.shape, dtype=torch.float32),
        key.new_empty(key.shape, dtype=torch.float32),
        value.new_empty(value.shape, dtype=torch.float32),
    )
    return tuple(
        gradient
        if gradient_mask & bit
        else gradient.new_empty((0,))
        for bit, gradient in zip((1, 2, 4), gradients)
    )


class _HardForwardSoftVjpFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        dropout_seed: Tensor,
        max_suffix_length: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
    ) -> Tensor:
        hard_output, packed_query_symbols, packed_key_symbols = (
            torch.ops.rosa_soft.hard_forward(
                query,
                key,
                value,
                int(max_suffix_length),
            )
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.save_for_backward(
            query,
            key,
            value,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
        )
        return hard_output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        (
            query,
            key,
            value,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
        ) = ctx.saved_tensors
        needs_input_grad = ctx.needs_input_grad[:3]
        gradients = torch.ops.rosa_soft.surrogate_vjp_masked(
            query,
            key,
            value,
            grad_output,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
            ctx.max_suffix_length,
            ctx.scale,
            ctx.dropout_p,
            ctx.mismatch_scale,
            _gradient_mask(needs_input_grad),
        )
        requested_gradients = _cast_requested_gradients(
            gradients,
            (query, key, value),
            needs_input_grad,
        )
        return (
            *requested_gradients,
            None,
            None,
            None,
            None,
            None,
        )


class _HardForwardSoftVjpVarlenFunction(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        cu_seqlens: Tensor,
        dropout_seed: Tensor,
        max_suffix_length: int,
        scale: float,
        dropout_p: float,
        mismatch_scale: float,
    ) -> Tensor:
        hard_output, packed_query_symbols, packed_key_symbols = (
            torch.ops.rosa_soft.hard_forward_varlen(
                query,
                key,
                value,
                cu_seqlens,
                int(max_suffix_length),
            )
        )
        ctx.max_suffix_length = int(max_suffix_length)
        ctx.scale = float(scale)
        ctx.dropout_p = float(dropout_p)
        ctx.mismatch_scale = float(mismatch_scale)
        ctx.save_for_backward(
            query,
            key,
            value,
            cu_seqlens,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
        )
        return hard_output

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output: Tensor):
        (
            query,
            key,
            value,
            cu_seqlens,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
        ) = ctx.saved_tensors
        needs_input_grad = ctx.needs_input_grad[:3]
        gradients = torch.ops.rosa_soft.surrogate_vjp_varlen_masked(
            query,
            key,
            value,
            cu_seqlens,
            grad_output,
            packed_query_symbols,
            packed_key_symbols,
            dropout_seed,
            ctx.max_suffix_length,
            ctx.scale,
            ctx.dropout_p,
            ctx.mismatch_scale,
            _gradient_mask(needs_input_grad),
        )
        requested_gradients = _cast_requested_gradients(
            gradients,
            (query, key, value),
            needs_input_grad,
        )
        return (
            *requested_gradients,
            None,
            None,
            None,
            None,
            None,
            None,
        )


def rosa_soft(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = ROSA_SOFT_DEFAULT_DROPOUT_P,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    """Run exact hard ROSA forward with a dense attention-style CUDA VJP."""

    max_suffix_length = _validate_cuda_call(
        query,
        key,
        value,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not needs_backward:
        return torch.ops.rosa_soft.hard_forward(
            query,
            key,
            value,
            max_suffix_length,
        )[0]

    dropout_seed = make_dropout_seed(query, dropout_p, needs_backward)
    return _HardForwardSoftVjpFunction.apply(
        query,
        key,
        value,
        dropout_seed,
        max_suffix_length,
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
    )


def rosa_soft_varlen(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    cu_seqlens: Tensor,
    *,
    max_suffix_length: int = 32,
    scale: float = ROSA_SOFT_DEFAULT_SCALE,
    dropout_p: float = ROSA_SOFT_DEFAULT_DROPOUT_P,
    mismatch_scale: float = ROSA_SOFT_DEFAULT_MISMATCH_SCALE,
) -> Tensor:
    """Run RosaSoft independently over CUDA-packed variable-length sequences."""

    max_suffix_length = _validate_cuda_varlen_call(
        query,
        key,
        value,
        cu_seqlens,
        max_suffix_length,
        scale,
        dropout_p,
        mismatch_scale,
    )
    needs_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad for tensor in (query, key, value)
    )
    if not needs_backward:
        return torch.ops.rosa_soft.hard_forward_varlen(
            query,
            key,
            value,
            cu_seqlens,
            max_suffix_length,
        )[0]

    dropout_seed = make_dropout_seed(query, dropout_p, needs_backward)
    return _HardForwardSoftVjpVarlenFunction.apply(
        query,
        key,
        value,
        cu_seqlens,
        dropout_seed,
        max_suffix_length,
        float(scale),
        float(dropout_p),
        float(mismatch_scale),
    )
