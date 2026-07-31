"""Validate the exact finite-window diagonal recurrence used by RosaSoft."""

from __future__ import annotations

import argparse
import json

import torch
from torch import Tensor


def _check_inputs(log_gates: Tensor, max_suffix_length: int) -> None:
    if log_gates.ndim < 1:
        raise ValueError("log_gates must have at least one dimension")
    if max_suffix_length < 1:
        raise ValueError("max_suffix_length must be >= 1")


def _affine_prefix_values(
    coefficient: Tensor,
    bias: Tensor,
    group_size: int = 32,
) -> Tensor:
    """Evaluate affine prefixes in warp-sized groups with a serial carry."""

    carry = torch.zeros_like(bias[..., 0])
    values = []
    for group_start in range(0, coefficient.size(-1), group_size):
        group_end = min(
            group_start + group_size,
            coefficient.size(-1),
        )
        scan_coefficient = coefficient[..., group_start:group_end]
        scan_bias = bias[..., group_start:group_end]
        shift = 1
        while shift < scan_coefficient.size(-1):
            right_coefficient = scan_coefficient[..., shift:]
            composed_coefficient = (
                right_coefficient * scan_coefficient[..., :-shift]
            )
            composed_bias = (
                right_coefficient * scan_bias[..., :-shift]
                + scan_bias[..., shift:]
            )
            scan_coefficient = torch.cat(
                (
                    scan_coefficient[..., :shift],
                    composed_coefficient,
                ),
                dim=-1,
            )
            scan_bias = torch.cat(
                (scan_bias[..., :shift], composed_bias),
                dim=-1,
            )
            shift <<= 1
        group_values = (
            scan_coefficient * carry.unsqueeze(-1) + scan_bias
        )
        values.append(group_values)
        carry = group_values[..., -1]
    return torch.cat(values, dim=-1)


def _window_correction(
    log_gates: Tensor,
    max_suffix_length: int,
) -> Tensor:
    """Return each removable product of length ``W + 1``.

    CUDA should derive these local sums from integer mismatch counts. The
    prototype uses a local window reduction to avoid cancellation from
    subtracting two long floating-point prefix sums.
    """

    correction = torch.zeros_like(log_gates)
    length = log_gates.size(-1)
    if length <= max_suffix_length:
        return correction
    window_logs = log_gates.unfold(
        -1,
        max_suffix_length + 1,
        1,
    ).sum(
        dim=-1,
    )
    correction[..., max_suffix_length :] = window_logs.exp()
    return correction


def finite_suffix_scores_scan(
    log_gates: Tensor,
    max_suffix_length: int,
) -> Tensor:
    """Compute finite suffix scores with an associative affine scan."""

    _check_inputs(log_gates, max_suffix_length)
    gates = log_gates.exp()
    correction = _window_correction(log_gates, max_suffix_length)
    return _affine_prefix_values(gates, gates - correction)


def finite_suffix_log_gate_vjp_scan(
    log_gates: Tensor,
    route_score_vjp: Tensor,
    max_suffix_length: int,
) -> Tensor:
    """Return the exact VJP with respect to ``log_gates``."""

    _check_inputs(log_gates, max_suffix_length)
    if route_score_vjp.shape != log_gates.shape:
        raise ValueError("route_score_vjp must match log_gates")
    gates = log_gates.exp()
    correction = _window_correction(log_gates, max_suffix_length)
    scores = finite_suffix_scores_scan(
        log_gates,
        max_suffix_length,
    )

    reverse_gates = gates.flip(-1)
    reverse_coefficient = torch.cat(
        (
            torch.zeros_like(reverse_gates[..., :1]),
            reverse_gates[..., :-1],
        ),
        dim=-1,
    )
    reverse_score_vjp = _affine_prefix_values(
        reverse_coefficient,
        route_score_vjp.flip(-1),
    )
    score_vjp = reverse_score_vjp.flip(-1)

    correction_vjp = score_vjp * correction
    padded_correction_vjp = torch.nn.functional.pad(
        correction_vjp,
        (0, max_suffix_length),
    )
    future_correction_vjp = padded_correction_vjp.unfold(
        -1,
        max_suffix_length + 1,
        1,
    ).sum(dim=-1)
    return (
        score_vjp * (scores + correction)
        - future_correction_vjp
    )


def direct_finite_suffix_scores(
    log_gates: Tensor,
    max_suffix_length: int,
) -> Tensor:
    """Vectorized oracle that explicitly advances every prefix product."""

    _check_inputs(log_gates, max_suffix_length)
    gates = log_gates.exp()
    product = gates
    scores = product
    for _ in range(1, min(max_suffix_length, gates.size(-1))):
        product = gates * torch.nn.functional.pad(
            product[..., :-1],
            (1, 0),
        )
        scores = scores + product
    return scores


def diagonal_suffix_scores(
    local_match_gates: Tensor,
    max_suffix_length: int,
) -> Tensor:
    """Map the scan to the causal non-null diagonals of ``[..., T, T]``."""

    if (
        local_match_gates.ndim < 2
        or local_match_gates.size(-1) != local_match_gates.size(-2)
    ):
        raise ValueError("local_match_gates must end in a square matrix")
    seq_len = local_match_gates.size(-1)
    scores = torch.zeros_like(local_match_gates)
    for diagonal_offset in range(seq_len - 1):
        routes = torch.arange(
            1,
            seq_len - diagonal_offset,
            device=local_match_gates.device,
        )
        rows = routes + diagonal_offset
        diagonal_log_gates = local_match_gates[
            ..., rows, routes
        ].log()
        scores[..., rows, routes] = finite_suffix_scores_scan(
            diagonal_log_gates,
            max_suffix_length,
        )
    return scores


def diagonal_suffix_log_gate_vjp(
    local_match_gates: Tensor,
    route_score_vjp: Tensor,
    max_suffix_length: int,
) -> Tensor:
    """Map exact suffix-score adjoints back to the causal local gates."""

    if (
        local_match_gates.ndim < 2
        or local_match_gates.size(-1) != local_match_gates.size(-2)
    ):
        raise ValueError("local_match_gates must end in a square matrix")
    if route_score_vjp.shape != local_match_gates.shape:
        raise ValueError("route_score_vjp must match local_match_gates")
    seq_len = local_match_gates.size(-1)
    log_gate_vjp = torch.zeros_like(local_match_gates)
    for diagonal_offset in range(seq_len - 1):
        routes = torch.arange(
            1,
            seq_len - diagonal_offset,
            device=local_match_gates.device,
        )
        rows = routes + diagonal_offset
        diagonal_gates = local_match_gates[..., rows, routes]
        log_gate_vjp[..., rows, routes] = (
            finite_suffix_log_gate_vjp_scan(
                diagonal_gates.log(),
                route_score_vjp[..., rows, routes],
                max_suffix_length,
            )
        )
    return log_gate_vjp


def diagonal_symbol_vjp(
    query_symbols: Tensor,
    key_symbols: Tensor,
    route_score_vjp: Tensor,
    max_suffix_length: int,
    mismatch_scale: float,
) -> tuple[Tensor, Tensor]:
    """Compute the exact diagonal-owned VJP for the Q/K symbol tensors.

    The final key position is intentionally unused: non-null route ``r``
    compares against key position ``r - 1``.
    """

    if query_symbols.shape != key_symbols.shape:
        raise ValueError("query_symbols and key_symbols must match")
    if query_symbols.ndim < 2:
        raise ValueError("symbol tensors must end in [T, D]")
    seq_len = query_symbols.size(-2)
    symbol_dim = query_symbols.size(-1)
    expected_route_shape = query_symbols.shape[:-1] + (seq_len,)
    if route_score_vjp.shape != expected_route_shape:
        raise ValueError(
            "route_score_vjp must match the symbol leading dimensions "
            "and end in [T, T]"
        )
    _check_inputs(query_symbols[..., 0], max_suffix_length)
    if mismatch_scale <= 0:
        raise ValueError("mismatch_scale must be > 0")

    grad_query = torch.zeros_like(query_symbols)
    grad_key = torch.zeros_like(key_symbols)
    symbol_vjp_scale = float(mismatch_scale) / (2.0 * symbol_dim)
    for diagonal_offset in range(seq_len - 1):
        routes = torch.arange(
            1,
            seq_len - diagonal_offset,
            device=query_symbols.device,
        )
        rows = routes + diagonal_offset
        key_positions = routes - 1
        diagonal_query = query_symbols[..., rows, :]
        diagonal_key = key_symbols[..., key_positions, :]
        mismatch_rate = 0.5 * (
            1.0 - diagonal_query * diagonal_key
        ).mean(dim=-1)
        log_gates = -float(mismatch_scale) * mismatch_rate
        log_gate_vjp = finite_suffix_log_gate_vjp_scan(
            log_gates,
            route_score_vjp[..., rows, routes],
            max_suffix_length,
        )
        scaled_vjp = symbol_vjp_scale * log_gate_vjp.unsqueeze(-1)
        grad_query.index_add_(
            -2,
            rows,
            scaled_vjp * diagonal_key,
        )
        grad_key.index_add_(
            -2,
            key_positions,
            scaled_vjp * diagonal_query,
        )
    return grad_query, grad_key


def _dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float64": torch.float64,
    }[name]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument(
        "--lengths",
        type=int,
        nargs="+",
        default=[1, 2, 7, 32, 65],
    )
    parser.add_argument(
        "--windows",
        type=int,
        nargs="+",
        default=[1, 2, 8, 32, 128],
    )
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--include-records", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = _dtype(args.dtype)
    records = []
    for length in args.lengths:
        for window in args.windows:
            for seed in range(args.seeds):
                generator = torch.Generator(device=device).manual_seed(seed)
                log_gates = (
                    -6.0
                    * torch.rand(
                        3,
                        length,
                        device=device,
                        dtype=dtype,
                        generator=generator,
                    )
                ).requires_grad_()
                route_score_vjp = torch.randn(
                    log_gates.shape,
                    device=device,
                    dtype=dtype,
                    generator=generator,
                )
                direct_scores = direct_finite_suffix_scores(
                    log_gates,
                    window,
                )
                expected_vjp = torch.autograd.grad(
                    (direct_scores * route_score_vjp).sum(),
                    log_gates,
                )[0]
                scan_scores = finite_suffix_scores_scan(
                    log_gates.detach(),
                    window,
                )
                scan_vjp = finite_suffix_log_gate_vjp_scan(
                    log_gates.detach(),
                    route_score_vjp,
                    window,
                )
                records.append(
                    {
                        "length": length,
                        "window": window,
                        "seed": seed,
                        "score_max_abs_error": float(
                            (scan_scores - direct_scores.detach())
                            .abs()
                            .max()
                        ),
                        "vjp_max_abs_error": float(
                            (scan_vjp - expected_vjp).abs().max()
                        ),
                    }
                )
    result = {
        "device": str(device),
        "dtype": str(dtype),
        "cases": len(records),
        "score_max_abs_error": max(
            record["score_max_abs_error"]
            for record in records
        ),
        "vjp_max_abs_error": max(
            record["vjp_max_abs_error"]
            for record in records
        ),
    }
    if args.include_records:
        result["records"] = records
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
