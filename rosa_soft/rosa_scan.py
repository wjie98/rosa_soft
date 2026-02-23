import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from fla.ops.linear_attn import chunk_linear_attn

from .rosa_sam import RosaContext, RosaWork
from .utils import QuantizeFunction, repeat_kv, unfold_qk, decay_qk, gather_x

from .rosa_soft import RosaSoftWork


__all__ = [
    "rosa_scan_ops",
]


def rosa_scan_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: Optional[float] = None,
        exponent: float = 2.0,
        suffix_window: int = 8,
        suffix_factor: Optional[float] = 0.5,
        quant_mode: str = "soft",
        quant_scale: Optional[float] = None,
        schmitt_trigger: float = 0.0,
        expansion_factor: int = 2,
        async_op: bool = False,
) -> Union[Tensor, RosaSoftWork]:
    """
    Performs the Rapid Online Suffix Automaton (ROSA) scan-like operation.

    This function computes a differentiable, scan-like mechanism based on the
    longest common suffix match between query and key sequences. The inputs are
    expected to be tensors of logits that will be binarized. The operation is designed
    to be efficient on parallel hardware like GPUs.

    Args:
        query (Tensor): (B, H, T, D) Logits for Query bits.
        key (Tensor): (B, H, T, D) Logits for Key bits.
        value (Tensor): (B, H, T, D_v) Logits for Value bits.
        scale (Optional[float]): Scale factor for the attention scores. If not provided, it will default to `1 / sqrt(D)`. Default: `None`.
        exponent (float): Exponent for the feature transformation. Default: `2.0`.
        suffix_window (int): Size of the lookback window for fingerprinting. Default: `8`.
        suffix_factor (Optional[float]): Decay factor for the window. If None, it will be dynamically set based on the window size. Default: `0.5`.
        quant_mode (str): Mode for quantizing the inputs. Supported values are "tanh" for tanh-based quantization and "soft" for softsign-based quantization. Default: `"soft"`.
        quant_scale (Optional[float]): Scale factor for quantization. If not provided, it will default to `1.0`. Default: `None`.
        schmitt_trigger (float): Threshold for Schmitt trigger to prevent noise. Default: `0.0`.
        expansion_factor (int): Factor to expand the feature dimension for better expressiveness. Default: `2`.
        async_op (bool): Whether to return a work object for asynchronous execution. Default: `False`.

    Returns:
        Tensor: The result of the Hard SAM lookup if async_op is False.
        RosaSoftWork: A work object for asynchronous execution if async_op is True.
    """

    work = RosaSoftWork()
    work._future = RosaContext().update(
        query=query, key=key, value=value,
        schmitt_trigger=schmitt_trigger,
        async_op=True,
    )

    work._params = RosaScanParams(
        scale=scale,
        exponent=exponent,
        suffix_window=suffix_window,
        suffix_factor=suffix_factor,
        quant_mode=quant_mode,
        quant_scale=quant_scale,
        expansion_factor=expansion_factor,
        schmitt_trigger=schmitt_trigger,
    )

    work._function_apply = RosaScanFunction.apply
    work._query_key_value = (query, key, value)

    if async_op:
        return work
    return work.wait()


class RosaScanParams:
    def __init__(self,
        scale: Optional[float],
        exponent: float,
        suffix_window: int,
        suffix_factor: Optional[float],
        quant_mode: str,
        quant_scale: Optional[float],
        expansion_factor: int,
        schmitt_trigger: float,
    ):
        self.scale = scale
        self.exponent = exponent
        self.suffix_window = suffix_window
        self.suffix_factor = suffix_factor
        self.quant_mode = quant_mode
        self.quant_scale = quant_scale
        self.expansion_factor = expansion_factor
        self.schmitt_trigger = schmitt_trigger

        self.info: Dict[str, Tensor] = {}
        self._ctx: Optional[RosaContext] = None


class RosaScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, params: RosaScanParams):
        if "x_hard" in params.info:
            x_hard = params.info.pop("x_hard")
        else:
            x_hard, info = RosaContext().update(
                query=query, key=key, value=value,
                schmitt_trigger=params.schmitt_trigger,
            )
            params.info.update(info)

        ctx.save_for_backward(
            query.detach(),
            key.detach(),
            value.detach(),
        )
        ctx.saved_params = params

        return x_hard
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, value = cast(Tuple[Tensor, ...], ctx.saved_tensors)
        params: RosaScanParams = ctx.saved_params

        length = params.info.pop("length")
        endpos = params.info.pop("endpos")

        with torch.enable_grad():
            query.requires_grad_(True)
            key.requires_grad_(True)
            value.requires_grad_(True)

            x_soft = suffix_linear_attention_proxy(
                query, key, value,
                endpos=endpos,
                scale=params.scale,
                exponent=params.exponent,
                suffix_window=params.suffix_window,
                suffix_factor=params.suffix_factor,
                quant_mode=params.quant_mode,
                quant_scale=params.quant_scale,
                expansion_factor=params.expansion_factor,
            )

            grad_query, grad_key, grad_value = torch.autograd.grad(
                outputs=x_soft,
                inputs=(query, key, value),
                grad_outputs=grad_output,
                retain_graph=False,
                only_inputs=True,
            )
        return grad_query, grad_key, grad_value, None


def suffix_linear_attention_proxy(
        query: Tensor, key: Tensor, value: Tensor,
        endpos: Tensor,
        scale: Optional[float],
        exponent: float,
        suffix_window: int,
        suffix_factor: Optional[float],
        quant_mode: str,
        quant_scale: Optional[float],
        expansion_factor: int,
):
    bsz, num_heads, seq_len, num_q_bits = query.size()
    bsz, num_kv_heads, seq_len, num_k_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

    assert num_q_bits == num_k_bits, "query and key must have the same number of bits"

    # q = torch.linspace(0, 1, 10, device=query.device)
    # qq = torch.quantile(query.flatten(), q, dim=0)
    # qk = torch.quantile(key.flatten(), q, dim=0)
    # qv = torch.quantile(value.flatten(), q, dim=0)
    # print(qq.tolist())
    # print(qk.tolist())
    # print(qv.tolist())

    ## quantize query, key, value

    if quant_scale is not None:
        assert quant_scale >= 1.0, "quant_scale must be >= 1.0"
        query = query * quant_scale
        key = key * quant_scale
        value = value * quant_scale
    else:
        quant_scale = 1.0

    if quant_mode == "tanh":
        xq = torch.tanh(query)
        xk = torch.tanh(key)
        xv = torch.tanh(value)
    elif quant_mode == "soft":
        xq = F.softsign(query)
        xk = F.softsign(key)
        xv = F.softsign(value)
    else:
        raise ValueError(f"Unsupported quant_mode: {quant_mode}, expected 'tanh' or 'soft'")
    
    ## feature transformation
    num_qk_bits = num_q_bits * expansion_factor
    mat = torch.randint(
        0, 2,
        (num_q_bits, num_qk_bits),
        dtype=xq.dtype,
        device=xq.device,
    ).mul_(2).sub_(1).div_(math.sqrt(num_qk_bits))
    
    xq = xq @ mat
    xk = xk @ mat

    xq = torch.cat([
        F.relu( xq).mul(2).pow(exponent),
        F.relu(-xq).mul(2).pow(exponent),
    ], dim=-1)

    xk = torch.cat([
        F.relu( xk).mul(2).pow(exponent),
        F.relu(-xk).mul(2).pow(exponent),
    ], dim=-1)

    ## repeat key and value for multi-head attention
    n_rep = num_heads // num_kv_heads
    xk = repeat_kv(xk, n_rep)
    xv = repeat_kv(xv, n_rep)

    ## unfold query and key for suffix attention
    xq = unfold_qk(xq, win_size=suffix_window, offset=0)
    xk = unfold_qk(xk, win_size=suffix_window, offset=1) # predict next token

    ## apply geometric decay to the query and key features within the suffix window
    xq, xk = decay_qk(xq, xk, decay_factor=suffix_factor)

    ## compute linear attention
    xq = xq.reshape(bsz, num_heads, seq_len, 2 * num_qk_bits * suffix_window)
    xk = xk.reshape(bsz, num_heads, seq_len, 2 * num_qk_bits * suffix_window)

    if scale is None:
        scale = 1.0 / math.sqrt(num_qk_bits)
    else:
        scale = float(scale)

    xo: Tensor = chunk_linear_attn(
        xq.transpose(1, 2),
        xk.transpose(1, 2),
        xv.transpose(1, 2),
        scale=scale,
        normalize=True,
    )[0].transpose(1, 2)

    ## apply gating based value detach
    pk = gather_x(xk, endpos)
    pv = gather_x(xv, endpos)

    gg = torch.sum(xq * pk, dim=-1, keepdim=True)
    gg = torch.sigmoid(gg * scale)
    
    xo = xo * (1 - gg) + pv * gg
    return xo

