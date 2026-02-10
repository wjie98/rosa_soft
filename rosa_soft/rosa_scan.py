import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from fla.ops.linear_attn import chunk_linear_attn

from .rosa_sam import RosaContext, RosaWork
from .utils import QuantizeFunction, repeat_kv, unfold_qk, decay_qk, gather_x


__all__ = [
    "rosa_scan_ops",
    "RosaScanWork",
]


def rosa_scan_ops(
        query: Tensor,
        key: Tensor,
        value: Tensor,
        scale: Optional[float] = None,
        exponent: float = 2.0,
        grad_eps: float = 1e-3,
        suffix_window: int = 8,
        suffix_factor: Optional[float] = 0.5,
        quantize_mode: str = "soft",
        schmitt_trigger: float = 0.0,
        async_op: bool = False,
) -> Union[Tensor, 'RosaScanWork']:
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
        grad_eps (float): Epsilon for gradient scaling in the STE quantization. Default: `1e-3`.
        suffix_window (int): Size of the lookback window for fingerprinting. Default: `8`.
        suffix_factor (Optional[float]): Decay factor for the window. If None, it will be dynamically set based on the window size. Default: `0.5`.
        quantize_mode (str): Mode for quantizing the inputs. One of "tanh", "soft", or "ste". Default: "ste".
        schmitt_trigger (float): Threshold for Schmitt trigger to prevent noise. Default: `0.0`.
        async_op (bool): Whether to return a work object for asynchronous execution. Default: `False`.

    Returns:
        Tensor: The result of the Hard SAM lookup if async_op is False.
        RosaScanWork: A work object for asynchronous execution if async_op is True.
    """

    work = RosaScanWork()
    work._future = RosaContext().update(
        query=query, key=key, value=value, mismatch=0,
        inspect=True, schmitt_trigger=schmitt_trigger,
        async_op=True,
    )

    work._params = RosaScanParams(
        scale=scale,
        exponent=exponent,
        grad_eps=grad_eps,
        suffix_window=suffix_window,
        suffix_factor=suffix_factor,
        quantize_mode=quantize_mode,
    )
    work._query_key_value = (query, key, value)

    if async_op:
        return work
    return work.wait()


class RosaScanWork:
    def __init__(self):
        self._future: RosaWork
        self._params: RosaScanParams
        self._query_key_value: Tuple[Tensor, Tensor, Tensor]

    def wait(self):
        if self._future is None:
            raise RuntimeError("wait() called twice")
        
        work = self._future
        params = self._params
        query, key, value = self._query_key_value

        x_hard, info = work.wait()
        params.info["x_hard"] = x_hard
        params.info.update(info)

        self._future = None
        self._params = None
        self._query_key_value = None

        return RosaScanFunction.apply(query, key, value, params)


class RosaScanParams:
    def __init__(self,
        scale: Optional[float],
        exponent: float,
        grad_eps: float,
        suffix_window: int,
        suffix_factor: Optional[float],
        quantize_mode: str,
    ):
        self.scale = scale
        self.exponent = exponent
        self.suffix_window = suffix_window
        self.suffix_factor = suffix_factor
        self.quantize_mode = quantize_mode
        self.grad_eps = grad_eps

        self.info: Dict[str, Tensor] = {}
        self._ctx: Optional[RosaContext] = None


class RosaScanFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, params: RosaScanParams):
        if "x_hard" in params.info:
            x_hard = params.info.pop("x_hard")
        else:
            x_hard, info = RosaContext().update(query, key, value, 0, inspect=True)
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
                length=length,
                endpos=endpos,
                scale=params.scale,
                exponent=params.exponent,
                grad_eps=params.grad_eps,
                suffix_window=params.suffix_window,
                suffix_factor=params.suffix_factor,
                quantize_mode=params.quantize_mode,
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
        length: Tensor, endpos: Tensor,
        scale: Optional[float], exponent: float,
        suffix_window: int,
        suffix_factor: Optional[float],
        quantize_mode: str,
        grad_eps: float,
):
    bsz, num_heads, seq_len, num_q_bits = query.size()
    bsz, num_kv_heads, seq_len, num_k_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

    num_qk_bits = num_q_bits
    assert num_q_bits == num_k_bits, "query and key must have the same number of bits"

    # q = torch.linspace(0, 1, 10, device=query.device)
    # qq = torch.quantile(query.flatten(), q, dim=0)
    # qk = torch.quantile(key.flatten(), q, dim=0)
    # qv = torch.quantile(value.flatten(), q, dim=0)
    # print(qq.tolist())
    # print(qk.tolist())
    # print(qv.tolist())

    ## quantize query, key, value
    if quantize_mode == "tanh":
        xq = torch.tanh(query)
        xk = torch.tanh(key)
        xv = torch.tanh(value)
    elif quantize_mode == "soft":
        xq = F.softsign(query)
        xk = F.softsign(key)
        xv = F.softsign(value)
    elif quantize_mode == "ste":
        xq = QuantizeFunction.apply(query, grad_eps)
        xk = QuantizeFunction.apply(key, grad_eps)
        xv = QuantizeFunction.apply(value, grad_eps)
    else:
        raise ValueError(f"Unsupported quantize_mode: {quantize_mode}, expected one of 'tanh', 'soft', or 'ste'")
    
    ## feature transformation
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
    xq = xq.reshape(bsz, num_heads, seq_len, 2 * num_q_bits * suffix_window)
    xk = xk.reshape(bsz, num_heads, seq_len, 2 * num_k_bits * suffix_window)

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

