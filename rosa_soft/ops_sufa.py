import torch
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from .sam import RosaContext
from .future import RosaSoftWork

from .utils import quantize


__all__ = [
    "rosa_sufa_ops",
]


def rosa_sufa_ops(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    scale: Optional[float] = None,
    suffix_window: int = 8,
    suffix_factor: Optional[float] = 0.5,
    quant_mode: str = "soft",
    quant_scale: Optional[float] = None,
    schmitt_trigger: float = 0.0,
    async_op: bool = False,
) -> Union[Tensor, RosaSoftWork]:
    """ROSA Suffix Attention operator (SUFA, Production Default).

    Approximates soft DP via windowed suffix vectors (suffix fingerprints,
    essentially n-gram representations with positional decay). Constructs
    hyper-vectors from last W tokens for dot-product matching.
    Compatible with Flash Attention optimization.

    Complexity: O(T^2) time, O(T) space (with Flash Attention).

    Args:
        query: (B, T, H, D) Query logits.
        key: (B, T, H, D) Key logits.
        value: (B, T, H_v, D_v) Value logits.
        scale: Attention scale factor.
        suffix_window: Size of suffix lookback window (W).
        suffix_factor: Decay factor for windowed aggregation.
        quant_mode: Quantization mode ("tanh" or "soft").
        quant_scale: Quantization scale factor.
        schmitt_trigger: Threshold for noise filtering.
        async_op: Return work object for async execution.

    Returns:
        Tensor if async_op=False, RosaSoftWork if async_op=True.
    """

    batch_size, _, num_heads, _ = query.size()

    work = RosaSoftWork()

    work._future = RosaContext(
        batch_size=batch_size,
        num_heads=num_heads,
    ).update(
        query=query, key=key, value=value,
        schmitt_trigger=schmitt_trigger,
        async_op=True,
    )

    work._params = RosaSufaParams(
        scale=scale,
        suffix_window=suffix_window,
        suffix_factor=suffix_factor,
        quant_mode=quant_mode,
        quant_scale=quant_scale,
        schmitt_trigger=schmitt_trigger,
    )

    work._function_apply = RosaSufaFunction.apply
    work._query_key_value = (query, key, value)

    if async_op:
        return work
    return work.wait()


class RosaSufaParams:
    def __init__(self,
        scale: Optional[float],
        suffix_window: int,
        suffix_factor: float,
        quant_mode: str,
        quant_scale: Optional[float],
        schmitt_trigger: float,
    ):
        self.scale = scale
        self.suffix_window = suffix_window
        self.suffix_factor = suffix_factor
        self.quant_mode = quant_mode
        self.quant_scale = quant_scale
        self.schmitt_trigger = schmitt_trigger

        self.info: Dict[str, Tensor] = {}


class RosaSufaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, value: Tensor, params: RosaSufaParams):
        x_hard = params.info.pop("x_hard")

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
        params: RosaSufaParams = ctx.saved_params

        length = params.info.pop("length")
        endpos = params.info.pop("endpos")

        query.requires_grad_(True)
        key.requires_grad_(True)
        value.requires_grad_(True)

        with torch.enable_grad():
            x_soft = suffix_attention_proxy(
                query, key, value,
                endpos=endpos,
                scale=params.scale,
                suffix_window=params.suffix_window,
                suffix_factor=params.suffix_factor,
                quant_mode=params.quant_mode,
                quant_scale=params.quant_scale,
            )

        grad_query, grad_key, grad_value = torch.autograd.grad(
            outputs=x_soft,
            inputs=(query, key, value),
            grad_outputs=grad_output,
            retain_graph=False,
            only_inputs=True,
        )

        return grad_query, grad_key, grad_value, None


def suffix_attention_proxy(
        query: Tensor, key: Tensor, value: Tensor,
        endpos: Tensor,
        scale: Optional[float],
        suffix_window: int,
        suffix_factor: float,
        quant_mode: str,
        quant_scale: Optional[float],
):
    bsz, seq_len, num_q_heads, num_q_bits = query.size()
    bsz, seq_len, num_k_heads, num_k_bits = key.size()
    bsz, seq_len, num_v_heads, num_v_bits = value.size()

    assert num_q_heads == num_k_heads, "num_q_heads must be equal to num_k_heads"
    assert num_q_heads % num_v_heads == 0, "num_q_heads must be divisible by num_v_heads"

    MAX_ATTENTION_HEAD_DIM = 256
    assert num_q_bits == num_k_bits, "num_q_bits must be equal to num_k_bits"
    assert 0 < num_q_bits * suffix_window <= MAX_ATTENTION_HEAD_DIM, \
        f"num_q_bits * suffix_window ({num_q_bits * suffix_window}) must be <= {MAX_ATTENTION_HEAD_DIM}"

    assert suffix_window >= 1, "suffix_window must be >= 1"
    assert 0 < suffix_factor <= 1.0, "suffix_factor must be in (0, 1]"

    query = query.permute(0, 2, 1, 3)
    key   = key.permute(0, 2, 1, 3)
    value = value.permute(0, 2, 1, 3)

    # q = torch.linspace(0, 1, 10, device=query.device)
    # qq = torch.quantile(query.flatten(), q, dim=0)
    # qk = torch.quantile(key.flatten(), q, dim=0)
    # qv = torch.quantile(value.flatten(), q, dim=0)
    # print(qq.tolist())
    # print(qk.tolist())
    # print(qv.tolist())

    ## quantize query, key, value
    xq, xk, xv = quantize(
        query=query, key=key, value=value,
        quant_mode=quant_mode, quant_scale=quant_scale,
    )
    
    ## repeat value for multi-head attention
    n_rep = num_q_heads // num_v_heads
    if n_rep > 1:
        xv = xv[:, :, None, :, :].expand(bsz, num_v_heads, n_rep, seq_len, num_v_bits)
        xv = xv.reshape(bsz, num_q_heads, seq_len, num_v_bits)

    ## unfold query and key for suffix attention
    xq = F.pad(xq, (0, 0, suffix_window - 1, 0)).unfold(-2, suffix_window, 1).transpose(-2, -1)
    xk = F.pad(xk, (0, 0, suffix_window,    -1)).unfold(-2, suffix_window, 1).transpose(-2, -1)

    ## apply geometric decay to the query and key features within the suffix window
    with torch.no_grad():
        decs = suffix_window - 1 - torch.arange(suffix_window, device=xq.device)
        decs = torch.pow(suffix_factor, decs.float())
        decs = torch.sqrt(decs / torch.sum(decs)).view(-1, 1)

    xq = xq * decs.to(xq.dtype)
    xk = xk * decs.to(xk.dtype)

    ## compute scaled dot product attention
    xq = xq.reshape(bsz, num_q_heads, seq_len, num_q_bits * suffix_window)
    xk = xk.reshape(bsz, num_k_heads, seq_len, num_k_bits * suffix_window)

    if scale is None:
        scale = 1.0 / math.sqrt(num_q_bits * suffix_window)
    else:
        scale = float(scale)

    xo = F.scaled_dot_product_attention(
        xq, xk, xv, scale=scale,
        is_causal=True, attn_mask=None,
    )

    ## apply gating based value detach
    with torch.no_grad():
        epos = endpos.permute(0, 2, 1)
        mask = (epos >= 0).unsqueeze(-1).type_as(xq)
        inds = (epos + 1).clamp(0, seq_len - 1).unsqueeze(-1).long()
    
    sk = torch.gather(xk, dim=2, index=inds) * mask
    sv = torch.gather(xv, dim=2, index=inds) * mask

    gg = torch.sum(xq * sk, dim=-1, keepdim=True)
    gg = torch.sigmoid(gg * scale)
    xo = xo * (1 - gg) + sv * gg
    
    return xo.permute(0, 2, 1, 3)

