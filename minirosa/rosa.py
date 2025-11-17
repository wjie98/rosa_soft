import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import torch._dynamo as dynamo

from torch import Tensor
from typing import *

from pathlib import Path
from functools import lru_cache

import logging
logger = logging.getLogger(__name__)


__all__ = [
    "RosaAttention",
]


from .base import RosaConfig, RosaBase


class RosaAttention(RosaBase):
    def __init__(
            self,
            config: RosaConfig,
            layer_idx: int,
    ):
        super().__init__(config=config)

        self.layer_idx = layer_idx
        self.num_heads = config.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_qk_bits = config.num_query_key_bits
        self.num_v_bits = config.num_value_bits

        self.n_rep = self.num_heads // self.num_kv_heads
        self.bits_tau = config.bits_tau
        self.attn_tau = config.attn_tau
        self.proxy_type = config.proxy_type

        bias = config.bias
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.num_qk_bits, bias=bias)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.num_qk_bits, bias=bias)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.num_v_bits, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.num_v_bits, config.hidden_size, bias=bias)

        self.v_emb0 = nn.Parameter(torch.full((self.num_heads * self.num_v_bits,), -1e-5))
        self.v_emb1 = nn.Parameter(torch.full((self.num_heads * self.num_v_bits,), +1e-5))

    def forward(
            self,
            hidden_states: Tensor,
            attention_mask = None,
            past_key_values = None,
            **kwargs,
    ):
        bsz, seq_len, _ = hidden_states.size()

        query_states: Tensor = self.q_proj(hidden_states).view(bsz, seq_len, -1, self.num_qk_bits).transpose(1, 2)
        key_states: Tensor = self.k_proj(hidden_states).view(bsz, seq_len, -1, self.num_qk_bits).transpose(1, 2)
        value_states: Tensor = self.v_proj(hidden_states).view(bsz, seq_len, -1, self.num_v_bits).transpose(1, 2)

        if past_key_values is None:
            output = rosa_bits_ops(
                query_states, key_states, value_states, attn_mask=attention_mask,
                alpha=self.get_rosa_ops_alpha(), proxy=self.proxy_type,
                bits_tau=self.bits_tau, attn_tau=self.attn_tau,
                host_ops=True, training=self.training,
            )
        else:
            raise NotImplementedError("RosaAttention with past_key_values is not implemented yet.")
        
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.v_emb1 * output + self.v_emb0 * (1 - output)
        return self.o_proj(output)


class RosaCache:
    def __init__(self):
        self.layers: Dict[int, Dict[int, Tensor]] = {}
    
    def __del__(self):
        if torch is None:
            return
        for _, cache in self.layers.items():
            for _, ctx in cache.items():
                torch.ops.torch_rosa.rosa_sam_free(ctx)
    
    def update(self,
            query_states: Tensor,
            key_states: Tensor,
            value_states: Tensor,
            layer_idx: int,
    ):
        if layer_idx not in self.layers:
            self.layers[layer_idx] = {}
        
        rq = torch.arange(query_states.size(-1), device=query_states.device)
        rk = torch.arange(key_states.size(-1), device=key_states.device)
        rv = torch.arange(value_states.size(-1), device=value_states.device)

        bq = ((query_states > 0).long() << rq).sum(dim=-1).cpu()
        bk = ((key_states > 0).long() << rk).sum(dim=-1).cpu()
        bv = ((value_states > 0).long() << rv).sum(dim=-1).cpu()

        cache = self.layers[layer_idx]

        bo = torch.zeros_like(bq)
        for b, (q, k, v) in enumerate(zip(bq, bk, bv)):
            if b not in cache:
                cache[b] = torch.zeros(q.size(0), dtype=torch.long, device="cpu")
                torch.ops.torch_rosa.rosa_sam_init(cache[b])
            
            bo[b] = torch.ops.torch_rosa.rosa_sam_update(cache[b], q, k, v, 0)
        
        output = (bo.to(query_states.device).unsqueeze_(-1) >> rv) & 1
        return output.type_as(query_states)


def rosa_bits_ops(
        query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
        alpha: Union[Tensor, float] = 1.0, proxy: Literal["rosa", "sufa"] = "rosa",
        bits_tau: Union[Tensor, float] = 1.0, attn_tau: Union[Tensor, float] = 1.0,
        host_ops: bool = False, training: bool = False, sufa_head_dim: int = 128,
):
    """
    Performs the Rapid Online Suffix Automaton (ROSA) attention-like operation.

    This function computes a differentiable, attention-like mechanism based on the
    longest common suffix match between query and key sequences. The inputs are
    expected to be binarized tensors (or tensors of logits that will be binarized).
    The operation is designed to be efficient on parallel hardware like GPUs.

    Args:
        query (Tensor): The query tensor of shape (B, H, T, D_qk). Can be logits.
        key (Tensor): The key tensor of shape (B, H_kv, T, D_qk). Can be logits.
        value (Tensor): The value tensor of shape (B, H_kv, T, D_v). Can be logits.
        attn_mask (Optional[Tensor]): An optional boolean attention mask.
        alpha (Union[Tensor, float]): The interpolation coefficient for the STE.
            - `alpha = 0.0`: The output is purely from the soft, differentiable proxy.
            - `alpha = 1.0`: The forward pass output is purely from the hard, discrete op.
            This should be annealed from 0 to 1 during training.
        proxy (Literal["rosa", "sufa"]): The type of soft proxy to use for gradient
            computation. 'rosa' uses a custom dynamic programming approach, while
            'sufa' uses a standard scaled dot-product attention as the proxy.
        bits_tau (Union[Tensor, float]): Temperature for converting query/key/value logits
            into continuous bit representations (e.g., via tanh or sigmoid). A smaller
            tau makes the bits "harder".
        attn_tau (Union[Tensor, float]): Temperature for the softmax in the soft attention
            calculation. A smaller tau makes the attention distribution sharper.
        host_ops (bool): If True, the 'hard' part of the operation is offloaded to the CPU
            for potentially faster execution. This requires a custom C++ extension.
        training (bool): If True, enables the training path which uses a soft proxy for
            gradients via a Straight-Through Estimator (STE).
        sufa_head_dim (int): The head dimension to use for the 'sufa' proxy's
            scaled dot-product attention.

    Returns:
        Tensor: The resulting attention output tensor, with the forward pass
                determined by the interpolation of hard and soft ops, and the
                backward pass determined solely by the soft op.
    """

    if host_ops:
        load_torch_rosa()
        xq, xk, xv = rosa_bits_host_ops_d2h(query, key, value, attn_mask=attn_mask)

    if training:
        if proxy == "rosa":
            x_soft = rosa_bits_soft_ops(
                query, key, value, attn_mask=attn_mask,
                bits_tau=bits_tau, attn_tau=attn_tau,
            )
        elif proxy == "sufa":
            x_soft = sufa_bits_soft_ops(
                query, key, value, attn_mask=attn_mask,
                bits_tau=bits_tau, attn_tau=attn_tau, head_dim=sufa_head_dim,
            )
        else:
            raise ValueError(f"Unknown proxy type: {proxy}")
    
    if host_ops:
        x_hard = rosa_bits_host_ops_h2d(query, key, value, xq, xk, xv)
    else:
        x_hard = rosa_bits_hard_ops(query, key, value, attn_mask=attn_mask)
    
    if training:
        return x_soft + (x_hard - x_soft).detach() * alpha
    else:
        return x_hard


@lru_cache(maxsize=None)
def load_torch_rosa():
    """
    JIT compiles and loads the 'torch_rosa' C++ extension.
    
    Thanks to the @lru_cache decorator, the compilation and loading process
    will only run once per Python session, no matter how many times this
    function is called.
    """
    logger.info("--- Compiling and loading torch_rosa C++ extension (this should happen only once) ---")

    rosa_cpp = Path(__file__).resolve().with_name("rosa.cpp")

    if not rosa_cpp.is_file():
        raise FileNotFoundError(
            f"The C++ source file was not found at the expected location: {rosa_cpp}"
        )
    
    from torch.utils.cpp_extension import load

    load(
        name="torch_rosa",
        sources=[str(rosa_cpp)],
        extra_cflags=["-O3", "-fopenmp"],
        is_python_module=False,
        verbose=True,
    )
    logger.info("--- torch_rosa C++ extension successfully loaded. ---")

ROSA_HOST_STREAM = torch.cuda.Stream() if torch.cuda.is_available() else None

@torch.no_grad()
def rosa_bits_host_ops_d2h(query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None):
    global ROSA_HOST_STREAM

    if attn_mask is not None and not getattr(rosa_bits_host_ops_d2h, "_warning_shown", False):
        logger.warning("The 'attn_mask' argument is provided but will be ignored in this bit-packing operation. "
                       "This warning will not be shown again.")
        rosa_bits_host_ops_d2h._warning_shown = True
    
    bsz, num_heads, seq_len, num_qk_bits = query.size()
    bsz, num_kv_heads, seq_len, num_qk_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = key.size()

    r = torch.arange(0, num_qk_bits, device=query.device)
    xq = ((query > 0).long() << r).sum(dim=-1).view(bsz, num_heads, seq_len)
    xk = ((key > 0).long() << r).sum(dim=-1).view(bsz, num_kv_heads, seq_len)

    r = torch.arange(0, num_v_bits, device=query.device)
    xv = ((value > 0).long() << r).sum(dim=-1).view(bsz, num_kv_heads, seq_len)

    if num_qk_bits <= 8 and num_v_bits <= 8:
        xq = xq.to(torch.uint8)
        xk = xk.to(torch.uint8)
        xv = xv.to(torch.uint8)
    
    if ROSA_HOST_STREAM is not None:
        ROSA_HOST_STREAM.wait_stream(torch.cuda.current_stream())

    with torch.cuda.stream(ROSA_HOST_STREAM):
        xq = xq.to("cpu", non_blocking=True)
        xk = xk.to("cpu", non_blocking=True)
        xv = xv.to("cpu", non_blocking=True)
    
    return xq, xk, xv

@torch.no_grad()
def rosa_bits_host_ops_h2d(query: Tensor, key: Tensor, value: Tensor, xq: Tensor, xk: Tensor, xv: Tensor):
    global ROSA_HOST_STREAM

    bsz, num_heads, seq_len, num_qk_bits = query.size()
    bsz, num_kv_heads, seq_len, num_qk_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = key.size()

    n_rep = num_heads // num_kv_heads
    if n_rep > 1:
        xk = xk.view(bsz, num_kv_heads, 1, seq_len).repeat(1, 1, n_rep, 1)
        xv = xv.view(bsz, num_kv_heads, 1, seq_len).repeat(1, 1, n_rep, 1)
    
    xq = xq.reshape(-1, seq_len)
    xk = xk.reshape(-1, seq_len)
    xv = xv.reshape(-1, seq_len)

    ctx = torch.zeros(bsz * num_heads, dtype=torch.long, device="cpu")
    try:
        if num_qk_bits <= 8 and num_v_bits <= 8:
            torch.ops.torch_rosa.rosa_sam_8bits_init(ctx)
            xo = torch.ops.torch_rosa.rosa_sam_8bits_update(ctx, xq, xk, xv, 0)
        else:
            torch.ops.torch_rosa.rosa_sam_init(ctx)
            xo = torch.ops.torch_rosa.rosa_sam_update(ctx, xq, xk, xv, 0)
    finally:
        if num_qk_bits <= 8 and num_v_bits <= 8:
            torch.ops.torch_rosa.rosa_sam_8bits_free(ctx)
        else:
            torch.ops.torch_rosa.rosa_sam_free(ctx)
    
    with torch.cuda.stream(ROSA_HOST_STREAM):
        xo = xo.to(query.device, non_blocking=True)
    
    if ROSA_HOST_STREAM is not None:
        torch.cuda.current_stream().wait_stream(ROSA_HOST_STREAM)
    
    r = torch.arange(0, num_v_bits, device=query.device)
    xo = (xo.long().view(bsz, num_heads, seq_len, 1) >> r) & 1
    return xo.type_as(value)


def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
    bsz, n_kvs, seq_len, dim = hidden_states.size()
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kvs, n_rep, seq_len, dim)
    return hidden_states.reshape(bsz, n_kvs * n_rep, seq_len, dim)


@torch.no_grad()
def rosa_bits_hard_ops(query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
    bsz, num_heads, seq_len, num_qk_bits = query.size()
    bsz, num_kv_heads, seq_len, num_qk_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = key.size()

    n_rep = num_heads // num_kv_heads

    r = torch.arange(0, num_qk_bits, device=query.device)
    xq = ((query > 0).long() << r).sum(dim=-1).view(bsz, num_heads, seq_len, 1)
    xk = ((key > 0).long() << r).sum(dim=-1).view(bsz, num_kv_heads, 1, seq_len)
    xv = (value > 0).type_as(value)

    xk = repeat_kv(xk, n_rep=n_rep)
    xv = repeat_kv(xv, n_rep=n_rep)

    ss = (xq == xk).tril_(-1)
    if attn_mask is not None:
        ss &= attn_mask
    
    ss = ss.long().roll(1, -1) # predict next token
    r = torch.arange(seq_len, dtype=torch.long, device=ss.device)

    # diagonals to columns
    c = (r.view(-1, 1) + r.view(1, -1) + 1) % seq_len
    ss = ss.gather(-1, c.expand(bsz, num_heads, seq_len, seq_len))

    # dynamic programming
    dp = ss.cumsum(dim=-2)
    ss = dp - (dp * (1 - ss)).cummax(dim=-2).values

    # columns to diagonals
    c = (r.view(1, -1) - r.view(-1, 1) + seq_len - 1) % seq_len
    ss = ss.gather(-1, c.expand(bsz, num_heads, seq_len, seq_len))

    # gather the largest scores
    mv = ss.max(dim=-1, keepdim=True).values
    ii = torch.where(ss == mv, r.view(1, 1, 1, -1), -1).max(dim=-1, keepdim=True).values

    output = xv.gather(-2, ii.repeat(1, 1, 1, num_v_bits))
    output = output.masked_fill_(mv <= 0, 0) # unmatched positions get zeroed out
    return output


def rosa_bits_soft_ops(
        query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
        bits_tau: Union[Tensor, float] = 1.0, attn_tau: Union[Tensor, float] = 1.0,
):
    bsz, num_heads, seq_len, num_qk_bits = query.size()
    bsz, num_kv_heads, seq_len, num_qk_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = key.size()

    n_rep = num_heads // num_kv_heads

    xq = torch.tanh(query / bits_tau)
    xk = torch.tanh(key / bits_tau)
    xv = torch.sigmoid(value / bits_tau)

    xk = repeat_kv(xk, n_rep=n_rep)
    xv = repeat_kv(xv, n_rep=n_rep)

    ss = xq @ xk.transpose(-1, -2) - (num_qk_bits - 1) * (1 - bits_tau)
    ss = torch.sigmoid(ss / bits_tau).tril(-1)
    if attn_mask is not None:
        ss = ss.masked_fill(~attn_mask, 0.0)
    
    ss = F.pad(ss, (1, -1), value=0.0) # predict next token
    r = torch.arange(seq_len, dtype=torch.long, device=ss.device)

    # diagonals to columns
    c = (r.view(-1, 1) + r.view(1, -1) + 1) % seq_len
    ss = ss.gather(-1, c.expand(bsz, num_heads, seq_len, seq_len))

    # dynamic programming
    dp = ss.cumsum(dim=-2)
    ss = dp - (dp * (1 - ss)).cummax(dim=-2).values

    # columns to diagonals
    c = (r.view(1, -1) - r.view(-1, 1) + seq_len - 1) % seq_len
    ss = ss.gather(-1, c.expand(bsz, num_heads, seq_len, seq_len))

    # calculate attention bias
    sharpness = 8192.0
    attn_bias = r.view(-1, 1) - r.view(1, -1) + 2
    attn_bias = 1 - torch.log(attn_bias.type_as(ss)).clamp_min_(1.0) / math.log(sharpness)
    attn_bias = attn_bias.clamp_min_(0.0).view(1, 1, seq_len, seq_len).tril_()
    attn_bias = attn_bias + torch.full_like(attn_bias, -torch.inf).triu_(1)

    # softmax attention
    mv = ss.max(dim=-1, keepdim=True).values
    ii = torch.softmax((ss + attn_bias) / attn_tau, dim=-1)

    output = ii @ xv
    output = output * mv.clamp(0.0, 1.0) # unmatched positions get zeroed out
    return output


@torch.no_grad()
def sufa_attn_bias(
        query: Tensor,
        attention_mask: Optional[Tensor] = None,
        sharpness: float = 8192.0,
):
    bsz, _, seq_len, _ = query.size()

    attn_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=query.device).tril_(-1)
    if attention_mask is not None:
        attn_mask = attn_mask & attention_mask
    
    r = torch.arange(0, seq_len, dtype=query.dtype, device=query.device)
    attn_bias = 1 - torch.log(r.view(-1, 1) - r.view(1, -1) + 1).clamp_min_(1.0) / math.log(sharpness)
    attn_bias = attn_bias.clamp_min_(0.0).expand_as(attn_mask).clone()

    attn_bias[~attn_mask] = -torch.inf
    return attn_bias

def sufa_bits_soft_ops(
        query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
        bits_tau: Union[Tensor, float] = 1.0, attn_tau: Union[Tensor, float] = 1.0,
        head_dim: int = 128,
):
    bsz, num_heads, seq_len, num_qk_bits = query.size()
    bsz, num_kv_heads, seq_len, num_qk_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = key.size()

    n_rep = num_heads // num_kv_heads
    enable_gqa = n_rep > 1

    xq = torch.tanh(query / bits_tau)
    xk = torch.tanh(key / bits_tau)
    xv = torch.sigmoid(value / bits_tau)

    # xk = repeat_kv(xk, n_rep=n_rep)
    # xv = repeat_kv(xv, n_rep=n_rep)

    win_qk_size = head_dim // num_qk_bits

    xq = F.pad(xq, (0, 0, win_qk_size - 1, 0), value=0.0).unfold(-2, win_qk_size, 1).transpose(-1, -2).reshape(bsz, -1, seq_len, head_dim)
    xk = F.pad(xk, (0, 0, win_qk_size - 1, 0), value=0.0).unfold(-2, win_qk_size, 1).transpose(-1, -2).reshape(bsz, -1, seq_len, head_dim)

    attn_bias = sufa_attn_bias(query, attention_mask=attn_mask) / attn_tau
    with dynamo.config.patch(capture_scalar_outputs=True):
        scale = 1.0 / math.sqrt(head_dim) / (attn_tau.item() if isinstance(attn_tau, Tensor) else attn_tau)
        output = F.scaled_dot_product_attention(
            xq, xk, xv, scale=scale,
            attn_mask=attn_bias, enable_gqa=enable_gqa,
        )
    
    return output
