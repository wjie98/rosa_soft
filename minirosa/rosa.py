import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import math
import torch._dynamo as dynamo

from torch import Tensor
from typing import *

from pathlib import Path

import logging
logger = logging.getLogger(__name__)


__all__ = [
    "RosaAttention",
]


def load_torch_rosa():
    from torch.utils.cpp_extension import load
    rosa_cpp = Path(__file__).resolve().with_name("rosa.cpp")
    load(
        name="torch_rosa",
        sources=[str(rosa_cpp)],
        extra_cflags=["-O3", "-fopenmp"],
        is_python_module=False,
        verbose=True,
    )

if os.getenv("TORCH_ROSA_CPP", "0") == "1":
    load_torch_rosa()


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
        bits_tau: Union[Tensor, float] = 1.0, attn_tau: Union[Tensor, float] = 1.0, sufa_head_dim: int = 128,
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
        sufa_head_dim (int): The head dimension to use for the 'sufa' proxy's
            scaled dot-product attention.

    Returns:
        Tensor: The resulting attention output tensor, with the forward pass
                determined by the interpolation of hard and soft ops, and the
                backward pass determined solely by the soft op.
    """

    if proxy == "rosa":
        x_hard = rosa_bits_hard_ops(query, key, value, attn_mask=attn_mask)
        x_soft = rosa_bits_soft_ops(
            query, key, value, attn_mask=attn_mask,
            bits_tau=bits_tau, attn_tau=attn_tau,
        )
    elif proxy == "sufa":
        x_hard = rosa_bits_hard_ops(query, key, value, attn_mask=attn_mask)
        x_soft = sufa_bits_soft_ops(
            query, key, value, attn_mask=attn_mask,
            bits_tau=bits_tau, attn_tau=attn_tau, head_dim=sufa_head_dim,
        )
    else:
        raise ValueError(f"Unknown proxy type: {proxy}")
    
    return x_soft + (x_hard - x_soft).detach() * alpha


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


class _RosaState:
    __slots__ = ("endpos", "length", "suffix_link", "transitions")

    def __init__(self):
        self.endpos = -1
        self.length = 0
        self.suffix_link: Optional[_RosaState] = None
        self.transitions: Dict[int, _RosaState] = {}


class RapidOnlineSuffixAutomaton:
    """
    Implements a classic Suffix Automaton for online sequence processing.

    This class builds a suffix automaton character by character (or token by token)
    to efficiently track all substrings of a given key sequence. It is extended here
    to also handle a query sequence, enabling it to find the longest suffix of the
    current query that is also a substring of the keys processed so far.
    """

    def __init__(self):
        self.query_states: List[int] = []
        self.key_states: List[int] = []
        self.value_states: List[int] = []

        self._root: _RosaState = _RosaState()
        self._last_query: _RosaState = self._root
        self._last_key: _RosaState = self._root
    
    def append(self, query: int, key: int, value: int, default: int = -1) -> int:
        """
        Appends a new (query, key, value) triplet and computes the output value.

        This method performs two main actions in one step:
        1.  Extends the automaton with the new `key` token, updating the internal
            state structure according to the standard suffix automaton algorithm.
        2.  Traverses the automaton with the new `query` token to find the longest
            suffix of the query history that exists in the key history.

        Args:
            query (int): The current query token.
            key (int): The current key token to extend the automaton with.
            value (int): The current value token, associated with the key.
            default (int): The default value to return if no match is found.

        Returns:
            int: The value from a previous timestep corresponding to the end of the
                 longest matched suffix. If no non-empty suffix matches, returns `default`.
        """

        i = len(self.value_states)

        self.query_states.append(query)
        self.key_states.append(key)
        self.value_states.append(value)

        r = _RosaState()
        r.length = self._last_key.length + 1

        p = self._last_key
        while (p is not None) and (key not in p.transitions):
            p.transitions[key] = r
            p = p.suffix_link
        
        if p is None:
            r.suffix_link = self._root
        else:
            q = p.transitions[key]

            if p.length + 1 == q.length:
                r.suffix_link = q
            else:
                u = _RosaState()
                u.endpos = q.endpos
                u.length = p.length + 1
                u.suffix_link = q.suffix_link
                u.transitions.update(q.transitions)

                q.suffix_link = u
                r.suffix_link = u

                while (p is not None) and (p.transitions.get(key) is q):
                    p.transitions[key] = u
                    p = p.suffix_link

        j = -1
        
        p = self._last_query
        while (p is not None) and (query not in p.transitions):
            p = p.suffix_link
        
        if p is None:
            self._last_query = self._root
        else:
            self._last_query = p.transitions[query]

            p = self._last_query
            while p is not None:
                if p.length > 0 and p.endpos >= 0:
                    j = p.endpos + 1
                    break
                p = p.suffix_link
        
        self._last_key = r
        while (r is not None) and (r.endpos < i):
            r.endpos = i
            r = r.suffix_link
        
        return self.value_states[j] if j >= 0 else default
    
    def extend(
            self,
            query_states: List[int],
            key_states: List[int],
            value_states: List[int],
            default: int = -1,
    ) -> List[int]:
        """
        Processes entire sequences by calling `append` iteratively.

        Args:
            query_states (List[int]): The full sequence of query tokens.
            key_states (List[int]): The full sequence of key tokens.
            value_states (List[int]): The full sequence of value tokens.
            default (int): The default value to use for non-matches.

        Returns:
            List[int]: A list containing the output value for each timestep.
        """

        outs = []
        for q, k, v in zip(query_states, key_states, value_states):
            x = self.append(q, k, v, default)
            outs.append(x)
        return outs
    