import torch
import torch.nn.functional as F

import math
import torch._dynamo as dynamo

from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union, Literal, cast


__all__ = [
    "rosa_bits_ops",
    "RapidOnlineSuffixAutomaton",
]

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


if __name__ == "__main__":
    B, T, H, C, V = 4, 8, 2, 4, 5

    def test_rosa_qkv_attn_hard(q, k, v):
        q = torch.tensor(q).long().view(1, 1, -1, 1) >> torch.arange(16)
        k = torch.tensor(k).long().view(1, 1, -1, 1) >> torch.arange(16)
        v = torch.tensor(v).long().view(1, 1, -1, 1) >> torch.arange(16)

        q = q & 1
        k = k & 1
        v = v & 1

        x = rosa_bits_hard_ops(q, k, v)

        x = (x > 0).long() << torch.arange(16)
        x = x.sum(dim=-1)

        return x.flatten().tolist()
    
    def test_rosa_qkv_attn_soft(q, k, v):
        q = torch.tensor(q).long().view(1, 1, -1, 1) >> torch.arange(16)
        k = torch.tensor(k).long().view(1, 1, -1, 1) >> torch.arange(16)
        v = torch.tensor(v).long().view(1, 1, -1, 1) >> torch.arange(16)

        q = (q & 1).float() - 0.5
        k = (k & 1).float() - 0.5
        v = (v & 1).float() - 0.5

        x = rosa_bits_soft_ops(q, k, v, bits_tau=1e-5, attn_tau=1e-5)

        x = (x > 0).long() << torch.arange(16)
        x = x.sum(dim=-1)

        return x.flatten().tolist()
    
    try:    
        for _ in range(10):
            q = torch.randint(0, 2, size=(8,)).tolist()
            k = torch.randint(0, 2, size=(8,)).tolist()
            v = torch.randint(0, 2, size=(8,)).tolist()

            r = RapidOnlineSuffixAutomaton()

            o1 = torch.tensor(r.extend(q, k, v, 0))
            o2 = torch.tensor(test_rosa_qkv_attn_hard(q, k, v))
            o3 = torch.tensor(test_rosa_qkv_attn_soft(q, k, v))

            assert (o1 == o2).all()
            assert (o1 == o3).all()

        print("✅ Forward Pass Passed!")
    except AssertionError as e:
        print("❌ Forward Pass Failed!")
        print(e)
    print()

    try:
        q = k = v = torch.randn(1, 8, 128, 64).requires_grad_()

        rosa_bits_ops(q, k, v, proxy="rosa").sum().backward()
        rosa_bits_ops(q, k, v, proxy="sufa").sum().backward()

        assert not q.grad.isnan().any()
        assert not k.grad.isnan().any()
        assert not v.grad.isnan().any()

        print("✅ Backward Pass Passed!")
    except AssertionError as e:
        print("❌ Backward Pass Failed!")
        print(e)
    print()

