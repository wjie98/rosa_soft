import torch
import torch.nn.functional as F

import math
import torch._dynamo as dynamo

from pathlib import Path
from functools import lru_cache

from torch import Tensor
from typing import Dict, List, Tuple, Optional, Union, Literal, cast

import logging
logger = logging.getLogger(__name__)


__all__ = [
    "rosa_bits_ops",
    "RapidOnlineSuffixAutomaton",
]

def rosa_bits_ops(
        query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
        alpha: Union[Tensor, float] = 1.0, proxy: Literal["rosa", "sufa"] = "sufa",
        bits_tau: Union[Tensor, float] = 1.0, attn_tau: Union[Tensor, float] = 1.0,
        host_ops: bool = False, training: bool = False,
        sufa_head_dim: int = 128, sufa_add_position: bool = True,
):
    """
    Performs the Rapid Online Suffix Automaton (ROSA) attention-like operation.

    This function computes a differentiable, attention-like mechanism based on the
    longest common suffix match between query and key sequences. The inputs are
    expected to be tensors of logits that will be binarized). The operation is designed
    to be efficient on parallel hardware like GPUs.

    Args:
        query (Tensor): The query tensor of shape (B, H, T, D_qk).
        key (Tensor): The key tensor of shape (B, H_kv, T, D_qk).
        value (Tensor): The value tensor of shape (B, H_kv, T, D_v).
        attn_mask (Optional[Tensor]): An optional boolean attention mask.
        alpha (Union[Tensor, float]): The interpolation coefficient for the STE.
            - `alpha = 0.0`: The output is purely from the soft, differentiable proxy.
            - `alpha = 1.0`: The forward pass output is purely from the hard, discrete op.
            This should be annealed from 0 to 1 during training.
        proxy (Literal["rosa", "sufa"]): The type of soft proxy to use for gradient
            computation. 'rosa' uses a custom dynamic programming approach to
            soft-simulate the ROSA matching algorithm. 'sufa' uses Suffix Attention
            (a standard scaled dot-product attention) as the proxy.
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
        sufa_add_position (bool): If True, adds positional information to attention scores
            when using the SUFA proxy to facilitate learning of positional dependencies.

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
                bits_tau=bits_tau, attn_tau=attn_tau,
                head_dim=sufa_head_dim, add_position=sufa_add_position,
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
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

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
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

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
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

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
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

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
    position_scale = 8192.0
    attn_bias = (r.view(1, -1) - r.view(-1, 1) - 1) / position_scale
    attn_bias = torch.exp(attn_bias).tril_() + torch.full_like(attn_bias, -torch.inf).triu_(1)

    # sharpness = 8192.0
    # attn_bias = r.view(-1, 1) - r.view(1, -1) + 2
    # attn_bias = 1 - torch.log(attn_bias.type_as(ss)).clamp_min_(1.0) / math.log(sharpness)
    # attn_bias = attn_bias.clamp_min_(0.0).view(1, 1, seq_len, seq_len).tril_()
    # attn_bias = attn_bias + torch.full_like(attn_bias, -torch.inf).triu_(1)

    # print(attn_bias)

    # softmax attention
    mv = ss.max(dim=-1, keepdim=True).values
    ii = torch.softmax((ss + attn_bias) / attn_tau, dim=-1)

    output = ii @ xv
    output = output * mv.clamp(0.0, 1.0) # unmatched positions get zeroed out
    return output


@torch.no_grad()
def sufa_attn_bias(
        query: Tensor,
        attn_mask: Optional[Tensor] = None,
        add_position: bool = True,
        position_scale: float = 8192.0,
        tau: Union[Tensor, float] = 1.0,
):
    bsz, _, seq_len, _ = query.size()
    if add_position:
        if attn_mask is None:
            attn_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=query.device).tril_()
        else:
            attn_mask = attn_mask.tril()

        r = torch.arange(0, seq_len, dtype=query.dtype, device=query.device)
        attn_bias = (r.view(1, -1) - r.view(-1, 1) - 1) / position_scale
        attn_bias = torch.exp(attn_bias).expand_as(attn_mask).masked_fill(~attn_mask, -torch.inf)
        attn_bias = attn_bias / tau
        return attn_bias, False
    else:
        return attn_mask, True

def sufa_bits_soft_ops(
        query: Tensor, key: Tensor, value: Tensor, attn_mask: Optional[Tensor] = None,
        bits_tau: Union[Tensor, float] = 1.0, attn_tau: Union[Tensor, float] = 1.0,
        head_dim: int = 128, add_position: bool = True,
):
    bsz, num_heads, seq_len, num_qk_bits = query.size()
    bsz, num_kv_heads, seq_len, num_qk_bits = key.size()
    bsz, num_kv_heads, seq_len, num_v_bits = value.size()

    n_rep = num_heads // num_kv_heads
    enable_gqa = n_rep > 1

    xq = torch.tanh(query / bits_tau)
    xk = torch.tanh(key / bits_tau)
    xv = torch.sigmoid(value / bits_tau)

    win_qk_size = head_dim // num_qk_bits

    # predict next token
    xq = F.pad(xq, (0, 0, win_qk_size - 1, 0), value=0.0).unfold(-2, win_qk_size, 1).transpose(-1, -2).reshape(bsz, -1, seq_len, head_dim)
    xk = F.pad(xk, (0, 0, win_qk_size, -1), value=0.0).unfold(-2, win_qk_size, 1).transpose(-1, -2).reshape(bsz, -1, seq_len, head_dim)

    attn_mask, is_causal = sufa_attn_bias(query, attn_mask=attn_mask, add_position=add_position, tau=attn_tau)
    with dynamo.config.patch(capture_scalar_outputs=True):
        scale = 1.0 / math.sqrt(head_dim) / (attn_tau.item() if isinstance(attn_tau, Tensor) else attn_tau)
        output = F.scaled_dot_product_attention(
            xq, xk, xv, scale=scale, is_causal=is_causal,
            attn_mask=attn_mask, enable_gqa=enable_gqa,
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

    load_torch_rosa()

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

        x = rosa_bits_soft_ops(q, k, v, bits_tau=1e-5, attn_tau=1e-7)

        x = (x > 0).long() << torch.arange(16)
        x = x.sum(dim=-1)

        return x.flatten().tolist()
    
    def test_rosa_qkv_attn_host(q, k, v):
        q = torch.tensor(q).long().view(1, 1, -1, 1) >> torch.arange(16)
        k = torch.tensor(k).long().view(1, 1, -1, 1) >> torch.arange(16)
        v = torch.tensor(v).long().view(1, 1, -1, 1) >> torch.arange(16)

        q = (q & 1).float() - 0.5
        k = (k & 1).float() - 0.5
        v = (v & 1).float() - 0.5

        xq, xk, xv = rosa_bits_host_ops_d2h(q, k, v)
        x = rosa_bits_host_ops_h2d(q, k, v, xq, xk, xv)

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
            o4 = torch.tensor(test_rosa_qkv_attn_host(q, k, v))

            assert (o1 == o2).all()
            assert (o1 == o3).all()
            assert (o1 == o4).all()

        print("✅ Forward Pass Passed!")
    except AssertionError as e:
        print("❌ Forward Pass Failed!")
        print(e)
    print()

    try:
        q = k = v = torch.randn(1, 8, 128, 64).requires_grad_()

        rosa_bits_ops(q, k, v, proxy="rosa", training=True).sum().backward()
        rosa_bits_ops(q, k, v, proxy="sufa", training=True).sum().backward()

        assert not q.grad.isnan().any()
        assert not k.grad.isnan().any()
        assert not v.grad.isnan().any()

        print("✅ Backward Pass Passed!")
    except AssertionError as e:
        print("❌ Backward Pass Failed!")
        print(e)
    print()

