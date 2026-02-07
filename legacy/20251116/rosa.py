import torch
import torch.nn.functional as F

from torch import Tensor
from typing import Dict, List, Tuple, Optional, cast


__all__ = [
    "rosa_bits_ops",
    "RapidOnlineSuffixAutomaton",
]


def rosa_bits_ops(
        query: Tensor, key: Tensor, value: Tensor,
        attn_mask: Optional[Tensor] = None,
):
    """
    Performs the Rapid Online Suffix Automaton (ROSA) attention-like operation.

    This function computes a differentiable, attention-like mechanism based on the
    longest common suffix match between query and key sequences. The inputs are
    expected to be binarized tensors (or tensors of logits that will be binarized).
    The operation is designed to be efficient on parallel hardware like GPUs.

    Args:
        query (Tensor): The query tensor of shape `(bsz, num_heads, seq_len, num_qk_bits)`.
                        Represents a sequence of binarized vectors.
        key (Tensor): The key tensor of shape `(bsz, num_kv_heads, seq_len, num_qk_bits)`.
                      Represents a sequence of binarized vectors. Supports Grouped-Query
                      Attention where `num_kv_heads` can be a divisor of `num_heads`.
        value (Tensor): The value tensor of shape `(bsz, num_kv_heads, seq_len, num_v_bits)`.
                        The values to be gathered based on the matching results.
        attn_mask (Optional[Tensor]): An optional boolean attention mask of shape
                                     `(bsz, num_heads, seq_len, seq_len)`. If provided,
                                     positions where the mask is `False` are ignored.

    Returns:
        Tensor: The output tensor of shape `(bsz, num_heads, seq_len, num_v_bits)`,
                containing the gathered values.
    """

    attn_scores: Tensor = ROSAScoresFunction.apply(query, key, attn_mask)

    bsz, num_heads, seq_len, _ = attn_scores.size()
    r = torch.arange(seq_len, dtype=torch.long, device=attn_scores.device)

    # diagonals to columns
    c = (r.view(-1, 1) + r.view(1, -1) + 1) % seq_len
    attn_scores = attn_scores.gather(-1, c.expand(bsz, num_heads, seq_len, seq_len))

    # dynamic programming
    a = attn_scores.cumsum(dim=-2)
    attn_scores = a - (a * (1 - attn_scores)).cummax(dim=-2).values

    # columns to diagonals
    c = (r.view(1, -1) - r.view(-1, 1) + seq_len - 1) % seq_len
    attn_scores = attn_scores.gather(-1, c.expand(bsz, num_heads, seq_len, seq_len))

    output: Tensor = ROSAGatherFunction.apply(attn_scores, value)
    return output


class ROSAScoresFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, query: Tensor, key: Tensor, attn_mask: Optional[Tensor] = None) -> Tensor:
        bsz, num_heads, seq_len, num_qk_bits = query.size()
        bsz, num_kv_heads, seq_len, num_qk_bits = key.size()

        rr = torch.arange(0, num_qk_bits, device=query.device)

        xq = ((query > 0).long() << rr).sum(dim=-1)
        xk = ((key > 0).long() << rr).sum(dim=-1)

        n_rep = num_heads // num_kv_heads
        if n_rep > 1:
            xk = xk.view(bsz, num_kv_heads, 1, seq_len)
            xk = xk.expand(bsz, num_kv_heads, n_rep, seq_len)
            xk = xk.reshape(bsz, num_heads, seq_len)
        
        xq = xq.view(bsz, num_heads, seq_len, 1)
        xk = xk.view(bsz, num_heads, 1, seq_len)

        attn_scores = (xq == xk).float().tril_(diagonal=-1)
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(~attn_mask, 0.0)

        attn_scores = F.pad(attn_scores, (1, -1), value=0.0) # predict next token

        ctx.save_for_backward(query, key, attn_mask)
        return attn_scores
    
    @staticmethod
    def backward(ctx, grad_output: Tensor):
        query, key, attn_mask = cast(Tuple[Tensor, Tensor, Optional[Tensor]], ctx.saved_tensors)

        bsz, num_heads, seq_len, num_qk_bits = query.size()
        bsz, num_kv_heads, seq_len, num_qk_bits = key.size()

        xq = (query > 0).float()
        xk = (key > 0).float()

        n_rep = num_heads // num_kv_heads
        if n_rep > 1:
            xk = xk.view(bsz, num_kv_heads, 1, seq_len, num_qk_bits)
            xk = xk.expand(bsz, num_kv_heads, n_rep, seq_len, num_qk_bits)
            xk = xk.reshape(bsz, num_heads, seq_len, num_qk_bits)
        
        grad_output = F.pad(grad_output, (-1, 1), value=0.0)
        if attn_mask is not None:
            grad_output = grad_output.masked_fill(~attn_mask, 0.0)
        
        ss = xq @ xk.transpose(-1, -2) - (num_qk_bits - 1) # [1 - 2 * num_qk_bits, 1]
        ss = torch.sigmoid(ss).tril_(diagonal=-1)
        grad_output = grad_output * ss * (1 - ss)

        grad_query = grad_output @ xk
        grad_key = grad_output.transpose(-1, -2) @ xq

        if n_rep > 1:
            grad_key = grad_key.view(bsz, num_kv_heads, n_rep, seq_len, num_qk_bits).sum(dim=2)
        
        sq = torch.sigmoid(query)
        sk = torch.sigmoid(key)

        grad_query = grad_query * sq * (1 - sq)
        grad_key = grad_key * sk * (1 - sk)

        return grad_query, grad_key, None


class ROSAGatherFunction(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx,
            attn_scores: Tensor,
            value: Tensor,
    ) -> Tensor:
        ctx.save_for_backward(attn_scores, value)

        bsz, num_heads, seq_len, _ = attn_scores.size()
        bsz, num_kv_heads, seq_len, num_v_bits = value.size()

        n_rep = num_heads // num_kv_heads

        attn_scores = attn_scores.tril()

        max_val = torch.max(attn_scores, dim=-1, keepdim=True).values
        indices = torch.arange(seq_len, device=attn_scores.device)
        indices = torch.where(attn_scores == max_val, indices.view(1, 1, 1, -1), -1).max(dim=-1, keepdim=True).values

        xv = (value > 0).float()
        if n_rep > 1:
            xv = xv.view(bsz, num_kv_heads, 1, seq_len, num_v_bits)
            xv = xv.repeat(1, 1, n_rep, 1, 1)
            xv = xv.reshape(bsz, num_heads, seq_len, num_v_bits)
        
        output = xv.gather(-2, indices.repeat(1, 1, 1, num_v_bits))
        output = output * (max_val > 0).float() # unmatched positions get zeroed out
        return output
    
    @staticmethod
    def backward(
            ctx,
            grad_output: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        attn_scores, value = cast(Tuple[Tensor, ...], ctx.saved_tensors)

        bsz, num_heads, seq_len, _ = attn_scores.size()
        _, num_kv_heads, _, num_v_bits = value.size()

        n_rep = num_heads // num_kv_heads

        xv = (value > 0).float()
        if n_rep > 1:
            xv = xv.view(bsz, num_kv_heads, 1, seq_len, num_v_bits)
            xv = xv.repeat(1, 1, n_rep, 1, 1)
            xv = xv.reshape(bsz, num_heads, seq_len, num_v_bits)
        
        with torch.enable_grad():
            attn_scores.requires_grad_(True)
            xv.requires_grad_(True)

            m = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=attn_scores.device).tril_()
            a = attn_scores.masked_fill(~m, -torch.inf)
            a = F.softmax(a, dim=-1)

            output_proxy = a @ xv
            grad_attn_scores, grad_value = torch.autograd.grad(
                outputs=output_proxy,
                inputs=(attn_scores, xv),
                grad_outputs=grad_output,
            )
        
        if n_rep > 1:
            grad_value = grad_value.view(bsz, num_kv_heads, n_rep, seq_len, num_v_bits).sum(dim=2)
        
        sv = torch.sigmoid(value)
        grad_value = grad_value * sv * (1 - sv)

        return grad_attn_scores, grad_value


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

        x = rosa_bits_ops(q, k, v)

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

            assert (o1 == o2).all()

        print("✅ Forward Pass Passed!")
    except AssertionError as e:
        print("❌ Forward Pass Failed!")
        print(e)
    print()

    try:
        q = k = v = torch.randn(1, 8, 128, 64).requires_grad_()

        rosa_bits_ops(q, k, v).sum().backward()

        assert not q.grad.isnan().any()
        assert not k.grad.isnan().any()
        assert not v.grad.isnan().any()

        print("✅ Backward Pass Passed!")
    except AssertionError as e:
        print("❌ Backward Pass Failed!")
        print(e)
    print()

