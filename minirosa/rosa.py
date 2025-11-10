import torch
import torch.nn as nn
import torch.nn.functional as F

import re
import os

from torch import Tensor
from typing import *

from pathlib import Path

from transformers.utils import logging
logger = logging.get_logger(__name__)


__all__ = [
    "rosa_qkv_ops",
    "RapidOnlineSuffixAutomaton",
    "RosaAttention",
    "RosaCache",
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

        query_states: Tensor = self.q_proj(hidden_states).view(bsz, seq_len, -1, self.num_qk_bits)
        key_states: Tensor = self.k_proj(hidden_states).view(bsz, seq_len, -1, self.num_qk_bits)
        value_states: Tensor = self.v_proj(hidden_states).view(bsz, seq_len, -1, self.num_v_bits)

        query_states = query_states.transpose(1, 2)
        key_states = self.repeat_kv(key_states.transpose(1, 2), self.n_rep)
        value_states = self.repeat_kv(value_states.transpose(1, 2), self.n_rep)

        if past_key_values is not None:
            if not hasattr(past_key_values, "rosa_cache"):
                setattr(past_key_values, "rosa_cache", RosaCache())

            cache: RosaCache = getattr(past_key_values, "rosa_cache")
            hidden_states = cache.update(query_states, key_states, value_states, self.layer_idx)

            v_emb0 = self.v_emb0.view(1, -1, 1, self.num_v_bits)
            v_emb1 = self.v_emb1.view(1, -1, 1, self.num_v_bits)
            output = v_emb0 * (1 - hidden_states) + v_emb1 * hidden_states
        else:
            tau = self.get_buffer("tau")

            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask[:, :, :, :key_states.size(-2)]
            
            v_emb0 = self.v_emb0.view(1, -1, 1, self.num_v_bits)
            v_emb1 = self.v_emb1.view(1, -1, 1, self.num_v_bits)
            output = rosa_qkv_ops(
                query_states, key_states,
                value_states, v_emb0, v_emb1,
                attn_mask=attn_mask, tau=tau,
            )
            
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output
    
    @classmethod
    def apply_adapter_to_modules_(
        cls,
        module: nn.Module,
        config: RosaConfig,
        target_modules: str = r".*self_attn",
        autocast_dtype: bool = True,
    ):
        adapter_factory = lambda i, m: cls(config, layer_idx=i)
        super().apply_adapter_to_modules_(
            module=module,
            adapter_factory=adapter_factory,
            adapter_name="rosa_attn",
            target_modules=target_modules,
            autocast_dtype=autocast_dtype,
        )

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


def _rosa_diag_to_cols(x: Tensor) -> Tensor:
    *B, _, T = x.size()
    r = torch.arange(T, dtype=torch.long, device=x.device)
    c = (r.view(-1, 1) + r.view(1, -1) + 1) % T
    x = x.gather(-1, c.expand(*B, T, T))
    return x

def _rosa_cols_to_diag(x: Tensor) -> Tensor:
    *B, _, T = x.size()
    r = torch.arange(T, dtype=torch.long, device=x.device)
    c = (r.view(1, -1) - r.view(-1, 1) + T - 1) % T
    x = x.gather(-1, c.expand(*B, T, T))
    return x

def _rosa_qkv_scan(x: Tensor) -> Tensor:
    x = _rosa_diag_to_cols(x)
    a = x.cumsum(dim=-2)
    x = a - (a * (1 - x)).cummax(dim=-2).values
    x = _rosa_cols_to_diag(x)
    return x

def _rosa_qkv_attn_hard(q: Tensor, k: Tensor, v: Tensor, bits_mode: bool, attn_mask: Tensor | None):
    B, H, T, C = q.size()
    if bits_mode:
        c = torch.arange(C, device=q.device)
        q = torch.sum((q > 0).long() << c, dim=-1)
        k = torch.sum((k > 0).long() << c, dim=-1)
        v = (v > 0).long()
    else:
        q = q.argmax(dim=-1)
        k = k.argmax(dim=-1)
        v = v.argmax(dim=-1)

    x = (q.view(B, H, T, 1) == k.view(B, H, 1, T)).long().tril_(diagonal=-1)
    if attn_mask is not None:
        x = x.masked_fill_(~attn_mask, 0)

    a = _rosa_qkv_scan(x)
    a = F.pad(a, (1, -1), value=0)

    r = torch.arange(1, T + 1, dtype=a.dtype, device=a.device)
    r = ((a << 32) | r).argmax(dim=-1) # (B, H, T)

    g = x.max(dim=-1).values
    if bits_mode:
        r = r.view(B, H, T, 1).expand(*v.size())
        r = v.gather(-2, r) * g[..., None]
    else:
        r = v.gather(-1, r) * g
    return r

def _rosa_qkv_attn_soft(q: Tensor, k: Tensor, v: Tensor, bits_mode: bool, attn_mask: Tensor | None, tau: Tensor | float):
    B, H, T, C = q.size()
    if bits_mode:
        q = torch.tanh(q / tau)
        k = torch.tanh(k / tau)
        v = torch.sigmoid(v / tau)

        x = q @ k.transpose(-1, -2) - (C - 1) * (1 - tau)
        x = torch.sigmoid(x / tau)
    else:
        q = torch.softmax(q / tau, dim=-1)
        k = torch.softmax(k / tau, dim=-1)
        v = torch.softmax(v / tau, dim=-1)

        x = q @ k.transpose(-1, -2)

    x = torch.tril(x, diagonal=-1)
    if attn_mask is not None:
        x = x.masked_fill(~attn_mask, 0.0)
        
    a = _rosa_qkv_scan(x)
    a = F.pad(a, (1, -1), value=0.0)

    r = torch.arange(T, dtype=a.dtype, device=a.device)
    a = a + r.view(1, -1) / (r.view(-1, 1) + 1)

    m = torch.ones(T, T, dtype=torch.bool, device=a.device).triu_(diagonal=1)
    a = a.masked_fill(m, -torch.inf)
    
    r = torch.softmax(a.float() / tau, dim=-1).type_as(a)
    r = r @ v
    
    g = x.max(dim=-1, keepdim=True).values
    if bits_mode:
        r = r * g
    else:
        z = torch.zeros(1, dtype=torch.long, device=r.device)
        u = F.one_hot(z, r.size(-1)).type_as(r)
        r = r * g + (1 - g) * u
    return r

def rosa_qkv_ops(
    q: Tensor, k: Tensor, v: Tensor,
    e0: Tensor, e1: Tensor | None = None,
    attn_mask: Tensor | None = None,
    tau: Tensor | float | None = None,
) -> Tensor:
    B, H, T, _ = q.size()

    if e1 is None:
        bits_mode = False
        _, _, V, C = e0.size()
        assert e0.size(0) in {1, B}
        assert e0.size(1) in {1, H}
    else:
        bits_mode = True
        _, _, _, C = e0.size()
        assert e0.size(0) in {1, B}
        assert e0.size(1) in {1, H}
        assert e0.size(2) in {1, T}
        assert e0.size(3) == v.size(3)
    
    if tau is None:
        r = _rosa_qkv_attn_hard(q, k, v, bits_mode=bits_mode, attn_mask=attn_mask)
    else:
        r = _rosa_qkv_attn_soft(q, k, v, bits_mode=bits_mode, attn_mask=attn_mask, tau=tau)
    
    if bits_mode:
        r = r.type_as(e0)
        o = r * e1 + (1 - r) * e0
    elif tau is None:
        r = r.view(B, H, T, 1).expand(B, H, T, C)
        o = e0.expand(B, H, V, C).gather(-2, r)
    else:
        o = r.view(B, H, T, V) @ e0
    return o


class _RosaState:
    __slots__ = ("endpos", "length", "suffix_link", "transitions")

    def __init__(self):
        self.endpos = -1
        self.length = 0
        self.suffix_link: Optional[_RosaState] = None
        self.transitions: Dict[int, _RosaState] = {}


class RapidOnlineSuffixAutomaton:
    def __init__(self):
        self.query_states: List[int] = []
        self.key_states: List[int] = []
        self.value_states: List[int] = []

        self._root: _RosaState = _RosaState()
        self._last_query: _RosaState = self._root
        self._last_key: _RosaState = self._root
    
    def append(self, query: int, key: int, value: int, default: int = -1) -> int:
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
        outs = []
        for q, k, v in zip(query_states, key_states, value_states):
            x = self.append(q, k, v, default)
            outs.append(x)
        return outs


if __name__ == "__main__":
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    config = Qwen3Config(num_hidden_layers=2, hidden_size=512, intermediate_size=1024)
    config.use_cache = False
    
    model = Qwen3ForCausalLM(config)
    model = RosaAttention.patch_attention(
        model,
        dim=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
    )

    model = RosaAttention.freeze_base_model_and_enable_patch_params(model)
    for name, _ in RosaAttention.state_dict_for_patch(model).items():
        print(name)

    x = torch.randint(0, 2, (3, 10))
    y = model(x)
    print(y)
