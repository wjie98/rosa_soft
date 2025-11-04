import torch
import torch.nn as nn
import torch.nn.functional as F

import math

from torch import Tensor
from typing import *

from transformers.activations import ACT2FN
from transformers import PreTrainedModel, GenerationMixin, PretrainedConfig
from transformers.modeling_outputs import CausalLMOutputWithPast, BaseModelOutputWithPast

try:
    from .rosa import rosa_qkv_ops
    from .configuration_minirosa import MiniRosaConfig
except ImportError:
    from rosa import rosa_qkv_ops
    from configuration_minirosa import MiniRosaConfig


class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dims))
    
    def _norm(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    
    def forward(self, x: Tensor) -> Tensor:
        return self.weight * self._norm(x.float()).type_as(x)


def precompute_freqs_cis(dims: int, end: int, rope_base: float, rope_scaling: Dict[str, int | str] | None = None):
    freqs = 1.0 / (rope_base ** (torch.arange(0, dims, 2)[:dims//2].float() / dims))
    
    if rope_scaling is not None:
        beta_fast = rope_scaling["beta_fast"]
        beta_slow = rope_scaling["beta_slow"]
        factor = rope_scaling["factor"]
        orig_max = rope_scaling["original_max_position_embeddings"]

        if end / orig_max > 1.0:
            corr_dim = next(
                (i for i in range(dims // 2) if 2 * math.pi / freqs[i].item() > orig_max),
                dims // 2)
            power = torch.arange(0, dims // 2, device=freqs.device).float() / max(dims // 2 - 1, 1)
            beta = beta_slow + (beta_fast - beta_slow) * power
            scale = torch.where(
                torch.arange(dims // 2, device=freqs.device) < corr_dim,
                (beta * factor - beta + 1) / (beta * factor), # λ = (β·α - β + 1)/(β·α) YaRN标准公式
                1.0 / factor,
            )
            freqs = freqs * scale

    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs).float()
    freqs_cos = torch.cos(freqs)
    freqs_sin = torch.sin(freqs)
    return torch.cat([freqs_cos, freqs_sin], dim=-1)


def apply_rotary_pos_emb(q: Tensor, k: Tensor, freqs: Tensor) -> Tensor:
    _, seq_len, _, _ = q.size()

    cos, sin = freqs[None, :seq_len, None, :].chunk(2, -1)

    q1, q2 = q.chunk(2, -1)
    q_embed = torch.cat([
        q1 * cos - q2 * sin,
        q1 * sin + q2 * cos,
    ], dim=-1)

    k1, k2 = k.chunk(2, -1)
    k_embed = torch.cat([
        k1 * cos - k2 * sin,
        k1 * sin + k2 * cos,
    ], dim=-1)
    return q_embed, k_embed


def repeat_kv(x: Tensor, n_rep: int) -> Tensor:
    bs, slen, num_key_value_heads, head_dims = x.size()
    if n_rep == 1:
        return x
    x = x.unsqueeze(-2).expand(bs, slen, num_key_value_heads, n_rep, head_dims)
    return x.reshape(bs, slen, -1, head_dims)


class Attention(nn.Module):
    def __init__(self, config: MiniRosaConfig):
        super().__init__()
        self.config = config
        assert config.num_attention_heads % config.num_key_value_heads == 0
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dims = config.hidden_size // config.num_attention_heads

        self.q_proj = nn.Linear(config.hidden_size, config.num_attention_heads * self.head_dims, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dims, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_key_value_heads * self.head_dims, bias=False)
        self.o_proj = nn.Linear(config.num_attention_heads * self.head_dims, config.hidden_size, bias=False)
    
    def forward(
            self,
            x: Tensor,
            position_embeddings: Tensor,
            attention_mask: Tensor | None = None,
            past_key_value: Tuple[Tensor, Tensor] | None = None,
            use_cache: bool = False,
    ):
        bsz, seq_len, _ = x.size()
        xq: Tensor = self.q_proj(x)
        xk: Tensor = self.k_proj(x)
        xv: Tensor = self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.head_dims)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.head_dims)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.head_dims)

        xq, xk = apply_rotary_pos_emb(xq, xk, position_embeddings)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        if seq_len > 1:
            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask.view(bsz, 1, 1, seq_len).bool()

            output = F.scaled_dot_product_attention(
                xq, xk, xv,
                attn_mask=attn_mask,
                is_causal=True,
            )
        else:
            scores = (xq @ xk.transpose(-2, -1)) / math.sqrt(self.head_dims)
            
            attn_mask = torch.ones((1, 1, seq_len, seq_len), dtype=torch.bool, device=scores.device).tril_()
            if attention_mask is not None:
                attn_mask = attn_mask & attention_mask.view(bsz, 1, 1, seq_len)

            scores = scores.masked_fill(~attn_mask, -torch.inf)
            scores = F.softmax(scores.float(), dim=-1).type_as(xq)
            output = scores @ xv
        
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output, past_kv


class RosaAttention(nn.Module):
    def __init__(self, config: MiniRosaConfig, tau: float = 1.0) -> None:
        super().__init__()
        self.config = config
        assert config.use_rosa_attention
        assert config.num_rosa_heads % config.num_rosa_key_value_heads == 0

        self.n_local_heads = config.num_rosa_heads
        self.n_local_kv_heads = config.num_rosa_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dims = config.hidden_size // config.num_attention_heads
        
        self.n_local_bits = config.num_rosa_value_bits
        self.n_local_qk_bits = config.num_rosa_query_key_bits

        self.q_proj = nn.Linear(config.hidden_size, config.num_rosa_heads * config.num_rosa_query_key_bits, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_rosa_key_value_heads * config.num_rosa_query_key_bits, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_rosa_key_value_heads * config.num_rosa_value_bits, bias=False)
        self.o_proj = nn.Linear(config.num_rosa_heads * config.num_rosa_value_bits, config.hidden_size, bias=False)

        self.emb0 = nn.Parameter(torch.full((config.num_rosa_heads, config.num_rosa_value_bits), -1e-5))
        self.emb1 = nn.Parameter(torch.full((config.num_rosa_heads, config.num_rosa_value_bits), +1e-5))

        self.register_buffer("tau", torch.tensor([tau], dtype=torch.float32))
    
    def forward(
            self,
            x: Tensor,
            attention_mask: Tensor | None = None,
            past_query_key_value: Tuple[Tensor, Tensor, Tensor] | None = None,
            use_cache: bool = False,
    ):
        bsz, seq_len, _ = x.size()
        xq: Tensor = self.q_proj(x)
        xk: Tensor = self.k_proj(x)
        xv: Tensor = self.v_proj(x)

        xq = xq.view(bsz, seq_len, self.n_local_heads, self.n_local_qk_bits)
        xk = xk.view(bsz, seq_len, self.n_local_kv_heads, self.n_local_qk_bits)
        xv = xv.view(bsz, seq_len, self.n_local_kv_heads, self.n_local_bits)

        if past_query_key_value is not None:
            xq = torch.cat([past_query_key_value[0], xq], dim=1)
            xk = torch.cat([past_query_key_value[1], xk], dim=1)
            xv = torch.cat([past_query_key_value[2], xv], dim=1)

        past_qkv = (xq, xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        tau = self.get_buffer("tau")

        attn_mask = None
        if attention_mask is not None:
            attn_mask = attention_mask.view(bsz, 1, 1, seq_len).bool()

        output = rosa_qkv_ops(xq, xk, xv, self.emb0, self.emb1, attn_mask=attn_mask, tau=tau)
        
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output, past_qkv


class FeedForward(nn.Module):
    def __init__(self, config: MiniRosaConfig):
        super().__init__()
        if config.intermediate_size is None:
            intermediate_size = config.hidden_size * 4
            intermediate_size = intermediate_size * 2 / 3
            intermediate_size = math.ceil(intermediate_size / 64) * 64
            config.intermediate_size = intermediate_size
            
        self.config = config
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]
    
    def forward(self, x: Tensor):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniRosaBlock(nn.Module):
    def __init__(self, layer_id: int, config: MiniRosaConfig):
        super().__init__()
        self.layer_id = layer_id
        self.config = config

        self.self_attn = Attention(config)
        if config.use_rosa_attention:
            self.rosa_attn = RosaAttention(config)
        
        self.feed_forward = FeedForward(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
    
    def forward(
            self,
            hidden_states: Tensor,
            position_embeddings: Tensor,
            attention_mask: Tensor | None = None,
            past_key_value: Tuple[Tensor, ...] | None = None,
            use_cache: bool = False,
    ):
        past_attn_key_value = None if past_key_value is None else past_key_value[:2]
        past_rosa_key_value = None if past_key_value is None else past_key_value[2:]

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        attn_states, present_key_value = self.self_attn(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_value=past_attn_key_value, use_cache=use_cache,
        )

        if self.config.use_rosa_attention:
            rosa_states, present_rosa_key_value = self.rosa_attn(
                hidden_states,
                attention_mask=attention_mask,
                past_query_key_value=past_rosa_key_value, use_cache=use_cache,
            )

            if use_cache:
                present_key_value = present_key_value + present_rosa_key_value

            hidden_states = attn_states + rosa_states + residual
        else:
            hidden_states = attn_states + residual
        
        hidden_states = self.feed_forward(self.post_attention_layernorm(hidden_states)) + hidden_states

        return hidden_states, present_key_value


class MiniRosaModel(nn.Module):
    def __init__(self, config: MiniRosaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([MiniRosaBlock(i, config) for i in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        freqs = precompute_freqs_cis(
            dims=config.hidden_size // config.num_attention_heads,
            end=config.max_position_embeddings, rope_base=config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.register_buffer("freqs", freqs, persistent=False)
    
    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor | None = None,
            past_key_values: List[Tuple[Tensor, ...]] | None = None,
            use_cache: bool = False,
            **kwargs,
    ):
        bsz, seq_len = input_ids.size()

        if hasattr(past_key_values, "layers"):
            past_key_values = None
        past_key_values = past_key_values or [None] * len(self.layers)
        start_pos = 0 if past_key_values[0] is None else past_key_values[0][0].size(1)

        hidden_states = self.embed_tokens(input_ids)
        position_embeddings = self.get_buffer("freqs")[start_pos : start_pos+seq_len]

        presents = []
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            hidden_states, present = layer(
                hidden_states,
                position_embeddings,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            presents.append(present)
        
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=present if use_cache else None,
        )


class MiniRosaForCausalLM(PreTrainedModel, GenerationMixin):
    config_class = MiniRosaConfig

    def __init__(self, config: MiniRosaConfig | None = None) -> None:
        self.config = config or MiniRosaConfig()
        super().__init__(self.config)
        
        self.model = MiniRosaModel(self.config)
        self.lm_head = nn.Linear(self.config.hidden_size, self.config.vocab_size, bias=False)

        self.post_init()

        self.model.embed_tokens._dynamic_tied_weights_keys = ["weight"]
        self.lm_head._dynamic_tied_weights_keys = ["weight"]
    
    def forward(
            self,
            input_ids: Tensor,
            attention_mask: Tensor | None = None,
            past_key_values: List[Tuple[Tensor, ...]] | None = None,
            use_cache: bool = False,
            labels: Tensor | None = None,
            logits_to_keep: Union[int, Tensor] = 0,
            **kwargs,
    ):
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values if use_cache else None,
            hidden_states=hidden_states,
        )

if __name__ == "__main__":
    config = MiniRosaConfig()
    config.use_rosa_attention = True

    model = MiniRosaForCausalLM(config)

    x = torch.randint(0, config.vocab_size, size=(1, 1024))
    print(x.size())
    m = torch.randint(0, 2, size=(1, 1024))

    x = model(x, labels=x, attention_mask=m)
    print(x)
