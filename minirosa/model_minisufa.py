import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import *

from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3DecoderLayer,
    Qwen3Model,
    Qwen3ForCausalLM,
    repeat_kv,
)

from transformers import AutoConfig, AutoModel, AutoModelForCausalLM
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg
from transformers.cache_utils import Cache, DynamicCache, DynamicLayer, DynamicSlidingWindowLayer
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from .rosa import rosa_qkv_ops, RapidOnlineSuffixAutomaton
except ImportError:
    from rosa import rosa_qkv_ops, RapidOnlineSuffixAutomaton



class MiniSufaConfig(Qwen3Config):
    model_type = "minisufa"

    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",

        "layers.*.sufa_attn.q_proj": "colwise",
        "layers.*.sufa_attn.k_proj": "colwise",
        "layers.*.sufa_attn.v_proj": "colwise",
        "layers.*.sufa_attn.o_proj": "rowwise",
        "layers.*.sufa_attn.v_emb0": "local_rowwise",
        "layers.*.sufa_attn.v_emb1": "local_rowwise",

        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }

    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
            self,
            vocab_size = 6400,
            hidden_size = 512,
            intermediate_size = 1536,
            num_hidden_layers = 2,
            num_attention_heads = 8,
            num_key_value_heads = 2,
            head_dim: int = 64,
            hidden_act: str = "silu",
            use_cache: bool = False,
            use_sliding_window: bool = True,
            sliding_window: int = 128,
            use_sufa_attention: bool = True,
            num_sufa_heads: int = 8,
            num_sufa_key_value_heads: int = 2,
            num_sufa_query_key_bits: int = 8,
            num_sufa_value_bits: int = 16,
            **kwargs,
    ) -> None:
        if use_sliding_window:
            max_window_layers = num_hidden_layers
            layer_types = ["sliding_attention"] * num_hidden_layers
        else:
            max_window_layers = 0
            layer_types = ["full_attention"] * num_hidden_layers

        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            hidden_act=hidden_act,
            use_sliding_window=use_sliding_window,
            sliding_window=sliding_window,
            max_window_layers=max_window_layers,
            layer_types=layer_types,
            use_cache=use_cache,
            **kwargs
        )

        self.use_sufa_attention = use_sufa_attention
        self.num_sufa_heads = num_sufa_heads
        self.num_sufa_key_value_heads = num_sufa_key_value_heads
        self.num_sufa_query_key_bits = num_sufa_query_key_bits
        self.num_sufa_value_bits = num_sufa_value_bits


class SufaCache(DynamicCache):
    def __init__(self,
            ddp_cache_data = None,
            config: MiniSufaConfig = None,
            offloading = False,
            offload_only_non_sliding = False,
    ):
        super().__init__(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        self.sufa_window_size = max(1, 256 // config.num_sufa_query_key_bits)
        self.sufa_query_layers = [DynamicSlidingWindowLayer(self.sufa_window_size) for _ in range(config.num_hidden_layers)]
        self.sufa_key_value_layers: List[DynamicLayer] = [DynamicLayer() for _ in range(config.num_hidden_layers)]
    
    def update_sufa(self,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ):
        if self.sufa_query_layers[layer_idx].get_seq_length() == 0:
            query_states = F.pad(query_states, (0, 0, self.sufa_window_size - 1, 0))
        
        if self.sufa_key_value_layers[layer_idx].get_seq_length() == 0:
            key_states = F.pad(key_states, (0, 0, self.sufa_window_size - 1, 0))
            value_states = F.pad(value_states, (0, 0, self.sufa_window_size - 1, 0))

        query_states, _ = self.sufa_query_layers[layer_idx].update(query_states, query_states)
        key_states, value_states = self.sufa_key_value_layers[layer_idx].update(key_states, value_states)
        value_states = F.pad(value_states, (0, 0, -self.sufa_window_size, 1))
        return query_states, key_states, value_states
    

class SuffixAttention(nn.Module):
    def __init__(self, config: MiniSufaConfig, layer_idx: int, tau: float = 1.0):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        assert config.num_sufa_heads % config.num_sufa_key_value_heads == 0

        self.num_heads = config.num_sufa_heads
        self.num_kv_heads = config.num_sufa_key_value_heads
        self.n_rep = self.num_heads // self.num_kv_heads

        self.num_qk_bits = config.num_sufa_query_key_bits
        self.num_v_bits = config.num_sufa_value_bits

        self.window_size = 256 // self.num_qk_bits
        assert self.num_qk_bits * self.window_size % 16 == 0

        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.num_qk_bits, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.num_qk_bits, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.num_v_bits, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.num_v_bits, config.hidden_size, bias=False)

        self.v_emb0 = nn.Parameter(torch.full((self.num_heads * self.num_v_bits,), -1e-5))
        self.v_emb1 = nn.Parameter(torch.full((self.num_heads * self.num_v_bits,), +1e-5))

        self.register_buffer("tau", torch.tensor([tau], dtype=torch.float32))
    
    def _attn_bias(self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        bsz, seq_len, _ = hidden_states.size()

        attn_mask = torch.ones(1, 1, seq_len, seq_len, dtype=torch.bool, device=hidden_states.device).tril_(-1)
        if attention_mask is not None:
            attn_mask = attn_mask & attention_mask
        
        r = torch.arange(0, seq_len, dtype=hidden_states.dtype, device=hidden_states.device)
        attn_bias = r.view(1, -1) / (r.view(-1, 1) + 1)
        attn_bias = attn_bias.expand_as(attn_mask).clone()
        attn_bias[~attn_mask] = -torch.inf
        return attn_bias
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[SufaCache] = None,
            **kwargs
    ):
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, -1, self.num_qk_bits)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, -1, self.num_qk_bits)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, -1, self.num_v_bits)

        tau = self.get_buffer("tau")
        
        query_states = torch.tanh(query_states / tau).transpose(1, 2)
        key_states = torch.tanh(key_states / tau).transpose(1, 2)
        value_states = torch.sigmoid(value_states / tau).transpose(1, 2)

        if past_key_values is not None:
            query_states, key_states, value_states = past_key_values.update_sufa(
                query_states, key_states, value_states,
                layer_idx=self.layer_idx,
            )
        else:
            query_states = F.pad(query_states, (0, 0, self.window_size - 1, 0))
            key_states = F.pad(key_states, (0, 0, self.window_size - 1, 0))
            value_states = F.pad(value_states, (0, 0, -1, 1)) # recall next token

        qk_dim = self.window_size * self.num_qk_bits
        
        query_states = query_states.unfold(-2, self.window_size, 1).transpose(-2, -1).reshape(bsz, -1, seq_len, qk_dim)
        key_states = key_states.unfold(-2, self.window_size, 1).transpose(-2, -1).reshape(bsz, -1, seq_len, qk_dim)

        scale = 1.0 / math.sqrt(qk_dim) / tau.item()
        attn_bias = self._attn_bias(hidden_states, attention_mask) / tau
        
        output = F.scaled_dot_product_attention(
            query_states, key_states, value_states, scale=scale,
            attn_mask=attn_bias, enable_gqa=True,
        )

        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = output * self.v_emb1 + (1 - output) * self.v_emb0
        output = self.o_proj(output)
        return output


class MiniSufaDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: MiniSufaConfig, layer_idx):
        super().__init__(config, layer_idx)
        if config.use_sufa_attention:
            self.sufa_attn = SuffixAttention(config, layer_idx)
        else:
            self.sufa_attn = None
    
    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # SUFA Attention
        if self.sufa_attn is not None:
            residual = residual + self.sufa_attn(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
            )

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class MiniSufaModel(Qwen3Model):
    config_class = MiniSufaConfig

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            MiniSufaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.post_init()
    
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)

        if isinstance(module, SuffixAttention):
            nn.init.constant_(module.v_emb0, -1e-5)
            nn.init.constant_(module.v_emb1, +1e-5)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if use_cache is None:
            use_cache = self.config.use_cache

        if use_cache and past_key_values is None:
            past_key_values = SufaCache(config=self.config)

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )
    

class MiniSufaForCausalLM(Qwen3ForCausalLM):
    config_class = MiniSufaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniSufaModel(config)
        self.post_init()

AutoConfig.register("minisufa", MiniSufaConfig)
AutoModel.register(MiniSufaConfig, MiniSufaModel)
AutoModelForCausalLM.register(MiniSufaConfig, MiniSufaForCausalLM)


if __name__ == "__main__":
    config = MiniSufaConfig(num_hidden_layers=1)

    config.use_cache = True

    model = MiniSufaForCausalLM(config)
    # model = torch.compile(model)
    print(model)

    x = torch.randint(0, config.vocab_size, size=(1, 512))
    print(x.size())
    m = torch.randint(0, 2, size=(1, 512))

    x = model(x, labels=x, attention_mask=m)
    print(x)