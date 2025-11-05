import torch
import torch.nn as nn

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
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

try:
    from .rosa import rosa_qkv_ops, RapidOnlineSuffixAutomaton
except ImportError:
    from rosa import rosa_qkv_ops, RapidOnlineSuffixAutomaton



class MiniRosaConfig(Qwen3Config):
    model_type = "minirosa"

    keys_to_ignore_at_inference = ["past_key_values"]

    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",

        "layers.*.rosa_attn.q_proj": "colwise",
        "layers.*.rosa_attn.k_proj": "colwise",
        "layers.*.rosa_attn.v_proj": "colwise",
        "layers.*.rosa_attn.o_proj": "rowwise",
        "layers.*.rosa_attn.v_emb0": "local_rowwise",
        "layers.*.rosa_attn.v_emb1": "local_rowwise",

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
            use_rosa_attention: bool = True,
            num_rosa_heads: int = 8,
            num_rosa_key_value_heads: int = 2,
            num_rosa_query_key_bits: int = 8,
            num_rosa_value_bits: int = 16,
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

        self.use_rosa_attention = use_rosa_attention
        self.num_rosa_heads = num_rosa_heads
        self.num_rosa_key_value_heads = num_rosa_key_value_heads
        self.num_rosa_query_key_bits = num_rosa_query_key_bits
        self.num_rosa_value_bits = num_rosa_value_bits


class RosaCache(DynamicCache):
    def __init__(self,
            ddp_cache_data = None,
            config: MiniRosaConfig = None,
            offloading = False,
            offload_only_non_sliding = False,
    ):
        super().__init__(
            ddp_cache_data=ddp_cache_data,
            config=config,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

        self.roas_layers: Dict[int, Dict[int, List[RapidOnlineSuffixAutomaton]]] = [{} for _ in range(config.num_hidden_layers)]
    
    def update_rosa_sam(self,
            query_states: torch.Tensor,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
    ):
        import torch._dynamo as dynamo
        from contextlib import nullcontext

        disable_compile = dynamo.disable() if dynamo.is_compiling() else nullcontext()
        with disable_compile:
            qqq = (query_states > 0).long() << torch.arange(query_states.size(-1), device=query_states.device)
            qqq = qqq.sum(dim=-1).cpu().tolist()

            kkk = (key_states > 0).long() << torch.arange(key_states.size(-1), device=key_states.device)
            kkk = kkk.sum(dim=-1).cpu().tolist()

            vvv = (value_states > 0).long() << torch.arange(value_states.size(-1), device=value_states.device)
            vvv = vvv.sum(dim=-1).cpu().tolist()

            ooo = []
            for b, (qq, kk, vv) in enumerate(zip(qqq, kkk, vvv)):
                states = self.roas_layers[layer_idx]
                if b not in states:
                    states[b] = [RapidOnlineSuffixAutomaton() for _ in range(len(qq))]
                
                oo = []
                for i, (q, k, v) in enumerate(zip(qq, kk, vv)):
                    o = states[b][i].extend(q, k, v, 0)
                    oo.append(o)
                ooo.append(oo)
            output = torch.tensor(ooo).to(value_states.device)
            output = output.unsqueeze(-1) >> torch.arange(value_states.size(-1), device=value_states.device)
            output = output & 1
            return output.type_as(value_states)
    

class RosaAttention(nn.Module):
    def __init__(self, config: MiniRosaConfig, layer_idx: int, tau: float = 1.0) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        assert config.use_rosa_attention
        assert config.num_rosa_heads % config.num_rosa_key_value_heads == 0

        self.n_local_heads = config.num_rosa_heads
        self.n_local_kv_heads = config.num_rosa_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.hidden_size // config.num_attention_heads

        self.num_value_bits = config.num_rosa_value_bits
        self.num_query_key_bits = config.num_rosa_query_key_bits

        self.q_proj = nn.Linear(config.hidden_size, config.num_rosa_heads * config.num_rosa_query_key_bits, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, config.num_rosa_key_value_heads * config.num_rosa_query_key_bits, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, config.num_rosa_key_value_heads * config.num_rosa_value_bits, bias=False)
        self.o_proj = nn.Linear(config.num_rosa_heads * config.num_rosa_value_bits, config.hidden_size, bias=False)

        self.v_emb0 = nn.Parameter(torch.full((config.num_rosa_heads * config.num_rosa_value_bits,), -1e-5))
        self.v_emb1 = nn.Parameter(torch.full((config.num_rosa_heads * config.num_rosa_value_bits,), +1e-5))

        self.register_buffer("tau", torch.tensor([tau], dtype=torch.float32))
    
    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            past_key_values: Optional[RosaCache] = None,
            **kwargs
    ):
        bsz, seq_len, _ = hidden_states.size()

        query_states = self.q_proj(hidden_states).view(bsz, seq_len, -1, self.num_query_key_bits)
        key_states = self.k_proj(hidden_states).view(bsz, seq_len, -1, self.num_query_key_bits)
        value_states = self.v_proj(hidden_states).view(bsz, seq_len, -1, self.num_value_bits)

        query_states = query_states.transpose(1, 2)
        key_states = repeat_kv(key_states.transpose(1, 2), self.n_rep)
        value_states = repeat_kv(value_states.transpose(1, 2), self.n_rep)

        if past_key_values is not None:
            hidden_states = past_key_values.update_rosa_sam(query_states, key_states, value_states, self.layer_idx)
            v_emb0 = self.v_emb0.view(1, self.n_local_heads, 1, -1)
            v_emb1 = self.v_emb1.view(1, self.n_local_heads, 1, -1)
            output = v_emb0 * (1 - hidden_states) + v_emb1 * hidden_states
        else:
            tau = self.get_buffer("tau")

            attn_mask = None
            if attention_mask is not None:
                attn_mask = attention_mask[:, :, :, :key_states.size(-2)]
            
            v_emb0 = self.v_emb0.view(1, self.n_local_heads, 1, -1)
            v_emb1 = self.v_emb1.view(1, self.n_local_heads, 1, -1)
            output = rosa_qkv_ops(
                query_states, key_states,
                value_states, v_emb0, v_emb1,
                attn_mask=attn_mask, tau=tau,
            )
            
        output = output.transpose(1, 2).reshape(bsz, seq_len, -1)
        output = self.o_proj(output)
        return output


class MiniRosaDecoderLayer(Qwen3DecoderLayer):
    def __init__(self, config: MiniRosaConfig, layer_idx):
        super().__init__(config, layer_idx)
        if config.use_rosa_attention:
            self.rosa_attn = RosaAttention(config, layer_idx)
        else:
            self.rosa_attn = None
    
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

        # Rosa Attention
        if self.rosa_attn is not None:
            residual = residual + self.rosa_attn(
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


class MiniRosaModel(Qwen3Model):
    config_class = MiniRosaConfig

    def __init__(self, config):
        super().__init__(config)
        self.layers = nn.ModuleList([
            MiniRosaDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ])
        self.post_init()
    
    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)

        if isinstance(module, RosaAttention):
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
            past_key_values = RosaCache(config=self.config)

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
    

class MiniRosaForCausalLM(Qwen3ForCausalLM):
    config_class = MiniRosaConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = MiniRosaModel(config)
        self.post_init()

AutoConfig.register("minirosa", MiniRosaConfig)
AutoModel.register(MiniRosaConfig, MiniRosaForCausalLM)
AutoModelForCausalLM.register(MiniRosaConfig, MiniRosaForCausalLM)


if __name__ == "__main__":
    config = MiniRosaConfig(num_hidden_layers=1)

    # config.use_cache = True

    model = MiniRosaForCausalLM(config)
    # model = torch.compile(model)
    print(model)

    x = torch.randint(0, config.vocab_size, size=(1, 512))
    print(x.size())
    m = torch.randint(0, 2, size=(1, 512))

    x = model(x, labels=x, attention_mask=m)
    print(x)