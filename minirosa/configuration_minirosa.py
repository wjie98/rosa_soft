from transformers import PretrainedConfig

__all__ = ["MiniRosaConfig"]


class MiniRosaConfig(PretrainedConfig):
    model_type = "minirosa"

    def __init__(
            self,
            hidden_act: str = 'silu',
            hidden_size: int = 512,
            intermediate_size: int | None = None,
            max_position_embeddings: int = 32768,
            num_attention_heads: int = 8,
            num_key_value_heads: int = 2,
            num_hidden_layers: int = 2,
            vocab_size: int = 6400,
            rms_norm_eps: float = 1e-5,
            rope_theta: int = 1e6,
            inference_rope_scaling: bool = False,
            use_rosa_attention: bool = False,
            num_rosa_heads: int = 8,
            num_rosa_key_value_heads: int = 2,
            num_rosa_query_key_bits: int = 8,
            num_rosa_value_bits: int = 16,
            tie_word_embeddings: bool = True,
            **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.num_key_value_heads = num_key_value_heads
        self.vocab_size = vocab_size
        self.rms_norm_eps = rms_norm_eps
        self.rope_theta = rope_theta
        self.inference_rope_scaling = inference_rope_scaling
        self.rope_scaling = dict(
            beta_fast = 4,
            beta_slow = 1,
            factor = 4,
            original_max_position_embeddings = 2048,
            type = "yarn",
        ) if self.inference_rope_scaling else None

        self.use_rosa_attention = use_rosa_attention
        self.num_rosa_heads = num_rosa_heads
        self.num_rosa_key_value_heads = num_rosa_key_value_heads
        self.num_rosa_query_key_bits = num_rosa_query_key_bits
        self.num_rosa_value_bits = num_rosa_value_bits
        self.tie_word_embeddings = tie_word_embeddings

