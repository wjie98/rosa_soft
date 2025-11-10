import torch
import torch.nn as nn
import torch.nn.functional as F

import re
import math
import json

from torch import Tensor
from typing import *

import inspect
from functools import wraps
from pathlib import Path
from contextlib import contextmanager

from transformers import PreTrainedModel
from transformers import TrainingArguments, TrainerCallback, TrainerState, TrainerControl, Trainer
from transformers.trainer import ADAPTER_WEIGHTS_NAME, ADAPTER_CONFIG_NAME

from transformers.utils import logging
logger = logging.get_logger(__name__)


__all__ = [
    "RosaConfig",
    "RosaBase",
    "RosaTauDecayCallback",
    "RosaTrainer",
]



class RosaConfig:
    def __init__(
            self,
            hidden_size: int = 512,
            num_heads: int = 8,
            num_key_value_heads: int = 2,
            num_query_key_bits: int = 8,
            num_value_bits: int = 8,
            bias: bool = False,
            max_attention_head_size: int = 256,
            **kwargs
    ) -> None:
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_query_key_bits = num_query_key_bits
        self.num_value_bits = num_value_bits
        self.bias = bias
        self.max_attention_head_size = max_attention_head_size


class RosaBase(nn.Module):
    def __init__(self, config: RosaConfig) -> None:
        super().__init__()
        
        assert config.num_heads % config.num_key_value_heads == 0
        self.config = config

        self.register_buffer("tau", torch.ones((1,)), persistent=False)
        self.rosa_enabled = True
    
    @staticmethod
    def repeat_kv(hidden_states: Tensor, n_rep: int) -> Tensor:
        bsz, n_kvs, seq_len, dim = hidden_states.size()
        if n_rep == 1:
            return hidden_states
        hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kvs, n_rep, seq_len, dim)
        return hidden_states.reshape(bsz, n_kvs * n_rep, seq_len, dim)
    
    @classmethod
    def update_tau_(cls, module: nn.Module, tau: float):
        for m in module.modules():
            if isinstance(m, cls):
                m.get_buffer("tau").fill_(tau)
    
    @classmethod
    def apply_adapter_to_modules_(
        cls,
        module: nn.Module,
        adapter_factory: Callable[[int, nn.Module], nn.Module],
        adapter_name: str = "rosa_attn",
        target_modules: str = r".*self_attn",
        autocast_dtype: bool = True,
    ):
        def apply_forward(m: nn.Module, adapter_name: str):
            origin_forward = m.forward

            @wraps(origin_forward)
            def forward(*args, **kwargs):
                adapter: RosaBase = getattr(m, adapter_name)
                if adapter.rosa_enabled:
                    a, *others = origin_forward(*args, **kwargs)
                    b, *_ = getattr(m, adapter_name)(*args, **kwargs)
                    return a + b, *others
                else:
                    return origin_forward(*args, **kwargs)
            forward.__signature__ = inspect.signature(origin_forward)

            m.forward = forward

        count = 0
        for name, m in module.named_modules():
            if not re.fullmatch(target_modules, name):
                continue
            elif hasattr(m, adapter_name):
                continue

            layer_idx = getattr(m, "layer_idx", count)
            adapter = adapter_factory(layer_idx, m)

            for p in m.parameters():
                dtype = p.dtype if autocast_dtype else None
                adapter = adapter.to(device=p.device, dtype=dtype)
                break

            setattr(m, adapter_name, adapter)
            apply_forward(m, adapter_name)

            count += 1
        
        if count == 0:
            raise ValueError("No target modules found to patch.")
        
        logger.info(f"Found {count} attention modules to patch.")

        return module
    
    @classmethod
    def apply_trainable_parameters_(cls, module: nn.Module):
        for p in module.parameters():
            p.requires_grad_(False)

        for m in module.modules():
            if not isinstance(m, cls):
                continue
            for p in m.parameters():
                p.requires_grad_(True)
    
    @classmethod
    def print_trainable_parameters(cls, module: nn.Module):
        for name, p in module.named_parameters():
            if not p.requires_grad:
                continue
            print(name, p.size())


class RosaTauDecayCallback(TrainerCallback):
    def __init__(
            self,
            initial_tau: float = 1.0,
            final_tau: float = 1e-5,
            decay_type: str = "cosine",
    ):
        super().__init__()
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.decay_type = decay_type

        self._current_tau = initial_tau

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: nn.Module, **kwargs):
        if state.max_steps == 0:
            return
        
        progress = state.global_step / state.max_steps
        if self.decay_type == "linear":
            new_tau = self.initial_tau - progress * (self.initial_tau - self.final_tau)
        elif self.decay_type == "cosine":
            cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
            new_tau = self.final_tau + (self.initial_tau - self.final_tau) * cosine_progress
        elif self.decay_type == "exponential":
            decay_rate = (self.final_tau / self.initial_tau) ** (1 / state.max_steps)
            new_tau = self.initial_tau * (decay_rate ** state.global_step)
        else:
            raise ValueError(f"Unsupported decay_type: {self.decay_type}")
        
        new_tau = max(self.final_tau, new_tau)
        RosaBase.update_tau_(model, tau=new_tau)

        self._current_tau = new_tau
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        if state.is_world_process_zero:
            try:
                import swanlab
                swanlab.log({"train/tau": self._current_tau}, step=state.global_step)
            except:
                pass


class RosaTrainer(Trainer):
    def __init__(
            self,
            *args,
            teacher_model: PreTrainedModel,
            distillation_loss_alpha: float = 1.0,
            distillation_temperature: float = 2.0,
            teacher_share_frozen_params: bool = True,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.teacher_model = teacher_model
        self.teacher_model.eval()
            
        self.distillation_loss_alpha = distillation_loss_alpha
        self.distillation_temperature = distillation_temperature

        self.teacher_share_frozen_params = teacher_share_frozen_params
    
    def _share_frozen_params(self, teacher_model: nn.Module, student_model: nn.Module):
        if getattr(teacher_model, "shared_frozen_params", False):
            return
        
        student_parameters = dict(student_model.named_parameters())
        student_buffers = dict(student_model.named_buffers())

        for name, _ in list(teacher_model.named_parameters()):
            if (name not in student_parameters) or student_parameters[name].requires_grad:
                continue

            module_path, param_name = name.rsplit('.', 1)
            module = teacher_model.get_submodule(module_path)
                
            delattr(module, param_name)
            setattr(module, param_name, student_parameters[name])
        
        for name, _ in list(teacher_model.named_buffers()):
            if name not in student_buffers:
                continue

            module_path, param_name = name.rsplit('.', 1)
            module = teacher_model.get_submodule(module_path)
            
            module.register_buffer(param_name, student_buffers[name])
        
        setattr(teacher_model, "shared_frozen_params", True)

    def _save(self, output_dir = None, state_dict = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir

        output_dir = Path(output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        trainable_params = {k: v for k, v in self.model.named_parameters() if v.requires_grad}

        adapter_config = {}
        for name, m in self.model.named_modules():
            if isinstance(m, RosaBase):
                adapter_config = m.config.__dict__
                break
        
        params_save_path = output_dir / ADAPTER_WEIGHTS_NAME
        torch.save(trainable_params, params_save_path)

        config_save_path = output_dir / ADAPTER_CONFIG_NAME
        text = json.dumps(adapter_config)
        config_save_path.write_text(text, encoding="utf-8")

        if hasattr(self.model, "config"):
            self.model.config.save_pretrained(output_dir)
    
    def _load_from_checkpoint(self, resume_from_checkpoint, model = None):
        if model is None:
            model = self.model
        
        resume_from_checkpoint = Path(resume_from_checkpoint).expanduser().resolve()

        params_save_path = resume_from_checkpoint / ADAPTER_WEIGHTS_NAME
        if not params_save_path.exists():
            raise FileNotFoundError(f"Adapter file not found at {params_save_path}")
        
        adapter_weights = torch.load(params_save_path, map_location="cpu")
        incompatible_keys = model.load_state_dict(adapter_weights, strict=False)
        
        # for key in incompatible_keys.missing_keys:
        #     logger.warning(f"Adapter weight '{key}' was not found in the checkpoint.")
    
    def compute_loss(
            self,
            model: nn.Module,
            inputs: Dict[str, Any],
            return_outputs: bool = False,
            num_items_in_batch: Optional[Tensor] = None,
    ):
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        
        # teacher
        if self.teacher_share_frozen_params:
            self._share_frozen_params(self.teacher_model, student_model=model)
            
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # student
        student_outputs = model(**inputs)

        loss_ce = student_outputs.loss
        student_logits = student_outputs.logits

        # KL
        tau = self.distillation_temperature
        loss_distill = F.kl_div(
            input=F.log_softmax(student_logits / tau, dim=-1),
            target=F.softmax(teacher_logits / tau, dim=-1),
            reduction="batchmean",
        ) * (tau ** 2)

        alpha = self.distillation_loss_alpha

        if (loss_ce is None) or (alpha == 1.0):
            final_loss = loss_distill
        else:
            final_loss = alpha * loss_distill + (1.0 - alpha) * loss_ce
        
        if (
            self.args.average_tokens_across_devices
            and (self.model_accepts_loss_kwargs or self.compute_loss_func)
            and num_items_in_batch is not None
        ):
            final_loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (final_loss, student_outputs) if return_outputs else final_loss

