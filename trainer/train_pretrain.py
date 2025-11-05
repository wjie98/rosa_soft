import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import os
import sys
import math
import json

from pathlib import Path
from torch import Tensor
from typing import *


from tokenizers import Tokenizer
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
from transformers import Trainer, TrainingArguments, TrainerCallback, TrainerState, TrainerControl


MINIROSA_NAME = "minirosa"
MINIROSA_ROOT = Path(__file__).resolve().parent.parent / MINIROSA_NAME

sys.path.append(str(MINIROSA_ROOT.parent))
from minirosa.model_minirosa import MiniRosaConfig, MiniRosaForCausalLM, RosaAttention

AutoConfig.register(MINIROSA_NAME, MiniRosaConfig)
AutoModelForCausalLM.register(MiniRosaConfig, MiniRosaForCausalLM)



os.environ["TOKENIZERS_PARALLELISM"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer: Tokenizer, max_length: int = 512, max_item: int = 1024):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples: List[Dict[str, str]] = []

        with open(data_path, 'r', encoding="utf-8") as f:
            for i, line in enumerate(f):
                data = json.loads(line.strip())
                self.samples.append(data)
                if len(self.samples) >= max_item:
                    break

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index: int):
        sample = self.samples[index]

        encoding = self.tokenizer(
            str(sample["text"]),
            max_length=self.max_length + 1,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        input_ids = encoding.input_ids.squeeze()
        attention_mask = (input_ids != self.tokenizer.pad_token_id).long()
        
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        input_ids = input_ids[:-1]
        attention_mask = attention_mask[:-1]
        labels = labels[1:]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

class TauDecayCallback(TrainerCallback):
    def __init__(self, initial_tau: float = 1.0, final_tau: float = 1e-5, decay_type: str = "cosine"):
        super().__init__()
        self.initial_tau = initial_tau
        self.final_tau = final_tau
        self.decay_type = decay_type

    def on_step_begin(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, model: MiniRosaForCausalLM, **kwargs):
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

        for module in model.modules():
            if isinstance(module, RosaAttention):
                module.get_buffer("tau").fill_(new_tau)

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MINIROSA_ROOT)
    config = MiniRosaConfig(vocab_size=tokenizer.vocab_size)
    model = AutoModelForCausalLM.from_config(config)

    train_dataset = PretrainDataset(
        data_path="D:\\datasets\\minimind_dataset\\pretrain_hq.jsonl",
        tokenizer=tokenizer,
    )

    training_args = TrainingArguments(
        output_dir="output",
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        logging_steps=10,
        num_train_epochs=1,
        save_steps=100,
        learning_rate=1e-4,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        # adam_beta1=0.9,
        # adam_beta2=0.999,
        # adam_epsilon=1e-8,
    )

    tau_callback = TauDecayCallback()

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=None,
        callbacks=[tau_callback],
    )

    trainer.train()
