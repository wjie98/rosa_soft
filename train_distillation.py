import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import os
import re
import math
from pathlib import Path
from typing import *

from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    get_scheduler,
)

from accelerate import Accelerator

from minirosa import (
    RosaBase,
    RosaConfig,
    RosaAttention,
    SuffixAttention,
    calculate_distillation_loss,
    calculate_tau_decay,
    setup_tf32, verify_tf32_status,
)

import logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)

setup_tf32()
verify_tf32_status()


def load_student_model(args):
    sliding_window = args.sliding_window

    logger.info("Loading student model configuration...")
    config = AutoConfig.from_pretrained(args.model_path)
    config.use_cache = False

    logger.info(f"Enabling sliding window attention with window size {sliding_window}")
    config.use_sliding_window = True
    config.sliding_window = sliding_window
    config.max_window_layers = config.num_hidden_layers
    config.layer_types = ["sliding_attention"] * config.num_hidden_layers

    logger.info("Loading student model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, dtype=torch.bfloat16)

    if args.adapter_type == "rosa":
        logger.info("Applying RosaAttention adapter to the student model.")
        adapter_class = RosaAttention

    elif args.adapter_type == "sufa":
        logger.info("Applying SuffixAttention adapter to the student model.")
        adapter_class = SuffixAttention

    else:
        raise ValueError(f"Unknown adapter type: {args.adapter_type}")
    
    rosa_config = RosaConfig(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
    )

    adapter_class.apply_adapter_to_modules_(model, config=rosa_config)
    
    logger.info("Setting trainable parameters for the adapter.")
    adapter_class.apply_trainable_parameters_(model)

    if args.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    return model

def load_teacher_model(args):
    logger.info("Loading teacher model configuration...")
    config = AutoConfig.from_pretrained(args.model_path)
    config.use_cache = False

    logger.info("Loading teacher model...")
    model = AutoModelForCausalLM.from_pretrained(args.model_path, config=config, dtype=torch.bfloat16)
    model.requires_grad_(False)
    model.eval()
    return model


def evaluate_model(args, model: nn.Module, dataloader: DataLoader, accelerator: Accelerator):
    model.eval()

    eval_loss = []
    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            student_outputs = model(**batch)
        
        eval_loss.append(accelerator.gather(student_outputs.loss))

        if accelerator.is_local_main_process:
            logger.info(f"evaluation batch {i + 1}/{args.max_eval_batches} loss")
        
        if i + 1 >= args.max_eval_batches:
            break
    
    eval_loss = torch.cat(eval_loss).mean().item()
    eval_perplexity = math.exp(eval_loss)

    return {
        "eval/loss": eval_loss,
        "eval/perplexity": eval_perplexity,
    }

class StateTracker:
    def __init__(self):
        self.completed_steps = 0
        self.starting_epoch = 0
    
    def state_dict(self):
        return {
            k: getattr(self, k)
            for k in self.__dict__
        }
    
    def load_state_dict(self, state):
        for k, v in state.items():
            if k in self.__dict__:
                setattr(self, k, v)


def main(args):
    from tqdm.auto import tqdm

    num_train_epochs = args.num_train_epochs
    log_every_steps = args.log_every_steps
    save_every_steps = args.save_every_steps
    eval_every_steps = args.eval_every_steps

    per_device_train_batch_size = args.per_device_train_batch_size
    per_device_eval_batch_size = args.per_device_eval_batch_size
    gradient_accumulation_steps = args.gradient_accumulation_steps

    distill_alpha = args.distill_alpha
    distill_temperature = args.distill_temperature

    accelerator = Accelerator(
        gradient_accumulation_steps=gradient_accumulation_steps,
        log_with="swanlab",
    )

    accelerator.init_trackers(
        project_name=args.project_name,
        config=args.__dict__,
        init_kwargs={
            "swanlab": {"experiment_name": args.experiment_name},
        }
    )
    
    with accelerator.main_process_first():
        teacher_model = load_teacher_model(args)
    
    with accelerator.main_process_first():
        student_model = load_student_model(args)
    
    with accelerator.main_process_first():
        datasets = load_from_disk(args.dataset_path)
    
    datasets.set_format(type="torch", columns=["input_ids", "labels", "attention_mask"])

    train_dataloader = DataLoader(
        datasets["train"],
        batch_size=per_device_train_batch_size,
        shuffle=True,
    )

    eval_dataloader = DataLoader(
        datasets["test"],
        batch_size=per_device_eval_batch_size,
    )

    global_batch_size = (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * accelerator.num_processes
    )
    logger.info(f"Global batch size: {global_batch_size}")

    train_steps = (
        len(train_dataloader)
        // accelerator.num_processes
        * num_train_epochs
        // gradient_accumulation_steps
    )
    logger.info(f"Training steps: {train_steps}")
    
    logger.info("Trainable parameters:")
    trainable_named_parameters = {name: p for name, p in student_model.named_parameters() if p.requires_grad}
    if accelerator.is_local_main_process:
        for name, p in trainable_named_parameters.items():
            logger.info(f"{name} {p.size()} {p.numel()}")

    optimizer = torch.optim.AdamW(
        trainable_named_parameters.values(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=train_steps,
    )

    student_model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
        student_model, optimizer, train_dataloader, eval_dataloader, scheduler
    )
    
    teacher_model = teacher_model.to(accelerator.device)

    state_tracker = StateTracker()
    accelerator.register_for_checkpointing(state_tracker)

    output_path = Path(args.output_path).expanduser().resolve()
    if args.resume_from_checkpoint and output_path.exists():
        checkpoint_path = None
        resume_from_steps = 0
        for sub in output_path.iterdir():
            if not re.match(r"checkpoint_\d+", sub.name):
                continue
            
            s = int(sub.name.split("_")[-1])

            if s > resume_from_steps:
                resume_from_steps = s
                checkpoint_path = sub
        
        if checkpoint_path is not None:
            accelerator.load_state(checkpoint_path)
            accelerator.print(f"Resumed from checkpoint. Current state: {state_tracker}")

    progress_bar = tqdm(total=train_steps, desc="Training", disable=not accelerator.is_local_main_process)
    progress_bar.update(state_tracker.completed_steps)

    for _ in range(state_tracker.starting_epoch, num_train_epochs):
        for _, batch in enumerate(train_dataloader):
            if accelerator.sync_gradients and (state_tracker.completed_steps % eval_every_steps == 0):
                eval_outputs = evaluate_model(args, student_model, eval_dataloader, accelerator)
                accelerator.log(eval_outputs, step=state_tracker.completed_steps)
            
            student_model.train()
            with accelerator.accumulate(student_model):
                with torch.no_grad():
                    teacher_outputs = teacher_model(**batch)
                student_outputs = student_model(**batch)
                
                loss = calculate_distillation_loss(
                    loss_ce=student_outputs.loss,
                    student_logits=student_outputs.logits,
                    teacher_logits=teacher_outputs.logits,
                    alpha=distill_alpha,
                    temperature=distill_temperature,
                )

                accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(trainable_named_parameters.values(), max_norm = 1.0)
            
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                current_tau = calculate_tau_decay(
                    step=state_tracker.completed_steps,
                    total_steps=train_steps,
                )
                RosaBase.update_tau_(student_model, current_tau)

                if state_tracker.completed_steps % log_every_steps == 0:
                    avg_loss = accelerator.gather(loss).mean().item()
                    accelerator.log({
                        "train/loss": avg_loss,
                        "train/learning_rate": scheduler.get_last_lr()[0],
                        "train/tau": current_tau,
                    }, step=state_tracker.completed_steps)

                state_tracker.completed_steps += 1
                progress_bar.update(1)
            
                if state_tracker.completed_steps % save_every_steps == 0: # skip the first step
                    checkpoint_path = output_path / f"checkpoint_{state_tracker.completed_steps}"
                    checkpoint_path.mkdir(parents=True, exist_ok=True)
                    accelerator.save_state(checkpoint_path)
            
            del student_outputs, teacher_outputs, loss
        
        state_tracker.starting_epoch += 1
    
    checkpoint_path = output_path / f"checkpoint_{state_tracker.completed_steps}"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    accelerator.save_state(checkpoint_path)
    
    accelerator.end_training()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ROSA Distillation Training")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base pretrained model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the preprocessed dataset.")
    parser.add_argument("--output_path", type=str, default="output", help="Path to save checkpoints and final model.")
    parser.add_argument("--project_name", type=str, default="rosa_soft", help="Name of the project.")
    parser.add_argument("--experiment_name", type=str, default="rosa_distillation", help="Name of the experiment.")

    # model args
    parser.add_argument("--adapter_type", type=str, default="rosa", help="Type of ROSA adapter [rosa, sufa]", choices=["rosa", "sufa"])
    parser.add_argument("--sliding_window", type=int, default=128, help="Sliding window size for the student model.")

    # logging and saving
    parser.add_argument("--log_every_steps", type=int, default=10, help="Number of steps between logging.")
    parser.add_argument("--save_every_steps", type=int, default=2000, help="Number of steps between model saves.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the last checkpoint in output_dir.")

    parser.add_argument("--eval_every_steps", type=int, default=100, help="Number of steps between evaluations.")
    parser.add_argument("--max_eval_batches", type=int, default=10, help="Maximum number of evaluation samples.")
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8, help="Batch size per GPU for evaluation.")

    # optimizer args
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5, help="Initial learning rate.")
    parser.add_argument("--warmup_steps", type=int, default=2000, help="Warmup steps for learning rate scheduler.")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer.")

    # distillation args
    parser.add_argument("--distill_alpha", type=float, default=1.0, help="Alpha for distillation.")
    parser.add_argument("--distill_temperature", type=float, default=1.0, help="Temperature for distillation.")

    # training args
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--gradient_checkpointing", action="store_true", help="Use gradient checkpointing.")

    args = parser.parse_args()
    main(args)
