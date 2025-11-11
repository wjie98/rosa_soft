import logging
import os
from pathlib import Path
from typing import Dict, Any

import torch
from datasets import load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    TrainingArguments,
    default_data_collator,
)
from minirosa import RosaAttention, SuffixAttention, RosaConfig, RosaTauDecayCallback, RosaTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger(__name__)


def create_student_model_config(model_path: Path, sliding_window: int) -> AutoConfig:
    config = AutoConfig.from_pretrained(model_path)
    config.use_cache = False

    logger.info(f"Enabling sliding window attention with window size {sliding_window}")
    config.use_sliding_window = True
    config.sliding_window = sliding_window
    config.max_window_layers = config.num_hidden_layers
    config.layer_types = ["sliding_attention"] * config.num_hidden_layers

    return config

def setup_rosa_adapter(model: PreTrainedModel, base_config: AutoConfig) -> PreTrainedModel:
    logger.info("Applying RosaAttention adapter to the student model.")
    rosa_config = RosaConfig(
        hidden_size=base_config.hidden_size,
        num_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
    )
    RosaAttention.apply_adapter_to_modules_(model, config=rosa_config)
    
    logger.info("Setting trainable parameters for the adapter.")
    RosaAttention.apply_trainable_parameters_(model)
    
    logger.info("Trainable parameters:")
    RosaAttention.print_trainable_parameters(model)

    model.enable_input_require_grads()

    return model


def setup_sufa_adapter(model: PreTrainedModel, base_config: AutoConfig) -> PreTrainedModel:
    logger.info("Applying SuffixAttention adapter to the student model.")
    rosa_config = RosaConfig(
        hidden_size=base_config.hidden_size,
        num_heads=base_config.num_attention_heads,
        num_key_value_heads=base_config.num_key_value_heads,
    )
    SuffixAttention.apply_adapter_to_modules_(model, config=rosa_config)
    
    logger.info("Setting trainable parameters for the adapter.")
    SuffixAttention.apply_trainable_parameters_(model)
    
    logger.info("Trainable parameters:")
    SuffixAttention.print_trainable_parameters(model)

    model.enable_input_require_grads()

    return model


def main(
    model_path: str,
    dataset_path: str,
    output_dir: str,
    sliding_window: int = 128,
    resume_from_checkpoint: bool = False,
    per_device_train_batch_size: int = 1,
    gradient_accumulation_steps: int = 1,
    learning_rate: float = 2e-5,
    num_train_epochs: int = 1,
    adapter_type: str = "rosa",
):
    model_path = Path(model_path).expanduser().resolve()
    dataset_path = Path(dataset_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()

    logger.info("Starting ROSA distillation training run...")
    logger.info(f"  - Model Path: {model_path}")
    logger.info(f"  - Dataset Path: {dataset_path}")
    logger.info(f"  - Output Directory: {output_dir}")

    logger.info("Loading teacher model...")
    teacher_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)

    logger.info("Loading student model configuration...")
    student_config = create_student_model_config(model_path, sliding_window)
    
    logger.info("Loading student model...")
    student_model = AutoModelForCausalLM.from_pretrained(model_path, config=student_config, dtype=torch.bfloat16)
    if adapter_type == "rosa":
        student_model = setup_rosa_adapter(student_model, student_config)
    elif adapter_type == "sufa":
        student_model = setup_sufa_adapter(student_model, student_config)
    else:
        raise ValueError(f"Unknown adapter type: {adapter_type}")
    
    logger.info("Loading dataset from disk...")
    datasets = load_from_disk(dataset_path)
    logger.info(f"Dataset loaded: {datasets}")

    logger.info("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        
        logging_steps=1,
        save_strategy="steps",
        save_steps=2000,
        # evaluation_strategy="no",
        
        lr_scheduler_type="cosine",
        warmup_steps=2000,
        # warmup_ratio=0.1,
        weight_decay=0.01,
        adam_beta1=0.9,
        adam_beta2=0.95,
        max_grad_norm=1.0,
        
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=max(1, os.cpu_count() // 2 // torch.cuda.device_count()),
        
        # report_to="none",
    )
    
    global_batch_size = (
        per_device_train_batch_size
        * gradient_accumulation_steps
        * int(os.environ.get("WORLD_SIZE", 1))
    )
    logger.info(f"Global batch size: {global_batch_size}")

    logger.info("Initializing RosaTrainer...")
    trainer = RosaTrainer(
        model=student_model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets.get("test"),
        callbacks=[RosaTauDecayCallback()],
        data_collator=default_data_collator,
        teacher_model=teacher_model,
        distillation_loss_alpha=1.0,
        distillation_temperature=2.0,
    )

    logger.info("Starting training...")
    train_result = trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    
    trainer.save_model()
    trainer.save_state()
    logger.info("Training finished successfully.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run ROSA Distillation Training")
    
    parser.add_argument("--model_path", type=str, required=True, help="Path to the base pretrained model.")
    parser.add_argument("--dataset_path", type=str, required=True, help="Path to the preprocessed dataset.")
    parser.add_argument("--output_dir", type=str, default="output", help="Directory to save checkpoints and final model.")
    parser.add_argument("--sliding_window", type=int, default=128, help="Sliding window size for the student model.")
    parser.add_argument("--resume_from_checkpoint", action="store_true", help="Resume training from the last checkpoint in output_dir.")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Batch size per GPU.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Initial learning rate.")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Total number of training epochs.")
    parser.add_argument("--adapter_type", type=str, default="rosa", help="Type of ROSA adapter [rosa, sufa]", choices=["rosa", "sufa"])

    args = parser.parse_args()
    
    main(
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        sliding_window=args.sliding_window,
        resume_from_checkpoint=args.resume_from_checkpoint,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        adapter_type=args.adapter_type,
    )