#!/usr/bin/env python
"""
fine_tune_qlora.py
-------------------

This script provides a basic template for fine‑tuning a large language model using
QLoRA.  QLoRA (Quantized Low‑Rank Adapter) combines 4‑bit quantisation with
parameter‑efficient fine‑tuning to dramatically reduce the memory footprint and
compute required to adapt a base model.

The script expects a dataset of instruction–response pairs in JSONL format.  Each
line should be a JSON object with ``instruction``, ``input`` and ``output`` keys.
Only the ``instruction`` and ``output`` fields are used here; the ``input`` field
is concatenated into the prompt for completeness.

Usage:
    python fine_tune_qlora.py \
        --dataset training_data/instructions.jsonl \
        --model meta-llama/Llama-3-8b-instruct \
        --output_dir training_data/qlora-adapter \
        --epochs 2 \
        --batch_size 2 \
        --learning_rate 2e-4

Arguments:
    --dataset      Path to the JSONL file containing instruction pairs.
    --model        HuggingFace model identifier for the base model (e.g. meta-llama/Llama-3-8b-instruct).
    --output_dir   Directory to save the trained adapter.  Will be created if it doesn't exist.
    --epochs       Number of training epochs (default: 3).
    --batch_size   Per‑device training batch size (default: 2).
    --learning_rateLearning rate for the LoRA parameters (default: 2e‑4).
    --max_length   Maximum sequence length (default: 1024).

This script is designed for illustration; you should adjust hyperparameters,
tokeniser truncation strategy and prompt formatting for your specific use case.

Note: Running this script requires a GPU with enough memory to load the base model
and fine‑tune it.  If you encounter out‑of‑memory errors, try reducing the batch
size, lowering the ``max_length`` or using a smaller base model.
"""

import argparse
import json
import os
from typing import Dict

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine‑tune a model with QLoRA")
    parser.add_argument("--dataset", required=True, help="Path to JSONL dataset file")
    parser.add_argument("--model", required=True, help="Base model identifier (e.g. meta-llama/Llama-3-8b-instruct)")
    parser.add_argument("--output_dir", required=True, help="Where to save the trained adapter")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2, help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4, help="Learning rate for LoRA parameters")
    parser.add_argument("--max_length", type=int, default=1024, help="Maximum tokenised sequence length")
    return parser.parse_args()


def format_example(example: Dict[str, str]) -> str:
    """Format an instruction example into a prompt string.

    You can customise this function to include system prompts or different separators.
    """
    instruction = example.get("instruction", "").strip()
    input_field = example.get("input", "").strip()
    if input_field:
        prompt = f"### Instruction\n{instruction}\n\n### Input\n{input_field}\n\n### Response\n"
    else:
        prompt = f"### Instruction\n{instruction}\n\n### Response\n"
    return prompt


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load dataset from JSONL
    print(f"Loading dataset from {args.dataset}...")
    raw_dataset = load_dataset("json", data_files={"train": args.dataset})["train"]

    # Load tokenizer and model
    print(f"Loading model and tokenizer: {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    # Ensure padding and eos tokens are defined for generation
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Freeze base model weights
    for param in model.parameters():
        param.requires_grad = False
    model.enable_input_require_grads()

    # Prepare LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Tokenise dataset
    def tokenize_fn(example: Dict[str, str]) -> Dict[str, torch.Tensor]:
        prompt = format_example(example)
        target = example.get("output", "").strip() + tokenizer.eos_token
        full_text = prompt + target
        tokens = tokenizer(
            full_text,
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        tokens["labels"] = tokens["input_ids"].copy()
        return tokens

    tokenised_ds = raw_dataset.map(tokenize_fn, remove_columns=raw_dataset.column_names)

    # Data collator for language modelling
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=50,
        save_steps=500,
        evaluation_strategy="no",
        save_total_limit=3,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenised_ds,
        data_collator=data_collator,
    )

    # Train
    print("Starting fine‑tuning...")
    trainer.train()
    print("Training completed. Saving the adapter...")

    # Save PEFT adapter and tokenizer
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


if __name__ == "__main__":
    main()