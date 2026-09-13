#!/usr/bin/env python3
"""Fine-tune Qwen2.5-0.5B with PEFT LoRA or all model weights."""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import platform
import socket
import sys
import time
from pathlib import Path

from chat_format import CHAT_TEMPLATE


BASE_MODEL = "Qwen/Qwen2.5-0.5B"
EXPECTED_LORA_PARAMETERS_PER_RANK = 549_888


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("lora", "full"), required=True)
    parser.add_argument("--rank", type=int, choices=(1, 4, 16))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=512)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=601768)
    parser.add_argument("--num-workers", type=int, default=4)
    return parser.parse_args()


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def validate(args: argparse.Namespace) -> None:
    if args.mode == "lora" and args.rank is None:
        raise SystemExit("--rank is required in LoRA mode")
    if args.mode == "full" and args.rank is not None:
        raise SystemExit("--rank is only valid in LoRA mode")
    if (args.output_dir / "metrics.json").exists():
        raise SystemExit(
            f"A completed run already exists at {args.output_dir}. "
            "The launcher skips completed runs automatically."
        )
    for relative in ("train", "validation", "manifest.json"):
        if not (args.data_dir / relative).exists():
            raise SystemExit(f"Missing prepared data item: {args.data_dir / relative}")


def tokenize_dataset(dataset, tokenizer, max_length: int):
    columns = dataset.column_names

    def tokenize(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return tokenizer(text, truncation=True, max_length=max_length, add_special_tokens=False)

    return dataset.map(tokenize, remove_columns=columns, desc="Applying chat template and tokenizing")


def write_loss_history(log_history: list[dict], output_path: Path) -> None:
    rows = []
    for record in log_history:
        if "loss" in record:
            rows.append(
                {
                    "step": record.get("step"),
                    "epoch": record.get("epoch"),
                    "split": "train",
                    "loss": record["loss"],
                }
            )
        if "eval_loss" in record:
            rows.append(
                {
                    "step": record.get("step"),
                    "epoch": record.get("epoch"),
                    "split": "validation",
                    "loss": record["eval_loss"],
                }
            )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("step", "epoch", "split", "loss"))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    validate(args)
    try:
        import torch
        from datasets import load_from_disk
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit("Install hw_empirical/requirements.txt in the GPU environment") from exc
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for this training experiment")
    if torch.cuda.device_count() != 1:
        raise SystemExit(
            f"Expected exactly one visible GPU, but found {torch.cuda.device_count()}. "
            "Set CUDA_VISIBLE_DEVICES to one device so runtime and peak-memory "
            "measurements remain comparable."
        )
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("The experiment requires a GPU with bfloat16 support")

    set_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False

    if args.mode == "lora":
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError as exc:
            raise SystemExit("PEFT is required for LoRA runs") from exc
        config = LoraConfig(
            r=args.rank,
            lora_alpha=2 * args.rank,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
        model = get_peft_model(model, config)
        # Reentrant checkpointing needs at least one grad-requiring input. The
        # base embeddings are frozen by PEFT, so retain gradients on their output.
        model.enable_input_require_grads()

    trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    if args.mode == "lora":
        expected = EXPECTED_LORA_PARAMETERS_PER_RANK * args.rank
        if trainable_parameters != expected:
            raise RuntimeError(
                f"Expected {expected:,} trainable parameters for rank {args.rank}, "
                f"but PEFT exposed {trainable_parameters:,}. Check the model and PEFT versions."
            )

    train_dataset = tokenize_dataset(
        load_from_disk(str(args.data_dir / "train")), tokenizer, args.max_seq_length
    )
    validation_dataset = tokenize_dataset(
        load_from_disk(str(args.data_dir / "validation")), tokenizer, args.max_seq_length
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(args.output_dir / "checkpoints"),
        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        per_device_eval_batch_size=args.eval_batch_size,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="no",
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        # Non-reentrant checkpointing can fail metadata validation in SDPA when
        # the attention mask is recomputed. Reentrant mode avoids that code path.
        gradient_checkpointing_kwargs={"use_reentrant": True},
        optim="adamw_torch",
        lr_scheduler_type="constant",
        warmup_steps=0,
        weight_decay=0.0,
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        data_collator=collator,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    baseline_reserved = torch.cuda.memory_reserved()
    started = time.perf_counter()
    train_result = trainer.train()
    final_validation = trainer.evaluate()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started

    saved_model_dir = args.output_dir / "model"
    trainer.save_model(str(saved_model_dir))
    tokenizer.save_pretrained(str(saved_model_dir))
    write_loss_history(trainer.state.log_history, args.output_dir / "loss_history.csv")

    metrics = {
        "variant": "full" if args.mode == "full" else f"lora_r{args.rank}",
        "mode": args.mode,
        "rank": args.rank,
        "base_model": BASE_MODEL,
        "trainable_parameters": trainable_parameters,
        "expected_lora_parameters": (
            EXPECTED_LORA_PARAMETERS_PER_RANK * args.rank if args.mode == "lora" else None
        ),
        "total_parameters_in_process_model": total_parameters,
        "trainable_fraction": trainable_parameters / total_parameters,
        "peak_gpu_memory_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_gpu_memory_reserved_bytes": torch.cuda.max_memory_reserved(),
        "baseline_gpu_memory_allocated_bytes": baseline_allocated,
        "baseline_gpu_memory_reserved_bytes": baseline_reserved,
        "wall_time_seconds_including_final_eval": elapsed,
        "train_metrics": train_result.metrics,
        "final_validation_metrics": final_validation,
        "effective_batch_size": args.micro_batch_size * args.gradient_accumulation_steps,
        "training_examples": len(train_dataset),
        "validation_examples": len(validation_dataset),
        "gpu": torch.cuda.get_device_name(0),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "versions": {
            name: package_version(name)
            for name in ("torch", "transformers", "datasets", "peft", "accelerate")
        },
        "command": sys.argv,
        "settings": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: metrics[key] for key in (
        "variant", "trainable_parameters", "peak_gpu_memory_reserved_bytes", "wall_time_seconds_including_final_eval"
    )}, indent=2))


if __name__ == "__main__":
    main()
