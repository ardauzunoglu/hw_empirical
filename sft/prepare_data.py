#!/usr/bin/env python3
"""Materialize deterministic UltraChat train/validation subsets and prompts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--train-size", type=int, default=1000)
    parser.add_argument("--validation-size", type=int, default=100)
    parser.add_argument("--generation-prompts", type=int, default=5)
    parser.add_argument("--revision", default=None, help="Optional Hugging Face dataset revision for strict pinning")
    return parser.parse_args()


def first_user_turn(messages: list[dict[str, str]]) -> list[dict[str, str]] | None:
    for message in messages:
        if message.get("role") == "user" and message.get("content"):
            return [{"role": "user", "content": message["content"]}]
    return None


def main() -> None:
    args = parse_args()
    if min(args.train_size, args.validation_size, args.generation_prompts) <= 0:
        raise SystemExit("All subset sizes must be positive")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Install hw_empirical/requirements.txt before preparing data") from exc

    train_path = args.output_dir / "train"
    validation_path = args.output_dir / "validation"
    prompt_path = args.output_dir / "heldout_prompts.json"
    occupied = [path for path in (train_path, validation_path, prompt_path) if path.exists()]
    if occupied:
        raise SystemExit(
            "Refusing to overwrite an existing fixed subset: "
            + ", ".join(map(str, occupied))
            + ". Remove or move it explicitly to prepare a new subset."
        )

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", revision=args.revision)
    train_source = dataset["train_sft"]
    validation_source = dataset["test_sft"]
    if len(train_source) < args.train_size or len(validation_source) < args.validation_size:
        raise SystemExit("Requested subset is larger than the available split")

    # Selecting leading indices makes the exact sample identity shared across all runs.
    train = train_source.select(range(args.train_size))
    validation = validation_source.select(range(args.validation_size))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    train.save_to_disk(str(train_path))
    validation.save_to_disk(str(validation_path))

    prompts = []
    for index, example in enumerate(validation_source):
        messages = first_user_turn(example["messages"])
        if messages is not None:
            prompts.append({"id": f"test_sft_{index}", "messages": messages})
        if len(prompts) == args.generation_prompts:
            break
    if len(prompts) != args.generation_prompts:
        raise SystemExit("Could not find enough held-out user prompts")
    prompt_path.write_text(json.dumps(prompts, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    manifest = {
        "dataset": "HuggingFaceH4/ultrachat_200k",
        "revision": args.revision,
        "train_split": "train_sft",
        "train_indices": [0, args.train_size - 1],
        "train_size": len(train),
        "validation_split": "test_sft",
        "validation_indices": [0, args.validation_size - 1],
        "validation_size": len(validation),
        "train_fingerprint": train._fingerprint,
        "validation_fingerprint": validation._fingerprint,
        "generation_prompt_ids": [item["id"] for item in prompts],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Saved fixed subsets and prompt manifest to {args.output_dir}")


if __name__ == "__main__":
    main()

