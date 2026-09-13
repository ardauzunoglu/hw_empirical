#!/usr/bin/env python3
"""Combine SFT measurements and plot train/validation loss trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


ORDER = ["base", "lora_r1", "lora_r4", "lora_r16", "full"]
LABELS = {"lora_r1": "LoRA r=1", "lora_r4": "LoRA r=4", "lora_r16": "LoRA r=16", "full": "Full"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-dir", type=Path, required=True)
    parser.add_argument("--generations-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metric_paths = sorted(args.runs_dir.glob("*/metrics.json"))
    if not metric_paths:
        raise SystemExit(f"No run metrics found under {args.runs_dir}")
    metrics = [json.loads(path.read_text(encoding="utf-8")) for path in metric_paths]
    variants = [item["variant"] for item in metrics]
    if len(variants) != len(set(variants)):
        raise SystemExit("Duplicate SFT variants found")

    summary_rows = []
    histories = []
    for item, path in zip(metrics, metric_paths):
        summary_rows.append(
            {
                "variant": item["variant"],
                "rank": item["rank"],
                "trainable_parameters": item["trainable_parameters"],
                "trainable_fraction": item["trainable_fraction"],
                "peak_gpu_memory_allocated_gib": item["peak_gpu_memory_allocated_bytes"] / 2**30,
                "peak_gpu_memory_reserved_gib": item["peak_gpu_memory_reserved_bytes"] / 2**30,
                "training_time_seconds": item["train_metrics"].get("train_runtime"),
                "wall_time_seconds_including_final_eval": item["wall_time_seconds_including_final_eval"],
                "final_training_loss": item["train_metrics"].get("train_loss"),
                "final_validation_loss": item["final_validation_metrics"].get("eval_loss"),
            }
        )
        history = pd.read_csv(path.parent / "loss_history.csv")
        history["variant"] = item["variant"]
        histories.append(history)
    summary = pd.DataFrame(summary_rows)
    summary["sort_order"] = summary.variant.map({name: index for index, name in enumerate(ORDER)})
    summary = summary.sort_values("sort_order").drop(columns="sort_order")
    history = pd.concat(histories, ignore_index=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "sft_summary.csv", index=False)
    history.to_csv(args.output_dir / "loss_history_all.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for variant in [name for name in ORDER if name not in ("base",)]:
        selected = history[history.variant == variant]
        for axis, split in zip(axes, ("train", "validation")):
            points = selected[selected.split == split].sort_values("step")
            if not points.empty:
                axis.plot(points.step, points.loss, marker="o", markersize=3, label=LABELS[variant])
    for axis, title in zip(axes, ("Training loss", "Validation loss")):
        axis.set_title(title)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel("Cross-entropy loss")
        axis.grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"sft_loss_curves.{extension}", dpi=220)
    plt.close(fig)

    generation_paths = sorted(args.generations_dir.glob("*.jsonl"))
    with (args.output_dir / "qualitative_generations.jsonl").open("w", encoding="utf-8") as output:
        for path in generation_paths:
            content = path.read_text(encoding="utf-8")
            output.write(content)
            if content and not content.endswith("\n"):
                output.write("\n")
    print(f"Wrote SFT tables, loss plots, and generations to {args.output_dir}")


if __name__ == "__main__":
    main()

