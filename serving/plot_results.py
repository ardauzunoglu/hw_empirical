#!/usr/bin/env python3
"""Aggregate raw serving CSV files and create the two homework figures."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


MODEL_ORDER = [
    "allenai/OLMo-2-0425-1B-Instruct",
    "allenai/OLMoE-1B-7B-0924-Instruct",
    "allenai/OLMo-2-1124-7B",
]
MODEL_LABELS = {
    MODEL_ORDER[0]: "OLMo-2 1B dense",
    MODEL_ORDER[1]: "OLMoE 1B-active/7B-total",
    MODEL_ORDER[2]: "OLMo-2 7B dense",
}
COLORS = {MODEL_ORDER[0]: "#0072B2", MODEL_ORDER[1]: "#D55E00", MODEL_ORDER[2]: "#009E73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def plot_workload(summary: pd.DataFrame, workload: str, output_dir: Path) -> None:
    subset = summary[summary.workload == workload]
    x_column = "input_length" if workload == "prefill" else "parallel_generations"
    x_label = "Input context length (tokens)" if workload == "prefill" else "Parallel generations"
    y_label = "Prefill throughput (input tokens/s)" if workload == "prefill" else "Decode throughput (output tokens/s)"

    fig, axis = plt.subplots(figsize=(7.2, 4.5))
    for model in MODEL_ORDER:
        model_rows = subset[subset.model == model].sort_values(x_column)
        if model_rows.empty:
            continue
        axis.errorbar(
            model_rows[x_column],
            model_rows["mean_tokens_per_second"],
            yerr=model_rows["std_tokens_per_second"].fillna(0),
            marker="o",
            linewidth=2,
            capsize=3,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
    axis.set_xlabel(x_label)
    axis.set_ylabel(y_label)
    axis.grid(alpha=0.25)
    axis.legend(frameon=False)
    fig.tight_layout()
    for extension in ("png", "pdf"):
        fig.savefig(output_dir / f"{workload}_throughput.{extension}", dpi=220)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    paths = sorted(args.input_dir.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No raw CSV files found in {args.input_dir}")
    raw = pd.concat((pd.read_csv(path) for path in paths), ignore_index=True)
    required = {"model", "workload", "input_length", "parallel_generations", "tokens_per_second"}
    missing = required - set(raw.columns)
    if missing:
        raise SystemExit(f"Raw data is missing columns: {sorted(missing)}")
    unknown = set(raw.model.unique()) - set(MODEL_ORDER)
    if unknown:
        raise SystemExit(f"Unexpected model names: {sorted(unknown)}")

    group_columns = ["model", "architecture", "role", "workload", "input_length", "parallel_generations"]
    summary = (
        raw.groupby(group_columns, as_index=False)
        .agg(
            mean_tokens_per_second=("tokens_per_second", "mean"),
            std_tokens_per_second=("tokens_per_second", "std"),
            repetitions=("tokens_per_second", "size"),
            mean_elapsed_seconds=("elapsed_seconds", "mean"),
        )
        .sort_values(["workload", "model", "input_length", "parallel_generations"])
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / "serving_summary.csv", index=False)
    plot_workload(summary, "prefill", args.output_dir)
    plot_workload(summary, "decode", args.output_dir)
    print(f"Wrote aggregate table and figures to {args.output_dir}")


if __name__ == "__main__":
    main()

