#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RAW_DIR="${PROJECT_DIR}/artifacts/serving/raw"
FIGURE_DIR="${PROJECT_DIR}/artifacts/serving/figures"

mkdir -p "${RAW_DIR}" "${FIGURE_DIR}"

models=(
  "allenai/OLMo-2-0425-1B-Instruct"
  "allenai/OLMoE-1B-7B-0924-Instruct"
  "allenai/OLMo-2-1124-7B"
)
names=("olmo2_1b_dense" "olmoe_1b_7b" "olmo2_7b_dense")

for index in "${!models[@]}"; do
  "${PYTHON_BIN}" "${SCRIPT_DIR}/benchmark_vllm.py" \
    --model "${models[$index]}" \
    --output "${RAW_DIR}/${names[$index]}.csv" \
    "$@"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/plot_results.py" \
  --input-dir "${RAW_DIR}" \
  --output-dir "${FIGURE_DIR}"

