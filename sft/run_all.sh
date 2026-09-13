#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_DIR="${PROJECT_DIR}/artifacts/sft/data"
RUNS_DIR="${PROJECT_DIR}/artifacts/sft/runs"
GEN_DIR="${PROJECT_DIR}/artifacts/sft/generations"
REPORT_DIR="${PROJECT_DIR}/artifacts/sft/report"

# Keep the runtime and peak-memory comparison genuinely single-GPU. Respect an
# explicit override; otherwise retain the first scheduler-assigned visible GPU.
if [[ -n "${SFT_CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES}"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
else
  export CUDA_VISIBLE_DEVICES=0
fi
echo "SFT using CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

mkdir -p "${RUNS_DIR}" "${GEN_DIR}" "${REPORT_DIR}"

if [[ ! -f "${DATA_DIR}/manifest.json" ]]; then
  "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py" --output-dir "${DATA_DIR}"
fi

for rank in 1 4 16; do
  if [[ -f "${RUNS_DIR}/lora_r${rank}/metrics.json" ]]; then
    echo "Skipping completed run lora_r${rank}"
  else
    "${PYTHON_BIN}" "${SCRIPT_DIR}/train_sft.py" \
      --mode lora \
      --rank "${rank}" \
      --data-dir "${DATA_DIR}" \
      --output-dir "${RUNS_DIR}/lora_r${rank}"
  fi
done

if [[ -f "${RUNS_DIR}/full/metrics.json" ]]; then
  echo "Skipping completed run full"
else
  "${PYTHON_BIN}" "${SCRIPT_DIR}/train_sft.py" \
    --mode full \
    --data-dir "${DATA_DIR}" \
    --output-dir "${RUNS_DIR}/full"
fi

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_samples.py" \
  --variant base \
  --kind base \
  --prompts "${DATA_DIR}/heldout_prompts.json" \
  --output "${GEN_DIR}/base.jsonl"

for rank in 1 4 16; do
  "${PYTHON_BIN}" "${SCRIPT_DIR}/generate_samples.py" \
    --variant "lora_r${rank}" \
    --kind lora \
    --model-path "${RUNS_DIR}/lora_r${rank}/model" \
    --prompts "${DATA_DIR}/heldout_prompts.json" \
    --output "${GEN_DIR}/lora_r${rank}.jsonl"
done

"${PYTHON_BIN}" "${SCRIPT_DIR}/generate_samples.py" \
  --variant full \
  --kind full \
  --model-path "${RUNS_DIR}/full/model" \
  --prompts "${DATA_DIR}/heldout_prompts.json" \
  --output "${GEN_DIR}/full.jsonl"

"${PYTHON_BIN}" "${SCRIPT_DIR}/summarize_results.py" \
  --runs-dir "${RUNS_DIR}" \
  --generations-dir "${GEN_DIR}" \
  --output-dir "${REPORT_DIR}"
