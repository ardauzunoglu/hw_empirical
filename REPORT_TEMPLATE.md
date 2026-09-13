# Empirical results template

Replace every bracketed field with values or observations from the generated artifacts. Do not submit this template with placeholders.

## Dense versus MoE serving

Hardware/software: `[GPU from metadata]`, vLLM `[version]`, PyTorch `[version]`. All models used dynamically quantized FP8 weights, an FP8 KV cache, tensor parallelism 1, and identical benchmark settings. The code is `hw_empirical/serving/benchmark_vllm.py`.

Embed:

- `artifacts/serving/figures/prefill_throughput.pdf`
- `artifacts/serving/figures/decode_throughput.pdf`

Raw measurements are in `artifacts/serving/raw/*.csv`; aggregate values are in `artifacts/serving/figures/serving_summary.csv`.

Brief interpretation: `[Describe the ordering and gaps in prefill throughput as context length changes.]` `[Describe the ordering and any crossover in decode throughput as concurrency increases.]` `[State where OLMoE resembles the 1B dense model, consistent with active compute, and where it resembles the 7B dense model, consistent with total parameter/memory traffic. Only make claims supported by the curves.]`

## Supervised fine-tuning

Hardware/software: `[GPU and package versions from metrics.json]`. All variants used the fixed data fingerprints in `artifacts/sft/data/manifest.json` and the common settings documented in the README. The code is `hw_empirical/sft/train_sft.py`.

Copy the requested comparison columns from `artifacts/sft/report/sft_summary.csv`:

| Variant | Trainable parameters | Peak reserved GPU GiB | Training seconds | Final train loss | Final validation loss |
|---|---:|---:|---:|---:|---:|
| LoRA r=1 | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| LoRA r=4 | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| LoRA r=16 | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |
| Full | `[ ]` | `[ ]` | `[ ]` | `[ ]` | `[ ]` |

Embed `artifacts/sft/report/sft_loss_curves.pdf`. Briefly discuss `[loss behavior]`, `[rank/memory/time scaling]`, and `[whether validation loss distinguishes the variants]`.

Select two or three prompt IDs from `artifacts/sft/report/qualitative_generations.jsonl`. Compare the base response with all fine-tuned variants on the same prompt, focusing on instruction following and conversational structure. Mention failures as well as improvements; greedy decoding makes the comparison deterministic but does not estimate average response quality.

