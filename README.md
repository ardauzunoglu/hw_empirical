# CS 601.768 homework empirical experiments

This folder implements the two empirical questions in the homework:

1. single-GPU FP8 vLLM serving for the two dense OLMo checkpoints and OLMoE;
2. Qwen2.5-0.5B supervised fine-tuning with LoRA ranks 1, 4, and 16 and with full weights.

The scripts intentionally do not contain precomputed or synthetic homework results. Running them on a GPU creates raw measurements, figures, metadata, checkpoints, and qualitative generations under `artifacts/`.

## Environment

Use a fresh environment on a GPU node. Install the PyTorch build matching that node's CUDA runtime, then install the remaining dependencies:

```bash
cd /home/jhu/auzunog1/dkhasha1-project/auzunog1/hw_empirical
python -m pip install -r requirements.txt
```

Set `HF_HOME` to a filesystem with enough capacity before running. The serving experiment dynamically quantizes BF16 checkpoints to FP8 during model loading, so its first load is slow and can temporarily use more host memory. An Ada/Hopper-or-newer GPU (CUDA compute capability at least 8.9) is expected for vLLM FP8 weight serving. All commands use one visible GPU; select it through the scheduler or `CUDA_VISIBLE_DEVICES`.

## 1. Dense versus MoE serving

The implementation pointer for the homework is [`serving/benchmark_vllm.py`](serving/benchmark_vllm.py). It uses:

- `quantization="fp8"` for FP8 weights;
- `kv_cache_dtype="fp8"` and runtime KV-scale calculation for the FP8 KV cache;
- tensor parallelism 1 and prefix caching disabled for all checkpoints;
- exact token-ID inputs, keeping tokenization outside the timed region;
- one output token for prefill trials and 512 forced output tokens for decode trials;
- one warmup followed by three recorded repetitions at every point.

Run all checkpoints sequentially on the same allocated GPU:

```bash
bash serving/run_all.sh
```

Optional benchmark settings can be passed through the launcher, for example:

```bash
bash serving/run_all.sh \
  --prefill-lengths 128,512,1024,2048,3072 \
  --decode-parallelism 1,2,4,8,16,32 \
  --repetitions 5
```

Each checkpoint is launched in a separate Python process so its GPU allocations are released before the next model. The generated deliverables are:

- `artifacts/serving/raw/*.csv`: one row per timed repetition;
- `artifacts/serving/raw/*.metadata.json`: hardware, software, command, and FP8 settings;
- `artifacts/serving/figures/serving_summary.csv`: means and standard deviations;
- `artifacts/serving/figures/prefill_throughput.{png,pdf}`;
- `artifacts/serving/figures/decode_throughput.{png,pdf}`.

Prefill throughput is total input tokens divided by elapsed generation-call time. Decode throughput is total emitted output tokens divided by elapsed time. Error bars show one standard deviation over repetitions. Do not compare runs taken under different GPU loads; use one GPU type and otherwise identical settings.

Interpret the measured curves rather than presupposing the result. The useful comparison is whether OLMoE follows the small dense model when computation is governed by its roughly 1B active parameters, and whether it follows the 7B dense model when memory traffic/capacity is governed by its roughly 7B stored parameters. Look for changes with input length and concurrency, and quote the actual crossover points (if any) from `serving_summary.csv`.

## 2. SFT with LoRA and full weights

The implementation pointer for the homework is [`sft/train_sft.py`](sft/train_sft.py). The default controlled setup is:

- base checkpoint `Qwen/Qwen2.5-0.5B`;
- the first 1,000 `train_sft` conversations for training;
- the first 100 `test_sft` conversations for validation;
- one epoch, sequence length 512, effective batch size 16, learning rate `5e-5`;
- BF16, gradient checkpointing, constant learning rate, and the same seed;
- PEFT `target_modules="all-linear"` for every LoRA run.

Because the requested checkpoint is a base model, training and generation both set the same explicit Qwen-style chat template from `sft/chat_format.py`; this does not depend on a remotely supplied tokenizer template.

Run the full workflow:

```bash
bash sft/run_all.sh
```

The launcher narrows `CUDA_VISIBLE_DEVICES` to one GPU so training time and peak memory remain comparable. To choose a particular assigned GPU, use `SFT_CUDA_VISIBLE_DEVICES`, for example `SFT_CUDA_VISIBLE_DEVICES=2 bash sft/run_all.sh`.

This first saves the exact shared dataset subsets, trains LoRA ranks 1/4/16 and full weights in separate processes, produces deterministic greedy responses to the same five held-out prompts, and builds the comparison artifacts:

- `artifacts/sft/data/manifest.json`: subset identity and dataset fingerprints;
- `artifacts/sft/runs/*/metrics.json`: trainable count, GPU peak memory, runtime, and final metrics;
- `artifacts/sft/runs/*/loss_history.csv`: raw stepwise training/validation losses;
- `artifacts/sft/runs/*/model/`: adapter or full model used for generation;
- `artifacts/sft/generations/*.jsonl`: raw matched generations;
- `artifacts/sft/report/sft_summary.csv`;
- `artifacts/sft/report/loss_history_all.csv` and `sft_loss_curves.{png,pdf}`;
- `artifacts/sft/report/qualitative_generations.jsonl`.

The measured training runtime is Transformers' `train_runtime`; a second wall-clock field includes the final held-out evaluation. CUDA peak memory includes the already-loaded model and all allocations made during training. Both allocated and reserved peaks are retained; report which definition you use (reserved memory is closest to process GPU footprint). The loss trains on the complete rendered conversation, including user and assistant tokens, consistently for all variants.

The full run saves roughly a model-sized checkpoint and every run tokenizes its local fixed subset. If a scheduled allocation is interrupted, rerun the launcher: variants containing a completed `metrics.json` are skipped, while an incomplete variant is restarted with the same configuration. You can also invoke the four `train_sft.py` commands separately, then run the generation and summary commands shown in [`sft/run_all.sh`](sft/run_all.sh).

## Writing the homework response

After the jobs finish, [`empirical_answers.tex`](empirical_answers.tex) provides a submission-ready write-up populated from the generated evidence. [`REPORT_TEMPLATE.md`](REPORT_TEMPLATE.md) remains available as a shorter checklist. Include the two script pointers above and disclose AI assistance as required by the assignment instructions.
