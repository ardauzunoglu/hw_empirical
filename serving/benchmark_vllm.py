#!/usr/bin/env python3
"""Benchmark prefill- and decode-dominated vLLM workloads on one GPU.

Every timed sample is written as one raw CSV row. Inputs are supplied as token
IDs so host-side tokenization is deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import json
import os
import platform
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "allenai/OLMo-2-0425-1B-Instruct": ("dense", "small dense (~1B)"),
    "allenai/OLMoE-1B-7B-0924-Instruct": ("moe", "~1B active / 7B total"),
    "allenai/OLMo-2-1124-7B": ("dense", "large dense (~7B)"),
}

FIELDS = [
    "timestamp_utc",
    "model",
    "architecture",
    "role",
    "workload",
    "repetition",
    "input_length",
    "parallel_generations",
    "requested_output_length",
    "prompt_tokens",
    "output_tokens",
    "elapsed_seconds",
    "tokens_per_second",
]


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected a comma-separated list of positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=MODELS, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prefill-lengths", type=parse_int_list, default=parse_int_list("128,512,2048,3072"))
    parser.add_argument("--prefill-batch-size", type=int, default=1)
    parser.add_argument("--decode-parallelism", type=parse_int_list, default=parse_int_list("1,2,4,8,16,32"))
    parser.add_argument("--decode-prompt-length", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--seed", type=int, default=601768)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--allow-unsupported-fp8-gpu",
        action="store_true",
        help="Try FP8 even when the detected CUDA compute capability is below 8.9.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "prefill_batch_size": args.prefill_batch_size,
        "decode_prompt_length": args.decode_prompt_length,
        "decode_tokens": args.decode_tokens,
        "repetitions": args.repetitions,
    }
    for name, value in positive.items():
        if value <= 0:
            raise SystemExit(f"--{name.replace('_', '-')} must be positive")
    if args.warmups < 0:
        raise SystemExit("--warmups cannot be negative")
    if not 0 < args.gpu_memory_utilization <= 1:
        raise SystemExit("--gpu-memory-utilization must be in (0, 1]")


def require_runtime(allow_unsupported: bool):
    try:
        import torch
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Missing benchmark dependencies. Install hw_empirical/requirements.txt "
            "inside the GPU environment. Original error: " + str(exc)
        ) from exc
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required; torch.cuda.is_available() is false.")
    capability = torch.cuda.get_device_capability(0)
    if capability < (8, 9) and not allow_unsupported:
        raise SystemExit(
            f"Detected CUDA compute capability {capability[0]}.{capability[1]}. "
            "vLLM FP8 weight serving normally requires capability >= 8.9; use a "
            "supported GPU or pass --allow-unsupported-fp8-gpu to let vLLM decide."
        )
    return torch, AutoTokenizer, LLM, SamplingParams


def exact_token_ids(tokenizer, length: int, offset: int = 0) -> list[int]:
    source = tokenizer.encode(
        "The quick brown fox studies efficient language model inference. ",
        add_special_tokens=False,
    )
    if not source:
        raise RuntimeError("Tokenizer produced no IDs for benchmark seed text")
    rotated = source[offset % len(source) :] + source[: offset % len(source)]
    return (rotated * ((length + len(rotated) - 1) // len(rotated)))[:length]


def synchronize(torch) -> None:
    torch.cuda.synchronize()


def timed_generate(torch, llm, prompts, sampling_params):
    synchronize(torch)
    started = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    synchronize(torch)
    elapsed = time.perf_counter() - started
    output_tokens = sum(len(candidate.token_ids) for request in outputs for candidate in request.outputs)
    return elapsed, output_tokens


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    args = parse_args()
    validate_args(args)
    torch, AutoTokenizer, LLM, SamplingParams = require_runtime(args.allow_unsupported_fp8_gpu)

    maximum_length = max(
        max(args.prefill_lengths) + 1,
        args.decode_prompt_length + args.decode_tokens,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
    llm = LLM(
        model=args.model,
        tokenizer=args.model,
        dtype="bfloat16",
        quantization="fp8",
        kv_cache_dtype="fp8",
        calculate_kv_scales=True,
        tensor_parallel_size=1,
        max_model_len=maximum_length,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        trust_remote_code=args.trust_remote_code,
        seed=args.seed,
    )
    prefill_sampling = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
    decode_sampling = SamplingParams(
        temperature=0.0,
        max_tokens=args.decode_tokens,
        min_tokens=args.decode_tokens,
        ignore_eos=True,
    )
    architecture, role = MODELS[args.model]
    rows: list[dict[str, object]] = []

    for length in args.prefill_lengths:
        prompts = [
            {"prompt_token_ids": exact_token_ids(tokenizer, length, offset=index)}
            for index in range(args.prefill_batch_size)
        ]
        for _ in range(args.warmups):
            timed_generate(torch, llm, prompts, prefill_sampling)
        for repetition in range(args.repetitions):
            elapsed, output_tokens = timed_generate(torch, llm, prompts, prefill_sampling)
            prompt_tokens = length * args.prefill_batch_size
            rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": args.model,
                    "architecture": architecture,
                    "role": role,
                    "workload": "prefill",
                    "repetition": repetition,
                    "input_length": length,
                    "parallel_generations": args.prefill_batch_size,
                    "requested_output_length": 1,
                    "prompt_tokens": prompt_tokens,
                    "output_tokens": output_tokens,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": prompt_tokens / elapsed,
                }
            )

    for parallelism in args.decode_parallelism:
        prompts = [
            {"prompt_token_ids": exact_token_ids(tokenizer, args.decode_prompt_length, offset=index)}
            for index in range(parallelism)
        ]
        for _ in range(args.warmups):
            timed_generate(torch, llm, prompts, decode_sampling)
        for repetition in range(args.repetitions):
            elapsed, output_tokens = timed_generate(torch, llm, prompts, decode_sampling)
            rows.append(
                {
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "model": args.model,
                    "architecture": architecture,
                    "role": role,
                    "workload": "decode",
                    "repetition": repetition,
                    "input_length": args.decode_prompt_length,
                    "parallel_generations": parallelism,
                    "requested_output_length": args.decode_tokens,
                    "prompt_tokens": args.decode_prompt_length * parallelism,
                    "output_tokens": output_tokens,
                    "elapsed_seconds": elapsed,
                    "tokens_per_second": output_tokens / elapsed,
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "command": sys.argv,
        "model": args.model,
        "architecture": architecture,
        "role": role,
        "fp8_weights": True,
        "fp8_kv_cache": True,
        "prefix_caching": False,
        "tokenization_in_timed_region": False,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "gpu": torch.cuda.get_device_name(0),
        "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "versions": {name: package_version(name) for name in ("torch", "vllm", "transformers")},
        "environment": {key: os.environ.get(key) for key in ("CUDA_VISIBLE_DEVICES", "SLURM_JOB_ID")},
        "settings": vars(args) | {"output": str(args.output)},
    }
    metadata_path = args.output.with_suffix(".metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(rows)} measurements to {args.output}")
    print(f"Wrote run metadata to {metadata_path}")


if __name__ == "__main__":
    main()

