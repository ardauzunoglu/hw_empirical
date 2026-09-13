#!/usr/bin/env python3
"""Generate deterministic qualitative samples from a base, LoRA, or full model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from chat_format import CHAT_TEMPLATE


BASE_MODEL = "Qwen/Qwen2.5-0.5B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--kind", choices=("base", "lora", "full"), required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.kind != "base" and args.model_path is None:
        raise SystemExit("--model-path is required for LoRA and full variants")
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit("Install hw_empirical/requirements.txt in the GPU environment") from exc
    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU is required for generation")

    tokenizer_source = str(args.model_path) if args.kind == "full" else BASE_MODEL
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    tokenizer.chat_template = CHAT_TEMPLATE
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if args.kind == "full":
        model = AutoModelForCausalLM.from_pretrained(
            str(args.model_path), dtype=torch.bfloat16, device_map="cuda"
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map="cuda"
        )
        if args.kind == "lora":
            try:
                from peft import PeftModel
            except ImportError as exc:
                raise SystemExit("PEFT is required to load LoRA adapters") from exc
            model = PeftModel.from_pretrained(model, str(args.model_path))
    model.eval()

    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))
    rows = []
    for item in prompts:
        rendered = tokenizer.apply_chat_template(
            item["messages"], tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(rendered, return_tensors="pt", add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        completion_ids = generated[0, encoded["input_ids"].shape[1] :]
        rows.append(
            {
                "variant": args.variant,
                "prompt_id": item["id"],
                "messages": item["messages"],
                "completion": tokenizer.decode(completion_ids, skip_special_tokens=True),
                "generated_tokens": int(completion_ids.numel()),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} {args.variant} generations to {args.output}")


if __name__ == "__main__":
    main()
