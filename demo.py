"""End-to-end demo: baseline generations -> LoRA fine-tune -> after generations.

Runs on a tiny randomly-initialized model (no downloads) in under ~2 minutes.
For a real model, edit configs/lora_sft.yaml (model.name = HF id).

Usage:
    python demo.py [--config configs/lora_sft.yaml]
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
import yaml

from src.model import load_base_model, apply_lora
from src.train import train_loop
from src.compare import compare, print_comparison

ADAPTER_OUT = Path("data/lora_adapter")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/lora_sft.yaml")
    args = ap.parse_args()

    torch.manual_seed(42)
    cfg = yaml.safe_load(open(args.config))
    print("lora-finetune-lab — parameter-efficient fine-tuning demo")
    print(f"config: {args.config}")

    print("\n[1/4] Loading base model...")
    base_model, tokenizer = load_base_model(cfg)

    print("\n[2/4] Applying LoRA adapter...")
    base_for_baseline = copy.deepcopy(base_model)  # peft wraps in place; keep a clean baseline
    tuned_model = apply_lora(base_model, cfg)

    print("\n[3/4] Fine-tuning on instruction data...")
    history = train_loop(tuned_model, tokenizer, cfg)
    print(f"  loss trajectory: "
          f"{history['train_loss'][0]:.4f} -> {history['train_loss'][-1]:.4f} "
          f"(train), {history['val_loss'][0]:.4f} -> {history['val_loss'][-1]:.4f} (val)")

    print("\n[4/4] Before/after generation comparison...")
    result = compare(base_for_baseline, tuned_model, tokenizer,
                     n_prompts=cfg["eval"].get("n_prompts", 5))
    print_comparison(result)

    ADAPTER_OUT.mkdir(parents=True, exist_ok=True)
    tuned_model.save_pretrained(str(ADAPTER_OUT))
    print(f"\nLoRA adapter saved to {ADAPTER_OUT}/")


if __name__ == "__main__":
    main()
