"""Model loading + LoRA adapter application.

Two paths, selected by config:
  - "tiny-random": a small randomly-initialized GPT-2 (no downloads) for the
    fast demo. Exercises the exact same PEFT code path as real models.
  - any HuggingFace model id: AutoModelForCausalLM / AutoTokenizer, with an
    optional 4-bit QLoRA path when bitsandbytes is installed.
"""

from __future__ import annotations

from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (AutoModelForCausalLM, AutoTokenizer, GPT2Config,
                          GPT2LMHeadModel)

from .data import SimpleTokenizer, load_instructions, make_full


def load_base_model(cfg: dict):
    model_name = cfg["model"]["name"]
    if model_name == "tiny-random":
        texts = [make_full(e) for e in load_instructions()]
        tokenizer = SimpleTokenizer(texts)
        gcfg = cfg["model"].get("tiny_config", {})
        config = GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=gcfg.get("n_positions", 128),
            n_embd=gcfg.get("n_embd", 128),
            n_layer=gcfg.get("n_layer", 2),
            n_head=gcfg.get("n_head", 2),
        )
        model = GPT2LMHeadModel(config)
        print(f"  base model: tiny random GPT-2 "
              f"({sum(p.numel() for p in model.parameters()) / 1e6:.2f}M params, "
              f"vocab={len(tokenizer)})")
        return model, tokenizer

    # ---- real HuggingFace model path ----
    use_4bit = cfg["model"].get("load_in_4bit", False)
    if use_4bit:
        try:
            from transformers import BitsAndBytesConfig
            bnb = BitsAndBytesConfig(load_in_4bit=True,
                                     bnb_4bit_quant_type="nf4",
                                     bnb_4bit_compute_dtype="bfloat16")
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("bitsandbytes is required for load_in_4bit=true") from e
        model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=bnb, device_map="auto")
        model = prepare_model_for_kbit_training(model)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"  base model: {model_name} "
          f"({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)")
    return model, tokenizer


def apply_lora(model, cfg: dict):
    lc = cfg["lora"]
    lora_cfg = LoraConfig(
        r=lc.get("r", 8),
        lora_alpha=lc.get("alpha", 16),
        lora_dropout=lc.get("dropout", 0.05),
        target_modules=lc.get("target_modules",
                              ["c_attn", "c_proj", "c_fc"]),
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_cfg)
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA adapter: r={lc.get('r', 8)} alpha={lc.get('alpha', 16)} "
          f"trainable={trainable / 1e6:.3f}M / {total / 1e6:.2f}M "
          f"({100 * trainable / total:.2f}% of params)")
    return peft_model
