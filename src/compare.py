"""Before/after generation comparison: the fine-tuning eval.

Generates responses to held-out prompts with the base model and with the
LoRA-tuned model, then reports (a) side-by-side samples and (b) how often
each model produces the learned response format. A successful LoRA run shows
the tuned model adopting the instruction-following format the base model
lacks.
"""

from __future__ import annotations

import torch

from .data import load_instructions, make_prompt


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int = 24) -> str:
    model.eval()
    batch = tokenizer(make_prompt({"instruction": prompt, "response": ""}),
                      max_length=64)
    input_ids = batch["input_ids"]
    out = model.generate(input_ids, max_new_tokens=max_new_tokens,
                         do_sample=False,
                         pad_token_id=tokenizer.pad_token_id,
                         eos_token_id=tokenizer.eos_token_id)
    gen_ids = out[0][input_ids.shape[1]:].tolist()
    model.train()
    return tokenizer.decode(gen_ids)


def format_adherence(generations: list[str]) -> float:
    """Share of generations that follow the learned 'Answer:' format."""
    if not generations:
        return 0.0
    return sum(1 for g in generations if g.strip().lower().startswith("answer:")) / len(generations)


def compare(base_model, tuned_model, tokenizer, n_prompts: int = 5) -> dict:
    rows = load_instructions()
    prompts = [r["instruction"] for r in rows[-n_prompts:]]
    base_gens, tuned_gens = [], []
    for p in prompts:
        base_gens.append(generate(base_model, tokenizer, p))
        tuned_gens.append(generate(tuned_model, tokenizer, p))
    return {
        "prompts": prompts,
        "base": base_gens,
        "tuned": tuned_gens,
        "base_format_rate": format_adherence(base_gens),
        "tuned_format_rate": format_adherence(tuned_gens),
    }


def print_comparison(result: dict) -> None:
    print("\n=== Before / After generation comparison ===")
    for p, b, t in zip(result["prompts"], result["base"], result["tuned"]):
        print(f"\n  prompt: {p}")
        print(f"  before: {b!r}")
        print(f"  after:  {t!r}")
    print(f"\n  'Answer:' format rate — before: {result['base_format_rate']:.0%}, "
          f"after: {result['tuned_format_rate']:.0%}")
