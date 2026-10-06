"""Causal-LM fine-tuning loop with train/validation loss tracking.

Loss is computed on response tokens only (prompt tokens masked with -100),
the standard instruction-tuning setup.
"""

from __future__ import annotations

import random

import torch

from .data import load_instructions, make_prompt, make_full


def build_examples(tokenizer, max_length: int = 64, val_frac: float = 0.2,
                   seed: int = 42):
    rows = load_instructions()
    rng = random.Random(seed)
    rng.shuffle(rows)
    n_val = max(2, int(len(rows) * val_frac))
    return rows[n_val:], rows[:n_val]


def encode_example(tokenizer, example: dict, max_length: int = 64):
    prompt = make_prompt(example)
    full = make_full(example)
    full_ids = tokenizer.encode(full)[:max_length]
    prompt_len = len(tokenizer.encode(prompt)[:max_length])
    input_ids = full_ids + [tokenizer.eos_token_id]
    labels = ([-100] * min(prompt_len, len(input_ids))
              + input_ids[min(prompt_len, len(input_ids)):])
    return (torch.tensor(input_ids, dtype=torch.long),
            torch.tensor(labels, dtype=torch.long))


def train_loop(model, tokenizer, cfg: dict):
    tcfg = cfg["training"]
    train_rows, val_rows = build_examples(tokenizer, seed=cfg.get("seed", 42))
    train_items = [encode_example(tokenizer, r) for r in train_rows]
    val_items = [encode_example(tokenizer, r) for r in val_rows]

    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=float(tcfg.get("lr", 5e-4)))
    batch_size = int(tcfg.get("batch_size", 8))
    epochs = int(tcfg.get("epochs", 3))
    history = {"train_loss": [], "val_loss": []}
    model.train()
    step = 0
    for epoch in range(epochs):
        random.Random(cfg.get("seed", 42) + epoch).shuffle(train_items)
        for i in range(0, len(train_items), batch_size):
            batch = train_items[i:i + batch_size]
            input_ids = torch.nn.utils.rnn.pad_sequence(
                [b[0] for b in batch], batch_first=True,
                padding_value=tokenizer.pad_token_id)
            labels = torch.nn.utils.rnn.pad_sequence(
                [b[1] for b in batch], batch_first=True, padding_value=-100)
            out = model(input_ids=input_ids, labels=labels)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
            step += 1
        tr_loss = evaluate_loss(model, tokenizer, train_items[:16])
        va_loss = evaluate_loss(model, tokenizer, val_items)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(va_loss)
        print(f"  epoch {epoch + 1}/{epochs}  "
              f"train_loss={tr_loss:.4f}  val_loss={va_loss:.4f}")
    return history


@torch.no_grad()
def evaluate_loss(model, tokenizer, items) -> float:
    model.eval()
    losses = []
    for input_ids, labels in items:
        out = model(input_ids=input_ids.unsqueeze(0),
                    labels=labels.unsqueeze(0))
        losses.append(out.loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)
