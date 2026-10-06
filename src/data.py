"""Instruction dataset + a tiny word-level tokenizer.

The demo ships a toy instruction dataset (no downloads) so `python demo.py`
runs anywhere in under two minutes. Point the config at any HuggingFace
model id + tokenizer to fine-tune a real model with the same code path.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch

DATA_PATH = Path(__file__).parent.parent / "data" / "instructions.jsonl"


def load_instructions(path: str | Path = DATA_PATH) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def make_prompt(example: dict) -> str:
    return f"Instruction: {example['instruction']}\nResponse:"


def make_full(example: dict) -> str:
    return f"Instruction: {example['instruction']}\nResponse: {example['response']}"


class SimpleTokenizer:
    """Minimal word-level tokenizer with an HF-style call interface.

    Used for the zero-download demo. For real runs, swap in
    `AutoTokenizer.from_pretrained(...)` — the training loop only relies on
    `encode`, `decode`, `pad_token_id`, and the `__call__` batch interface.
    """

    def __init__(self, texts: list[str], max_vocab: int = 4000):
        self.pad_token, self.bos_token, self.eos_token, self.unk_token = (
            "<pad>", "<bos>", "<eos>", "<unk>")
        counts: dict[str, int] = {}
        for t in texts:
            for w in re.findall(r"\S+", t.lower()):
                counts[w] = counts.get(w, 0) + 1
        vocab = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        vocab += [w for w, _ in sorted(counts.items(), key=lambda kv: -kv[1])
                  ][:max_vocab - len(vocab)]
        self.stoi = {w: i for i, w in enumerate(vocab)}
        self.itos = vocab
        self.pad_token_id = 0
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, text: str) -> list[int]:
        return [self.stoi.get(w, self.unk_token_id)
                for w in re.findall(r"\S+", text.lower())]

    def decode(self, ids: list[int]) -> str:
        words = []
        for i in ids:
            if i in (self.pad_token_id, self.bos_token_id):
                continue
            if i == self.eos_token_id:
                break
            words.append(self.itos[i] if i < len(self.itos) else self.unk_token)
        return " ".join(words)

    def __call__(self, texts, padding=True, truncation=True,
                 max_length: int = 64, return_tensors: str = "pt"):
        if isinstance(texts, str):
            texts = [texts]
        ids = [self.encode(t)[:max_length] for t in texts]
        if padding:
            ml = max(len(x) for x in ids)
            attn = [[1] * len(x) + [0] * (ml - len(x)) for x in ids]
            ids = [x + [self.pad_token_id] * (ml - len(x)) for x in ids]
        else:
            attn = [[1] * len(x) for x in ids]
        out = {"input_ids": torch.tensor(ids, dtype=torch.long),
               "attention_mask": torch.tensor(attn, dtype=torch.long)}
        return out
