# lora-finetune-lab — Parameter-Efficient LLM Fine-Tuning Toolkit

A config-driven **LoRA/QLoRA fine-tuning toolkit** built on HuggingFace
`transformers` + `peft`: load a base model, attach a LoRA adapter, run
supervised fine-tuning with train/validation loss tracking, and evaluate with
a **before/after generation comparison**.

The demo runs on a tiny randomly-initialized GPT-2 (no downloads, ~30s on
CPU) through the *exact same PEFT code path* as real models — point the YAML
config at any HuggingFace causal-LM id to fine-tune the real thing, with an
optional 4-bit QLoRA path via bitsandbytes.

## Architecture

```
configs/lora_sft.yaml ──► model / lora / training / eval settings
        │
        ▼
load_base_model() ──► "tiny-random" GPT-2 (demo) ──► real HF id + optional
        │               + word-level tokenizer          4-bit QLoRA (config)
        ▼
apply_lora() ──► peft.get_peft_model (r, alpha, dropout, target_modules)
        │         e.g. 7.2% of params trainable on the demo model
        ▼
train_loop() ──► causal-LM SFT, loss on RESPONSE tokens only (prompt masked),
        │         per-epoch train/val loss
        ▼
compare() ──► before/after generations + format-adherence rate
```

## Quickstart

```bash
pip install -r requirements.txt

# Fast demo: tiny random model, no downloads (~30s CPU)
python demo.py

# Real model: edit configs/lora_sft.yaml -> model.name: meta-llama/Llama-3.2-1B
# (for QLoRA also set load_in_4bit: true; needs bitsandbytes + CUDA)
python demo.py --config configs/lora_sft.yaml
```

## Sample output

```
[1/4] Loading base model...
  base model: tiny random GPT-2 (0.42M params, vocab=83)

[2/4] Applying LoRA adapter...
  LoRA adapter: r=8 alpha=16 trainable=0.033M / 0.46M (7.18% of params)

[3/4] Fine-tuning on instruction data...
  epoch 1/3  train_loss=4.0349  val_loss=3.9326
  epoch 2/3  train_loss=3.6214  val_loss=3.5132
  epoch 3/3  train_loss=3.3139  val_loss=3.2476

[4/4] Before/after generation comparison...
  prompt: Give the sentiment in one word: Weak script
  before: 'response: response: response: ...'
  after:  'answer: answer: answer: ...'

  'Answer:' format rate — before: 0%, after: 100%
LoRA adapter saved to data/lora_adapter/
```

## How it works

- **Data** (`src/data.py`) — 32 instruction/response pairs in
  `data/instructions.jsonl`; every response follows the `Answer: <word>`
  format so the eval can measure format learning. `SimpleTokenizer` is a
  zero-download word-level tokenizer with an HF-style batch interface — swap
  in `AutoTokenizer` for real runs.
- **Model** (`src/model.py`) — `load_base_model` handles both the tiny-random
  demo model and real HF ids (with `prepare_model_for_kbit_training` for the
  QLoRA path); `apply_lora` wraps with `peft.get_peft_model` and reports the
  trainable-parameter share.
- **Training** (`src/train.py`) — plain PyTorch SFT loop; prompt tokens are
  masked (`-100`) so loss is computed on responses only.
- **Eval** (`src/compare.py`) — greedy generations before/after on held-out
  prompts, side-by-side samples, and the share of generations adopting the
  learned `Answer:` format.

## Honest scope

The demo model is intentionally tiny and randomly initialized: it learns the
response *format*, not fluent language. The value is the real PEFT machinery —
adapter config, parameter accounting, masked SFT loss, before/after eval —
which transfers unchanged to real base models.

## Tech stack

Python, PyTorch, HuggingFace transformers, PEFT (LoRA), PyYAML.
Optional: bitsandbytes (4-bit QLoRA), CUDA.
