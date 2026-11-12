#!/usr/bin/env python3
"""
Perplexity evaluation on standard corpora.

Two metrics are reported for every model:

* **Token-level perplexity** -- ``exp(total_nll / total_tokens)``. Only
  directly comparable between models that share a tokenizer (GPT-2 BPE:
  ``gpt2``, ``opt-*``, and HSKM).
* **Word-level perplexity** -- ``exp(total_nll / n_words)`` where ``n_words``
  is the number of whitespace-separated words in the raw corpus. This is
  **tokenizer independent**, so it is the metric to use when comparing HSKM
  against models with different vocabularies (Pythia, SmolLM2, Qwen, Mamba).

The corpus is scored with non-overlapping windows of ``--context`` tokens.
By default every model is evaluated at the *same* context length so the
comparison isolates the model rather than its context window; pass
``--native-context`` to use each model's own maximum instead.

Examples
--------
    python benchmarks/scripts/30_perplexity.py --models all --context 512
    python benchmarks/scripts/30_perplexity.py --models gpt2,pythia-160m
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "src"))

from data import get_gpt2_encoder, tokenize_corpus  # noqa: E402
from model_registry import BY_KEY, REGISTRY, Baseline, load_model  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results", "perplexity")

CORPORA = {
    "wikitext103": ("wikitext103", "validation"),
    "wikitext2": ("wikitext2", "validation"),
    "tinystories": ("tinystories", "validation"),
}


def count_words(dataset: str, split: str) -> int:
    """Number of whitespace-separated words in the raw corpus."""
    from datasets import load_dataset

    if dataset == "wikitext103":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    elif dataset == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    else:
        ds = load_dataset("roneneldan/TinyStories", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    return len(text.split())


@torch.inference_mode()
def nll_for_ids(model, ids, context: int, kind: str, device, batch_size: int = 4):
    """
    Sum of negative log-likelihood over ``ids[1:]`` given ``ids[:-1]``.

    The corpus is cut into non-overlapping windows of ``context`` tokens.
    """
    total_nll = 0.0
    total_tokens = 0
    n_windows = (len(ids) - 1) // context

    for w in range(n_windows):
        chunk = ids[w * context : (w + 1) * context]
        x = torch.tensor(chunk[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(chunk[1:], dtype=torch.long, device=device).unsqueeze(0)

        if kind == "hskm":
            _, logits = model(x)
        else:
            logits = model(input_ids=x).logits

        lp = F.log_softmax(logits[0].float(), dim=-1)
        tgt = y[0]
        total_nll += -lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
        total_tokens += tgt.numel()

    return total_nll, total_tokens


def evaluate(baseline: Baseline, args, device, corpus_name: str):
    dataset, split = CORPORA[corpus_name]
    ids = tokenize_corpus(corpus_name, split, dataset=dataset)
    ids = ids.astype(np.int64)

    context = args.context
    if context is None:
        context = baseline.max_length

    model = None
    kind = "hf"
    if baseline.loader == "ours_hskm":
        from hskm_lm import load_hskm_checkpoint
        from hskm_stable import patch_hskm_stable

        ckpt = args.checkpoints or os.path.join(
            _ROOT, "benchmarks", "checkpoints", "matched_params", "hskm.pt"
        )
        model = load_hskm_checkpoint(ckpt)
        patch_hskm_stable(model)
        model = model.to(device).eval()
        kind = "hskm"
        context = min(context, model.config.max_seq_len)
    elif baseline.loader in ("legacy_mamba", "legacy_mamba2"):
        model = load_model(baseline, device=device, dtype=torch.float16)
    else:
        from transformers import AutoModelForCausalLM

        dtype = {"float16": torch.float16, "float32": torch.float32}[baseline.dtype]
        model = AutoModelForCausalLM.from_pretrained(baseline.hf_id, torch_dtype=dtype)
        model = model.to(device).eval()

    # Tokenizer: HSKM and the GPT-2-family models share tiktoken gpt2.
    # For every other model we must use the model's own tokenizer, otherwise
    # the token-level numbers are meaningless.
    if kind == "hskm" or baseline.key in ("gpt2", "opt-125m", "opt-350m", "gpt2-ours"):
        text_ids = ids  # already tokenised with GPT-2 BPE above
    else:
        raise RuntimeError(
            f"{baseline.key}: perplexity for non-GPT-2 tokenizers is handled by "
            f"evaluate_native_tokenizer()"
        )

    t0 = time.time()
    nll, ntok = nll_for_ids(model, text_ids, context, kind, device)
    elapsed = time.time() - t0

    del model
    torch.cuda.empty_cache()
    return nll, ntok, context, elapsed


@torch.inference_mode()
def evaluate_native_tokenizer(baseline, args, device, corpus_name):
    """Perplexity for models whose tokenizer is not GPT-2 BPE."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dataset, split = CORPORA[corpus_name]
    tok_id = baseline.tokenizer_id or baseline.hf_id
    tok = AutoTokenizer.from_pretrained(tok_id)
    dtype = {"float16": torch.float16, "float32": torch.float32}[baseline.dtype]
    model = AutoModelForCausalLM.from_pretrained(baseline.hf_id, torch_dtype=dtype)
    model = model.to(device).eval()

    context = args.context or baseline.max_length

    # Re-tokenise the raw corpus with this model's own tokenizer.
    from datasets import load_dataset

    if dataset == "wikitext103":
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split=split)
    elif dataset == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    else:
        ds = load_dataset("roneneldan/TinyStories", split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())

    enc = tok(text, return_tensors="pt").input_ids[0].to(device)
    total_nll, total_tokens = 0.0, 0
    n_win = (enc.numel() - 1) // context
    t0 = time.time()
    for w in range(n_win):
        chunk = enc[w * context : (w + 1) * context]
        x = chunk[:-1].unsqueeze(0)
        y = chunk[1:]
        logits = model(input_ids=x).logits
        lp = F.log_softmax(logits[0].float(), dim=-1)
        total_nll += -lp.gather(-1, y.unsqueeze(-1)).sum().item()
        total_tokens += y.numel()
    elapsed = time.time() - t0

    del model
    torch.cuda.empty_cache()
    return total_nll, total_tokens, context, elapsed


GPT2_TOKENIZER_MODELS = {"gpt2", "opt-125m", "opt-350m", "hskm-ours"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--corpora", default="wikitext103,tinystories")
    ap.add_argument("--context", type=int, default=512,
                    help="context window used for every model (None = native)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoints", default=None)
    args = ap.parse_args()

    if args.models == "all":
        selected = list(REGISTRY)
    else:
        selected = [BY_KEY[k.strip()] for k in args.models.split(",") if k.strip()]

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "perplexity.json")
    all_results = {}
    if os.path.exists(out_path):
        all_results = json.load(open(out_path))

    for corpus in args.corpora.split(","):
        n_words = count_words(CORPORA[corpus][0], CORPORA[corpus][1])
        print(f"\n=== {corpus} ({n_words:,} words) ===", flush=True)
        for b in selected:
            key = f"{b.key}/{corpus}"
            try:
                if b.key in GPT2_TOKENIZER_MODELS:
                    nll, ntok, ctx, el = evaluate(b, args, args.device, corpus)
                else:
                    nll, ntok, ctx, el = evaluate_native_tokenizer(b, args, args.device, corpus)
            except Exception as e:  # noqa: BLE001
                print(f"  {b.key:16s} FAILED: {type(e).__name__}: {e}")
                continue

            tok_ppl = math.exp(nll / ntok)
            word_ppl = math.exp(nll / n_words)
            all_results[key] = {
                "key": b.key, "corpus": corpus, "context": ctx,
                "token_ppl": tok_ppl, "word_ppl": word_ppl,
                "nll": nll, "n_tokens": ntok, "n_words": n_words,
                "elapsed_s": el,
            }
            print(f"  {b.key:16s} ctx={ctx:<5d} tok_ppl={tok_ppl:9.3f}  word_ppl={word_ppl:9.3f}  ({el:.0f}s)",
                  flush=True)
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
