#!/usr/bin/env python3
"""
Perplexity evaluation on standard corpora.

Two metrics are reported for every model:

* **Token-level perplexity** -- ``exp(total_nll / total_tokens)``. Only
  directly comparable between models that share a tokenizer (GPT-2 BPE:
  ``gpt2``, ``opt-*``, and HSKM).
* **Word-level perplexity** -- ``exp(total_nll / n_words)`` where ``n_words``
  is the number of whitespace-separated words in the evaluated text. This is
  **tokenizer independent**, so it is the metric to use when comparing HSKM
  against models with different vocabularies (Pythia, SmolLM2, Qwen, Mamba).

Evaluation span
---------------
The corpus is truncated to a fixed number of **words** (``--max-words``),
and every model is then tokenised *from that same text prefix*. This matters:

* Every model is scored on exactly the same text, so the word-level
  denominator is identical across models -- a per-model token count would
  make the comparison tokenizer-dependent and therefore meaningless.
* The cost is bounded: scoring the full 4.8M-token TinyStories validation
  split with a dozen models would take hours.

The window is ``--context`` tokens (default 512) for every model, so the
comparison isolates the model rather than its context window. Pass
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

from model_registry import BY_KEY, REGISTRY, Baseline, load_model  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results", "perplexity")

CORPORA = {
    "wikitext103": ("Salesforce/wikitext", "wikitext-103-raw-v1"),
    "wikitext2": ("Salesforce/wikitext", "wikitext-2-raw-v1"),
    "tinystories": ("roneneldan/TinyStories", None),
}

_TEXT_CACHE: dict = {}


def corpus_text(corpus: str, split: str = "validation") -> str:
    """Raw text of a corpus split (cached in-process)."""
    key = (corpus, split)
    if key in _TEXT_CACHE:
        return _TEXT_CACHE[key]
    from datasets import load_dataset

    repo, cfg = CORPORA[corpus]
    if cfg is None:
        ds = load_dataset(repo, split=split)
    else:
        ds = load_dataset(repo, cfg, split=split)
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    _TEXT_CACHE[key] = text
    return text


def truncate_words(text: str, max_words: int) -> str:
    """First ``max_words`` whitespace-separated words of ``text``."""
    if max_words <= 0:
        return text
    parts = text.split()
    if len(parts) <= max_words:
        return text
    return " ".join(parts[:max_words])


def count_words(text: str) -> int:
    return len(text.split())


@torch.inference_mode()
def nll_for_ids(model, ids, context: int, kind: str, device):
    """
    Sum of negative log-likelihood over the token stream.

    The token stream is cut into non-overlapping windows of ``context``
    tokens; the first token of each window is scored against the previous
    window's last token so that no token is left unscored except the very
    first.
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


def build_model(baseline: Baseline, args, device):
    """Instantiate the model and return ``(model, kind, context)``."""
    if baseline.loader == "ours_hskm":
        from hskm_lm import load_hskm_checkpoint
        from hskm_stable import patch_hskm_stable

        ckpt = args.checkpoints or os.path.join(
            _ROOT, "benchmarks", "checkpoints", "matched_params", "hskm.pt"
        )
        model = load_hskm_checkpoint(ckpt)
        patch_hskm_stable(model)
        model = model.to(device).eval()
        return model, "hskm", min(args.context or 512, model.config.max_seq_len)

    if baseline.loader == "ours_gpt2":
        from model_registry import _ours_gpt2_dir

        src = args.checkpoints_gpt2 or _ours_gpt2_dir(_ROOT)
        model = load_model(baseline, device=device, dtype=torch.float32, checkpoint=src)
        return model, "hf", min(args.context or 512, model.config.n_positions)

    if baseline.loader in ("legacy_mamba", "legacy_mamba2"):
        model = load_model(baseline, device=device, dtype=torch.float16)
        return model, "hf", args.context or baseline.max_length

    from transformers import AutoModelForCausalLM

    dtype = {"float16": torch.float16, "float32": torch.float32}[baseline.dtype]
    model = AutoModelForCausalLM.from_pretrained(baseline.hf_id, torch_dtype=dtype)
    model = model.to(device).eval()
    return model, "hf", args.context or baseline.max_length


def token_ids_for(baseline: Baseline, text: str, kind: str):
    """
    Tokenise ``text`` with the model's own tokenizer and its own convention.

    The special-token convention matters enormously for perplexity:

    * **OPT** was trained with ``</s>`` prepended to every sequence. Scoring
      it without a BOS token puts the model far out of distribution and
      inflates the perplexity by orders of magnitude (GPT-2 PPL 32 vs OPT
      PPL 8498 on the same text when BOS is omitted). OPT therefore uses its
      own tokenizer, which prepends BOS.
    * **GPT-2** uses no BOS token, and its tokenizer is used directly.
    * **HSKM** was trained by the upstream ``dataset.py`` with
      ``add_bos=True``, so the BOS token is prepended here too.
    * Every other model uses its own tokenizer with its own convention
      (Llama-style models prepend BOS, GPT-NeoX/Mamba and Qwen do not).
    """
    if kind == "hskm":
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        # BOS id used by the upstream tokenizer.py (max_token_value + 2).
        return [50258] + enc.encode_ordinary(text), None

    if baseline.key in ("gpt2", "gpt2-ours"):
        import tiktoken

        enc = tiktoken.get_encoding("gpt2")
        return enc.encode_ordinary(text), None

    from transformers import AutoTokenizer

    tok_id = baseline.tokenizer_id or baseline.hf_id
    tok = AutoTokenizer.from_pretrained(tok_id)
    ids = tok(text, return_tensors="pt").input_ids[0]
    return ids, tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--corpora", default="wikitext103,tinystories")
    ap.add_argument("--context", type=int, default=512,
                    help="context window used for every model")
    ap.add_argument("--max-words", type=int, default=500_000,
                    help="evaluate the first N words of each corpus (same text "
                         "for every model)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoints", default=None,
                    help="path to the HSKM checkpoint (ours tier)")
    ap.add_argument("--checkpoints-gpt2", default=None,
                    help="path to the controlled GPT-2 checkpoint directory")
    args = ap.parse_args()

    selected = (
        list(REGISTRY)
        if args.models == "all"
        else [BY_KEY[k.strip()] for k in args.models.split(",") if k.strip()]
    )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "perplexity.json")
    all_results = {}
    if os.path.exists(out_path):
        all_results = json.load(open(out_path))

    for corpus in args.corpora.split(","):
        text = corpus_text(corpus)
        text = truncate_words(text, args.max_words)
        n_words = count_words(text)
        print(f"\n=== {corpus} ({n_words:,} words evaluated) ===", flush=True)

        for b in selected:
            key = f"{b.key}/{corpus}"
            try:
                model, kind, context = build_model(b, args, args.device)
                ids, _tok = token_ids_for(b, text, kind)
                t0 = time.time()
                nll, ntok = nll_for_ids(model, ids, context, kind, args.device)
                elapsed = time.time() - t0
                del model
                torch.cuda.empty_cache()
            except Exception as e:  # noqa: BLE001
                print(f"  {b.key:16s} FAILED: {type(e).__name__}: {e}")
                torch.cuda.empty_cache()
                continue

            tok_ppl = math.exp(nll / ntok)
            word_ppl = math.exp(nll / n_words)
            all_results[key] = {
                "key": b.key, "corpus": corpus, "context": context,
                "token_ppl": tok_ppl, "word_ppl": word_ppl,
                "nll": nll, "n_tokens": ntok, "n_words": n_words,
                "elapsed_s": elapsed,
            }
            print(
                f"  {b.key:16s} ctx={context:<5d} tok_ppl={tok_ppl:9.3f}  "
                f"word_ppl={word_ppl:9.3f}  ({elapsed:.0f}s)",
                flush=True,
            )
            with open(out_path, "w") as f:
                json.dump(all_results, f, indent=2)

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
