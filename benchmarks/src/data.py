"""
Shared, cached data pipeline for the controlled from-scratch comparison.

Both the HSKM checkpoint and the controlled GPT-2 baseline are trained on the
*identical* token stream produced here, so any difference in their held-out
perplexity is attributable to the architecture alone.

The corpus is WikiText-103 (raw), tokenised with the GPT-2 BPE tokenizer
(tiktoken ``gpt2``). WikiText-103 is the standard language-modelling
benchmark corpus at this model scale and is the corpus for which published
reference perplexities exist, which makes the absolute numbers interpretable.

Tokenised corpora are cached as ``uint16`` ``.npy`` files under
``benchmarks/.cache`` so a re-run does not re-tokenise.
"""

from __future__ import annotations

import os
import time

import numpy as np

WIKITEXT_103 = ("Salesforce/wikitext", "wikitext-103-raw-v1")
TINYSTORIES = ("roneneldan/TinyStories", None)

DEFAULT_CACHE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".cache")


def get_gpt2_encoder():
    import tiktoken

    return tiktoken.get_encoding("gpt2")


def _cache_path(cache_dir: str, name: str) -> str:
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"{name}.npy")


def tokenize_corpus(
    name: str = "wikitext-103-train",
    split: str = "train",
    cache_dir: str = DEFAULT_CACHE,
    dataset: str = "wikitext103",
    max_tokens: int | None = None,
) -> np.ndarray:
    """
    Return a ``uint16`` array of token ids for ``dataset``/``split``.

    ``max_tokens`` truncates the corpus, which is useful for quick smoke tests.
    """
    path = _cache_path(cache_dir, f"{dataset}-{split}-{name}" + (f"-{max_tokens}" if max_tokens else ""))
    if os.path.exists(path):
        return np.load(path)

    from datasets import load_dataset

    enc = get_gpt2_encoder()

    if dataset == "wikitext103":
        repo, cfg = WIKITEXT_103
        ds = load_dataset(repo, cfg, split=split)
        texts = [t for t in ds["text"] if t.strip()]
        sep = "\n\n"
    elif dataset == "tinystories":
        repo, cfg = TINYSTORIES
        ds = load_dataset(repo, split=split)
        texts = [t for t in ds["text"] if t.strip()]
        sep = "\n\n"
    elif dataset == "wikitext2":
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        texts = [t for t in ds["text"] if t.strip()]
        sep = "\n\n"
    else:
        raise ValueError(f"unknown dataset {dataset}")

    t0 = time.time()
    print(f"[data] tokenising {len(texts)} documents from {dataset}/{split} ...", flush=True)
    ids = enc.encode_ordinary(sep.join(texts))
    print(f"[data] {len(ids):,} tokens in {time.time() - t0:.1f}s", flush=True)

    arr = np.array(ids, dtype=np.uint16)
    if max_tokens is not None:
        arr = arr[:max_tokens]
    np.save(path, arr)
    return arr


def make_blocks(tokens: np.ndarray, seq_len: int):
    """
    Turn a flat token stream into ``(x, y)`` blocks for next-token prediction.

    Returns ``(x, y)`` where ``x`` is ``tokens[:-1]`` shaped into rows of
    ``seq_len`` and ``y`` is the corresponding next-token targets.
    """
    n = (len(tokens) - 1) // seq_len
    usable = n * seq_len
    x = tokens[:usable].reshape(n, seq_len)
    y = tokens[1 : usable + 1].reshape(n, seq_len)
    return x, y


def iter_batches(x, y, batch_size: int, device: str = "cuda", shuffle: bool = True, seed: int = 0):
    """Yield ``(xb, yb)`` tensors on ``device``."""
    import torch

    n = x.shape[0]
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)
    for start in range(0, n - batch_size + 1, batch_size):
        sel = idx[start : start + batch_size]
        xb = torch.from_numpy(x[sel].astype(np.int64)).to(device, non_blocking=True)
        yb = torch.from_numpy(y[sel].astype(np.int64)).to(device, non_blocking=True)
        yield xb, yb
