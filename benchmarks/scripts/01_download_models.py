#!/usr/bin/env python3
"""
Download every model checkpoint and dataset needed by the benchmark suite.

Downloads are idempotent -- anything already in the HF cache is skipped.

Examples
--------
    python benchmarks/scripts/01_download_models.py
    python benchmarks/scripts/01_download_models.py --models gpt2,mamba-130m
"""

from __future__ import annotations

import argparse
import os
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "src"))

from model_registry import BY_KEY, REGISTRY  # noqa: E402


def download_model(repo_id: str, kind: str = "model"):
    from huggingface_hub import snapshot_download

    patterns = (
        ["*.json", "*.txt", "*.model", "*.safetensors", "*.bin", "*.tiktoken", "*.py"]
        if kind == "model"
        else ["*.json", "*.txt", "*.parquet", "*.arrow"]
    )
    t0 = time.time()
    try:
        snapshot_download(repo_id, allow_patterns=patterns)
        print(f"  OK      {repo_id}  ({time.time()-t0:.1f}s)")
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  FAILED  {repo_id}: {type(e).__name__}: {str(e)[:160]}")
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="all")
    ap.add_argument("--datasets", action="store_true", help="also pre-fetch datasets")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    keys = (
        list(REGISTRY)
        if args.models == "all"
        else [BY_KEY[k.strip()] for k in args.models.split(",") if k.strip()]
    )

    print("=" * 78)
    print("  Downloading model checkpoints")
    print("=" * 78)

    ok, failed = [], []
    for b in keys:
        if not b.hf_id:
            print(f"  SKIP    {b.key} (trained locally)")
            continue
        (ok if download_model(b.hf_id) else failed).append(b.key)

        # Some repos ship no tokenizer; fetch the one the registry points at.
        if b.tokenizer_id and b.tokenizer_id != b.hf_id:
            download_model(b.tokenizer_id)

    if args.datasets:
        print("\n" + "=" * 78)
        print("  Preparing datasets")
        print("=" * 78)
        from data import tokenize_corpus

        for name, ds in [("wikitext103", "wikitext103"), ("tinystories", "tinystories")]:
            for split in ("train", "validation"):
                try:
                    tokenize_corpus(name, split, dataset=ds)
                    print(f"  OK      {ds}/{split}")
                except Exception as e:  # noqa: BLE001
                    print(f"  FAILED  {ds}/{split}: {type(e).__name__}: {e}")

    print("\n" + "=" * 78)
    print(f"  downloaded: {len(ok)}   failed: {len(failed)}")
    if failed:
        print(f"  failures: {failed}")
    print("=" * 78)


if __name__ == "__main__":
    main()
