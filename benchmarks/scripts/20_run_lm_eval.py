#!/usr/bin/env python3
"""
Run the lm-evaluation-harness suite across every model in the registry.

The task list is the standard "small decoder LM" panel used across the
literature, so that our numbers can be compared with published results:

    lambada_openai    last-word prediction (word-level acc + ppl)
    piqa              physical commonsense
    hellaswag         sentence completion (adversarially filtered)
    arc_easy          grade-school science, easy
    arc_challenge     grade-school science, hard
    winogrande        pronoun resolution
    boolq             yes/no reading comprehension
    openbookqa        open-book science questions

Every model is evaluated with *identical* task versions, few-shot counts,
batching and scoring rules. HF baselines go through ``HFLM``; HSKM goes
through the custom adapter in ``benchmarks/src/hskm_lm.py`` which implements
the same loglikelihood contract.

Examples
--------
    python benchmarks/scripts/20_run_lm_eval.py --models gpt2,mamba-130m
    python benchmarks/scripts/20_run_lm_eval.py --all --limit 100      # smoke test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "src"))

from model_registry import BY_KEY, REGISTRY, Baseline, count_params, load_model  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results", "lm_eval")

DEFAULT_TASKS = [
    "lambada_openai",
    "piqa",
    "hellaswag",
    "arc_easy",
    "arc_challenge",
    "winogrande",
    "boolq",
    "openbookqa",
]

# Metrics to extract per task.
METRIC_KEYS = {
    "lambada_openai": ["acc,none", "acc_norm,none", "perplexity,none", "word_perplexity,none"],
    "piqa": ["acc,none", "acc_norm,none"],
    "hellaswag": ["acc,none", "acc_norm,none"],
    "arc_easy": ["acc,none", "acc_norm,none"],
    "arc_challenge": ["acc,none", "acc_norm,none"],
    "winogrande": ["acc,none"],
    "boolq": ["acc,none"],
    "openbookqa": ["acc,none", "acc_norm,none"],
}


def build_lm(baseline: Baseline, args, device):
    """Instantiate an lm-eval ``LM`` object for a baseline."""
    if baseline.loader == "ours_hskm":
        from hskm_lm import HSKMLM

        ckpt = args.checkpoints or os.path.join(
            _ROOT, "benchmarks", "checkpoints", "matched_params", "hskm.pt"
        )
        return HSKMLM(
            checkpoint=ckpt,
            device=device,
            batch_size=args.batch_size,
            max_length=baseline.max_length,
            stable=not args.unstable_hskm,
        )

    if baseline.loader == "ours_gpt2":
        # Our controlled GPT-2 baseline, saved in HF format by the training
        # script. It shares the GPT-2 tokenizer.
        from transformers import AutoTokenizer

        from model_registry import _ours_gpt2_dir

        src = args.checkpoints_gpt2 or _ours_gpt2_dir(_ROOT)
        model = load_model(baseline, device=device, dtype=dtype, checkpoint=src)
        tok = AutoTokenizer.from_pretrained(src)
        return HFLM(
            pretrained=model,
            tokenizer=tok,
            batch_size=args.batch_size,
            device=device,
            max_length=baseline.max_length,
            backend="causal",
        )

    from lm_eval.models.huggingface import HFLM

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[baseline.dtype]

    if baseline.loader in ("legacy_mamba", "legacy_mamba2"):
        # HFLM cannot AutoConfig the legacy state-spaces repos, so we build
        # the model ourselves and hand it to HFLM. Those repos ship no
        # tokenizer, so we point at the upstream GPT-NeoX tokenizer.
        model = load_model(baseline, device=device, dtype=dtype)
        return HFLM(
            pretrained=model,
            tokenizer=baseline.tokenizer_id or baseline.hf_id,
            batch_size=args.batch_size,
            device=device,
            max_length=baseline.max_length,
            backend="causal",
        )

    return HFLM(
        pretrained=baseline.hf_id,
        tokenizer=baseline.tokenizer_id or baseline.hf_id,
        batch_size=args.batch_size,
        device=device,
        max_length=baseline.max_length,
        dtype=dtype,
        backend="causal",
    )


def extract_metrics(results, tasks):
    """Flatten lm-eval results into ``{task: {metric: value}}``."""
    out = {}
    for task in tasks:
        r = results.get("results", {}).get(task)
        if r is None:
            continue
        vals = {}
        for k in METRIC_KEYS.get(task, ["acc,none", "acc_norm,none"]):
            if k in r and isinstance(r[k], (int, float)):
                vals[k.replace(",none", "")] = float(r[k])
        out[task] = vals
    return out


def run_one(baseline: Baseline, args, device):
    import lm_eval

    print("\n" + "=" * 78)
    print(f"  {baseline.key}  ({baseline.hf_id or 'ours'})  [{baseline.tier}]")
    print("=" * 78, flush=True)

    t0 = time.time()
    try:
        lm = build_lm(baseline, args, device)
    except Exception as e:  # noqa: BLE001
        print(f"  !! failed to build model: {type(e).__name__}: {e}")
        return None

    try:
        results = lm_eval.simple_evaluate(
            model=lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            batch_size=args.batch_size,
            limit=args.limit,
            bootstrap_iters=0,
            verbosity="ERROR",
        )
    except Exception as e:  # noqa: BLE001
        import traceback

        traceback.print_exc()
        print(f"  !! evaluation failed: {type(e).__name__}: {e}")
        return None

    elapsed = time.time() - t0
    metrics = extract_metrics(results, args.tasks)
    record = {
        "key": baseline.key,
        "hf_id": baseline.hf_id,
        "family": baseline.family,
        "tier": baseline.tier,
        "params_m": baseline.params_m,
        "num_fewshot": args.num_fewshot,
        "limit": args.limit,
        "tasks": args.tasks,
        "metrics": metrics,
        "elapsed_s": elapsed,
        "raw": results.get("results", {}),
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, f"{baseline.key}.json"), "w") as f:
        json.dump(record, f, indent=2)

    for task, vals in metrics.items():
        pretty = "  ".join(f"{k}={v:.4f}" for k, v in vals.items())
        print(f"  {task:16s} {pretty}")
    print(f"  -> {elapsed:.0f}s")

    # free GPU memory
    del lm
    torch.cuda.empty_cache()
    return record


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="", help="comma-separated registry keys")
    ap.add_argument("--all", action="store_true", help="run every model in the registry")
    ap.add_argument("--tasks", default=",".join(DEFAULT_TASKS))
    ap.add_argument("--num-fewshot", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--checkpoints", default=None,
                    help="path to the HSKM checkpoint (ours tier)")
    ap.add_argument("--checkpoints-gpt2", default=None,
                    help="path to the controlled GPT-2 checkpoint directory")
    ap.add_argument("--unstable-hskm", action="store_true",
                    help="use the upstream (numerically unstable) EMA for HSKM")
    args = ap.parse_args()

    args.tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    if args.all:
        selected = list(REGISTRY)
    else:
        keys = [k.strip() for k in args.models.split(",") if k.strip()]
        selected = []
        for k in keys:
            if k not in BY_KEY:
                raise SystemExit(f"unknown model key '{k}'. known: {sorted(BY_KEY)}")
            selected.append(BY_KEY[k])

    os.makedirs(RESULTS_DIR, exist_ok=True)
    # Merge into any existing summary so that models evaluated in separate
    # invocations (e.g. the from-scratch checkpoints on a second GPU)
    # accumulate rather than overwrite each other.
    summary_path = os.path.join(RESULTS_DIR, "_summary.json")
    summary = {}
    if os.path.exists(summary_path):
        with open(summary_path) as f:
            summary = json.load(f)

    for b in selected:
        rec = run_one(b, args, args.device)
        if rec:
            summary[b.key] = rec["metrics"]
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2)

    print(f"\nwrote {summary_path}")


if __name__ == "__main__":
    main()
