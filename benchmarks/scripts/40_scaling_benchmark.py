#!/usr/bin/env python3
"""
Scaling benchmark: does HSKM actually deliver O(N) sequence modelling?

This is the experiment that tests the project's central claim. For every
model we measure, as a function of sequence length:

* forward+backward wall-clock time per step  -> fitted scaling exponent
* peak GPU memory                            -> fitted scaling exponent
* raw training throughput (tokens/s)

A least-squares fit of ``log(metric) = a + b * log(seq_len)`` gives the
empirical exponent ``b``: ``b ~ 1`` for linear, ``b ~ 2`` for quadratic
attention.

Methodological notes
--------------------
* **Causal baselines only.** The upstream ``benchmark.py`` compares HSKM
  against ``nn.TransformerEncoderLayer``, which is a *bidirectional* encoder
  layer -- not a valid decoder baseline. Here the attention baseline is the
  real ``GPT2LMHeadModel`` (genuinely causal, and the direct ancestor of
  HSKM's design), plus Pythia (GPT-NeoX) and Mamba (linear SSM).
* **Numerically stable HSKM.** Upstream HSKM overflows float32 at >= 1024
  tokens, so the long-context points would be NaN. The segmented-scan EMA
  (``hskm_stable.py``) is used instead; it is mathematically identical and
  verified in ``tests/test_numerics.py``.
* **Fair dtype.** Every model runs in the same dtype so the timings are
  comparable.
* **Architecture, not weights.** Attention models (GPT-2, Pythia) are built
  from config with random weights and a position table widened to cover the
  sweep: a scaling study measures the architecture, and learned absolute
  position embeddings sized to a 2048-token training context would
  otherwise cap the sweep for a reason unrelated to complexity. HSKM and
  Mamba are recurrent/position-independent by construction. (GPT-2-style
  absolute positions *are* a genuine handicap at long context — which is
  exactly why the field moved to RoPE.)
* OOM is recorded rather than crashing the sweep.

Examples
--------
    python benchmarks/scripts/40_scaling_benchmark.py
    python benchmarks/scripts/40_scaling_benchmark.py --seq-lens 512,1024,2048,4096
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "src"))

from model_registry import BY_KEY, REGISTRY, Baseline, count_params  # noqa: E402

RESULTS_DIR = os.path.join(_ROOT, "benchmarks", "results", "scaling")


# --------------------------------------------------------------------------
# Model construction (architecture only -- weights are irrelevant for timing)
# --------------------------------------------------------------------------

def build_for_scaling(key: str, args, device, dtype):
    """
    Build a model for timing.

    For HSKM we use the *trained* checkpoint when available (identical
    parameter count to the published configuration otherwise).
    """
    from model import HSKM, HSKMConfig
    from hskm_stable import patch_hskm_stable

    if key == "hskm":
        ckpt_path = os.path.join(
            _ROOT, "benchmarks", "checkpoints", "matched_params", "hskm.pt"
        )
        if os.path.exists(ckpt_path):
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cfg = HSKMConfig(**ckpt["config"])
            model = HSKM(cfg)
            model.load_state_dict(ckpt["model"])
        else:
            cfg = HSKMConfig(
                vocab_size=50259, d_model=512, n_layers=8, n_heads=8,
                d_medium=128, n_kernels=64, top_k=16, n_patterns=4096,
                max_seq_len=max(args.seq_lens), dropout=0.0,
            )
            model = HSKM(cfg)
        patch_hskm_stable(model)
        return model.to(device).to(dtype), "hskm"

    if key == "gpt2":
        from transformers import GPT2Config, GPT2LMHeadModel

        # Random weights: a scaling study measures the architecture.
        # GPT-2 uses learned absolute position embeddings, widened to cover
        # the sweep (see the note on pythia-160m below).
        cfg = GPT2Config(
            vocab_size=50257, n_positions=max(args.seq_lens) + 8,
            n_embd=args.d_model, n_layer=args.n_layers, n_head=args.n_heads,
        )
        torch.manual_seed(0)
        return GPT2LMHeadModel(cfg).to(device).to(dtype), "hf"

    if key == "gpt2-ctrl":
        # The controlled from-scratch baseline (matched parameter count).
        from transformers import GPT2Config, GPT2LMHeadModel

        n_layer = args.ctrl_layers
        cfg = GPT2Config(
            vocab_size=50259, n_positions=max(args.seq_lens) + 8,
            n_embd=args.d_model, n_layer=n_layer, n_head=args.n_heads,
        )
        torch.manual_seed(0)
        return GPT2LMHeadModel(cfg).to(device).to(dtype), "hf"

    if key == "pythia-160m":
        # Built from config (random weights) rather than from the checkpoint:
        # Pythia uses *learned absolute* position embeddings sized to its
        # training context (2048), which would cap the sweep at 2048 tokens
        # for a reason that has nothing to do with the architecture's
        # complexity. A scaling study measures the architecture, so we widen
        # the position table and note that GPT-2-style absolute positions are
        # a genuine handicap at long context -- which is precisely why the
        # field moved to RoPE.
        from transformers import AutoConfig, AutoModelForCausalLM

        cfg = AutoConfig.from_pretrained("EleutherAI/pythia-160m")
        cfg.max_position_embeddings = max(args.seq_lens) + 8
        torch.manual_seed(0)
        return AutoModelForCausalLM.from_config(cfg).to(device).to(dtype), "hf"

    if key == "mamba-130m":
        from model_registry import load_model

        b = BY_KEY["mamba-130m"]
        m = load_model(b, device=device, dtype=dtype)
        return m, "hf"

    if key == "mamba2-130m":
        from model_registry import load_model

        b = BY_KEY["mamba2-130m"]
        m = load_model(b, device=device, dtype=dtype)
        return m, "hf"

    raise ValueError(key)


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------

def body_forward(model, kind, input_ids):
    """
    Run only the sequence-mixing body, excluding the LM head.

    The LM head produces a ``[B, T, vocab]`` tensor whose cost is ``O(N)``
    for *every* model. Including it dilutes the measured exponent of a
    quadratic model towards 1 and would flatter the attention baselines.
    Measuring the body alone isolates the architecture that the O(N) claim is
    actually about; the full-step numbers are reported alongside it.
    """
    if kind == "hskm":
        x = model.token_emb(input_ids)
        x = model.dropout(x)
        freqs = model.freqs_cis.to(x.device)
        for block in model.blocks:
            x = block(x, freqs)
        return model.ln_f(x)
    return model.base_model(input_ids).last_hidden_state


def measure_body_forward(model, kind, seq_len, batch_size, device, vocab,
                          warmup=2, steps=5):
    """Time a forward pass through the body only (no LM head, no backward)."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randint(0, vocab, (batch_size, seq_len), device=device, dtype=torch.long)

    try:
        with torch.inference_mode():
            for _ in range(warmup):
                body_forward(model, kind, x)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(steps):
                body_forward(model, kind, x)
            torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) / steps
        peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    except torch.cuda.OutOfMemoryError:
        return {"seq_len": seq_len, "oom": True}
    tokens = batch_size * seq_len
    return {
        "seq_len": seq_len,
        "ms_per_step": elapsed * 1000,
        "tok_per_s": tokens / elapsed,
        "peak_mem_mb": peak_mb,
        "oom": False,
    }


def measure_train_step(model, kind, seq_len, batch_size, device, vocab, dtype,
                       warmup=2, steps=5):
    """Time a forward+backward step and record peak memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    x = torch.randint(0, vocab, (batch_size, seq_len), device=device, dtype=torch.long)

    def one_step():
        if kind == "hskm":
            loss, _ = model(x, labels=x)
        else:
            out = model(input_ids=x)
            loss = F.cross_entropy(
                out.logits.view(-1, out.logits.size(-1)).float(), x.view(-1)
            )
        loss.backward()
        model.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one_step()
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / steps

    peak_mb = torch.cuda.max_memory_allocated() / 1024**2
    tokens = batch_size * seq_len
    return {
        "seq_len": seq_len,
        "ms_per_step": elapsed * 1000,
        "tok_per_s": tokens / elapsed,
        "peak_mem_mb": peak_mb,
    }


def measure_generation(model, kind, ctx_len, device, vocab, new_tokens=64):
    """
    Measure generation throughput at a given context length.

    HSKM's ``generate`` re-runs a full forward pass per token (it has no KV
    cache), so its per-token cost grows with the context. A transformer with
    a KV cache pays O(1) per token. This measurement quantifies that.
    """
    x = torch.randint(0, vocab, (1, ctx_len), device=device, dtype=torch.long)
    try:
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            if kind == "hskm":
                model.generate(x, max_new_tokens=new_tokens)
            else:
                model.generate(
                    x,
                    max_new_tokens=new_tokens,
                    do_sample=False,
                    pad_token_id=vocab - 1,
                    use_cache=True,
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
    except torch.cuda.OutOfMemoryError:
        return {"ctx_len": ctx_len, "oom": True}
    return {
        "ctx_len": ctx_len,
        "gen_tok_per_s": new_tokens / elapsed,
        "ms_per_token": elapsed / new_tokens * 1000,
        "oom": False,
    }


def fit_exponent(xs, ys):
    """Least-squares slope of log(y) vs log(x)."""
    xs = [x for x, y in zip(xs, ys) if y and y > 0 and not math.isnan(y)]
    ys = [y for x, y in zip(xs, ys) if y and y > 0 and not math.isnan(y)]
    if len(xs) < 2:
        return float("nan")
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    n = len(lx)
    mx = sum(lx) / n
    my = sum(ly) / n
    num = sum((a - mx) * (b - my) for a, b in zip(lx, ly))
    den = sum((a - mx) ** 2 for a in lx)
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="hskm,gpt2-ctrl,gpt2,pythia-160m,mamba-130m,mamba2-130m")
    ap.add_argument("--seq-lens", default="512,1024,2048,4096,8192,16384")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--dtype", default="float16")
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--ctrl-layers", type=int, default=14,
                    help="depth of the parameter-matched GPT-2 control")
    ap.add_argument("--skip-generation", action="store_true")
    ap.add_argument("--gen-contexts", default="512,1024,2048,4096,8192")
    args = ap.parse_args()

    args.seq_lens = [int(s) for s in args.seq_lens.split(",")]
    args.gen_contexts = [int(s) for s in args.gen_contexts.split(",")]
    dtype = {"float16": torch.float16, "float32": torch.float32}[args.dtype]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "scaling.json")
    results = json.load(open(out_path)) if os.path.exists(out_path) else {}

    for key in args.models.split(","):
        key = key.strip()
        if not key:
            continue
        print("\n" + "=" * 78)
        print(f"  scaling: {key}")
        print("=" * 78, flush=True)

        try:
            model, kind = build_for_scaling(key, args, device, dtype)
        except Exception as e:  # noqa: BLE001
            print(f"  !! could not build {key}: {type(e).__name__}: {e}")
            continue

        n_params = count_params(model)
        vocab = model.config.vocab_size
        print(f"  params {n_params/1e6:.1f}M   dtype {args.dtype}   batch {args.batch_size}")

        rows = []
        body_rows = []
        for L in args.seq_lens:
            try:
                r = measure_train_step(
                    model, kind, L, args.batch_size, device, vocab, dtype, steps=args.steps
                )
                r["oom"] = False
                print(
                    f"  L={L:6d}  step {r['ms_per_step']:9.1f} ms  {r['tok_per_s']:11,.0f} tok/s  "
                    f"{r['peak_mem_mb']:9.1f} MB"
                )
            except torch.cuda.OutOfMemoryError:
                r = {"seq_len": L, "oom": True}
                print(f"  L={L:6d}  step OOM")
                torch.cuda.empty_cache()
            rows.append(r)

            b = measure_body_forward(
                model, kind, L, args.batch_size, device, vocab, steps=args.steps
            )
            if not b.get("oom"):
                print(
                    f"           body {b['ms_per_step']:9.1f} ms  {b['tok_per_s']:11,.0f} tok/s  "
                    f"{b['peak_mem_mb']:9.1f} MB"
                )
            body_rows.append(b)
            torch.cuda.empty_cache()

            results.setdefault(key, {})["train"] = rows
            results[key]["body"] = body_rows
            with open(out_path, "w") as f:
                json.dump(results, f, indent=2)

        ok = [r for r in rows if not r.get("oom")]
        bok = [r for r in body_rows if not r.get("oom")]
        results.setdefault(key, {})["params_m"] = n_params
        if len(ok) >= 2:
            results[key]["time_exponent"] = fit_exponent(
                [r["seq_len"] for r in ok], [r["ms_per_step"] for r in ok]
            )
            results[key]["mem_exponent"] = fit_exponent(
                [r["seq_len"] for r in ok], [r["peak_mem_mb"] for r in ok]
            )
            print(
                f"  -> fitted full-step time exponent b = {results[key]['time_exponent']:.2f} "
                f"(1 = linear, 2 = quadratic)"
            )
        if len(bok) >= 2:
            results[key]["body_time_exponent"] = fit_exponent(
                [r["seq_len"] for r in bok], [r["ms_per_step"] for r in bok]
            )
            results[key]["body_mem_exponent"] = fit_exponent(
                [r["seq_len"] for r in bok], [r["peak_mem_mb"] for r in bok]
            )
            print(
                f"  -> fitted body-forward time exponent b = "
                f"{results[key]['body_time_exponent']:.2f}"
            )

        if not args.skip_generation:
            grows = []
            for C in args.gen_contexts:
                try:
                    g = measure_generation(model, kind, C, device, vocab)
                except torch.cuda.OutOfMemoryError:
                    g = {"ctx_len": C, "oom": True}
                grows.append(g)
                if not g.get("oom"):
                    print(f"  gen ctx={C:6d}  {g['gen_tok_per_s']:9.1f} tok/s  {g['ms_per_token']:7.2f} ms/tok")
                else:
                    print(f"  gen ctx={C:6d}  OOM")
                torch.cuda.empty_cache()
            results.setdefault(key, {})["generation"] = grows

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

        del model
        torch.cuda.empty_cache()

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
