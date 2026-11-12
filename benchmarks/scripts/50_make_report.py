#!/usr/bin/env python3
"""
Turn the raw benchmark outputs into comparison tables and figures.

Reads:
    benchmarks/results/lm_eval/_summary.json
    benchmarks/results/perplexity/perplexity.json
    benchmarks/results/scaling/scaling.json
    benchmarks/results/training/controlled_training_summary.json

Writes:
    benchmarks/results/REPORT.md
    benchmarks/results/figures/*.png
    benchmarks/results/tables/*.md

Examples
--------
    python benchmarks/scripts/50_make_report.py
"""

from __future__ import annotations

import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "src"))

from model_registry import REGISTRY  # noqa: E402

RES = os.path.join(_ROOT, "benchmarks", "results")
FIGS = os.path.join(RES, "figures")
TABLES = os.path.join(RES, "tables")

ORDER = [b.key for b in REGISTRY]
TIER_OF = {b.key: b.tier for b in REGISTRY}
FAMILY_OF = {b.key: b.family for b in REGISTRY}

COLORS = {
    "ours": "#d62728",
    "linear": "#2ca02c",
    "attention": "#1f77b4",
}


def load(path, default=None):
    p = os.path.join(RES, path)
    if not os.path.exists(p):
        return default
    with open(p) as f:
        return json.load(f)


def color_of(key):
    return COLORS.get(TIER_OF.get(key, "attention"), "#1f77b4")


def md_table(headers, rows):
    out = ["| " + " | ".join(headers) + " |",
           "|" + "|".join("---" for _ in headers) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def fmt(x, nd=3):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if x != x:
            return "n/a"
        return f"{x:.{nd}f}"
    return str(x)


# --------------------------------------------------------------------------
# Table builders
# --------------------------------------------------------------------------

def lm_eval_table(lm):
    if not lm:
        return None
    tasks = []
    for key in ORDER:
        for t in (lm.get(key) or {}):
            if t not in tasks:
                tasks.append(t)
    if not tasks:
        return None

    headers = ["Model", "Tier", "Family"] + tasks + ["Avg acc"]
    rows = []
    for key in ORDER:
        m = lm.get(key)
        if not m:
            continue
        cells = []
        accs = []
        for t in tasks:
            v = m.get(t, {})
            # Prefer acc_norm when available (the standard headline metric).
            val = v.get("acc_norm", v.get("acc"))
            cells.append(fmt(val))
            if val is not None:
                accs.append(val)
        avg = sum(accs) / len(accs) if accs else None
        rows.append([f"`{key}`", TIER_OF.get(key, ""), FAMILY_OF.get(key, ""),
                     *cells, fmt(avg)])

    out = ["### lm-evaluation-harness (0-shot)", ""]
    out.append(md_table(headers, rows))
    out.append("")
    out.append("`acc_norm` is shown when the task provides it, otherwise `acc`.")
    return "\n".join(out)


def ppl_table(ppl):
    if not ppl:
        return None
    corpora = []
    for k in ppl:
        c = k.split("/")[-1]
        if c not in corpora:
            corpora.append(c)
    if not corpora:
        return None

    headers = ["Model", "Tier"]
    for c in corpora:
        headers += [f"{c} tok-PPL", f"{c} word-PPL"]
    rows = []
    for key in ORDER:
        entries = [v for k, v in ppl.items() if k.startswith(f"{key}/")]
        if not entries:
            continue
        cells = []
        for c in corpora:
            e = next((v for v in entries if v["corpus"] == c), None)
            cells.append(fmt(e["token_ppl"], 2) if e else "n/a")
            cells.append(fmt(e["word_ppl"], 2) if e else "n/a")
        rows.append([f"`{key}`", TIER_OF.get(key, ""), *cells])
    out = ["### Perplexity", ""]
    out.append(md_table(headers, rows))
    out.append("")
    out.append(
        "Word-level PPL is tokenizer-independent and is the metric to compare "
        "HSKM against models with a different vocabulary."
    )
    return "\n".join(out)


def scaling_table(sc):
    if not sc:
        return None
    headers = [
        "Model", "Params (M)", "Body time exponent b", "Body mem exponent b",
        "Full-step time exponent b", "Verdict",
    ]
    rows = []
    for key in ORDER:
        e = sc.get(key)
        if not e:
            continue
        b = e.get("body_time_exponent", e.get("time_exponent"))
        verdict = "n/a"
        if b is not None and b == b:
            if b < 1.35:
                verdict = "linear"
            elif b < 1.7:
                verdict = "sub-quadratic"
            else:
                verdict = "quadratic"
        rows.append([
            f"`{key}`", fmt(e.get("params_m"), 1), fmt(b, 2),
            fmt(e.get("body_mem_exponent"), 2),
            fmt(e.get("time_exponent"), 2), verdict,
        ])
    out = ["### Scaling exponents", ""]
    out.append(md_table(headers, rows))
    out.append("")
    out.append(
        "`b` is the least-squares slope of `log(metric)` against `log(seq_len)`. "
        "A quadratic attention model gives `b ~ 2`; a linear model gives `b ~ 1`. "
        "The **body** columns exclude the LM head, whose `O(N)` cost is identical "
        "for every model and would otherwise dilute the measured exponent of the "
        "attention baselines towards 1."
    )
    return "\n".join(out)


def load_training_summaries():
    """
    Merge the controlled-training summaries.

    The two models may be trained in separate processes (e.g. on separate
    GPUs), which writes separate ``controlled_*.json`` files. They are merged
    here into one dict keyed by the registry key.
    """
    d = os.path.join(RES, "training")
    if not os.path.isdir(d):
        return {}
    merged = {}
    for fn in sorted(os.listdir(d)):
        if not fn.startswith("controlled") or not fn.endswith(".json"):
            continue
        try:
            part = json.load(open(os.path.join(d, fn)))
        except Exception:  # noqa: BLE001
            continue
        for k, v in part.items():
            merged[k] = v
    # Map onto the registry keys used elsewhere.
    if "hskm" in merged:
        merged["hskm-ours"] = merged.pop("hskm")
    if "gpt2-matched-params" in merged:
        merged["gpt2-ours"] = merged.pop("gpt2-matched-params")
    if "gpt2-matched-depth" in merged:
        merged["gpt2-ours-depth"] = merged.pop("gpt2-matched-depth")
    return merged


def controlled_table(ctrl):
    if not ctrl:
        return None
    headers = ["Model", "Params (M)", "Tokens seen", "Val loss", "Val PPL"]
    rows = []
    for key in ["hskm-ours", "gpt2-ours", "gpt2-ours-depth"]:
        e = ctrl.get(key)
        if not e:
            continue
        rows.append([
            f"`{key}`", fmt(e["params"] / 1e6, 1), f"{e['tokens_seen']:,}",
            fmt(e["val_loss"], 4), fmt(e["val_ppl"], 2),
        ])
    if not rows:
        return None
    out = ["### Controlled from-scratch comparison", ""]
    out.append(md_table(headers, rows))
    out.append("")
    out.append(
        "Identical corpus, token budget, optimiser and schedule; the only "
        "difference is the architecture. `gpt2-ours` is matched to HSKM on "
        "**parameter count**; `gpt2-ours-depth` (if present) is matched on "
        "**depth** instead."
    )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def fig_task_bars(lm):
    if not lm:
        return
    tasks = []
    for key in ORDER:
        for t in (lm.get(key) or {}):
            if t not in tasks:
                tasks.append(t)
    if not tasks:
        return
    models = [k for k in ORDER if lm.get(k)]

    fig, axes = plt.subplots(1, len(tasks), figsize=(3.1 * len(tasks), 4.6), sharey=True)
    if len(tasks) == 1:
        axes = [axes]
    for ax, task in zip(axes, tasks):
        vals = []
        for k in models:
            v = (lm[k].get(task) or {})
            val = v.get("acc_norm", v.get("acc"))
            vals.append(val if val is not None else 0.0)
        ax.bar(range(len(models)), vals,
               color=[color_of(k) for k in models], edgecolor="none")
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=75, ha="right", fontsize=7)
        ax.set_title(task, fontsize=9)
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("accuracy")
    fig.suptitle("lm-evaluation-harness results (0-shot)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "lm_eval_tasks.png"), dpi=160)
    plt.close(fig)


def fig_scaling(sc):
    if not sc:
        return
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.6))

    # --- body forward time (isolates the sequence-mixing architecture) ---
    ax = axes[0]
    for key, e in sc.items():
        rows = [r for r in e.get("body", []) if not r.get("oom")]
        if len(rows) < 2:
            continue
        xs = [r["seq_len"] for r in rows]
        ys = [r["ms_per_step"] for r in rows]
        ax.plot(xs, ys, "o-", label=f"{key} (b={e.get('body_time_exponent', float('nan')):.2f})",
                color=color_of(key))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("ms / forward (body only)")
    ax.set_title("Body forward time (no LM head)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[1]
    for key, e in sc.items():
        rows = [r for r in e.get("train", []) if not r.get("oom")]
        if len(rows) < 2:
            continue
        xs = [r["seq_len"] for r in rows]
        ys = [r["ms_per_step"] for r in rows]
        ax.plot(xs, ys, "o-", label=f"{key} (b={e.get('time_exponent', float('nan')):.2f})",
                color=color_of(key))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("ms / step (fwd+bwd)")
    ax.set_title("Full training step")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[2]
    for key, e in sc.items():
        rows = [r for r in e.get("body", []) if not r.get("oom")]
        if len(rows) < 2:
            continue
        ax.plot([r["seq_len"] for r in rows], [r["peak_mem_mb"] for r in rows], "o-",
                label=f"{key} (b={e.get('body_mem_exponent', float('nan')):.2f})",
                color=color_of(key))
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("sequence length")
    ax.set_ylabel("peak VRAM (MB)")
    ax.set_title("Peak memory (body only)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    ax = axes[3]
    for key, e in sc.items():
        rows = [r for r in e.get("generation", []) if not r.get("oom")]
        if len(rows) < 2:
            continue
        ax.plot([r["ctx_len"] for r in rows], [r["gen_tok_per_s"] for r in rows], "o-",
                label=key, color=color_of(key))
    ax.set_xscale("log", base=2)
    ax.set_xlabel("context length")
    ax.set_ylabel("generation tokens/s")
    ax.set_title("Generation throughput")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)

    fig.suptitle("HSKM scaling behaviour vs. baselines", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "scaling.png"), dpi=160)
    plt.close(fig)


def fig_ppl(ppl):
    if not ppl:
        return
    corpora = sorted({k.split("/")[-1] for k in ppl})
    models = [k for k in ORDER if any(x.startswith(f"{k}/") for x in ppl)]
    if not models:
        return
    fig, axes = plt.subplots(1, len(corpora), figsize=(6 * len(corpora), 4.2))
    if len(corpora) == 1:
        axes = [axes]
    for ax, corpus in zip(axes, corpora):
        vals = []
        for k in models:
            e = ppl.get(f"{k}/{corpus}")
            vals.append(e["word_ppl"] if e else np.nan)
        ax.bar(range(len(models)), vals, color=[color_of(k) for k in models])
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels(models, rotation=75, ha="right", fontsize=7)
        ax.set_ylabel("word-level perplexity (lower is better)")
        ax.set_title(corpus)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Held-out perplexity (lower is better)", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGS, "perplexity.png"), dpi=160)
    plt.close(fig)


def main():
    os.makedirs(FIGS, exist_ok=True)
    os.makedirs(TABLES, exist_ok=True)

    lm = load("lm_eval/_summary.json")
    ppl = load("perplexity/perplexity.json")
    sc = load("scaling/scaling.json")
    ctrl = load_training_summaries()

    sections = []
    for title, builder in [
        ("Controlled from-scratch comparison", controlled_table),
        ("Perplexity", ppl_table),
        ("lm-evaluation-harness", lm_eval_table),
        ("Scaling", scaling_table),
    ]:
        s = builder({"Controlled from-scratch comparison": ctrl,
                     "Perplexity": ppl,
                     "lm-evaluation-harness": lm,
                     "Scaling": sc}[title])
        if s:
            sections.append(s)

    parts = ["# HSKM Benchmark Report", "",
             "Auto-generated by `benchmarks/scripts/50_make_report.py`.", ""]
    parts.extend(sections)

    with open(os.path.join(RES, "REPORT.md"), "w") as f:
        f.write("\n\n".join(parts) + "\n")

    fig_lm_eval = fig_task_bars(lm)
    fig_scaling(sc)
    fig_ppl(ppl)

    print(f"wrote {os.path.join(RES, 'REPORT.md')}")
    print(f"wrote figures to {FIGS}")


if __name__ == "__main__":
    main()
