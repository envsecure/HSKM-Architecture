#!/usr/bin/env python3
"""
Controlled from-scratch training: HSKM vs. a standard causal GPT-2.

Both models are trained on the *identical* token stream with the *identical*
optimiser, schedule, batch size, sequence length and total token budget.
The only difference is the architecture. This is the only fully apples-to-apples
quality comparison in the suite, because every other baseline is pretrained on
orders of magnitude more data.

Examples
--------
    # Train both, 100M tokens, WikiText-103
    python benchmarks/scripts/10_train_controlled.py --model both --tokens 100_000_000

    # Smoke test
    python benchmarks/scripts/10_train_controlled.py --model both --tokens 2_000_000
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

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "src"))

from data import iter_batches, make_blocks, tokenize_corpus  # noqa: E402
from model import HSKM, HSKMConfig  # noqa: E402
from hskm_stable import patch_hskm_stable  # noqa: E402

OUT_ROOT = os.path.join(_ROOT, "benchmarks", "checkpoints")
LOG_ROOT = os.path.join(_ROOT, "benchmarks", "results", "training")


# --------------------------------------------------------------------------
# Parameter matching
# --------------------------------------------------------------------------

def hskm_param_count(cfg: HSKMConfig) -> int:
    m = HSKM(cfg)
    seen, total = set(), 0
    for p in m.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def gpt2_param_count(n_embd: int, n_layer: int, n_head: int, vocab: int) -> int:
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=vocab, n_positions=1024, n_embd=n_embd, n_layer=n_layer, n_head=n_head
    )
    m = GPT2LMHeadModel(cfg)
    seen, total = set(), 0
    for p in m.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total


def find_matched_gpt2(target_params: int, vocab: int, n_embd: int, n_head: int,
                      max_layer: int = 32):
    """
    Find the GPT-2 depth whose parameter count is closest to ``target_params``.

    We match on *parameters* (the standard way to compare architectures of
    different shapes) rather than on depth, because HSKM's long-term memory
    bank carries a large parameter block that a transformer of the same depth
    does not have. The matched-depth variant is trained as well (see
    ``--also-matched-depth``) so the two effects can be separated.
    """
    best = None
    for n_layer in range(2, max_layer + 1):
        p = gpt2_param_count(n_embd, n_layer, n_head, vocab)
        delta = abs(p - target_params)
        if best is None or delta < best[0]:
            best = (delta, n_layer, p)
    return best[1], best[2]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

def build_hskm(args, vocab: int):
    cfg = HSKMConfig(
        vocab_size=vocab,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_medium=args.d_medium,
        n_kernels=args.n_kernels,
        top_k=args.top_k,
        n_patterns=args.n_patterns,
        max_seq_len=args.seq_len,
        dropout=args.dropout,
        use_gradient_checkpointing=False,
    )
    model = HSKM(cfg)
    # Train with the numerically stable EMA: the upstream implementation
    # overflows float32 at >= 1024 tokens, and training at seq_len 512 is
    # already close to that boundary.
    patch_hskm_stable(model)
    return model, cfg


def build_gpt2(args, vocab: int, n_layer: int):
    from transformers import GPT2Config, GPT2LMHeadModel

    cfg = GPT2Config(
        vocab_size=vocab,
        n_positions=args.seq_len,
        n_embd=args.d_model,
        n_layer=n_layer,
        n_head=args.n_heads,
        resid_pdrop=args.dropout,
        embd_pdrop=args.dropout,
        attn_pdrop=args.dropout,
        use_cache=False,
    )
    model = GPT2LMHeadModel(cfg)
    return model, cfg


def forward_loss(model, x, y, kind, amp=False):
    if kind == "hskm":
        with torch.amp.autocast("cuda", enabled=amp, dtype=torch.float16):
            loss, _ = model(x, labels=y)
        return loss
    with torch.amp.autocast("cuda", enabled=amp, dtype=torch.float16):
        out = model(input_ids=x, labels=y)
    return out.loss


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_one(kind, args, train_x, train_y, val_x, val_y, vocab, out_dir, n_layer=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if kind == "hskm":
        model, cfg = build_hskm(args, vocab)
        cfg_dict = cfg.__dict__
    else:
        model, cfg = build_gpt2(args, vocab, n_layer)
        cfg_dict = cfg.to_dict()

    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters())

    steps_total = max(1, args.tokens // (args.batch_size * args.seq_len))
    warmup = max(10, int(0.03 * steps_total))
    opt = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1
    )

    def lr_at(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, steps_total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    print(
        f"\n[{kind}] params={n_params/1e6:.1f}M  steps={steps_total}  "
        f"batch={args.batch_size}x{args.seq_len}  tokens={steps_total*args.batch_size*args.seq_len:,}",
        flush=True,
    )

    model.train()
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step = 0
    t0 = time.time()
    tokens_seen = 0
    history = []
    log = []
    epoch = 0

    while step < steps_total:
        for xb, yb in iter_batches(train_x, train_y, args.batch_size, device=str(device),
                                   shuffle=True, seed=args.seed + epoch):
            if step >= steps_total:
                break
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            loss = forward_loss(model, xb, yb, kind, amp=amp)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()

            tokens_seen += xb.numel()
            l = loss.item()
            history.append(l)
            if step % max(1, steps_total // 50) == 0:
                el = time.time() - t0
                tps = tokens_seen / max(el, 1e-9)
                print(
                    f"  step {step:6d}/{steps_total}  loss {l:7.4f}  ppl {math.exp(min(l,20)):8.2f}  "
                    f"{tps:,.0f} tok/s  lr {sched.get_last_lr()[0]:.2e}",
                    flush=True,
                )
                log.append({"step": step, "loss": l, "tokens": tokens_seen, "tok_per_s": tps})
            step += 1
        epoch += 1

    # ---- validation ----
    model.eval()
    val_losses = []
    with torch.no_grad():
        n_val = min(200, val_x.shape[0] // args.batch_size)
        for xb, yb in iter_batches(val_x, val_y, args.batch_size, device=str(device),
                                   shuffle=False):
            val_losses.append(forward_loss(model, xb, yb, kind, amp=False).item())
            if len(val_losses) >= n_val:
                break
    val_loss = float(np.mean(val_losses))
    print(f"[{kind}] final val loss {val_loss:.4f} | ppl {math.exp(min(val_loss,20)):.2f}", flush=True)

    os.makedirs(out_dir, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "config": cfg_dict,
        "kind": kind,
        "params": n_params,
        "steps": steps_total,
        "tokens_seen": tokens_seen,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "val_loss": val_loss,
        "val_ppl": math.exp(min(val_loss, 20)),
        "train_loss_last100": float(np.mean(history[-100:])),
        "corpus": args.dataset,
        "amp": bool(amp),
        "wallclock_s": time.time() - t0,
    }
    ckpt_path = os.path.join(out_dir, f"{kind}.pt")
    torch.save(payload, ckpt_path)
    with open(os.path.join(out_dir, f"{kind}_train_log.json"), "w") as f:
        json.dump(log, f, indent=2)

    # Additionally save the GPT-2 control in the standard Hugging Face format
    # so it can be loaded with `from_pretrained` by the evaluation scripts.
    if kind == "gpt2":
        hf_dir = os.path.join(out_dir, "hskm-ours-gpt2-ctrl")
        model.save_pretrained(hf_dir)
        try:
            from transformers import AutoTokenizer

            AutoTokenizer.from_pretrained("openai-community/gpt2").save_pretrained(hf_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  !! could not save tokenizer: {type(e).__name__}: {e}")
        print(f"  saved HF-format control to {hf_dir}")

    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="both", choices=["hskm", "gpt2", "both"])
    ap.add_argument("--dataset", default="wikitext103",
                    choices=["wikitext103", "tinystories"])
    ap.add_argument("--tokens", type=int, default=100_000_000,
                    help="total training tokens (identical for both models)")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=8)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--d-medium", type=int, default=128)
    ap.add_argument("--n-kernels", type=int, default=64)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--n-patterns", type=int, default=4096)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--amp", action="store_true", default=True,
                    help="mixed precision (fp16 + GradScaler); ~2.6x faster on T4")
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    ap.add_argument("--also-matched-depth", action="store_true",
                    help="also train a GPT-2 with the same depth as HSKM")
    ap.add_argument("--ctrl-n-layer", type=int, default=0,
                    help="explicit depth for the GPT-2 control (0 = auto-match)")
    ap.add_argument("--out-dir", default=OUT_ROOT)
    ap.add_argument("--summary-name", default="controlled_training_summary.json",
                    help="filename under benchmarks/results/training/ (use distinct "
                         "names when training the two models on separate GPUs)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("=" * 78)
    print("  HSKM controlled from-scratch training")
    print("=" * 78)

    # ---- data ----
    train_tokens = tokenize_corpus("train", "train", dataset=args.dataset)
    val_tokens = tokenize_corpus("validation", "validation", dataset=args.dataset)
    train_x, train_y = make_blocks(train_tokens, args.seq_len)
    val_x, val_y = make_blocks(val_tokens, args.seq_len)
    print(f"[data] train blocks {train_x.shape}, val blocks {val_x.shape}")

    vocab = int(max(train_tokens.max(), val_tokens.max())) + 1
    vocab = 50259  # GPT-2 BPE + special tokens (matches upstream tokenizer.py)
    print(f"[data] vocab_size = {vocab}")

    os.makedirs(LOG_ROOT, exist_ok=True)
    summary = {}

    # The GPT-2 control is matched to HSKM on parameter count. HSKM's
    # long-term pattern bank carries a large parameter block that a
    # transformer of the same depth does not have, so we match capacity
    # rather than depth.
    hcfg = HSKMConfig(
        vocab_size=vocab, d_model=args.d_model, n_layers=args.n_layers,
        n_heads=args.n_heads, d_medium=args.d_medium, n_kernels=args.n_kernels,
        top_k=args.top_k, n_patterns=args.n_patterns, max_seq_len=args.seq_len,
        dropout=args.dropout,
    )
    h_params = hskm_param_count(hcfg)
    matched_layer, matched_params = find_matched_gpt2(
        h_params, vocab, args.d_model, args.n_heads
    )
    if args.ctrl_n_layer:
        matched_layer = args.ctrl_n_layer
        matched_params = gpt2_param_count(args.d_model, matched_layer, args.n_heads, vocab)
    print(f"[match] HSKM {h_params/1e6:.1f}M -> GPT-2 control n_layer={matched_layer} "
          f"({matched_params/1e6:.1f}M, {100*abs(matched_params-h_params)/h_params:.1f}% off)")

    out_dir = os.path.join(args.out_dir, "matched_params")

    if args.model in ("hskm", "both"):
        p = train_one("hskm", args, train_x, train_y, val_x, val_y, vocab, out_dir)
        summary["hskm"] = {k: p[k] for k in
                           ("params", "steps", "tokens_seen", "val_loss", "val_ppl")}

    if args.model in ("gpt2", "both"):
        p = train_one("gpt2", args, train_x, train_y, val_x, val_y, vocab,
                      out_dir, n_layer=matched_layer)
        summary["gpt2-matched-params"] = {
            k: p[k] for k in ("params", "steps", "tokens_seen", "val_loss", "val_ppl")
        }
        summary["gpt2-matched-params"]["n_layer"] = matched_layer

    if args.also_matched_depth:
        out_dir = os.path.join(args.out_dir, "matched_depth")
        p = train_one("gpt2", args, train_x, train_y, val_x, val_y, vocab,
                      out_dir, n_layer=args.n_layers)
        summary["gpt2-matched-depth"] = {
            k: p[k] for k in ("params", "steps", "tokens_seen", "val_loss", "val_ppl")
        }
        summary["gpt2-matched-depth"]["n_layer"] = args.n_layers

    with open(os.path.join(LOG_ROOT, args.summary_name), "w") as f:
        json.dump(summary, f, indent=2)
    print("\n" + json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
