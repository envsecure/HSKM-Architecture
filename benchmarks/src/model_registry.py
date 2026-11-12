"""
Single source of truth for every model in the HSKM benchmark suite.

Each entry records *why* the model is a suitable baseline, which tier it
belongs to, and how to load it (some legacy checkpoints need special
handling -- see ``load_model``).

Tiers
-----
``linear``    Architecturally closest peers: other *linear-time* sequence
              models (state-space / recurrent). These test HSKM's central
              O(N) claim head-on.
``attention`` Attention-based decoder LMs at matched parameter count. These
              test whether HSKM's novel memory hierarchy buys anything over
              standard causal attention at the same size.
``ours``      Models we train from scratch under a *controlled* budget so the
              quality comparison isolates architecture from pretraining data.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

import torch

# --------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------

@dataclass
class Baseline:
    key: str                       # short identifier used in tables/plots
    hf_id: Optional[str]           # Hugging Face repo id (None for our models)
    family: str
    tier: str
    params_m: float                # approximate parameter count (millions)
    reason: str                    # why this is a suitable baseline
    loader: str = "hf"             # hf | legacy_mamba | legacy_mamba2 | ours_hskm
    max_length: int = 2048         # max context used for loglikelihood scoring
    dtype: str = "float16"
    tokenizer_id: Optional[str] = None  # override when the repo ships no tokenizer
    notes: str = ""


REGISTRY: list[Baseline] = [
    # ------------------------------------------------------------------
    # Tier 1 -- Linear-time sequence models (the true architectural peers)
    # ------------------------------------------------------------------
    Baseline(
        key="mamba-130m",
        hf_id="state-spaces/mamba-130m",
        family="Mamba (selective SSM)",
        tier="linear",
        params_m=130,
        loader="legacy_mamba",
        max_length=2048,
        tokenizer_id="EleutherAI/gpt-neox-20b",
        reason=(
            "The canonical linear-time language model. HSKM's central claim is "
            "O(N) sequence modelling without quadratic attention; Mamba is the "
            "reference implementation of that idea and the paper it must be "
            "measured against. Matched at ~130M parameters so the comparison "
            "is at equal capacity."
        ),
    ),
    Baseline(
        key="mamba2-130m",
        hf_id="state-spaces/mamba2-130m",
        family="Mamba-2 (SSM v2)",
        tier="linear",
        params_m=130,
        loader="legacy_mamba2",
        max_length=2048,
        tokenizer_id="EleutherAI/gpt-neox-20b",
        reason=(
            "Mamba-2 is the current state of the art for linear-time sequence "
            "modelling, replacing the selective SSM with a structured SSM that "
            "is faster and more hardware-efficient. At the same 130M size it "
            "isolates whether HSKM's advantage holds against the newer variant."
        ),
    ),
    Baseline(
        key="mamba2-1.3b",
        hf_id="state-spaces/mamba2-1.3b",
        family="Mamba-2 (SSM v2)",
        tier="linear",
        params_m=1300,
        loader="legacy_mamba2",
        max_length=2048,
        tokenizer_id="EleutherAI/gpt-neox-20b",
        dtype="float16",
        reason=(
            "Scale point for the linear-time family. Shows whether HSKM's "
            "architecture is competitive once models are large enough that "
            "quality (not just efficiency) dominates."
        ),
    ),
    Baseline(
        key="mamba-2.8b",
        hf_id="state-spaces/mamba-2.8b-hf",
        family="Mamba (selective SSM)",
        tier="linear",
        params_m=2800,
        loader="hf",
        max_length=2048,
        notes="Released in native transformers format (has model_type).",
        reason=(
            "Large-scale reference for the SSM family, and the release that "
            "made Mamba available in the transformers `MambaForCausalLM` "
            "implementation -- so it also validates that our custom loader for "
            "the legacy 130M checkpoints is faithful."
        ),
    ),

    # ------------------------------------------------------------------
    # Tier 2 -- Attention-based decoder LMs at matched scale
    # ------------------------------------------------------------------
    Baseline(
        key="gpt2",
        hf_id="openai-community/gpt2",
        family="GPT-2 (causal Transformer)",
        tier="attention",
        params_m=124,
        max_length=1024,
        reason=(
            "HSKM is a GPT-2-shaped decoder (RMSNorm, SwiGLU, RoPE, tied "
            "embeddings, GPT-2 BPE vocabulary of 50257). GPT-2 is its direct "
            "architectural ancestor and the reference point for 'what does a "
            "standard causal transformer achieve at this size'. It is also the "
            "only baseline that shares HSKM's tokenizer exactly, so token-level "
            "perplexities are directly comparable without any re-tokenisation."
        ),
    ),
    Baseline(
        key="opt-125m",
        hf_id="facebook/opt-125m",
        family="OPT (causal Transformer)",
        tier="attention",
        params_m=125,
        max_length=2048,
        reason=(
            "Same size, same GPT-2 BPE tokenizer family, but with learned "
            "absolute position embeddings instead of RoPE. Isolates the "
            "contribution of HSKM's RoPE-in-kernel-attention design against a "
            "contemporary absolute-PE decoder."
        ),
    ),
    Baseline(
        key="pythia-160m",
        hf_id="EleutherAI/pythia-160m",
        family="GPT-NeoX (causal Transformer)",
        tier="attention",
        params_m=160,
        max_length=2048,
        reason=(
            "Modern open causal LM trained on the Pile. Uses rotary position "
            "embeddings and parallel residual attention like HSKM, so it is the "
            "closest *pretrained* relative of HSKM's design at a comparable "
            "parameter count."
        ),
    ),
    Baseline(
        key="smollm2-135m",
        hf_id="HuggingFaceTB/SmolLM2-135M",
        family="Llama (causal Transformer)",
        tier="attention",
        params_m=135,
        max_length=2048,
        reason=(
            "A modern (2024) small decoder trained on 2T tokens. Same order of "
            "magnitude as the matched-scale peers but far better trained, so it "
            "shows what the *current* state of the art achieves at 135M "
            "parameters -- the number HSKM would have to reach to be worth "
            "deploying."
        ),
    ),
    Baseline(
        key="pythia-410m",
        hf_id="EleutherAI/pythia-410m",
        family="GPT-NeoX (causal Transformer)",
        tier="attention",
        params_m=410,
        max_length=2048,
        reason=(
            "Scale point for the Pile-trained attention family, letting us to "
            "plot quality against parameter count for HSKM's closest "
            "pretrained relatives."
        ),
    ),
    Baseline(
        key="smollm2-360m",
        hf_id="HuggingFaceTB/SmolLM2-360M",
        family="Llama (causal Transformer)",
        tier="attention",
        params_m=360,
        max_length=2048,
        reason=(
            "Modern scale point. Pairs with SmolLM2-135M to show the quality "
            "trajectory of a well-trained modern decoder, against which the "
            "from-scratch HSKM checkpoints are evaluated."
        ),
    ),
    Baseline(
        key="opt-350m",
        hf_id="facebook/opt-350m",
        family="OPT (causal Transformer)",
        tier="attention",
        params_m=350,
        max_length=2048,
        reason=(
            "Second size point for the OPT family, giving a controlled "
            "scaling curve (125M -> 350M) within a single architecture so that "
            "size effects are not confounded with architecture changes."
        ),
    ),
    Baseline(
        key="qwen2.5-0.5b",
        hf_id="Qwen/Qwen2.5-0.5B",
        family="Qwen2 (causal Transformer)",
        tier="attention",
        params_m=494,
        max_length=2048,
        reason=(
            "Strongest small-scale open decoder in its class. Serves as the "
            "quality ceiling for 'what a modern architecture delivers at <0.5B "
            "parameters' that HSKM's O(N) efficiency has to be weighed against."
        ),
    ),

    # ------------------------------------------------------------------
    # Tier 3 -- Controlled from-scratch models (ours)
    # ------------------------------------------------------------------
    Baseline(
        key="hskm-ours",
        hf_id=None,
        family="HSKM (ours, from scratch)",
        tier="ours",
        params_m=0,  # filled in from the checkpoint
        loader="ours_hskm",
        max_length=512,
        dtype="float32",
        reason=(
            "HSKM trained by us under a controlled budget (identical corpus, "
            "token count, optimiser and schedule as the controlled GPT-2 "
            "baseline). This is the only configuration in which the "
            "architecture comparison is apples-to-apples, because every other "
            "model here is pretrained on orders of magnitude more data."
        ),
    ),
    Baseline(
        key="gpt2-ours",
        hf_id=None,
        family="GPT-2 (ours, from scratch)",
        tier="ours",
        params_m=0,
        loader="ours_gpt2",
        max_length=512,
        dtype="float32",
        reason=(
            "A standard causal GPT-2 trained by us under the *identical* "
            "budget as the HSKM checkpoint (same data, same number of tokens, "
            "same optimiser, same schedule). Any difference in held-out "
            "perplexity between this and `hskm-ours` is attributable to the "
            "architecture alone."
        ),
    ),
]

BY_KEY = {b.key: b for b in REGISTRY}


# --------------------------------------------------------------------------
# Loaders
# --------------------------------------------------------------------------

def _hf_snapshot(model_id: str) -> str:
    """Locate a locally cached snapshot of ``model_id``."""
    from huggingface_hub import snapshot_download

    return snapshot_download(
        model_id,
        allow_patterns=[
            "*.json", "*.txt", "*.model", "*.safetensors",
            "*.bin", "*.tiktoken", "*.py",
        ],
    )


def _build_legacy_mamba(model_id: str, family: str):
    """
    Load a *legacy* ``state-spaces`` checkpoint.

    Those repos predate the transformers hub format: their ``config.json``
    has no ``model_type`` key, so ``AutoConfig`` rejects them, and their
    weight names use the original Mamba implementation
    (``backbone.embedding.weight``) rather than the transformers one
    (``backbone.embeddings.weight``).

    We therefore rebuild the config explicitly and rename the embedding key.
    Everything else matches one-to-one. The rename is verified by
    ``tests/test_model_registry.py`` which asserts that *every* parameter
    receives a value from the checkpoint (no missing / unexpected keys).
    """
    from transformers import (
        MambaConfig, MambaForCausalLM, Mamba2Config, Mamba2ForCausalLM,
    )

    path = _hf_snapshot(model_id)
    raw = json.load(open(os.path.join(path, "config.json")))

    bin_path = os.path.join(path, "pytorch_model.bin")
    state = torch.load(bin_path, map_location="cpu", weights_only=True)

    # The vocabulary is padded to a multiple of 8/16 in the checkpoint.
    vocab_size = state["backbone.embedding.weight"].shape[0]
    hidden = raw["d_model"]
    layers = raw["n_layer"]

    if family == "mamba":
        cfg = MambaConfig(
            hidden_size=hidden, num_hidden_layers=layers, vocab_size=vocab_size
        )
        cls = MambaForCausalLM
    else:
        cfg = Mamba2Config(
            hidden_size=hidden, num_hidden_layers=layers, vocab_size=vocab_size
        )
        cls = Mamba2ForCausalLM

    model = cls(cfg)

    # Rename the one key that differs from the transformers naming scheme.
    renamed = {}
    for k, v in state.items():
        k2 = k.replace("backbone.embedding.weight", "backbone.embeddings.weight")
        renamed[k2] = v

    missing, unexpected = model.load_state_dict(renamed, strict=False)
    missing = [m for m in missing if "masked_bias" not in m]
    unexpected = [u for u in unexpected if "masked_bias" not in u]
    if missing or unexpected:
        raise RuntimeError(
            f"{model_id}: state dict mismatch. missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )

    model.config.vocab_size = vocab_size
    return model


def load_model(baseline: Baseline, device="cuda", dtype=torch.float16, checkpoint=None):
    """
    Instantiate a model for a :class:`Baseline`.

    ``checkpoint`` is used for the ``ours`` tier (path to a local checkpoint).
    """
    if baseline.loader == "legacy_mamba":
        model = _build_legacy_mamba(baseline.hf_id, "mamba")
    elif baseline.loader == "legacy_mamba2":
        model = _build_legacy_mamba(baseline.hf_id, "mamba2")
    elif baseline.loader == "ours_hskm":
        from hskm_lm import load_hskm_checkpoint

        model = load_hskm_checkpoint(checkpoint)
        dtype = torch.float32
    elif baseline.loader == "ours_gpt2":
        from transformers import GPT2LMHeadModel

        model = GPT2LMHeadModel.from_pretrained(checkpoint, torch_dtype=dtype)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            baseline.hf_id, torch_dtype=dtype, trust_remote_code=False
        )

    model = model.to(device)
    model.eval()
    return model


def count_params(model) -> float:
    """Parameter count in millions (unique parameters, so tied weights count once)."""
    seen = set()
    total = 0
    for p in model.parameters():
        if id(p) in seen:
            continue
        seen.add(id(p))
        total += p.numel()
    return total / 1e6


def registry_as_json(path: str) -> None:
    """Dump the registry (with live parameter counts) to JSON."""
    out = []
    for b in REGISTRY:
        d = asdict(b)
        out.append(d)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    import sys

    registry_as_json(sys.argv[1] if len(sys.argv) > 1 else "model_registry.json")
    for b in REGISTRY:
        print(f"{b.key:16s} {b.tier:10s} {b.params_m:7.0f}M  {b.family}")
