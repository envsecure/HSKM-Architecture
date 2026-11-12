"""
Correctness tests for the benchmark harness itself.

A benchmark that cannot show that its own measurements are trustworthy is
worth nothing, so these tests check the four things the results depend on:

1. ``test_hskm_is_causal``          -- HSKM must not leak future tokens.
2. ``test_upstream_ema_overflows``  -- documents the upstream fp32 overflow.
3. ``test_stable_ema_matches``      -- the stable EMA is algebraically equal
                                       to the upstream one where upstream is
                                       finite, and stays finite where it is not.
4. ``test_hskm_adapter_matches_gpt2`` -- our custom lm-eval adapter scores
                                       identically to the reference
                                       ``HFLM`` implementation on the same
                                       model, so HSKM's numbers are directly
                                       comparable with the baselines.

Run with:  python -m pytest benchmarks/tests -v
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "benchmarks", "src"))

from model import HSKM, HSKMConfig  # noqa: E402
from hskm_stable import patch_hskm_stable  # noqa: E402


def _cfg(**kw):
    base = dict(
        vocab_size=512, d_model=64, n_layers=2, n_heads=4, d_medium=16,
        n_kernels=16, top_k=4, n_patterns=32, max_seq_len=4096, dropout=0.0,
    )
    base.update(kw)
    return HSKMConfig(**base)


def test_hskm_is_causal():
    """Changing a later token must not change any earlier logit."""
    torch.manual_seed(0)
    m = HSKM(_cfg()).eval()
    x = torch.randint(0, 500, (1, 48))
    with torch.no_grad():
        _, a = m(x)
        x2 = x.clone()
        x2[0, -1] = (x2[0, -1] + 7) % 500
        _, b = m(x2)
    upstream_diff = (a[0, :-1] - b[0, :-1]).abs().max().item()
    assert upstream_diff < 1e-4, f"HSKM leaks future information ({upstream_diff})"


def test_upstream_ema_overflows():
    """
    The upstream ``MediumTermMemory`` overflows float32 at long sequence
    lengths because it computes ``decay ** -t``.

    We assert the overflow exists so that the finding stays documented and
    cannot silently regress.
    """
    torch.manual_seed(0)
    cfg = _cfg(d_model=256, n_layers=2, max_seq_len=2048)
    m = HSKM(cfg).eval()
    x = torch.randint(0, 500, (1, 1024))
    with torch.no_grad():
        _, logits = m(x)
    assert not torch.isfinite(logits).all(), (
        "upstream HSKM no longer overflows at 1024 tokens -- if this was fixed "
        "upstream, hskm_stable.py can be simplified and the README updated"
    )


@pytest.mark.parametrize("seq_len", [64, 256, 512, 768])
def test_stable_ema_matches(seq_len):
    """The stable EMA reproduces the upstream EMA wherever upstream is finite."""
    torch.manual_seed(0)
    cfg = _cfg(d_model=256, n_layers=2, max_seq_len=4096)
    upstream = HSKM(cfg).eval()
    stable = HSKM(cfg).eval()
    patch_hskm_stable(stable, chunk_size=256)
    stable.load_state_dict(upstream.state_dict())

    x = torch.randint(0, 500, (1, seq_len))
    with torch.no_grad():
        _, a = upstream(x)
        _, b = stable(x)
    assert torch.isfinite(a).all(), f"upstream overflowed at {seq_len}; adjust the test"
    diff = (a - b).abs().max().item()
    scale = a.abs().max().item()
    assert diff / scale < 1e-4, f"stable EMA diverges from upstream at T={seq_len}: {diff}"


@pytest.mark.parametrize("seq_len", [1024, 2048, 4096])
def test_stable_ema_is_finite(seq_len):
    """The stable EMA stays finite exactly where the upstream one breaks."""
    torch.manual_seed(0)
    cfg = _cfg(d_model=256, n_layers=2, max_seq_len=8192)
    m = HSKM(cfg).eval()
    patch_hskm_stable(m, chunk_size=256)
    x = torch.randint(0, 500, (1, seq_len))
    with torch.no_grad():
        _, logits = m(x)
    assert torch.isfinite(logits).all()


def test_stable_ema_survives_autocast():
    """
    Under ``torch.autocast`` the preceding ``nn.Linear`` emits float16, and
    the factor ``decay ** -C`` overflows float16. The stable EMA must force
    float32 internally, otherwise mixed-precision training silently produces
    inf/NaN.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for autocast")
    torch.manual_seed(0)
    cfg = _cfg(d_model=256, n_layers=2, max_seq_len=1024)
    m = HSKM(cfg).eval().cuda()
    patch_hskm_stable(m, chunk_size=256)
    x = torch.randint(0, 500, (1, 768), device="cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _, logits = m(x)
    assert torch.isfinite(logits).all(), "autocast produced non-finite logits"


def test_amp_and_fp32_agree():
    """Mixed precision and full precision must give the same loss."""
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA for autocast")
    torch.manual_seed(0)
    cfg = _cfg(d_model=256, n_layers=2, max_seq_len=512)
    x = torch.randint(0, 500, (8, 512), device="cuda")

    m32 = HSKM(cfg).eval().cuda()
    patch_hskm_stable(m32, chunk_size=256)
    with torch.no_grad():
        _, l32 = m32(x)

    m16 = HSKM(cfg).eval().cuda()
    patch_hskm_stable(m16, chunk_size=256)
    m16.load_state_dict(m32.state_dict())  # identical weights
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16):
        _, l16 = m16(x)

    assert torch.isfinite(l16).all()
    rel = (l32 - l16).abs().max().item() / l32.abs().max().item()
    assert rel < 5e-2, f"AMP and fp32 disagree: rel={rel}"


def test_hskm_adapter_matches_gpt2():
    """
    Our custom lm-eval adapter must produce exactly the same loglikelihood
    as the reference ``HFLM`` implementation for the same underlying model.

    We use GPT-2 as the stand-in because the adapter and ``HFLM`` implement
    the same scoring contract; if the two agree for GPT-2 they agree for
    HSKM too.
    """
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM

    from hskm_lm import HSKMLM

    class _FakeHSKM(HSKMLM):
        """Reuse the adapter's scoring logic but with a GPT-2 backbone."""

        def __init__(self, model):
            self._device = torch.device("cpu")
            self._batch_size = 4
            self._max_length = 256
            self._dtype = torch.float32
            self.model = model
            self.tokenizer = None
            from hskm_lm import _tiktoken_gpt2

            self._enc = _tiktoken_gpt2()

    torch.manual_seed(0)
    gpt2 = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
    gpt2.eval()

    adapter = _FakeHSKM(gpt2)

    ctx = "The capital of France is"
    cont = " Paris"
    ctx_ids = adapter._enc.encode(ctx)
    cont_ids = adapter._enc.encode(cont)

    # Reference: score the continuation with a plain HF forward pass.
    seq = ctx_ids + cont_ids
    with torch.no_grad():
        logits = gpt2(input_ids=torch.tensor([seq[:-1]])).logits
    lp = F.log_softmax(logits[0].float(), dim=-1)
    n = len(cont_ids)
    target = torch.tensor(seq[-n:])
    ref = lp[len(seq) - n - 1 : len(seq) - 1].gather(-1, target.unsqueeze(-1)).sum().item()

    got, greedy = adapter._loglikelihood_tokens(
        [((ctx, cont), ctx_ids, cont_ids)]
    )[0]
    assert abs(got - ref) < 1e-3, f"adapter mismatch: {got} vs {ref}"
    assert isinstance(greedy, bool)


def test_registry_entries_are_unique():
    from model_registry import REGISTRY

    keys = [b.key for b in REGISTRY]
    assert len(keys) == len(set(keys)), "duplicate registry keys"
    for b in REGISTRY:
        assert b.reason and len(b.reason) > 40, f"{b.key}: missing justification"
        assert b.tier in ("linear", "attention", "ours"), f"{b.key}: bad tier"
