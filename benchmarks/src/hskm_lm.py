"""
lm-evaluation-harness adapter for HSKM.

HSKM is not a ``transformers`` model, so it cannot be wrapped by
``lm_eval.models.huggingface.HFLM``. This module provides a ``TemplateLM``
implementation that exposes HSKM to the harness with the exact same
loglikelihood semantics as ``HFLM``, so that every number produced here is
directly comparable with the numbers produced for the Hugging Face baselines.

Two implementation details are worth calling out:

1. **Right padding, never left padding.**
   HSKM's ``MediumTermMemory`` is a causal EMA over the token axis, so the
   state at position ``t`` depends on every token before it. Left-padding a
   batch would shift every real token to a different position *and* inject
   padding into the EMA, changing the result. Right-padding is safe: the
   padded positions come *after* every position we score, and the EMA at a
   scored position only ever sees real tokens.

2. **The numerically stable EMA is used by default.**
   Upstream HSKM overflows float32 for sequences >= 1024 tokens (see
   ``hskm_stable.py``). ``stable=True`` swaps in the segmented-scan EMA so
   long-context evaluation is numerically sound. Set ``stable=False`` to
   reproduce the upstream behaviour exactly (including its NaNs).
"""

from __future__ import annotations

import os
import sys
from typing import Optional

import torch
import torch.nn.functional as F

# Allow `import model` (repo root) and `import hskm_stable` (this dir).
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
for p in (_ROOT, _HERE):
    if p not in sys.path:
        sys.path.insert(0, p)

from lm_eval.api.model import TemplateLM  # noqa: E402

from hskm_stable import patch_hskm_stable  # noqa: E402

GPT2_EOT = 50256  # tiktoken gpt2 <|endoftext|>


def load_hskm_checkpoint(path: str):
    """Load an HSKM checkpoint saved by ``benchmarks/scripts/10_train_controlled.py``."""
    from model import HSKM, HSKMConfig

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = HSKMConfig(**ckpt["config"])
    model = HSKM(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


class HSKMLM(TemplateLM):
    """lm-evaluation-harness wrapper around an HSKM checkpoint."""

    def __init__(
        self,
        checkpoint: str,
        device: str = "cuda",
        batch_size: int = 8,
        max_length: int = 512,
        stable: bool = True,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self._device = torch.device(device)
        self._batch_size = batch_size
        self._max_length = max_length
        self._dtype = dtype

        self.model = load_hskm_checkpoint(checkpoint)
        if stable:
            patch_hskm_stable(self.model)
        self.model = self.model.to(self._device).to(self._dtype)
        self.model.eval()

        self._enc = _tiktoken_gpt2()
        self.tokenizer = None  # no HF tokenizer object exists for HSKM
        self._max_length = min(self._max_length, self.model.config.max_seq_len)

    # -- properties required by the harness -----------------------------
    @property
    def eot_token_id(self) -> int:
        return GPT2_EOT

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return 256

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self):
        return self._device

    def tok_encode(self, string: str, add_special_tokens: Optional[bool] = None, **kwargs):
        """
        GPT-2 style: *no* special tokens are added.

        This deliberately matches ``HFLM``'s behaviour for ``openai-community/gpt2``
        so that token-level perplexities are directly comparable -- both models
        score the identical token sequence.
        """
        return self._enc.encode(string)

    # -- core scoring ---------------------------------------------------
    def _forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Run the model and return raw logits.

        HSKM's ``model(...)`` returns ``(loss, logits)``; a ``transformers``
        causal LM returns an object with a ``.logits`` attribute. Supporting
        both keeps this scoring path identical to ``HFLM``'s, which is what
        makes the two directly comparable (and is what the adapter test
        checks).
        """
        out = self.model(input_ids)
        if isinstance(out, tuple):
            return out[1]
        if hasattr(out, "logits"):
            return out.logits
        return out

    def _loglikelihood_tokens(self, requests, disable_tqdm: bool = False, **kwargs):
        """
        Score a batch of ``(context_enc, continuation_enc)`` pairs.

        Returns one ``(total_logprob, is_greedy)`` tuple per request, matching
        the contract of ``HFLM._loglikelihood_tokens``.
        """
        res: list[tuple[float, bool]] = []

        # Sort by descending total length so that batches are homogeneous in
        # size (less padding) and so that an OOM shows up immediately.
        prepared = []
        for req in requests:
            _, context_enc, continuation_enc = req
            context_enc = list(context_enc)
            continuation_enc = list(continuation_enc)

            # Keep the continuation intact; drop context from the left.
            budget = self._max_length - len(continuation_enc)
            if budget < 1:
                continuation_enc = continuation_enc[: self._max_length - 1]
                budget = 1
            if len(context_enc) > budget:
                context_enc = context_enc[-budget:]

            seq = context_enc + continuation_enc
            prepared.append((seq, len(continuation_enc)))

        prepared.sort(key=lambda x: -len(x[0]))

        for start in range(0, len(prepared), self._batch_size):
            chunk = prepared[start : start + self._batch_size]
            res.extend(self._score_chunk(chunk))

        # Restore the original request order.
        return res

    def _score_chunk(self, chunk):
        """Score one padded batch. Right padding (see module docstring)."""
        seqs = [s for s, _ in chunk]
        n_conts = [n for _, n in chunk]
        max_len = max(len(s) for s in seqs)

        pad_id = GPT2_EOT
        inp = torch.full((len(seqs), max_len - 1), pad_id, dtype=torch.long, device=self._device)

        for i, s in enumerate(seqs):
            # inp holds seq[:-1]; the final token is only ever a target.
            inp[i, : len(s) - 1] = torch.tensor(s[:-1], dtype=torch.long, device=self._device)

        with torch.inference_mode():
            logits = self._forward_logits(inp)

        out = []
        for i, (s, n_cont) in enumerate(zip(seqs, n_conts)):
            L = len(s)
            # Position p predicts token p+1. The continuation tokens are
            # s[L-n_cont : L], so the predicting logits are at
            # s-indices [L-n_cont-1 : L-1].
            sl = logits[i, L - n_cont - 1 : L - 1, :].float()
            targets = torch.tensor(
                s[L - n_cont : L], dtype=torch.long, device=self._device
            )
            logprobs = F.log_softmax(sl, dim=-1)
            gathered = logprobs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            greedy = bool((logprobs.argmax(dim=-1) == targets).all().item())
            out.append((gathered.sum().item(), greedy))

        return out

    def loglikelihood_rolling(self, requests, disable_tqdm: bool = False, **kwargs):
        """
        Total loglikelihood of each string under the model.

        Used by the perplexity-style tasks (e.g. ``wikitext``). The string is
        scored in overlapping windows so that sequences longer than
        ``max_length`` are still handled.
        """
        res = []
        for req in requests:
            string = req.args
            ids = self._enc.encode(string)
            if len(ids) < 2:
                res.append(0.0)
                continue

            total = 0.0
            window = self._max_length
            stride = window - 1
            for start in range(0, len(ids) - 1, stride):
                end = min(start + window, len(ids))
                chunk = ids[start:end]
                if len(chunk) < 2:
                    break
                inp = torch.tensor(chunk[:-1], dtype=torch.long, device=self._device).unsqueeze(0)
                with torch.inference_mode():
                    logits = self._forward_logits(inp)
                lp = F.log_softmax(logits[0].float(), dim=-1)
                tgt = torch.tensor(chunk[1:], dtype=torch.long, device=self._device)
                total += lp.gather(-1, tgt.unsqueeze(-1)).sum().item()
                if end == len(ids):
                    break
            res.append(total)
        return res

    def generate_until(self, requests, disable_tqdm: bool = False, **kwargs):
        """Greedy / sampled generation (used by a handful of tasks)."""
        res = []
        for req in requests:
            ctx, kwargs_ = req.args
            until = kwargs_.get("until", [])
            max_gen = kwargs_.get("max_gen_toks", self.max_gen_toks)
            ids = self._enc.encode(ctx)
            inp = torch.tensor(ids, dtype=torch.long, device=self._device).unsqueeze(0)
            with torch.inference_mode():
                out = self.model.generate(inp, max_new_tokens=max_gen)
            text = self._enc.decode(out[0, len(ids) :].tolist())
            for stop in until:
                if stop and stop in text:
                    text = text.split(stop)[0]
            res.append(text)
        return res


def _tiktoken_gpt2():
    import tiktoken

    return tiktoken.get_encoding("gpt2")
