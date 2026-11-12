"""
Numerically-stable variant of HSKM's MediumTermMemory.

Why this exists
---------------
The upstream ``model.MediumTermMemory`` evaluates the EMA recurrence

    h_t = decay * h_{t-1} + (1 - decay) * x_t

using the "geometric power" parallel-scan trick:

    h_scaled_t = x_t * decay ** (-t)
    h_sum_t    = cumsum(h_scaled)
    h_ema_t    = h_sum_t * decay ** t * (1 - decay)

This is mathematically exact but **numerically unstable in float32**: the
factor ``decay ** (-t)`` grows without bound. For the default ``decay = 0.9``:

    t =  512 -> 0.9 ** -512  ~ 2.2e23   (still inside float32 range)
    t = 1024 -> 0.9 ** -1024 ~ 4.6e46   (overflows float32 -> inf -> NaN)

We verified empirically that upstream HSKM emits NaN logits for sequence
lengths >= 1024 (see ``tests/test_numerics.py``). That directly contradicts
the project's long-context/O(N) claim, so every long-context number in this
benchmark suite is produced with the stable implementation below.

Method
------
The recurrence is a first-order linear filter, so it can be evaluated
*exactly* with a segmented scan:

1. Split the sequence into chunks of ``C`` steps.
2. Within a chunk, the local EMA is computed with the same power trick --
   but now the exponent is bounded by ``C`` instead of ``T``, so
   ``decay ** -C`` stays well inside float32 range.
3. Chunk-local EMA values are then combined with a sequential scan over the
   (tiny) chunk axis:

       carry_0 = 0
       carry_{j+1} = decay ** C * carry_j + local_ema_j[-1]

   and every position inside chunk ``j`` receives the carry of chunk ``j-1``
   propagated through the remaining steps of the chunk.

This is algebraically identical to the unbounded recurrence, but the largest
intermediate magnitude is ``O(decay ** -C)`` instead of ``O(decay ** -T)``.

Correctness is asserted in ``tests/test_numerics.py``: for sequence lengths
where the upstream implementation is still finite, the two agree to within
float32 round-off.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from model import HSKMConfig, MediumTermMemory

__all__ = ["StableMediumTermMemory", "patch_hskm_stable", "stable_hskm"]


class StableMediumTermMemory(MediumTermMemory):
    """
    Drop-in replacement for :class:`model.MediumTermMemory` that computes the
    *same* EMA recurrence using a segmented scan.

    Args:
        config: HSKM configuration (``d_model``, ``d_medium``, ``mtm_decay``).
        chunk_size: maximum chunk length ``C``. ``C = 256`` keeps
            ``decay ** -C ~ 4.9e11`` for ``decay = 0.9`` -- comfortably inside
            float32 range while retaining good GPU utilisation.
    """

    def __init__(self, config: HSKMConfig, chunk_size: int = 256):
        super().__init__(config)
        self.chunk_size = chunk_size

    def forward(self, x: Tensor) -> Tensor:
        B, T, _ = x.shape
        h = self.in_proj(x)  # [B, T, D_m]

        # The scan is always evaluated in float32.
        #
        # Under ``torch.autocast`` the preceding ``nn.Linear`` produces a
        # float16 tensor, and the factor ``decay ** -C`` (4.9e11 for
        # ``decay = 0.9``, ``C = 256``) overflows float16 (max 65504) --
        # silently producing inf/NaN. Forcing float32 here keeps the
        # computation exact under mixed precision.
        out_dtype = h.dtype
        h = h.float()

        decay = self.decay
        C = self.chunk_size

        if T <= C:
            # Single chunk: identical to the upstream formulation, but the
            # exponent is bounded by T <= C so it cannot overflow.
            return self.out_proj(self._chunk_local_ema(h).to(out_dtype))

        n_full = T // C
        tail = T - n_full * C

        # Pad the tail so that the sequence length becomes an exact multiple
        # of C. The amount of padding is ``C - tail`` (not ``tail``): we need
        # to round the length *up* to the next chunk boundary.
        #
        # Zero padding contributes nothing to the EMA. It does advance the
        # decay clock for the padded chunk, but that chunk's output is
        # discarded, so no correction is needed.
        if tail > 0:
            pad_len = C - tail
            h = torch.cat([h, h.new_zeros(B, pad_len, h.shape[-1])], dim=1)
            T_pad = h.shape[1]
        else:
            T_pad = T

        n_chunks = T_pad // C
        assert n_chunks * C == T_pad, "chunking does not tile the sequence"
        chunks = h.view(B, n_chunks, C, h.shape[-1])  # [B, n_chunks, C, D_m]

        # 1) Local EMA inside every chunk (bounded exponents).
        local = self._chunk_local_ema(chunks)  # [B, n_chunks, C, D_m]

        # 2) Sequential combine across chunks.
        #    carry_j = EMA state entering chunk j.
        dec = decay ** C
        local_last = local[:, :, -1, :]  # [B, n_chunks, D_m]

        # exclusive prefix combination:
        #   carry_{j+1} = dec * carry_j + local_last_j
        # With carry_0 = 0 this is a simple cumulative recurrence.
        carry_in = h.new_zeros(B, local.shape[-1])  # [B, D_m], carry_0 = 0
        carries = []
        c = carry_in
        for j in range(n_chunks):
            carries.append(c)
            c = dec * c + local_last[:, j, :]
        carries = torch.stack(carries, dim=1)  # [B, n_chunks, D_m]

        # 3) Expand the incoming carry over the positions of its chunk.
        #
        #    With l_s = EMA of the chunk starting from a ZERO state and
        #    carry_j = state entering chunk j, linearity of the recurrence
        #    gives the exact true state at local step s:
        #
        #        h_s = l_s + decay ** (s + 1) * carry_j
        #
        #    (the +1 is because the incoming state is itself decayed once
        #    before it can influence the first output step).
        steps = torch.arange(C, device=h.device, dtype=h.dtype)
        decay_pow_pos = self.decay ** (steps + 1)  # [C]
        expanded = carries.unsqueeze(2) * decay_pow_pos.view(1, 1, C, 1)

        out_chunks = expanded + local  # [B, n_chunks, C, D_m]
        out = out_chunks.reshape(B, T_pad, h.shape[-1])

        if tail > 0:
            # Drop the zero-padded tail: it contributed nothing to the EMA.
            out = out[:, :T, :]

        return self.out_proj(out.to(out_dtype))

    def _chunk_local_ema(self, h: Tensor) -> Tensor:
        """EMA within each chunk; exponents bounded by the chunk length."""
        *lead, C, D = h.shape
        h_flat = h.reshape(-1, C, D)
        steps = torch.arange(C, device=h.device, dtype=h.dtype)
        decay_pow = self.decay ** steps
        h_scaled = h_flat * (self.decay ** -steps).view(1, C, 1)
        h_sum = torch.cumsum(h_scaled, dim=1)
        ema = h_sum * decay_pow.view(1, C, 1) * (1 - self.decay)
        return ema.view(*lead, C, D)


def decay_pow_tensor(decay: float, steps: Tensor) -> Tensor:
    """``decay ** steps`` computed in float32."""
    return torch.pow(torch.as_tensor(decay, device=steps.device, dtype=steps.dtype), steps)


def patch_hskm_stable(model: nn.Module, chunk_size: int = 256) -> nn.Module:
    """
    Replace every :class:`model.MediumTermMemory` inside ``model`` with
    :class:`StableMediumTermMemory`, in-place.

    The replacement re-uses the *original* module's parameters (``in_proj``,
    ``out_proj``) so the state dict stays fully compatible -- a checkpoint
    trained with the upstream implementation loads unchanged.
    """
    for name, child in list(model.named_children()):
        if isinstance(child, MediumTermMemory) and not isinstance(child, StableMediumTermMemory):
            new = StableMediumTermMemory.__new__(StableMediumTermMemory)
            nn.Module.__init__(new)
            new.d_model = child.d_model
            new.d_medium = child.d_medium
            new.decay = child.decay
            new.chunk_size = chunk_size
            new.in_proj = child.in_proj
            new.out_proj = child.out_proj
            setattr(model, name, new)
        else:
            patch_hskm_stable(child, chunk_size=chunk_size)
    return model


def stable_hskm(model: nn.Module, chunk_size: int = 256) -> nn.Module:
    """Convenience wrapper: patch ``model`` in place and return it."""
    return patch_hskm_stable(model, chunk_size=chunk_size)
