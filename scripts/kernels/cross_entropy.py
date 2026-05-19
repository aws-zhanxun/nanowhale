"""NKI fused cross-entropy wrapper.

Calls `nkilib.experimental.loss.cross_entropy_forward / _backward` via a
`torch.autograd.Function`. Drop-in replacement for
`F.cross_entropy(logits, targets, reduction="mean")`.

Limitations:
- No `ignore_index` support — caller must ensure no -100 in `targets`.
- `inplace=False` for the backward kernel: we do not overwrite `logits`,
  which costs an extra `[N, V]` HBM buffer but keeps the call site simple.
"""

from __future__ import annotations

import torch


_LNC_CACHE: int | None = None
_CE_FN_CACHE = None


def _get_lnc() -> int:
    """Logical Neuron Core count (LNC=2 on trn2). Cached after first lookup."""
    global _LNC_CACHE
    if _LNC_CACHE is None:
        from torch_neuronx.utils import get_logical_neuron_cores
        _LNC_CACHE = int(get_logical_neuron_cores())
    return _LNC_CACHE


def _get_ce_autograd_fn():
    """Build the cross-entropy autograd Function on first use, then cache.

    Lazy because the imports below pull the Neuron SDK; we want this file to
    be importable on CPU without those packages installed.
    """
    global _CE_FN_CACHE
    if _CE_FN_CACHE is not None:
        return _CE_FN_CACHE

    import nki.language as nl
    from nkilib.experimental.loss import (
        cross_entropy_forward,
        cross_entropy_backward,
    )

    LNC = _get_lnc()

    class NKICrossEntropyFn(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits, targets):
            # logits: [N, V] fp32 ; targets: [N] int32
            # Returns scalar mean loss to match F.cross_entropy(reduction="mean").
            loss_vec, lse = cross_entropy_forward[LNC](
                logits, targets, dtype=nl.float32,
            )
            ctx.save_for_backward(logits, targets, lse)
            return loss_vec.mean()

        @staticmethod
        def backward(ctx, grad_loss_scalar):
            logits, targets, lse = ctx.saved_tensors
            # The kernel bakes 1/N into the gradient when reduction="mean",
            # so its output is already ∂L_mean/∂logits. We only need to scale
            # by `grad_loss_scalar` (which is 1.0 in the common loss.backward()
            # case but could differ if the user sets grad_output).
            grad_logits = cross_entropy_backward[LNC](
                logits, targets, lse,
                reduction="mean",
                inplace=False,           # don't clobber logits — caller may keep it
                dtype=nl.float32,
            )
            return grad_logits * grad_loss_scalar, None

    _CE_FN_CACHE = NKICrossEntropyFn
    return NKICrossEntropyFn


def nki_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Drop-in replacement for `F.cross_entropy(logits, targets, reduction="mean")`.

    Args:
        logits: 2D [N, V], fp32 or bf16. Will be passed straight to the kernel.
        targets: 1D [N], int32 or int64 (cast to int32 here). **Must not contain
            -100** — the underlying kernel has no `ignore_index` support and
            -100 would index out of bounds.

    Returns:
        Scalar tensor, mean cross-entropy across positions.
    """
    if logits.dim() != 2:
        raise ValueError(f"logits must be 2D [N, V], got shape {tuple(logits.shape)}")
    if targets.dim() != 1:
        raise ValueError(f"targets must be 1D [N], got shape {tuple(targets.shape)}")
    if logits.shape[0] != targets.shape[0]:
        raise ValueError(
            f"logits.shape[0]={logits.shape[0]} != targets.shape[0]={targets.shape[0]}"
        )
    targets_i32 = targets.to(torch.int32)
    return _get_ce_autograd_fn().apply(logits, targets_i32)
