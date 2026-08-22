"""Small compatibility shims for the pinned FireRed runtime."""

from __future__ import annotations

from collections.abc import Callable
from functools import wraps
from typing import Any


def bf16_forced_align_compatible(
    forced_align: Callable[..., Any],
    *,
    bf16_dtype: object | None = None,
) -> Callable[..., Any]:
    """Cast only BF16 log probabilities at the torchaudio boundary.

    The installed torchaudio CUDA kernel accepts FP16/FP32/FP64 but rejects
    BF16.  Keeping this conversion at the boundary leaves the AED model in its
    configured production precision.
    """

    @wraps(forced_align)
    def wrapped(log_probs: Any, targets: Any, *args: Any, **kwargs: Any) -> Any:
        dtype = getattr(log_probs, "dtype", None)
        expected = bf16_dtype
        if expected is None:
            try:
                import torch

                expected = torch.bfloat16
            except ImportError:
                expected = "bfloat16"
        if dtype == expected or str(dtype).lower() in {"bf16", "bfloat16", "torch.bfloat16"}:
            log_probs = log_probs.float()
        return forced_align(log_probs, targets, *args, **kwargs)

    return wrapped


def install_bf16_forced_align_compat() -> None:
    """Patch torchaudio once in the worker process before FireRed is loaded."""

    import torchaudio.functional

    if getattr(torchaudio.functional, "_funasr_bf16_forced_align", False):
        return
    torchaudio.functional.forced_align = bf16_forced_align_compatible(
        torchaudio.functional.forced_align
    )
    torchaudio.functional._funasr_bf16_forced_align = True
