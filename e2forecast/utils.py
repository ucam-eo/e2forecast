from __future__ import annotations

import random

import numpy as np
import torch


def resolve_device(value: str = "auto", tf32: bool = False) -> torch.device:
    """``auto`` = CUDA when available.  ``tf32`` enables TF32 matmuls (used by the S1 / S2 experiments)."""
    device = torch.device("cuda" if value == "auto" and torch.cuda.is_available() else
                          ("cpu" if value == "auto" else value))
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.backends.cuda.matmul.allow_tf32 = tf32
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def l2_normalize(values: np.ndarray, check_finite: bool = True) -> np.ndarray:
    """Row-wise L2 normalisation in float32 (norm floor 1e-8); by default fails on non-finite values."""
    values = values.astype(np.float32, copy=False)
    values /= np.maximum(np.linalg.norm(values, axis=-1, keepdims=True), 1e-8)
    if check_finite and not np.isfinite(values).all():
        raise ValueError("embedding contains non-finite values")
    return values


def repeat_batch(x: torch.Tensor, times: int) -> torch.Tensor:
    """(B, ...) -> (times * B, ...), sample-major, so ``.view(times, B, ...)`` recovers the ensemble."""
    return x.repeat(times, *([1] * (x.dim() - 1)))
