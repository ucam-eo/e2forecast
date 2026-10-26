"""Fixed location encoder: GeoCLIP-style random Fourier features of (lon, lat)."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class GeoCLIPRFF(nn.Module):
    """Parameter-free random Fourier features of (lon, lat) in radians.

    ``omega`` ~ N(0, sigma^2) of shape (2, embedding_dim / 2) is drawn once from ``seed`` and stored as a buffer,
    so the encoding is identical across runs and saved with every checkpoint.  Output: [cos(x omega), sin(x omega)].
    """

    def __init__(self, embedding_dim: int = 256, sigma: float = 1.0, seed: int = 42) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        generator = torch.Generator().manual_seed(seed)
        self.register_buffer("omega", torch.randn(2, embedding_dim // 2, generator=generator) * sigma)

    def forward(self, lonlat: Tensor) -> Tensor:
        projection = torch.deg2rad(lonlat).to(self.omega.dtype) @ self.omega
        return torch.cat((projection.cos(), projection.sin()), dim=-1)
