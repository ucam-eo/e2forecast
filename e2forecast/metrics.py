"""Forecast metrics of the paper (Sec. 4.1), accumulated over batches in float64.

For a model that outputs a Gaussian (mu, sigma) and an ensemble of J samples:

* ``mean_z`` / ``std_z``: mean and population std of z = (e* - mu) / sigma over all finite (item, dim) values
  (ideal 0 and 1);
* ``point_cosine_distance``: mean of 1 - cos(mu, e*);
* ``ensemble_cosine_distance``: mean over items of min_j 1 - cos(e_hat_j, e*);
* ``crps``: closed-form CRPS of N(mu, sigma^2), averaged over items and dims (Gaussian mode);
* ``crps_ensemble_fair``: fair (unbiased) ensemble CRPS of the J samples, reported for reference.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor


def crps_gaussian(mean: Tensor, std: Tensor, target: Tensor) -> Tensor:
    """(B, D) x 3 -> (B,): closed-form CRPS of N(mean, std^2) at target, averaged over finite dims.

    CRPS = std * (z (2 Phi(z) - 1) + 2 phi(z) - 1 / sqrt(pi)),  z = (target - mean) / std.
    """
    z = (target - mean) / std
    value = std * (z * (2.0 * torch.special.ndtr(z) - 1.0) + 2.0 * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
                   - 1.0 / math.sqrt(math.pi))
    return torch.where(torch.isfinite(value), value, torch.full_like(value, float("nan"))).nanmean(-1)


def crps_ensemble(samples: Tensor, target: Tensor, fair: bool = True) -> Tensor:
    """(S, B, D), (B, D) -> (B,): E|X - y| - 0.5 E|X - X'| averaged over dims.

    ``fair=True`` uses 1 / (S (S - 1)) over distinct pairs (Ferro 2014); ``fair=False`` the 1 / S^2 plug-in form.
    """
    term1 = (samples - target).abs().mean(0)
    ordered = samples.sort(dim=0).values  # sum_{i,j} |x_i - x_j| in O(S log S) for sorted samples
    count = ordered.shape[0]
    weight = (2 * torch.arange(count, device=ordered.device, dtype=ordered.dtype) - count + 1).view(count, 1, 1)
    pair_sum = (weight * ordered).sum(0) * 2.0
    term2 = pair_sum / (count * (count - 1) if fair else count ** 2)
    return (term1 - 0.5 * term2).mean(-1)


class ForecastMetrics:
    """Accumulates sums so that results do not depend on the batch size."""

    KEYS = ("items", "z_count", "z_sum", "z_square", "point", "ensemble", "crps", "crps_ensemble_fair")

    def __init__(self) -> None:
        self.sums = dict.fromkeys(self.KEYS, 0.0)

    @torch.no_grad()
    def update(self, mean: Tensor, std: Tensor, samples: Tensor | None, target: Tensor) -> None:
        """mean, std, target: (B, D); samples: (J, B, D) normalised ensemble forecasts or None."""
        mean, std, target = mean.double(), std.double(), target.double()
        z = (target - mean) / std
        z = z[torch.isfinite(z)]
        s = self.sums
        s["items"] += len(target)
        s["z_count"] += z.numel()
        s["z_sum"] += z.sum().item()
        s["z_square"] += z.square().sum().item()
        s["point"] += (1 - F.cosine_similarity(mean, target, dim=-1)).sum().item()
        s["crps"] += crps_gaussian(mean, std, target).sum().item()
        if samples is not None:
            samples = samples.double()
            s["ensemble"] += (1 - F.cosine_similarity(samples, target.unsqueeze(0), dim=-1)).min(0).values.sum().item()
            s["crps_ensemble_fair"] += crps_ensemble(samples, target, fair=True).sum().item()

    def merge(self, sums: dict) -> None:
        for key in self.KEYS:
            self.sums[key] += sums[key]

    def summary(self) -> dict:
        s, n = self.sums, max(self.sums["items"], 1)
        mean_z = s["z_sum"] / max(s["z_count"], 1)
        return {"items": int(s["items"]), "mean_z": mean_z,
                "std_z": math.sqrt(max(s["z_square"] / max(s["z_count"], 1) - mean_z ** 2, 0.0)),
                "point_cosine_distance": s["point"] / n, "ensemble_cosine_distance": s["ensemble"] / n,
                "crps": s["crps"] / n, "crps_ensemble_fair": s["crps_ensemble_fair"] / n}
