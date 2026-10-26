#!/usr/bin/env python3
"""Monthly global fields of the date-conditioned INR model (query date = the 15th of every month).

Two 3 x 4 figures are written:

* ``<output>``: the 12 fields with one PCA fitted jointly over all months (colours comparable across panels);
* ``<output stem>_anomaly.png``: the seasonal signal only -- every grid point's 12-month mean is removed, a shared
  PCA is fitted on the deviations and mapped symmetrically around grey (clipped at the 99th percentile).

    python scripts/inr/plot_global_monthly.py --checkpoint checkpoints/inr_date.pt --output figures/global_monthly.png
"""

from __future__ import annotations

import argparse
import calendar
from datetime import date
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from e2forecast.checkpoint import load_model
from e2forecast.globe import embed_grid, global_grid, land_mask, pcs_to_rgb, plot_globes
from e2forecast.utils import resolve_device

MONTHS = list(calendar.month_name)[1:]
MID_MONTH_DOY = [date(2021, m, 15).timetuple().tm_yday for m in range(1, 13)]  # non-leap year


def monthly_rgb(embeddings: list[np.ndarray]) -> list[np.ndarray]:
    pca = PCA(n_components=3).fit(np.concatenate(embeddings))
    pcs = [pca.transform(e) for e in embeddings]
    stacked = np.concatenate(pcs)
    return [pcs_to_rgb(p, stacked.min(0), stacked.max(0)) for p in pcs]


def seasonal_rgb(embeddings: list[np.ndarray], gamma: float = 0.8) -> tuple[list[np.ndarray], np.ndarray]:
    deviations = [e - np.mean(embeddings, 0) for e in embeddings]
    pca = PCA(n_components=3).fit(np.concatenate(deviations))
    pcs = [pca.transform(d) for d in deviations]
    scale = np.percentile(np.abs(np.concatenate(pcs)), 99, axis=0)
    rgb = [0.5 + 0.5 * np.sign(p) * np.abs(p.clip(-scale, scale) / scale) ** gamma for p in pcs]
    return rgb, pca.explained_variance_ratio_


def main(args: argparse.Namespace) -> None:
    model, _ = load_model(args.checkpoint, resolve_device(args.device), "inr_date")
    lats, lons, shape = global_grid(args.step)
    embeddings = [embed_grid(model, lats, lons, args.batch_size, day_of_year=doy, flow_steps=args.flow_steps)
                  for doy in MID_MONTH_DOY]
    print("mean cosine to January: " + ", ".join(f"{m[:3]}={float((e * embeddings[0]).sum(-1).mean()):.4f}"
                                                for m, e in zip(MONTHS, embeddings)))
    if args.save_embeddings:
        np.save(args.output.with_suffix(".npy"), np.stack([e.reshape(shape + (-1,)) for e in embeddings]).astype(np.float16))
    land = None if args.keep_ocean else land_mask(lats, lons)
    reshape = lambda images: [image.reshape(shape + (3,)) for image in images]  # noqa: E731
    plot_globes(reshape(monthly_rgb(embeddings)), MONTHS, args.output, columns=4, dpi=args.dpi, land=land)
    rgb, ratio = seasonal_rgb(embeddings)
    print(f"seasonal PCA explained variance: {np.round(ratio, 3)}")
    plot_globes(reshape(rgb), MONTHS, args.output.with_name(args.output.stem + "_anomaly.png"), columns=4,
                dpi=args.dpi, land=land)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=float, default=1.0, help="grid spacing in degrees")
    p.add_argument("--flow-steps", type=int, default=0, help="0 = normalised Gaussian mean, else one flow sample")
    p.add_argument("--keep-ocean", action="store_true")
    p.add_argument("--save-embeddings", action="store_true")
    p.add_argument("--batch-size", type=int, default=3000)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--device", default="auto")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
