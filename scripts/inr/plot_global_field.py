#!/usr/bin/env python3
"""Global embedding field of the INR model: query a regular grid, project the embeddings to RGB with PCA
(3 components, per-component min-max, gamma 0.8) and draw them on a Robinson globe with the ocean masked.

    python scripts/inr/plot_global_field.py --checkpoint checkpoints/inr.pt --output figures/global_field.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

from e2forecast.checkpoint import load_model
from e2forecast.globe import embed_grid, global_grid, land_mask, pcs_to_rgb, plot_globes
from e2forecast.utils import resolve_device


def main(args: argparse.Namespace) -> None:
    model, _ = load_model(args.checkpoint, resolve_device(args.device), "inr")
    lats, lons, shape = global_grid(args.step)
    embeddings = embed_grid(model, lats, lons, args.batch_size, flow_steps=args.flow_steps)
    if args.save_embeddings:
        np.save(args.output.with_suffix(".npy"), embeddings.reshape(shape + (-1,)).astype(np.float16))
    pcs = PCA(n_components=3).fit_transform(embeddings)
    rgb = pcs_to_rgb(pcs, pcs.min(0), pcs.max(0)).reshape(shape + (3,))
    plot_globes([rgb], [args.title], args.output, columns=1, dpi=args.dpi,
                land=None if args.keep_ocean else land_mask(lats, lons))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--step", type=float, default=0.5, help="grid spacing in degrees")
    p.add_argument("--flow-steps", type=int, default=0, help="0 = normalised Gaussian mean, else one flow sample")
    p.add_argument("--title", default="")
    p.add_argument("--keep-ocean", action="store_true")
    p.add_argument("--save-embeddings", action="store_true", help="also save the grid embeddings (.npy, float16)")
    p.add_argument("--batch-size", type=int, default=3000)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--device", default="auto")
    return p


if __name__ == "__main__":
    main(parser().parse_args())
