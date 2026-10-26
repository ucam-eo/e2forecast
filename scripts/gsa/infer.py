#!/usr/bin/env python3
"""Anomaly maps for every GSA event with a trained E2Forecast model.

For each event the chronologically last image is the observation e_obs and the three images before it are the
context.  For every pixel the script computes (J = 10 ensemble forecasts, S = 10 Euler steps each)

* ``e2forecast``  : squared anomaly score min(||(e_obs - mu)/sigma||^2, min_j ||(e_obs - e_hat_j)/sigma||^2);
* ``mahalanobis`` : ||(e_obs - mu)/sigma||   (Gaussian branch only);
* ``cosine``      : 1 - cos(mu, e_obs)       (point forecast only),

and writes ``<output-dir>/<category>/<event>/anomaly_<score>_native.npy`` (embedding grid) and
``anomaly_<score>.npy`` (bilinearly upsampled to the 10 m label grid), the input of ``scripts/gsa/evaluate.py``.

    python scripts/gsa/infer.py --checkpoint checkpoints/gsa.pt --embedding-root data/GSA/embeddings_patch5 \
        --label-root data/GSA/original --output-dir results/gsa
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from e2forecast.anomaly import anomaly_scores, find_label, read_label, upsample_to
from e2forecast.checkpoint import load_model
from e2forecast.data import gsa
from e2forecast.utils import l2_normalize, resolve_device

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data/gsa_manifest.json"


def save_turbo_png(path: Path, values: np.ndarray) -> None:
    import matplotlib
    from PIL import Image

    finite = values[np.isfinite(values)]
    low, high = np.percentile(finite, [1, 99]) if finite.size else (0.0, 1.0)
    normed = np.clip((values - low) / max(float(high - low), 1e-12), 0.0, 1.0)
    rgb = matplotlib.colormaps["turbo"](np.nan_to_num(normed, nan=0.0))[..., :3]
    rgb[~np.isfinite(values)] = 0.0
    Image.fromarray((rgb * 255).round().astype(np.uint8), mode="RGB").save(path)


@torch.no_grad()
def event_maps(model, event: gsa.Event, cache: gsa.RasterCache, args, generator, device) -> dict[str, np.ndarray]:
    context_steps = range(len(event.images) - 1 - gsa.CONTEXT_COUNT, len(event.images) - 1)
    observed_step = len(event.images) - 1
    obs_dates = torch.tensor([[order, event.day_of_year(step)] for order, step in
                              zip(range(-gsa.CONTEXT_COUNT, 0), context_steps)], dtype=torch.float32, device=device)
    query_date = torch.tensor([0, event.day_of_year(observed_step)], dtype=torch.float32, device=device)
    total = event.height * event.width
    maps = {}
    for start in range(0, total, args.batch_size):
        index = np.arange(start, min(start + args.batch_size, total))
        rows, cols = np.unravel_index(index, (event.height, event.width))
        obs = np.stack([gsa.read_pixels(cache.get(event.images[i]), rows, cols) for i in context_steps], 1)
        observed = gsa.read_pixels(cache.get(event.images[observed_step]), rows, cols)
        n = len(index)
        scores = anomaly_scores(
            model, torch.from_numpy(l2_normalize(observed, False)).to(device), args.samples, args.flow_steps, generator,
            obs=torch.from_numpy(l2_normalize(obs, False)).to(device), obs_dates=obs_dates.expand(n, -1, -1),
            query_date=query_date.expand(n, -1), lonlat=torch.from_numpy(gsa.lonlat_for(event, rows, cols)).to(device))
        for name, value in scores.items():
            maps.setdefault(name, np.zeros(total, np.float32))[start:start + n] = value.cpu().numpy()
    return {name: value.reshape(event.height, event.width) for name, value in maps.items()}


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device, "gsa")
    events = gsa.load_manifest(args.manifest, args.embedding_root)
    if args.events:
        events = [e for e in events if e.name in set(args.events) or e.category in set(args.events)]
    cache, generator = gsa.RasterCache(), torch.Generator(device=device).manual_seed(args.seed)
    started = time.time()
    for number, event in enumerate(events, 1):
        out_dir = args.output_dir / event.name
        if (out_dir / "anomaly_e2forecast.npy").exists() and not args.overwrite:
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        maps = event_maps(model, event, cache, args, generator, device)
        label_shape = read_label(find_label(args.label_root, event.name)).shape
        factor = label_shape[0] // event.height
        for name, native in maps.items():
            upsampled = upsample_to(native, label_shape, factor, "bilinear")
            np.save(out_dir / f"anomaly_{name}_native.npy", native)
            np.save(out_dir / f"anomaly_{name}.npy", upsampled)
            if args.save_png:
                save_turbo_png(out_dir / f"anomaly_{name}.png", upsampled)
        (out_dir / "inference.json").write_text(json.dumps(
            {"event": event.name, "observed_date": event.dates[-1], "context_dates": event.dates[-4:-1],
             "samples": args.samples, "flow_steps": args.flow_steps, "upsample_factor": factor,
             "checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"]}, ensure_ascii=False, indent=2) + "\n")
        print(f"[{time.time() - started:7.0f}s] {number}/{len(events)} {event.name}", flush=True)
    cache.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--embedding-root", type=Path, required=True)
    p.add_argument("--label-root", type=Path, required=True, help="10 m GSA root with <category>/<event>/*label*.tif")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--events", nargs="*", help="optional subset: event names or categories")
    p.add_argument("--samples", type=int, default=10, help="J ensemble forecasts")
    p.add_argument("--flow-steps", type=int, default=10, help="S Euler steps per forecast")
    p.add_argument("--batch-size", type=int, default=2048)
    p.add_argument("--save-png", action="store_true", help="also write turbo-coloured PNGs of the upsampled maps")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=84513)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
