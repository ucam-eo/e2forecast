#!/usr/bin/env python3
"""Forecast metrics of E2Forecast on the S1_Embed test set (50 sites x 20,000 items, 2025).

Gaussian branch -> mean / std of z, point cosine distance and Gaussian CRPS; J = 10 ensemble forecasts with
S = 10 Euler steps -> ensemble cosine distance (closest member).  Per-site sums are written after every site to
``<output-dir>/per_site.json``, so an interrupted run resumes.

    python scripts/s1_embed/evaluate.py --data-root data/S1_Embed --checkpoint checkpoints/s1_embed.pt \
        --output-dir results/s1_embed
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from e2forecast.checkpoint import load_model
from e2forecast.data import s1_embed
from e2forecast.metrics import ForecastMetrics
from e2forecast.model import conditions_from_doys as conditions
from e2forecast.utils import repeat_batch, resolve_device


@torch.no_grad()
def evaluate_batch(model, batch, samples: int, steps: int, generator) -> tuple[torch.Tensor, ...]:
    cond = conditions(batch)
    mean, std = model.gaussian(**cond)
    ensemble = None
    if samples:
        repeated = {k: repeat_batch(v, samples) for k, v in cond.items()}
        ensemble = model.sample(steps, generator, **repeated).view(samples, len(mean), -1)
    return mean, std, ensemble


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device, tf32=True)
    model, checkpoint = load_model(args.checkpoint, device, "s1_embed")
    context_count = checkpoint["model_config"]["context_count"]
    items = s1_embed.EvalItems(args.data_root / "config/test_config.json", args.data_root / "embeddings",
                               args.load_workers, args.items_per_site)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "per_site.json"
    done = json.loads(state_path.read_text()) if state_path.exists() else {}
    todo = [i for i, name in enumerate(items.names) if name not in done]
    started, current, metrics = time.time(), None, None
    for i, batch in items.batches(context_count, args.batch_size, device, todo):
        if i != current:
            if current is not None:
                done[items.names[current]] = metrics.sums
                state_path.write_text(json.dumps(done, indent=1))
                print(f"[{time.time() - started:7.0f}s] {len(done)}/{len(items.names)} {items.names[current]}", flush=True)
            current, metrics = i, ForecastMetrics()
        metrics.update(*evaluate_batch(model, batch, args.samples, args.flow_steps, generator), batch["target"])
    if current is not None:
        done[items.names[current]] = metrics.sums
        state_path.write_text(json.dumps(done, indent=1))
    total = ForecastMetrics()
    for sums in done.values():
        total.merge(sums)
    summary = {"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"], "context_count": context_count,
               "ensemble_size": args.samples, "flow_steps": args.flow_steps, "sites": len(done), **total.summary()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--items-per-site", type=int, default=0, help="first N test items of every site (0 = all 20,000)")
    p.add_argument("--samples", type=int, default=10, help="J ensemble forecasts per item")
    p.add_argument("--flow-steps", type=int, default=10, help="S Euler steps per ensemble forecast")
    p.add_argument("--batch-size", type=int, default=4096)
    p.add_argument("--load-workers", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=84513)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
