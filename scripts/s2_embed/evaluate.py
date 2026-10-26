#!/usr/bin/env python3
"""Forecast metrics of E2Forecast on the S2_Embed test samples.

The paper reports the first 100,000 of the 1,000,000 fixed test samples (``--samples-limit 100000``; the file is
in random order).  Gaussian branch -> mean / std of z, point cosine distance and Gaussian CRPS; J = 10 ensemble
forecasts with S = 10 Euler steps -> ensemble cosine distance.  Sums are saved every ``--chunk`` samples to
``<output-dir>/chunks.json`` so an interrupted run resumes.

    python scripts/s2_embed/evaluate.py --data-root data/S2_Embed --checkpoint checkpoints/s2_embed.pt \
        --output-dir results/s2_embed --samples-limit 100000
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from e2forecast.checkpoint import load_model
from e2forecast.data import s2_embed
from e2forecast.metrics import ForecastMetrics
from e2forecast.model import conditions_from_doys as conditions
from e2forecast.utils import repeat_batch, resolve_device


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device, tf32=True)
    model, checkpoint = load_model(args.checkpoint, device, "s2_embed")
    data = s2_embed.SampleSet(s2_embed.Benchmark.from_root(args.data_root), "test", args.samples_limit, args.load_workers,
                              device if args.data_on_device else torch.device("cpu"))
    print(f"test: {data.count:,} samples from {data.patches} patches", flush=True)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    state_path = args.output_dir / "chunks.json"
    done = json.loads(state_path.read_text()) if state_path.exists() else {}
    started = time.time()
    for chunk_start in range(0, data.count, args.chunk):
        if str(chunk_start) in done:
            continue
        metrics = ForecastMetrics()
        for start in range(chunk_start, min(chunk_start + args.chunk, data.count), args.batch_size):
            batch = {k: v.to(device) for k, v in data.batch(start, min(start + args.batch_size,
                                                                       chunk_start + args.chunk)).items()}
            cond = conditions(batch)
            mean, std = model.gaussian(**cond)
            repeated = {k: repeat_batch(v, args.samples) for k, v in cond.items()}
            ensemble = model.sample(args.flow_steps, generator, **repeated).view(args.samples, len(mean), -1)
            metrics.update(mean, std, ensemble, batch["target"])
        done[str(chunk_start)] = metrics.sums
        state_path.write_text(json.dumps(done, indent=1))
        print(f"[{time.time() - started:7.0f}s] samples {chunk_start:,}-{chunk_start + args.chunk:,}", flush=True)
    total = ForecastMetrics()
    for sums in done.values():
        total.merge(sums)
    summary = {"checkpoint": str(args.checkpoint), "epoch": checkpoint["epoch"], "ensemble_size": args.samples,
               "flow_steps": args.flow_steps, **total.summary()}
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--samples-limit", type=int, default=100_000, help="first N test samples (0 = all 1,000,000)")
    p.add_argument("--samples", type=int, default=10, help="J ensemble forecasts per sample")
    p.add_argument("--flow-steps", type=int, default=10, help="S Euler steps per ensemble forecast")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--chunk", type=int, default=5000)
    p.add_argument("--data-on-device", action="store_true", help="keep all test embeddings on the GPU (~3 GB)")
    p.add_argument("--load-workers", type=int, default=16)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=84515)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
