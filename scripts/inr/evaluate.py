#!/usr/bin/env python3
"""Reconstruction metrics of an INR checkpoint on the held-out test shards: cosine similarity of the Gaussian mean
and of one flow sample to the true SatCLIP embedding, and the z-score statistics of the Gaussian branch.

    python scripts/inr/evaluate.py --checkpoint checkpoints/inr.pt \
        --shards 'data/satclip/satclip_embeddings_location_{000493..000542}.tar'
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from e2forecast.checkpoint import load_model
from e2forecast.data import satclip
from e2forecast.metrics import ForecastMetrics
from e2forecast.utils import resolve_device


@torch.no_grad()
def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    model, checkpoint = load_model(args.checkpoint, device)
    if checkpoint["task"] not in ("inr", "inr_date"):
        raise ValueError(f"{args.checkpoint} is not an INR checkpoint")
    with_date = checkpoint["task"] == "inr_date"
    _, _, test_urls = satclip.split_shards(args.shards, args.train_split, args.val_split, args.split_seed)
    loader = satclip.make_loader(test_urls, args.batch_size, args.num_workers, train=False)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    metrics, cosine_flow = ForecastMetrics(), 0.0
    for embedding, captions in loader:
        embedding = embedding.to(device)
        cond = {k: v.to(device) for k, v in satclip.parse_captions(captions, with_date).items()}
        mean, std = model.gaussian(**cond)
        sample = model.sample(args.flow_steps, generator, **cond)
        metrics.update(mean, std, sample.unsqueeze(0), embedding)
        cosine_flow += F.cosine_similarity(sample, embedding, dim=-1).sum().item()
    summary = metrics.summary()
    result = {"checkpoint": str(args.checkpoint), "task": checkpoint["task"], "test_shards": len(test_urls),
              "items": summary["items"], "cosine_mean": 1 - summary["point_cosine_distance"],
              "cosine_flow": cosine_flow / max(summary["items"], 1), "mean_z": summary["mean_z"],
              "std_z": summary["std_z"], "crps": summary["crps"]}
    print(json.dumps(result, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    satclip.add_shard_args(p)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--flow-steps", type=int, default=10)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=84513)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
