#!/usr/bin/env python3
"""Train E2Forecast as an implicit neural representation: location (+ optional query date) -> SatCLIP embedding.

The context is empty, so the token sequence is [location | (query date) | h_mu | h_sigma | flow time | z_t |
h_query].  Without ``--query-date`` the model is trained on S2-100K (as SatCLIP); with ``--query-date`` on
SSL4EO-S12, whose captions carry the acquisition date (fixed day-of-year encoding, no projection).

Paper setting: 300 epochs of 200,000 samples, batch 256, AdamW lr 1.1e-4, gradient clip 0.5, RFF sigma 1;
``best.pt`` (best validation cosine of the Gaussian mean or a flow sample, every 2 epochs) is used.

    python scripts/inr/train.py --shards 'data/satclip/satclip_embeddings_location_{000493..000542}.tar' \
        --output-dir runs/inr
    python scripts/inr/train.py --query-date \
        --shards 'data/satclip/satclip_embeddings_location_{000000..000492}.tar' --output-dir runs/inr_date
"""

from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from e2forecast.data import satclip
from e2forecast.model import build_model
from e2forecast.training import add_model_args, add_optimisation_args, fit, model_config_from_args
from e2forecast.utils import resolve_device, seed_everything


@torch.no_grad()
def cosine_scores(model, loader, with_date: bool, device, flow_steps: int, max_batches: int, seed: int) -> dict:
    """Mean cosine similarity of the Gaussian mean and of one flow sample to the true embedding."""
    generator = torch.Generator(device=device).manual_seed(seed)
    totals, count = {"cosine_mean": 0.0, "cosine_flow": 0.0}, 0
    for step, (embedding, captions) in enumerate(loader):
        if max_batches and step >= max_batches:
            break
        embedding = embedding.to(device)
        cond = {k: v.to(device) for k, v in satclip.parse_captions(captions, with_date).items()}
        totals["cosine_mean"] += F.cosine_similarity(model.point_forecast(**cond), embedding, dim=-1).sum().item()
        totals["cosine_flow"] += F.cosine_similarity(model.sample(flow_steps, generator, **cond), embedding,
                                                     dim=-1).sum().item()
        count += len(embedding)
    return {k: v / max(count, 1) for k, v in totals.items()}


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    seed_everything(args.seed)
    task = "inr_date" if args.query_date else "inr"
    train_urls, val_urls, _ = satclip.split_shards(args.shards, args.train_split, args.val_split, args.split_seed)
    print(f"{len(train_urls)} train / {len(val_urls)} val shards", flush=True)
    model_config = model_config_from_args(args, satclip.EMBED_DIM, 0, "fixed" if args.query_date else "none")
    model = build_model(model_config, args.nll_weight, args.t_eps)
    loader = satclip.make_loader(train_urls, args.batch_size, args.num_workers, True, args.shuffle_buffer,
                                 args.samples_per_epoch)

    def epoch_batches():
        for embedding, captions in loader:
            yield {"target": embedding, **satclip.parse_captions(captions, args.query_date)}

    def validate(m):
        val_loader = satclip.make_loader(val_urls, args.batch_size, args.num_workers, False)
        scores = cosine_scores(m, val_loader, args.query_date, device, args.flow_steps, args.val_batches, args.seed)
        return -max(scores.values()), scores

    fit(model, args, task=task, model_config=model_config, device=device, epoch_batches=epoch_batches,
        validate=validate if val_urls else None, metadata={"shards": args.shards})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    satclip.add_shard_args(p)
    p.add_argument("--query-date", action="store_true", help="add the day-of-year query-date token (SSL4EO-S12)")
    p.add_argument("--samples-per-epoch", type=int, default=200_000,
                   help="samples per epoch and per data-loader worker (webdataset with_epoch semantics)")
    p.add_argument("--shuffle-buffer", type=int, default=2000)
    p.add_argument("--validate-every", type=int, default=2)
    p.add_argument("--val-batches", type=int, default=200, help="batches per validation (0 = whole split)")
    p.add_argument("--flow-steps", type=int, default=10)
    add_optimisation_args(p, epochs=300, batch_size=256, steps_per_epoch=None)
    add_model_args(p, heads=4)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
