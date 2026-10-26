#!/usr/bin/env python3
"""Train E2Forecast on S1_Embed (OLMoEarth-nano patch-5 embeddings of Sentinel-1 RTC, 7 context observations).

Paper setting: 600 epochs x 1000 steps x batch 256, AdamW lr 1.1e-4, gradient clip 0.5, 8,192 fixed pixels per
site, the last epoch (``latest.pt``) is evaluated.

    python scripts/s1_embed/train.py --data-root data/S1_Embed --output-dir runs/s1_embed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from e2forecast.data import s1_embed
from e2forecast.metrics import ForecastMetrics
from e2forecast.model import build_model, conditions_from_doys as conditions
from e2forecast.training import add_model_args, add_optimisation_args, fit, model_config_from_args
from e2forecast.utils import resolve_device, seed_everything


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device, tf32=True)
    seed_everything(args.seed)
    root = args.data_root / "embeddings"
    bank = s1_embed.TrainBank(root, args.data_root / "config/split.json", args.context_count, args.pixels_per_site,
                              args.seed, args.load_workers, sites=args.sites)
    model_config = model_config_from_args(args, s1_embed.EMBED_DIM, args.context_count, "learned")
    model = build_model(model_config, args.nll_weight, args.t_eps)

    def epoch_batches():
        for _ in range(args.steps_per_epoch):
            batch = bank.sample(args.batch_size)
            yield {"target": batch["target"], **conditions(batch)}

    validate_enabled = args.val_items_per_site > 0
    if validate_enabled:  # optional: mean 1 - cos(mu, e*) on the fixed 2025 validation items
        val = s1_embed.EvalItems(args.data_root / "config/val_config.json", root, args.load_workers,
                                 args.val_items_per_site)

        def score_validation(m):
            metrics = ForecastMetrics()
            for _, batch in val.batches(args.context_count, 2048, device):
                mean, std = m.gaussian(**conditions(batch))
                metrics.update(mean, std, None, batch["target"])
            point = metrics.summary()["point_cosine_distance"]
            return point, {"val_point_cosine_distance": point}

    fit(model, args, task="s1_embed", model_config=model_config, device=device, epoch_batches=epoch_batches,
        validate=score_validation if validate_enabled else None, metadata={"data": bank.summary})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--context-count", type=int, default=7)
    p.add_argument("--pixels-per-site", type=int, default=8192, help="fixed random pixels kept per site; 0 = all")
    p.add_argument("--sites", nargs="*", help="optional site subset (smoke tests)")
    p.add_argument("--load-workers", type=int, default=8)
    p.add_argument("--val-items-per-site", type=int, default=0,
                   help="validate every epoch on this many 2025 items per validation site (0 = no validation)")
    add_optimisation_args(p, epochs=600, batch_size=256)
    add_model_args(p, heads=4)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
