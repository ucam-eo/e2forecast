#!/usr/bin/env python3
"""Train E2Forecast on S2_Embed (PASTIS Sentinel-2, Prithvi-EO-2.0-300M 1024-D embeddings, 37 context dates).

Paper setting: 72 epochs x 1000 steps x batch 64, 8 heads, AdamW lr 1.1e-4, gradient clip 0.5; the last epoch
(``latest.pt``) is evaluated.  ``best.pt`` (lowest validation point cosine distance) is kept as well.

    python scripts/s2_embed/train.py --data-root data/S2_Embed --output-dir runs/s2_embed
"""

from __future__ import annotations

import argparse
from pathlib import Path

from e2forecast.data import s2_embed
from e2forecast.metrics import ForecastMetrics
from e2forecast.model import build_model, conditions_from_doys as conditions
from e2forecast.training import add_model_args, add_optimisation_args, fit, model_config_from_args
from e2forecast.utils import resolve_device, seed_everything


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device, tf32=True)
    seed_everything(args.seed)
    bench = s2_embed.Benchmark.from_root(args.data_root)
    bank = s2_embed.TrainBank(bench, bench.context_count, args.seed, args.load_workers, args.max_patches)
    model_config = model_config_from_args(args, s2_embed.EMBED_DIM, bench.context_count, "learned")
    model = build_model(model_config, args.nll_weight, args.t_eps)

    def epoch_batches():
        for _ in range(args.steps_per_epoch):
            batch = bank.sample(args.batch_size)
            yield {"target": batch["target"], **conditions(batch)}

    validate_enabled = args.val_samples >= 0
    if validate_enabled:
        val = s2_embed.SampleSet(bench, "val", args.val_samples, args.load_workers)

        def score_validation(m):
            metrics = ForecastMetrics()
            for start in range(0, val.count, args.val_batch_size):
                batch = {k: v.to(device) for k, v in val.batch(start, start + args.val_batch_size).items()}
                mean, std = m.gaussian(**conditions(batch))
                metrics.update(mean, std, None, batch["target"])
            point = metrics.summary()["point_cosine_distance"]
            return point, {"val_point_cosine_distance": point}

    fit(model, args, task="s2_embed", model_config=model_config, device=device, epoch_batches=epoch_batches,
        validate=score_validation if validate_enabled else None, metadata={"data": bank.summary})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-root", type=Path, required=True,
                   help="holds benchmark_config/, embeddings_prithvi_eo_2_300M/ and PASTIS/metadata.geojson")
    p.add_argument("--max-patches", type=int, default=0, help="smoke tests: only the first N train patches")
    p.add_argument("--load-workers", type=int, default=16)
    p.add_argument("--val-samples", type=int, default=0,
                   help="validate on the first N of the 1M validation samples every epoch (0 = all, -1 = off)")
    p.add_argument("--val-batch-size", type=int, default=1024)
    add_optimisation_args(p, epochs=72, batch_size=64)
    add_model_args(p, heads=8)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
