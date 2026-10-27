#!/usr/bin/env python3
"""Train E2Forecast on the GSA events (OLMoEarth-nano patch-5 embeddings, 3 context observations).

Training is self-supervised on normal history only: every event's last (post-event) image is held out.
Paper setting: 300 epochs x 1000 steps x batch 256 (8 events per batch), AdamW lr 1.1e-4, gradient clip 0.5;
the last epoch is used for anomaly detection.

    python scripts/gsa/train.py --embedding-root data/GSA/embeddings_patch5 --output-dir runs/gsa
"""

from __future__ import annotations

import argparse
from pathlib import Path

from e2forecast.data import gsa
from e2forecast.model import build_model, conditions_from_doys as conditions
from e2forecast.training import (add_date_offset_arg, add_model_args, add_optimisation_args, fit,
                                 model_config_from_args)
from e2forecast.utils import resolve_device, seed_everything

DEFAULT_MANIFEST = Path(__file__).resolve().parents[2] / "data/gsa_manifest.json"


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    seed_everything(args.seed)
    events = gsa.load_manifest(args.manifest, args.embedding_root)
    sampler = gsa.Sampler(events, args.seed, args.events_per_batch)
    print(f"training events: {len(sampler.events)}/{len(events)}", flush=True)
    model_config = model_config_from_args(args, gsa.EMBED_DIM, gsa.CONTEXT_COUNT, "learned")
    model = build_model(model_config, args.nll_weight, args.t_eps)

    def epoch_batches():
        for _ in range(args.steps_per_epoch):
            batch = sampler.sample(args.batch_size)
            yield {"target": batch["target"], **conditions(batch, args.date_offset)}

    try:
        fit(model, args, task="gsa", model_config=model_config, device=device, epoch_batches=epoch_batches,
            metadata={"training_events": len(sampler.events), "date_offset": args.date_offset})
    finally:
        sampler.cache.close()


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--embedding-root", type=Path, required=True, help="root of the patch-5 embedding rasters")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--events-per-batch", type=int, default=8)
    add_optimisation_args(p, epochs=300, batch_size=256)
    add_model_args(p, heads=4)
    add_date_offset_arg(p, training=True)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
