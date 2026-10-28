#!/usr/bin/env python3
"""Convert a checkpoint written by the original research scripts into the release format (``e2forecast-v1``).

The parameter names of :class:`e2forecast.model.E2ForecastNetwork` are unchanged, so the weights load strictly;
optimizer state is dropped.  The original scripts encoded the observation order (-C..-1), so the released
checkpoints keep ``date_offset="order"`` (``--date-offset`` overrides it).

    python scripts/export_checkpoint.py --task s1_embed --source /path/to/ours_locenc/e2forecast_rff/latest.pt \
        --output checkpoints/s1_embed.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from e2forecast.checkpoint import FORECAST_TASKS, TASKS, checkpoint_payload, save_atomic
from e2forecast.model import build_model

OBJECTIVE = ("E2Forecast: joint probability estimation (diagonal Gaussian NLL) and embedding generation "
             "(rectified flow, x-prediction with velocity loss) in one masked Transformer pass")
EMBED_DIM = {"s1_embed": 128, "s2_embed": 1024, "gsa": 128, "inr": 256, "inr_date": 256}


def convert(task: str, legacy: dict) -> tuple[dict, dict]:
    """-> (model_config, loss_config) of the release format."""
    old = legacy["model_config"]
    config = {"embed_dim": EMBED_DIM[task], "depth": old["depth"], "heads": old["heads"], "ff_mult": 4,
              "dropout": old["dropout"], "min_std": old["min_std"], "rff_sigma": old.get("rff_sigma", 1.0),
              "rff_seed": old.get("rff_seed", 42)}
    if old.get("model_dim", EMBED_DIM[task]) != EMBED_DIM[task]:
        raise ValueError(f"model_dim {old['model_dim']} differs from the {task} embedding dimension")
    if task in ("s1_embed", "s2_embed"):
        if legacy.get("objective") != OBJECTIVE or legacy.get("location_encoding") != "geoclip_rff":
            raise ValueError("not an E2Forecast (GeoCLIP-RFF) forecasting checkpoint")
        config.update(context_count=old["context_count"], query_date="learned")
        loss = {"nll_weight": legacy.get("nll_weight", 1.0), "t_eps": legacy["t_eps"]}
    elif task == "gsa":
        if legacy.get("loss_config", {}).get("objective") != OBJECTIVE or legacy.get("condition"):
            raise ValueError("not an E2Forecast (GeoCLIP-RFF, no annual condition) GSA checkpoint")
        config.update(context_count=3, query_date="learned")
        loss = {"nll_weight": legacy["loss_config"]["nll_weight"], "t_eps": legacy["loss_config"]["t_eps"]}
    else:
        config.update(context_count=0, query_date="fixed" if task == "inr_date" else "none")
        loss = {"nll_weight": legacy.get("nll_weight", 1.0), "t_eps": legacy["t_eps"]}
    return config, loss


RELEASE_DATE_OFFSET = {"s1_embed": "order", "s2_embed": "order", "gsa": "order"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--date-offset", choices=("year", "order"), help="default: the release setting of the task")
    args = parser.parse_args()
    legacy = torch.load(args.source, map_location="cpu", weights_only=False)
    model_config, loss_config = convert(args.task, legacy)
    model = build_model(model_config, **loss_config)
    model.load_state_dict(legacy["model"])  # strict: every tensor name and shape must match
    epoch = int(legacy["epoch"]) + (1 if args.task == "gsa" else 0)  # the GSA script stored 0-based epochs
    extra = {"source": str(args.source.resolve())}
    if args.task in FORECAST_TASKS:
        extra["date_offset"] = args.date_offset or RELEASE_DATE_OFFSET[args.task]
    save_atomic(checkpoint_payload(model, args.task, model_config, epoch, **extra), args.output)
    print(f"{args.source} -> {args.output} (epoch {epoch}, "
          f"{sum(p.numel() for p in model.parameters()):,} parameters)")


if __name__ == "__main__":
    main()
