"""Training loop shared by every experiment.

Each epoch draws batches from ``epoch_batches()`` (an iterable of keyword dicts for :meth:`E2Forecast.loss`),
optimises with AdamW (constant learning rate) and gradient-norm clipping, and writes

* ``latest.pt``  every epoch (model + optimizer, used by ``--resume``);
* ``best.pt``    when ``validate`` is given and its score (lower is better) improves;
* ``training_log.jsonl`` one row per epoch.
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Callable, Iterable

import torch
from torch import nn

from .checkpoint import checkpoint_payload, load_checkpoint, save_atomic
from .model import E2Forecast

Validate = Callable[[E2Forecast], tuple[float, dict]]  # -> (selection score, lower is better; logged values)


def add_optimisation_args(parser: argparse.ArgumentParser, *, epochs: int, batch_size: int,
                          steps_per_epoch: int | None = 1000) -> None:
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=epochs)
    if steps_per_epoch is not None:
        parser.add_argument("--steps-per-epoch", type=int, default=steps_per_epoch)
    parser.add_argument("--batch-size", type=int, default=batch_size)
    parser.add_argument("--lr", type=float, default=1.1e-4, help="constant learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--mixed-precision", choices=("none", "bf16"), default="none")
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--resume", action="store_true", help="continue from <output-dir>/latest.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=84513)


def add_model_args(parser: argparse.ArgumentParser, *, heads: int = 4) -> None:
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--heads", type=int, default=heads)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--min-std", type=float, default=1e-4, help="sigma_min, floor of the Gaussian std")
    parser.add_argument("--nll-weight", type=float, default=1.0, help="weight of L_nll in the joint objective")
    parser.add_argument("--t-eps", type=float, default=0.05, help="clip on 1 - t in the x-to-velocity conversion")
    parser.add_argument("--rff-sigma", type=float, default=1.0, help="bandwidth of the location RFF encoding")
    parser.add_argument("--rff-seed", type=int, default=42)


def model_config_from_args(args: argparse.Namespace, embed_dim: int, context_count: int, query_date: str) -> dict:
    return {"embed_dim": embed_dim, "context_count": context_count, "query_date": query_date, "depth": args.depth,
            "heads": args.heads, "ff_mult": 4, "dropout": args.dropout, "min_std": args.min_std,
            "rff_sigma": args.rff_sigma, "rff_seed": args.rff_seed}


def fit(model: E2Forecast, args: argparse.Namespace, *, task: str, model_config: dict, device: torch.device,
        epoch_batches: Callable[[], Iterable[dict]], validate: Validate | None = None, metadata: dict | None = None) -> None:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    metadata = {**(metadata or {}), "lr": args.lr, "batch_size": args.batch_size}
    start_epoch, best = 0, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(output_dir / "latest.pt", task)
        if checkpoint["model_config"] != model_config:
            raise ValueError("latest.pt was trained with a different model configuration")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch, best = checkpoint["epoch"], checkpoint.get("best_score", float("inf"))
    (output_dir / "config.json").write_text(json.dumps(
        {"task": task, "model_config": model_config, "args": {k: str(v) for k, v in vars(args).items()
                                                              if k != "func"}, **metadata}, indent=2) + "\n")
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}", flush=True)
    autocast = (lambda: torch.autocast(device.type, dtype=torch.bfloat16)) if args.mixed_precision == "bf16" \
        else nullcontext
    started = time.time()
    with (output_dir / "training_log.jsonl").open("a") as log:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            model.train()
            totals, steps, skipped = {"loss": 0.0, "nll": 0.0, "flow": 0.0}, 0, 0
            for batch in epoch_batches():
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                with autocast():
                    terms = model.loss(**batch)
                if not torch.isfinite(terms["loss"]):
                    skipped += 1
                    continue
                optimizer.zero_grad(set_to_none=True)
                terms["loss"].backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                steps += 1
                for key in totals:
                    totals[key] += float(terms[key])
                if steps % args.log_every == 0:
                    print(f"epoch={epoch}/{args.epochs} step={steps} " +
                          " ".join(f"{k}={v / steps:.6f}" for k, v in totals.items()), flush=True)
            row = {"epoch": epoch, **{k: v / max(steps, 1) for k, v in totals.items()}, "steps": steps,
                   "skipped_non_finite": skipped}
            improved = False
            if validate is not None and epoch % getattr(args, "validate_every", 1) == 0:
                model.eval()
                with autocast():
                    score, values = validate(model)
                improved, best = score < best, min(best, score)
                row.update(values, selection_score=score, best_saved=improved)
            row["elapsed_seconds"] = time.time() - started
            payload = checkpoint_payload(model, task, model_config, epoch, best_score=best, **metadata)
            if improved:
                save_atomic(payload, output_dir / "best.pt")
            save_atomic({**payload, "optimizer": optimizer.state_dict()}, output_dir / "latest.pt")
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(json.dumps(row), flush=True)
