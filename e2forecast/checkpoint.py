"""Checkpoint format shared by all training, evaluation and export scripts.

A checkpoint is a dict::

    {"format": "e2forecast-v1", "task": "s1_embed" | "s2_embed" | "gsa" | "inr" | "inr_date",
     "model_config": {...E2ForecastNetwork kwargs...}, "loss_config": {"nll_weight", "t_eps"},
     "model": state_dict, "epoch": int, "optimizer": state_dict (training checkpoints only), ...metadata}
"""

from __future__ import annotations

from pathlib import Path

import torch

from .model import E2Forecast, build_model

FORMAT = "e2forecast-v1"
TASKS = ("s1_embed", "s2_embed", "gsa", "inr", "inr_date")


def save_atomic(payload: dict, path: Path) -> None:
    """Write to a temporary file and rename, so an interrupted save never leaves a truncated checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    torch.save(payload, partial)
    partial.replace(path)


def checkpoint_payload(model: E2Forecast, task: str, model_config: dict, epoch: int, **extra) -> dict:
    if task not in TASKS:
        raise ValueError(f"task must be one of {TASKS}")
    return {"format": FORMAT, "task": task, "model_config": dict(model_config),
            "loss_config": {"nll_weight": model.nll_weight, "t_eps": model.t_eps},
            "model": {k: v.detach().cpu() for k, v in model.state_dict().items()}, "epoch": epoch, **extra}


def load_checkpoint(path: Path, task: str | None = None) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != FORMAT:
        raise ValueError(f"{path} is not an {FORMAT} checkpoint; convert legacy checkpoints with "
                         "scripts/export_checkpoint.py")
    if task is not None and checkpoint["task"] != task:
        raise ValueError(f"{path} is a '{checkpoint['task']}' checkpoint, expected '{task}'")
    return checkpoint


def load_model(path: Path, device: torch.device, task: str | None = None) -> tuple[E2Forecast, dict]:
    """Build the model from a checkpoint, load its weights (strict) and switch to eval mode."""
    checkpoint = load_checkpoint(path, task)
    model = build_model(checkpoint["model_config"], **checkpoint["loss_config"])
    model.load_state_dict(checkpoint["model"])
    return model.to(device).eval(), checkpoint
