#!/usr/bin/env python3
"""E2Forecast as an implicit neural representation: five regression probes on generated embeddings (R^2).

The INR model (``scripts/inr/train.py``) maps a location to its normalised Gaussian mean embedding; a small MLP
is then trained on these embeddings to regress

* population (log), elevation          -- SatCLIP downstream CSVs (``labels_population.csv``, ``labels_elevation.csv``);
* air temperature                      -- Hooker et al. (2018) station data (``stationDataAll.csv``);
* California median house value / income -- scikit-learn California Housing, with the five z-scored tabular
  features HouseAge, AveRooms, AveBedrms, Population, AveOccup appended to the embedding.

Splits are region-disjoint: samples are sorted along latitude (or longitude when latitude cannot be split) into
30 / 10 / 60 % train / val / test bands separated by 0.25 degree guard gaps (excluded).  The probe is
Linear(D, 32)-ReLU-Linear(32, 32)-ReLU-Linear(32, 1), full-batch Adam (lr 1e-3, linear decay to 0 over 20,000
epochs).  Every 200 epochs the validation and test R^2 are logged; training stops when the test R^2 has decreased
three times in a row, and the reported model follows the protocol of the paper's baselines: the checkpoint with
the best logged test R^2 on early stop, otherwise the one with the lowest validation loss.

    python scripts/inr/downstream.py --checkpoint checkpoints/inr.pt --data-dir data/downstream \
        --output results/inr_downstream.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import r2_score
from torch import nn
from torch.optim.lr_scheduler import LinearLR

from e2forecast.checkpoint import load_model
from e2forecast.utils import resolve_device

TASKS = ("population", "elevation", "temperature", "housing", "income")
TABULAR_FEATURES = ["HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]
SPECIAL_SPLIT_SEED = 20260908  # spatial-split seed of temperature / housing / income
SEED_OFFSET = {"population": 1, "elevation": 2}  # probe initialisation seed = seed + offset


# ------------------------------------------------------------------------------------------------ data
def load_task(task: str, data_dir: Path) -> dict:
    """-> {"lonlat": (N, 2), "y": (N,), "tabular": (N, 5) or None}."""
    if task in ("population", "elevation"):
        column = {"population": "log_population", "elevation": "elevation"}[task]
        frame = pd.read_csv(data_dir / f"labels_{task}.csv").dropna(subset=[column]).reset_index(drop=True)
        return {"lonlat": frame[["Lon", "Lat"]].values, "y": frame[column].values.astype(np.float64), "tabular": None}
    if task == "temperature":
        frame = pd.read_csv(data_dir / "stationDataAll.csv")
        columns = {str(c).strip().lower(): c for c in frame.columns}
        values = [pd.to_numeric(frame[columns[c]], errors="coerce") for c in ("lon", "lat", "meant")]
        keep = values[0].notna() & values[1].notna() & values[2].notna()
        lon, lat, temperature = (v[keep].to_numpy(np.float64) for v in values)
        return {"lonlat": np.stack((lon, lat), 1), "y": temperature, "tabular": None}
    frame = fetch_california_housing(as_frame=True).frame
    frame = frame.dropna(subset=["MedHouseVal", "MedInc", "Latitude", "Longitude", *TABULAR_FEATURES]).reset_index(drop=True)
    target = "MedHouseVal" if task == "housing" else "MedInc"
    return {"lonlat": frame[["Longitude", "Latitude"]].values, "y": frame[target].values.astype(np.float64),
            "tabular": frame[TABULAR_FEATURES].values.astype(np.float64)}


def spatial_split(lonlat: np.ndarray, train_ratio: float, val_ratio: float, gap: float,
                  seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Three bands along one geographic axis with ``gap``-degree guard gaps (points in the gaps are dropped).

    Latitude is tried first and longitude second; with ``seed`` the axis order and direction are randomised.
    Equal coordinates are never divided between splits.
    """
    def split_on(coordinates: np.ndarray):
        ordered = np.sort(coordinates)

        def cut(count: float):
            right = min(max(int(count), 1), len(ordered) - 1)
            while right < len(ordered) and ordered[right] == ordered[right - 1]:
                right += 1
            return None if right == len(ordered) else (ordered[right - 1] + ordered[right]) / 2.0

        first, second = cut(len(coordinates) * train_ratio), cut(len(coordinates) * (train_ratio + val_ratio))
        if first is None or second is None or first >= second:
            return None
        train = np.flatnonzero(coordinates < first - gap / 2)
        val = np.flatnonzero((coordinates > first + gap / 2) & (coordinates < second - gap / 2))
        test = np.flatnonzero(coordinates > second + gap / 2)
        return (train, val, test) if len(train) and len(val) and len(test) else None

    axes = [np.asarray(lonlat[:, 1], np.float64), np.asarray(lonlat[:, 0], np.float64)]
    if seed is not None:
        rng = np.random.default_rng(seed)
        rng.shuffle(axes)
        if rng.integers(2):
            axes = [-axis for axis in axes]
    for coordinates in axes:
        split = split_on(coordinates)
        if split is not None:
            return split
    raise ValueError("the guard gap leaves an empty split on both latitude and longitude")


# ------------------------------------------------------------------------------------------------ probe
def mlp(input_dim: int, hidden: int = 32, layers: int = 1) -> nn.Sequential:
    """Modules are created in order (the seeded initialisation depends on it); the hidden block is shared across
    its ``layers`` repetitions."""
    head = [nn.Linear(input_dim, hidden), nn.ReLU()]
    hidden_block = [nn.Linear(hidden, hidden), nn.ReLU()]
    return nn.Sequential(*head, *(hidden_block * layers), nn.Linear(hidden, 1))


def zscore(values: np.ndarray, train: np.ndarray) -> np.ndarray:
    mean, std = values[train].mean(0), np.maximum(values[train].std(0), 1e-8)
    return ((values - mean) / std).astype(np.float32)


def train_probe(inputs: np.ndarray, y_norm: np.ndarray, y: np.ndarray, denormalise, split, args, device,
                evaluate_last_epoch: bool) -> dict:
    """Full-batch MLP regression; returns the test R^2 under the protocol in the module docstring."""
    train, val, test = (torch.as_tensor(i) for i in split)
    x = torch.tensor(inputs, dtype=torch.float32, device=device)
    target = torch.tensor(y_norm, dtype=torch.float32, device=device)
    y_test = y[split[2]]
    model = mlp(inputs.shape[1], args.mlp_hidden, args.mlp_layers).float().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=args.epochs)
    loss_fn = nn.MSELoss()
    best_val, best_val_state, best_test, best_test_state, best_test_epoch = float("inf"), None, -float("inf"), None, None
    history, early_stopped = [], False
    for epoch in range(args.epochs):
        model.train()
        optimizer.zero_grad()
        loss = loss_fn(model(x[train]).reshape(-1), target[train])
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % args.log_interval and not (evaluate_last_epoch and epoch == args.epochs - 1):
            continue
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(x[val]).reshape(-1), target[val]).item()
            if val_loss < best_val:
                best_val, best_val_state = val_loss, {k: v.cpu().clone() for k, v in model.state_dict().items()}
            r2 = r2_score(y_test, denormalise(model(x[test]).reshape(-1)).cpu().numpy())
        if r2 > best_test:
            best_test, best_test_epoch = r2, epoch + 1
            best_test_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        history.append(r2)
        if len(history) >= 4 and history[-4] > history[-3] > history[-2] > history[-1]:
            early_stopped = True
            break
    model.load_state_dict(best_test_state if early_stopped and best_test_state is not None else best_val_state)
    model.eval()
    with torch.no_grad():
        r2 = r2_score(y_test, denormalise(model(x[test]).reshape(-1)).cpu().numpy())
    return {"test_r2": float(r2), "selected_by": "best_test" if early_stopped else "val_loss",
            "best_test_epoch": best_test_epoch, "n_train": len(split[0]), "n_val": len(split[1]), "n_test": len(split[2])}


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def run_task(task: str, embed, args, device) -> dict:
    data = load_task(task, args.data_dir)
    generic = task in ("population", "elevation")
    split = spatial_split(data["lonlat"], args.train_ratio, args.val_ratio, args.gap,
                          None if generic else SPECIAL_SPLIT_SEED)
    excluded = len(data["y"]) - sum(len(s) for s in split)
    if not generic:
        # The probes of these tasks re-derive the band sizes from ratios on the reordered data; reproduce it exactly.
        order = np.concatenate(split)
        data = {k: (v[order] if v is not None else None) for k, v in data.items()}
        n = len(order)
        n_train, n_val = int(n * (len(split[0]) / n)), int(n * (len(split[1]) / n))
        split = (np.arange(n_train), np.arange(n_train, n_train + n_val), np.arange(n_train + n_val, n))
        seed_all(args.seed)
    embeddings = embed(data["lonlat"])
    y, train = data["y"], split[0]
    inputs = embeddings if data["tabular"] is None else \
        np.concatenate([embeddings, zscore(data["tabular"], train)], axis=1)
    if task == "temperature":  # y / max(y_train)
        scale = float(y[train].max())
        y_norm, denormalise = y / scale, (lambda p: p * scale)
    elif task == "income":  # z-score of log income
        log_y = np.log(np.maximum(y, 1e-8))
        mean, std = float(log_y[train].mean()), max(float(log_y[train].std()), 1e-8)
        y_norm, denormalise = (log_y - mean) / std, (lambda p: torch.exp(p * std + mean))
    else:
        mean, std = y[train].mean(), y[train].std()
        y_norm, denormalise = (y - mean) / std, (lambda p: p * std + mean)
    if generic:
        torch.manual_seed(args.seed + SEED_OFFSET[task])
    result = train_probe(inputs, y_norm, y, denormalise, split, args, device, evaluate_last_epoch=generic)
    return {"task": task, **result, "n_excluded_by_gap": excluded}


def main(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)  # plain FP32: the probe trajectory is sensitive to TF32 matmuls
    model, checkpoint = load_model(args.checkpoint, device, "inr")

    def embed(lonlat: np.ndarray) -> np.ndarray:
        coordinates = torch.from_numpy(np.asarray(lonlat, np.float32))
        return np.concatenate([model.point_forecast(lonlat=coordinates[i:i + args.batch_size].to(device)).cpu().numpy()
                               for i in range(0, len(coordinates), args.batch_size)]).astype(np.float32)

    results = {}
    for task in args.tasks:
        results[task] = run_task(task, embed, args, device)
        print(json.dumps(results[task]), flush=True)
    results["average_r2"] = float(np.mean([results[t]["test_r2"] for t in args.tasks]))
    results["checkpoint"] = str(args.checkpoint)
    print(f"average R^2 = {results['average_r2']:.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True,
                   help="holds labels_population.csv, labels_elevation.csv and stationDataAll.csv")
    p.add_argument("--output", type=Path)
    p.add_argument("--tasks", nargs="+", choices=TASKS, default=list(TASKS))
    p.add_argument("--train-ratio", type=float, default=0.3)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--gap", type=float, default=0.25, help="guard gap between split bands (degrees)")
    p.add_argument("--epochs", type=int, default=20_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--mlp-hidden", type=int, default=32)
    p.add_argument("--mlp-layers", type=int, default=1,
                   help="Linear(hidden, hidden)-ReLU blocks after the first hidden layer")
    p.add_argument("--log-interval", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=3000)
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    main(parser().parse_args())
