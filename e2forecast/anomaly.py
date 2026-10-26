"""Forecast-based anomaly change detection (paper Eq. (anomaly)) and the GSA evaluation protocol.

Anomaly score of an observation e_obs given the Gaussian forecast (mu, sigma) and J ensemble forecasts e_hat_j:

    a^2 = min( ||(e_obs - mu) / sigma||^2 , min_j ||(e_obs - e_hat_j) / sigma||^2 )

The maps store the squared score a^2 (a monotone transform of a); reported GSA numbers use these squared maps.

Evaluation follows HyperFree / ESIA: per event, recall and a weighted precision in which false-positive pixels
count with weight 0.5; event F1 values are averaged within each category and the headline value is the macro
average over categories.  One threshold is selected per category (the best mean F1), by a dense histogram sweep
(``exact=False``) or over every observed score (``exact=True``).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rasterio
import torch
import torch.nn.functional as F
from torch import Tensor

from .model import E2Forecast


@torch.no_grad()
def anomaly_scores(model: E2Forecast, observed: Tensor, samples: int, steps: int, generator: torch.Generator,
                   **conditions: Tensor) -> dict[str, Tensor]:
    """-> {"e2forecast": a^2, "mahalanobis": ||(e_obs - mu)/sigma||, "cosine": 1 - cos(mu, e_obs)} per item."""
    mean, std = model.gaussian(**conditions)
    gaussian_term = ((observed - mean) / std).square().sum(-1)
    sample_term = torch.full_like(gaussian_term, torch.inf)
    for _ in range(samples):  # J ensemble forecasts, S Euler steps each
        forecast = model.sample(steps, generator, **conditions)
        sample_term = torch.minimum(sample_term, ((observed - forecast) / std).square().sum(-1))
    return {"e2forecast": torch.minimum(gaussian_term, sample_term), "mahalanobis": gaussian_term.sqrt(),
            "cosine": 1 - F.cosine_similarity(mean, observed, dim=-1)}


def upsample_to(score: np.ndarray, shape: tuple[int, int], factor: int, mode: str = "bilinear") -> np.ndarray:
    """Upsample a (h, w) patch-level map onto the (H, W) label grid, H = factor * h + (0..factor-1).

    ``bilinear`` places each patch score at its block centre and interpolates (edge values outside the outermost
    centres); ``nearest`` repeats every score over its block.
    """
    pad = (shape[0] - factor * score.shape[0], shape[1] - factor * score.shape[1])
    if not (0 <= pad[0] < factor and 0 <= pad[1] < factor):
        raise ValueError(f"score {score.shape} x{factor} does not cover label {shape}")
    if mode == "nearest":
        return np.pad(np.repeat(np.repeat(score, factor, 0), factor, 1), ((0, pad[0]), (0, pad[1])), mode="edge")

    def axis(size: int, source: int):
        position = np.clip((np.arange(size) + 0.5) / factor - 0.5, 0, source - 1)
        low = np.floor(position).astype(np.int64)
        return low, np.minimum(low + 1, source - 1), (position - low).astype(np.float32)

    r0, r1, wr = axis(shape[0], score.shape[0])
    c0, c1, wc = axis(shape[1], score.shape[1])
    score = score.astype(np.float32, copy=False)
    rows = score[r0] * (1 - wr[:, None]) + score[r1] * wr[:, None]
    return rows[:, c0] * (1 - wc) + rows[:, c1] * wc


def reference_binary_label(label: np.ndarray) -> np.ndarray:
    """Label conversion of the reference evaluator: {0, 1} labels are kept, otherwise 1 -> 0 and > 1 -> 1."""
    unique = np.unique(label)
    if unique.size == 2 and np.array_equal(unique, np.array([0, 1], dtype=unique.dtype)):
        return label.astype(np.uint8, copy=False)
    result = label.copy()
    result[result == 1] = 0
    result[result > 0] = 1
    return result.astype(np.uint8, copy=False)


def read_label(path: Path) -> np.ndarray:
    with rasterio.open(path) as src:
        return reference_binary_label(src.read(1))


def find_label(label_root: Path, event_name: str) -> Path:
    labels = sorted((label_root / event_name).glob("*label*.tif"))
    if len(labels) != 1:
        raise ValueError(f"expected one label in {label_root / event_name}, found {len(labels)}")
    return labels[0]


# ------------------------------------------------------------------------------------------------ evaluation
def weighted_metrics(true_positives, false_positives, positives: int, background_weight: float):
    """Recall, weighted precision tp / (tp + w fp) and F1 (zero when undefined)."""
    tp = np.asarray(true_positives, dtype=np.float64)
    fp = np.asarray(false_positives, dtype=np.float64)
    recall = np.divide(tp, positives, out=np.zeros_like(tp), where=positives != 0)
    denominator = tp + background_weight * fp
    precision = np.divide(tp, denominator, out=np.zeros_like(tp), where=denominator != 0)
    f1 = np.divide(2 * recall * precision, recall + precision, out=np.zeros_like(tp), where=(recall + precision) != 0)
    return recall, precision, f1


@dataclass(frozen=True)
class EventScores:
    category: str
    event: str
    score_path: Path
    label_path: Path

    def load(self) -> tuple[np.ndarray, np.ndarray]:
        score, label = np.load(self.score_path, mmap_mode="r"), read_label(self.label_path)
        if score.shape != label.shape:
            raise ValueError(f"{self.category}/{self.event}: score {score.shape} vs label {label.shape}")
        return score, label


def find_events(results_root: Path, label_root: Path, score_name: str) -> dict[str, list[EventScores]]:
    grouped: dict[str, list[EventScores]] = {}
    for path in sorted(results_root.glob(f"*/*/anomaly_{score_name}.npy")):
        category, event = path.parent.parent.name, path.parent.name
        grouped.setdefault(category, []).append(EventScores(category, event, path,
                                                            find_label(label_root, f"{category}/{event}")))
    if not grouped:
        raise ValueError(f"no */*/anomaly_{score_name}.npy below {results_root}")
    return grouped


def best_index(thresholds, recall, precision, f1) -> int:
    """F1 is the objective; ties prefer precision, then recall, then the higher threshold."""
    return int(np.lexsort((thresholds, recall, precision, f1))[-1])


def confusion_at(event: EventScores, threshold: float, chunk: int = 2_000_000) -> tuple[int, int, int, int]:
    score, label = event.load()
    score, label = score.reshape(-1), label.reshape(-1)
    tp = fp = 0
    for start in range(0, score.size, chunk):
        values, targets = np.asarray(score[start:start + chunk]), label[start:start + chunk]
        predicted = np.isfinite(values) & (values >= threshold)
        tp += int(np.count_nonzero(predicted & (targets == 1)))
        fp += int(np.count_nonzero(predicted & (targets == 0)))
    return tp, fp, int(label.sum()), int(label.size)


def histogram_threshold(events: list[EventScores], background_weight: float, bins: int, chunk: int = 2_000_000) -> float:
    """Best mean-F1 threshold among the edges of ``bins`` uniform bins spanning the category's finite scores."""
    low, high = np.inf, -np.inf
    for event in events:
        values = np.load(event.score_path).reshape(-1)
        values = values[np.isfinite(values)]
        low, high = min(low, float(values.min())), max(high, float(values.max()))
    if high <= low:
        return low
    edges = np.linspace(low, np.nextafter(high, np.inf), bins + 1, dtype=np.float64)
    scale = bins / (high - low)
    sums = np.zeros((3, bins))
    for event in events:
        score, label = event.load()
        score, label = score.reshape(-1), label.reshape(-1)
        histogram = np.zeros((bins, 2), np.int64)
        for start in range(0, score.size, chunk):
            values, targets = np.asarray(score[start:start + chunk]), label[start:start + chunk]
            valid = np.isfinite(values)
            index = np.clip(np.floor((values[valid] - low) * scale).astype(np.int64), 0, bins - 1)
            histogram += np.bincount(index * 2 + targets[valid], minlength=bins * 2).reshape(-1, 2)
        tp = np.cumsum(histogram[::-1, 1])[::-1]
        fp = np.cumsum(histogram[::-1, 0])[::-1]
        sums += np.stack(weighted_metrics(tp, fp, int(label.sum()), background_weight))
    sums /= len(events)
    return float(edges[best_index(edges[:-1], *sums)])


def exact_threshold(events: list[EventScores], background_weight: float, candidate_chunk: int = 500_000) -> float:
    """Best mean-F1 threshold over every distinct observed score of the category."""
    data = []
    for event in events:
        score, label = event.load()
        values, targets = np.asarray(score).reshape(-1), label.reshape(-1)
        finite = np.isfinite(values)
        order = np.argsort(values[finite], kind="stable")
        prefix = np.concatenate([[0], np.cumsum(targets[finite][order], dtype=np.int64)])
        data.append((values[finite][order], prefix, int(label.sum())))
    thresholds = np.unique(np.concatenate([d[0] for d in data]))
    best_key, best = None, float(thresholds[0])
    for start in range(0, thresholds.size, candidate_chunk):
        candidates = thresholds[start:start + candidate_chunk]
        sums = np.zeros((3, candidates.size))
        for scores, prefix, positives in data:
            index = np.searchsorted(scores, candidates, side="left")
            tp = prefix[-1] - prefix[index]
            sums += np.stack(weighted_metrics(tp, scores.size - index - tp, positives, background_weight))
        sums /= len(data)
        i = best_index(candidates, *sums)
        key = (sums[2, i], sums[1, i], sums[0, i], float(candidates[i]))
        if best_key is None or key > best_key:
            best_key, best = key, float(candidates[i])
    return best


def evaluate_score(results_root: Path, label_root: Path, score_name: str, output_dir: Path,
                   background_weight: float = 0.5, bins: int = 65_536, exact: bool = False) -> dict:
    """Tune one threshold per category and report recall / weighted precision / F1 (category macro averages)."""
    category_rows, event_rows = [], []
    for category, events in sorted(find_events(results_root, label_root, score_name).items()):
        threshold = exact_threshold(events, background_weight) if exact else \
            histogram_threshold(events, background_weight, bins)
        rows = []
        for event in events:  # recount the chosen operating point directly on the float maps
            tp, fp, positives, pixels = confusion_at(event, threshold)
            recall, precision, f1 = weighted_metrics(float(tp), float(fp), positives, background_weight)
            rows.append({"category": category, "event": event.event, "threshold": threshold, "recall": float(recall),
                         "weighted_precision": float(precision), "weighted_f1": float(f1), "true_positives": tp,
                         "false_positives": fp, "positive_pixels": positives, "pixels": pixels})
        category_rows.append({"category": category, "events": len(events), "threshold": threshold,
                              **{f"mean_{k}": float(np.mean([r[k] for r in rows]))
                                 for k in ("recall", "weighted_precision", "weighted_f1")}})
        event_rows += rows
        print(f"{score_name} {category:<18} threshold={threshold:.8g} recall={category_rows[-1]['mean_recall']:.4f} "
              f"precision={category_rows[-1]['mean_weighted_precision']:.4f} "
              f"F1={category_rows[-1]['mean_weighted_f1']:.4f}", flush=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in (("category_metrics.csv", category_rows), ("event_metrics.csv", event_rows)):
        with (output_dir / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    summary = {"score": score_name, "events": len(event_rows), "categories": len(category_rows),
               "background_weight": background_weight,
               "threshold_search": "exact_observed_scores" if exact else f"histogram_{bins}_bins",
               **{f"macro_{k}": float(np.mean([r[f'mean_{k}'] for r in category_rows]))
                  for k in ("recall", "weighted_precision", "weighted_f1")},
               "per_category": category_rows}
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    return summary
