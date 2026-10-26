"""GSA anomaly change detection on OLMoEarth-nano (patch 5, 128-D) embeddings of Sentinel-2 L2A images.

The event manifest (``data/gsa_manifest.json``) lists, for each of the 55 events, its embedding rasters in
chronological order (paths relative to ``--embedding-root``), acquisition dates, raster size and WGS84 bounds.
Every event's chronologically last image is the post-event (anomalous) observation and is never used for
training: a training item is a normal target plus three random earlier normal observations of the same pixel.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import rasterio.windows
import torch

from ..utils import l2_normalize

EMBED_DIM = 128
CONTEXT_COUNT = 3


@dataclass(frozen=True)
class Event:
    name: str  # <category>/<event>
    images: tuple[str, ...]  # chronological; absolute after ``load_manifest``
    dates: tuple[str, ...]  # ISO dates
    height: int
    width: int
    bounds_wgs84: tuple[float, float, float, float]  # left, bottom, right, top

    @property
    def category(self) -> str:
        return self.name.split("/")[0]

    def day_of_year(self, step: int) -> int:
        return date.fromisoformat(self.dates[step]).timetuple().tm_yday


def load_manifest(path: Path, embedding_root: Path) -> list[Event]:
    payload = json.loads(Path(path).read_text())
    return [Event(x["name"], tuple(str(embedding_root / p) for p in x["images"]), tuple(x["dates"]),
                  x["height"], x["width"], tuple(x["bounds_wgs84"])) for x in payload["events"]]


class RasterCache:
    """Keeps every opened raster open (each event is read many times during training)."""

    def __init__(self) -> None:
        self.files = {}

    def get(self, path: str):
        if path not in self.files:
            self.files[path] = rasterio.open(path)
        return self.files[path]

    def close(self) -> None:
        for src in self.files.values():
            src.close()
        self.files.clear()


def read_pixels(src, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """(n, bands) float32 values at (rows, cols); reads are grouped by raster block."""
    result = np.empty((len(rows), src.count), dtype=np.float32)
    block_h, block_w = src.block_shapes[0]
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (r, c) in enumerate(zip(rows, cols)):
        groups.setdefault((int(r) // block_h, int(c) // block_w), []).append(i)
    for (br, bc), indices in groups.items():
        window = rasterio.windows.Window(bc * block_w, br * block_h, min(block_w, src.width - bc * block_w),
                                         min(block_h, src.height - br * block_h))
        block = src.read(window=window, out_dtype="float32")
        ii = np.asarray(indices)
        result[ii] = block[:, rows[ii] - br * block_h, cols[ii] - bc * block_w].T
    return result


def lonlat_for(event: Event, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    left, bottom, right, top = event.bounds_wgs84
    lon = left + (cols + .5) / event.width * (right - left)
    lat = top - (rows + .5) / event.height * (top - bottom)
    return np.stack((lon, lat), -1).astype(np.float64)


class Sampler:
    """Random training items: target = any normal step with >= 3 earlier steps (the last image is excluded).

    Pixels of a batch are grouped into ``events_per_batch`` events and one 128 x 128 block per event, which keeps
    the marginal distribution uniform while sharing raster-block decompression across items.
    """

    def __init__(self, events: list[Event], seed: int, events_per_batch: int = 8, spatial_block: int = 128) -> None:
        # Events with only four images have no normal target after three normal inputs.
        self.events = [event for event in events if len(event.images) >= CONTEXT_COUNT + 2]
        if not self.events:
            raise ValueError("no event has three earlier normal inputs plus a normal target")
        self.rng, self.cache = np.random.default_rng(seed), RasterCache()
        self.events_per_batch, self.spatial_block = events_per_batch, spatial_block

    def sample(self, batch: int) -> dict[str, torch.Tensor]:
        event_ids = np.empty(batch, np.int64)
        all_rows, all_cols = np.empty(batch, np.int64), np.empty(batch, np.int64)
        for indices in np.array_split(np.arange(batch), min(self.events_per_batch, batch)):
            event_id = int(self.rng.integers(0, len(self.events)))
            event = self.events[event_id]
            row_start = int(self.rng.integers(0, event.height)) // self.spatial_block * self.spatial_block
            col_start = int(self.rng.integers(0, event.width)) // self.spatial_block * self.spatial_block
            event_ids[indices] = event_id
            all_rows[indices] = self.rng.integers(row_start, min(row_start + self.spatial_block, event.height), len(indices))
            all_cols[indices] = self.rng.integers(col_start, min(col_start + self.spatial_block, event.width), len(indices))
        obs = np.empty((batch, CONTEXT_COUNT, EMBED_DIM), np.float32)
        target = np.empty((batch, EMBED_DIM), np.float32)
        obs_dates = np.empty((batch, CONTEXT_COUNT, 2), np.float32)  # (relative order -3..-1, day of year)
        query_date = np.zeros((batch, 2), np.float32)  # (0, day of year)
        lonlat = np.empty((batch, 2), np.float64)
        obs_dates[:, :, 0] = np.arange(-CONTEXT_COUNT, 0)
        for event_id in np.unique(event_ids):
            event = self.events[int(event_id)]
            idx = np.flatnonzero(event_ids == event_id)
            rows, cols = all_rows[idx], all_cols[idx]
            targets = self.rng.integers(CONTEXT_COUNT, len(event.images) - 1, len(idx))
            chosen = np.sort(np.stack([self.rng.choice(t, CONTEXT_COUNT, replace=False) for t in targets]), axis=1)
            for j in range(CONTEXT_COUNT):
                for step in np.unique(chosen[:, j]):
                    take = np.flatnonzero(chosen[:, j] == step)
                    obs[idx[take], j] = read_pixels(self.cache.get(event.images[int(step)]), rows[take], cols[take])
                    obs_dates[idx[take], j, 1] = event.day_of_year(int(step))
            for step in np.unique(targets):
                take = np.flatnonzero(targets == step)
                target[idx[take]] = read_pixels(self.cache.get(event.images[int(step)]), rows[take], cols[take])
                query_date[idx[take], 1] = event.day_of_year(int(step))
            lonlat[idx] = lonlat_for(event, rows, cols)
        return {k: torch.from_numpy(v) for k, v in
                {"obs": l2_normalize(obs, False), "target": l2_normalize(target, False), "obs_dates": obs_dates,
                 "query_date": query_date, "lonlat": lonlat}.items()}
