"""S1_Embed: OLMoEarth-nano (patch 5, 128-D) embeddings of descending Sentinel-1 RTC scenes.

Layout (``--data-root``)::

    embeddings/embedding_manifest.json
    embeddings/<site>/<year>/<YYYYMMDDTHHMMSS.ffffffZ_...>.tif     128-band embedding grid of one scene
    config/split.json                                              {"train_sites": [...], ...}
    config/{val,test}_config.json + *_items.npz                    fixed evaluation items

Protocol: train on 2024 only; a training item is (site, pixel, target scene) with C earlier 2024 scenes of
the same pixel drawn at random (distinct, sorted by time).  Evaluation uses the fixed 2025 test items; every
item stores 12 earlier scenes and a context of length C uses the C most recent of them.
"""

from __future__ import annotations

import json
import re
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterator

import numpy as np
import pyproj
import rasterio
import torch

from ..utils import l2_normalize

EMBED_DIM = 128
TRAIN_YEAR, EVAL_YEAR = 2024, 2025
MIN_SERIES_LENGTH = 13
DATE_RE = re.compile(r"^(\d{8})T(\d{6})\.(\d{6})Z_")


@dataclass(frozen=True)
class Scene:
    path: str
    datetime: str
    day_of_year: int


def scene_datetime(path: Path) -> datetime:
    match = DATE_RE.match(path.name)
    if not match:
        raise ValueError(f"expected a compact acquisition timestamp at the start of {path.name}")
    return datetime.strptime("".join(match.groups()[:2]), "%Y%m%d%H%M%S")


def discover_site(root: Path, site: str, year: int) -> list[Scene]:
    """Chronological scenes of one site and year; same-timestamp reprocessings count once."""
    scenes = []
    for path in sorted((root / site / str(year)).glob("*.tif")):
        stamp = scene_datetime(path)
        scenes.append(Scene(str(path), stamp.isoformat(), int(stamp.strftime("%j"))))
    scenes.sort(key=lambda s: (s.datetime, s.path))
    unique: list[Scene] = []
    for scene in scenes:
        if not unique or unique[-1].datetime != scene.datetime:
            unique.append(scene)
    return unique


def list_sites(root: Path, min_series: int = MIN_SERIES_LENGTH) -> list[str]:
    """Sites with more than ``min_series`` scenes in 2024 or in 2025."""
    manifest = json.loads((root / "embedding_manifest.json").read_text())
    if not manifest.get("complete"):
        raise RuntimeError(f"embedding run is incomplete: {root}")
    names = sorted({record["output"].split("/")[0] for record in manifest["records"]})
    return [n for n in names
            if max(len(discover_site(root, n, TRAIN_YEAR)), len(discover_site(root, n, EVAL_YEAR))) > min_series]


def read_scene(path: str, pixels: np.ndarray) -> np.ndarray:
    """Normalised (len(pixels), 128) float16 embeddings of the given row-major pixel indices."""
    with rasterio.open(path) as src:
        values = np.moveaxis(src.read(out_dtype="float32"), 0, -1).reshape(-1, EMBED_DIM)
    return l2_normalize(values[pixels]).astype(np.float16)


def pixel_lonlat(crs, transform: rasterio.Affine, width: int, pixels: np.ndarray) -> np.ndarray:
    rows, cols = np.divmod(pixels, width)
    x, y = transform * (cols + 0.5, rows + 0.5)
    lon, lat = pyproj.Transformer.from_crs(crs, 4326, always_xy=True).transform(x, y)
    return np.stack((lon, lat), -1).astype(np.float64)


class Site:
    def __init__(self, name: str, scenes: list[Scene], values: np.ndarray, lonlat: np.ndarray) -> None:
        self.name, self.scenes, self.values, self.lonlat = name, scenes, values, lonlat  # values: (T, P, 128) fp16
        self.doys = np.array([s.day_of_year for s in scenes], dtype=np.int64)


class TrainBank:
    """All 2024 scenes of the training sites (a fixed random pixel subset per site) with random-context sampling."""

    def __init__(self, root: Path, split_file: Path, context_count: int, pixels_per_site: int, seed: int,
                 workers: int = 8, min_series: int = MIN_SERIES_LENGTH, sites: list[str] | None = None) -> None:
        self.context_count, self.rng = context_count, np.random.default_rng(seed)
        train_sites = set(json.loads(split_file.read_text())["train_sites"])
        names = [n for n in list_sites(root, min_series) if n in train_sites]
        self.sites: list[Site] = []
        started = time.time()
        with ThreadPoolExecutor(workers) as pool:
            for index, name in enumerate(names):
                if sites and name not in sites:
                    continue
                scenes = discover_site(root, name, TRAIN_YEAR)
                if len(scenes) < context_count + 1:
                    print(f"skip {name}: fewer than {context_count + 1} {TRAIN_YEAR} scenes", flush=True)
                    continue
                with rasterio.open(scenes[0].path) as src:
                    crs, transform, width, count = src.crs, src.transform, src.width, src.width * src.height
                pixels = np.arange(count)
                if 0 < pixels_per_site < count:
                    pixels = np.sort(np.random.default_rng([seed, index]).choice(count, pixels_per_site, replace=False))
                values = np.stack(list(pool.map(lambda s: read_scene(s.path, pixels), scenes)))
                self.sites.append(Site(name, scenes, values, pixel_lonlat(crs, transform, width, pixels)))
                print(f"[{index + 1}/{len(names)}] {name}: {len(scenes)} scenes x {len(pixels)} px "
                      f"({time.time() - started:.0f}s)", flush=True)
        if not self.sites:
            raise RuntimeError("no usable training sites")
        self.summary = {"sites": [s.name for s in self.sites], "scenes": sum(len(s.scenes) for s in self.sites),
                        "pixels_per_site": int(self.sites[0].values.shape[1])}

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Site uniform; target uniform among scenes >= C; C random earlier scenes; pixel uniform."""
        C = self.context_count
        obs = np.empty((batch_size, C, EMBED_DIM), np.float32)
        obs_doys = np.empty((batch_size, C), np.int64)
        target = np.empty((batch_size, EMBED_DIM), np.float32)
        query_doy = np.empty(batch_size, np.int64)
        lonlat = np.empty((batch_size, 2), np.float64)
        for row in range(batch_size):
            site = self.sites[int(self.rng.integers(len(self.sites)))]
            t = int(self.rng.integers(C, len(site.scenes)))
            chosen = np.sort(self.rng.choice(t, C, replace=False))
            p = int(self.rng.integers(site.values.shape[1]))
            obs[row], obs_doys[row] = site.values[chosen, p], site.doys[chosen]
            target[row], query_doy[row], lonlat[row] = site.values[t, p], site.doys[t], site.lonlat[p]
        return {k: torch.from_numpy(v) for k, v in
                {"obs": obs, "obs_doys": obs_doys, "target": target, "query_doy": query_doy, "lonlat": lonlat}.items()}


class EvalItems:
    """Fixed evaluation items (``val_config.json`` / ``test_config.json``).

    Item = (site, pixel, target scene, 12 earlier scenes); a context of length k is the k most recent of the 12,
    so all context lengths share the same pixel and target date.
    """

    def __init__(self, config: Path, root: Path, workers: int = 16, items_per_site: int = 0) -> None:
        self.meta = json.loads(config.read_text())
        arrays = np.load(config.with_name(self.meta["items_file"]))
        self.root, self.workers = root, workers
        self.names = [s["id"] for s in self.meta["sites"]]
        self.groups = {s["id"]: s["group"] for s in self.meta["sites"]}
        self.items = []
        for i in range(len(self.names)):
            chosen = np.flatnonzero(arrays["site"] == i)
            if items_per_site:  # items are stored in random order: the first ones are a fixed random subsample
                chosen = chosen[:items_per_site]
            self.items.append({key: arrays[key][chosen].astype(np.int64) for key in ("pixel", "target", "context")})
        self.count = sum(len(x["pixel"]) for x in self.items)
        self.max_context = self.meta["max_context"]

    def load_site(self, i: int, device: torch.device) -> dict[str, torch.Tensor | np.ndarray]:
        site, used = self.meta["sites"][i], np.unique(self.items[i]["pixel"])
        with ThreadPoolExecutor(self.workers) as pool:
            values = np.stack(list(pool.map(lambda s: read_scene(str(self.root / s["path"]), used), site["scenes"])))
        grid = site["grid"]
        lonlat = pixel_lonlat(grid["crs"], rasterio.Affine(*grid["transform"]), grid["width"], used)
        return {"values": torch.from_numpy(values).to(device), "lonlat": torch.from_numpy(lonlat).to(device),
                "doys": torch.tensor([s["day_of_year"] for s in site["scenes"]], device=device), "used": used}

    def batches(self, context_count: int, batch_size: int, device: torch.device,
                sites: list[int] | None = None) -> Iterator[tuple[int, dict[str, torch.Tensor]]]:
        """Yield (site index, batch) with keys obs, obs_doys, query_doy, lonlat, target."""
        if not 1 <= context_count <= self.max_context:
            raise ValueError(f"context length must be in 1..{self.max_context}")
        for i in (range(len(self.names)) if sites is None else sites):
            data, items = self.load_site(i, device), self.items[i]
            local = torch.from_numpy(np.searchsorted(data["used"], items["pixel"])).to(device)
            target = torch.from_numpy(items["target"]).to(device)
            context = torch.from_numpy(items["context"][:, -context_count:]).to(device)
            values, doys = data["values"], data["doys"]
            for start in range(0, len(local), batch_size):
                px, tg, cx = (x[start:start + batch_size] for x in (local, target, context))
                yield i, {"obs": values[cx, px[:, None]].float(), "obs_doys": doys[cx], "query_doy": doys[tg],
                          "lonlat": data["lonlat"][px], "target": values[tg, px].float()}
            del data, values
