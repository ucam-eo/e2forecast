"""S2_Embed: PASTIS Sentinel-2 L2A patches embedded per date with Prithvi-EO-2.0-300M (1024-D).

Layout::

    <embedding-root>/S2_<ID_PATCH>.npy        float16 (T, 8, 8, 1024): one 1024-D token per 16 x 16 px per date
    <metadata>                                PASTIS metadata.geojson (dates and patch polygon, EPSG:2154)
    <config-dir>/config.json                  benchmark definition (context_count = 37)
    <config-dir>/split_1461_486_486.json      patch split (train / val / test)
    <config-dir>/{val,test}_samples.npz       fixed (patch_id, token, target_index, context_index[37]) samples

A training sample is (patch, token, target date) with C earlier dates of the same token drawn at random
(distinct, sorted by time).  Token embeddings are L2-normalised.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path

import numpy as np
import pyproj
import torch

from ..utils import l2_normalize

EMBED_DIM = 1024
GRID = 8  # tokens per side (128 px / 16 px)


def load_metadata(path: Path) -> dict[str, dict]:
    """patch id -> {'dates': [date, ...] chronological, 'corners': (4, 2) EPSG:2154 TL, TR, BR, BL}."""
    out = {}
    for feature in json.loads(Path(path).read_text())["features"]:
        props = feature["properties"]
        dates = [props["dates-S2"][str(i)] for i in range(len(props["dates-S2"]))]
        dates = [date(d // 10000, d // 100 % 100, d % 100) for d in dates]
        if dates != sorted(dates):
            raise ValueError(f"patch {props['ID_PATCH']}: dates are not chronological")
        geometry = feature["geometry"]
        outer = geometry["coordinates"][0][0] if geometry["type"] == "MultiPolygon" else geometry["coordinates"][0]
        ring = np.array(outer[:4], dtype=np.float64)  # TL, TR, BR, BL (5th point closes the ring)
        if not (ring[0, 1] > ring[3, 1] and ring[1, 0] > ring[0, 0]):
            raise ValueError(f"patch {props['ID_PATCH']}: unexpected polygon corner order")
        out[str(props["ID_PATCH"])] = {"dates": dates, "corners": ring}
    return out


def token_lonlat(corners: np.ndarray) -> np.ndarray:
    """(64, 2) WGS84 lon/lat of every token centre (row-major), bilinear inside the patch polygon."""
    tl, tr, br, bl = corners
    centre = (np.arange(GRID) + 0.5) / GRID
    v, u = np.meshgrid(centre, centre, indexing="ij")
    u, v = u[..., None], v[..., None]
    xy = (1 - v) * ((1 - u) * tl + u * tr) + v * ((1 - u) * bl + u * br)
    lon, lat = pyproj.Transformer.from_crs(2154, 4326, always_xy=True).transform(xy[..., 0].ravel(), xy[..., 1].ravel())
    return np.stack((lon, lat), -1).astype(np.float64)


class Patch:
    def __init__(self, root: Path, name: str, meta: dict) -> None:
        raw = np.load(root / f"S2_{name}.npy")
        if raw.shape != (len(meta["dates"]), GRID, GRID, EMBED_DIM):
            raise ValueError(f"patch {name}: embedding shape {raw.shape} does not match {len(meta['dates'])} dates")
        self.name = name
        self.values = l2_normalize(raw.reshape(len(raw), GRID * GRID, EMBED_DIM)).astype(np.float16)  # (T, 64, D)
        self.doys = np.array([d.timetuple().tm_yday for d in meta["dates"]], dtype=np.int64)
        self.years = np.array([d.year for d in meta["dates"]], dtype=np.int64)
        self.lonlat = token_lonlat(meta["corners"])


class Benchmark:
    """Paths of the benchmark files, resolved relative to the config directory."""

    def __init__(self, config: Path, embedding_root: Path, metadata: Path) -> None:
        self.config = json.loads(config.read_text())
        self.directory, self.embedding_root, self.metadata_path = config.parent, embedding_root, metadata
        self.context_count = self.config["context_count"]
        self.split = json.loads((self.directory / Path(self.config["splits"]["file"]).name).read_text())["patches"]
        self._metadata = None

    @classmethod
    def from_root(cls, root: Path) -> "Benchmark":
        """Standard layout: <root>/benchmark_config, <root>/embeddings_prithvi_eo_2_300M, <root>/PASTIS."""
        return cls(root / "benchmark_config/config.json", root / "embeddings_prithvi_eo_2_300M",
                   root / "PASTIS/metadata.geojson")

    @property
    def metadata(self) -> dict[str, dict]:
        if self._metadata is None:
            self._metadata = load_metadata(self.metadata_path)
        return self._metadata

    def sample_file(self, split: str) -> Path:
        return self.directory / f"{split}_samples.npz"

    def load_patches(self, names: list[str], workers: int) -> list[Patch]:
        with ThreadPoolExecutor(workers) as pool:
            return list(pool.map(lambda n: Patch(self.embedding_root, n, self.metadata[n]), names))


class TrainBank:
    """All dates and tokens of the train-split patches in RAM, with random-context sampling."""

    def __init__(self, bench: Benchmark, context_count: int, seed: int, workers: int = 16, max_patches: int = 0) -> None:
        self.context_count, self.rng = context_count, np.random.default_rng(seed)
        names = bench.split["train"][:max_patches or None]
        started = time.time()
        self.patches = [p for p in bench.load_patches(names, workers) if len(p.values) > context_count]
        if not self.patches:
            raise RuntimeError("no usable training patches")
        self.summary = {"train_patches": len(self.patches), "dates": int(sum(len(p.values) for p in self.patches))}
        print(f"loaded {len(self.patches)} train patches, {self.summary['dates']} dates "
              f"in {time.time() - started:.0f}s", flush=True)

    def sample(self, batch_size: int) -> dict[str, torch.Tensor]:
        """Patch uniform; target date uniform among dates >= C; C random earlier dates; token uniform."""
        C = self.context_count
        obs = np.empty((batch_size, C, EMBED_DIM), np.float32)
        obs_doys, obs_years = np.empty((batch_size, C), np.int64), np.empty((batch_size, C), np.int64)
        target = np.empty((batch_size, EMBED_DIM), np.float32)
        query_doy, query_year = np.empty(batch_size, np.int64), np.empty(batch_size, np.int64)
        lonlat = np.empty((batch_size, 2), np.float64)
        for row in range(batch_size):
            patch = self.patches[int(self.rng.integers(len(self.patches)))]
            t = int(self.rng.integers(C, len(patch.values)))
            chosen = np.sort(self.rng.choice(t, C, replace=False))
            p = int(self.rng.integers(GRID * GRID))
            obs[row], obs_doys[row], obs_years[row] = patch.values[chosen, p], patch.doys[chosen], patch.years[chosen]
            target[row], query_doy[row], query_year[row] = patch.values[t, p], patch.doys[t], patch.years[t]
            lonlat[row] = patch.lonlat[p]
        return {k: torch.from_numpy(v) for k, v in
                {"obs": obs, "obs_doys": obs_doys, "obs_years": obs_years, "target": target, "query_doy": query_doy,
                 "query_year": query_year, "lonlat": lonlat}.items()}


class SampleSet:
    """Fixed validation / test samples; ``limit`` uses the first N (the files are in random order)."""

    def __init__(self, bench: Benchmark, split: str, limit: int = 0, workers: int = 16,
                 device: torch.device = torch.device("cpu")) -> None:
        arrays = np.load(bench.sample_file(split))
        n = limit or len(arrays["patch_id"])
        patch_id = arrays["patch_id"][:n].astype(np.int64)
        ids = np.unique(patch_id)
        patches = bench.load_patches([str(i) for i in ids], workers)
        lengths = np.array([len(p.values) for p in patches])
        offsets = np.concatenate([[0], np.cumsum(lengths)[:-1]])
        row = np.searchsorted(ids, patch_id)
        target = arrays["target_index"][:n].astype(np.int64)
        context = arrays["context_index"][:n].astype(np.int64)
        if (target >= lengths[row]).any() or context.shape[1] != bench.context_count:
            raise ValueError(f"{split} sample indices do not match the patch dates")
        to = lambda a: torch.from_numpy(np.ascontiguousarray(a)).to(device)  # noqa: E731
        self.values = to(np.concatenate([p.values for p in patches]))  # (all dates, 64, D) fp16
        self.doys = to(np.concatenate([p.doys for p in patches]))
        self.years = to(np.concatenate([p.years for p in patches]))
        self.lonlat = to(np.stack([p.lonlat for p in patches]))  # (P, 64, 2)
        self.row, self.offset = to(row), to(offsets[row])
        self.token, self.target, self.context = to(arrays["token"][:n].astype(np.int64)), to(target), to(context)
        self.count, self.patches = n, len(ids)

    def batch(self, start: int, stop: int) -> dict[str, torch.Tensor]:
        context = self.offset[start:stop, None] + self.context[start:stop]
        target = self.offset[start:stop] + self.target[start:stop]
        token = self.token[start:stop]
        return {"obs": self.values[context, token[:, None]].float(), "obs_doys": self.doys[context],
                "obs_years": self.years[context], "query_doy": self.doys[target], "query_year": self.years[target],
                "lonlat": self.lonlat[self.row[start:stop], token],
                "target": self.values[target, token].float()}
