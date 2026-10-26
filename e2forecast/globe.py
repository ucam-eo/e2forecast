"""Helpers for the global embedding-field figures of the INR experiments (needs matplotlib, scikit-learn, cartopy)."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch

from .model import E2Forecast


def global_grid(step: float) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """Flattened (lat, lon) of a regular grid with ``step`` degrees and its (rows, cols) shape."""
    lon_grid, lat_grid = np.meshgrid(np.arange(-180, 180, step), np.arange(-90, 90 + step, step))
    return lat_grid.ravel(), lon_grid.ravel(), lat_grid.shape


@torch.no_grad()
def embed_grid(model: E2Forecast, lats: np.ndarray, lons: np.ndarray, batch_size: int = 3000,
               day_of_year: float | None = None, flow_steps: int = 0, seed: int = 0) -> np.ndarray:
    """Normalised Gaussian mean (``flow_steps == 0``) or one flow sample per grid point -> (N, D) float32."""
    device = next(model.parameters()).device
    coordinates = torch.from_numpy(np.stack([lons, lats], 1).astype(np.float32))
    generator = torch.Generator(device=device).manual_seed(seed)
    out = []
    for start in range(0, len(coordinates), batch_size):
        cond = {"lonlat": coordinates[start:start + batch_size].to(device)}
        if day_of_year is not None:
            cond["query_date"] = torch.tensor([0.0, day_of_year], device=device).expand(len(cond["lonlat"]), -1)
        embedding = model.point_forecast(**cond) if flow_steps == 0 else model.sample(flow_steps, generator, **cond)
        out.append(embedding.float().cpu().numpy())
    return np.concatenate(out)


def land_mask(lats: np.ndarray, lons: np.ndarray) -> np.ndarray:
    """(N,) bool from the Natural Earth 10 m land polygons shipped with cartopy (downloaded on first use)."""
    import cartopy.io.shapereader as shapereader
    import shapely
    from shapely.ops import unary_union

    land = unary_union(list(shapereader.Reader(shapereader.natural_earth("10m", "physical", "land")).geometries()))
    shapely.prepare(land)
    return shapely.contains_xy(land, lons, lats)


def pcs_to_rgb(pcs: np.ndarray, low: np.ndarray, high: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    return np.power(((pcs - low) / (high - low + 1e-8)).clip(0, 1), gamma)


def plot_globes(images: list[np.ndarray], titles: list[str], path: Path, columns: int, dpi: int = 300,
                land: np.ndarray | None = None) -> None:
    """Robinson-projection panels of (rows, cols, 3) RGB images; with ``land`` the ocean is transparent."""
    import cartopy.crs as ccrs
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = int(np.ceil(len(images) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 3.07 * rows), squeeze=False,
                                subplot_kw={"projection": ccrs.Robinson()}, facecolor="none")
    figure.subplots_adjust(left=0.003, right=0.997, bottom=0.003, top=0.955, wspace=0.01, hspace=0.22)
    for axis, image, title in zip(axes.ravel(), images, titles):
        if land is not None:
            image = np.concatenate([image, land.reshape(image.shape[:2] + (1,)).astype(image.dtype)], -1)
        axis.set_global()
        axis.patch.set_alpha(0)
        axis.spines["geo"].set_visible(False)
        axis.coastlines(linewidth=0.6, color="#333333", alpha=0.6)
        axis.imshow(image, origin="lower", extent=[-180, 180, -90, 90], transform=ccrs.PlateCarree(),
                    interpolation="nearest")
        if title:
            axis.set_title(title, fontsize=22, pad=3)
    for axis in axes.ravel()[len(images):]:
        axis.set_visible(False)
    os.makedirs(Path(path).parent, exist_ok=True)
    figure.savefig(path, dpi=dpi, bbox_inches="tight", transparent=True)
    plt.close(figure)
    print(f"saved {path}")
