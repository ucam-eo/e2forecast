"""Location -> embedding data for the INR experiments.

Webdataset shards with one sample per image: ``.npy`` = 256-D SatCLIP image embedding (L2-normalised here) and
``.txt`` = ``"lon,lat[,YYYY-MM-DDTHH:MM:SS,...]"``.  The S2-100K shards are
``satclip_embeddings_location_{000493..000542}.tar``, the SSL4EO-S12 shards (with acquisition dates)
``satclip_embeddings_location_{000000..000492}.tar``.  Shards are split at random into train / val / test.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime

import torch
import webdataset as wds
from torch.utils.data import DataLoader

EMBED_DIM = 256


def add_shard_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--shards", required=True, help="webdataset brace pattern, e.g. '.../name_{000493..000542}.tar'")
    parser.add_argument("--train-split", type=float, default=0.9, help="fraction of shards for training")
    parser.add_argument("--val-split", type=float, default=0.05, help="fraction of shards for validation; rest = test")
    parser.add_argument("--split-seed", type=int, default=84513)
    parser.add_argument("--num-workers", type=int, default=4)


def split_shards(url: str, train: float, val: float, seed: int) -> tuple[list[str], list[str], list[str]]:
    """Shard-level random split of a ``{start..end}`` brace pattern (remainder = test; empty test falls back to val)."""
    match = re.search(r"\{([0-9]+)\.\.([0-9]+)\}", url)
    if not match:
        raise ValueError("--shards must contain a range like {000493..000542}")
    start, end, width = int(match.group(1)), int(match.group(2)), len(match.group(1))
    template = url[:match.start()] + "{}" + url[match.end():]
    indices = list(range(start, end + 1))
    order = torch.randperm(len(indices), generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [indices[i] for i in order]
    n_train, n_val = int(len(indices) * train), int(len(indices) * val)
    names = lambda part: [template.format(str(i).zfill(width)) for i in part]  # noqa: E731
    test = shuffled[n_train + n_val:] or shuffled[n_train:n_train + n_val]
    return names(shuffled[:n_train]), names(shuffled[n_train:n_train + n_val]), names(test)


def _decode(sample: dict) -> tuple[torch.Tensor, str]:
    embedding = torch.as_tensor(sample["npy"]).float().squeeze()
    caption = sample["txt"]
    return embedding / embedding.norm(dim=-1, keepdim=True).clamp(min=1e-8), \
        caption.decode("utf-8", errors="replace") if isinstance(caption, bytes) else str(caption)


def _collate(batch):
    embeddings, captions = zip(*batch)
    return torch.stack(embeddings), list(captions)


def make_loader(urls: list[str], batch_size: int, num_workers: int, train: bool, shuffle_buffer: int = 2000,
                samples_per_epoch: int | None = None) -> DataLoader:
    """Yields (embeddings (B, 256), captions); training shuffles shards and samples, evaluation is in shard order."""
    pipeline = wds.DataPipeline(wds.SimpleShardList(urls))
    if train:
        pipeline.append(wds.filters.shuffle(1000))
    pipeline.append(wds.tarfile_to_samples(handler=wds.handlers.reraise_exception))
    pipeline.append(wds.decode(handler=wds.handlers.reraise_exception))
    pipeline.append(wds.map(_decode))
    if train and shuffle_buffer > 0:
        pipeline.append(wds.shuffle(shuffle_buffer))
    if train and samples_per_epoch:
        pipeline = pipeline.with_epoch(nsamples=samples_per_epoch)
    return DataLoader(pipeline, batch_size=batch_size, num_workers=num_workers, pin_memory=True, shuffle=False,
                      collate_fn=_collate)


def parse_captions(captions: list[str], with_date: bool) -> dict[str, torch.Tensor]:
    """-> {"lonlat": (B, 2) degrees} and, with ``with_date``, {"query_date": (B, 2) = (0, day of year)}."""
    lonlat, doy = [], []
    for caption in captions:
        fields = caption.split(",")
        lonlat.append([float(fields[0]), float(fields[1])])
        if with_date:
            if len(fields) < 3:
                raise ValueError(f"caption has no date field: {caption!r}")
            doy.append(datetime.fromisoformat(fields[2].strip()).timetuple().tm_yday)
    out = {"lonlat": torch.tensor(lonlat, dtype=torch.float32)}
    if with_date:
        doy = torch.tensor(doy, dtype=torch.float32)
        out["query_date"] = torch.stack((torch.zeros_like(doy), doy), -1)
    return out
