# E²Forecast

Code for **Forecasting How Earth's Surface Changes in Embedding Space**.

E²Forecast forecasts the embedding of a location at any query date from an irregular history of
foundation-model embeddings. One masked Transformer pass gives two branches:

- a **probability branch**: a diagonal Gaussian `N(μ, diag(σ²))`, used for the point forecast `μ/‖μ‖` and the per-dimension uncertainty;
- a **generation branch**: a rectified flow with x-prediction, used for ensemble forecasts in `S` Euler steps.

The same network detects anomalous change: it compares a new observation with the Gaussian forecast and with the
closest ensemble forecast. With an empty history it becomes an implicit neural representation (INR) of the
global embedding field.

```
[location | C observations | query date | h_μ | h_σ | flow time t | noisy target z_t | h_query]
 └────────────── probability estimation ──────────────┘ └──────────── generation ────────────┘
          (cannot attend to the generation tokens)
```

## Repository layout

```
e2forecast/                 installable package
  model.py                  E2ForecastNetwork, E2Forecast (loss, Gaussian forecast, flow sampling), encodings
  location.py               fixed GeoCLIP random-Fourier-feature location encoder
  training.py               shared training loop (AdamW, checkpoints, resume, validation)
  metrics.py                z-score, point / ensemble cosine distance, CRPS
  anomaly.py                anomaly score, upsampling, GSA threshold evaluation
  checkpoint.py             checkpoint format and loading
  globe.py                  global-field plotting helpers
  data/                     s1_embed.py, s2_embed.py, gsa.py, satclip.py
scripts/
  s1_embed/                 train.py, evaluate.py
  s2_embed/                 train.py, evaluate.py
  gsa/                      train.py, infer.py, evaluate.py
  inr/                      train.py, evaluate.py, downstream.py, plot_global_field.py, plot_global_monthly.py
  export_checkpoint.py      convert checkpoints of the original research code to the release format
data/gsa_manifest.json      the 55 GSA events (relative paths, dates, bounds)
```

## Installation

```bash
conda create -n e2forecast python=3.12 && conda activate e2forecast
pip install torch            # pick the build that matches your CUDA version
pip install -r requirements.txt
pip install -e .
```

## Checkpoints

| file | experiment | paper | selection |
| --- | --- | --- | --- |
| `s1_embed.pt` | S1_Embed, 7 context observations, 128-D | forecasting table, S1_Embed | last epoch (600) |
| `s2_embed.pt` | S2_Embed, 37 context observations, 1024-D | forecasting table, S2_Embed | last epoch (72) |
| `gsa.pt` | GSA, 3 context observations, 128-D | GSA table | last epoch (300) |
| `inr.pt` | INR on S2-100K SatCLIP embeddings, 256-D | INR regression table, global-field figure | best validation epoch (108) |
| `inr_date.pt` | INR + query date on SSL4EO-S12, 256-D | monthly global-field figure | best validation epoch (288) |

Place the files in `checkpoints/`. Each one stores its model configuration, so every script rebuilds the network
from the checkpoint alone.

## Data

Checkpoints and data are on HuggingFace: https://huggingface.co/JingtaoLi/E2Forecast and https://huggingface.co/datasets/JingtaoLi/E2Forecast-data (S1_Embed, S2_Embed, INR S2-100K shards).
Download them into `checkpoints/` and `data/`:

```bash
hf download JingtaoLi/E2Forecast --local-dir checkpoints
hf download JingtaoLi/E2Forecast-data --repo-type dataset --local-dir data
```

The GSA data and the dated SSL4EO-S12 shards (used for `inr_date.pt`) are not released yet.

| experiment | expected layout |
| --- | --- |
| S1_Embed | `<root>/embeddings/` (OLMoEarth-nano patch-5 rasters per site and year, `embedding_manifest.json`) and `<root>/config/` (`split.json`, `{val,test}_config.json`, `*_items.npz`) |
| S2_Embed | `<root>/embeddings_prithvi_eo_2_300M/S2_<patch>.npy`, `<root>/benchmark_config/` (`config.json`, split, `{val,test}_samples.npz`) and `<root>/PASTIS/metadata.geojson` |
| GSA | patch-5 OLMoEarth-nano embedding rasters (`--embedding-root`, paths as in `data/gsa_manifest.json`) and the original 10 m GSA folders with `*label*.tif` (`--label-root`) |
| INR | webdataset shards with `.npy` (SatCLIP embedding) and `.txt` (`lon,lat[,date]`); downstream CSVs `labels_population.csv`, `labels_elevation.csv` (SatCLIP) and `stationDataAll.csv` (Hooker et al., 2018) |

## Reproducing the paper

All commands run from the repository root.

### Forecasting on S1_Embed and S2_Embed

```bash
python scripts/s1_embed/evaluate.py --data-root data/S1_Embed --checkpoint checkpoints/s1_embed.pt \
    --output-dir results/s1_embed
python scripts/s2_embed/evaluate.py --data-root data/S2_Embed --checkpoint checkpoints/s2_embed.pt \
    --output-dir results/s2_embed --samples-limit 100000
```

The Gaussian branch gives the mean and std of `z`, the point cosine distance and the CRPS (closed form). `J = 10`
ensemble forecasts with `S = 10` Euler steps give the ensemble cosine distance (closest member). S1_Embed uses all
1,000,000 test items. S2_Embed uses the first 100,000 of the 1,000,000 test samples.

| | mean z | std z | point cos. | ensemble cos. | CRPS |
| --- | --- | --- | --- | --- | --- |
| S1_Embed | −0.0005 | 1.1382 | 0.1087 | 0.0911 | 0.0213 |
| S2_Embed | 0.0006 | 1.0250 | 0.0667 | 0.0613 | 0.0038 |

Training (paper settings are the defaults):

```bash
python scripts/s1_embed/train.py --data-root data/S1_Embed --output-dir runs/s1_embed      # 600 epochs, batch 256
python scripts/s2_embed/train.py --data-root data/S2_Embed --output-dir runs/s2_embed      # 72 epochs, batch 64, 8 heads
```

### Anomaly change detection on GSA

```bash
python scripts/gsa/train.py --embedding-root data/GSA/embeddings_patch5 --output-dir runs/gsa
python scripts/gsa/infer.py --checkpoint checkpoints/gsa.pt --embedding-root data/GSA/embeddings_patch5 \
    --label-root data/GSA/original --output-dir results/gsa
python scripts/gsa/evaluate.py --results-root results/gsa --label-root data/GSA/original
```

For every event, the last image is the observation and the three images before it are the context. The score map
holds `a² = min(‖(e−μ)/σ‖², min_j ‖(e−ê_j)/σ‖²)` with `J = S = 10`. It is computed on the 50 m embedding grid and
bilinearly upsampled to the 10 m labels. Evaluation follows HyperFree / ESIA:

- false positives get weight 0.5 in the precision;
- one threshold is tuned per category;
- event F1 is averaged within each category, then over categories.

Expected averages: recall 65.48, precision 48.84, F1 48.70.

### E²Forecast as an INR

```bash
python scripts/inr/train.py --shards 'data/INR/satclip_s2_100k/satclip_embeddings_location_{000493..000542}.tar' --output-dir runs/inr
python scripts/inr/downstream.py --checkpoint checkpoints/inr.pt --data-dir data/downstream --output results/inr_downstream.json
python scripts/inr/plot_global_field.py --checkpoint checkpoints/inr.pt --output figures/global_field.png

python scripts/inr/train.py --query-date --shards 'data/satclip/satclip_embeddings_location_{000000..000492}.tar' \
    --output-dir runs/inr_date
python scripts/inr/plot_global_monthly.py --checkpoint checkpoints/inr_date.pt --output figures/global_monthly.png
```

The downstream probe regresses each target from the normalised Gaussian mean at the sample's location:

- splits are region-disjoint: 30/10/60 % latitude (or longitude) bands with 0.25° guard gaps;
- the probe is an MLP with one hidden layer of width 32;
- the probe-selection rule is the same as for every baseline in the INR table (see `scripts/inr/downstream.py`).

Expected R²:

| population | elevation | temperature | house value | income | average |
| --- | --- | --- | --- | --- | --- |
| 0.6064 | 0.1687 | 0.7628 | 0.2183 | 0.5282 | 0.4569 |

## Using the model

```python
import torch
from e2forecast import load_model, relative_dates

model, checkpoint = load_model("checkpoints/s1_embed.pt", torch.device("cuda"))
obs_dates, query_date = relative_dates(context_doys, query_doy)          # (B, C) and (B,) day of year
conditions = dict(obs=context, obs_dates=obs_dates, query_date=query_date, lonlat=lonlat)  # context: (B, C, D), L2-normalised
mu, sigma = model.gaussian(**conditions)                                 # probability branch
point = model.point_forecast(**conditions)                               # mu / ||mu||
ensemble = [model.sample(steps=10, **conditions) for _ in range(10)]     # J = 10 ensemble forecasts
```

## Citation

```bibtex
@article{li2026e2forecast,
  title  = {Forecasting How Earth's Surface Changes in Embedding Space},
  author = {Li, Jingtao and Jaffer, Sadiq and Madhavapeddy, Anil and others},
  year   = {2026}
}
```
