"""E2Forecast: joint probability estimation and embedding generation in one masked Transformer pass.

Token sequence (paper, Sec. 3):

    [location | C observations | query date | h_mu | h_sigma | flow time | noisy target z_t | h_query]

* The first group of tokens (location ... h_sigma) is the *probability-estimation* branch: the outputs at
  h_mu / h_sigma give the diagonal Gaussian surrogate N(mu, diag(sigma^2)), trained with the Gaussian NLL.
* The last three tokens are the *generation* branch: the output at h_query is an x-prediction z1_hat of the
  (scaled) target embedding, trained with a rectified-flow loss in velocity space.
* Probability-estimation tokens may not attend to generation tokens, so (mu, sigma) never depends on z_t or t.

The same network is used for every experiment in the paper; only the token set changes:

* S1_Embed / S2_Embed / GSA forecasting: C context observations and a learned query-date token;
* INR (location -> embedding): no context and no query date;
* INR with query date: no context, a fixed (unprojected) day-of-year query-date token.

All embeddings are L2-normalised.  The flow operates on ``sqrt(D) * e`` (unit variance per dimension).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .location import GeoCLIPRFF

QUERY_DATE_MODES = ("learned", "fixed", "none")
DATE_OFFSETS = ("order", "year")
DEFAULT_DATE_OFFSET = "order"  # used whenever a checkpoint or caller does not specify one


# ------------------------------------------------------------------------------------------------ encodings
def date_encoding(dates: Tensor, dim: int) -> Tensor:
    """psi(tau): sine-cosine encoding of (relative offset r, day of year); no absolute year.

    ``dates`` is (N, 2): column 0 = offset r relative to the query date (see :func:`relative_dates`; 0 for the query),
    column 1 = day of year (1..366).
    """
    if dim % 4:
        raise ValueError("date encoding dimension must be divisible by four")
    offset, day_of_year = dates[:, 0], dates[:, 1]
    frequency = torch.exp(torch.linspace(0, math.log(10000), dim // 4, device=dates.device, dtype=dates.dtype))
    offset_angle = offset[:, None] / frequency
    doy_angle = 2 * math.pi * (day_of_year[:, None] - 1.0) / 365.25 * frequency.sqrt()
    return torch.cat((offset_angle.sin(), offset_angle.cos(), doy_angle.sin(), doy_angle.cos()), -1)


def day_of_year_encoding(day_of_year: Tensor, dim: int) -> Tensor:
    """Fixed sine-cosine encoding of the day of year only, using all ``dim`` dimensions (INR query date)."""
    if dim % 2:
        raise ValueError("day-of-year encoding dimension must be even")
    frequency = torch.exp(torch.linspace(0, math.log(10000), dim // 2, device=day_of_year.device,
                                         dtype=torch.float32)).sqrt()
    angle = 2 * math.pi * (day_of_year.float()[:, None] - 1.0) / 365.25 * frequency
    return torch.cat((angle.sin(), angle.cos()), -1)


def flow_time_encoding(flow_time: Tensor, dim: int) -> Tensor:
    """zeta(t): sinusoidal encoding of the flow time t in [0, 1]."""
    frequency = torch.exp(torch.linspace(0, math.log(1000.0), dim // 2, device=flow_time.device,
                                         dtype=flow_time.dtype))
    angle = 2 * math.pi * flow_time[:, None] * frequency
    return torch.cat((angle.sin(), angle.cos()), -1)


def relative_dates(context_doys: Tensor, query_doy: Tensor, date_offset: str = DEFAULT_DATE_OFFSET,
                   context_years: Tensor | None = None, query_year: Tensor | None = None) -> tuple[Tensor, Tensor]:
    """(offset r, day of year) for C chronologically sorted observations and the query date (r = 0).

    ``date_offset="order"``: r = -C..-1, the position of each observation (default; histories within one year);
    ``date_offset="year"`` : r = year(query) - year(observation), which needs ``context_years`` and ``query_year``
    (histories spanning several years).
    """
    batch, count = context_doys.shape
    if date_offset == "order":
        offset = torch.arange(-count, 0, device=context_doys.device, dtype=torch.float32).expand(batch, count)
    elif date_offset == "year":
        if context_years is None or query_year is None:
            raise ValueError("date_offset='year' needs context_years and query_year")
        offset = (query_year[:, None] - context_years).float()
    else:
        raise ValueError(f"date_offset must be one of {DATE_OFFSETS}")
    obs_dates = torch.stack((offset, context_doys.float()), dim=-1)
    query_date = torch.stack((torch.zeros(batch, device=query_doy.device), query_doy.float()), dim=-1)
    return obs_dates, query_date


def conditions_from_doys(batch: dict[str, Tensor], date_offset: str = DEFAULT_DATE_OFFSET) -> dict[str, Tensor]:
    """Batch with ``obs``, ``obs_doys``, ``query_doy``, ``lonlat`` (and ``obs_years``, ``query_year`` for
    ``date_offset="year"``) -> keyword conditions of :class:`E2Forecast`."""
    obs_dates, query_date = relative_dates(batch["obs_doys"], batch["query_doy"], date_offset,
                                           batch.get("obs_years"), batch.get("query_year"))
    return {"obs": batch["obs"], "obs_dates": obs_dates, "query_date": query_date, "lonlat": batch["lonlat"]}


def diagonal_gaussian_nll(target: Tensor, mean: Tensor, std: Tensor) -> Tensor:
    """L_nll of N(mu, diag(sigma^2)), averaged over batch and dimensions."""
    if target.shape != mean.shape or std.shape != target.shape:
        raise ValueError("target, mean, and std must all have shape B x embedding_dim")
    return 0.5 * ((target - mean) / std).square().mean() + std.log().mean() + 0.5 * math.log(2.0 * math.pi)


# ------------------------------------------------------------------------------------------------ network
class E2ForecastNetwork(nn.Module):
    """f_theta: one masked Transformer pass returning ``(z1_hat, mu, sigma)``.

    The Transformer width equals the embedding dimension ``embed_dim``: the fixed GeoCLIP random-Fourier-feature
    location token is used without projection.
    """

    def __init__(self, embed_dim: int, context_count: int = 0, query_date: str = "learned", depth: int = 8,
                 heads: int = 4, ff_mult: int = 4, dropout: float = 0.05, min_std: float = 1e-4,
                 rff_sigma: float = 1.0, rff_seed: int = 42) -> None:
        super().__init__()
        if query_date not in QUERY_DATE_MODES:
            raise ValueError(f"query_date must be one of {QUERY_DATE_MODES}")
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads")
        dim = embed_dim
        self.embed_dim, self.context_count, self.query_date, self.min_std = embed_dim, context_count, query_date, min_std
        self.location_encoder = GeoCLIPRFF(embedding_dim=dim, sigma=rff_sigma, seed=rff_seed)  # fixed, no parameters
        if context_count:
            self.observation = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))  # W_s
            self.obs_date = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))  # W_tau (observation dates)
        if query_date == "learned":
            self.target_date = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))  # W_tau (query date)
        self.noisy = nn.Linear(dim, dim)  # W_z
        self.flow_time = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.learned_mean = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # h_mu
        self.learned_std = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # h_sigma
        self.flow_readout = nn.Parameter(torch.randn(1, 1, dim) * 0.02)  # h_query
        self.mean_index = 1 + context_count + (query_date != "none")
        self.std_index = self.mean_index + 1
        self.flow_time_index = self.std_index + 1
        self.query_index = self.flow_time_index + 2
        self.token_count = self.query_index + 1
        self.types_and_positions = nn.Parameter(torch.randn(1, self.token_count, dim) * 0.02)
        layer = nn.TransformerEncoderLayer(dim, heads, dim * ff_mult, dropout, "gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, depth, nn.LayerNorm(dim), enable_nested_tensor=False)
        self.mean_head = nn.Linear(dim, dim)  # W_mu
        self.std_head = nn.Linear(dim, dim)  # W_sigma
        self.x_head = nn.Linear(dim, dim)  # W_query
        mask = torch.zeros(self.token_count, self.token_count, dtype=torch.bool)  # True = may not attend
        mask[:self.std_index + 1, self.flow_time_index:] = True  # probability estimation ignores generation tokens
        self.register_buffer("attention_mask", mask, persistent=False)

    def location_token(self, lonlat: Tensor) -> Tensor:
        with torch.no_grad(), torch.autocast(device_type=lonlat.device.type, enabled=False):
            return self.location_encoder(lonlat.float()).float()

    def forward(self, lonlat: Tensor, noisy: Tensor, flow_time: Tensor, obs: Tensor | None = None,
                obs_dates: Tensor | None = None, query_date: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor]:
        """lonlat (B, 2) degrees; noisy z_t (B, D); flow_time t (B,); obs (B, C, D); obs_dates (B, C, 2);
        query_date (B, 2) = (0, day of year).  Returns ``(z1_hat, mu, sigma)``."""
        batch, dim = len(noisy), self.embed_dim
        tokens = [self.location_token(lonlat).to(noisy.dtype).unsqueeze(1)]
        if self.context_count:
            if obs is None or obs.shape[1:] != (self.context_count, dim):
                raise ValueError(f"obs must have shape B x {self.context_count} x {dim}")
            dates = date_encoding(obs_dates.reshape(-1, 2), dim).reshape(batch, self.context_count, -1)
            tokens.append(self.observation(obs) + self.obs_date(dates.to(obs.dtype)))
        if self.query_date == "learned":
            tokens.append(self.target_date(date_encoding(query_date, dim).to(noisy.dtype)).unsqueeze(1))
        elif self.query_date == "fixed":
            tokens.append(day_of_year_encoding(query_date[:, 1], dim).to(noisy.dtype).unsqueeze(1))
        time_token = self.flow_time(flow_time_encoding(flow_time, dim).to(noisy.dtype)).unsqueeze(1)
        tokens += [self.learned_mean.expand(batch, -1, -1), self.learned_std.expand(batch, -1, -1),
                   time_token, self.noisy(noisy).unsqueeze(1), self.flow_readout.expand(batch, -1, -1)]
        encoded = self.transformer(torch.cat(tokens, dim=1) + self.types_and_positions, mask=self.attention_mask)
        mean = self.mean_head(encoded[:, self.mean_index])
        std = F.softplus(self.std_head(encoded[:, self.std_index])) + self.min_std
        return self.x_head(encoded[:, self.query_index]), mean, std


# ------------------------------------------------------------------------------------------------ model
class E2Forecast(nn.Module):
    """Joint objective L = w_nll * L_nll + L_flow and the two inference modes.

    ``conditions`` are the keyword arguments of :meth:`E2ForecastNetwork.forward` other than the flow state:
    ``lonlat`` and, depending on the token set, ``obs``, ``obs_dates`` and ``query_date``.
    """

    def __init__(self, network: E2ForecastNetwork, nll_weight: float = 1.0, t_eps: float = 0.05) -> None:
        super().__init__()
        self.network, self.nll_weight, self.t_eps = network, nll_weight, t_eps

    @property
    def embed_dim(self) -> int:
        return self.network.embed_dim

    def velocity(self, z1_hat: Tensor, z_t: Tensor, flow_time: Tensor) -> Tensor:
        """v = (z1_hat - z_t) / (1 - t), with 1 - t clipped at t_eps."""
        return (z1_hat - z_t) / (1.0 - flow_time[:, None]).clamp_min(self.t_eps)

    def loss(self, target: Tensor, **conditions: Tensor) -> dict[str, Tensor]:
        """Returns ``{"loss", "nll", "flow"}``; ``loss`` is the training objective."""
        z1 = target * math.sqrt(self.embed_dim)  # flow operates on the embedding scaled to unit variance per dimension
        noise = torch.randn_like(z1)
        flow_time = torch.sigmoid(torch.randn(len(z1), device=target.device) * 1.2 - 1.2)  # logit-normal t
        z_t = flow_time[:, None] * z1 + (1.0 - flow_time[:, None]) * noise
        z1_hat, mean, std = self.network(noisy=z_t, flow_time=flow_time, **conditions)
        nll = diagonal_gaussian_nll(target, mean, std)
        flow = F.mse_loss(self.velocity(z1_hat, z_t, flow_time), self.velocity(z1, z_t, flow_time))
        return {"loss": self.nll_weight * nll + flow, "nll": nll.detach(), "flow": flow.detach()}

    @torch.no_grad()
    def gaussian(self, **conditions: Tensor) -> tuple[Tensor, Tensor]:
        """(mu, sigma) in one pass with z_t = 0 and t = 0."""
        reference = conditions["lonlat"]
        zeros = torch.zeros(len(reference), self.embed_dim, device=reference.device)
        if conditions.get("obs") is not None:
            zeros = zeros.to(conditions["obs"].dtype)
        _, mean, std = self.network(noisy=zeros, flow_time=zeros.new_zeros(len(reference)), **conditions)
        return mean.float(), std.float()

    @torch.no_grad()
    def point_forecast(self, **conditions: Tensor) -> Tensor:
        """Normalised Gaussian mean mu / ||mu||."""
        return F.normalize(self.gaussian(**conditions)[0], dim=-1)

    @torch.no_grad()
    def sample(self, steps: int = 10, generator: torch.Generator | None = None, **conditions: Tensor) -> Tensor:
        """One ensemble forecast: ``steps`` Euler steps from Gaussian noise, then normalised."""
        if steps < 1:
            raise ValueError("steps must be positive")
        reference = conditions["lonlat"]
        batch, device = len(reference), reference.device
        z = torch.randn((batch, self.embed_dim), device=device, generator=generator)
        for step in range(steps):
            flow_time = torch.full((batch,), step / steps, device=device)
            z1_hat = self.network(noisy=z, flow_time=flow_time, **conditions)[0]
            z += self.velocity(z1_hat, z, flow_time) / steps
        return F.normalize(z / math.sqrt(self.embed_dim), dim=-1)


def build_model(model_config: dict, nll_weight: float = 1.0, t_eps: float = 0.05) -> E2Forecast:
    return E2Forecast(E2ForecastNetwork(**model_config), nll_weight, t_eps)
