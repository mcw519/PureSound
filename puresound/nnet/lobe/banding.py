"""Non-uniform frequency banding for the recurrent bottleneck.

A 512-point STFT at 16 kHz gives 256 bins of equal width, and half of them sit
above 4 kHz where voiced speech has almost no harmonic structure left. Striding
that grid uniformly -- which is what a frequency-strided conv does -- throws away
resolution at 200 Hz, where the harmonics are, at exactly the same rate as at
7 kHz, where there is nothing to lose.

Banding on a perceptual scale spends the same number of units differently: dense
low, sparse high. Thirty-two ERB bands cost the recurrent path exactly what
thirty-two uniform units cost, and keep more of the part that carries pitch.
This is the grouping RNNoise, PercepNet and DeepFilterNet all use, for this reason.

What this module is NOT: a replacement for the STFT. Analysis and synthesis stay
on the full 256-bin complex grid and the mask is still applied there -- only the
recurrent bottleneck sees bands. Band gains alone cannot reconstruct structure
*inside* a band, which is why systems that band all the way through need a comb
filter or a per-bin refinement stage to put it back.

Both directions are a fixed linear map over frequency with no state in time, so a
per-frame streaming graph carries them unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def erb_rate(hz: torch.Tensor | float) -> torch.Tensor | float:
    """Glasberg & Moore ERB-number scale."""
    if isinstance(hz, torch.Tensor):
        return 21.4 * torch.log10(1.0 + 0.00437 * hz)
    return 21.4 * math.log10(1.0 + 0.00437 * hz)


def erb_rate_to_hz(rate: torch.Tensor) -> torch.Tensor:
    return (torch.pow(10.0, rate / 21.4) - 1.0) / 0.00437


def mel_rate(hz: float) -> float:
    return 2595.0 * math.log10(1.0 + hz / 700.0)


def mel_rate_to_hz(rate: torch.Tensor) -> torch.Tensor:
    return 700.0 * (torch.pow(10.0, rate / 2595.0) - 1.0)


SCALES = {
    "erb": (erb_rate, erb_rate_to_hz),
    "mel": (mel_rate, mel_rate_to_hz),
}


def band_edges_hz(
    n_bands: int, f_min: float, f_max: float, scale: str = "erb"
) -> torch.Tensor:
    """``n_bands + 1`` edges, equally spaced on ``scale`` and returned in Hz."""
    if scale not in SCALES:
        raise ValueError(f"scale must be one of {sorted(SCALES)}, got {scale!r}")
    if n_bands < 1:
        raise ValueError("n_bands must be >= 1")
    if not 0 <= f_min < f_max:
        raise ValueError(f"need 0 <= f_min < f_max, got {f_min} and {f_max}")

    to_rate, from_rate = SCALES[scale]
    rates = torch.linspace(float(to_rate(f_min)), float(to_rate(f_max)), n_bands + 1)
    return from_rate(rates)


def triangular_band_matrix(
    n_bands: int,
    n_units: int,
    *,
    f_min: float,
    f_max: float,
    scale: str = "erb",
) -> torch.Tensor:
    """``[n_bands, n_units]`` overlapping triangular weights, rows summing to 1.

    Rows sum to one so banding is an average rather than a sum: a wide band and a
    narrow one come out on the same scale, and the values reaching the recurrent
    path do not grow with band width.

    Triangular rather than rectangular so neighbouring bands overlap. A hard
    partition puts a discontinuity at every edge, and the edges are where a
    harmonic that drifts with pitch crosses over.
    """
    edges = band_edges_hz(n_bands, f_min, f_max, scale)
    # Unit u is centred at the middle of its slice of [f_min, f_max].
    centres = f_min + (torch.arange(n_units, dtype=torch.float32) + 0.5) * (
        (f_max - f_min) / n_units
    )

    matrix = torch.zeros(n_bands, n_units)
    for band in range(n_bands):
        low, high = float(edges[band]), float(edges[band + 1])
        centre = 0.5 * (low + high)
        # Reach one band outward on each side so neighbours overlap.
        left = float(edges[band - 1]) if band > 0 else low - (centre - low)
        right = float(edges[band + 2]) if band + 2 <= n_bands else high + (high - centre)
        rising = (centres - left) / max(centre - left, 1e-6)
        falling = (right - centres) / max(right - centre, 1e-6)
        weights = torch.clamp(torch.minimum(rising, falling), min=0.0)
        total = float(weights.sum())
        if total <= 0.0:
            # A band narrower than one unit: fall back to its nearest unit, so no
            # band is silently empty and no row of the matrix is all zeros.
            weights = torch.zeros(n_units)
            weights[int(torch.argmin((centres - centre).abs()))] = 1.0
            total = 1.0
        matrix[band] = weights / total

    return matrix


class BandBottleneck(nn.Module):
    """Pool the bottleneck onto perceptual bands, and expand it back.

    Sits around the recurrent blocks: the U-net's strided convs, the mask and the
    heads all keep seeing ``n_units``, so nothing downstream changes shape.

    ``learnable`` makes both maps trainable, initialised from the fixed
    perceptual ones. Off by default: a fixed map keeps the band layout an
    explicit design choice, and letting it drift makes any effect of the
    banding impossible to attribute.
    """

    def __init__(
        self,
        n_units: int,
        n_bands: int,
        *,
        sample_rate: int = 16000,
        f_min: float = 50.0,
        f_max: float | None = None,
        scale: str = "erb",
        learnable: bool = False,
    ):
        super().__init__()
        if n_bands > n_units:
            raise ValueError(
                f"banding to {n_bands} from {n_units} units would not reduce anything"
            )
        f_max = f_max if f_max is not None else sample_rate / 2
        self.n_units, self.n_bands, self.scale = n_units, n_bands, scale

        pool = triangular_band_matrix(
            n_bands, n_units, f_min=f_min, f_max=f_max, scale=scale
        )
        # Expansion is the transpose, normalised so every unit receives weight 1
        # in total -- otherwise units at a band edge come back quieter than units
        # at a centre, which reads as a comb the model never asked for.
        expand = pool.t().clone()
        expand = expand / expand.sum(dim=1, keepdim=True).clamp(min=1e-6)

        if learnable:
            self.pool = nn.Parameter(pool)
            self.expand = nn.Parameter(expand)
        else:
            self.register_buffer("pool", pool)
            self.register_buffer("expand", expand)

    def to_bands(self, x: torch.Tensor) -> torch.Tensor:
        """``[N, CH, n_units, T]`` -> ``[N, CH, n_bands, T]``."""
        return torch.einsum("bu,ncut->ncbt", self.pool, x)

    def to_units(self, x: torch.Tensor) -> torch.Tensor:
        """``[N, CH, n_bands, T]`` -> ``[N, CH, n_units, T]``."""
        return torch.einsum("ub,ncbt->ncut", self.expand, x)

    def extra_repr(self) -> str:
        return (
            f"n_units={self.n_units}, n_bands={self.n_bands}, scale={self.scale}, "
            f"learnable={isinstance(self.pool, nn.Parameter)}"
        )
