"""Band-limited fractional delay: Kaiser-windowed sinc kernels and reads.

Interpolating a sampled signal between samples is the same operation as
delaying it by a non-integer amount.  Linear and low-order Lagrange
interpolation attenuate high frequencies by an amount that depends on the
fractional part (at 6 kHz / 16 kHz, linear loses up to 8.3 dB and third-order
Lagrange swings by 4.4 dB), so a path whose delay drifts — a moving source —
is amplitude-modulated at the rate its fractional part cycles.  A windowed
sinc keeps the response flat almost to Nyquist for every fraction (Laakso et
al., "Splitting the unit delay", IEEE SPM 1996).

With the defaults (32 taps, Kaiser beta 6) the magnitude stays within 0.01 dB
up to 0.875 x Nyquist (7 kHz at 16 kHz) for every fraction.  The kernel is
symmetric, so a delayed impulse rings for ``half_width - 1`` samples before
its arrival; that is the band-limited representation of the arrival, not a
causality violation of the modelled sound.
"""

from __future__ import annotations

from functools import lru_cache
import math

import numpy as np

WINDOWED_SINC_HALF_WIDTH = 16
WINDOWED_SINC_BETA = 6.0
_TABLE_PHASES = 1024


def windowed_sinc_taps(
    fraction: float | np.ndarray,
    *,
    half_width: int = WINDOWED_SINC_HALF_WIDTH,
    beta: float = WINDOWED_SINC_BETA,
) -> tuple[np.ndarray, np.ndarray]:
    """Taps that evaluate ``x(i + fraction)`` from ``x[i + offsets]``.

    Returns ``(offsets, taps)``; ``taps`` has a trailing axis of
    ``2 * half_width`` matching ``offsets = -half_width + 1 .. half_width``.
    """
    _check(half_width, beta)
    offsets = np.arange(-half_width + 1, half_width + 1)
    t = offsets - np.asarray(fraction, dtype=np.float64)[..., None]
    window = np.i0(beta * np.sqrt(np.clip(1.0 - (t / half_width) ** 2, 0.0, 1.0)))
    return offsets, np.sinc(t) * window / np.i0(beta)


def windowed_sinc_kernel(
    delay_samples: float,
    *,
    half_width: int = WINDOWED_SINC_HALF_WIDTH,
    beta: float = WINDOWED_SINC_BETA,
) -> tuple[int, np.ndarray]:
    """An impulse delayed by ``delay_samples``, as ``(start, kernel)``.

    ``kernel[m]`` is the band-limited impulse at sample ``start + m``.
    ``start`` is negative when the delay is shorter than ``half_width``;
    callers drop the samples before time zero.
    """
    delay = float(delay_samples)
    if not math.isfinite(delay) or delay < 0.0:
        raise ValueError("delay_samples must be finite and non-negative")
    whole = math.floor(delay)
    offsets, taps = windowed_sinc_taps(
        delay - whole, half_width=half_width, beta=beta
    )
    # kernel[m] = h(start + m - delay); with start = whole + offsets[0] this
    # is h(offsets[m] - fraction), the read taps for that fraction.
    return whole + int(offsets[0]), taps


def fractional_read(
    signals: np.ndarray,
    positions: np.ndarray,
    *,
    half_width: int = WINDOWED_SINC_HALF_WIDTH,
    beta: float = WINDOWED_SINC_BETA,
    block_size: int = 4096,
) -> np.ndarray:
    """Read ``signals`` at fractional sample ``positions``; outside reads zero.

    ``signals`` is ``(N,)`` or ``(channels, N)``; every channel is read at
    the same positions.  Taps come from a phase table with linear
    interpolation between adjacent phases (error below -120 dB).
    """
    values = np.asarray(signals, dtype=np.float64)
    single = values.ndim == 1
    values = np.atleast_2d(values)
    where = np.asarray(positions, dtype=np.float64)
    if values.ndim != 2 or where.ndim != 1:
        raise ValueError("signals must be (N,) or (channels, N); positions 1-D")
    if not np.all(np.isfinite(where)):
        raise ValueError("read positions must be finite")
    table = _phase_table(int(half_width), float(beta))
    pad = 2 * half_width
    padded = np.pad(values, ((0, 0), (pad, pad)))
    offsets = np.arange(-half_width + 1, half_width + 1)
    out = np.zeros((values.shape[0], where.size))
    for begin in range(0, where.size, block_size):
        block = where[begin : begin + block_size]
        whole_block = np.floor(block)
        # Reads whose taps all miss the signal stay zero.
        inside = (whole_block >= -half_width) & (
            whole_block < values.shape[1] + half_width
        )
        if not np.any(inside):
            continue
        local = block[inside]
        whole = whole_block[inside]
        phase = (local - whole) * _TABLE_PHASES
        index = np.minimum(phase.astype(np.int64), _TABLE_PHASES - 1)
        blend = (phase - index)[:, None]
        taps = table[index] * (1.0 - blend) + table[index + 1] * blend
        gather = whole.astype(np.int64)[:, None] + offsets[None, :] + pad
        out[:, np.flatnonzero(inside) + begin] = np.einsum(
            "cij,ij->ci", padded[:, gather], taps
        )
    return out[0] if single else out


@lru_cache(maxsize=8)
def _phase_table(half_width: int, beta: float) -> np.ndarray:
    fractions = np.arange(_TABLE_PHASES + 1) / _TABLE_PHASES
    return windowed_sinc_taps(fractions, half_width=half_width, beta=beta)[1]


def _check(half_width: int, beta: float) -> None:
    if type(half_width) is not int or not 2 <= half_width <= 64:
        raise ValueError("half_width must be an integer in [2, 64]")
    if not math.isfinite(beta) or beta < 0.0:
        raise ValueError("beta must be finite and non-negative")


__all__ = [
    "WINDOWED_SINC_BETA",
    "WINDOWED_SINC_HALF_WIDTH",
    "fractional_read",
    "windowed_sinc_kernel",
    "windowed_sinc_taps",
]
