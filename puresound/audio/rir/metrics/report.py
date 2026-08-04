"""``analyze_rir``: the one report facade every bank tool consumes.

Moved out of ``puresound.audio.rir.metrics`` in R4 of
``RIR_EXP_LOG.md``.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

from puresound.audio.rir.metrics.core import (
    _as_mono_numpy,
    _decay_dict,
    _finite_or_none,
    direct_sample,
)
from puresound.audio.rir.metrics.density import analyze_echo_density
from puresound.audio.rir.metrics.spectral import (
    DEFAULT_OCTAVE_CENTERS_HZ,
    octave_band_rir,
    spectral_tilt_db_per_octave,
    valid_octave_centers,
)
from puresound.audio.rir.metrics.temporal import (
    _fit_decay_curve,
    clarity_db,
    compute_drr_db,
    estimate_noise_floor_lundeby,
    noise_compensated_schroeder_decay_db,
    schroeder_decay_db,
)


def analyze_rir(
    rir: Any,
    sample_rate: int,
    direct_window_ms: float = 2.5,
    direct_index: Optional[int] = None,
    octave_centers_hz: Optional[Iterable[float]] = None,
    noise_compensation: bool = True,
    echo_density: bool = False,
    echo_density_window_ms: float = 20.0,
    echo_density_hop_ms: float = 1.0,
    echo_density_threshold: float = 0.9,
    echo_density_minimum_sustain_ms: float = 10.0,
) -> dict[str, Any]:
    """Compute a JSON-safe set of broadband and optional octave-band metrics."""
    x = _as_mono_numpy(rir)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    peak = direct_sample(x) if direct_index is None else int(direct_index)
    peak = int(np.clip(peak, 0, max(0, x.size - 1)))
    if noise_compensation:
        decay_curve, noise_floor = noise_compensated_schroeder_decay_db(
            x,
            sample_rate=sample_rate,
            direct_index=peak,
        )
    else:
        decay_curve = schroeder_decay_db(x, direct_index=peak)
        noise_floor = estimate_noise_floor_lundeby(
            x,
            sample_rate=sample_rate,
            direct_index=peak,
        )
    edt = _fit_decay_curve(decay_curve, sample_rate, 0.0, -10.0)
    t20 = _fit_decay_curve(decay_curve, sample_rate, -5.0, -25.0)
    t30 = _fit_decay_curve(decay_curve, sample_rate, -5.0, -35.0)
    result: dict[str, Any] = {
        "sample_rate": int(sample_rate),
        "num_samples": int(x.size),
        "direct_sample": int(peak),
        "direct_delay_ms": float(peak / float(sample_rate) * 1000.0),
        "peak_abs": float(np.max(np.abs(x))) if x.size else 0.0,
        "energy": float(np.square(x, dtype=np.float64).sum()),
        "drr_db": _finite_or_none(
            compute_drr_db(
                x,
                sample_rate,
                direct_window_ms=direct_window_ms,
                direct_index=peak,
            )
        ),
        "c50_db": _finite_or_none(clarity_db(x, sample_rate, 50.0, peak)),
        "c80_db": _finite_or_none(clarity_db(x, sample_rate, 80.0, peak)),
        "spectral_tilt_db_per_octave": _finite_or_none(
            spectral_tilt_db_per_octave(x, sample_rate)
        ),
        "noise_floor": noise_floor.to_dict(),
        "edt": _decay_dict(edt),
        "t20": _decay_dict(t20),
        "t30": _decay_dict(t30),
        "edt_s": edt.rt60_s if edt is not None else None,
        "t20_s": t20.rt60_s if t20 is not None else None,
        "t30_s": t30.rt60_s if t30 is not None else None,
    }

    if octave_centers_hz is not None:
        octave_metrics: dict[str, Any] = {}
        for center in valid_octave_centers(sample_rate, octave_centers_hz):
            band = octave_band_rir(x, sample_rate, center)
            band_peak = direct_sample(band)
            if noise_compensation:
                (
                    band_curve,
                    band_noise_floor,
                ) = noise_compensated_schroeder_decay_db(
                    band,
                    sample_rate=sample_rate,
                    direct_index=band_peak,
                )
            else:
                band_curve = schroeder_decay_db(band, direct_index=band_peak)
                band_noise_floor = estimate_noise_floor_lundeby(
                    band,
                    sample_rate=sample_rate,
                    direct_index=band_peak,
                )
            band_edt = _fit_decay_curve(band_curve, sample_rate, 0.0, -10.0)
            band_t20 = _fit_decay_curve(band_curve, sample_rate, -5.0, -25.0)
            band_t30 = _fit_decay_curve(band_curve, sample_rate, -5.0, -35.0)
            octave_metrics[f"{center:g}"] = {
                "center_hz": center,
                "drr_db": _finite_or_none(
                    compute_drr_db(
                        band,
                        sample_rate,
                        direct_window_ms=direct_window_ms,
                        direct_index=band_peak,
                    )
                ),
                "c50_db": _finite_or_none(
                    clarity_db(band, sample_rate, 50.0, band_peak)
                ),
                "edt_s": band_edt.rt60_s if band_edt is not None else None,
                "t20_s": band_t20.rt60_s if band_t20 is not None else None,
                "t30_s": band_t30.rt60_s if band_t30 is not None else None,
                "decay_fit_r2": {
                    "edt": band_edt.r_squared if band_edt is not None else None,
                    "t20": band_t20.r_squared if band_t20 is not None else None,
                    "t30": band_t30.r_squared if band_t30 is not None else None,
                },
                "noise_floor": band_noise_floor.to_dict(),
            }
        result["octave_bands"] = octave_metrics
    if echo_density:
        result["echo_density"] = analyze_echo_density(
            x,
            sample_rate,
            direct_index=peak,
            window_ms=echo_density_window_ms,
            hop_ms=echo_density_hop_ms,
            threshold=echo_density_threshold,
            minimum_sustain_ms=echo_density_minimum_sustain_ms,
        )
    return result
