"""Echo density, mixing time, and the multiband late field.

Abel-Huang normalized echo density plus the octave-band late-field analysis
that truncates at the Lundeby intersection so measurement noise is not read as
a physical diffuse tail.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Optional

import numpy as np

from puresound.audio.rir.metrics.core import (
    _as_mono_numpy,
    _decay_dict,
    direct_sample,
)
from puresound.audio.rir.metrics.spectral import (
    DEFAULT_OCTAVE_CENTERS_HZ,
    octave_band_rir,
    valid_octave_centers,
)
from puresound.audio.rir.metrics.temporal import (
    _fit_decay_curve,
    noise_compensated_schroeder_decay_db,
)


GAUSSIAN_OUTSIDE_ONE_STD_FRACTION = math.erfc(1.0 / math.sqrt(2.0))

ABEL_ECHO_DENSITY_POLICY = "puresound.abel_huang_echo_density.v1"

MULTIBAND_LATE_FIELD_POLICY = "puresound.multiband_late_field.v1"


def abel_normalized_echo_density_profile(
    rir: Any,
    sample_rate: int,
    *,
    direct_index: Optional[int] = None,
    window_ms: float = 20.0,
    hop_ms: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the Abel-Huang normalized echo-density profile.

    Times are seconds relative to the detected or provided direct sample. Only
    full analysis windows after that sample are evaluated so pre-arrival
    silence cannot bias the first estimates. Values are intentionally not
    clipped: a finite statistical estimate can be greater than one.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if window_ms <= 0.0:
        raise ValueError("window_ms must be positive")
    if hop_ms <= 0.0:
        raise ValueError("hop_ms must be positive")

    x = _as_mono_numpy(rir)
    if x.size == 0:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty.copy()
    if direct_index is None:
        direct_index = direct_sample(x)
    direct_index = int(np.clip(direct_index, 0, x.size - 1))

    half_window = max(
        1,
        int(round(0.5 * window_ms * 1e-3 * sample_rate)),
    )
    window_size = 2 * half_window + 1
    hop_size = max(1, int(round(hop_ms * 1e-3 * sample_rate)))
    first_center = direct_index + half_window
    last_center = x.size - half_window
    if first_center >= last_center:
        empty = np.asarray([], dtype=np.float64)
        return empty, empty.copy()

    centers = np.arange(
        first_center,
        last_center,
        hop_size,
        dtype=np.int64,
    )
    density = np.empty(centers.size, dtype=np.float64)
    normalizer = window_size * GAUSSIAN_OUTSIDE_ONE_STD_FRACTION
    tiny = np.finfo(np.float64).tiny

    for output_index, center in enumerate(centers):
        frame = x[center - half_window : center + half_window + 1]
        sigma = float(np.std(frame))
        if sigma <= tiny:
            density[output_index] = 0.0
            continue
        exceedances = int(np.count_nonzero(np.abs(frame) > sigma))
        density[output_index] = exceedances / normalizer

    times_s = (centers.astype(np.float64) - direct_index) / sample_rate
    return times_s, density


def estimate_abel_mixing_time(
    times_s: Any,
    normalized_density: Any,
    *,
    threshold: float = 0.9,
    minimum_sustain_ms: float = 0.0,
) -> Optional[float]:
    """Estimate direct-relative mixing time from normalized echo density.

    With ``minimum_sustain_ms=0`` this implements the original first threshold
    crossing. A positive sustain time is a PureSound robustness extension for
    rejecting isolated crossings in a strongly fluctuating finite-window
    profile.
    """

    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError("threshold must be finite and positive")
    if not np.isfinite(minimum_sustain_ms) or minimum_sustain_ms < 0.0:
        raise ValueError("minimum_sustain_ms must be finite and non-negative")

    times = np.asarray(times_s, dtype=np.float64).reshape(-1)
    density = np.asarray(normalized_density, dtype=np.float64).reshape(-1)
    if times.size != density.size:
        raise ValueError("times_s and normalized_density must have equal length")
    if times.size == 0:
        return None
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(density)):
        raise ValueError("profile values must be finite")
    if np.any(np.diff(times) <= 0.0):
        raise ValueError("times_s must be strictly increasing")

    above = density >= threshold
    if minimum_sustain_ms == 0.0 or times.size == 1:
        crossings = np.flatnonzero(above)
        return None if crossings.size == 0 else float(times[crossings[0]])

    hop_s = float(np.median(np.diff(times)))
    required = max(1, int(math.ceil(minimum_sustain_ms * 1e-3 / hop_s)))
    run_start = 0
    run_length = 0
    for index, is_above in enumerate(above):
        if is_above:
            if run_length == 0:
                run_start = index
            run_length += 1
            if run_length >= required:
                return float(times[run_start])
        else:
            run_length = 0
    return None


def analyze_echo_density(
    rir: Any,
    sample_rate: int,
    *,
    direct_index: Optional[int] = None,
    window_ms: float = 20.0,
    hop_ms: float = 1.0,
    threshold: float = 0.9,
    minimum_sustain_ms: float = 10.0,
    probe_times_ms: Iterable[float] = (20.0, 50.0, 100.0, 200.0),
) -> dict[str, Any]:
    """Summarize monophonic echo density and direct-relative mixing time."""

    times_s, density = abel_normalized_echo_density_profile(
        rir,
        sample_rate,
        direct_index=direct_index,
        window_ms=window_ms,
        hop_ms=hop_ms,
    )
    mixing_time_s = estimate_abel_mixing_time(
        times_s,
        density,
        threshold=threshold,
        minimum_sustain_ms=minimum_sustain_ms,
    )

    probes: dict[str, Optional[float]] = {}
    for probe_time_ms in probe_times_ms:
        probe_time_ms = float(probe_time_ms)
        if not np.isfinite(probe_time_ms) or probe_time_ms < 0.0:
            raise ValueError("probe_times_ms must be finite and non-negative")
        key = f"{probe_time_ms:g}"
        if (
            times_s.size == 0
            or probe_time_ms * 1e-3 < times_s[0]
            or probe_time_ms * 1e-3 > times_s[-1]
        ):
            probes[key] = None
            continue
        nearest = int(np.argmin(np.abs(times_s - probe_time_ms * 1e-3)))
        probes[key] = float(density[nearest])

    if density.size == 0:
        late_median = None
        profile_max = None
    else:
        late_start = max(0, int(math.floor(0.75 * density.size)))
        late_median = float(np.median(density[late_start:]))
        profile_max = float(np.max(density))

    return {
        "policy": ABEL_ECHO_DENSITY_POLICY,
        "time_reference": "seconds_after_direct_sample",
        "window_ms": float(window_ms),
        "hop_ms": float(hop_ms),
        "threshold": float(threshold),
        "minimum_sustain_ms": float(minimum_sustain_ms),
        "mixing_time_s": mixing_time_s,
        "profile_sample_count": int(density.size),
        "profile_max_normalized_density": profile_max,
        "late_median_normalized_density": late_median,
        "normalized_density_at_ms": probes,
    }


def analyze_multiband_late_field(
    rir: Any,
    sample_rate: int,
    *,
    direct_index: Optional[int] = None,
    centers_hz: Iterable[float] = DEFAULT_OCTAVE_CENTERS_HZ,
    filter_order: int = 4,
    base_window_ms: float = 20.0,
    minimum_window_cycles: float = 4.0,
    hop_ms: float = 2.0,
    threshold: float = 0.9,
    minimum_sustain_ms: float = 10.0,
    probe_times_ms: Iterable[float] = (50.0, 100.0, 200.0, 400.0),
) -> dict[str, Any]:
    """Analyze noise-aware octave-band decay and echo density.

    Every band keeps the broadband direct sample as its physical time anchor.
    If Lundeby analysis finds a reliable stationary-noise intersection, echo
    density is evaluated only up to that exclusive sample so measurement noise
    is not mislabeled as a physical diffuse tail.

    The local echo-density window is at least ``minimum_window_cycles`` at the
    nominal lower octave edge. This makes low-band estimates explicit rather
    than applying a sub-cycle 20 ms window at every center frequency.
    """

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if filter_order < 1:
        raise ValueError("filter_order must be positive")
    if base_window_ms <= 0.0:
        raise ValueError("base_window_ms must be positive")
    if not np.isfinite(minimum_window_cycles) or minimum_window_cycles <= 0.0:
        raise ValueError("minimum_window_cycles must be finite and positive")

    x = _as_mono_numpy(rir)
    peak = direct_sample(x) if direct_index is None else int(direct_index)
    peak = int(np.clip(peak, 0, max(0, x.size - 1)))
    if x.size == 0:
        return {
            "policy": MULTIBAND_LATE_FIELD_POLICY,
            "filter": "causal_butterworth_nominal_octave",
            "filter_order": int(filter_order),
            "direct_sample": 0,
            "direct_policy": (
                "absolute_peak"
                if direct_index is None
                else "provided_broadband_anchor"
            ),
            "window_policy": "max_base_or_minimum_cycles_at_lower_octave_edge",
            "base_window_ms": float(base_window_ms),
            "minimum_window_cycles": float(minimum_window_cycles),
            "hop_ms": float(hop_ms),
            "threshold": float(threshold),
            "minimum_sustain_ms": float(minimum_sustain_ms),
            "bands": {},
        }
    bands: dict[str, Any] = {}
    for center_hz in valid_octave_centers(sample_rate, centers_hz):
        band = octave_band_rir(
            x,
            sample_rate,
            center_hz,
            filter_order=filter_order,
        )
        decay_curve, noise_floor = noise_compensated_schroeder_decay_db(
            band,
            sample_rate=sample_rate,
            direct_index=peak,
        )
        analysis_end = (
            int(noise_floor.truncation_sample)
            if noise_floor.correction_applied
            else int(band.size)
        )
        analysis_end = int(np.clip(analysis_end, peak + 1, band.size))
        lower_edge_hz = center_hz / math.sqrt(2.0)
        cycle_window_ms = 1000.0 * minimum_window_cycles / lower_edge_hz
        window_ms = max(float(base_window_ms), float(cycle_window_ms))
        echo_density = analyze_echo_density(
            band[:analysis_end],
            sample_rate,
            direct_index=peak,
            window_ms=window_ms,
            hop_ms=hop_ms,
            threshold=threshold,
            minimum_sustain_ms=minimum_sustain_ms,
            probe_times_ms=probe_times_ms,
        )
        edt = _fit_decay_curve(decay_curve, sample_rate, 0.0, -10.0)
        t20 = _fit_decay_curve(decay_curve, sample_rate, -5.0, -25.0)
        t30 = _fit_decay_curve(decay_curve, sample_rate, -5.0, -35.0)
        bands[f"{center_hz:g}"] = {
            "center_hz": float(center_hz),
            "lower_edge_hz": float(lower_edge_hz),
            "analysis_window_ms": float(window_ms),
            "analysis_end_sample": int(analysis_end),
            "analysis_end_s_after_direct": float(
                max(0, analysis_end - peak) / sample_rate
            ),
            "echo_density_truncated_at_noise_intersection": bool(
                noise_floor.correction_applied
            ),
            "echo_density": echo_density,
            "noise_floor": noise_floor.to_dict(),
            "edt": _decay_dict(edt),
            "t20": _decay_dict(t20),
            "t30": _decay_dict(t30),
            "edt_s": edt.rt60_s if edt is not None else None,
            "t20_s": t20.rt60_s if t20 is not None else None,
            "t30_s": t30.rt60_s if t30 is not None else None,
        }
    return {
        "policy": MULTIBAND_LATE_FIELD_POLICY,
        "filter": "causal_butterworth_nominal_octave",
        "filter_order": int(filter_order),
        "direct_sample": int(peak),
        "direct_policy": (
            "absolute_peak" if direct_index is None else "provided_broadband_anchor"
        ),
        "window_policy": "max_base_or_minimum_cycles_at_lower_octave_edge",
        "base_window_ms": float(base_window_ms),
        "minimum_window_cycles": float(minimum_window_cycles),
        "hop_ms": float(hop_ms),
        "threshold": float(threshold),
        "minimum_sustain_ms": float(minimum_sustain_ms),
        "bands": bands,
    }
