"""Causal PathEvent-early / multiband-FDN-late coupling for M4.4.

The coupling is deliberately independent of the production hybrid generator.
Callers opt in with a PathEvent waveform, a physical direct-arrival sample, and
material-derived octave decay targets.  Samples through the transition start
are preserved exactly; a complementary equal-power crossfade then replaces
the coherent later paths with an internally contractive FDN tail.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from puresound.audio.multiband_fdn import (
    MultibandFDNDesign,
    design_multiband_fdn,
    render_multiband_fdn,
)


PATH_EVENT_FDN_COUPLING_POLICY = "puresound.path_event_fdn_coupling.v1"


@dataclass(frozen=True)
class PathEventFDNCouplingResult:
    """One coupled RIR plus its timing, energy, and FDN design evidence."""

    rir: np.ndarray
    coherent_component: np.ndarray
    diffuse_component: np.ndarray
    early_weight: np.ndarray
    late_weight: np.ndarray
    design: MultibandFDNDesign
    metadata: Mapping[str, Any]


def equal_power_transition_weights(
    sample_count: int,
    transition_start_sample: int,
    transition_end_sample: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return complementary cosine/sine weights with exact endpoint values."""

    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    start = int(transition_start_sample)
    end = int(transition_end_sample)
    if not 0 <= start < end < sample_count:
        raise ValueError(
            "transition must satisfy 0 <= start < end < sample_count"
        )

    early = np.ones(sample_count, dtype=np.float64)
    late = np.zeros(sample_count, dtype=np.float64)
    phase = np.linspace(0.0, 0.5 * math.pi, end - start + 1)
    early[start : end + 1] = np.cos(phase)
    late[start : end + 1] = np.sin(phase)
    early[end + 1 :] = 0.0
    late[end + 1 :] = 1.0
    return early, late


def _transition_samples(
    sample_count: int,
    sample_rate: int,
    direct_sample: int,
    mixing_time_s: float,
    transition_duration_s: float,
) -> tuple[int, int, int]:
    center = direct_sample + int(round(mixing_time_s * sample_rate))
    width = max(2, int(round(transition_duration_s * sample_rate)))
    start = max(direct_sample + 1, center - width // 2)
    end = start + width
    if end >= sample_count:
        end = sample_count - 1
        start = end - width
    if start <= direct_sample or start < 0 or end <= start:
        raise ValueError(
            "RIR is too short for the requested direct-relative mixing transition"
        )
    return int(center), int(start), int(end)


def transition_samples(
    sample_count: int,
    sample_rate: int,
    direct_sample: int,
    mixing_time_s: float,
    transition_duration_s: float,
) -> tuple[int, int, int]:
    """Public validated direct-relative transition timing contract."""

    return _transition_samples(
        sample_count,
        sample_rate,
        direct_sample,
        mixing_time_s,
        transition_duration_s,
    )


def _energy_preserving_diffuse_gain(
    original: np.ndarray,
    coherent: np.ndarray,
    diffuse: np.ndarray,
    transition_start_sample: int,
    *,
    target_energy: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Solve the positive quadratic root that preserves post-start energy."""

    start = int(transition_start_sample)
    original_tail = original[start:]
    coherent_tail = coherent[start:]
    diffuse_tail = diffuse[start:]
    finite_original_energy = float(np.dot(original_tail, original_tail))
    target = (
        finite_original_energy
        if target_energy is None
        else float(target_energy)
    )
    if not np.isfinite(target) or target < finite_original_energy:
        raise ValueError(
            "diffuse target energy must be finite and no smaller than the "
            "finite coherent reference"
        )
    coherent_energy = float(np.dot(coherent_tail, coherent_tail))
    diffuse_energy = float(np.dot(diffuse_tail, diffuse_tail))
    cross_inner_product = float(np.dot(coherent_tail, diffuse_tail))

    if target <= np.finfo(np.float64).tiny or diffuse_energy <= np.finfo(
        np.float64
    ).tiny:
        gain = 0.0
    else:
        remaining = max(0.0, target - coherent_energy)
        discriminant = max(
            0.0,
            cross_inner_product**2 + diffuse_energy * remaining,
        )
        gain = (
            -cross_inner_product + math.sqrt(discriminant)
        ) / diffuse_energy
    output_tail = coherent_tail + gain * diffuse_tail
    realized = float(np.dot(output_tail, output_tail))
    relative_error = (
        abs(realized - target) / target
        if target > np.finfo(np.float64).tiny
        else 0.0
    )
    return float(gain), {
        "target_original_post_transition_energy": target,
        "finite_original_post_transition_energy": finite_original_energy,
        "weighted_coherent_energy": coherent_energy,
        "weighted_fdn_energy_before_gain": diffuse_energy,
        "coherent_fdn_cross_inner_product": cross_inner_product,
        "fdn_output_gain": float(gain),
        "realized_coupled_post_transition_energy": realized,
        "relative_energy_error": float(relative_error),
    }


def energy_preserving_diffuse_gain(
    original: np.ndarray,
    coherent: np.ndarray,
    diffuse: np.ndarray,
    transition_start_sample: int,
    *,
    target_energy: float | None = None,
) -> tuple[float, dict[str, float]]:
    """Public positive-root solve used by mono and synchronized renderers."""

    return _energy_preserving_diffuse_gain(
        original,
        coherent,
        diffuse,
        transition_start_sample,
        target_energy=target_energy,
    )


def extrapolated_path_tail_energy_target(
    original: np.ndarray,
    transition_start_sample: int,
    sample_rate: int,
    target_rt60_s_by_hz: Mapping[float, float],
    *,
    terminal_window_s: float = 0.05,
) -> tuple[float, dict[str, float | int | str]]:
    """Extend a finite PathEvent tail with its material RT60 decay law.

    A maximum-order path renderer stops producing events long before a long
    material decay reaches the end of the RIR.  The final active window gives
    a local energy density; the material-derived median RT60 analytically
    integrates that density from the last event to the render boundary.
    """

    signal = np.asarray(original, dtype=np.float64).squeeze()
    start = int(transition_start_sample)
    if signal.ndim != 1 or not 0 <= start < signal.size:
        raise ValueError("invalid PathEvent tail for energy extrapolation")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    rt60_values = np.asarray(
        [float(value) for value in target_rt60_s_by_hz.values()],
        dtype=np.float64,
    )
    if (
        rt60_values.size == 0
        or not np.all(np.isfinite(rt60_values))
        or np.any(rt60_values <= 0.0)
    ):
        raise ValueError("material RT60 targets must be finite and positive")

    tail = signal[start:]
    finite_energy = float(np.dot(tail, tail))
    peak = float(np.max(np.abs(tail))) if tail.size else 0.0
    active = np.flatnonzero(np.abs(tail) > max(peak * 1e-10, 1e-15))
    effective_rt60 = float(np.median(rt60_values))
    if active.size == 0:
        return finite_energy, {
            "policy": "material_rt60_finite_path_extrapolation.v1",
            "finite_energy": finite_energy,
            "extrapolated_energy": 0.0,
            "effective_rt60_s": effective_rt60,
            "last_active_sample": start,
            "remaining_samples": int(signal.size - start - 1),
        }

    last = start + int(active[-1])
    remaining = int(signal.size - last - 1)
    window = max(8, int(round(float(terminal_window_s) * sample_rate)))
    window_start = max(start, last - window + 1)
    terminal = signal[window_start : last + 1]
    local_mean_energy = float(np.dot(terminal, terminal) / max(terminal.size, 1))
    decay_per_sample = math.exp(
        -6.0 * math.log(10.0) / (effective_rt60 * float(sample_rate))
    )
    extrapolated = (
        local_mean_energy
        * decay_per_sample
        * (1.0 - decay_per_sample**remaining)
        / max(1.0 - decay_per_sample, np.finfo(np.float64).eps)
        if remaining > 0
        else 0.0
    )
    target = finite_energy + extrapolated
    return float(target), {
        "policy": "material_rt60_finite_path_extrapolation.v1",
        "finite_energy": finite_energy,
        "extrapolated_energy": float(extrapolated),
        "effective_rt60_s": effective_rt60,
        "terminal_window_samples": int(terminal.size),
        "terminal_mean_energy_per_sample": local_mean_energy,
        "last_active_sample": int(last),
        "remaining_samples": remaining,
    }


def couple_path_event_rir_with_fdn(
    path_event_rir: Any,
    sample_rate: int,
    direct_sample: int,
    target_rt60_s_by_hz: Mapping[float, float],
    *,
    mixing_time_s: float = 0.024,
    transition_duration_s: float = 0.016,
    delay_line_count: int = 16,
    seed: int = 0,
    filter_order: int = 4,
) -> PathEventFDNCouplingResult:
    """Replace coherent later paths with a causal, energy-matched FDN tail.

    The FDN is excited by the PathEvent response multiplied by the early
    transition weight.  Direct and early event amplitudes therefore seed the
    late state, while paths after the transition cannot keep injecting a sparse
    coherent pattern.  A scalar positive-root solve preserves total energy from
    the transition start through the end of the finite RIR.
    """

    signal = np.asarray(path_event_rir, dtype=np.float64).squeeze()
    if signal.ndim == 0:
        signal = signal.reshape(1)
    if signal.ndim != 1:
        raise ValueError("PathEvent RIR must be one-dimensional")
    if signal.size < 3 or not np.all(np.isfinite(signal)):
        raise ValueError("PathEvent RIR must contain at least three finite samples")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    direct = int(direct_sample)
    if not 0 <= direct < signal.size:
        raise ValueError("direct_sample lies outside the PathEvent RIR")
    if not np.isfinite(mixing_time_s) or mixing_time_s <= 0.0:
        raise ValueError("mixing_time_s must be finite and positive")
    if not np.isfinite(transition_duration_s) or transition_duration_s <= 0.0:
        raise ValueError("transition_duration_s must be finite and positive")

    center, start, end = _transition_samples(
        signal.size,
        int(sample_rate),
        direct,
        float(mixing_time_s),
        float(transition_duration_s),
    )
    early_weight, late_weight = equal_power_transition_weights(
        signal.size,
        start,
        end,
    )
    design = design_multiband_fdn(
        int(sample_rate),
        target_rt60_s_by_hz,
        target_mixing_time_s=float(mixing_time_s),
        delay_line_count=int(delay_line_count),
        seed=int(seed),
    )

    excitation = signal * early_weight
    fdn = render_multiband_fdn(
        design,
        excitation,
        filter_order=int(filter_order),
    )
    coherent = excitation
    diffuse_before_gain = fdn.rir * late_weight
    target_energy, extrapolation_metadata = extrapolated_path_tail_energy_target(
        signal,
        start,
        int(sample_rate),
        target_rt60_s_by_hz,
    )
    diffuse_gain, energy_metadata = _energy_preserving_diffuse_gain(
        signal,
        coherent,
        diffuse_before_gain,
        start,
        target_energy=target_energy,
    )
    diffuse = diffuse_gain * diffuse_before_gain
    coupled = coherent + diffuse
    pre_transition_error = float(
        np.max(np.abs(coupled[: start + 1] - signal[: start + 1]))
    )
    metadata = {
        "policy": PATH_EVENT_FDN_COUPLING_POLICY,
        "scope": "opt_in_path_event_early_multiband_fdn_late",
        "sample_rate": int(sample_rate),
        "sample_count": int(signal.size),
        "seed": int(seed),
        "direct_sample": direct,
        "direct_time_s": float(direct / sample_rate),
        "mixing_time_s_after_direct": float(mixing_time_s),
        "transition_duration_s": float(transition_duration_s),
        "transition_center_sample": center,
        "transition_start_sample": start,
        "transition_end_sample": end,
        "transition_start_s_after_direct": float((start - direct) / sample_rate),
        "transition_end_s_after_direct": float((end - direct) / sample_rate),
        "crossfade": "complementary_equal_power_cosine_sine",
        "injection": "path_event_response_times_early_weight",
        "post_transition_energy_policy": (
            "material_rt60_extrapolated_positive_quadratic_root"
        ),
        "pre_transition_max_abs_error": pre_transition_error,
        "finite_output": bool(np.all(np.isfinite(coupled))),
        "energy": energy_metadata,
        "path_tail_extrapolation": extrapolation_metadata,
        "fdn_design": design.to_dict(include_coefficients=False),
    }
    return PathEventFDNCouplingResult(
        rir=np.asarray(coupled, dtype=np.float64),
        coherent_component=np.asarray(coherent, dtype=np.float64),
        diffuse_component=np.asarray(diffuse, dtype=np.float64),
        early_weight=early_weight,
        late_weight=late_weight,
        design=design,
        metadata=metadata,
    )


__all__ = [
    "PATH_EVENT_FDN_COUPLING_POLICY",
    "PathEventFDNCouplingResult",
    "couple_path_event_rir_with_fdn",
    "energy_preserving_diffuse_gain",
    "equal_power_transition_weights",
    "extrapolated_path_tail_energy_target",
    "transition_samples",
]
