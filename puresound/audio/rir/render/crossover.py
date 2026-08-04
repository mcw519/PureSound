"""Low/high band crossover, direct alignment, and causal clipping.

The causal Linkwitz-Riley crossover, the M2.11 energy-matching policy, the
Pyroomacoustics fixed-delay removal, and the ``floor(distance / c * fs)``
causality clip.  Moved out of ``puresound.audio.rir.render.hybrid`` in R2 of
``RIR_EXP_LOG.md``.

The clip preserves the arrival sample itself and zeroes everything strictly
before it; ``puresound.audio.rir.contracts.RIRArray.violates_causality``
checks the same boundary from the consumer side.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.render.arrays import coerce_rir_array
from puresound.audio.rir.scene.sampling import HybridRIRScene
from puresound.audio.rir.scene.schema import RoomSceneV2


def effective_crossover_match_band(
    config: HybridRIRConfig,
) -> tuple[float, float]:
    """Return a valid energy-audit band centered on the actual crossover."""
    nyquist = float(config.sample_rate) / 2.0
    if config.crossover_match_band_hz is None:
        lo_hz = 0.7 * float(config.crossover_hz)
        hi_hz = 1.3 * float(config.crossover_hz)
    else:
        lo_hz, hi_hz = config.crossover_match_band_hz
        if not float(lo_hz) < float(config.crossover_hz) < float(hi_hz):
            raise ValueError(
                "crossover_match_band_hz must contain crossover_hz"
            )
    lo_hz = max(20.0, min(float(lo_hz), nyquist * 0.95))
    hi_hz = max(lo_hz + 1.0, min(float(hi_hz), nyquist * 0.99))
    return float(lo_hz), float(hi_hz)


def match_low_band_to_high_band(
    low_band: np.ndarray,
    high_band: np.ndarray,
    config: HybridRIRConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scale the low band to the high band's level across the match band.

    Returns the scaled band, the applied gain, and the gain *before* clipping.
    The two differ whenever ``crossover_match_gain_range`` binds, and that has
    to stay visible: see the note at the clip.
    """
    lo_hz, hi_hz = effective_crossover_match_band(config)
    if hi_hz <= lo_hz:
        gain = np.ones((low_band.shape[0], 1), dtype=np.float64)
        return low_band, gain, gain

    sos_bp = butter(
        2,
        [lo_hz, hi_hz],
        btype="bandpass",
        fs=int(config.sample_rate),
        output="sos",
    )
    filt = sosfiltfilt if low_band.shape[-1] > 64 else sosfilt
    low_ref = filt(sos_bp, low_band, axis=-1)
    high_ref = filt(sos_bp, high_band, axis=-1)
    low_rms = np.sqrt(np.mean(low_ref**2, axis=-1, keepdims=True))
    high_rms = np.sqrt(np.mean(high_ref**2, axis=-1, keepdims=True))
    target_ratio = 10.0 ** (float(config.crossover_match_target_db) / 20.0)
    raw_gain = high_rms * target_ratio / np.maximum(low_rms, 1e-12)
    gain_min, gain_max = config.crossover_match_gain_range
    gain = np.clip(raw_gain, float(gain_min), float(gain_max))
    # Hitting the bound is not a detail.  The pytARD low band is
    # peak-normalized, so this match is the only thing setting its level
    # against the high band; a clipped gain ships a low band that is under- or
    # over-level by an amount nothing else records.  Carry the raw value out so
    # the caller can report it instead of leaving the truncation silent.
    return low_band * gain, gain, raw_gain


def hybrid_crossover(
    rir_low: np.ndarray,
    rir_high: np.ndarray,
    config: HybridRIRConfig,
) -> np.ndarray:
    output, _metadata = hybrid_crossover_with_metadata(
        rir_low,
        rir_high,
        config,
    )
    return output


def hybrid_crossover_with_metadata(
    rir_low: np.ndarray,
    rir_high: np.ndarray,
    config: HybridRIRConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    low = coerce_rir_array(rir_low, config.num_sources, config.num_samples)
    high = coerce_rir_array(rir_high, config.num_sources, config.num_samples)
    # Linkwitz-Riley (4th order = Butterworth applied twice), filtered causally
    # so the RIR stays causal (no pre-ringing before the direct path). The LR
    # low- and high-pass share an identical phase response, so the two
    # independently simulated bands stay time-aligned and sum to a flat
    # magnitude across the crossover.
    sos_lp = butter(
        2,
        float(config.crossover_hz),
        btype="lowpass",
        fs=int(config.sample_rate),
        output="sos",
    )
    sos_hp = butter(
        2,
        float(config.crossover_hz),
        btype="highpass",
        fs=int(config.sample_rate),
        output="sos",
    )
    low_band = sosfilt(sos_lp, sosfilt(sos_lp, low, axis=-1), axis=-1)
    high_band = sosfilt(sos_hp, sosfilt(sos_hp, high, axis=-1), axis=-1)
    low_gain = np.ones((low_band.shape[0], 1), dtype=np.float64)
    raw_low_gain = low_gain
    effective_match_band_hz = effective_crossover_match_band(config)
    if config.match_crossover_energy:
        low_band, low_gain, raw_low_gain = match_low_band_to_high_band(
            low_band,
            high_band,
            config,
        )
    out = low_band + high_band
    fade_samples = int(
        round(float(config.tail_fade_ms) * 1e-3 * float(config.sample_rate))
    )
    if fade_samples < 0:
        raise ValueError("tail_fade_ms cannot be negative")
    fade_samples = min(fade_samples, out.shape[-1])
    if fade_samples > 1:
        fade = np.square(
            np.cos(np.linspace(0.0, 0.5 * math.pi, fade_samples))
        )
        fade[-1] = 0.0
        out[..., -fade_samples:] *= fade
    elif fade_samples == 1:
        out[..., -1] = 0.0
    peak = float(np.max(np.abs(out)))
    normalization_gain = 1.0
    if (
        config.output_mode == "peak_normalized"
        and peak > 1e-9
        and config.normalize_peak > 0
    ):
        normalization_gain = float(config.normalize_peak) / peak
        out = out * normalization_gain
    metadata = {
        "filter": "causal_linkwitz_riley_fourth_order",
        "crossover_hz": float(config.crossover_hz),
        "energy_matching_requested": bool(config.match_crossover_energy),
        "energy_matching_applied": bool(config.match_crossover_energy),
        "effective_match_band_hz": [
            float(effective_match_band_hz[0]),
            float(effective_match_band_hz[1]),
        ],
        "low_band_gain_by_channel": [
            float(value) for value in low_gain[:, 0]
        ],
        "low_band_gain_requested_by_channel": [
            float(value) for value in raw_low_gain[:, 0]
        ],
        "low_band_gain_range": [
            float(config.crossover_match_gain_range[0]),
            float(config.crossover_match_gain_range[1]),
        ],
        "low_band_gain_clipped_channels": [
            index
            for index, (applied, requested) in enumerate(
                zip(low_gain[:, 0], raw_low_gain[:, 0])
            )
            if not math.isclose(
                float(applied), float(requested), rel_tol=1e-9, abs_tol=1e-12
            )
        ],
        "post_sum_peak_normalization_gain": float(normalization_gain),
        "tail_fade": {
            "policy": "raised_cosine_squared_to_zero",
            "duration_ms": float(config.tail_fade_ms),
            "sample_count": int(fade_samples),
        },
    }
    return out.astype(np.float32), metadata


def align_high_band_direct(
    rir: np.ndarray,
    scene: HybridRIRScene,
    config: HybridRIRConfig,
) -> np.ndarray:
    """Shift the geometric RIR so the direct path lands at ``distance / c``.

    Pyroomacoustics offsets every RIR by a constant fractional-delay length, so
    its direct path arrives later than the true geometric time. The low-frequency
    wave band carries the physical propagation delay from injection, so removing
    this constant offset keeps the two bands time-aligned at the direct path.
    """
    rir = np.asarray(rir, dtype=np.float64)
    mic = np.asarray(scene.mic_pos, dtype=np.float64)
    srcs = np.asarray(scene.source_pos, dtype=np.float64)
    fs = int(config.sample_rate)
    c = max(float(config.sound_speed), 1e-6)
    n = rir.shape[-1]
    shifts: list[int] = []
    for idx in range(min(rir.shape[0], srcs.shape[0])):
        distance = float(np.linalg.norm(srcs[idx] - mic))
        expected = int(round(distance / c * fs))
        lo = max(0, expected - 16)
        hi = min(n, expected + 160)
        if hi <= lo:
            continue
        channel = np.abs(rir[idx])
        peak = float(channel.max())
        if peak <= 0.0:
            continue
        seg = channel[lo:hi]
        threshold = 0.3 * peak
        crossing = int(np.argmax(seg >= threshold))
        if seg[crossing] < threshold:
            continue
        shifts.append((lo + crossing) - expected)
    shift = int(round(float(np.median(shifts)))) if shifts else 0
    if shift > 0:
        out = np.zeros_like(rir)
        out[:, : n - shift] = rir[:, shift:]
    else:
        out = rir.copy()

    # Fractional-delay kernels and the stochastic ray-tracing tail can leave
    # low-level samples before the physical source-to-receiver travel time.
    # Moving the common Pyroomacoustics filter delay to the left also moves
    # those samples to t=0. A real acoustic path cannot arrive before d/c, so
    # enforce that invariant independently for every source after alignment.
    for idx in range(min(out.shape[0], srcs.shape[0])):
        distance = float(np.linalg.norm(srcs[idx] - mic))
        first_physical_sample = int(math.floor(distance / c * fs))
        out[idx, : max(0, min(first_physical_sample, n))] = 0.0
    return out


def clip_rir_before_physical_arrival(
    rir: np.ndarray,
    scene: HybridRIRScene | RoomSceneV2,
    config: HybridRIRConfig,
) -> np.ndarray:
    """Enforce the discrete geometric-arrival support on each source channel.

    The low-frequency ARD/modal approximation is causal in its continuous wave
    model, but a finite voxel/modal reconstruction can produce tiny numerical
    support before the source-to-receiver travel time.  Keeping those samples
    would make a hybrid RIR fail the bank's hard causality invariant and would
    smear the direct arrival.  We therefore apply the explicit M6 contract to
    every low-band backend before crossover.  The first allowed sample is
    ``floor(distance / c * fs)``; the boundary sample itself is preserved.
    """
    output = np.asarray(rir, dtype=np.float64).copy()
    if output.ndim != 2:
        raise ValueError(f"expected [channels, samples] RIR, got {output.shape}")
    distances = scene.source_distances()
    sound_speed = max(float(config.sound_speed), 1e-6)
    sample_rate = float(config.sample_rate)
    for channel, distance in enumerate(distances[: output.shape[0]]):
        first_physical = int(
            math.floor(float(distance) / sound_speed * sample_rate)
        )
        if first_physical > 0:
            output[channel, : min(first_physical, output.shape[1])] = 0.0
    return output


__all__ = [
    "align_high_band_direct",
    "clip_rir_before_physical_arrival",
    "effective_crossover_match_band",
    "hybrid_crossover",
    "hybrid_crossover_with_metadata",
    "match_low_band_to_high_band",
]
