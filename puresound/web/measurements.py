"""Lightweight, reproducible audio measurements for the web workspace.

The functions in this module intentionally return JSON-safe scalar values. They
are useful for a quick model comparison when no full benchmark manifest is
available; reference-dependent scores are left absent rather than fabricated.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from puresound.inference.processors.base import load_audio


def _db(value: float, floor: float = -120.0) -> float:
    if value <= 1e-12:
        return floor
    return float(max(floor, 20.0 * np.log10(value)))


def _finite(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


def audio_metrics(samples: Any, sample_rate: int) -> dict[str, Any]:
    """Return level, clipping, silence, and spectral summary metrics."""

    values = np.asarray(samples, dtype=np.float32).reshape(-1)
    if values.size == 0:
        raise ValueError("audio input is empty")
    values = np.nan_to_num(values, nan=0.0, posinf=1.0, neginf=-1.0)
    peak = float(np.max(np.abs(values)))
    rms = float(np.sqrt(np.mean(np.square(values))))
    crossings = np.count_nonzero(np.signbit(values[1:]) != np.signbit(values[:-1])) if values.size > 1 else 0
    frame_size = min(1024, values.size)
    hop = max(1, frame_size // 2)
    centroid_values: list[float] = []
    if frame_size >= 16:
        window = np.hanning(frame_size)
        frequencies = np.fft.rfftfreq(frame_size, 1.0 / sample_rate)
        for start in range(0, max(1, values.size - frame_size + 1), hop):
            frame = values[start : start + frame_size]
            if frame.size < frame_size:
                frame = np.pad(frame, (0, frame_size - frame.size))
            magnitude = np.abs(np.fft.rfft(frame * window))
            total = float(np.sum(magnitude))
            if total > 1e-12:
                centroid_values.append(float(np.dot(frequencies, magnitude) / total))
    return {
        "sample_rate": int(sample_rate),
        "samples": int(values.size),
        "duration_seconds": float(values.size / sample_rate),
        "peak_dbfs": _db(peak),
        "rms_dbfs": _db(rms),
        "crest_factor_db": _db(peak / max(rms, 1e-12)),
        "clipping_ratio": float(np.mean(np.abs(values) >= 0.999)),
        "silence_ratio": float(np.mean(np.abs(values) < 1e-4)),
        "zero_crossing_rate": float(crossings / max(1, values.size - 1)),
        "spectral_centroid_hz": _finite(float(np.mean(centroid_values)) if centroid_values else None),
    }


def audio_metrics_from_path(path: str | Path, sample_rate: int = 16_000) -> tuple[np.ndarray, dict[str, Any]]:
    values, actual_rate = load_audio(path, sample_rate=sample_rate)
    return values, audio_metrics(values, actual_rate)


def reference_metrics(reference: Any, candidate: Any, sample_rate: int) -> dict[str, float | None]:
    """Compare two aligned waveforms using reference-dependent quality scores."""

    target = np.asarray(reference, dtype=np.float64).reshape(-1)
    estimate = np.asarray(candidate, dtype=np.float64).reshape(-1)
    length = min(target.size, estimate.size)
    if length < 2:
        return {"si_sdr_db": None, "snr_db": None, "correlation": None, "stoi": None, "pesq_wb": None}
    target = target[:length]
    estimate = estimate[:length]
    target = target - np.mean(target)
    estimate = estimate - np.mean(estimate)
    target_energy = float(np.dot(target, target))
    error = estimate - target
    alpha = float(np.dot(estimate, target) / max(target_energy, 1e-12))
    projected = alpha * target
    residual = estimate - projected
    si_sdr = 10.0 * np.log10(max(float(np.dot(projected, projected)), 1e-12) / max(float(np.dot(residual, residual)), 1e-12))
    snr = 10.0 * np.log10(max(target_energy, 1e-12) / max(float(np.dot(error, error)), 1e-12))
    correlation = float(np.dot(target, estimate) / np.sqrt(max(target_energy * float(np.dot(estimate, estimate)), 1e-12)))
    result: dict[str, float | None] = {
        "si_sdr_db": _finite(float(si_sdr)),
        "snr_db": _finite(float(snr)),
        "correlation": _finite(correlation),
        "stoi": None,
        "pesq_wb": None,
    }
    try:
        from pystoi.stoi import stoi

        result["stoi"] = _finite(float(stoi(target.astype(np.float32), estimate.astype(np.float32), sample_rate)))
        if sample_rate == 16_000:
            from pesq import pesq

            result["pesq_wb"] = _finite(float(pesq(16_000, target.astype(np.float32), estimate.astype(np.float32), "wb")))
    except Exception:
        # Optional metric dependencies and short-reference constraints should
        # not prevent the always-available NumPy scores from being reported.
        pass
    return result
