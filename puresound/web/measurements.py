"""Lightweight, reproducible audio measurements for the web workspace.

The functions in this module intentionally return JSON-safe scalar values. They
are useful for a quick model comparison when no full benchmark manifest is
available; reference-dependent scores are left absent rather than fabricated.
Reference-free DNSMOS is opt-in because its scorer may load an additional model.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

import numpy as np

from puresound.evaluation.reference import reference_metrics as reference_metrics
from puresound.inference.processors.base import load_audio


_DNSMOS_LOCK = threading.Lock()


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


def align_streaming_output(
    candidate: Any,
    *,
    latency_samples: int = 0,
    target_samples: int | None = None,
) -> np.ndarray:
    """Remove streaming warm-up and bound an output to the source duration."""

    values = np.asarray(candidate, dtype=np.float32).reshape(-1)
    latency = max(0, int(latency_samples))
    if latency >= values.size:
        raise ValueError("streaming latency consumes the entire model output")
    values = values[latency:]
    if target_samples is not None:
        values = values[: max(0, int(target_samples))]
    if values.size == 0:
        raise ValueError("aligned model output is empty")
    return values


def reference_free_metrics(
    samples: Any,
    sample_rate: int,
    *,
    include_dnsmos: bool = False,
) -> dict[str, Any]:
    """Return quality metrics that do not require a clean reference.

    DNSMOS is intentionally optional at runtime.  Its model and audio
    dependencies can be relatively heavy (and may not be installed in a
    minimal CPU environment), so an unavailable scorer is represented in the
    report instead of making the whole inference fail.
    """

    result: dict[str, Any] = {"dnsmos": {}}
    if not include_dnsmos:
        return result
    try:
        import torch

        from puresound.metrics import Metrics

        values = np.asarray(samples, dtype=np.float32).reshape(-1)
        with _DNSMOS_LOCK:
            scores = Metrics.dnsmos_p835(
                torch.zeros(1, dtype=torch.float32),
                torch.from_numpy(values),
                sr=int(sample_rate),
            )
        result["dnsmos"] = {
            str(name): _finite(float(value)) for name, value in scores.items()
        }
    except Exception as exc:  # optional scorer must not block core metrics
        result["dnsmos_error"] = str(exc)
    return result
