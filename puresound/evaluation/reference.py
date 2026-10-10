"""Reference-dependent quality scores for two aligned waveforms."""

from __future__ import annotations

from typing import Any

import numpy as np


def _finite(value: float | None) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return float(value)


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


__all__ = ["reference_metrics"]
