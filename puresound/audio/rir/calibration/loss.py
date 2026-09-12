"""Reference multi-objective measured/synthetic RIR loss contract for M5.1."""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.signal import csd, stft, welch

from puresound.audio.rir.metrics import (
    direct_sample,
    estimate_decay_time,
    octave_band_rir,
    valid_octave_centers,
)


RIR_CALIBRATION_LOSS_POLICY = "puresound.rir_calibration_loss.v1"


@dataclass(frozen=True)
class CalibrationLossWeights:
    """Non-negative weights for independently auditable calibration terms."""

    multiresolution_stft: float = 1.0
    energy_decay: float = 1.0
    arrival_timing: float = 1.0
    octave_acoustics: float = 1.0
    spatial_coherence: float = 1.0
    causality: float = 10.0
    decay_regularization: float = 1.0

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(value) or value < 0.0 for value in self.to_dict().values()
        ):
            raise ValueError("calibration loss weights must be finite and non-negative")

    def to_dict(self) -> dict[str, float]:
        return {
            "multiresolution_stft": float(self.multiresolution_stft),
            "energy_decay": float(self.energy_decay),
            "arrival_timing": float(self.arrival_timing),
            "octave_acoustics": float(self.octave_acoustics),
            "spatial_coherence": float(self.spatial_coherence),
            "causality": float(self.causality),
            "decay_regularization": float(self.decay_regularization),
        }


@dataclass(frozen=True)
class CalibrationLossReport:
    """Scalar objective plus term-level evidence and configuration."""

    total: float
    terms: Mapping[str, float]
    diagnostics: Mapping[str, Any]
    weights: CalibrationLossWeights

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": RIR_CALIBRATION_LOSS_POLICY,
            "implementation": "numpy_scipy_reference_not_autograd",
            "total": float(self.total),
            "terms": {key: float(value) for key, value in self.terms.items()},
            "weights": self.weights.to_dict(),
            "diagnostics": dict(self.diagnostics),
        }


def _channels_first(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 2:
        raise ValueError(f"{name} must have shape [channel, sample]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite samples")
    return array


def _matched_signals(measured: Any, synthetic: Any) -> tuple[np.ndarray, np.ndarray]:
    reference = _channels_first(measured, "measured RIR")
    candidate = _channels_first(synthetic, "synthetic RIR")
    if reference.shape[0] != candidate.shape[0]:
        raise ValueError("measured and synthetic RIRs must have equal channel count")
    length = min(reference.shape[1], candidate.shape[1])
    return reference[:, :length], candidate[:, :length]


def _direct_samples(
    values: np.ndarray,
    provided: Sequence[int] | None,
    name: str,
) -> tuple[int, ...]:
    if provided is None:
        return tuple(direct_sample(channel) for channel in values)
    result = tuple(int(value) for value in provided)
    if len(result) != values.shape[0] or any(
        value < 0 or value >= values.shape[1] for value in result
    ):
        raise ValueError(f"{name} must match channels and lie inside each RIR")
    return result


def multiresolution_stft_distance(
    measured: Any,
    synthetic: Any,
    *,
    fft_sizes: Sequence[int] = (256, 512, 1024),
    log_floor: float = 1e-7,
) -> dict[str, Any]:
    """Return spectral-convergence plus log-magnitude distance at each scale."""

    reference, candidate = _matched_signals(measured, synthetic)
    if not fft_sizes or any(int(size) < 16 for size in fft_sizes):
        raise ValueError("fft_sizes must contain values of at least 16")
    if not math.isfinite(log_floor) or log_floor <= 0.0:
        raise ValueError("log_floor must be finite and positive")
    resolutions: dict[str, Any] = {}
    totals = []
    for requested_size in fft_sizes:
        fft_size = int(requested_size)
        channel_convergence = []
        channel_log_magnitude = []
        for measured_channel, synthetic_channel in zip(reference, candidate):
            segment_size = min(fft_size, measured_channel.size)
            hop_size = max(1, segment_size // 4)
            _, _, measured_stft = stft(
                measured_channel,
                nperseg=segment_size,
                noverlap=segment_size - hop_size,
                nfft=fft_size,
                boundary=None,
                padded=False,
            )
            _, _, synthetic_stft = stft(
                synthetic_channel,
                nperseg=segment_size,
                noverlap=segment_size - hop_size,
                nfft=fft_size,
                boundary=None,
                padded=False,
            )
            measured_magnitude = np.abs(measured_stft)
            synthetic_magnitude = np.abs(synthetic_stft)
            denominator = float(np.linalg.norm(measured_magnitude))
            convergence = float(
                np.linalg.norm(synthetic_magnitude - measured_magnitude)
                / max(denominator, log_floor)
            )
            log_magnitude = float(
                np.mean(
                    np.abs(
                        np.log(synthetic_magnitude + log_floor)
                        - np.log(measured_magnitude + log_floor)
                    )
                )
            )
            channel_convergence.append(convergence)
            channel_log_magnitude.append(log_magnitude)
        spectral_convergence = float(np.mean(channel_convergence))
        log_magnitude = float(np.mean(channel_log_magnitude))
        total = spectral_convergence + log_magnitude
        resolutions[str(fft_size)] = {
            "fft_size": fft_size,
            "spectral_convergence": spectral_convergence,
            "log_magnitude_l1": log_magnitude,
            "total": total,
        }
        totals.append(total)
    return {
        "distance": float(np.mean(totals)),
        "resolutions": resolutions,
    }


def _direct_relative_pair(
    measured: np.ndarray,
    synthetic: np.ndarray,
    measured_direct: int,
    synthetic_direct: int,
) -> tuple[np.ndarray, np.ndarray]:
    length = min(
        measured.size - measured_direct,
        synthetic.size - synthetic_direct,
    )
    return (
        measured[measured_direct : measured_direct + length],
        synthetic[synthetic_direct : synthetic_direct + length],
    )


def _normalized_decay_db(signal: np.ndarray, floor_db: float) -> np.ndarray:
    energy = np.square(signal)
    integrated = np.cumsum(energy[::-1], dtype=np.float64)[::-1]
    maximum = float(integrated[0]) if integrated.size else 0.0
    if maximum <= np.finfo(np.float64).tiny:
        return np.full(signal.shape, float(floor_db), dtype=np.float64)
    decay = 10.0 * np.log10(np.maximum(integrated / maximum, 10.0 ** (floor_db / 10.0)))
    return np.maximum(decay, float(floor_db))


def energy_decay_curve_distance(
    measured: Any,
    synthetic: Any,
    measured_direct_samples: Sequence[int],
    synthetic_direct_samples: Sequence[int],
    *,
    floor_db: float = -80.0,
) -> dict[str, Any]:
    """Compare normalized direct-relative Schroeder energy-decay curves."""

    reference, candidate = _matched_signals(measured, synthetic)
    measured_direct = _direct_samples(
        reference,
        measured_direct_samples,
        "measured_direct_samples",
    )
    synthetic_direct = _direct_samples(
        candidate,
        synthetic_direct_samples,
        "synthetic_direct_samples",
    )
    if not math.isfinite(floor_db) or floor_db >= 0.0:
        raise ValueError("floor_db must be finite and negative")
    channel_rmse_db = []
    for channel_index in range(reference.shape[0]):
        measured_tail, synthetic_tail = _direct_relative_pair(
            reference[channel_index],
            candidate[channel_index],
            measured_direct[channel_index],
            synthetic_direct[channel_index],
        )
        measured_decay = _normalized_decay_db(measured_tail, floor_db)
        synthetic_decay = _normalized_decay_db(synthetic_tail, floor_db)
        channel_rmse_db.append(
            float(np.sqrt(np.mean(np.square(synthetic_decay - measured_decay))))
        )
    rmse_db = float(np.mean(channel_rmse_db))
    return {
        "distance": float(rmse_db / abs(floor_db)),
        "mean_rmse_db": rmse_db,
        "channel_rmse_db": channel_rmse_db,
        "floor_db": float(floor_db),
    }


def arrival_timing_distance(
    measured_direct_samples: Sequence[int],
    synthetic_direct_samples: Sequence[int],
    sample_rate: int,
    *,
    tolerance_ms: float = 1.0,
) -> dict[str, Any]:
    """Return direct-arrival error normalized by an explicit tolerance."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not math.isfinite(tolerance_ms) or tolerance_ms <= 0.0:
        raise ValueError("arrival tolerance must be finite and positive")
    measured_values = np.asarray(measured_direct_samples, dtype=np.float64)
    synthetic_values = np.asarray(synthetic_direct_samples, dtype=np.float64)
    if (
        measured_values.shape != synthetic_values.shape
        or measured_values.ndim != 1
        or measured_values.size == 0
        or not np.all(np.isfinite(measured_values))
        or not np.all(np.isfinite(synthetic_values))
    ):
        raise ValueError("measured and synthetic direct samples must have equal shape")
    difference_ms = (synthetic_values - measured_values) / float(sample_rate) * 1000.0
    mean_absolute_ms = float(np.mean(np.abs(difference_ms)))
    return {
        "distance": float(mean_absolute_ms / tolerance_ms),
        "difference_ms_by_channel": difference_ms.astype(float).tolist(),
        "mean_absolute_difference_ms": mean_absolute_ms,
        "tolerance_ms": float(tolerance_ms),
    }


def octave_acoustic_distance(
    measured: Any,
    synthetic: Any,
    sample_rate: int,
    measured_direct_samples: Sequence[int],
    synthetic_direct_samples: Sequence[int],
    *,
    centers_hz: Sequence[float] = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0),
) -> dict[str, Any]:
    """Compare direct-relative octave energy and qualified T20 decay."""

    reference, candidate = _matched_signals(measured, synthetic)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    measured_direct = _direct_samples(
        reference,
        measured_direct_samples,
        "measured_direct_samples",
    )
    synthetic_direct = _direct_samples(
        candidate,
        synthetic_direct_samples,
        "synthetic_direct_samples",
    )
    valid_centers = valid_octave_centers(sample_rate, centers_hz)
    bands: dict[str, Any] = {}
    distances = []
    for center_hz in valid_centers:
        energy_errors_db = []
        log_t20_errors = []
        for channel_index in range(reference.shape[0]):
            measured_tail, synthetic_tail = _direct_relative_pair(
                reference[channel_index],
                candidate[channel_index],
                measured_direct[channel_index],
                synthetic_direct[channel_index],
            )
            measured_band = octave_band_rir(
                measured_tail,
                sample_rate,
                center_hz,
            )
            synthetic_band = octave_band_rir(
                synthetic_tail,
                sample_rate,
                center_hz,
            )
            measured_energy = float(np.dot(measured_band, measured_band))
            synthetic_energy = float(np.dot(synthetic_band, synthetic_band))
            floor = np.finfo(np.float64).tiny
            energy_errors_db.append(
                float(
                    abs(
                        10.0
                        * math.log10(
                            max(synthetic_energy, floor) / max(measured_energy, floor)
                        )
                    )
                )
            )
            measured_t20 = estimate_decay_time(
                measured_band,
                sample_rate,
                -5.0,
                -25.0,
                direct_index=0,
            )
            synthetic_t20 = estimate_decay_time(
                synthetic_band,
                sample_rate,
                -5.0,
                -25.0,
                direct_index=0,
            )
            if measured_t20 is not None and synthetic_t20 is not None:
                log_t20_errors.append(
                    float(abs(math.log(synthetic_t20.rt60_s / measured_t20.rt60_s)))
                )
        energy_distance = float(np.mean(energy_errors_db) / 20.0)
        decay_distance = float(np.mean(log_t20_errors)) if log_t20_errors else 0.0
        distance = energy_distance + decay_distance
        bands[f"{center_hz:g}"] = {
            "center_hz": float(center_hz),
            "mean_absolute_energy_error_db": float(np.mean(energy_errors_db)),
            "qualified_t20_channels": len(log_t20_errors),
            "mean_absolute_log_t20_ratio": decay_distance,
            "distance": distance,
        }
        distances.append(distance)
    return {
        "distance": float(np.mean(distances)) if distances else 0.0,
        "bands": bands,
    }


def _complex_pair_coherence(
    first: np.ndarray,
    second: np.ndarray,
    sample_rate: int,
    nperseg: int,
) -> tuple[np.ndarray, np.ndarray]:
    segment_size = min(int(nperseg), first.size, second.size)
    if segment_size < 16:
        return np.zeros(0), np.zeros(0, dtype=np.complex128)
    overlap = segment_size // 2
    frequencies, first_psd = welch(
        first,
        fs=sample_rate,
        nperseg=segment_size,
        noverlap=overlap,
    )
    _, second_psd = welch(
        second,
        fs=sample_rate,
        nperseg=segment_size,
        noverlap=overlap,
    )
    _, cross = csd(
        first,
        second,
        fs=sample_rate,
        nperseg=segment_size,
        noverlap=overlap,
    )
    denominator = np.sqrt(np.maximum(first_psd * second_psd, 0.0))
    coherence = np.zeros(cross.shape, dtype=np.complex128)
    valid = denominator > np.finfo(np.float64).tiny
    coherence[valid] = cross[valid] / denominator[valid]
    return frequencies[valid], coherence[valid]


def spatial_coherence_distance(
    measured: Any,
    synthetic: Any,
    sample_rate: int,
    measured_direct_samples: Sequence[int],
    synthetic_direct_samples: Sequence[int],
    *,
    start_ms: float = 80.0,
    nperseg: int = 256,
) -> dict[str, Any]:
    """Compare measured and synthetic late complex coherence for every pair."""

    reference, candidate = _matched_signals(measured, synthetic)
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if int(nperseg) < 16:
        raise ValueError("spatial coherence nperseg must be at least 16")
    measured_direct = _direct_samples(
        reference,
        measured_direct_samples,
        "measured_direct_samples",
    )
    synthetic_direct = _direct_samples(
        candidate,
        synthetic_direct_samples,
        "synthetic_direct_samples",
    )
    if reference.shape[0] < 2:
        return {
            "distance": 0.0,
            "evaluable": False,
            "reason": "fewer_than_two_synchronized_receivers",
            "pairs": {},
        }
    if not math.isfinite(start_ms) or start_ms < 0.0:
        raise ValueError("spatial coherence start_ms must be finite and non-negative")
    pairs: dict[str, Any] = {}
    distances = []
    offset = int(round(start_ms * 1e-3 * sample_rate))
    for first_index, second_index in itertools.combinations(
        range(reference.shape[0]),
        2,
    ):
        measured_start = (
            min(
                measured_direct[first_index],
                measured_direct[second_index],
            )
            + offset
        )
        synthetic_start = (
            min(
                synthetic_direct[first_index],
                synthetic_direct[second_index],
            )
            + offset
        )
        common_length = min(
            reference.shape[1] - measured_start,
            candidate.shape[1] - synthetic_start,
        )
        measured_frequency, measured_coherence = _complex_pair_coherence(
            reference[
                first_index,
                measured_start : measured_start + common_length,
            ],
            reference[
                second_index,
                measured_start : measured_start + common_length,
            ],
            sample_rate,
            nperseg,
        )
        synthetic_frequency, synthetic_coherence = _complex_pair_coherence(
            candidate[
                first_index,
                synthetic_start : synthetic_start + common_length,
            ],
            candidate[
                second_index,
                synthetic_start : synthetic_start + common_length,
            ],
            sample_rate,
            nperseg,
        )
        length = min(measured_coherence.size, synthetic_coherence.size)
        if length == 0:
            continue
        if not np.array_equal(
            measured_frequency[:length],
            synthetic_frequency[:length],
        ):
            raise RuntimeError("measured and synthetic coherence grids differ")
        rmse = float(
            np.sqrt(
                np.mean(
                    np.square(
                        np.abs(
                            synthetic_coherence[:length] - measured_coherence[:length]
                        )
                    )
                )
            )
        )
        key = f"{first_index}-{second_index}"
        pairs[key] = {
            "first_channel": first_index,
            "second_channel": second_index,
            "frequency_bin_count": int(length),
            "complex_rmse": rmse,
        }
        distances.append(rmse)
    return {
        "distance": float(np.mean(distances)) if distances else 0.0,
        "evaluable": bool(distances),
        "reason": "ok" if distances else "insufficient_late_samples",
        "start_ms": float(start_ms),
        "nperseg": int(nperseg),
        "pairs": pairs,
    }


def causality_penalty(
    synthetic: Any,
    physical_first_samples: Sequence[int],
) -> dict[str, Any]:
    """Penalize synthetic energy before the measured physical arrival bound."""

    candidate = _channels_first(synthetic, "synthetic RIR")
    first_samples = _direct_samples(
        candidate,
        physical_first_samples,
        "physical_first_samples",
    )
    fractions = []
    for channel, first_sample in zip(candidate, first_samples):
        total = float(np.dot(channel, channel))
        prearrival = float(np.dot(channel[:first_sample], channel[:first_sample]))
        fractions.append(prearrival / max(total, np.finfo(np.float64).tiny))
    return {
        "distance": float(np.mean(fractions)),
        "prearrival_energy_fraction_by_channel": fractions,
    }


def decay_growth_penalty(
    synthetic: Any,
    sample_rate: int,
    direct_samples: Sequence[int],
    *,
    start_ms: float = 80.0,
    window_ms: float = 20.0,
    hop_ms: float = 10.0,
    allowed_growth_db: float = 1.0,
) -> dict[str, Any]:
    """Penalize sustained late-window energy growth beyond a small tolerance."""

    candidate = _channels_first(synthetic, "synthetic RIR")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    direct = _direct_samples(candidate, direct_samples, "direct_samples")
    scalar_configuration = (
        start_ms,
        window_ms,
        hop_ms,
        allowed_growth_db,
    )
    if any(not math.isfinite(value) for value in scalar_configuration):
        raise ValueError("decay regularization configuration must be finite")
    if start_ms < 0.0 or window_ms <= 0.0 or hop_ms <= 0.0:
        raise ValueError("decay windows must be positive after a non-negative start")
    if allowed_growth_db < 0.0:
        raise ValueError("allowed_growth_db must be non-negative")
    window = max(2, int(round(window_ms * 1e-3 * sample_rate)))
    hop = max(1, int(round(hop_ms * 1e-3 * sample_rate)))
    offset = int(round(start_ms * 1e-3 * sample_rate))
    channel_penalties = []
    for channel, direct_sample_index in zip(candidate, direct):
        start = direct_sample_index + offset
        energies = np.asarray(
            [
                np.mean(np.square(channel[index : index + window]))
                for index in range(start, channel.size - window + 1, hop)
            ],
            dtype=np.float64,
        )
        if energies.size < 2:
            channel_penalties.append(0.0)
            continue
        energy_db = 10.0 * np.log10(np.maximum(energies, np.finfo(np.float64).tiny))
        growth = np.maximum(0.0, np.diff(energy_db) - allowed_growth_db)
        channel_penalties.append(float(np.mean(np.square(growth / 10.0))))
    return {
        "distance": float(np.mean(channel_penalties)),
        "channel_penalties": channel_penalties,
        "allowed_growth_db_per_hop": float(allowed_growth_db),
    }


def analyze_rir_calibration_loss(
    measured: Any,
    synthetic: Any,
    sample_rate: int,
    *,
    measured_direct_samples: Sequence[int] | None = None,
    synthetic_direct_samples: Sequence[int] | None = None,
    physical_first_samples: Sequence[int] | None = None,
    weights: CalibrationLossWeights | None = None,
    fft_sizes: Sequence[int] = (256, 512, 1024),
    octave_centers_hz: Sequence[float] = (
        125.0,
        250.0,
        500.0,
        1000.0,
        2000.0,
        4000.0,
    ),
) -> CalibrationLossReport:
    """Evaluate every M5.1 reference loss without hiding unevaluable terms."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    reference, candidate = _matched_signals(measured, synthetic)
    measured_direct = _direct_samples(
        reference,
        measured_direct_samples,
        "measured_direct_samples",
    )
    synthetic_direct = _direct_samples(
        candidate,
        synthetic_direct_samples,
        "synthetic_direct_samples",
    )
    physical_first = (
        measured_direct
        if physical_first_samples is None
        else tuple(int(value) for value in physical_first_samples)
    )
    active_weights = weights or CalibrationLossWeights()
    stft_report = multiresolution_stft_distance(
        reference,
        candidate,
        fft_sizes=fft_sizes,
    )
    decay_report = energy_decay_curve_distance(
        reference,
        candidate,
        measured_direct,
        synthetic_direct,
    )
    arrival_report = arrival_timing_distance(
        measured_direct,
        synthetic_direct,
        sample_rate,
    )
    octave_report = octave_acoustic_distance(
        reference,
        candidate,
        sample_rate,
        measured_direct,
        synthetic_direct,
        centers_hz=octave_centers_hz,
    )
    spatial_report = spatial_coherence_distance(
        reference,
        candidate,
        sample_rate,
        measured_direct,
        synthetic_direct,
    )
    causality_report = causality_penalty(candidate, physical_first)
    regularization_report = decay_growth_penalty(
        candidate,
        sample_rate,
        synthetic_direct,
    )
    terms = {
        "multiresolution_stft": float(stft_report["distance"]),
        "energy_decay": float(decay_report["distance"]),
        "arrival_timing": float(arrival_report["distance"]),
        "octave_acoustics": float(octave_report["distance"]),
        "spatial_coherence": float(spatial_report["distance"]),
        "causality": float(causality_report["distance"]),
        "decay_regularization": float(regularization_report["distance"]),
    }
    total = float(sum(terms[name] * active_weights.to_dict()[name] for name in terms))
    diagnostics = {
        "sample_rate": int(sample_rate),
        "channel_count": int(reference.shape[0]),
        "sample_count": int(reference.shape[1]),
        "measured_direct_samples": list(measured_direct),
        "synthetic_direct_samples": list(synthetic_direct),
        "physical_first_samples": list(physical_first),
        "multiresolution_stft": stft_report,
        "energy_decay": decay_report,
        "arrival_timing": arrival_report,
        "octave_acoustics": octave_report,
        "spatial_coherence": spatial_report,
        "causality": causality_report,
        "decay_regularization": regularization_report,
    }
    return CalibrationLossReport(
        total=total,
        terms=terms,
        diagnostics=diagnostics,
        weights=active_weights,
    )


__all__ = [
    "RIR_CALIBRATION_LOSS_POLICY",
    "CalibrationLossReport",
    "CalibrationLossWeights",
    "analyze_rir_calibration_loss",
    "arrival_timing_distance",
    "causality_penalty",
    "decay_growth_penalty",
    "energy_decay_curve_distance",
    "multiresolution_stft_distance",
    "octave_acoustic_distance",
    "spatial_coherence_distance",
]
