"""Deterministic passive multiband feedback delay network for M4 late fields.

This module implements the isolated M4.3 late-reverberation core. It does not
splice the generated tail into the production hybrid RIR; M4.4 owns that
direct/early/late coupling decision.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import numpy as np
from scipy.signal import butter, sosfilt, sosfreqz

from puresound.audio.rir.metrics import valid_octave_centers


MULTIBAND_FDN_POLICY = "puresound.multiband_fdn.v2"


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value == 2:
        return True
    if value % 2 == 0:
        return False
    limit = int(math.isqrt(value))
    for divisor in range(3, limit + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _prime_pool(low: int, high: int) -> list[int]:
    return [value for value in range(max(2, low), high + 1) if _is_prime(value)]


def select_prime_delay_lengths(
    sample_rate: int,
    delay_line_count: int,
    minimum_delay_ms: float,
    maximum_delay_ms: float,
    *,
    seed: int = 0,
) -> tuple[int, ...]:
    """Select deterministic distinct prime delays near geometric targets."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if delay_line_count < 2:
        raise ValueError("delay_line_count must be at least two")
    if not np.isfinite(minimum_delay_ms) or minimum_delay_ms <= 0.0:
        raise ValueError("minimum_delay_ms must be finite and positive")
    if not np.isfinite(maximum_delay_ms) or maximum_delay_ms <= minimum_delay_ms:
        raise ValueError("maximum_delay_ms must exceed minimum_delay_ms")

    low = max(2, int(round(minimum_delay_ms * 1e-3 * sample_rate)))
    requested_high = max(low + 1, int(round(maximum_delay_ms * 1e-3 * sample_rate)))
    high = requested_high
    primes = _prime_pool(low, high)
    while len(primes) < delay_line_count:
        high = max(high + 16, int(math.ceil(1.25 * high)))
        primes = _prime_pool(low, high)
        if high > 10 * sample_rate:
            raise RuntimeError("could not find enough prime delay lengths")

    rng = np.random.default_rng(int(seed))
    targets = np.geomspace(float(low), float(requested_high), delay_line_count)
    targets *= rng.uniform(0.98, 1.02, delay_line_count)
    available = set(primes)
    selected: list[int] = []
    for target in targets:
        candidate = min(available, key=lambda value: (abs(value - target), value))
        selected.append(int(candidate))
        available.remove(candidate)
    return tuple(sorted(selected))


def randomized_hadamard_matrix(size: int, *, seed: int = 0) -> np.ndarray:
    """Return a deterministic dense orthogonal signed/permuted Hadamard matrix."""

    if size < 2 or size & (size - 1):
        raise ValueError("Hadamard size must be a power of two and at least two")
    matrix = np.asarray([[1.0]], dtype=np.float64)
    while matrix.shape[0] < size:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    matrix /= math.sqrt(size)

    rng = np.random.default_rng(int(seed))
    row_order = rng.permutation(size)
    column_order = rng.permutation(size)
    row_sign = rng.choice((-1.0, 1.0), size=size)
    column_sign = rng.choice((-1.0, 1.0), size=size)
    matrix = matrix[row_order][:, column_order]
    matrix *= row_sign[:, None]
    matrix *= column_sign[None, :]
    return np.asarray(matrix, dtype=np.float64)


def delay_proportional_loop_gains(
    delay_lengths_samples: Any,
    sample_rate: int,
    target_rt60_s: float,
) -> np.ndarray:
    """Return per-delay pressure gains for a target 60 dB decay time."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if not np.isfinite(target_rt60_s) or target_rt60_s <= 0.0:
        raise ValueError("target_rt60_s must be finite and positive")
    delays = np.asarray(delay_lengths_samples, dtype=np.float64).reshape(-1)
    if delays.size == 0 or not np.all(np.isfinite(delays)) or np.any(delays <= 0.0):
        raise ValueError("delay lengths must be finite and positive")
    return np.power(10.0, -3.0 * delays / (sample_rate * target_rt60_s))


@dataclass(frozen=True)
class MultibandFDNDesign:
    """Complete deterministic FDN state-independent design."""

    sample_rate: int
    centers_hz: tuple[float, ...]
    target_rt60_s: tuple[float, ...]
    target_mixing_time_s: float
    delay_lengths_samples: tuple[int, ...]
    feedback_matrix: np.ndarray
    input_vector: np.ndarray
    output_matrix: np.ndarray
    band_weights: np.ndarray
    loop_gains: np.ndarray
    seed: int

    @property
    def delay_line_count(self) -> int:
        return len(self.delay_lengths_samples)

    @property
    def band_count(self) -> int:
        return len(self.centers_hz)

    def to_dict(self, *, include_coefficients: bool = False) -> dict[str, Any]:
        identity = np.eye(self.delay_line_count, dtype=np.float64)
        orthogonality_error = float(
            np.linalg.norm(
                self.feedback_matrix.T @ self.feedback_matrix - identity,
                ord=2,
            )
        )
        bands = {}
        for index, center_hz in enumerate(self.centers_hz):
            gains = self.loop_gains[index]
            bands[f"{center_hz:g}"] = {
                "center_hz": float(center_hz),
                "target_rt60_s": float(self.target_rt60_s[index]),
                "minimum_loop_gain": float(np.min(gains)),
                "maximum_loop_gain": float(np.max(gains)),
                "feedback_operator_2norm": float(np.max(gains)),
            }
        result: dict[str, Any] = {
            "policy": MULTIBAND_FDN_POLICY,
            "sample_rate": int(self.sample_rate),
            "seed": int(self.seed),
            "delay_line_count": int(self.delay_line_count),
            "delay_lengths_samples": list(self.delay_lengths_samples),
            "delay_lengths_ms": [
                float(1000.0 * delay / self.sample_rate)
                for delay in self.delay_lengths_samples
            ],
            "delays_are_distinct_primes": bool(
                len(set(self.delay_lengths_samples)) == self.delay_line_count
                and all(_is_prime(delay) for delay in self.delay_lengths_samples)
            ),
            "feedback_matrix": "signed_permuted_normalized_hadamard",
            "feedback_orthogonality_error_2norm": orthogonality_error,
            "target_mixing_time_s": float(self.target_mixing_time_s),
            "band_weight_energy_sum": float(np.sum(np.square(self.band_weights))),
            "bands": bands,
        }
        if include_coefficients:
            result["coefficients"] = {
                "feedback_matrix": self.feedback_matrix.tolist(),
                "input_vector": self.input_vector.tolist(),
                "output_matrix": self.output_matrix.tolist(),
                "band_weights": self.band_weights.tolist(),
                "loop_gains": self.loop_gains.tolist(),
            }
        return result


@dataclass(frozen=True)
class MultibandFDNRender:
    """Rendered full-band tail and its explicitly separated octave bands."""

    rir: np.ndarray
    band_rirs: Mapping[float, np.ndarray]
    raw_band_rirs: Mapping[float, np.ndarray]
    design: MultibandFDNDesign
    filterbank_metadata: Mapping[str, Any]


def _fdn_partition_sos(
    sample_rate: int,
    centers_hz: tuple[float, ...],
    band_index: int,
    filter_order: int,
) -> np.ndarray | None:
    """Return one causal endpoint-complete Butterworth partition filter."""

    if len(centers_hz) == 1:
        return None
    boundaries = tuple(
        math.sqrt(left * right)
        for left, right in zip(centers_hz[:-1], centers_hz[1:])
    )
    sections = []
    # A cascaded binary split is exactly power-complementary for Butterworth
    # low/high pairs: L0, H0*L1, H0*H1*L2, ..., H0*...*Hn.  This avoids both
    # the gaps of finite octave filters and their accumulated overlap ripple.
    for boundary in boundaries[:band_index]:
        sections.append(
            butter(
                filter_order,
                boundary,
                btype="highpass",
                fs=float(sample_rate),
                output="sos",
            )
        )
    if band_index < len(boundaries):
        sections.append(
            butter(
                filter_order,
                boundaries[band_index],
                btype="lowpass",
                fs=float(sample_rate),
                output="sos",
            )
        )
    return np.vstack(sections)


def fdn_filterbank_power_response(
    sample_rate: int,
    centers_hz: Any,
    *,
    filter_order: int = 4,
    frequency_count: int = 4096,
) -> tuple[np.ndarray, np.ndarray]:
    """Return summed squared magnitude for the endpoint-complete partitions."""

    centers = tuple(sorted(float(value) for value in centers_hz))
    if not centers:
        raise ValueError("at least one FDN center is required")
    if frequency_count < 2:
        raise ValueError("frequency_count must be at least two")
    frequencies = np.linspace(0.0, 0.5 * sample_rate, frequency_count)
    power = np.zeros_like(frequencies)
    for band_index in range(len(centers)):
        sos = _fdn_partition_sos(
            int(sample_rate),
            centers,
            band_index,
            int(filter_order),
        )
        if sos is None:
            power += 1.0
        else:
            _angular, response = sosfreqz(
                sos,
                worN=frequencies,
                fs=float(sample_rate),
            )
            power += np.square(np.abs(response))
    return frequencies, power


def design_multiband_fdn(
    sample_rate: int,
    target_rt60_s_by_hz: Mapping[float, float],
    *,
    target_mixing_time_s: float = 0.024,
    delay_line_count: int = 16,
    delay_range_ms: Optional[tuple[float, float]] = None,
    seed: int = 0,
) -> MultibandFDNDesign:
    """Design a deterministic internally contractive multiband FDN."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if delay_line_count < 2 or delay_line_count & (delay_line_count - 1):
        raise ValueError("delay_line_count must be a power of two and at least two")
    if not np.isfinite(target_mixing_time_s) or target_mixing_time_s <= 0.0:
        raise ValueError("target_mixing_time_s must be finite and positive")
    if not target_rt60_s_by_hz:
        raise ValueError("at least one target RT60 band is required")

    sorted_targets = sorted(
        (float(center), float(rt60_s))
        for center, rt60_s in target_rt60_s_by_hz.items()
    )
    centers_hz = tuple(center for center, _ in sorted_targets)
    target_rt60_s = tuple(rt60_s for _, rt60_s in sorted_targets)
    if len(set(centers_hz)) != len(centers_hz):
        raise ValueError("target octave centers must be unique")
    if any(not np.isfinite(value) or value <= 0.0 for value in target_rt60_s):
        raise ValueError("target RT60 values must be finite and positive")
    valid_centers = set(valid_octave_centers(sample_rate, centers_hz))
    if any(center not in valid_centers for center in centers_hz):
        raise ValueError("every target octave band must lie below Nyquist")

    if delay_range_ms is None:
        mixing_ms = 1000.0 * float(target_mixing_time_s)
        delay_range_ms = (max(1.0, 0.125 * mixing_ms), 0.75 * mixing_ms)
    minimum_delay_ms, maximum_delay_ms = map(float, delay_range_ms)
    delays = select_prime_delay_lengths(
        sample_rate,
        delay_line_count,
        minimum_delay_ms,
        maximum_delay_ms,
        seed=seed,
    )
    feedback_matrix = randomized_hadamard_matrix(delay_line_count, seed=seed)
    rng = np.random.default_rng(int(seed) + 1)
    input_vector = rng.choice((-1.0, 1.0), size=delay_line_count) / math.sqrt(
        delay_line_count
    )
    output_matrix = rng.choice(
        (-1.0, 1.0),
        size=(len(centers_hz), delay_line_count),
    ) / math.sqrt(delay_line_count)
    band_weights = np.full(
        len(centers_hz),
        1.0 / math.sqrt(len(centers_hz)),
        dtype=np.float64,
    )
    loop_gains = np.vstack(
        [
            delay_proportional_loop_gains(delays, sample_rate, rt60_s)
            for rt60_s in target_rt60_s
        ]
    )
    design = MultibandFDNDesign(
        sample_rate=int(sample_rate),
        centers_hz=centers_hz,
        target_rt60_s=target_rt60_s,
        target_mixing_time_s=float(target_mixing_time_s),
        delay_lengths_samples=delays,
        feedback_matrix=feedback_matrix,
        input_vector=np.asarray(input_vector, dtype=np.float64),
        output_matrix=np.asarray(output_matrix, dtype=np.float64),
        band_weights=band_weights,
        loop_gains=np.asarray(loop_gains, dtype=np.float64),
        seed=int(seed),
    )
    _validate_design(design)
    return design


def _validate_design(design: MultibandFDNDesign) -> None:
    delay_count = design.delay_line_count
    band_count = design.band_count
    if design.feedback_matrix.shape != (delay_count, delay_count):
        raise ValueError("feedback matrix shape does not match delay count")
    if design.input_vector.shape != (delay_count,):
        raise ValueError("input vector shape does not match delay count")
    if design.output_matrix.shape != (band_count, delay_count):
        raise ValueError("output matrix shape does not match bands and delays")
    if design.loop_gains.shape != (band_count, delay_count):
        raise ValueError("loop gain shape does not match bands and delays")
    if design.band_weights.shape != (band_count,):
        raise ValueError("band weight shape does not match band count")
    arrays = (
        design.feedback_matrix,
        design.input_vector,
        design.output_matrix,
        design.band_weights,
        design.loop_gains,
    )
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("FDN coefficients must be finite")
    identity = np.eye(delay_count, dtype=np.float64)
    orthogonality_error = np.linalg.norm(
        design.feedback_matrix.T @ design.feedback_matrix - identity,
        ord=2,
    )
    if orthogonality_error > 1e-12:
        raise ValueError("feedback matrix must be orthogonal")
    if np.any(design.loop_gains <= 0.0) or np.any(design.loop_gains >= 1.0):
        raise ValueError("every loop gain must lie strictly between zero and one")
    if not math.isclose(
        float(np.sum(np.square(design.band_weights))),
        1.0,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("band weights must have unit squared norm")


def render_multiband_fdn(
    design: MultibandFDNDesign,
    excitation: Any,
    *,
    output_gain: float = 1.0,
    filter_order: int = 4,
) -> MultibandFDNRender:
    """Render a mono excitation through every independently damped FDN band."""

    _validate_design(design)
    if not np.isfinite(output_gain):
        raise ValueError("output_gain must be finite")
    if filter_order < 1:
        raise ValueError("filter_order must be positive")
    signal = np.asarray(excitation, dtype=np.float64).squeeze()
    if signal.ndim == 0:
        signal = signal.reshape(1)
    if signal.ndim != 1:
        raise ValueError("FDN excitation must be one-dimensional")
    if not np.all(np.isfinite(signal)):
        raise ValueError("FDN excitation must be finite")

    sample_count = int(signal.size)
    band_count = design.band_count
    delay_count = design.delay_line_count
    buffers = [
        np.zeros((band_count, delay), dtype=np.float64)
        for delay in design.delay_lengths_samples
    ]
    pointers = np.zeros(delay_count, dtype=np.int64)
    raw = np.zeros((band_count, sample_count), dtype=np.float64)

    # No write can return to an output sooner than the shortest delay.  Rendering
    # blocks no longer than that delay is therefore sample-equivalent to the
    # scalar recurrence while moving the band/delay matrix work into NumPy.
    block_size = min(design.delay_lengths_samples)
    for block_start in range(0, sample_count, block_size):
        block_stop = min(sample_count, block_start + block_size)
        block_length = block_stop - block_start
        offsets = np.arange(block_length, dtype=np.int64)
        delayed = np.empty(
            (band_count, delay_count, block_length),
            dtype=np.float64,
        )
        buffer_indexes: list[np.ndarray] = []
        for delay_index, buffer in enumerate(buffers):
            indexes = (pointers[delay_index] + offsets) % buffer.shape[1]
            buffer_indexes.append(indexes)
            delayed[:, delay_index, :] = buffer[:, indexes]

        raw[:, block_start:block_stop] = np.einsum(
            "bd,bdt->bt",
            design.output_matrix,
            delayed,
            optimize=False,
        )
        attenuated = delayed * design.loop_gains[:, :, None]
        writes = np.einsum(
            "bdt,ed->bet",
            attenuated,
            design.feedback_matrix,
            optimize=False,
        )
        writes += (
            design.band_weights[:, None, None]
            * design.input_vector[None, :, None]
            * signal[None, None, block_start:block_stop]
        )
        for delay_index, buffer in enumerate(buffers):
            buffer[:, buffer_indexes[delay_index]] = writes[:, delay_index, :]
            pointers[delay_index] = (
                pointers[delay_index] + block_length
            ) % buffer.shape[1]

    raw_band_rirs: dict[float, np.ndarray] = {}
    band_rirs: dict[float, np.ndarray] = {}
    for band_index, center_hz in enumerate(design.centers_hz):
        raw_band = float(output_gain) * raw[band_index]
        sos = _fdn_partition_sos(
            design.sample_rate,
            design.centers_hz,
            band_index,
            filter_order,
        )
        band = raw_band if sos is None else sosfilt(sos, raw_band)
        raw_band_rirs[center_hz] = np.asarray(raw_band, dtype=np.float64)
        band_rirs[center_hz] = np.asarray(band, dtype=np.float64)
    combined = np.sum(np.vstack(list(band_rirs.values())), axis=0)
    frequencies, power = fdn_filterbank_power_response(
        design.sample_rate,
        design.centers_hz,
        filter_order=filter_order,
    )
    analysis = (frequencies >= 20.0) & (
        frequencies <= 0.99 * 0.5 * design.sample_rate
    )
    power_db = 10.0 * np.log10(np.maximum(power[analysis], 1e-20))
    return MultibandFDNRender(
        rir=np.asarray(combined, dtype=np.float64),
        band_rirs=band_rirs,
        raw_band_rirs=raw_band_rirs,
        design=design,
        filterbank_metadata={
            "policy": "causal_endpoint_complete_butterworth_power_partition",
            "filter_order": int(filter_order),
            "coverage_hz": [0.0, 0.5 * float(design.sample_rate)],
            "summed_power_min_db": float(np.min(power_db)),
            "summed_power_max_db": float(np.max(power_db)),
            "summed_power_ripple_db": float(np.max(power_db) - np.min(power_db)),
        },
    )


def render_multiband_fdn_impulse(
    design: MultibandFDNDesign,
    duration_s: float,
    *,
    output_gain: float = 1.0,
    filter_order: int = 4,
) -> MultibandFDNRender:
    """Render a unit impulse through a multiband FDN for ``duration_s``."""

    if not np.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("duration_s must be finite and positive")
    sample_count = max(1, int(round(duration_s * design.sample_rate)))
    impulse = np.zeros(sample_count, dtype=np.float64)
    impulse[0] = 1.0
    return render_multiband_fdn(
        design,
        impulse,
        output_gain=output_gain,
        filter_order=filter_order,
    )


def analyze_fdn_coloration(
    rir: Any,
    sample_rate: int,
    centers_hz: Any,
) -> dict[str, Any]:
    """Report within-octave spectral flatness and robust peak-to-median ripple."""

    signal = np.asarray(rir, dtype=np.float64).squeeze()
    if signal.ndim != 1:
        raise ValueError("coloration analysis expects one-dimensional audio")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    if signal.size < 2 or not np.any(signal):
        return {"policy": "puresound.fdn_coloration.v1", "bands": {}}
    fft_size = 1 << int(math.ceil(math.log2(max(2, signal.size))))
    spectrum_power = np.square(np.abs(np.fft.rfft(signal, n=fft_size)))
    frequencies = np.fft.rfftfreq(fft_size, d=1.0 / sample_rate)
    bands: dict[str, Any] = {}
    for center_hz in valid_octave_centers(sample_rate, centers_hz):
        low = center_hz / math.sqrt(2.0)
        high = center_hz * math.sqrt(2.0)
        values = spectrum_power[(frequencies >= low) & (frequencies < high)]
        if values.size < 4:
            continue
        floor = max(float(np.max(values)) * 1e-12, np.finfo(np.float64).tiny)
        values = np.maximum(values, floor)
        geometric_mean = float(np.exp(np.mean(np.log(values))))
        arithmetic_mean = float(np.mean(values))
        median = float(np.median(values))
        percentile_95 = float(np.percentile(values, 95.0))
        bands[f"{center_hz:g}"] = {
            "center_hz": float(center_hz),
            "frequency_bin_count": int(values.size),
            "spectral_flatness": float(geometric_mean / arithmetic_mean),
            "p95_to_median_db": float(
                10.0 * math.log10(percentile_95 / max(median, floor))
            ),
        }
    return {"policy": "puresound.fdn_coloration.v1", "bands": bands}


__all__ = [
    "MULTIBAND_FDN_POLICY",
    "MultibandFDNDesign",
    "MultibandFDNRender",
    "analyze_fdn_coloration",
    "delay_proportional_loop_gains",
    "design_multiband_fdn",
    "fdn_filterbank_power_response",
    "randomized_hadamard_matrix",
    "render_multiband_fdn",
    "render_multiband_fdn_impulse",
    "select_prime_delay_lengths",
]
