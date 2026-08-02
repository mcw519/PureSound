"""Physically constrained synthetic-recovery baseline for M5.2.

The model in this module is deliberately a lightweight approximate renderer.
It is used to test parameter identifiability and optimization plumbing before
real measurements or a learned residual are introduced. It does not replace
the M4 renderer and it is not a measured-room fit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from puresound.audio.rir.calibration.loss import (
    CalibrationLossWeights,
    analyze_rir_calibration_loss,
)
from puresound.audio.rir.metrics import octave_band_rir, valid_octave_centers


RIR_SYNTHETIC_RECOVERY_POLICY = "puresound.rir_synthetic_recovery.v1"
RIR_ROBUST_RECOVERY_OBJECTIVE_POLICY = "puresound.rir_robust_recovery_objective.v1"


def _finite_mapping(name: str, values: Mapping[float, float]) -> dict[float, float]:
    result = {float(key): float(value) for key, value in values.items()}
    if not result or any(
        not math.isfinite(key) or not math.isfinite(value)
        for key, value in result.items()
    ):
        raise ValueError(f"{name} must contain finite values")
    return dict(sorted(result.items()))


@dataclass(frozen=True)
class SyntheticRecoveryParameters:
    """Shared room parameters exposed by the M5.2 approximate renderer."""

    mixing_time_s: float
    early_reflection_gain_db: float
    rt60_s_by_hz: Mapping[float, float]
    late_gain_db_by_hz: Mapping[float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.mixing_time_s) or self.mixing_time_s <= 0.0:
            raise ValueError("mixing_time_s must be finite and positive")
        if not math.isfinite(self.early_reflection_gain_db):
            raise ValueError("early_reflection_gain_db must be finite")
        rt60 = _finite_mapping("rt60_s_by_hz", self.rt60_s_by_hz)
        late_gain = _finite_mapping("late_gain_db_by_hz", self.late_gain_db_by_hz)
        if set(rt60) != set(late_gain):
            raise ValueError("RT60 and late-gain octave centers must match")
        if any(value <= 0.0 for value in rt60.values()):
            raise ValueError("every RT60 must be positive")
        object.__setattr__(self, "rt60_s_by_hz", rt60)
        object.__setattr__(self, "late_gain_db_by_hz", late_gain)

    @property
    def centers_hz(self) -> tuple[float, ...]:
        return tuple(self.rt60_s_by_hz)

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.mixing_time_s,
                self.early_reflection_gain_db,
                *(self.rt60_s_by_hz[center] for center in self.centers_hz),
                *(self.late_gain_db_by_hz[center] for center in self.centers_hz),
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_vector(
        cls,
        centers_hz: Sequence[float],
        values: Any,
    ) -> "SyntheticRecoveryParameters":
        centers = tuple(float(center) for center in centers_hz)
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        expected = 2 + 2 * len(centers)
        if vector.size != expected or not np.all(np.isfinite(vector)):
            raise ValueError(f"parameter vector must contain {expected} finite values")
        return cls(
            mixing_time_s=float(vector[0]),
            early_reflection_gain_db=float(vector[1]),
            rt60_s_by_hz={
                center: float(vector[2 + index]) for index, center in enumerate(centers)
            },
            late_gain_db_by_hz={
                center: float(vector[2 + len(centers) + index])
                for index, center in enumerate(centers)
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mixing_time_s": float(self.mixing_time_s),
            "early_reflection_gain_db": float(self.early_reflection_gain_db),
            "rt60_s_by_hz": {
                f"{center:g}": float(value)
                for center, value in self.rt60_s_by_hz.items()
            },
            "late_gain_db_by_hz": {
                f"{center:g}": float(value)
                for center, value in self.late_gain_db_by_hz.items()
            },
        }


@dataclass(frozen=True)
class SyntheticRecoveryBounds:
    """Box constraints that keep every fitted solution physically meaningful."""

    mixing_time_s: tuple[float, float] = (0.012, 0.080)
    early_reflection_gain_db: tuple[float, float] = (-12.0, 12.0)
    rt60_s: tuple[float, float] = (0.20, 2.00)
    late_gain_db: tuple[float, float] = (-18.0, 12.0)

    def vectors(self, band_count: int) -> tuple[np.ndarray, np.ndarray]:
        pairs = (
            self.mixing_time_s,
            self.early_reflection_gain_db,
            self.rt60_s,
            self.late_gain_db,
        )
        if band_count < 1 or any(
            len(pair) != 2
            or not all(math.isfinite(value) for value in pair)
            or pair[0] >= pair[1]
            for pair in pairs
        ):
            raise ValueError("synthetic recovery bounds must be finite and increasing")
        lower = np.asarray(
            [
                self.mixing_time_s[0],
                self.early_reflection_gain_db[0],
                *([self.rt60_s[0]] * band_count),
                *([self.late_gain_db[0]] * band_count),
            ],
            dtype=np.float64,
        )
        upper = np.asarray(
            [
                self.mixing_time_s[1],
                self.early_reflection_gain_db[1],
                *([self.rt60_s[1]] * band_count),
                *([self.late_gain_db[1]] * band_count),
            ],
            dtype=np.float64,
        )
        return lower, upper


@dataclass(frozen=True)
class SyntheticRecoveryObjectiveConfig:
    """Select the exact-recovery objective or the noise-aware M4 proxy."""

    mode: str = "waveform_v1"
    waveform_weight: float = 0.05
    early_waveform_weight: float = 0.50
    broadband_decay_weight: float = 1.00
    octave_decay_weight: float = 1.00
    energy_window_ms: float = 8.0
    noise_margin_db: float = 6.0

    def __post_init__(self) -> None:
        if self.mode not in {"waveform_v1", "m4_multiterm_v2"}:
            raise ValueError("unsupported synthetic recovery objective mode")
        scalars = (
            self.waveform_weight,
            self.early_waveform_weight,
            self.broadband_decay_weight,
            self.octave_decay_weight,
            self.energy_window_ms,
            self.noise_margin_db,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in scalars):
            raise ValueError("objective weights and configuration must be non-negative")
        if self.energy_window_ms <= 0.0:
            raise ValueError("energy_window_ms must be positive")
        if self.mode == "m4_multiterm_v2" and (
            self.broadband_decay_weight == 0.0 or self.octave_decay_weight == 0.0
        ):
            raise ValueError("M4 multi-term objective requires decay terms")

    @property
    def policy(self) -> str:
        return (
            RIR_SYNTHETIC_RECOVERY_POLICY
            if self.mode == "waveform_v1"
            else RIR_ROBUST_RECOVERY_OBJECTIVE_POLICY
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "mode": self.mode,
            "implementation": "scipy_finite_difference_not_autograd",
            "waveform_weight": float(self.waveform_weight),
            "early_waveform_weight": float(self.early_waveform_weight),
            "broadband_decay_weight": float(self.broadband_decay_weight),
            "octave_decay_weight": float(self.octave_decay_weight),
            "energy_window_ms": float(self.energy_window_ms),
            "noise_margin_db": float(self.noise_margin_db),
        }


@dataclass(frozen=True)
class SyntheticMeasurementPerturbation:
    """Controlled acquisition error and renderer mismatch for M5.2b."""

    snr_db: float
    gain_error_db: float = 0.0
    latency_offset_samples: int = 0
    early_model_mismatch_fraction: float = 0.0
    late_model_mismatch_fraction: float = 0.0
    seed: int = 0

    def __post_init__(self) -> None:
        scalars = (
            self.snr_db,
            self.gain_error_db,
            self.early_model_mismatch_fraction,
            self.late_model_mismatch_fraction,
        )
        if any(not math.isfinite(value) for value in scalars):
            raise ValueError("measurement perturbation values must be finite")
        if self.snr_db <= 0.0:
            raise ValueError("snr_db must be positive")
        if not 0.0 <= self.early_model_mismatch_fraction <= 1.0:
            raise ValueError("early mismatch fraction must lie in [0, 1]")
        if not 0.0 <= self.late_model_mismatch_fraction <= 1.0:
            raise ValueError("late mismatch fraction must lie in [0, 1]")
        object.__setattr__(
            self,
            "latency_offset_samples",
            int(self.latency_offset_samples),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "snr_db": float(self.snr_db),
            "gain_error_db": float(self.gain_error_db),
            "latency_offset_samples": int(self.latency_offset_samples),
            "early_model_mismatch_fraction": float(self.early_model_mismatch_fraction),
            "late_model_mismatch_fraction": float(self.late_model_mismatch_fraction),
            "seed": int(self.seed),
        }


@dataclass(frozen=True)
class PerturbedSyntheticMeasurement:
    """Raw perturbed target plus the target after known nuisance correction."""

    raw_rir: np.ndarray
    corrected_rir: np.ndarray
    clean_rir: np.ndarray
    mismatch_component: np.ndarray
    noise_component_raw: np.ndarray
    perturbation: SyntheticMeasurementPerturbation
    metadata: Mapping[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "perturbation": self.perturbation.to_dict(),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class SyntheticRecoveryObservation:
    """Position-specific known basis used to identify shared room parameters."""

    observation_id: str
    sample_rate: int
    direct_sample: int
    direct_component: np.ndarray
    early_component: np.ndarray
    late_basis_by_hz: Mapping[float, np.ndarray]

    def __post_init__(self) -> None:
        direct = np.asarray(self.direct_component, dtype=np.float64).reshape(-1)
        early = np.asarray(self.early_component, dtype=np.float64).reshape(-1)
        if not self.observation_id or self.sample_rate <= 0:
            raise ValueError("observation id and positive sample rate are required")
        if direct.size < 32 or early.shape != direct.shape:
            raise ValueError("direct and early components must share a useful length")
        if not 0 <= int(self.direct_sample) < direct.size:
            raise ValueError("direct sample lies outside the observation")
        if not np.all(np.isfinite(direct)) or not np.all(np.isfinite(early)):
            raise ValueError("observation components must be finite")
        late = {
            float(center): np.asarray(value, dtype=np.float64).reshape(-1)
            for center, value in self.late_basis_by_hz.items()
        }
        if not late or any(
            value.shape != direct.shape or not np.all(np.isfinite(value))
            for value in late.values()
        ):
            raise ValueError("every late basis must match the observation length")
        if set(late) != set(valid_octave_centers(self.sample_rate, late)):
            raise ValueError("late basis octave centers must lie below Nyquist")
        first = int(self.direct_sample)
        if (
            np.any(direct[:first] != 0.0)
            or np.any(early[:first] != 0.0)
            or any(np.any(value[:first] != 0.0) for value in late.values())
        ):
            raise ValueError("synthetic recovery bases must be causal")
        object.__setattr__(self, "direct_sample", first)
        object.__setattr__(self, "direct_component", direct)
        object.__setattr__(self, "early_component", early)
        object.__setattr__(self, "late_basis_by_hz", dict(sorted(late.items())))

    @property
    def centers_hz(self) -> tuple[float, ...]:
        return tuple(self.late_basis_by_hz)

    @property
    def sample_count(self) -> int:
        return int(self.direct_component.size)


@dataclass(frozen=True)
class SyntheticRecoveryFit:
    """Fitted parameters plus optimization and local-identifiability evidence."""

    parameters: SyntheticRecoveryParameters
    initial_parameters: SyntheticRecoveryParameters
    success: bool
    message: str
    evaluations: int
    initial_cost: float
    final_cost: float
    scaled_jacobian_singular_values: tuple[float, ...]
    scaled_jacobian_condition_number: float
    locally_full_rank: bool
    objective: SyntheticRecoveryObjectiveConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": RIR_SYNTHETIC_RECOVERY_POLICY,
            "parameters": self.parameters.to_dict(),
            "initial_parameters": self.initial_parameters.to_dict(),
            "success": bool(self.success),
            "message": self.message,
            "evaluations": int(self.evaluations),
            "initial_cost": float(self.initial_cost),
            "final_cost": float(self.final_cost),
            "objective": self.objective.to_dict(),
            "relative_cost_reduction": float(
                1.0 - self.final_cost / max(self.initial_cost, np.finfo(float).tiny)
            ),
            "local_identifiability": {
                "scaled_jacobian_singular_values": list(
                    self.scaled_jacobian_singular_values
                ),
                "condition_number": float(self.scaled_jacobian_condition_number),
                "full_column_rank": bool(self.locally_full_rank),
                "warning": "Local numerical evidence is not global identifiability proof.",
            },
        }


def build_synthetic_recovery_observation(
    observation_id: str,
    sample_rate: int,
    sample_count: int,
    distance_m: float,
    *,
    centers_hz: Sequence[float] = (500.0, 1000.0, 2000.0, 4000.0),
    sound_speed_m_s: float = 343.0,
    seed: int = 0,
) -> SyntheticRecoveryObservation:
    """Build one deterministic causal multi-position recovery fixture."""

    if sample_rate <= 0 or sample_count < 512:
        raise ValueError("sample rate must be positive and sample count at least 512")
    if not math.isfinite(distance_m) or distance_m <= 0.0:
        raise ValueError("distance_m must be finite and positive")
    if not math.isfinite(sound_speed_m_s) or sound_speed_m_s <= 0.0:
        raise ValueError("sound speed must be finite and positive")
    centers = valid_octave_centers(sample_rate, centers_hz)
    if len(centers) != len(tuple(centers_hz)):
        raise ValueError("all requested octave centers must lie below Nyquist")
    direct_sample = int(round(distance_m / sound_speed_m_s * sample_rate))
    if direct_sample + int(round(0.12 * sample_rate)) >= sample_count:
        raise ValueError("observation is too short for direct and early components")
    direct = np.zeros(sample_count, dtype=np.float64)
    direct[direct_sample] = 1.0 / (4.0 * math.pi * distance_m)
    early = np.zeros(sample_count, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    delays_ms = np.asarray((7.0, 12.0, 19.0, 28.0, 40.0, 57.0, 78.0, 105.0))
    delays_ms += rng.uniform(-0.8, 0.8, size=delays_ms.size)
    amplitudes = (
        rng.choice((-1.0, 1.0), size=delays_ms.size)
        * np.exp(-delays_ms / 72.0)
        * rng.uniform(0.45, 0.9, size=delays_ms.size)
        / (4.0 * math.pi * distance_m)
    )
    pulse = np.asarray((0.25, 1.0, 0.25), dtype=np.float64)
    for delay_ms, amplitude in zip(delays_ms, amplitudes):
        index = direct_sample + int(round(delay_ms * 1e-3 * sample_rate))
        early[index - 1 : index + 2] += float(amplitude) * pulse
    white = rng.normal(size=sample_count)
    late: dict[float, np.ndarray] = {}
    for center in centers:
        band = octave_band_rir(white, sample_rate, center)
        band[:direct_sample] = 0.0
        rms = float(np.sqrt(np.mean(np.square(band[direct_sample:]))))
        band *= 0.012 / max(rms, np.finfo(np.float64).tiny)
        late[center] = band
    return SyntheticRecoveryObservation(
        observation_id=observation_id,
        sample_rate=int(sample_rate),
        direct_sample=direct_sample,
        direct_component=direct,
        early_component=early,
        late_basis_by_hz=late,
    )


def render_synthetic_recovery_rir(
    observation: SyntheticRecoveryObservation,
    parameters: SyntheticRecoveryParameters,
    *,
    transition_duration_s: float = 0.016,
) -> np.ndarray:
    """Render one causal direct + coherent-early + decaying multiband RIR."""

    if set(observation.centers_hz) != set(parameters.centers_hz):
        raise ValueError("observation and parameter octave centers must match")
    if not math.isfinite(transition_duration_s) or transition_duration_s <= 0.0:
        raise ValueError("transition_duration_s must be finite and positive")
    relative_time, early_weight, late_weight = _recovery_transition_weights(
        observation,
        parameters,
        transition_duration_s,
    )
    result = observation.direct_component.copy()
    result += (
        10.0 ** (parameters.early_reflection_gain_db / 20.0)
        * observation.early_component
        * early_weight
    )
    decay_time = np.maximum(relative_time, 0.0)
    for center in observation.centers_hz:
        envelope = np.power(
            10.0,
            -3.0 * decay_time / parameters.rt60_s_by_hz[center],
        )
        result += (
            10.0 ** (parameters.late_gain_db_by_hz[center] / 20.0)
            * observation.late_basis_by_hz[center]
            * envelope
            * late_weight
        )
    result[: observation.direct_sample] = 0.0
    return np.asarray(result, dtype=np.float64)


def _recovery_transition_weights(
    observation: SyntheticRecoveryObservation,
    parameters: SyntheticRecoveryParameters,
    transition_duration_s: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    relative_time = (
        np.arange(observation.sample_count, dtype=np.float64)
        - observation.direct_sample
    ) / observation.sample_rate
    progress = np.clip(
        (relative_time - (parameters.mixing_time_s - 0.5 * transition_duration_s))
        / transition_duration_s,
        0.0,
        1.0,
    )
    early_weight = np.cos(0.5 * math.pi * progress)
    late_weight = np.sin(0.5 * math.pi * progress)
    return relative_time, early_weight, late_weight


def _integer_shift(signal: np.ndarray, offset_samples: int) -> np.ndarray:
    result = np.zeros_like(signal)
    offset = int(offset_samples)
    if offset == 0:
        result[:] = signal
    elif offset > 0:
        result[offset:] = signal[:-offset]
    else:
        result[:offset] = signal[-offset:]
    return result


def _windowed_peak_sample(
    signal: np.ndarray,
    expected_sample: int,
    radius_samples: int = 24,
) -> int:
    start = max(0, int(expected_sample) - int(radius_samples))
    end = min(signal.size, int(expected_sample) + int(radius_samples) + 1)
    if end <= start:
        raise ValueError("direct-arrival search window lies outside the RIR")
    return int(start + np.argmax(np.abs(signal[start:end])))


def perturb_synthetic_recovery_measurement(
    observation: SyntheticRecoveryObservation,
    parameters: SyntheticRecoveryParameters,
    perturbation: SyntheticMeasurementPerturbation,
    *,
    transition_duration_s: float = 0.016,
) -> PerturbedSyntheticMeasurement:
    """Create noisy/model-mismatched data and apply known gain/latency correction."""

    clean = render_synthetic_recovery_rir(
        observation,
        parameters,
        transition_duration_s=transition_duration_s,
    )
    relative_time, early_weight, late_weight = _recovery_transition_weights(
        observation,
        parameters,
        transition_duration_s,
    )
    rng = np.random.default_rng(int(perturbation.seed))
    early_mismatch = np.zeros(observation.sample_count, dtype=np.float64)
    delays_ms = np.asarray((15.5, 34.0, 67.0, 93.0), dtype=np.float64)
    delays_ms += rng.uniform(-1.0, 1.0, size=delays_ms.size)
    for delay_ms in delays_ms:
        index = observation.direct_sample + int(
            round(delay_ms * 1e-3 * observation.sample_rate)
        )
        if 1 <= index < observation.sample_count - 1:
            early_mismatch[index - 1 : index + 2] += rng.uniform(
                -1.0, 1.0
            ) * np.asarray((0.2, 1.0, 0.2))
    nominal_early = (
        10.0 ** (parameters.early_reflection_gain_db / 20.0)
        * observation.early_component
        * early_weight
    )
    early_norm = float(np.linalg.norm(early_mismatch))
    if early_norm > np.finfo(np.float64).tiny:
        early_mismatch *= float(np.linalg.norm(nominal_early)) / early_norm
    decay_time = np.maximum(relative_time, 0.0)
    late_mismatch = np.zeros(observation.sample_count, dtype=np.float64)
    for center in observation.centers_hz:
        independent = octave_band_rir(
            rng.normal(size=observation.sample_count),
            observation.sample_rate,
            center,
        )
        independent[: observation.direct_sample] = 0.0
        rms = float(
            np.sqrt(np.mean(np.square(independent[observation.direct_sample :])))
        )
        independent *= 0.012 / max(rms, np.finfo(np.float64).tiny)
        mismatch_rt60 = 1.25 * parameters.rt60_s_by_hz[center]
        envelope = np.power(10.0, -3.0 * decay_time / mismatch_rt60)
        late_mismatch += (
            10.0 ** (parameters.late_gain_db_by_hz[center] / 20.0)
            * independent
            * envelope
            * late_weight
        )
    mismatch = (
        perturbation.early_model_mismatch_fraction * early_mismatch
        + perturbation.late_model_mismatch_fraction * late_mismatch
    )
    acoustic = clean + mismatch
    gain = 10.0 ** (perturbation.gain_error_db / 20.0)
    shifted = _integer_shift(
        gain * acoustic,
        perturbation.latency_offset_samples,
    )
    active_start = max(
        0,
        observation.direct_sample + perturbation.latency_offset_samples,
    )
    signal_rms = float(np.sqrt(np.mean(np.square(shifted[active_start:]))))
    noise_rms = signal_rms * 10.0 ** (-perturbation.snr_db / 20.0)
    noise = rng.normal(scale=noise_rms, size=observation.sample_count)
    raw = shifted + noise
    corrected = (
        _integer_shift(
            raw,
            -perturbation.latency_offset_samples,
        )
        / gain
    )
    expected_raw_direct = observation.direct_sample + int(
        perturbation.latency_offset_samples
    )
    metadata = {
        "policy": "puresound.rir_synthetic_measurement_perturbation.v1",
        "known_nuisance_correction": {
            "gain_error_db_removed": float(perturbation.gain_error_db),
            "latency_offset_samples_removed": int(perturbation.latency_offset_samples),
        },
        "expected_raw_direct_sample": int(expected_raw_direct),
        "windowed_detected_raw_direct_sample": _windowed_peak_sample(
            raw,
            expected_raw_direct,
        ),
        "global_raw_peak_sample_diagnostic": int(np.argmax(np.abs(raw))),
        "expected_corrected_direct_sample": int(observation.direct_sample),
        "windowed_detected_corrected_direct_sample": _windowed_peak_sample(
            corrected,
            observation.direct_sample,
        ),
        "global_corrected_peak_sample_diagnostic": int(np.argmax(np.abs(corrected))),
        "noise_rms_raw": noise_rms,
        "clean_energy": float(np.dot(clean, clean)),
        "mismatch_energy": float(np.dot(mismatch, mismatch)),
        "mismatch_to_clean_energy_ratio": float(
            np.dot(mismatch, mismatch)
            / max(float(np.dot(clean, clean)), np.finfo(np.float64).tiny)
        ),
    }
    return PerturbedSyntheticMeasurement(
        raw_rir=np.asarray(raw, dtype=np.float64),
        corrected_rir=np.asarray(corrected, dtype=np.float64),
        clean_rir=clean,
        mismatch_component=np.asarray(mismatch, dtype=np.float64),
        noise_component_raw=np.asarray(noise, dtype=np.float64),
        perturbation=perturbation,
        metadata=metadata,
    )


def _window_log_energy(signal: np.ndarray, window: int) -> np.ndarray:
    length = signal.size // window * window
    if length == 0:
        return np.zeros(0, dtype=np.float64)
    energy = np.mean(np.square(signal[:length].reshape(-1, window)), axis=1)
    return np.log(np.maximum(energy, 1e-14))


def _recovery_residual_v1(
    vector: np.ndarray,
    centers_hz: tuple[float, ...],
    observations: Sequence[SyntheticRecoveryObservation],
    targets: Sequence[np.ndarray],
) -> np.ndarray:
    parameters = SyntheticRecoveryParameters.from_vector(centers_hz, vector)
    residuals = []
    for observation, target in zip(observations, targets):
        candidate = render_synthetic_recovery_rir(observation, parameters)
        scale = max(float(np.linalg.norm(target)), 1e-8)
        waveform = (candidate - target) / scale
        early_end = min(
            observation.sample_count,
            observation.direct_sample + int(round(0.12 * observation.sample_rate)),
        )
        early = (
            candidate[observation.direct_sample : early_end]
            - target[observation.direct_sample : early_end]
        ) / scale
        window = max(16, int(round(0.008 * observation.sample_rate)))
        log_energy = (
            _window_log_energy(candidate, window) - _window_log_energy(target, window)
        ) / 4.0
        residuals.extend((waveform, 3.0 * early, log_energy))
    return np.concatenate(residuals)


@dataclass(frozen=True)
class _RecoveryTargetFeatures:
    target: np.ndarray
    scale: float
    early_end: int
    broadband_log_energy: np.ndarray
    broadband_mask: np.ndarray
    octave_log_energy: Mapping[float, np.ndarray]
    octave_masks: Mapping[float, np.ndarray]


def _noise_energy(signal: np.ndarray, direct: int) -> float:
    quiet_end = max(1, direct // 2)
    quiet = signal[:quiet_end]
    return float(
        max(
            np.median(np.square(quiet)),
            np.finfo(np.float64).tiny,
        )
    )


def _target_log_energy_and_mask(
    signal: np.ndarray,
    direct: int,
    window: int,
    noise_margin_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    tail = signal[direct:]
    length = tail.size // window * window
    if length == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=bool)
    energy = np.mean(np.square(tail[:length].reshape(-1, window)), axis=1)
    threshold = _noise_energy(signal, direct) * 10.0 ** (noise_margin_db / 10.0)
    mask = energy >= threshold
    if not np.any(mask):
        mask[int(np.argmax(energy))] = True
    return np.log(np.maximum(energy, 1e-18)), mask


def _build_robust_target_features(
    observation: SyntheticRecoveryObservation,
    target: np.ndarray,
    config: SyntheticRecoveryObjectiveConfig,
) -> _RecoveryTargetFeatures:
    window = max(
        16,
        int(round(config.energy_window_ms * 1e-3 * observation.sample_rate)),
    )
    broadband_log, broadband_mask = _target_log_energy_and_mask(
        target,
        observation.direct_sample,
        window,
        config.noise_margin_db,
    )
    octave_log: dict[float, np.ndarray] = {}
    octave_masks: dict[float, np.ndarray] = {}
    for center in observation.centers_hz:
        band = octave_band_rir(target, observation.sample_rate, center)
        values, mask = _target_log_energy_and_mask(
            band,
            observation.direct_sample,
            window,
            config.noise_margin_db,
        )
        octave_log[center] = values
        octave_masks[center] = mask
    early_end = min(
        observation.sample_count,
        observation.direct_sample + int(round(0.12 * observation.sample_rate)),
    )
    return _RecoveryTargetFeatures(
        target=target,
        scale=max(float(np.linalg.norm(target)), 1e-8),
        early_end=early_end,
        broadband_log_energy=broadband_log,
        broadband_mask=broadband_mask,
        octave_log_energy=octave_log,
        octave_masks=octave_masks,
    )


def _candidate_log_energy(
    signal: np.ndarray,
    direct: int,
    window: int,
) -> np.ndarray:
    return _window_log_energy(signal[direct:], window)


def _recovery_residual_m4_multiterm(
    vector: np.ndarray,
    centers_hz: tuple[float, ...],
    observations: Sequence[SyntheticRecoveryObservation],
    features: Sequence[_RecoveryTargetFeatures],
    config: SyntheticRecoveryObjectiveConfig,
) -> np.ndarray:
    parameters = SyntheticRecoveryParameters.from_vector(centers_hz, vector)
    residuals = []
    for observation, target_features in zip(observations, features):
        candidate = render_synthetic_recovery_rir(observation, parameters)
        target = target_features.target
        scale = target_features.scale
        if config.waveform_weight > 0.0:
            residuals.append(
                math.sqrt(config.waveform_weight) * (candidate - target) / scale
            )
        if config.early_waveform_weight > 0.0:
            early_slice = slice(
                observation.direct_sample,
                target_features.early_end,
            )
            residuals.append(
                math.sqrt(config.early_waveform_weight)
                * (candidate[early_slice] - target[early_slice])
                / scale
            )
        window = max(
            16,
            int(round(config.energy_window_ms * 1e-3 * observation.sample_rate)),
        )
        if config.broadband_decay_weight > 0.0:
            candidate_log = _candidate_log_energy(
                candidate,
                observation.direct_sample,
                window,
            )
            mask = target_features.broadband_mask
            residuals.append(
                math.sqrt(config.broadband_decay_weight)
                * (candidate_log[mask] - target_features.broadband_log_energy[mask])
                / 4.0
            )
        if config.octave_decay_weight > 0.0:
            octave_scale = math.sqrt(
                config.octave_decay_weight / len(observation.centers_hz)
            )
            for center in observation.centers_hz:
                candidate_band = octave_band_rir(
                    candidate,
                    observation.sample_rate,
                    center,
                )
                candidate_log = _candidate_log_energy(
                    candidate_band,
                    observation.direct_sample,
                    window,
                )
                mask = target_features.octave_masks[center]
                residuals.append(
                    octave_scale
                    * (
                        candidate_log[mask]
                        - target_features.octave_log_energy[center][mask]
                    )
                    / 4.0
                )
    return np.concatenate(residuals)


def fit_synthetic_recovery_parameters(
    observations: Sequence[SyntheticRecoveryObservation],
    targets: Sequence[Any],
    initial_parameters: SyntheticRecoveryParameters,
    *,
    bounds: SyntheticRecoveryBounds | None = None,
    objective: SyntheticRecoveryObjectiveConfig | None = None,
    maximum_evaluations: int = 300,
) -> SyntheticRecoveryFit:
    """Fit shared parameters to multiple synthetic positions with box bounds."""

    observations = tuple(observations)
    if not observations or len(observations) != len(targets):
        raise ValueError("observations and targets must have equal non-zero length")
    centers = observations[0].centers_hz
    if initial_parameters.centers_hz != centers or any(
        observation.centers_hz != centers for observation in observations
    ):
        raise ValueError("all observations and parameters must share octave centers")
    arrays = tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in targets)
    if any(
        value.shape != (observation.sample_count,) or not np.all(np.isfinite(value))
        for value, observation in zip(arrays, observations)
    ):
        raise ValueError("each target must be finite and match its observation")
    if maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be positive")
    active_bounds = bounds or SyntheticRecoveryBounds()
    active_objective = objective or SyntheticRecoveryObjectiveConfig()
    lower, upper = active_bounds.vectors(len(centers))
    initial = initial_parameters.to_vector()
    if np.any(initial <= lower) or np.any(initial >= upper):
        raise ValueError("initial parameters must lie strictly inside the bounds")
    if active_objective.mode == "waveform_v1":
        residual_function = _recovery_residual_v1
        residual_args: tuple[Any, ...] = (centers, observations, arrays)
    else:
        features = tuple(
            _build_robust_target_features(observation, target, active_objective)
            for observation, target in zip(observations, arrays)
        )
        residual_function = _recovery_residual_m4_multiterm
        residual_args = (
            centers,
            observations,
            features,
            active_objective,
        )
    initial_residual = residual_function(initial, *residual_args)
    result = least_squares(
        residual_function,
        initial,
        bounds=(lower, upper),
        args=residual_args,
        x_scale=upper - lower,
        max_nfev=int(maximum_evaluations),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    scaled_jacobian = np.asarray(result.jac, dtype=np.float64) * (upper - lower)
    singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
    threshold = (
        max(scaled_jacobian.shape) * np.finfo(np.float64).eps * singular_values[0]
    )
    full_rank = bool(
        singular_values.size == initial.size and singular_values[-1] > threshold
    )
    condition = float(
        singular_values[0] / max(singular_values[-1], np.finfo(np.float64).tiny)
    )
    fitted = SyntheticRecoveryParameters.from_vector(centers, result.x)
    return SyntheticRecoveryFit(
        parameters=fitted,
        initial_parameters=initial_parameters,
        success=bool(result.success),
        message=str(result.message),
        evaluations=int(result.nfev),
        initial_cost=float(0.5 * np.dot(initial_residual, initial_residual)),
        final_cost=float(result.cost),
        scaled_jacobian_singular_values=tuple(
            float(value) for value in singular_values
        ),
        scaled_jacobian_condition_number=condition,
        locally_full_rank=full_rank,
        objective=active_objective,
    )


def evaluate_synthetic_recovery(
    observations: Sequence[SyntheticRecoveryObservation],
    targets: Sequence[Any],
    parameters: SyntheticRecoveryParameters,
) -> list[dict[str, Any]]:
    """Evaluate fitted responses with the independent M5.1 loss oracle."""

    if len(observations) != len(targets):
        raise ValueError("observations and targets must have equal length")
    weights = CalibrationLossWeights(decay_regularization=0.0)
    reports = []
    for observation, target in zip(observations, targets):
        candidate = render_synthetic_recovery_rir(observation, parameters)
        report = analyze_rir_calibration_loss(
            target,
            candidate,
            observation.sample_rate,
            measured_direct_samples=(observation.direct_sample,),
            synthetic_direct_samples=(observation.direct_sample,),
            physical_first_samples=(observation.direct_sample,),
            weights=weights,
            octave_centers_hz=observation.centers_hz,
        )
        reports.append(
            {
                "observation_id": observation.observation_id,
                **report.to_dict(),
            }
        )
    return reports


__all__ = [
    "RIR_ROBUST_RECOVERY_OBJECTIVE_POLICY",
    "RIR_SYNTHETIC_RECOVERY_POLICY",
    "PerturbedSyntheticMeasurement",
    "SyntheticMeasurementPerturbation",
    "SyntheticRecoveryBounds",
    "SyntheticRecoveryFit",
    "SyntheticRecoveryObservation",
    "SyntheticRecoveryObjectiveConfig",
    "SyntheticRecoveryParameters",
    "build_synthetic_recovery_observation",
    "evaluate_synthetic_recovery",
    "fit_synthetic_recovery_parameters",
    "perturb_synthetic_recovery_measurement",
    "render_synthetic_recovery_rir",
]
