"""Passivity-by-construction fitting of complex impedance measurements."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.optimize import least_squares

from puresound.audio.rir.physics.impedance.admittance import (
    PassiveMultiPoleAdmittance,
    PassiveResonantAdmittance,
    normal_incidence_reflection_coefficient,
)
from puresound.audio.rir.physics.impedance.measurements import (
    ComplexImpedanceMeasurement,
    NormalizedComplexImpedanceMeasurement,
)


@dataclass(frozen=True)
class PassiveMultiPoleFit:
    model: PassiveMultiPoleAdmittance
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    num_frequencies: int
    rms_complex_reflection_error: float
    maximum_complex_reflection_error: float
    maximum_magnitude_error: float
    maximum_phase_error_rad: float
    acceptance_threshold: float
    source_measurement_id: str | None = None
    weighting_strategy: str = "uniform"

    @property
    def accepted(self) -> bool:
        return self.maximum_complex_reflection_error <= self.acceptance_threshold

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model.metadata(),
            "fit_frequency_range_hz": [
                float(self.minimum_frequency_hz),
                float(self.maximum_frequency_hz),
            ],
            "num_frequencies": int(self.num_frequencies),
            "rms_complex_reflection_error": float(
                self.rms_complex_reflection_error
            ),
            "maximum_complex_reflection_error": float(
                self.maximum_complex_reflection_error
            ),
            "maximum_magnitude_error": float(self.maximum_magnitude_error),
            "maximum_phase_error_rad": float(self.maximum_phase_error_rad),
            "acceptance_threshold": float(self.acceptance_threshold),
            "accepted": bool(self.accepted),
            "source_measurement_id": self.source_measurement_id,
            "fit_domain": "complex_pressure_reflection_coefficient",
            "weighting_strategy": self.weighting_strategy,
            "passivity_strategy": (
                "non_negative_parallel_positive_real_branches"
            ),
            "pole_strategy": "fixed_real_relaxation_poles",
        }


def fit_passive_multi_pole_admittance(
    frequencies_hz: Iterable[float],
    surface_impedance_pa_s_m: Iterable[complex],
    *,
    air_density_kg_m3: float,
    sound_speed_m_s: float,
    pole_frequencies_hz: Iterable[float],
    reflection_weights: Iterable[float] | None = None,
    acceptance_threshold: float = 0.05,
    source_measurement_id: str | None = None,
    weighting_strategy: str = "uniform",
) -> PassiveMultiPoleFit:
    """Fit non-negative parallel relaxation branches at fixed real poles.

    Fixed poles avoid an unstable pole-relocation stage. Non-negative static,
    low-pass, and high-pass branch strengths make the resulting admittance
    positive-real by construction.
    """
    frequencies = np.asarray(tuple(frequencies_hz), dtype=np.float64)
    impedance = np.asarray(
        tuple(surface_impedance_pa_s_m),
        dtype=np.complex128,
    )
    poles = np.asarray(tuple(pole_frequencies_hz), dtype=np.float64)
    if (
        frequencies.ndim != 1
        or frequencies.size < 3
        or impedance.shape != frequencies.shape
    ):
        raise ValueError("frequency and impedance inputs must have equal length >= 3")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("fit frequencies must be finite and positive")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("fit frequencies must be strictly increasing")
    if np.any(~np.isfinite(impedance.real)) or np.any(~np.isfinite(impedance.imag)):
        raise ValueError("fit impedance must be finite")
    if np.any(impedance.real < 0.0):
        raise ValueError("fit impedance must be passive")
    if (
        poles.ndim != 1
        or poles.size < 1
        or np.any(~np.isfinite(poles))
        or np.any(poles <= 0.0)
        or np.any(np.diff(poles) <= 0.0)
    ):
        raise ValueError("pole frequencies must be finite, positive, and increasing")
    if not math.isfinite(acceptance_threshold) or acceptance_threshold <= 0.0:
        raise ValueError("acceptance threshold must be finite and positive")
    if reflection_weights is None:
        weights = np.ones_like(frequencies)
    else:
        weights = np.asarray(
            tuple(reflection_weights),
            dtype=np.float64,
        )
        if (
            weights.shape != frequencies.shape
            or np.any(~np.isfinite(weights))
            or np.any(weights <= 0.0)
        ):
            raise ValueError(
                "reflection weights must match frequencies and be positive"
            )
        weights = weights / float(np.median(weights))
    if not weighting_strategy:
        raise ValueError("weighting strategy cannot be empty")

    target = np.asarray(
        [
            normal_incidence_reflection_coefficient(
                value,
                air_density_kg_m3,
                sound_speed_m_s,
            )
            for value in impedance
        ]
    )
    if np.any(np.abs(target) > 1.0 + 1e-9):
        raise ValueError("target impedance violates passive reflection magnitude")
    num_poles = int(poles.size)

    def model_from_coefficients(
        coefficients: Iterable[float],
    ) -> PassiveMultiPoleAdmittance:
        values = np.asarray(tuple(coefficients), dtype=np.float64)
        return PassiveMultiPoleAdmittance(
            normalized_admittance_static=float(values[0]),
            pole_frequencies_hz=tuple(float(value) for value in poles),
            normalized_admittance_lowpass=tuple(
                float(value) for value in values[1 : 1 + num_poles]
            ),
            normalized_admittance_highpass=tuple(
                float(value) for value in values[1 + num_poles :]
            ),
        )

    def residual(coefficients: Iterable[float]) -> np.ndarray:
        model = model_from_coefficients(coefficients)
        predicted = np.asarray(
            [
                model.reflection_coefficient(float(frequency))
                for frequency in frequencies
            ]
        )
        error = predicted - target
        return np.concatenate((weights * error.real, weights * error.imag))

    branch_scale = 0.5 / num_poles
    starts = (
        np.concatenate(
            (
                [0.001],
                np.full(num_poles, 0.001),
                np.full(num_poles, branch_scale),
            )
        ),
        np.concatenate(
            (
                [0.05],
                np.full(num_poles, branch_scale),
                np.full(num_poles, 0.001),
            )
        ),
        np.full(1 + 2 * num_poles, 0.05),
    )
    results = [
        least_squares(
            residual,
            start,
            bounds=(0.0, 100.0),
            max_nfev=30_000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        for start in starts
    ]
    result = min(results, key=lambda item: float(item.cost))
    model = model_from_coefficients(result.x)
    predicted = np.asarray(
        [
            model.reflection_coefficient(float(frequency))
            for frequency in frequencies
        ]
    )
    complex_error = np.abs(predicted - target)
    magnitude_error = np.abs(np.abs(predicted) - np.abs(target))
    safe_target = np.where(np.abs(target) > 1e-12, target, 1.0 + 0.0j)
    phase_error = np.abs(np.angle(predicted / safe_target))
    return PassiveMultiPoleFit(
        model=model,
        minimum_frequency_hz=float(frequencies[0]),
        maximum_frequency_hz=float(frequencies[-1]),
        num_frequencies=int(frequencies.size),
        rms_complex_reflection_error=float(
            np.sqrt(np.mean(np.square(complex_error)))
        ),
        maximum_complex_reflection_error=float(np.max(complex_error)),
        maximum_magnitude_error=float(np.max(magnitude_error)),
        maximum_phase_error_rad=float(np.max(phase_error)),
        acceptance_threshold=float(acceptance_threshold),
        source_measurement_id=source_measurement_id,
        weighting_strategy=str(weighting_strategy),
    )


def fit_complex_impedance_measurement(
    measurement: ComplexImpedanceMeasurement,
    *,
    num_poles: int = 3,
    pole_frequencies_hz: Iterable[float] | None = None,
    acceptance_threshold: float = 0.05,
) -> PassiveMultiPoleFit:
    """Fit one strict measurement with fixed, log-spaced real poles."""
    if int(num_poles) < 1:
        raise ValueError("num_poles must be positive")
    if pole_frequencies_hz is None:
        minimum = float(measurement.frequencies_hz[0])
        maximum = float(measurement.frequencies_hz[-1])
        if int(num_poles) == 1:
            poles = (math.sqrt(minimum * maximum),)
        else:
            poles = tuple(
                float(value)
                for value in np.geomspace(
                    0.5 * minimum,
                    2.0 * maximum,
                    int(num_poles),
                )
            )
    else:
        poles = tuple(float(value) for value in pole_frequencies_hz)
        if len(poles) != int(num_poles):
            raise ValueError("num_poles must match pole_frequencies_hz")

    reflection_weights = None
    weighting_strategy = "uniform"
    if measurement.impedance_real_std_pa_s_m:
        impedance = measurement.complex_impedance_pa_s_m
        real_std = np.asarray(
            measurement.impedance_real_std_pa_s_m,
            dtype=np.float64,
        )
        imag_std = np.asarray(
            measurement.impedance_imag_std_pa_s_m,
            dtype=np.float64,
        )
        z0 = measurement.air_density_kg_m3 * measurement.sound_speed_m_s
        derivative_magnitude = np.abs(
            2.0 * z0 / np.square(impedance + z0)
        )
        reflection_std = derivative_magnitude * np.sqrt(
            0.5 * (np.square(real_std) + np.square(imag_std))
        )
        positive = reflection_std[reflection_std > 0.0]
        if positive.size:
            floor = max(1e-6, 0.1 * float(np.median(positive)))
            reflection_weights = 1.0 / np.maximum(reflection_std, floor)
            weighting_strategy = (
                "inverse_propagated_complex_reflection_standard_deviation"
            )
    return fit_passive_multi_pole_admittance(
        measurement.frequencies_hz,
        measurement.complex_impedance_pa_s_m,
        air_density_kg_m3=measurement.air_density_kg_m3,
        sound_speed_m_s=measurement.sound_speed_m_s,
        pole_frequencies_hz=poles,
        reflection_weights=reflection_weights,
        acceptance_threshold=acceptance_threshold,
        source_measurement_id=measurement.measurement_id,
        weighting_strategy=weighting_strategy,
    )


@dataclass(frozen=True)
class ComplexReflectionFitMetrics:
    minimum_frequency_hz: float
    maximum_frequency_hz: float
    num_frequencies: int
    rms_complex_reflection_error: float
    maximum_complex_reflection_error: float
    maximum_magnitude_error: float
    maximum_phase_error_rad: float

    def metadata(self) -> dict[str, Any]:
        return {
            "frequency_range_hz": [
                float(self.minimum_frequency_hz),
                float(self.maximum_frequency_hz),
            ],
            "num_frequencies": int(self.num_frequencies),
            "rms_complex_reflection_error": float(
                self.rms_complex_reflection_error
            ),
            "maximum_complex_reflection_error": float(
                self.maximum_complex_reflection_error
            ),
            "maximum_magnitude_error": float(self.maximum_magnitude_error),
            "maximum_phase_error_rad": float(self.maximum_phase_error_rad),
        }


@dataclass(frozen=True)
class PassiveResonantFit:
    model: PassiveResonantAdmittance
    training_metrics: ComplexReflectionFitMetrics
    held_out_metrics: ComplexReflectionFitMetrics | None
    acceptance_threshold: float
    source_measurement_id: str | None = None

    @property
    def accepted(self) -> bool:
        if (
            self.training_metrics.maximum_complex_reflection_error
            > self.acceptance_threshold
        ):
            return False
        return self.held_out_metrics is None or (
            self.held_out_metrics.maximum_complex_reflection_error
            <= self.acceptance_threshold
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model": self.model.metadata(),
            "training": self.training_metrics.metadata(),
            "held_out": (
                None
                if self.held_out_metrics is None
                else self.held_out_metrics.metadata()
            ),
            "acceptance_threshold": float(self.acceptance_threshold),
            "accepted": bool(self.accepted),
            "source_measurement_id": self.source_measurement_id,
            "fit_domain": "bounded_impedance_cayley_transform",
            "passivity_strategy": (
                "non_negative_parallel_series_r_l_c_branches"
            ),
            "pole_strategy": "fitted_stable_conjugate_pair",
        }


def _complex_reflection_fit_metrics(
    frequencies_hz: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
) -> ComplexReflectionFitMetrics:
    complex_error = np.abs(predicted - target)
    magnitude_error = np.abs(np.abs(predicted) - np.abs(target))
    safe_target = np.where(np.abs(target) > 1e-12, target, 1.0 + 0.0j)
    phase_error = np.abs(np.angle(predicted / safe_target))
    return ComplexReflectionFitMetrics(
        minimum_frequency_hz=float(frequencies_hz[0]),
        maximum_frequency_hz=float(frequencies_hz[-1]),
        num_frequencies=int(frequencies_hz.size),
        rms_complex_reflection_error=float(
            np.sqrt(np.mean(np.square(complex_error)))
        ),
        maximum_complex_reflection_error=float(np.max(complex_error)),
        maximum_magnitude_error=float(np.max(magnitude_error)),
        maximum_phase_error_rad=float(np.max(phase_error)),
    )


def fit_passive_single_resonance_admittance(
    frequencies_hz: Iterable[float],
    normalized_surface_impedance: Iterable[complex],
    *,
    training_frequency_indices: Iterable[int] | None = None,
    acceptance_threshold: float = 0.1,
    source_measurement_id: str | None = None,
) -> PassiveResonantFit:
    """Fit a passive series-RLC admittance with a free conjugate pole pair."""
    frequencies = np.asarray(tuple(frequencies_hz), dtype=np.float64)
    impedance = np.asarray(
        tuple(normalized_surface_impedance),
        dtype=np.complex128,
    )
    if (
        frequencies.ndim != 1
        or frequencies.size < 3
        or impedance.shape != frequencies.shape
    ):
        raise ValueError("frequency and impedance inputs must have equal length >= 3")
    if np.any(~np.isfinite(frequencies)) or np.any(frequencies <= 0.0):
        raise ValueError("fit frequencies must be finite and positive")
    if np.any(np.diff(frequencies) <= 0.0):
        raise ValueError("fit frequencies must be strictly increasing")
    if np.any(~np.isfinite(impedance.real)) or np.any(~np.isfinite(impedance.imag)):
        raise ValueError("fit impedance must be finite")
    if np.any(impedance.real < 0.0):
        raise ValueError("fit impedance must be passive")
    if not math.isfinite(acceptance_threshold) or acceptance_threshold <= 0.0:
        raise ValueError("acceptance threshold must be finite and positive")

    target = (impedance - 1.0) / (impedance + 1.0)
    if np.any(np.abs(target) > 1.0 + 1e-9):
        raise ValueError("target impedance violates passive Cayley magnitude")
    if training_frequency_indices is None:
        training_indices = np.arange(frequencies.size, dtype=np.int64)
    else:
        training_indices = np.asarray(
            tuple(int(value) for value in training_frequency_indices),
            dtype=np.int64,
        )
        if (
            training_indices.ndim != 1
            or training_indices.size < 3
            or np.any(training_indices < 0)
            or np.any(training_indices >= frequencies.size)
            or np.unique(training_indices).size != training_indices.size
        ):
            raise ValueError(
                "training frequency indices must contain at least three "
                "unique in-range indices"
            )
        training_indices = np.sort(training_indices)
    held_out_indices = np.setdiff1d(
        np.arange(frequencies.size, dtype=np.int64),
        training_indices,
    )

    def model_from_parameters(
        parameters: Iterable[float],
    ) -> PassiveResonantAdmittance:
        static, peak, quality_factor, resonance_frequency_hz = (
            float(value) for value in parameters
        )
        return PassiveResonantAdmittance(
            normalized_admittance_static=static,
            resonance_frequencies_hz=(resonance_frequency_hz,),
            quality_factors=(quality_factor,),
            peak_normalized_admittances=(peak,),
        )

    def predict(
        model: PassiveResonantAdmittance,
        indices: np.ndarray,
    ) -> np.ndarray:
        return np.asarray(
            [
                model.reflection_coefficient(float(frequencies[index]))
                for index in indices
            ],
            dtype=np.complex128,
        )

    def residual(parameters: Iterable[float]) -> np.ndarray:
        error = predict(
            model_from_parameters(parameters),
            training_indices,
        ) - target[training_indices]
        return np.concatenate((error.real, error.imag))

    resonance_guess_index = int(np.argmin(np.abs(impedance)))
    resonance_guess = float(frequencies[resonance_guess_index])
    peak_guess = float(
        min(
            100.0,
            max(0.1, 1.0 / max(impedance[resonance_guess_index].real, 0.01)),
        )
    )
    starts = tuple(
        np.asarray(
            [static, peak_guess, quality_factor, resonance_guess],
            dtype=np.float64,
        )
        for static in (0.001, 0.05)
        for quality_factor in (2.0, 10.0, 40.0)
    )
    lower = np.asarray(
        [0.0, 0.0, 0.05, 0.5 * frequencies[0]],
        dtype=np.float64,
    )
    upper = np.asarray(
        [100.0, 100.0, 500.0, 2.0 * frequencies[-1]],
        dtype=np.float64,
    )
    results = [
        least_squares(
            residual,
            start,
            bounds=(lower, upper),
            max_nfev=30_000,
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
        )
        for start in starts
    ]
    result = min(results, key=lambda item: float(item.cost))
    model = model_from_parameters(result.x)
    training_metrics = _complex_reflection_fit_metrics(
        frequencies[training_indices],
        target[training_indices],
        predict(model, training_indices),
    )
    held_out_metrics = (
        None
        if held_out_indices.size == 0
        else _complex_reflection_fit_metrics(
            frequencies[held_out_indices],
            target[held_out_indices],
            predict(model, held_out_indices),
        )
    )
    return PassiveResonantFit(
        model=model,
        training_metrics=training_metrics,
        held_out_metrics=held_out_metrics,
        acceptance_threshold=float(acceptance_threshold),
        source_measurement_id=source_measurement_id,
    )


def fit_normalized_complex_impedance_measurement(
    measurement: NormalizedComplexImpedanceMeasurement,
    *,
    alternating_frequency_holdout: bool = True,
    acceptance_threshold: float = 0.1,
) -> PassiveResonantFit:
    """Fit direct normalized data with an alternating-frequency holdout."""
    training_indices = (
        tuple(range(0, len(measurement.frequencies_hz), 2))
        if alternating_frequency_holdout
        else None
    )
    return fit_passive_single_resonance_admittance(
        measurement.frequencies_hz,
        measurement.complex_normalized_impedance,
        training_frequency_indices=training_indices,
        acceptance_threshold=acceptance_threshold,
        source_measurement_id=measurement.measurement_id,
    )


__all__ = [
    "ComplexReflectionFitMetrics",
    "PassiveMultiPoleFit",
    "PassiveResonantFit",
    "fit_complex_impedance_measurement",
    "fit_normalized_complex_impedance_measurement",
    "fit_passive_multi_pole_admittance",
    "fit_passive_single_resonance_admittance",
]
