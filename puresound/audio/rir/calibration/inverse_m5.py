"""Readiness-gated inverse-calibration utilities for M5.

The module keeps three evidence levels separate:

* synthetic local identifiability of grouped PathEvent/M4 parameters;
* measured-campaign readiness and room/position split semantics;
* constrained residual and spatial candidate evaluation.

No synthetic result can promote a measured or production exit.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from puresound.audio.rir_calibration import (
    CalibrationLossWeights,
    analyze_rir_calibration_loss,
)
from puresound.audio.rir_late_coupling import (
    PathEventFDNCouplingResult,
    couple_path_event_rir_with_fdn,
)
from puresound.audio.rir_path_events import (
    ComplexPathGainSpectrum,
    PathEvent,
    PathEventSet,
    render_path_events,
)


M5_GROUPED_PATH_INVERSE_POLICY = "puresound.m5_grouped_path_inverse.v1"
M5_LOCAL_IDENTIFIABILITY_POLICY = "puresound.m5_local_identifiability.v1"
M5_SPATIAL_CANDIDATE_PROFILE_POLICY = "puresound.m5_spatial_candidate_profile.v1"


def _finite_mapping(name: str, values: Mapping[str, float]) -> dict[str, float]:
    result = {str(key): float(value) for key, value in sorted(values.items())}
    if not result or any(
        not key or not math.isfinite(value) for key, value in result.items()
    ):
        raise ValueError(f"{name} must contain named finite values")
    return result


def _finite_band_mapping(
    name: str,
    values: Mapping[float, float],
) -> dict[float, float]:
    result = dict(
        sorted((float(center), float(value)) for center, value in values.items())
    )
    if not result or any(
        not math.isfinite(center)
        or center <= 0.0
        or not math.isfinite(value)
        or value <= 0.0
        for center, value in result.items()
    ):
        raise ValueError(f"{name} must contain positive finite octave values")
    return result


@dataclass(frozen=True)
class LocalIdentifiabilityReport:
    """Rank, conditioning, correlation, and an explicit accepted subset."""

    parameter_names: tuple[str, ...]
    numerical_rank: int
    full_column_rank: bool
    normalized_condition_number: float
    maximum_absolute_column_correlation: float
    accepted_parameters: tuple[str, ...]
    rejected_parameters: Mapping[str, str]
    singular_values: tuple[float, ...]
    absolute_column_correlation: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": M5_LOCAL_IDENTIFIABILITY_POLICY,
            "parameter_names": list(self.parameter_names),
            "numerical_rank": int(self.numerical_rank),
            "full_column_rank": bool(self.full_column_rank),
            "normalized_condition_number": float(self.normalized_condition_number),
            "maximum_absolute_column_correlation": float(
                self.maximum_absolute_column_correlation
            ),
            "accepted_parameters": list(self.accepted_parameters),
            "rejected_parameters": dict(self.rejected_parameters),
            "singular_values": list(self.singular_values),
            "absolute_column_correlation": self.absolute_column_correlation.astype(
                float
            ).tolist(),
        }


def analyze_local_identifiability(
    jacobian: Any,
    parameter_names: Sequence[str],
    *,
    maximum_condition_number: float = 100.0,
    maximum_absolute_correlation: float = 0.995,
) -> LocalIdentifiabilityReport:
    """Analyze scaled sensitivity columns and reject redundant parameters.

    ``parameter_names`` is also the priority order. Earlier physical groups are
    retained when a later column is numerically redundant with them.
    """

    matrix = np.asarray(jacobian, dtype=np.float64)
    names = tuple(str(value) for value in parameter_names)
    if (
        matrix.ndim != 2
        or matrix.shape[1] != len(names)
        or matrix.shape[0] < 1
        or not names
        or len(set(names)) != len(names)
        or any(not name for name in names)
        or not np.all(np.isfinite(matrix))
    ):
        raise ValueError("identifiability Jacobian and parameter names are invalid")
    if (
        not math.isfinite(maximum_condition_number)
        or maximum_condition_number <= 1.0
        or not math.isfinite(maximum_absolute_correlation)
        or not 0.0 < maximum_absolute_correlation < 1.0
    ):
        raise ValueError("identifiability thresholds are invalid")
    norms = np.linalg.norm(matrix, axis=0)
    nonzero = norms > np.finfo(np.float64).tiny
    normalized = np.zeros_like(matrix)
    normalized[:, nonzero] = matrix[:, nonzero] / norms[nonzero]
    correlation = np.abs(normalized.T @ normalized)
    np.fill_diagonal(correlation, np.where(nonzero, 1.0, 0.0))
    singular_values = np.linalg.svd(normalized, compute_uv=False)
    rank_threshold = (
        max(normalized.shape)
        * np.finfo(np.float64).eps
        * (singular_values[0] if singular_values.size else 0.0)
    )
    rank = int(np.sum(singular_values > rank_threshold))
    condition = float(
        min(
            np.finfo(np.float64).max,
            singular_values[0]
            / max(
                singular_values[-1],
                singular_values[0] / np.finfo(np.float64).max,
                np.finfo(np.float64).tiny,
            ),
        )
    )
    off_diagonal = correlation.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    maximum_correlation = float(np.max(off_diagonal, initial=0.0))

    accepted_indices: list[int] = []
    rejected: dict[str, str] = {}
    for index, name in enumerate(names):
        if not nonzero[index]:
            rejected[name] = "zero_local_sensitivity"
            continue
        if accepted_indices and any(
            correlation[index, previous] >= maximum_absolute_correlation
            for previous in accepted_indices
        ):
            previous = max(
                accepted_indices,
                key=lambda value: correlation[index, value],
            )
            rejected[name] = f"redundant_with:{names[previous]}"
            continue
        trial = normalized[:, (*accepted_indices, index)]
        trial_singular = np.linalg.svd(trial, compute_uv=False)
        trial_condition = float(
            min(
                np.finfo(np.float64).max,
                trial_singular[0]
                / max(
                    trial_singular[-1],
                    trial_singular[0] / np.finfo(np.float64).max,
                    np.finfo(np.float64).tiny,
                ),
            )
        )
        if trial_condition > maximum_condition_number:
            rejected[name] = f"subset_condition_exceeds:{maximum_condition_number:g}"
            continue
        accepted_indices.append(index)
    return LocalIdentifiabilityReport(
        parameter_names=names,
        numerical_rank=rank,
        full_column_rank=bool(rank == len(names)),
        normalized_condition_number=condition,
        maximum_absolute_column_correlation=maximum_correlation,
        accepted_parameters=tuple(names[index] for index in accepted_indices),
        rejected_parameters=rejected,
        singular_values=tuple(float(value) for value in singular_values),
        absolute_column_correlation=correlation,
    )


@dataclass(frozen=True)
class GroupedPathObservation:
    """One inspectable PathEvent channel with named surface groups."""

    observation_id: str
    event_set: PathEventSet
    surface_group_by_id: Mapping[str, str]
    sample_rate: int
    sample_count: int
    direct_sample: int
    fdn_seed: int
    delay_line_count: int = 4
    transition_duration_s: float = 0.016
    filter_order: int = 4

    def __post_init__(self) -> None:
        groups = {
            str(surface_id): str(group)
            for surface_id, group in self.surface_group_by_id.items()
        }
        referenced = {
            surface_id
            for event in self.event_set.events
            for surface_id in event.surface_ids
        }
        if not self.observation_id or self.sample_rate <= 0 or self.sample_count < 256:
            raise ValueError("grouped PathEvent observation shape is invalid")
        if not 0 <= int(self.direct_sample) < self.sample_count:
            raise ValueError("grouped PathEvent direct sample is invalid")
        if referenced.difference(groups):
            raise ValueError("every PathEvent surface must have a parameter group")
        if any(not group for group in groups.values()):
            raise ValueError("surface group names cannot be empty")
        if self.delay_line_count < 2 or self.delay_line_count & (
            self.delay_line_count - 1
        ):
            raise ValueError("grouped M4 delay-line count must be a power of two")
        if (
            not math.isfinite(self.transition_duration_s)
            or self.transition_duration_s <= 0.0
            or self.filter_order < 1
        ):
            raise ValueError("grouped M4 transition/filter configuration is invalid")
        object.__setattr__(self, "surface_group_by_id", groups)
        object.__setattr__(self, "direct_sample", int(self.direct_sample))

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(sorted(set(self.surface_group_by_id.values())))


@dataclass(frozen=True)
class GroupedPathParameters:
    """Effective per-boundary pressure adjustment plus fixed M4 late groups."""

    mixing_time_s: float
    reflection_adjustment_db_by_group: Mapping[str, float]
    target_rt60_s_by_hz: Mapping[float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.mixing_time_s) or self.mixing_time_s <= 0.0:
            raise ValueError("grouped M4 mixing time must be positive")
        object.__setattr__(
            self,
            "reflection_adjustment_db_by_group",
            _finite_mapping(
                "reflection_adjustment_db_by_group",
                self.reflection_adjustment_db_by_group,
            ),
        )
        object.__setattr__(
            self,
            "target_rt60_s_by_hz",
            _finite_band_mapping(
                "target_rt60_s_by_hz",
                self.target_rt60_s_by_hz,
            ),
        )

    @property
    def group_names(self) -> tuple[str, ...]:
        return tuple(self.reflection_adjustment_db_by_group)

    def gain_vector(self) -> np.ndarray:
        return np.asarray(
            [self.reflection_adjustment_db_by_group[name] for name in self.group_names],
            dtype=np.float64,
        )

    def with_gain_vector(self, values: Any) -> "GroupedPathParameters":
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != len(self.group_names) or not np.all(np.isfinite(vector)):
            raise ValueError("grouped path gain vector has invalid shape")
        return replace(
            self,
            reflection_adjustment_db_by_group={
                name: float(vector[index])
                for index, name in enumerate(self.group_names)
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mixing_time_s": float(self.mixing_time_s),
            "reflection_adjustment_db_by_group": dict(
                self.reflection_adjustment_db_by_group
            ),
            "target_rt60_s_by_hz": {
                f"{center:g}": float(value)
                for center, value in self.target_rt60_s_by_hz.items()
            },
        }


def _scaled_event(
    event: PathEvent,
    observation: GroupedPathObservation,
    parameters: GroupedPathParameters,
) -> PathEvent:
    adjustment_db = sum(
        parameters.reflection_adjustment_db_by_group[
            observation.surface_group_by_id[surface_id]
        ]
        for surface_id in event.surface_ids
    )
    scale = 10.0 ** (adjustment_db / 20.0)
    spectrum = event.gain_spectrum
    return replace(
        event,
        gain_spectrum=ComplexPathGainSpectrum(
            frequencies_hz=list(spectrum.frequencies_hz),
            real=[scale * value for value in spectrum.real],
            imag=[scale * value for value in spectrum.imag],
            provenance=(
                f"{spectrum.provenance}; M5 grouped pressure adjustment "
                f"{adjustment_db:.9g} dB"
            ),
            quantity=spectrum.quantity,
            interpolation=spectrum.interpolation,
        ),
    )


def render_grouped_path_observation(
    observation: GroupedPathObservation,
    parameters: GroupedPathParameters,
) -> PathEventFDNCouplingResult:
    """Render grouped physical paths through the actual M4 coupling."""

    if observation.group_names != parameters.group_names:
        raise ValueError("observation and grouped parameters must share group names")
    coherent = render_path_events(
        [
            _scaled_event(event, observation, parameters)
            for event in observation.event_set.events
        ],
        sample_rate_hz=observation.sample_rate,
        num_samples=observation.sample_count,
    )
    return couple_path_event_rir_with_fdn(
        coherent,
        observation.sample_rate,
        observation.direct_sample,
        parameters.target_rt60_s_by_hz,
        mixing_time_s=parameters.mixing_time_s,
        transition_duration_s=observation.transition_duration_s,
        delay_line_count=observation.delay_line_count,
        seed=observation.fdn_seed,
        filter_order=observation.filter_order,
    )


@dataclass(frozen=True)
class GroupedPathGainFit:
    parameters: GroupedPathParameters
    initial_cost: float
    final_cost: float
    evaluations: int
    success: bool
    message: str
    identifiability: LocalIdentifiabilityReport
    scaled_jacobian: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": M5_GROUPED_PATH_INVERSE_POLICY,
            "parameters": self.parameters.to_dict(),
            "initial_cost": float(self.initial_cost),
            "final_cost": float(self.final_cost),
            "evaluations": int(self.evaluations),
            "success": bool(self.success),
            "message": self.message,
            "local_identifiability": self.identifiability.to_dict(),
            "scaled_jacobian_shape": list(self.scaled_jacobian.shape),
        }


def _grouped_gain_residual(
    vector: np.ndarray,
    observations: Sequence[GroupedPathObservation],
    targets: Sequence[np.ndarray],
    template: GroupedPathParameters,
) -> np.ndarray:
    parameters = template.with_gain_vector(vector)
    residuals = []
    for observation, target in zip(observations, targets):
        candidate = render_grouped_path_observation(observation, parameters).rir
        scale = max(float(np.linalg.norm(target)), 1e-8)
        residuals.append((candidate - target) / scale)
    return np.concatenate(residuals)


def fit_grouped_path_gains(
    observations: Sequence[GroupedPathObservation],
    targets: Sequence[Any],
    initial_parameters: GroupedPathParameters,
    *,
    gain_bounds_db: tuple[float, float] = (-12.0, 3.0),
    maximum_evaluations: int = 80,
) -> GroupedPathGainFit:
    """Fit only grouped coherent-path gains while M4 topology/RT60 are fixed."""

    observations = tuple(observations)
    arrays = tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in targets)
    if not observations or len(observations) != len(arrays):
        raise ValueError("grouped observations and targets must have equal size")
    if any(
        observation.group_names != initial_parameters.group_names
        or target.size != observation.sample_count
        or not np.all(np.isfinite(target))
        for observation, target in zip(observations, arrays)
    ):
        raise ValueError("grouped observation, target, or parameter groups mismatch")
    lower, upper = (float(value) for value in gain_bounds_db)
    if (
        not math.isfinite(lower)
        or not math.isfinite(upper)
        or lower >= upper
        or maximum_evaluations < 1
    ):
        raise ValueError("grouped gain bounds/evaluation limit are invalid")
    initial = initial_parameters.gain_vector()
    if np.any(initial <= lower) or np.any(initial >= upper):
        raise ValueError("initial grouped gains must lie strictly inside bounds")
    initial_residual = _grouped_gain_residual(
        initial,
        observations,
        arrays,
        initial_parameters,
    )
    result = least_squares(
        _grouped_gain_residual,
        initial,
        bounds=(
            np.full(initial.size, lower, dtype=np.float64),
            np.full(initial.size, upper, dtype=np.float64),
        ),
        args=(observations, arrays, initial_parameters),
        x_scale=np.full(initial.size, upper - lower, dtype=np.float64),
        max_nfev=int(maximum_evaluations),
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    scaled_jacobian = np.asarray(result.jac, dtype=np.float64) * (upper - lower)
    identifiability = analyze_local_identifiability(
        scaled_jacobian,
        initial_parameters.group_names,
    )
    return GroupedPathGainFit(
        parameters=initial_parameters.with_gain_vector(result.x),
        initial_cost=float(0.5 * np.dot(initial_residual, initial_residual)),
        final_cost=float(result.cost),
        evaluations=int(result.nfev),
        success=bool(result.success),
        message=str(result.message),
        identifiability=identifiability,
        scaled_jacobian=scaled_jacobian,
    )


@dataclass(frozen=True)
class SpatialCalibrationSelection:
    """Auditable discrete selection using synchronized receiver evidence."""

    best_candidate_id: str
    candidate_reports: Mapping[str, Mapping[str, Any]]
    weights: CalibrationLossWeights

    def to_dict(self) -> dict[str, Any]:
        ordered = sorted(
            self.candidate_reports.items(),
            key=lambda item: float(item[1]["total"]),
        )
        second_total = float(ordered[1][1]["total"]) if len(ordered) > 1 else math.inf
        best_total = float(self.candidate_reports[self.best_candidate_id]["total"])
        return {
            "policy": M5_SPATIAL_CANDIDATE_PROFILE_POLICY,
            "best_candidate_id": self.best_candidate_id,
            "weights": self.weights.to_dict(),
            "candidate_reports": {
                key: dict(value) for key, value in self.candidate_reports.items()
            },
            "best_to_second_total_ratio": float(
                best_total / max(second_total, np.finfo(np.float64).tiny)
            ),
            "requires_synchronized_receivers": True,
        }


def select_spatial_calibration_candidate(
    measured: Any,
    candidates: Mapping[str, Any],
    sample_rate: int,
    measured_direct_samples: Sequence[int],
    *,
    synthetic_direct_samples_by_candidate: (Mapping[str, Sequence[int]] | None) = None,
    physical_first_samples: Sequence[int] | None = None,
    octave_centers_hz: Sequence[float] = (500.0, 1000.0, 2000.0),
    weights: CalibrationLossWeights | None = None,
) -> SpatialCalibrationSelection:
    """Profile scattering/directivity/late candidates on synchronized arrays."""

    reference = np.asarray(measured, dtype=np.float64)
    if (
        reference.ndim != 2
        or reference.shape[0] < 2
        or reference.shape[1] < 256
        or not np.all(np.isfinite(reference))
    ):
        raise ValueError(
            "M5.4 spatial calibration requires at least two synchronized receivers"
        )
    if sample_rate <= 0 or not candidates:
        raise ValueError("spatial candidate sample rate/grid is invalid")
    active_weights = weights or CalibrationLossWeights(
        multiresolution_stft=0.25,
        energy_decay=0.50,
        arrival_timing=0.25,
        octave_acoustics=0.50,
        spatial_coherence=2.00,
        causality=10.0,
        decay_regularization=1.0,
    )
    direct_by_candidate = dict(synthetic_direct_samples_by_candidate or {})
    reports = {}
    for candidate_id, value in sorted(candidates.items()):
        if not candidate_id:
            raise ValueError("spatial candidate ids cannot be empty")
        candidate = np.asarray(value, dtype=np.float64)
        if candidate.shape != reference.shape or not np.all(np.isfinite(candidate)):
            raise ValueError("every spatial candidate must match measured array shape")
        synthetic_direct = direct_by_candidate.get(
            candidate_id,
            measured_direct_samples,
        )
        report = analyze_rir_calibration_loss(
            reference,
            candidate,
            sample_rate,
            measured_direct_samples=measured_direct_samples,
            synthetic_direct_samples=synthetic_direct,
            physical_first_samples=physical_first_samples,
            weights=active_weights,
            fft_sizes=(256, 512),
            octave_centers_hz=octave_centers_hz,
        ).to_dict()
        if not report["diagnostics"]["spatial_coherence"]["evaluable"]:
            raise RuntimeError("synchronized spatial term unexpectedly unevaluable")
        reports[str(candidate_id)] = report
    best = min(reports, key=lambda key: float(reports[key]["total"]))
    return SpatialCalibrationSelection(
        best_candidate_id=best,
        candidate_reports=reports,
        weights=active_weights,
    )


__all__ = [
    "M5_GROUPED_PATH_INVERSE_POLICY",
    "M5_LOCAL_IDENTIFIABILITY_POLICY",
    "M5_SPATIAL_CANDIDATE_PROFILE_POLICY",
    "GroupedPathGainFit",
    "GroupedPathObservation",
    "GroupedPathParameters",
    "LocalIdentifiabilityReport",
    "SpatialCalibrationSelection",
    "analyze_local_identifiability",
    "fit_grouped_path_gains",
    "render_grouped_path_observation",
    "select_spatial_calibration_candidate",
]
