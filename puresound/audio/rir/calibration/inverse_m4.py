"""Profile-based inverse calibration of the actual M4 PathEvent/FDN coupling.

M4 mixing time changes the prime-delay topology and is therefore treated as a
discrete outer profile. Coherent-path gain and octave RT60 targets remain in a
bounded continuous inner solve. This is an opt-in calibration reference; it
does not alter the production renderer or its defaults.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.optimize import least_squares

from puresound.audio.rir.render.coupling import (
    PathEventFDNCouplingResult,
    couple_path_event_rir_with_fdn,
)
from puresound.audio.rir.metrics import octave_band_rir, valid_octave_centers


M4_PARAMETER_PROFILE_INVERSE_POLICY = "puresound.m4_parameter_profile_inverse.v1"


def _finite_band_mapping(
    name: str,
    values: Mapping[float, float],
) -> dict[float, float]:
    result = dict(
        sorted((float(center), float(value)) for center, value in values.items())
    )
    if not result or any(
        not math.isfinite(center) or not math.isfinite(value)
        for center, value in result.items()
    ):
        raise ValueError(f"{name} must contain finite octave values")
    return result


@dataclass(frozen=True)
class M4InverseParameters:
    """First M5 mapping onto actual M4 coupling parameter groups."""

    mixing_time_s: float
    coherent_reflection_gain_db: float
    target_rt60_s_by_hz: Mapping[float, float]

    def __post_init__(self) -> None:
        if not math.isfinite(self.mixing_time_s) or self.mixing_time_s <= 0.0:
            raise ValueError("mixing_time_s must be finite and positive")
        if not math.isfinite(self.coherent_reflection_gain_db):
            raise ValueError("coherent reflection gain must be finite")
        rt60 = _finite_band_mapping(
            "target_rt60_s_by_hz",
            self.target_rt60_s_by_hz,
        )
        if any(value <= 0.0 for value in rt60.values()):
            raise ValueError("every M4 target RT60 must be positive")
        object.__setattr__(self, "target_rt60_s_by_hz", rt60)

    @property
    def centers_hz(self) -> tuple[float, ...]:
        return tuple(self.target_rt60_s_by_hz)

    def continuous_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.coherent_reflection_gain_db,
                *(self.target_rt60_s_by_hz[center] for center in self.centers_hz),
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_continuous_vector(
        cls,
        mixing_time_s: float,
        centers_hz: Sequence[float],
        values: Any,
    ) -> "M4InverseParameters":
        centers = tuple(float(center) for center in centers_hz)
        vector = np.asarray(values, dtype=np.float64).reshape(-1)
        if vector.size != 1 + len(centers) or not np.all(np.isfinite(vector)):
            raise ValueError("M4 continuous parameter vector has invalid shape")
        return cls(
            mixing_time_s=float(mixing_time_s),
            coherent_reflection_gain_db=float(vector[0]),
            target_rt60_s_by_hz={
                center: float(vector[1 + index]) for index, center in enumerate(centers)
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mixing_time_s": float(self.mixing_time_s),
            "coherent_reflection_gain_db": float(self.coherent_reflection_gain_db),
            "target_rt60_s_by_hz": {
                f"{center:g}": float(value)
                for center, value in self.target_rt60_s_by_hz.items()
            },
        }


@dataclass(frozen=True)
class M4InverseObservation:
    """Known coherent PathEvent response and fixed M4 topology identity."""

    observation_id: str
    path_event_rir: np.ndarray
    sample_rate: int
    direct_sample: int
    fdn_seed: int
    delay_line_count: int = 4
    transition_duration_s: float = 0.016
    direct_exclusion_ms: float = 0.75
    filter_order: int = 4
    fractional_arrival_tolerance_samples: int = 2

    def __post_init__(self) -> None:
        rir = np.asarray(self.path_event_rir, dtype=np.float64).reshape(-1)
        if not self.observation_id or self.sample_rate <= 0:
            raise ValueError("M4 observation id and sample rate are required")
        if rir.size < 256 or not np.all(np.isfinite(rir)):
            raise ValueError("M4 PathEvent RIR must contain finite samples")
        if not 0 <= int(self.direct_sample) < rir.size:
            raise ValueError("M4 direct sample lies outside the PathEvent RIR")
        causal_bound = max(
            0,
            int(self.direct_sample) - int(self.fractional_arrival_tolerance_samples),
        )
        if self.fractional_arrival_tolerance_samples < 0 or np.any(
            rir[:causal_bound] != 0.0
        ):
            raise ValueError("M4 PathEvent RIR violates fractional-arrival causality")
        if self.delay_line_count < 2 or self.delay_line_count & (
            self.delay_line_count - 1
        ):
            raise ValueError("M4 delay line count must be a power of two")
        if (
            not math.isfinite(self.transition_duration_s)
            or self.transition_duration_s <= 0.0
            or not math.isfinite(self.direct_exclusion_ms)
            or self.direct_exclusion_ms < 0.0
        ):
            raise ValueError("M4 transition and direct exclusion must be valid")
        if self.filter_order < 1:
            raise ValueError("M4 filter order must be positive")
        object.__setattr__(self, "path_event_rir", rir)
        object.__setattr__(self, "direct_sample", int(self.direct_sample))


@dataclass(frozen=True)
class M4ProfileObjectiveConfig:
    """Noise-aware feature residual used for every discrete topology profile."""

    waveform_weight: float = 0.10
    early_waveform_weight: float = 0.50
    broadband_decay_weight: float = 0.50
    octave_decay_weight: float = 1.00
    early_window_ms: float = 12.0
    energy_window_ms: float = 8.0
    noise_margin_db: float = 15.0

    def __post_init__(self) -> None:
        values = (
            self.waveform_weight,
            self.early_waveform_weight,
            self.broadband_decay_weight,
            self.octave_decay_weight,
            self.early_window_ms,
            self.energy_window_ms,
            self.noise_margin_db,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("M4 profile objective values must be non-negative")
        if (
            self.early_window_ms <= 0.0
            or self.energy_window_ms <= 0.0
            or self.octave_decay_weight <= 0.0
        ):
            raise ValueError("M4 profile objective requires energy and octave terms")

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy": "puresound.m4_profile_objective.v1",
            "implementation": "discrete_topology_plus_finite_difference_inner_solve",
            "waveform_weight": float(self.waveform_weight),
            "early_waveform_weight": float(self.early_waveform_weight),
            "broadband_decay_weight": float(self.broadband_decay_weight),
            "octave_decay_weight": float(self.octave_decay_weight),
            "early_window_ms": float(self.early_window_ms),
            "energy_window_ms": float(self.energy_window_ms),
            "noise_margin_db": float(self.noise_margin_db),
        }


@dataclass(frozen=True)
class M4ProfileBounds:
    """Continuous bounds; mixing-time candidates are supplied separately."""

    coherent_reflection_gain_db: tuple[float, float] = (-8.0, 8.0)
    target_rt60_s: tuple[float, float] = (0.20, 1.50)

    def vectors(self, band_count: int) -> tuple[np.ndarray, np.ndarray]:
        pairs = (self.coherent_reflection_gain_db, self.target_rt60_s)
        if band_count < 1 or any(
            len(pair) != 2
            or not all(math.isfinite(value) for value in pair)
            or pair[0] >= pair[1]
            for pair in pairs
        ):
            raise ValueError("M4 profile bounds must be finite and increasing")
        return (
            np.asarray(
                [
                    self.coherent_reflection_gain_db[0],
                    *([self.target_rt60_s[0]] * band_count),
                ],
                dtype=np.float64,
            ),
            np.asarray(
                [
                    self.coherent_reflection_gain_db[1],
                    *([self.target_rt60_s[1]] * band_count),
                ],
                dtype=np.float64,
            ),
        )


@dataclass(frozen=True)
class M4MixingProfilePoint:
    """One fixed-topology inner optimization result."""

    parameters: M4InverseParameters
    cost: float
    evaluations: int
    success: bool
    message: str
    scaled_jacobian_condition_number: float
    locally_full_rank: bool
    delay_lengths_by_observation: Mapping[str, tuple[int, ...]]

    def to_dict(self) -> dict[str, Any]:
        return {
            "parameters": self.parameters.to_dict(),
            "cost": float(self.cost),
            "evaluations": int(self.evaluations),
            "success": bool(self.success),
            "message": self.message,
            "local_identifiability": {
                "scaled_jacobian_condition_number": float(
                    self.scaled_jacobian_condition_number
                ),
                "full_column_rank": bool(self.locally_full_rank),
            },
            "delay_lengths_by_observation": {
                key: list(value)
                for key, value in self.delay_lengths_by_observation.items()
            },
        }


@dataclass(frozen=True)
class M4ProfileFit:
    """Selected M4 topology plus every rejected profile point."""

    best: M4MixingProfilePoint
    profiles: tuple[M4MixingProfilePoint, ...]
    objective: M4ProfileObjectiveConfig
    initial_parameters: M4InverseParameters

    def to_dict(self) -> dict[str, Any]:
        ordered = sorted(self.profiles, key=lambda point: point.cost)
        second_cost = ordered[1].cost if len(ordered) > 1 else math.inf
        return {
            "policy": M4_PARAMETER_PROFILE_INVERSE_POLICY,
            "objective": self.objective.to_dict(),
            "initial_parameters": self.initial_parameters.to_dict(),
            "best": self.best.to_dict(),
            "profiles": [point.to_dict() for point in self.profiles],
            "best_to_second_cost_ratio": float(
                self.best.cost / max(second_cost, np.finfo(np.float64).tiny)
            ),
            "warning": (
                "Discrete-profile selection is local to the supplied mixing-time grid."
            ),
        }


def _scaled_path_event(
    observation: M4InverseObservation,
    gain_db: float,
) -> np.ndarray:
    signal = observation.path_event_rir.copy()
    first_reflection = observation.direct_sample + int(
        round(observation.direct_exclusion_ms * 1e-3 * observation.sample_rate)
    )
    signal[first_reflection:] *= 10.0 ** (float(gain_db) / 20.0)
    return signal


def render_m4_inverse_observation(
    observation: M4InverseObservation,
    parameters: M4InverseParameters,
) -> PathEventFDNCouplingResult:
    """Render through the actual M4 equal-power PathEvent/FDN implementation."""

    valid = valid_octave_centers(
        observation.sample_rate,
        parameters.centers_hz,
    )
    if tuple(valid) != parameters.centers_hz:
        raise ValueError("M4 inverse octave centers must lie below Nyquist")
    return couple_path_event_rir_with_fdn(
        _scaled_path_event(
            observation,
            parameters.coherent_reflection_gain_db,
        ),
        observation.sample_rate,
        observation.direct_sample,
        parameters.target_rt60_s_by_hz,
        mixing_time_s=parameters.mixing_time_s,
        transition_duration_s=observation.transition_duration_s,
        delay_line_count=observation.delay_line_count,
        seed=observation.fdn_seed,
        filter_order=observation.filter_order,
    )


def _window_log_energy(signal: np.ndarray, window: int) -> np.ndarray:
    length = signal.size // window * window
    if length == 0:
        return np.zeros(0, dtype=np.float64)
    energy = np.mean(np.square(signal[:length].reshape(-1, window)), axis=1)
    return np.log(np.maximum(energy, 1e-18))


def _noise_energy(signal: np.ndarray, direct_sample_index: int) -> float:
    quiet = signal[: max(1, direct_sample_index // 2)]
    return float(max(np.median(np.square(quiet)), np.finfo(np.float64).tiny))


@dataclass(frozen=True)
class _M4TargetFeatures:
    target: np.ndarray
    scale: float
    early_end: int
    broadband_log: np.ndarray
    broadband_mask: np.ndarray
    octave_log: Mapping[float, np.ndarray]
    octave_masks: Mapping[float, np.ndarray]


def _log_energy_and_mask(
    signal: np.ndarray,
    direct_sample_index: int,
    window: int,
    noise_margin_db: float,
) -> tuple[np.ndarray, np.ndarray]:
    tail = signal[direct_sample_index:]
    values = _window_log_energy(tail, window)
    length = tail.size // window * window
    energy = np.mean(np.square(tail[:length].reshape(-1, window)), axis=1)
    threshold = _noise_energy(signal, direct_sample_index) * 10.0 ** (
        noise_margin_db / 10.0
    )
    mask = energy >= threshold
    if not np.any(mask):
        mask[int(np.argmax(energy))] = True
    return values, mask


def _target_features(
    observation: M4InverseObservation,
    target: np.ndarray,
    centers_hz: tuple[float, ...],
    objective: M4ProfileObjectiveConfig,
) -> _M4TargetFeatures:
    window = max(
        16,
        int(round(objective.energy_window_ms * 1e-3 * observation.sample_rate)),
    )
    broadband_log, broadband_mask = _log_energy_and_mask(
        target,
        observation.direct_sample,
        window,
        objective.noise_margin_db,
    )
    octave_log = {}
    octave_masks = {}
    for center in centers_hz:
        band = octave_band_rir(target, observation.sample_rate, center)
        values, mask = _log_energy_and_mask(
            band,
            observation.direct_sample,
            window,
            objective.noise_margin_db,
        )
        octave_log[center] = values
        octave_masks[center] = mask
    return _M4TargetFeatures(
        target=target,
        scale=max(float(np.linalg.norm(target)), 1e-8),
        early_end=min(
            target.size,
            observation.direct_sample
            + int(round(objective.early_window_ms * 1e-3 * observation.sample_rate)),
        ),
        broadband_log=broadband_log,
        broadband_mask=broadband_mask,
        octave_log=octave_log,
        octave_masks=octave_masks,
    )


def _profile_residual(
    vector: np.ndarray,
    mixing_time_s: float,
    centers_hz: tuple[float, ...],
    observations: Sequence[M4InverseObservation],
    targets: Sequence[_M4TargetFeatures],
    objective: M4ProfileObjectiveConfig,
) -> np.ndarray:
    parameters = M4InverseParameters.from_continuous_vector(
        mixing_time_s,
        centers_hz,
        vector,
    )
    residuals = []
    for observation, target_features in zip(observations, targets):
        candidate = render_m4_inverse_observation(observation, parameters).rir
        target = target_features.target
        if objective.waveform_weight > 0.0:
            residuals.append(
                math.sqrt(objective.waveform_weight)
                * (candidate - target)
                / target_features.scale
            )
        if objective.early_waveform_weight > 0.0:
            early = slice(observation.direct_sample, target_features.early_end)
            residuals.append(
                math.sqrt(objective.early_waveform_weight)
                * (candidate[early] - target[early])
                / target_features.scale
            )
        window = max(
            16,
            int(round(objective.energy_window_ms * 1e-3 * observation.sample_rate)),
        )
        if objective.broadband_decay_weight > 0.0:
            candidate_log = _window_log_energy(
                candidate[observation.direct_sample :],
                window,
            )
            mask = target_features.broadband_mask
            residuals.append(
                math.sqrt(objective.broadband_decay_weight)
                * (candidate_log[mask] - target_features.broadband_log[mask])
                / 4.0
            )
        octave_scale = math.sqrt(objective.octave_decay_weight / len(centers_hz))
        for center in centers_hz:
            candidate_band = octave_band_rir(
                candidate,
                observation.sample_rate,
                center,
            )
            candidate_log = _window_log_energy(
                candidate_band[observation.direct_sample :],
                window,
            )
            mask = target_features.octave_masks[center]
            residuals.append(
                octave_scale
                * (candidate_log[mask] - target_features.octave_log[center][mask])
                / 4.0
            )
    return np.concatenate(residuals)


def fit_m4_parameter_profile(
    observations: Sequence[M4InverseObservation],
    targets: Sequence[Any],
    initial_parameters: M4InverseParameters,
    mixing_time_candidates_s: Sequence[float],
    *,
    objective: M4ProfileObjectiveConfig | None = None,
    bounds: M4ProfileBounds | None = None,
    maximum_evaluations: int = 60,
) -> M4ProfileFit:
    """Profile discrete M4 topologies and fit continuous groups within each."""

    observations = tuple(observations)
    if not observations or len(observations) != len(targets):
        raise ValueError("M4 observations and targets must have equal non-zero length")
    centers = initial_parameters.centers_hz
    if any(
        tuple(valid_octave_centers(item.sample_rate, centers)) != centers
        for item in observations
    ):
        raise ValueError("M4 inverse centers must be valid for every observation")
    arrays = tuple(np.asarray(value, dtype=np.float64).reshape(-1) for value in targets)
    if any(
        target.shape != observation.path_event_rir.shape
        or not np.all(np.isfinite(target))
        for observation, target in zip(observations, arrays)
    ):
        raise ValueError("every M4 target must match its observation")
    candidates = tuple(sorted(set(float(value) for value in mixing_time_candidates_s)))
    if not candidates or any(
        not math.isfinite(value) or value <= 0.0 for value in candidates
    ):
        raise ValueError("M4 mixing-time profile requires positive finite candidates")
    if maximum_evaluations < 1:
        raise ValueError("maximum_evaluations must be positive")
    active_objective = objective or M4ProfileObjectiveConfig()
    active_bounds = bounds or M4ProfileBounds()
    lower, upper = active_bounds.vectors(len(centers))
    initial = initial_parameters.continuous_vector()
    if np.any(initial <= lower) or np.any(initial >= upper):
        raise ValueError("M4 initial continuous parameters must lie inside bounds")
    target_features = tuple(
        _target_features(
            observation,
            target,
            centers,
            active_objective,
        )
        for observation, target in zip(observations, arrays)
    )
    profiles = []
    for mixing_time_s in candidates:
        result = least_squares(
            _profile_residual,
            initial,
            bounds=(lower, upper),
            args=(
                mixing_time_s,
                centers,
                observations,
                target_features,
                active_objective,
            ),
            x_scale=upper - lower,
            max_nfev=int(maximum_evaluations),
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        scaled_jacobian = np.asarray(result.jac) * (upper - lower)
        singular_values = np.linalg.svd(scaled_jacobian, compute_uv=False)
        threshold = (
            max(scaled_jacobian.shape) * np.finfo(np.float64).eps * singular_values[0]
        )
        parameters = M4InverseParameters.from_continuous_vector(
            mixing_time_s,
            centers,
            result.x,
        )
        delays = {
            observation.observation_id: tuple(
                render_m4_inverse_observation(
                    observation,
                    parameters,
                ).design.delay_lengths_samples
            )
            for observation in observations
        }
        profiles.append(
            M4MixingProfilePoint(
                parameters=parameters,
                cost=float(result.cost),
                evaluations=int(result.nfev),
                success=bool(result.success),
                message=str(result.message),
                scaled_jacobian_condition_number=float(
                    min(
                        np.finfo(np.float64).max,
                        singular_values[0]
                        / max(
                            singular_values[-1],
                            singular_values[0] / np.finfo(np.float64).max,
                            np.finfo(np.float64).tiny,
                        ),
                    )
                ),
                locally_full_rank=bool(
                    singular_values.size == initial.size
                    and singular_values[-1] > threshold
                ),
                delay_lengths_by_observation=delays,
            )
        )
    best = min(profiles, key=lambda point: point.cost)
    return M4ProfileFit(
        best=best,
        profiles=tuple(profiles),
        objective=active_objective,
        initial_parameters=initial_parameters,
    )


__all__ = [
    "M4_PARAMETER_PROFILE_INVERSE_POLICY",
    "M4InverseObservation",
    "M4InverseParameters",
    "M4MixingProfilePoint",
    "M4ProfileBounds",
    "M4ProfileFit",
    "M4ProfileObjectiveConfig",
    "fit_m4_parameter_profile",
    "render_m4_inverse_observation",
]
