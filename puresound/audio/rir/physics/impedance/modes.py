"""Complex one-dimensional cavity modes with rational impedance boundaries."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.optimize import root

from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
    PassiveMultiPoleAdmittance,
    PassiveResonantAdmittance,
)


AdmittanceModel = (
    FirstOrderRelaxationAdmittance
    | PassiveMultiPoleAdmittance
    | PassiveResonantAdmittance
)
RECTANGULAR_IMPEDANCE_BOUNDARY_SCHEMA_VERSION = (
    "puresound.rectangular_impedance_boundary.v1"
)
RECTANGULAR_BOUNDARIES = (
    "west",
    "east",
    "south",
    "north",
    "floor",
    "ceiling",
)
AXIS_BOUNDARIES = (
    ("west", "east"),
    ("south", "north"),
    ("floor", "ceiling"),
)


@dataclass(frozen=True)
class ImpedanceCavityMode1D:
    mode_index: int
    complex_angular_frequency_rad_s: complex
    frequency_hz: float
    amplitude_decay_rate_per_s: float
    q_factor: float
    residual: float

    def metadata(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "complex_angular_frequency_rad_s": {
                "real": float(self.complex_angular_frequency_rad_s.real),
                "imag": float(self.complex_angular_frequency_rad_s.imag),
            },
        }


def solve_1d_impedance_cavity_modes(
    length_m: float,
    boundary_admittance: AdmittanceModel,
    *,
    sound_speed_m_s: float = 343.0,
    mode_indices: Iterable[int] = (1,),
) -> tuple[ImpedanceCavityMode1D, ...]:
    """Solve ``1 - Gamma(s)^2 exp(-2 s L/c) = 0`` for cavity poles.

    The two ends use the same locally reacting boundary.  The complex pole is
    ``s = -gamma + j*omega``; its imaginary part gives modal frequency and its
    negative real part gives amplitude decay.
    """
    length = float(length_m)
    sound_speed = float(sound_speed_m_s)
    if not math.isfinite(length) or length <= 0.0:
        raise ValueError("cavity length must be finite and positive")
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound speed must be finite and positive")
    if not isinstance(
        boundary_admittance,
        (
            FirstOrderRelaxationAdmittance,
            PassiveMultiPoleAdmittance,
            PassiveResonantAdmittance,
        ),
    ):
        raise TypeError("unsupported boundary admittance model")
    indices = tuple(int(value) for value in mode_indices)
    if not indices or any(value < 1 for value in indices):
        raise ValueError("mode indices must contain positive integers")

    modes: list[ImpedanceCavityMode1D] = []
    for mode_index in indices:
        rigid_frequency = (
            mode_index * sound_speed / (2.0 * length)
        )
        initial_reflection = boundary_admittance.reflection_coefficient(
            rigid_frequency
        )
        initial_decay = -sound_speed / length * math.log(
            max(abs(initial_reflection), 1e-9)
        )
        initial_pole = complex(
            -initial_decay,
            2.0 * math.pi * rigid_frequency,
        )

        def characteristic(values: np.ndarray) -> np.ndarray:
            pole = complex(float(values[0]), float(values[1]))
            reflection = (
                boundary_admittance.reflection_coefficient_laplace(pole)
            )
            value = 1.0 - reflection**2 * np.exp(
                -2.0 * pole * length / sound_speed
            )
            return np.asarray([value.real, value.imag], dtype=np.float64)

        solution = root(
            characteristic,
            np.asarray([initial_pole.real, initial_pole.imag]),
        )
        pole = complex(float(solution.x[0]), float(solution.x[1]))
        residual = float(np.linalg.norm(characteristic(solution.x)))
        if not solution.success or residual > 1e-7:
            raise RuntimeError(
                f"impedance cavity mode {mode_index} did not converge "
                f"(residual={residual:.3g})"
            )
        decay_rate = -float(pole.real)
        frequency = float(pole.imag / (2.0 * math.pi))
        if decay_rate <= 0.0 or frequency <= 0.0:
            raise RuntimeError(
                f"impedance cavity mode {mode_index} is not a stable "
                "positive-frequency pole"
            )
        modes.append(
            ImpedanceCavityMode1D(
                mode_index=mode_index,
                complex_angular_frequency_rad_s=pole,
                frequency_hz=frequency,
                amplitude_decay_rate_per_s=decay_rate,
                q_factor=float(pole.imag / (2.0 * decay_rate)),
                residual=residual,
            )
        )
    return tuple(modes)


def admittance_model_from_metadata(metadata: dict[str, Any]) -> AdmittanceModel:
    """Reconstruct one supported passive rational admittance model."""
    if not isinstance(metadata, dict):
        raise ValueError("admittance model metadata must be an object")
    model = metadata.get("model")
    if model == "first_order_relaxation_admittance":
        return FirstOrderRelaxationAdmittance(
            normalized_admittance_infinite=float(
                metadata["normalized_admittance_infinite"]
            ),
            normalized_admittance_relaxation=float(
                metadata["normalized_admittance_relaxation"]
            ),
            relaxation_frequency_hz=float(
                metadata["relaxation_frequency_hz"]
            ),
        )
    if model == "passive_multi_pole_admittance":
        return PassiveMultiPoleAdmittance(
            normalized_admittance_static=float(
                metadata["normalized_admittance_static"]
            ),
            pole_frequencies_hz=tuple(
                float(value) for value in metadata["pole_frequencies_hz"]
            ),
            normalized_admittance_lowpass=tuple(
                float(value)
                for value in metadata["normalized_admittance_lowpass"]
            ),
            normalized_admittance_highpass=tuple(
                float(value)
                for value in metadata["normalized_admittance_highpass"]
            ),
        )
    if model == "passive_resonant_admittance":
        return PassiveResonantAdmittance(
            normalized_admittance_static=float(
                metadata["normalized_admittance_static"]
            ),
            resonance_frequencies_hz=tuple(
                float(value)
                for value in metadata["resonance_frequencies_hz"]
            ),
            quality_factors=tuple(
                float(value) for value in metadata["quality_factors"]
            ),
            peak_normalized_admittances=tuple(
                float(value)
                for value in metadata["peak_normalized_admittances"]
            ),
        )
    raise ValueError(f"unsupported admittance model {model!r}")


@dataclass(frozen=True)
class RectangularImpedanceBoundaryConfig:
    """Explicit six-wall rational boundary assignment for research use."""

    reference_id: str
    boundaries: dict[str, AdmittanceModel]
    source: dict[str, Any]
    applicability: dict[str, Any]

    def __post_init__(self) -> None:
        if not self.reference_id:
            raise ValueError("impedance boundary reference_id is required")
        if set(self.boundaries) != set(RECTANGULAR_BOUNDARIES):
            raise ValueError(
                "impedance boundary config must define exactly six shoebox walls"
            )
        if not all(
            isinstance(
                model,
                (
                    FirstOrderRelaxationAdmittance,
                    PassiveMultiPoleAdmittance,
                    PassiveResonantAdmittance,
                ),
            )
            for model in self.boundaries.values()
        ):
            raise TypeError("all impedance boundaries must be passive rational models")
        if not isinstance(self.source, dict) or not self.source.get(
            "evidence_tier"
        ):
            raise ValueError("impedance boundary source.evidence_tier is required")
        if not isinstance(self.applicability, dict) or not self.applicability.get(
            "scope"
        ):
            raise ValueError("impedance boundary applicability.scope is required")

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": RECTANGULAR_IMPEDANCE_BOUNDARY_SCHEMA_VERSION,
            "reference_id": self.reference_id,
            "boundaries": {
                boundary: self.boundaries[boundary].metadata()
                for boundary in RECTANGULAR_BOUNDARIES
            },
            "source": dict(self.source),
            "applicability": dict(self.applicability),
        }

    @classmethod
    def from_json(
        cls,
        path: str | Path,
    ) -> "RectangularImpedanceBoundaryConfig":
        metadata = json.loads(Path(path).read_text(encoding="utf-8"))
        if (
            metadata.get("schema_version")
            != RECTANGULAR_IMPEDANCE_BOUNDARY_SCHEMA_VERSION
        ):
            raise ValueError("unsupported rectangular impedance boundary schema")
        raw_boundaries = metadata.get("boundaries")
        if not isinstance(raw_boundaries, dict):
            raise ValueError("impedance boundary metadata requires boundaries")
        return cls(
            reference_id=str(metadata.get("reference_id", "")),
            boundaries={
                boundary: admittance_model_from_metadata(
                    raw_boundaries[boundary]
                )
                for boundary in RECTANGULAR_BOUNDARIES
                if boundary in raw_boundaries
            },
            source=dict(metadata.get("source", {})),
            applicability=dict(metadata.get("applicability", {})),
        )


def _complex_sin_over_wavenumber(
    wavenumber_rad_m: complex,
    length_m: float,
) -> complex:
    wavenumber = complex(wavenumber_rad_m)
    length = float(length_m)
    argument = wavenumber * length
    if abs(argument) < 1e-5:
        return complex(
            length
            * (
                1.0
                - argument**2 / 6.0
                + argument**4 / 120.0
            )
        )
    return complex(np.sin(argument) / wavenumber)


def _axis_characteristic(
    wavenumber_rad_m: complex,
    complex_pole_rad_s: complex,
    length_m: float,
    minus_boundary: AdmittanceModel,
    plus_boundary: AdmittanceModel,
    sound_speed_m_s: float,
    boundary_scale: float = 1.0,
) -> complex:
    pole = complex(complex_pole_rad_s)
    wavenumber = complex(wavenumber_rad_m)
    length = float(length_m)
    scale = float(boundary_scale)
    h_minus = (
        scale
        * pole
        / float(sound_speed_m_s)
        * minus_boundary.normalized_admittance_laplace(pole)
    )
    h_plus = (
        scale
        * pole
        / float(sound_speed_m_s)
        * plus_boundary.normalized_admittance_laplace(pole)
    )
    return complex(
        length
        * (
            (h_minus * h_plus - wavenumber**2)
            * _complex_sin_over_wavenumber(wavenumber, length)
            + (h_minus + h_plus)
            * np.cos(wavenumber * length)
        )
    )


def _is_rigid_admittance(model: AdmittanceModel) -> bool:
    return bool(
        math.isclose(
            float(model.maximum_normalized_admittance_bound),
            0.0,
            rel_tol=0.0,
            abs_tol=1e-15,
        )
    )


def _axis_mode_value(
    position_m: float,
    wavenumber_rad_m: complex,
    minimum_log_derivative_per_m: complex,
) -> complex:
    position = float(position_m)
    wavenumber = complex(wavenumber_rad_m)
    return complex(
        np.cos(wavenumber * position)
        + minimum_log_derivative_per_m
        * _complex_sin_over_wavenumber(wavenumber, position)
    )


@dataclass(frozen=True)
class ImpedanceRoomMode3D:
    """One separable rectangular-room pole with complex spatial wavenumbers."""

    indices: tuple[int, int, int]
    complex_angular_frequency_rad_s: complex
    axis_wavenumbers_rad_m: tuple[complex, complex, complex]
    axis_minimum_log_derivatives_per_m: tuple[complex, complex, complex]
    room_dim_m: tuple[float, float, float]
    frequency_hz: float
    amplitude_decay_rate_per_s: float
    q_factor: float | None
    residual: float

    def eigenfunction_at(
        self,
        position_m: Iterable[float],
        *,
        normalize: bool = True,
    ) -> complex:
        position = tuple(float(value) for value in position_m)
        if len(position) != 3 or any(
            not 0.0 <= value <= length
            for value, length in zip(position, self.room_dim_m)
        ):
            raise ValueError("mode position must lie inside the room")
        value = complex(1.0)
        for coordinate, wavenumber, derivative in zip(
            position,
            self.axis_wavenumbers_rad_m,
            self.axis_minimum_log_derivatives_per_m,
        ):
            value *= _axis_mode_value(
                coordinate,
                wavenumber,
                derivative,
            )
        if not normalize:
            return value

        for axis, (length, wavenumber, derivative) in enumerate(
            zip(
                self.room_dim_m,
                self.axis_wavenumbers_rad_m,
                self.axis_minimum_log_derivatives_per_m,
            )
        ):
            nodes, weights = np.polynomial.legendre.leggauss(32)
            coordinates = 0.5 * float(length) * (nodes + 1.0)
            samples = np.asarray(
                [
                    _axis_mode_value(value, wavenumber, derivative)
                    for value in coordinates
                ],
                dtype=np.complex128,
            )
            norm = 0.5 * float(length) * np.sum(weights * samples**2)
            if abs(norm) <= 1e-14:
                raise RuntimeError(
                    f"mode {self.indices} axis {axis} has singular norm"
                )
            value /= np.sqrt(norm)
        return complex(value)

    def metadata(self) -> dict[str, Any]:
        def encoded(value: complex) -> dict[str, float]:
            return {
                "real": float(complex(value).real),
                "imag": float(complex(value).imag),
            }

        return {
            "indices": list(self.indices),
            "complex_angular_frequency_rad_s": encoded(
                self.complex_angular_frequency_rad_s
            ),
            "axis_wavenumbers_rad_m": [
                encoded(value) for value in self.axis_wavenumbers_rad_m
            ],
            "frequency_hz": float(self.frequency_hz),
            "amplitude_decay_rate_per_s": float(
                self.amplitude_decay_rate_per_s
            ),
            "q_factor": (
                float(self.q_factor) if self.q_factor is not None else None
            ),
            "residual": float(self.residual),
        }


def solve_rectangular_impedance_modes(
    room_dim_m: Iterable[float],
    boundary_config: RectangularImpedanceBoundaryConfig,
    *,
    mode_indices: Iterable[tuple[int, int, int]],
    sound_speed_m_s: float = 343.0,
    continuation_steps: int = 8,
) -> tuple[ImpedanceRoomMode3D, ...]:
    """Solve the separable 3D nonlinear rational-impedance eigenproblem.

    For each axis, with ``h=(s/c)y(s)``, the two Robin boundaries give

    ``(h- h+ - k²) sin(kL) + k(h- + h+) cos(kL) = 0``.

    The three complex spatial wavenumbers share one temporal pole and satisfy
    ``kx² + ky² + kz² + (s/c)² = 0``.  Continuation from a weak boundary tracks
    the requested rigid-wall mode branch as the full admittance is enabled.
    """
    dimensions = tuple(float(value) for value in room_dim_m)
    if len(dimensions) != 3 or any(
        not math.isfinite(value) or value <= 0.0 for value in dimensions
    ):
        raise ValueError("room dimensions must contain three positive values")
    sound_speed = float(sound_speed_m_s)
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound speed must be finite and positive")
    if int(continuation_steps) < 1:
        raise ValueError("continuation_steps must be positive")
    requested = tuple(
        tuple(int(component) for component in indices)
        for indices in mode_indices
    )
    if not requested or any(
        len(indices) != 3
        or any(component < 0 for component in indices)
        or indices == (0, 0, 0)
        for indices in requested
    ):
        raise ValueError(
            "mode indices must contain non-negative 3-tuples excluding DC"
        )
    if len(set(requested)) != len(requested):
        raise ValueError("mode indices must be unique")

    boundaries = boundary_config.boundaries
    axis_models = tuple(
        (boundaries[minus], boundaries[plus])
        for minus, plus in AXIS_BOUNDARIES
    )
    active_axes = tuple(
        axis
        for axis, (minus, plus) in enumerate(axis_models)
        if not (_is_rigid_admittance(minus) and _is_rigid_admittance(plus))
    )
    length_scale = float(np.cbrt(np.prod(dimensions)))
    modes: list[ImpedanceRoomMode3D] = []

    for indices in requested:
        rigid_wavenumbers = np.asarray(
            [
                index * math.pi / length
                for index, length in zip(indices, dimensions)
            ],
            dtype=np.complex128,
        )
        rigid_omega = sound_speed * float(
            np.sqrt(np.sum(np.square(rigid_wavenumbers.real)))
        )
        if not active_axes:
            pole = complex(0.0, rigid_omega)
            modes.append(
                ImpedanceRoomMode3D(
                    indices=indices,
                    complex_angular_frequency_rad_s=pole,
                    axis_wavenumbers_rad_m=tuple(
                        complex(value) for value in rigid_wavenumbers
                    ),
                    axis_minimum_log_derivatives_per_m=(0j, 0j, 0j),
                    room_dim_m=dimensions,
                    frequency_hz=rigid_omega / (2.0 * math.pi),
                    amplitude_decay_rate_per_s=0.0,
                    q_factor=None,
                    residual=0.0,
                )
            )
            continue

        probe_pole = complex(0.0, rigid_omega)
        initial_decay = 0.0
        for axis, ((minus, plus), length, index) in enumerate(
            zip(axis_models, dimensions, indices)
        ):
            if axis not in active_axes:
                continue
            norm = length if index == 0 else 0.5 * length
            pair_admittance = (
                minus.normalized_admittance_laplace(probe_pole).real
                + plus.normalized_admittance_laplace(probe_pole).real
            )
            initial_decay += (
                0.5 * sound_speed * max(float(pair_admittance), 0.0) / norm
            )
        first_scale = 1.0 / int(continuation_steps)
        initial_pole = complex(
            -max(initial_decay * first_scale, 1e-6),
            rigid_omega,
        )
        initial_wavenumbers = rigid_wavenumbers.copy()
        for axis in active_axes:
            if indices[axis] != 0:
                continue
            minus, plus = axis_models[axis]
            h_minus = (
                first_scale
                * initial_pole
                / sound_speed
                * minus.normalized_admittance_laplace(initial_pole)
            )
            h_plus = (
                first_scale
                * initial_pole
                / sound_speed
                * plus.normalized_admittance_laplace(initial_pole)
            )
            estimate = (
                h_minus
                + h_plus
                + h_minus * h_plus * dimensions[axis]
            ) / dimensions[axis]
            initial_wavenumbers[axis] = np.sqrt(estimate + 0j)

        def pack(
            pole: complex,
            wavenumbers: np.ndarray,
        ) -> np.ndarray:
            values = [pole]
            values.extend(wavenumbers[axis] for axis in active_axes)
            return np.asarray(
                [
                    component
                    for value in values
                    for component in (complex(value).real, complex(value).imag)
                ],
                dtype=np.float64,
            )

        def unpack(values: np.ndarray) -> tuple[complex, np.ndarray]:
            complex_values = [
                complex(float(values[index]), float(values[index + 1]))
                for index in range(0, values.size, 2)
            ]
            wavenumbers = rigid_wavenumbers.copy()
            for axis, value in zip(active_axes, complex_values[1:]):
                wavenumbers[axis] = value
            return complex_values[0], wavenumbers

        state = pack(initial_pole, initial_wavenumbers)
        final_residual = float("inf")
        for step in range(1, int(continuation_steps) + 1):
            boundary_scale = step / float(continuation_steps)

            def characteristic(values: np.ndarray) -> np.ndarray:
                pole, wavenumbers = unpack(values)
                equations = [
                    length_scale**2
                    * (
                        np.sum(np.square(wavenumbers))
                        + (pole / sound_speed) ** 2
                    )
                ]
                for axis in active_axes:
                    equations.append(
                        _axis_characteristic(
                            wavenumbers[axis],
                            pole,
                            dimensions[axis],
                            axis_models[axis][0],
                            axis_models[axis][1],
                            sound_speed,
                            boundary_scale,
                        )
                    )
                return np.asarray(
                    [
                        component
                        for value in equations
                        for component in (complex(value).real, complex(value).imag)
                    ],
                    dtype=np.float64,
                )

            solution = root(characteristic, state, method="hybr", tol=1e-10)
            final_residual = float(
                np.linalg.norm(characteristic(solution.x))
            )
            if not solution.success and final_residual > 1e-7:
                raise RuntimeError(
                    f"3D impedance mode {indices} failed at continuation "
                    f"{step}/{continuation_steps}: {solution.message}; "
                    f"residual={final_residual:.3g}"
                )
            state = solution.x

        pole, wavenumbers = unpack(state)
        if pole.imag < 0.0:
            pole = pole.conjugate()
            wavenumbers = np.conjugate(wavenumbers)
        decay_rate = -float(pole.real)
        frequency = float(pole.imag / (2.0 * math.pi))
        if decay_rate <= 0.0 or frequency <= 0.0:
            raise RuntimeError(
                f"3D impedance mode {indices} is not a stable "
                "positive-frequency pole"
            )
        minimum_log_derivatives = []
        for axis, (minus, _plus) in enumerate(axis_models):
            if axis not in active_axes:
                minimum_log_derivatives.append(0j)
                continue
            minimum_log_derivatives.append(
                pole
                / sound_speed
                * minus.normalized_admittance_laplace(pole)
            )
        modes.append(
            ImpedanceRoomMode3D(
                indices=indices,
                complex_angular_frequency_rad_s=pole,
                axis_wavenumbers_rad_m=tuple(
                    complex(value) for value in wavenumbers
                ),
                axis_minimum_log_derivatives_per_m=tuple(
                    complex(value) for value in minimum_log_derivatives
                ),
                room_dim_m=dimensions,
                frequency_hz=frequency,
                amplitude_decay_rate_per_s=decay_rate,
                q_factor=float(pole.imag / (2.0 * decay_rate)),
                residual=final_residual,
            )
        )
    return tuple(
        sorted(modes, key=lambda mode: (mode.frequency_hz, mode.indices))
    )


__all__ = [
    "AdmittanceModel",
    "ImpedanceCavityMode1D",
    "ImpedanceRoomMode3D",
    "RECTANGULAR_BOUNDARIES",
    "RECTANGULAR_IMPEDANCE_BOUNDARY_SCHEMA_VERSION",
    "RectangularImpedanceBoundaryConfig",
    "admittance_model_from_metadata",
    "solve_1d_impedance_cavity_modes",
    "solve_rectangular_impedance_modes",
]
