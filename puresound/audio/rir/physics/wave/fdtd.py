"""Small independent 3D acoustic FDTD reference for modal validation.

This solver is intentionally validation-oriented, not a dataset renderer. It
uses a staggered pressure/particle-velocity grid and locally reacting boundary
impedances derived from normal-incidence energy absorption. The implementation
is independent of the cosine-mode recurrence in ``hybrid_rir``.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, replace
from typing import Any, Iterable

import numpy as np

from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
    PassiveMultiPoleAdmittance,
    PassiveResonantAdmittance,
    characteristic_impedance_pa_s_m,
    digital_normalized_admittance_filter,
    impedance_from_absorption_and_phase,
    normal_incidence_absorption_coefficient,
)


BOUNDARIES = ("west", "east", "south", "north", "floor", "ceiling")
FOURTH_ORDER_CLOSURE_CFL_LIMIT = 0.25
FOURTH_ORDER_QUADRATIC_BOUNDARY_COURANT_LIMIT = 0.25
AdmittanceModel = (
    FirstOrderRelaxationAdmittance
    | PassiveMultiPoleAdmittance
    | PassiveResonantAdmittance
)


@dataclass(frozen=True)
class FDTDReferenceConfig:
    room_dim_m: tuple[float, float, float] = (3.0, 2.5, 2.0)
    grid_spacing_m: float = 0.10
    duration_s: float = 0.8
    sound_speed_m_s: float = 343.0
    air_density_kg_m3: float = 1.204
    cfl: float = 0.92
    source_position_m: tuple[float, float, float] = (0.73, 0.81, 0.69)
    receiver_position_m: tuple[float, float, float] = (2.17, 1.63, 1.31)
    source_center_hz: float = 180.0
    source_delay_s: float = 0.025
    boundary_pressure_scheme: str = "cell_center"
    spatial_derivative_order: int = 2
    near_wall_closure: str = "second_order"

    def __post_init__(self) -> None:
        if len(self.room_dim_m) != 3 or any(value <= 0.0 for value in self.room_dim_m):
            raise ValueError("room dimensions must be three positive values")
        if self.grid_spacing_m <= 0.0 or self.duration_s <= 0.0:
            raise ValueError("grid spacing and duration must be positive")
        if self.sound_speed_m_s <= 0.0 or self.air_density_kg_m3 <= 0.0:
            raise ValueError("sound speed and air density must be positive")
        if not 0.0 < self.cfl < 1.0:
            raise ValueError("cfl must be in (0, 1)")
        if self.boundary_pressure_scheme not in {
            "cell_center",
            "face_extrapolated",
            "face_time_extrapolated",
            "face_quadratic_time_quadratic",
            "face_quadratic_time_centered",
        }:
            raise ValueError("unsupported boundary pressure scheme")
        if int(self.spatial_derivative_order) not in {2, 4}:
            raise ValueError("spatial_derivative_order must be 2 or 4")
        if self.near_wall_closure not in {
            "second_order",
            "third_order_one_sided",
            "fourth_order_mirrored",
        }:
            raise ValueError("unsupported near-wall closure")
        if int(self.spatial_derivative_order) == 2:
            if self.near_wall_closure != "second_order":
                raise ValueError(
                    "second-order interior requires second_order closure"
                )
            if (
                self.boundary_pressure_scheme
                in {
                    "face_quadratic_time_quadratic",
                    "face_quadratic_time_centered",
                }
            ):
                raise ValueError(
                    "quadratic face/time scheme requires fourth-order interior"
                )
        elif self.near_wall_closure not in {
            "third_order_one_sided",
            "fourth_order_mirrored",
        }:
            raise ValueError(
                "fourth-order interior requires a fourth-order closure"
            )
        if (
            self.boundary_pressure_scheme
            == "face_quadratic_time_centered"
            and self.near_wall_closure != "fourth_order_mirrored"
        ):
            raise ValueError(
                "centered quadratic boundary requires "
                "fourth_order_mirrored closure"
            )
        if (
            self.near_wall_closure == "fourth_order_mirrored"
            and self.boundary_pressure_scheme
            != "face_quadratic_time_centered"
        ):
            raise ValueError(
                "fourth_order_mirrored closure requires the centered "
                "quadratic boundary"
            )
        for name, position in (
            ("source", self.source_position_m),
            ("receiver", self.receiver_position_m),
        ):
            if len(position) != 3 or any(
                not 0.0 < float(value) < float(limit)
                for value, limit in zip(position, self.room_dim_m)
            ):
                raise ValueError(f"{name} position must be strictly inside the room")


@dataclass(frozen=True)
class FDTDReferenceResult:
    rir: np.ndarray
    sample_rate_hz: float
    time_step_s: float
    grid_shape_zyx: tuple[int, int, int]
    grid_spacing_xyz_m: tuple[float, float, float]
    source_cell_zyx: tuple[int, int, int]
    receiver_cell_zyx: tuple[int, int, int]
    boundary_impedance_pa_s_m: dict[str, float | None]
    boundary_model: dict[str, dict[str, Any]]
    interior_cfl_time_step_s: float
    effective_interior_cfl: float
    boundary_time_step_limit_s: float | None
    config: FDTDReferenceConfig
    source_spatial_weights_used: bool = False
    receiver_spatial_weights_used: bool = False
    output_dc_removed: bool = True
    custom_source_signal_used: bool = False
    reciprocity_averaged: bool = False
    raw_reciprocity_nrmse: float | None = None

    def metadata(self) -> dict[str, Any]:
        return {
            "sample_rate_hz": float(self.sample_rate_hz),
            "time_step_s": float(self.time_step_s),
            "num_samples": int(self.rir.size),
            "grid_shape_zyx": list(self.grid_shape_zyx),
            "grid_spacing_xyz_m": list(self.grid_spacing_xyz_m),
            "source_cell_zyx": list(self.source_cell_zyx),
            "receiver_cell_zyx": list(self.receiver_cell_zyx),
            "boundary_impedance_pa_s_m": dict(self.boundary_impedance_pa_s_m),
            "boundary_model": dict(self.boundary_model),
            "interior_cfl_time_step_s": float(
                self.interior_cfl_time_step_s
            ),
            "effective_interior_cfl": float(
                self.effective_interior_cfl
            ),
            "boundary_time_step_limit_s": (
                None
                if self.boundary_time_step_limit_s is None
                else float(self.boundary_time_step_limit_s)
            ),
            "source_spatial_weights_used": bool(
                self.source_spatial_weights_used
            ),
            "receiver_spatial_weights_used": bool(
                self.receiver_spatial_weights_used
            ),
            "output_dc_removed": bool(self.output_dc_removed),
            "custom_source_signal_used": bool(
                self.custom_source_signal_used
            ),
            "reciprocity_averaged": bool(self.reciprocity_averaged),
            "raw_reciprocity_nrmse": (
                None
                if self.raw_reciprocity_nrmse is None
                else float(self.raw_reciprocity_nrmse)
            ),
            "config": asdict(self.config),
        }


@dataclass(frozen=True)
class FDTDDiscretePlaneWaveReflection:
    """One exact plane-wave reflection of a staggered-grid boundary equation."""

    reflection_coefficient: complex
    digital_normalized_admittance: complex
    requested_incidence_cosine: float
    discrete_incidence_cosine: float
    normal_wavenumber_rad_m: float
    frequency_hz: float
    normal_axis: int
    boundary_pressure_scheme: str
    spatial_derivative_order: int
    time_domain_implemented: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "reflection_coefficient": {
                "real": float(self.reflection_coefficient.real),
                "imag": float(self.reflection_coefficient.imag),
            },
            "digital_normalized_admittance": {
                "real": float(self.digital_normalized_admittance.real),
                "imag": float(self.digital_normalized_admittance.imag),
            },
            "requested_incidence_cosine": (
                self.requested_incidence_cosine
            ),
            "discrete_incidence_cosine": self.discrete_incidence_cosine,
            "normal_wavenumber_rad_m": self.normal_wavenumber_rad_m,
            "frequency_hz": self.frequency_hz,
            "normal_axis": self.normal_axis,
            "boundary_pressure_scheme": self.boundary_pressure_scheme,
            "spatial_derivative_order": self.spatial_derivative_order,
            "time_domain_implemented": self.time_domain_implemented,
        }


def _staggered_spatial_symbol(
    half_phase_rad: float,
    spacing_m: float,
    derivative_order: int,
) -> float:
    sine = math.sin(float(half_phase_rad))
    symbol = sine / float(spacing_m)
    if int(derivative_order) == 4:
        symbol *= 1.0 + sine**2 / 6.0
    return symbol


def _normal_wavenumber_from_symbol(
    target_symbol_m_inv: float,
    spacing_m: float,
    derivative_order: int,
) -> float:
    maximum_symbol = _staggered_spatial_symbol(
        0.5 * math.pi,
        spacing_m,
        derivative_order,
    )
    if float(target_symbol_m_inv) >= maximum_symbol:
        raise ValueError("normal wavenumber lies outside the discrete passband")
    lower = 0.0
    upper = 0.5 * math.pi
    for _iteration in range(64):
        midpoint = 0.5 * (lower + upper)
        if (
            _staggered_spatial_symbol(
                midpoint,
                spacing_m,
                derivative_order,
            )
            < float(target_symbol_m_inv)
        ):
            lower = midpoint
        else:
            upper = midpoint
    return (lower + upper) / float(spacing_m)


def fdtd_discrete_plane_wave_reflection(
    model: AdmittanceModel,
    *,
    frequency_hz: float,
    incident_direction_unit: Iterable[float],
    normal_axis: int,
    grid_spacing_xyz_m: Iterable[float],
    time_step_s: float,
    sound_speed_m_s: float = 343.0,
    boundary_pressure_scheme: str = "cell_center",
    spatial_derivative_order: int = 2,
) -> FDTDDiscretePlaneWaveReflection:
    """Return a staggered-grid boundary's exact plane-wave reflection.

    Tangential wavenumbers use the requested physical direction. The normal
    wavenumber is then solved from the discrete 3D dispersion relation. For a
    normal discrete characteristic admittance ``q_d``, digital wall
    admittance ``Y_d``, angular step ``Omega``, and normal phase step
    ``k_n*dx_n``, the pressure reflection referenced to the physical wall is

    ``(q_d - Y_d T exp(-j*Omega/2) S_i) /
      (q_d + Y_d T exp(-j*Omega/2) S_r)``.

    The half-time factor is caused by pressure/velocity staggering. The
    spatial factors ``S_i`` and ``S_r`` describe the first cell center or a
    causal wall-face extrapolation. The second-order predictor is
    ``1.5-0.5*z**-1``; the quadratic predictor is
    ``15/8-5/4*z**-1+3/8*z**-2``.

    ``spatial_derivative_order=2`` is the implemented 3D solver. Order four is
    a harmonic candidate using the fourth-order staggered derivative
    ``D4(k)/2 = sin(k*dx/2)/dx * (1 + sin(k*dx/2)**2/6)``. The returned
    ``time_domain_implemented`` flag distinguishes implemented equations from
    harmonic-only candidates.
    """
    frequency = float(frequency_hz)
    time_step = float(time_step_s)
    sound_speed = float(sound_speed_m_s)
    axis = int(normal_axis)
    pressure_scheme = str(boundary_pressure_scheme)
    derivative_order = int(spatial_derivative_order)
    spacing = np.asarray(
        [float(value) for value in grid_spacing_xyz_m],
        dtype=np.float64,
    )
    direction = np.asarray(
        [float(value) for value in incident_direction_unit],
        dtype=np.float64,
    )
    if not math.isfinite(frequency) or frequency <= 0.0:
        raise ValueError("frequency_hz must be finite and positive")
    if not math.isfinite(time_step) or time_step <= 0.0:
        raise ValueError("time_step_s must be finite and positive")
    if not math.isfinite(sound_speed) or sound_speed <= 0.0:
        raise ValueError("sound_speed_m_s must be finite and positive")
    if axis not in {0, 1, 2}:
        raise ValueError("normal_axis must be 0, 1, or 2")
    if pressure_scheme not in {
        "cell_center",
        "face_extrapolated",
        "face_time_extrapolated",
        "face_quadratic_time_quadratic",
        "face_quadratic_time_centered",
    }:
        raise ValueError("unsupported boundary pressure scheme")
    if derivative_order not in {2, 4}:
        raise ValueError("spatial_derivative_order must be 2 or 4")
    if (
        spacing.shape != (3,)
        or not np.all(np.isfinite(spacing))
        or np.any(spacing <= 0.0)
    ):
        raise ValueError("grid spacing must contain three positive values")
    direction_norm = float(np.linalg.norm(direction))
    if (
        direction.shape != (3,)
        or not np.all(np.isfinite(direction))
        or not math.isclose(
            direction_norm,
            1.0,
            rel_tol=0.0,
            abs_tol=1e-10,
        )
    ):
        raise ValueError("incident direction must be a finite unit vector")
    requested_cosine = abs(float(direction[axis]))
    if requested_cosine <= 0.0:
        raise ValueError("incident direction must have a normal component")
    omega_step = 2.0 * math.pi * frequency * time_step
    if omega_step >= math.pi:
        raise ValueError("frequency must lie below the FDTD Nyquist rate")
    physical_wavenumber = 2.0 * math.pi * frequency / sound_speed
    dispersion_target = (
        math.sin(0.5 * omega_step) / (sound_speed * time_step)
    ) ** 2
    tangential_term = sum(
        _staggered_spatial_symbol(
            0.5
            * physical_wavenumber
            * float(direction[index])
            * float(spacing[index]),
            float(spacing[index]),
            derivative_order,
        )
        ** 2
        for index in range(3)
        if index != axis
    )
    normal_term = dispersion_target - tangential_term
    tolerance = 1e-12 * max(dispersion_target, 1.0)
    if normal_term <= tolerance:
        raise ValueError(
            "requested direction has no propagating discrete normal mode"
        )
    normal_wavenumber = _normal_wavenumber_from_symbol(
        math.sqrt(normal_term),
        float(spacing[axis]),
        derivative_order,
    )
    half_normal_phase = (
        0.5 * normal_wavenumber * float(spacing[axis])
    )
    discrete_cosine = (
        sound_speed
        * time_step
        * _staggered_spatial_symbol(
            half_normal_phase,
            float(spacing[axis]),
            derivative_order,
        )
        / math.sin(0.5 * omega_step)
    )
    numerator, denominator = digital_normalized_admittance_filter(
        model,
        1.0 / time_step,
    )
    unit_delay = np.exp(-1j * omega_step)
    digital_admittance = complex(
        sum(
            value * unit_delay**index
            for index, value in enumerate(numerator)
        )
        / sum(
            value * unit_delay**index
            for index, value in enumerate(denominator)
        )
    )
    staggered_admittance = digital_admittance * np.exp(
        -0.5j * omega_step
    )
    if pressure_scheme == "cell_center":
        incident_pressure_factor = np.exp(1j * half_normal_phase)
        reflected_pressure_factor = np.exp(-1j * half_normal_phase)
        time_prediction_factor = complex(1.0, 0.0)
    elif pressure_scheme in {
        "face_extrapolated",
        "face_time_extrapolated",
    }:
        incident_pressure_factor = (
            1.5 * np.exp(1j * half_normal_phase)
            - 0.5 * np.exp(3j * half_normal_phase)
        )
        reflected_pressure_factor = (
            1.5 * np.exp(-1j * half_normal_phase)
            - 0.5 * np.exp(-3j * half_normal_phase)
        )
        time_prediction_factor = (
            1.5 - 0.5 * unit_delay
            if pressure_scheme == "face_time_extrapolated"
            else complex(1.0, 0.0)
        )
    else:
        incident_pressure_factor = (
            15.0 / 8.0 * np.exp(1j * half_normal_phase)
            - 5.0 / 4.0 * np.exp(3j * half_normal_phase)
            + 3.0 / 8.0 * np.exp(5j * half_normal_phase)
        )
        reflected_pressure_factor = (
            15.0 / 8.0 * np.exp(-1j * half_normal_phase)
            - 5.0 / 4.0 * np.exp(-3j * half_normal_phase)
            + 3.0 / 8.0 * np.exp(-5j * half_normal_phase)
        )
        if pressure_scheme == "face_quadratic_time_centered":
            time_prediction_factor = 0.5 * (
                1.0 + 1.0 / unit_delay
            )
        else:
            time_prediction_factor = (
                15.0 / 8.0
                - 5.0 / 4.0 * unit_delay
                + 3.0 / 8.0 * unit_delay**2
            )
    effective_admittance = (
        staggered_admittance * time_prediction_factor
    )
    reflection = (
        discrete_cosine
        - effective_admittance * incident_pressure_factor
    ) / (
        discrete_cosine
        + effective_admittance * reflected_pressure_factor
    )
    return FDTDDiscretePlaneWaveReflection(
        reflection_coefficient=complex(reflection),
        digital_normalized_admittance=digital_admittance,
        requested_incidence_cosine=requested_cosine,
        discrete_incidence_cosine=float(discrete_cosine),
        normal_wavenumber_rad_m=float(normal_wavenumber),
        frequency_hz=frequency,
        normal_axis=axis,
        boundary_pressure_scheme=pressure_scheme,
        spatial_derivative_order=derivative_order,
        time_domain_implemented=bool(
            derivative_order == 4
            or (
                derivative_order == 2
                and pressure_scheme
                in {
                    "cell_center",
                    "face_extrapolated",
                    "face_time_extrapolated",
                }
            )
        ),
    )


def absorption_to_impedance(
    absorption: float,
    air_density_kg_m3: float,
    sound_speed_m_s: float,
) -> float:
    """Convert absorption to the zero-phase, high-impedance real branch.

    This compatibility helper is sufficient for the original real-boundary
    FDTD reference.  Absorption alone is not a unique impedance; phase-aware
    callers must use ``acoustic_impedance.impedance_from_absorption_and_phase``.
    """
    alpha = float(absorption)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("absorption must be in [0, 1]")
    impedance = impedance_from_absorption_and_phase(
        alpha,
        reflection_phase_rad=0.0,
        air_density_kg_m3=air_density_kg_m3,
        sound_speed_m_s=sound_speed_m_s,
    )
    return float(impedance.real)


def impedance_to_absorption(
    impedance_pa_s_m: float,
    air_density_kg_m3: float,
    sound_speed_m_s: float,
) -> float:
    return normal_incidence_absorption_coefficient(
        impedance_pa_s_m,
        air_density_kg_m3,
        sound_speed_m_s,
    )


def _cell_index(
    position_m: Iterable[float],
    spacing_xyz: tuple[float, float, float],
    shape_zyx: tuple[int, int, int],
) -> tuple[int, int, int]:
    x, y, z = (float(value) for value in position_m)
    dx, dy, dz = spacing_xyz
    nz, ny, nx = shape_zyx
    return (
        int(np.clip(math.floor(z / dz), 0, nz - 1)),
        int(np.clip(math.floor(y / dy), 0, ny - 1)),
        int(np.clip(math.floor(x / dx), 0, nx - 1)),
    )


def _accumulate_fourth_order_mimetic_divergence(
    target: np.ndarray,
    velocity: np.ndarray,
    *,
    axis: int,
    inverse_spacing_m: float,
    boundary_pressure_scheme: str,
    near_wall_closure: str,
) -> None:
    """Accumulate one adjoint-consistent fourth-order divergence axis.

    The interior divergence is the negative transpose of the pressure
    gradient used at staggered velocity faces. Boundary-face fluxes are lifted
    with the transpose of the same spatial face-pressure interpolation used by
    the admittance condition. This makes the semi-discrete room operator
    reciprocal, including cells adjacent to faces, edges, and corners.
    """
    output = np.moveaxis(target, axis, -1)
    faces = np.moveaxis(velocity, axis, -1)
    if output.shape[-1] < 6 or faces.shape[-1] != output.shape[-1] + 1:
        raise ValueError("fourth-order divergence requires at least six cells")

    if boundary_pressure_scheme == "cell_center":
        face_weights = (1.0,)
    elif boundary_pressure_scheme in {
        "face_extrapolated",
        "face_time_extrapolated",
    }:
        face_weights = (1.5, -0.5)
    elif boundary_pressure_scheme in {
        "face_quadratic_time_quadratic",
        "face_quadratic_time_centered",
    }:
        face_weights = (15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0)
    else:
        raise ValueError("unsupported FDTD boundary pressure scheme")

    scale = float(inverse_spacing_m)
    west_velocity = faces[..., 0]
    east_velocity = faces[..., -1]
    for offset, coefficient in enumerate(face_weights):
        output[..., offset] -= scale * coefficient * west_velocity
        output[..., -1 - offset] += scale * coefficient * east_velocity

    if near_wall_closure == "fourth_order_mirrored":
        one_sided_gradient = (
            -13.0 / 12.0,
            9.0 / 8.0,
            -1.0 / 24.0,
        )
    else:
        one_sided_gradient = (
            -23.0 / 24.0,
            7.0 / 8.0,
            1.0 / 8.0,
            -1.0 / 24.0,
        )
    west_interior_velocity = faces[..., 1]
    east_interior_velocity = faces[..., -2]
    for offset, coefficient in enumerate(one_sided_gradient):
        output[..., offset] -= (
            scale * coefficient * west_interior_velocity
        )
        output[..., -1 - offset] += (
            scale * coefficient * east_interior_velocity
        )

    centered_velocity = faces[..., 2:-2]
    output[..., :-3] -= scale / 24.0 * centered_velocity
    output[..., 1:-2] += 9.0 * scale / 8.0 * centered_velocity
    output[..., 2:-1] -= 9.0 * scale / 8.0 * centered_velocity
    output[..., 3:] += scale / 24.0 * centered_velocity


def _legacy_fourth_order_divergence(
    target: np.ndarray,
    velocity_x: np.ndarray,
    velocity_y: np.ndarray,
    velocity_z: np.ndarray,
    *,
    spacing_xyz_m: tuple[float, float, float],
) -> None:
    """Apply the scoped M3.9 one-sided divergence without changing it."""
    dx, dy, dz = spacing_xyz_m
    target[:, :, 1:-1] = (
        9.0
        / 8.0
        * (velocity_x[:, :, 2:-1] - velocity_x[:, :, 1:-2])
        - 1.0
        / 24.0
        * (velocity_x[:, :, 3:] - velocity_x[:, :, :-3])
    ) / dx
    target[:, :, 0] = (
        -23.0 / 24.0 * velocity_x[:, :, 0]
        + 7.0 / 8.0 * velocity_x[:, :, 1]
        + 1.0 / 8.0 * velocity_x[:, :, 2]
        - 1.0 / 24.0 * velocity_x[:, :, 3]
    ) / dx
    target[:, :, -1] = (
        1.0 / 24.0 * velocity_x[:, :, -4]
        - 1.0 / 8.0 * velocity_x[:, :, -3]
        - 7.0 / 8.0 * velocity_x[:, :, -2]
        + 23.0 / 24.0 * velocity_x[:, :, -1]
    ) / dx
    target[:, 1:-1, :] += (
        9.0
        / 8.0
        * (velocity_y[:, 2:-1, :] - velocity_y[:, 1:-2, :])
        - 1.0
        / 24.0
        * (velocity_y[:, 3:, :] - velocity_y[:, :-3, :])
    ) / dy
    target[:, 0, :] += (
        -23.0 / 24.0 * velocity_y[:, 0, :]
        + 7.0 / 8.0 * velocity_y[:, 1, :]
        + 1.0 / 8.0 * velocity_y[:, 2, :]
        - 1.0 / 24.0 * velocity_y[:, 3, :]
    ) / dy
    target[:, -1, :] += (
        1.0 / 24.0 * velocity_y[:, -4, :]
        - 1.0 / 8.0 * velocity_y[:, -3, :]
        - 7.0 / 8.0 * velocity_y[:, -2, :]
        + 23.0 / 24.0 * velocity_y[:, -1, :]
    ) / dy
    target[1:-1, :, :] += (
        9.0
        / 8.0
        * (velocity_z[2:-1, :, :] - velocity_z[1:-2, :, :])
        - 1.0
        / 24.0
        * (velocity_z[3:, :, :] - velocity_z[:-3, :, :])
    ) / dz
    target[0, :, :] += (
        -23.0 / 24.0 * velocity_z[0, :, :]
        + 7.0 / 8.0 * velocity_z[1, :, :]
        + 1.0 / 8.0 * velocity_z[2, :, :]
        - 1.0 / 24.0 * velocity_z[3, :, :]
    ) / dz
    target[-1, :, :] += (
        1.0 / 24.0 * velocity_z[-4, :, :]
        - 1.0 / 8.0 * velocity_z[-3, :, :]
        - 7.0 / 8.0 * velocity_z[-2, :, :]
        + 23.0 / 24.0 * velocity_z[-1, :, :]
    ) / dz


def ricker_source(
    num_samples: int,
    time_step_s: float,
    center_hz: float,
    delay_s: float,
) -> np.ndarray:
    """Return the unit-peak Ricker pulse used by the FDTD pressure source."""
    if int(num_samples) < 1:
        raise ValueError("num_samples must be positive")
    if float(time_step_s) <= 0.0 or float(center_hz) <= 0.0:
        raise ValueError("time step and center frequency must be positive")
    if float(delay_s) < 0.0:
        raise ValueError("source delay must be non-negative")
    time_s = np.arange(num_samples, dtype=np.float64) * float(time_step_s)
    argument = math.pi * float(center_hz) * (time_s - float(delay_s))
    source = (1.0 - 2.0 * argument**2) * np.exp(-(argument**2))
    peak = float(np.max(np.abs(source)))
    return source / max(peak, 1e-12)


def simulate_fdtd_reference(
    config: FDTDReferenceConfig,
    boundary_absorption: dict[str, float] | float | None = None,
    *,
    boundary_admittance: (
        dict[str, AdmittanceModel] | AdmittanceModel | None
    ) = None,
    source_spatial_weights_zyx: np.ndarray | None = None,
    receiver_spatial_weights_zyx: np.ndarray | None = None,
    remove_output_dc: bool = True,
    source_signal: np.ndarray | None = None,
) -> FDTDReferenceResult:
    """Simulate one band-limited pressure RIR on a staggered 3D grid.

    Pass exactly one boundary representation. ``boundary_absorption`` retains
    the original frequency-independent real-impedance reference.
    ``boundary_admittance`` applies passive causal relaxation or resonant
    states per wall cell and active branch. Optional source and receiver
    spatial weights are validation-only hooks for plane-mode excitation and
    projection; the default remains one source and one receiver cell.
    ``remove_output_dc=False`` is likewise intended for transfer-ratio probes
    whose subtraction already cancels common offsets. ``source_signal`` can
    replace the Ricker pulse in controlled harmonic validation.
    """
    if (boundary_absorption is None) == (boundary_admittance is None):
        raise ValueError(
            "provide exactly one of boundary_absorption or boundary_admittance"
        )

    rho = float(config.air_density_kg_m3)
    c = float(config.sound_speed_m_s)
    z0 = characteristic_impedance_pa_s_m(rho, c)
    if boundary_admittance is not None:
        if isinstance(
            boundary_admittance,
            (
                FirstOrderRelaxationAdmittance,
                PassiveMultiPoleAdmittance,
                PassiveResonantAdmittance,
            ),
        ):
            admittance_models = {
                boundary: boundary_admittance for boundary in BOUNDARIES
            }
        else:
            missing = set(BOUNDARIES).difference(boundary_admittance)
            if missing:
                raise ValueError(f"missing boundary admittance: {sorted(missing)}")
            admittance_models = {
                boundary: boundary_admittance[boundary]
                for boundary in BOUNDARIES
            }
        if any(
            not isinstance(
                model,
                (
                    FirstOrderRelaxationAdmittance,
                    PassiveMultiPoleAdmittance,
                    PassiveResonantAdmittance,
                ),
            )
            for model in admittance_models.values()
        ):
            raise TypeError(
                "boundary_admittance values must be "
                "FirstOrderRelaxationAdmittance or "
                "PassiveMultiPoleAdmittance or "
                "PassiveResonantAdmittance"
            )
        impedance: dict[str, float | None] = {
            boundary: None for boundary in BOUNDARIES
        }
        boundary_model = {
            boundary: model.metadata()
            for boundary, model in admittance_models.items()
        }
    else:
        if isinstance(boundary_absorption, (int, float)):
            absorption = {
                boundary: float(boundary_absorption) for boundary in BOUNDARIES
            }
        else:
            missing = set(BOUNDARIES).difference(boundary_absorption)
            if missing:
                raise ValueError(f"missing boundary absorption: {sorted(missing)}")
            absorption = {
                boundary: float(boundary_absorption[boundary])
                for boundary in BOUNDARIES
            }
        for boundary, value in absorption.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{boundary} absorption must be in [0, 1]")
        impedance = {
            boundary: absorption_to_impedance(
                absorption[boundary],
                air_density_kg_m3=rho,
                sound_speed_m_s=c,
            )
            for boundary in BOUNDARIES
        }
        admittance_models = {
            boundary: FirstOrderRelaxationAdmittance(
                normalized_admittance_infinite=(
                    0.0
                    if math.isinf(impedance[boundary])
                    else z0 / max(float(impedance[boundary]), 1e-12)
                ),
                normalized_admittance_relaxation=0.0,
                relaxation_frequency_hz=1.0,
            )
            for boundary in BOUNDARIES
        }
        boundary_model = {
            boundary: {
                "model": "frequency_independent_real_impedance",
                "surface_impedance_pa_s_m": (
                    None
                    if math.isinf(impedance[boundary])
                    else float(impedance[boundary])
                ),
                "input_absorption": float(absorption[boundary]),
                "reflection_phase_rad": 0.0,
                "branch": "high_impedance",
            }
            for boundary in BOUNDARIES
        }

    lx, ly, lz = (float(value) for value in config.room_dim_m)
    minimum_cells = (
        6 if int(config.spatial_derivative_order) == 4 else 4
    )
    nx = max(
        minimum_cells,
        int(round(lx / float(config.grid_spacing_m))),
    )
    ny = max(
        minimum_cells,
        int(round(ly / float(config.grid_spacing_m))),
    )
    nz = max(
        minimum_cells,
        int(round(lz / float(config.grid_spacing_m))),
    )
    dx, dy, dz = lx / nx, ly / ny, lz / nz
    derivative_maximum = (
        7.0 / 6.0
        if int(config.spatial_derivative_order) == 4
        else 1.0
    )
    inverse_spacing_squared = derivative_maximum**2 * (
        dx**-2 + dy**-2 + dz**-2
    )
    requested_interior_cfl = (
        min(float(config.cfl), FOURTH_ORDER_CLOSURE_CFL_LIMIT)
        if int(config.spatial_derivative_order) == 4
        else float(config.cfl)
    )
    interior_cfl_dt = requested_interior_cfl / (
        float(config.sound_speed_m_s) * math.sqrt(inverse_spacing_squared)
    )
    maximum_admittance = {
        boundary: float(model.maximum_normalized_admittance_bound)
        for boundary, model in admittance_models.items()
    }
    boundary_rate = c * (
        max(maximum_admittance["west"], maximum_admittance["east"]) / dx
        + max(maximum_admittance["south"], maximum_admittance["north"]) / dy
        + max(maximum_admittance["floor"], maximum_admittance["ceiling"]) / dz
    )
    # The interior CFL condition alone does not bound an explicit admittance
    # term accumulated at edges/corners. Quadratic time prediction has a
    # larger high-frequency gain, so it needs a separately validated,
    # conservative boundary Courant cap.
    if config.boundary_pressure_scheme == (
        "face_quadratic_time_centered"
    ):
        boundary_dt = math.inf
    else:
        boundary_courant_limit = (
            FOURTH_ORDER_QUADRATIC_BOUNDARY_COURANT_LIMIT
            if (
                int(config.spatial_derivative_order) == 4
                and config.boundary_pressure_scheme
                == "face_quadratic_time_quadratic"
            )
            else 0.9
        )
        boundary_dt = (
            boundary_courant_limit / boundary_rate
            if boundary_rate > 0.0
            else math.inf
        )
    dt = min(interior_cfl_dt, boundary_dt)
    effective_interior_cfl = (
        c * dt * math.sqrt(inverse_spacing_squared)
    )
    num_samples = int(math.ceil(float(config.duration_s) / dt))
    if source_signal is None:
        source = ricker_source(
            num_samples,
            time_step_s=dt,
            center_hz=float(config.source_center_hz),
            delay_s=float(config.source_delay_s),
        )
    else:
        source = np.asarray(source_signal, dtype=np.float64)
        if (
            source.shape != (num_samples,)
            or not np.all(np.isfinite(source))
        ):
            raise ValueError(
                "source signal must be finite and match the simulated "
                "sample count"
            )

    pressure = np.zeros((nz, ny, nx), dtype=np.float64)
    velocity_x = np.zeros((nz, ny, nx + 1), dtype=np.float64)
    velocity_y = np.zeros((nz, ny + 1, nx), dtype=np.float64)
    velocity_z = np.zeros((nz + 1, ny, nx), dtype=np.float64)
    source_cell = _cell_index(
        config.source_position_m,
        (dx, dy, dz),
        (nz, ny, nx),
    )
    receiver_cell = _cell_index(
        config.receiver_position_m,
        (dx, dy, dz),
        (nz, ny, nx),
    )
    source_weights = None
    if source_spatial_weights_zyx is not None:
        source_weights = np.asarray(
            source_spatial_weights_zyx,
            dtype=np.float64,
        )
        if (
            source_weights.shape != pressure.shape
            or not np.all(np.isfinite(source_weights))
            or not np.any(source_weights != 0.0)
        ):
            raise ValueError(
                "source spatial weights must be finite, nonzero, and match "
                "the pressure grid"
            )
    receiver_weights = None
    if receiver_spatial_weights_zyx is not None:
        receiver_weights = np.asarray(
            receiver_spatial_weights_zyx,
            dtype=np.float64,
        )
        if (
            receiver_weights.shape != pressure.shape
            or not np.all(np.isfinite(receiver_weights))
            or not np.any(receiver_weights != 0.0)
        ):
            raise ValueError(
                "receiver spatial weights must be finite, nonzero, and match "
                "the pressure grid"
            )
    rir = np.zeros(num_samples, dtype=np.float64)
    velocity_scale_x = dt / (rho * dx)
    velocity_scale_y = dt / (rho * dy)
    velocity_scale_z = dt / (rho * dz)
    pressure_scale = rho * c**2 * dt
    boundary_shapes = {
        "west": (nz, ny),
        "east": (nz, ny),
        "south": (nz, nx),
        "north": (nz, nx),
        "floor": (ny, nx),
        "ceiling": (ny, nx),
    }
    boundary_state: dict[str, dict[str, Any]] = {}
    for boundary, model in admittance_models.items():
        static, poles, lowpass, highpass = model.relaxation_sections()
        if isinstance(model, FirstOrderRelaxationAdmittance):
            coefficients = (model.digital_lowpass_coefficients(1.0 / dt),)
        else:
            coefficients = model.digital_lowpass_coefficients(1.0 / dt)
        active = [
            index
            for index, (low_strength, high_strength) in enumerate(
                zip(lowpass, highpass)
            )
            if low_strength > 0.0 or high_strength > 0.0
        ]
        active_coefficients = [coefficients[index] for index in active]
        num_active = len(active)
        if isinstance(model, PassiveResonantAdmittance):
            resonant_coefficients = model.digital_biquad_coefficients(1.0 / dt)
            active_resonances = [
                index
                for index, peak in enumerate(
                    model.peak_normalized_admittances
                )
                if peak > 0.0
            ]
        else:
            resonant_coefficients = ()
            active_resonances = []
        active_resonant_coefficients = [
            resonant_coefficients[index] for index in active_resonances
        ]
        num_resonances = len(active_resonances)
        normalized_feedthrough = float(static)
        normalized_feedthrough += sum(
            float(lowpass[index]) * float(coefficients[index][0])
            + float(highpass[index])
            * (1.0 - float(coefficients[index][0]))
            for index in active
        )
        normalized_feedthrough += sum(
            float(resonant_coefficients[index][0])
            for index in active_resonances
        )
        if normalized_feedthrough < -1e-12:
            raise ValueError(
                f"{boundary} admittance has negative digital feedthrough"
            )
        boundary_state[boundary] = {
            "static": float(static),
            "normalized_feedthrough": max(
                0.0,
                normalized_feedthrough,
            ),
            "pole_frequencies_hz": np.asarray(
                [poles[index] for index in active],
                dtype=np.float64,
            ),
            "lowpass": np.asarray(
                [lowpass[index] for index in active],
                dtype=np.float64,
            ),
            "highpass": np.asarray(
                [highpass[index] for index in active],
                dtype=np.float64,
            ),
            "previous_pressure": np.zeros(
                boundary_shapes[boundary],
                dtype=np.float64,
            ),
            "relaxed_pressure": np.zeros(
                (num_active, *boundary_shapes[boundary]),
                dtype=np.float64,
            ),
            "b0": np.asarray(
                [value[0] for value in active_coefficients],
                dtype=np.float64,
            ),
            "b1": np.asarray(
                [value[1] for value in active_coefficients],
                dtype=np.float64,
            ),
            "a1": np.asarray(
                [value[2] for value in active_coefficients],
                dtype=np.float64,
            ),
            "resonance_frequencies_hz": np.asarray(
                [
                    model.resonance_frequencies_hz[index]
                    for index in active_resonances
                ]
                if isinstance(model, PassiveResonantAdmittance)
                else [],
                dtype=np.float64,
            ),
            "resonant_previous_pressure_1": np.zeros(
                boundary_shapes[boundary],
                dtype=np.float64,
            ),
            "resonant_previous_pressure_2": np.zeros(
                boundary_shapes[boundary],
                dtype=np.float64,
            ),
            "resonant_output_1": np.zeros(
                (num_resonances, *boundary_shapes[boundary]),
                dtype=np.float64,
            ),
            "resonant_output_2": np.zeros(
                (num_resonances, *boundary_shapes[boundary]),
                dtype=np.float64,
            ),
            "resonant_b0": np.asarray(
                [value[0] for value in active_resonant_coefficients],
                dtype=np.float64,
            ),
            "resonant_b1": np.asarray(
                [value[1] for value in active_resonant_coefficients],
                dtype=np.float64,
            ),
            "resonant_b2": np.asarray(
                [value[2] for value in active_resonant_coefficients],
                dtype=np.float64,
            ),
            "resonant_a1": np.asarray(
                [value[3] for value in active_resonant_coefficients],
                dtype=np.float64,
            ),
            "resonant_a2": np.asarray(
                [value[4] for value in active_resonant_coefficients],
                dtype=np.float64,
            ),
            "previous_face_pressure": np.zeros(
                boundary_shapes[boundary],
                dtype=np.float64,
            ),
            "previous_face_pressure_2": np.zeros(
                boundary_shapes[boundary],
                dtype=np.float64,
            ),
        }

    def sampled_boundary_pressure(boundary: str) -> np.ndarray:
        if boundary == "west":
            center = pressure[:, :, 0]
            adjacent = pressure[:, :, 1]
            second_adjacent = pressure[:, :, 2]
        elif boundary == "east":
            center = pressure[:, :, -1]
            adjacent = pressure[:, :, -2]
            second_adjacent = pressure[:, :, -3]
        elif boundary == "south":
            center = pressure[:, 0, :]
            adjacent = pressure[:, 1, :]
            second_adjacent = pressure[:, 2, :]
        elif boundary == "north":
            center = pressure[:, -1, :]
            adjacent = pressure[:, -2, :]
            second_adjacent = pressure[:, -3, :]
        elif boundary == "floor":
            center = pressure[0, :, :]
            adjacent = pressure[1, :, :]
            second_adjacent = pressure[2, :, :]
        elif boundary == "ceiling":
            center = pressure[-1, :, :]
            adjacent = pressure[-2, :, :]
            second_adjacent = pressure[-3, :, :]
        else:
            raise ValueError("unsupported FDTD boundary")
        scheme = config.boundary_pressure_scheme
        if scheme == "cell_center":
            return center
        if scheme == "face_quadratic_time_quadratic":
            face_pressure = (
                15.0 / 8.0 * center
                - 5.0 / 4.0 * adjacent
                + 3.0 / 8.0 * second_adjacent
            )
            state = boundary_state[boundary]
            predicted = (
                15.0 / 8.0 * face_pressure
                - 5.0 / 4.0 * state["previous_face_pressure"]
                + 3.0 / 8.0 * state["previous_face_pressure_2"]
            )
            state["previous_face_pressure_2"][...] = state[
                "previous_face_pressure"
            ]
            state["previous_face_pressure"][...] = face_pressure
            return predicted
        if scheme == "face_quadratic_time_centered":
            return (
                15.0 / 8.0 * center
                - 5.0 / 4.0 * adjacent
                + 3.0 / 8.0 * second_adjacent
            )
        face_pressure = 1.5 * center - 0.5 * adjacent
        if scheme == "face_extrapolated":
            return face_pressure
        state = boundary_state[boundary]
        predicted = (
            1.5 * face_pressure - 0.5 * state["previous_face_pressure"]
        )
        state["previous_face_pressure"][...] = face_pressure
        return predicted

    def boundary_velocity(
        boundary: str,
        boundary_pressure: np.ndarray,
    ) -> np.ndarray:
        state = boundary_state[boundary]
        static = float(state["static"])
        normalized_velocity_pressure = static * boundary_pressure
        if state["relaxed_pressure"].shape[0] > 0:
            coefficient_shape = (
                state["relaxed_pressure"].shape[0],
                *([1] * boundary_pressure.ndim),
            )
            relaxed_pressure = (
                state["b0"].reshape(coefficient_shape) * boundary_pressure
                + state["b1"].reshape(coefficient_shape)
                * state["previous_pressure"]
                - state["a1"].reshape(coefficient_shape)
                * state["relaxed_pressure"]
            )
            state["previous_pressure"][...] = boundary_pressure
            state["relaxed_pressure"][...] = relaxed_pressure
            normalized_velocity_pressure += np.sum(
                state["lowpass"].reshape(coefficient_shape) * relaxed_pressure
                + state["highpass"].reshape(coefficient_shape)
                * (boundary_pressure - relaxed_pressure),
                axis=0,
            )
        if state["resonant_output_1"].shape[0] > 0:
            resonant_shape = (
                state["resonant_output_1"].shape[0],
                *([1] * boundary_pressure.ndim),
            )
            resonant_output = (
                state["resonant_b0"].reshape(resonant_shape)
                * boundary_pressure
                + state["resonant_b1"].reshape(resonant_shape)
                * state["resonant_previous_pressure_1"]
                + state["resonant_b2"].reshape(resonant_shape)
                * state["resonant_previous_pressure_2"]
                - state["resonant_a1"].reshape(resonant_shape)
                * state["resonant_output_1"]
                - state["resonant_a2"].reshape(resonant_shape)
                * state["resonant_output_2"]
            )
            state["resonant_previous_pressure_2"][...] = state[
                "resonant_previous_pressure_1"
            ]
            state["resonant_previous_pressure_1"][...] = boundary_pressure
            state["resonant_output_2"][...] = state["resonant_output_1"]
            state["resonant_output_1"][...] = resonant_output
            normalized_velocity_pressure += np.sum(
                resonant_output,
                axis=0,
            )
        return normalized_velocity_pressure / z0

    def boundary_normalized_history(boundary: str) -> np.ndarray:
        state = boundary_state[boundary]
        history = np.zeros(
            boundary_shapes[boundary],
            dtype=np.float64,
        )
        if state["relaxed_pressure"].shape[0] > 0:
            coefficient_shape = (
                state["relaxed_pressure"].shape[0],
                *([1] * history.ndim),
            )
            relaxed_zero_input = (
                state["b1"].reshape(coefficient_shape)
                * state["previous_pressure"]
                - state["a1"].reshape(coefficient_shape)
                * state["relaxed_pressure"]
            )
            history += np.sum(
                (
                    state["lowpass"] - state["highpass"]
                ).reshape(coefficient_shape)
                * relaxed_zero_input,
                axis=0,
            )
        if state["resonant_output_1"].shape[0] > 0:
            resonant_shape = (
                state["resonant_output_1"].shape[0],
                *([1] * history.ndim),
            )
            resonant_zero_input = (
                state["resonant_b1"].reshape(resonant_shape)
                * state["resonant_previous_pressure_1"]
                + state["resonant_b2"].reshape(resonant_shape)
                * state["resonant_previous_pressure_2"]
                - state["resonant_a1"].reshape(resonant_shape)
                * state["resonant_output_1"]
                - state["resonant_a2"].reshape(resonant_shape)
                * state["resonant_output_2"]
            )
            history += np.sum(resonant_zero_input, axis=0)
        return history

    def quadratic_face_pressure(
        field: np.ndarray,
        boundary: str,
    ) -> np.ndarray:
        if boundary == "west":
            return (
                15.0 / 8.0 * field[:, :, 0]
                - 5.0 / 4.0 * field[:, :, 1]
                + 3.0 / 8.0 * field[:, :, 2]
            )
        if boundary == "east":
            return (
                15.0 / 8.0 * field[:, :, -1]
                - 5.0 / 4.0 * field[:, :, -2]
                + 3.0 / 8.0 * field[:, :, -3]
            )
        if boundary == "south":
            return (
                15.0 / 8.0 * field[:, 0, :]
                - 5.0 / 4.0 * field[:, 1, :]
                + 3.0 / 8.0 * field[:, 2, :]
            )
        if boundary == "north":
            return (
                15.0 / 8.0 * field[:, -1, :]
                - 5.0 / 4.0 * field[:, -2, :]
                + 3.0 / 8.0 * field[:, -3, :]
            )
        if boundary == "floor":
            return (
                15.0 / 8.0 * field[0, :, :]
                - 5.0 / 4.0 * field[1, :, :]
                + 3.0 / 8.0 * field[2, :, :]
            )
        if boundary == "ceiling":
            return (
                15.0 / 8.0 * field[-1, :, :]
                - 5.0 / 4.0 * field[-2, :, :]
                + 3.0 / 8.0 * field[-3, :, :]
            )
        raise ValueError("unsupported FDTD boundary")

    def subtract_quadratic_face_lift(
        field: np.ndarray,
        boundary: str,
        normalized_face_value: np.ndarray,
    ) -> None:
        weights = (15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0)
        axis_spacing = {
            "west": dx,
            "east": dx,
            "south": dy,
            "north": dy,
            "floor": dz,
            "ceiling": dz,
        }[boundary]
        scale = c * dt / axis_spacing
        for offset, coefficient in enumerate(weights):
            contribution = scale * coefficient * normalized_face_value
            if boundary == "west":
                field[:, :, offset] -= contribution
            elif boundary == "east":
                field[:, :, -1 - offset] -= contribution
            elif boundary == "south":
                field[:, offset, :] -= contribution
            elif boundary == "north":
                field[:, -1 - offset, :] -= contribution
            elif boundary == "floor":
                field[offset, :, :] -= contribution
            else:
                field[-1 - offset, :, :] -= contribution

    implicit_centered_boundary = (
        int(config.spatial_derivative_order) == 4
        and config.boundary_pressure_scheme
        == "face_quadratic_time_centered"
    )
    implicit_axis_eigenvectors: dict[
        int,
        tuple[np.ndarray, np.ndarray],
    ] = {}
    implicit_eigenvalues = np.zeros((nz, ny, nx), dtype=np.float64)
    if implicit_centered_boundary:
        weights = np.asarray(
            (15.0 / 8.0, -5.0 / 4.0, 3.0 / 8.0),
            dtype=np.float64,
        )
        axis_definitions = (
            (
                2,
                nx,
                dx,
                "west",
                "east",
                (1, 1, nx),
            ),
            (
                1,
                ny,
                dy,
                "south",
                "north",
                (1, ny, 1),
            ),
            (
                0,
                nz,
                dz,
                "floor",
                "ceiling",
                (nz, 1, 1),
            ),
        )
        for (
            axis,
            count,
            spacing_m,
            negative_boundary,
            positive_boundary,
            reshape,
        ) in axis_definitions:
            negative_scale = (
                c
                * dt
                * float(
                    boundary_state[negative_boundary][
                        "normalized_feedthrough"
                    ]
                )
                / (2.0 * spacing_m)
            )
            positive_scale = (
                c
                * dt
                * float(
                    boundary_state[positive_boundary][
                        "normalized_feedthrough"
                    ]
                )
                / (2.0 * spacing_m)
            )
            negative_values, negative_vectors = np.linalg.eigh(
                negative_scale * np.outer(weights, weights)
            )
            positive_values, positive_vectors = np.linalg.eigh(
                positive_scale
                * np.outer(weights[::-1], weights[::-1])
            )
            implicit_axis_eigenvectors[axis] = (
                negative_vectors,
                positive_vectors,
            )
            axis_values = np.zeros(count, dtype=np.float64)
            axis_values[:3] = negative_values
            axis_values[-3:] = positive_values
            implicit_eigenvalues += axis_values.reshape(reshape)
        implicit_eigenvalues += 1.0

    def transform_implicit_boundary_basis(
        field: np.ndarray,
        *,
        inverse: bool,
    ) -> None:
        for axis in (2, 1, 0):
            negative_vectors, positive_vectors = (
                implicit_axis_eigenvectors[axis]
            )
            moved = np.moveaxis(field, axis, -1)
            if inverse:
                moved[..., :3] = (
                    moved[..., :3] @ negative_vectors.T
                )
                moved[..., -3:] = (
                    moved[..., -3:] @ positive_vectors.T
                )
            else:
                moved[..., :3] = moved[..., :3] @ negative_vectors
                moved[..., -3:] = moved[..., -3:] @ positive_vectors

    fourth_order = int(config.spatial_derivative_order) == 4
    divergence = np.empty_like(pressure)
    for sample in range(num_samples):
        if fourth_order:
            velocity_x[:, :, 2:-2] -= velocity_scale_x * (
                9.0
                / 8.0
                * (pressure[:, :, 2:-1] - pressure[:, :, 1:-2])
                - 1.0
                / 24.0
                * (pressure[:, :, 3:] - pressure[:, :, :-3])
            )
            if config.near_wall_closure == "fourth_order_mirrored":
                velocity_x[:, :, 1] -= velocity_scale_x * (
                    -13.0 / 12.0 * pressure[:, :, 0]
                    + 9.0 / 8.0 * pressure[:, :, 1]
                    - 1.0 / 24.0 * pressure[:, :, 2]
                )
                velocity_x[:, :, -2] -= velocity_scale_x * (
                    1.0 / 24.0 * pressure[:, :, -3]
                    - 9.0 / 8.0 * pressure[:, :, -2]
                    + 13.0 / 12.0 * pressure[:, :, -1]
                )
            else:
                velocity_x[:, :, 1] -= velocity_scale_x * (
                    -23.0 / 24.0 * pressure[:, :, 0]
                    + 7.0 / 8.0 * pressure[:, :, 1]
                    + 1.0 / 8.0 * pressure[:, :, 2]
                    - 1.0 / 24.0 * pressure[:, :, 3]
                )
                velocity_x[:, :, -2] -= velocity_scale_x * (
                    1.0 / 24.0 * pressure[:, :, -4]
                    - 1.0 / 8.0 * pressure[:, :, -3]
                    - 7.0 / 8.0 * pressure[:, :, -2]
                    + 23.0 / 24.0 * pressure[:, :, -1]
                )
            velocity_y[:, 2:-2, :] -= velocity_scale_y * (
                9.0
                / 8.0
                * (pressure[:, 2:-1, :] - pressure[:, 1:-2, :])
                - 1.0
                / 24.0
                * (pressure[:, 3:, :] - pressure[:, :-3, :])
            )
            if config.near_wall_closure == "fourth_order_mirrored":
                velocity_y[:, 1, :] -= velocity_scale_y * (
                    -13.0 / 12.0 * pressure[:, 0, :]
                    + 9.0 / 8.0 * pressure[:, 1, :]
                    - 1.0 / 24.0 * pressure[:, 2, :]
                )
                velocity_y[:, -2, :] -= velocity_scale_y * (
                    1.0 / 24.0 * pressure[:, -3, :]
                    - 9.0 / 8.0 * pressure[:, -2, :]
                    + 13.0 / 12.0 * pressure[:, -1, :]
                )
            else:
                velocity_y[:, 1, :] -= velocity_scale_y * (
                    -23.0 / 24.0 * pressure[:, 0, :]
                    + 7.0 / 8.0 * pressure[:, 1, :]
                    + 1.0 / 8.0 * pressure[:, 2, :]
                    - 1.0 / 24.0 * pressure[:, 3, :]
                )
                velocity_y[:, -2, :] -= velocity_scale_y * (
                    1.0 / 24.0 * pressure[:, -4, :]
                    - 1.0 / 8.0 * pressure[:, -3, :]
                    - 7.0 / 8.0 * pressure[:, -2, :]
                    + 23.0 / 24.0 * pressure[:, -1, :]
                )
            velocity_z[2:-2, :, :] -= velocity_scale_z * (
                9.0
                / 8.0
                * (pressure[2:-1, :, :] - pressure[1:-2, :, :])
                - 1.0
                / 24.0
                * (pressure[3:, :, :] - pressure[:-3, :, :])
            )
            if config.near_wall_closure == "fourth_order_mirrored":
                velocity_z[1, :, :] -= velocity_scale_z * (
                    -13.0 / 12.0 * pressure[0, :, :]
                    + 9.0 / 8.0 * pressure[1, :, :]
                    - 1.0 / 24.0 * pressure[2, :, :]
                )
                velocity_z[-2, :, :] -= velocity_scale_z * (
                    1.0 / 24.0 * pressure[-3, :, :]
                    - 9.0 / 8.0 * pressure[-2, :, :]
                    + 13.0 / 12.0 * pressure[-1, :, :]
                )
            else:
                velocity_z[1, :, :] -= velocity_scale_z * (
                    -23.0 / 24.0 * pressure[0, :, :]
                    + 7.0 / 8.0 * pressure[1, :, :]
                    + 1.0 / 8.0 * pressure[2, :, :]
                    - 1.0 / 24.0 * pressure[3, :, :]
                )
                velocity_z[-2, :, :] -= velocity_scale_z * (
                    1.0 / 24.0 * pressure[-4, :, :]
                    - 1.0 / 8.0 * pressure[-3, :, :]
                    - 7.0 / 8.0 * pressure[-2, :, :]
                    + 23.0 / 24.0 * pressure[-1, :, :]
                )
        else:
            velocity_x[:, :, 1:-1] -= velocity_scale_x * (
                pressure[:, :, 1:] - pressure[:, :, :-1]
            )
            velocity_y[:, 1:-1, :] -= velocity_scale_y * (
                pressure[:, 1:, :] - pressure[:, :-1, :]
            )
            velocity_z[1:-1, :, :] -= velocity_scale_z * (
                pressure[1:, :, :] - pressure[:-1, :, :]
            )

        if implicit_centered_boundary:
            old_face_pressure = {
                boundary: quadratic_face_pressure(pressure, boundary).copy()
                for boundary in BOUNDARIES
            }
            boundary_history = {
                boundary: boundary_normalized_history(boundary)
                for boundary in BOUNDARIES
            }
            velocity_x[:, :, 0] = 0.0
            velocity_x[:, :, -1] = 0.0
            velocity_y[:, 0, :] = 0.0
            velocity_y[:, -1, :] = 0.0
            velocity_z[0, :, :] = 0.0
            velocity_z[-1, :, :] = 0.0
        else:
            velocity_x[:, :, 0] = -boundary_velocity(
                "west",
                sampled_boundary_pressure("west"),
            )
            velocity_x[:, :, -1] = boundary_velocity(
                "east",
                sampled_boundary_pressure("east"),
            )
            velocity_y[:, 0, :] = -boundary_velocity(
                "south",
                sampled_boundary_pressure("south"),
            )
            velocity_y[:, -1, :] = boundary_velocity(
                "north",
                sampled_boundary_pressure("north"),
            )
            velocity_z[0, :, :] = -boundary_velocity(
                "floor",
                sampled_boundary_pressure("floor"),
            )
            velocity_z[-1, :, :] = boundary_velocity(
                "ceiling",
                sampled_boundary_pressure("ceiling"),
            )

        if fourth_order:
            if config.near_wall_closure == "fourth_order_mirrored":
                divergence.fill(0.0)
                _accumulate_fourth_order_mimetic_divergence(
                    divergence,
                    velocity_x,
                    axis=2,
                    inverse_spacing_m=1.0 / dx,
                    boundary_pressure_scheme=(
                        config.boundary_pressure_scheme
                    ),
                    near_wall_closure=config.near_wall_closure,
                )
                _accumulate_fourth_order_mimetic_divergence(
                    divergence,
                    velocity_y,
                    axis=1,
                    inverse_spacing_m=1.0 / dy,
                    boundary_pressure_scheme=(
                        config.boundary_pressure_scheme
                    ),
                    near_wall_closure=config.near_wall_closure,
                )
                _accumulate_fourth_order_mimetic_divergence(
                    divergence,
                    velocity_z,
                    axis=0,
                    inverse_spacing_m=1.0 / dz,
                    boundary_pressure_scheme=(
                        config.boundary_pressure_scheme
                    ),
                    near_wall_closure=config.near_wall_closure,
                )
            else:
                _legacy_fourth_order_divergence(
                    divergence,
                    velocity_x,
                    velocity_y,
                    velocity_z,
                    spacing_xyz_m=(dx, dy, dz),
                )
        else:
            divergence[...] = (
                (velocity_x[:, :, 1:] - velocity_x[:, :, :-1]) / dx
                + (velocity_y[:, 1:, :] - velocity_y[:, :-1, :]) / dy
                + (velocity_z[1:, :, :] - velocity_z[:-1, :, :]) / dz
            )
        if implicit_centered_boundary:
            right_hand_side = pressure - pressure_scale * divergence
            if source_weights is None:
                right_hand_side[source_cell] += float(source[sample])
            else:
                right_hand_side += float(source[sample]) * source_weights
            for boundary in BOUNDARIES:
                feedthrough = float(
                    boundary_state[boundary][
                        "normalized_feedthrough"
                    ]
                )
                explicit_face_value = (
                    0.5
                    * feedthrough
                    * old_face_pressure[boundary]
                    + boundary_history[boundary]
                )
                subtract_quadratic_face_lift(
                    right_hand_side,
                    boundary,
                    explicit_face_value,
                )
            transform_implicit_boundary_basis(
                right_hand_side,
                inverse=False,
            )
            right_hand_side /= implicit_eigenvalues
            transform_implicit_boundary_basis(
                right_hand_side,
                inverse=True,
            )
            pressure[...] = right_hand_side
            midpoint_face_pressure = {
                boundary: 0.5
                * (
                    old_face_pressure[boundary]
                    + quadratic_face_pressure(pressure, boundary)
                )
                for boundary in BOUNDARIES
            }
            velocity_x[:, :, 0] = -boundary_velocity(
                "west",
                midpoint_face_pressure["west"],
            )
            velocity_x[:, :, -1] = boundary_velocity(
                "east",
                midpoint_face_pressure["east"],
            )
            velocity_y[:, 0, :] = -boundary_velocity(
                "south",
                midpoint_face_pressure["south"],
            )
            velocity_y[:, -1, :] = boundary_velocity(
                "north",
                midpoint_face_pressure["north"],
            )
            velocity_z[0, :, :] = -boundary_velocity(
                "floor",
                midpoint_face_pressure["floor"],
            )
            velocity_z[-1, :, :] = boundary_velocity(
                "ceiling",
                midpoint_face_pressure["ceiling"],
            )
        elif source_weights is None:
            pressure -= pressure_scale * divergence
            pressure[source_cell] += float(source[sample])
        else:
            pressure -= pressure_scale * divergence
            pressure += float(source[sample]) * source_weights
        if receiver_weights is None:
            rir[sample] = pressure[receiver_cell]
        else:
            rir[sample] = float(np.sum(pressure * receiver_weights))
        if (
            sample % 64 == 63
            and (
                not np.all(np.isfinite(pressure))
                or not np.all(np.isfinite(velocity_x))
                or not np.all(np.isfinite(velocity_y))
                or not np.all(np.isfinite(velocity_z))
            )
        ):
            raise FloatingPointError(
                "FDTD state became non-finite at sample "
                f"{sample}; reduce the CFL or inspect the boundary model"
            )

    if bool(remove_output_dc):
        rir -= float(np.mean(rir[-max(8, rir.size // 20) :]))
    return FDTDReferenceResult(
        rir=rir.astype(np.float64),
        sample_rate_hz=float(1.0 / dt),
        time_step_s=float(dt),
        grid_shape_zyx=(nz, ny, nx),
        grid_spacing_xyz_m=(dx, dy, dz),
        source_cell_zyx=source_cell,
        receiver_cell_zyx=receiver_cell,
        boundary_impedance_pa_s_m={
            boundary: (
                None
                if value is None or math.isinf(value)
                else float(value)
            )
            for boundary, value in impedance.items()
        },
        boundary_model=boundary_model,
        interior_cfl_time_step_s=float(interior_cfl_dt),
        effective_interior_cfl=float(effective_interior_cfl),
        boundary_time_step_limit_s=(
            None if math.isinf(boundary_dt) else float(boundary_dt)
        ),
        config=config,
        source_spatial_weights_used=source_weights is not None,
        receiver_spatial_weights_used=receiver_weights is not None,
        output_dc_removed=bool(remove_output_dc),
        custom_source_signal_used=source_signal is not None,
    )


def simulate_reciprocal_fdtd_reference(
    config: FDTDReferenceConfig,
    boundary_absorption: dict[str, float] | float | None = None,
    *,
    boundary_admittance: (
        dict[str, AdmittanceModel] | AdmittanceModel | None
    ) = None,
    source_spatial_weights_zyx: np.ndarray | None = None,
    receiver_spatial_weights_zyx: np.ndarray | None = None,
    remove_output_dc: bool = True,
    source_signal: np.ndarray | None = None,
) -> FDTDReferenceResult:
    """Return the symmetric part of the discrete Green transfer.

    The scoped M3.9 one-sided closure has accurate coarse-grid wall reflection
    magnitude/phase but is not a self-adjoint spatial operator near faces,
    edges, and corners. Running the exchanged source/receiver problem and
    averaging ``(G_sr + G_rs) / 2`` removes only that anti-reciprocal
    discretization component. Metadata retains its raw NRMSE so callers cannot
    mistake the operation for a natively reciprocal field solve.
    """
    if (source_spatial_weights_zyx is None) != (
        receiver_spatial_weights_zyx is None
    ):
        raise ValueError(
            "reciprocal spatial validation requires both source and "
            "receiver weights"
        )
    forward = simulate_fdtd_reference(
        config,
        boundary_absorption,
        boundary_admittance=boundary_admittance,
        source_spatial_weights_zyx=source_spatial_weights_zyx,
        receiver_spatial_weights_zyx=receiver_spatial_weights_zyx,
        remove_output_dc=remove_output_dc,
        source_signal=source_signal,
    )
    reverse_config = replace(
        config,
        source_position_m=config.receiver_position_m,
        receiver_position_m=config.source_position_m,
    )
    reverse = simulate_fdtd_reference(
        reverse_config,
        boundary_absorption,
        boundary_admittance=boundary_admittance,
        source_spatial_weights_zyx=receiver_spatial_weights_zyx,
        receiver_spatial_weights_zyx=source_spatial_weights_zyx,
        remove_output_dc=remove_output_dc,
        source_signal=source_signal,
    )
    difference = forward.rir - reverse.rir
    denominator = max(
        float(np.sqrt(np.mean(forward.rir**2))),
        float(np.sqrt(np.mean(reverse.rir**2))),
        1e-30,
    )
    raw_nrmse = float(
        np.sqrt(np.mean(difference**2)) / denominator
    )
    return replace(
        forward,
        rir=0.5 * (forward.rir + reverse.rir),
        reciprocity_averaged=True,
        raw_reciprocity_nrmse=raw_nrmse,
    )


__all__ = [
    "BOUNDARIES",
    "FOURTH_ORDER_CLOSURE_CFL_LIMIT",
    "FOURTH_ORDER_QUADRATIC_BOUNDARY_COURANT_LIMIT",
    "FDTDDiscretePlaneWaveReflection",
    "FDTDReferenceConfig",
    "FDTDReferenceResult",
    "absorption_to_impedance",
    "fdtd_discrete_plane_wave_reflection",
    "impedance_to_absorption",
    "ricker_source",
    "simulate_fdtd_reference",
    "simulate_reciprocal_fdtd_reference",
]
