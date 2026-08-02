#!/usr/bin/env python3
"""Build the M3.9 opt-in fourth-order 3D FDTD validation report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from egs.rir_generation.phases.m3_wave_path.scripts.validate_corrected_fdtd_boundary import (
    _raised_cosine_interval,
)
from egs.rir_generation.phases.m2_impedance.scripts.validate_full_room_crossover import (
    _sampled_frequency_response,
)
from egs.rir_generation.phases.m3_wave_path.scripts.validate_oblique_fdtd_boundary import (
    _error_summary,
    _uniform_model,
)
from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
)
from puresound.audio.rir.physics.wave.fdtd import (
    BOUNDARIES,
    FDTDReferenceConfig,
    FOURTH_ORDER_CLOSURE_CFL_LIMIT,
    fdtd_discrete_plane_wave_reflection,
    simulate_fdtd_reference,
)
from puresound.audio.rir.physics.impedance.modes import (
    RectangularImpedanceBoundaryConfig,
)


REPORT_SCHEMA_VERSION = "puresound.fourth_order_fdtd_3d.v1"
_SCHEME = "face_quadratic_time_quadratic"
_DERIVATIVE_ORDER = 4
_CLOSURE = "third_order_one_sided"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the opt-in fourth-order 3D FDTD update with normal and "
            "oblique rigid-tangential plane modes."
        )
    )
    parser.add_argument(
        "--boundary-config",
        type=Path,
        default=Path(
            "egs/rir_generation/phases/m2_impedance/config/"
            "impedance_reference_glass_wool_14kgm3_100mm.json"
        ),
    )
    parser.add_argument(
        "--time-domain-report",
        type=Path,
        default=Path(
            "egs/rir_generation/phases/m3_wave_path/reports/"
            "higher_order_fdtd_time_domain_m3_8_report.json"
        ),
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--maximum-normal-complex-error",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--maximum-normal-phase-error-deg",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--maximum-oblique-complex-error",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--maximum-oblique-phase-error-deg",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def _rigid_model() -> FirstOrderRelaxationAdmittance:
    return FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.0,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )


def _reference_model() -> FirstOrderRelaxationAdmittance:
    return FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.35,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )


def _grid_shape_and_spacing(
    config: FDTDReferenceConfig,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    lx, ly, lz = config.room_dim_m
    nx = max(6, int(round(lx / config.grid_spacing_m)))
    ny = max(6, int(round(ly / config.grid_spacing_m)))
    nz = max(6, int(round(lz / config.grid_spacing_m)))
    return (nz, ny, nx), (lx / nx, ly / ny, lz / nz)


def _rigid_staggered_pressure_mode(
    num_cells: int,
    spacing_m: float,
    mode_index: int,
    *,
    near_wall_closure: str = "third_order_one_sided",
) -> tuple[np.ndarray, dict[str, float]]:
    if not 0 <= int(mode_index) < int(num_cells):
        raise ValueError("mode index must lie inside the pressure grid")
    gradient = np.zeros(
        (num_cells + 1, num_cells),
        dtype=np.float64,
    )
    if near_wall_closure == "fourth_order_mirrored":
        gradient[1, :3] = (-13.0 / 12.0, 9.0 / 8.0, -1.0 / 24.0)
        gradient[-2, -3:] = (
            1.0 / 24.0,
            -9.0 / 8.0,
            13.0 / 12.0,
        )
    elif near_wall_closure == "third_order_one_sided":
        gradient[1, :4] = (
            -23.0 / 24.0,
            7.0 / 8.0,
            1.0 / 8.0,
            -1.0 / 24.0,
        )
        gradient[-2, -4:] = (
            1.0 / 24.0,
            -1.0 / 8.0,
            -7.0 / 8.0,
            23.0 / 24.0,
        )
    else:
        raise ValueError("unsupported fourth-order closure")
    for face in range(2, num_cells - 1):
        gradient[face, face - 2 : face + 2] = (
            1.0 / 24.0,
            -9.0 / 8.0,
            9.0 / 8.0,
            -1.0 / 24.0,
        )
    divergence = np.zeros(
        (num_cells, num_cells + 1),
        dtype=np.float64,
    )
    if near_wall_closure == "fourth_order_mirrored":
        divergence[:, 1:-1] = -gradient[1:-1, :].T
    else:
        divergence[0, :4] = (
            -23.0 / 24.0,
            7.0 / 8.0,
            1.0 / 8.0,
            -1.0 / 24.0,
        )
        divergence[-1, -4:] = (
            1.0 / 24.0,
            -1.0 / 8.0,
            -7.0 / 8.0,
            23.0 / 24.0,
        )
        for cell in range(1, num_cells - 1):
            divergence[cell, cell - 1 : cell + 3] = (
                1.0 / 24.0,
                -9.0 / 8.0,
                9.0 / 8.0,
                -1.0 / 24.0,
            )
    eigenvalues, eigenvectors = np.linalg.eig(divergence @ gradient)
    order = np.argsort(np.abs(eigenvalues))
    selected = int(order[int(mode_index)])
    eigenvalue = complex(eigenvalues[selected])
    eigenvector = np.asarray(eigenvectors[:, selected], dtype=np.complex128)
    if abs(eigenvalue.imag) > 1e-10 or np.max(np.abs(eigenvector.imag)) > 1e-10:
        raise ValueError("rigid closure produced a complex pressure mode")
    vector = eigenvector.real
    vector /= max(float(np.max(np.abs(vector))), 1e-30)
    if vector[0] < 0.0:
        vector *= -1.0
    half_derivative_symbol = (
        math.sqrt(max(0.0, -float(eigenvalue.real)))
        / (2.0 * float(spacing_m))
    )
    lower = 0.0
    upper = 0.5 * math.pi
    for _iteration in range(64):
        midpoint = 0.5 * (lower + upper)
        sine = math.sin(midpoint)
        symbol = (
            sine
            / float(spacing_m)
            * (1.0 + sine**2 / 6.0)
        )
        if symbol < half_derivative_symbol:
            lower = midpoint
        else:
            upper = midpoint
    equivalent_wavenumber = (lower + upper) / float(spacing_m)
    return vector, {
        "dimensionless_pressure_laplacian_eigenvalue": float(
            eigenvalue.real
        ),
        "half_derivative_symbol_m_inv": half_derivative_symbol,
        "equivalent_wavenumber_rad_m": equivalent_wavenumber,
    }


def _plane_mode_weights(
    config: FDTDReferenceConfig,
    *,
    tangential_mode_y: int,
    tangential_mode_z: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shape, spacing = _grid_shape_and_spacing(config)
    nz, ny, nx = shape
    dx, dy, dz = spacing
    lx, ly, lz = config.room_dim_m
    source_x = int(
        np.clip(math.floor(config.source_position_m[0] / dx), 0, nx - 1)
    )
    receiver_x = int(
        np.clip(
            math.floor(config.receiver_position_m[0] / dx),
            0,
            nx - 1,
        )
    )
    mode_y, mode_y_metadata = _rigid_staggered_pressure_mode(
        ny,
        dy,
        tangential_mode_y,
    )
    mode_z, mode_z_metadata = _rigid_staggered_pressure_mode(
        nz,
        dz,
        tangential_mode_z,
    )
    mode_zy = mode_z[:, np.newaxis] * mode_y[np.newaxis, :]
    source = np.zeros(shape, dtype=np.float64)
    source[:, :, source_x] = mode_zy
    receiver = np.zeros(shape, dtype=np.float64)
    receiver[:, :, receiver_x] = mode_zy / float(np.sum(mode_zy**2))
    return source, receiver, {
        "grid_shape_zyx": list(shape),
        "grid_spacing_xyz_m": list(spacing),
        "source_plane_x_m": (source_x + 0.5) * dx,
        "receiver_plane_x_m": (receiver_x + 0.5) * dx,
        "tangential_mode_y": int(tangential_mode_y),
        "tangential_mode_z": int(tangential_mode_z),
        "tangential_wavenumber_y_rad_m": mode_y_metadata[
            "equivalent_wavenumber_rad_m"
        ],
        "tangential_wavenumber_z_rad_m": mode_z_metadata[
            "equivalent_wavenumber_rad_m"
        ],
        "tangential_mode_y_operator": mode_y_metadata,
        "tangential_mode_z_operator": mode_z_metadata,
        "continuous_cosine_wavenumber_y_rad_m": (
            float(tangential_mode_y) * math.pi / ly
        ),
        "continuous_cosine_wavenumber_z_rad_m": (
            float(tangential_mode_z) * math.pi / lz
        ),
    }


def _simulate_plane_mode(
    west_model,
    *,
    config: FDTDReferenceConfig,
    tangential_mode_y: int,
    tangential_mode_z: int,
    source_signal: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    source_weights, receiver_weights, mode = _plane_mode_weights(
        config,
        tangential_mode_y=tangential_mode_y,
        tangential_mode_z=tangential_mode_z,
    )
    rigid = _rigid_model()
    boundaries = {boundary: rigid for boundary in BOUNDARIES}
    boundaries["west"] = west_model
    result = simulate_fdtd_reference(
        config,
        boundary_admittance=boundaries,
        source_spatial_weights_zyx=source_weights,
        receiver_spatial_weights_zyx=receiver_weights,
        remove_output_dc=False,
        source_signal=source_signal,
    )
    return result.rir, {
        **mode,
        "sample_rate_hz": result.sample_rate_hz,
        "time_step_s": result.time_step_s,
        "interior_cfl_time_step_s": result.interior_cfl_time_step_s,
        "effective_interior_cfl": result.effective_interior_cfl,
        "boundary_time_step_limit_s": result.boundary_time_step_limit_s,
    }


def _plane_mode_cross_ratio(
    model,
    *,
    config: FDTDReferenceConfig,
    tangential_mode_y: int,
    tangential_mode_z: int,
    frequencies_hz: np.ndarray,
    window_s: tuple[float, float],
    window_ramp_s: float,
) -> dict[str, Any]:
    target, geometry = _simulate_plane_mode(
        model,
        config=config,
        tangential_mode_y=tangential_mode_y,
        tangential_mode_z=tangential_mode_z,
    )
    rigid, _rigid_geometry = _simulate_plane_mode(
        _rigid_model(),
        config=config,
        tangential_mode_y=tangential_mode_y,
        tangential_mode_z=tangential_mode_z,
    )
    reference_model = _reference_model()
    reference, _reference_geometry = _simulate_plane_mode(
        reference_model,
        config=config,
        tangential_mode_y=tangential_mode_y,
        tangential_mode_z=tangential_mode_z,
    )
    window = _raised_cosine_interval(
        num_samples=target.size,
        sample_rate_hz=geometry["sample_rate_hz"],
        start_s=float(window_s[0]),
        stop_s=float(window_s[1]),
        ramp_s=float(window_ramp_s),
    )
    target_difference = (target - reference) * window
    rigid_difference = (rigid - reference) * window
    target_response = _sampled_frequency_response(
        target_difference,
        geometry["sample_rate_hz"],
        frequencies_hz,
    )
    rigid_response = _sampled_frequency_response(
        rigid_difference,
        geometry["sample_rate_hz"],
        frequencies_hz,
    )
    measured = target_response / rigid_response
    kt_y = float(geometry["tangential_wavenumber_y_rad_m"])
    kt_z = float(geometry["tangential_wavenumber_z_rad_m"])
    target_reflections = []
    reference_reflections = []
    requested_cosines = []
    discrete_cosines = []
    dx, dy, dz = geometry["grid_spacing_xyz_m"]
    for frequency_hz in frequencies_hz:
        wavenumber = 2.0 * math.pi * float(frequency_hz) / 343.0
        tangential_ratio_y = kt_y / wavenumber
        tangential_ratio_z = kt_z / wavenumber
        cosine = math.sqrt(
            1.0 - tangential_ratio_y**2 - tangential_ratio_z**2
        )
        direction = (
            cosine,
            tangential_ratio_y,
            tangential_ratio_z,
        )
        target_discrete = fdtd_discrete_plane_wave_reflection(
            model,
            frequency_hz=float(frequency_hz),
            incident_direction_unit=direction,
            normal_axis=0,
            grid_spacing_xyz_m=(dx, dy, dz),
            time_step_s=geometry["time_step_s"],
            boundary_pressure_scheme=_SCHEME,
            spatial_derivative_order=_DERIVATIVE_ORDER,
        )
        reference_discrete = fdtd_discrete_plane_wave_reflection(
            reference_model,
            frequency_hz=float(frequency_hz),
            incident_direction_unit=direction,
            normal_axis=0,
            grid_spacing_xyz_m=(dx, dy, dz),
            time_step_s=geometry["time_step_s"],
            boundary_pressure_scheme=_SCHEME,
            spatial_derivative_order=_DERIVATIVE_ORDER,
        )
        target_reflections.append(target_discrete.reflection_coefficient)
        reference_reflections.append(
            reference_discrete.reflection_coefficient
        )
        requested_cosines.append(cosine)
        discrete_cosines.append(target_discrete.discrete_incidence_cosine)
    target_reflections = np.asarray(
        target_reflections,
        dtype=np.complex128,
    )
    reference_reflections = np.asarray(
        reference_reflections,
        dtype=np.complex128,
    )
    predicted = (
        target_reflections - reference_reflections
    ) / (
        1.0 - reference_reflections
    )
    relative_denominator = np.abs(rigid_response) / max(
        float(np.max(np.abs(rigid_response))),
        1e-30,
    )
    return {
        "geometry": geometry,
        "frequency_band_hz": [
            float(frequencies_hz[0]),
            float(frequencies_hz[-1]),
        ],
        "window_s": [float(window_s[0]), float(window_s[1])],
        "window_ramp_s": float(window_ramp_s),
        "requested_incidence_cosine_range": [
            float(min(requested_cosines)),
            float(max(requested_cosines)),
        ],
        "discrete_incidence_cosine_range": [
            float(min(discrete_cosines)),
            float(max(discrete_cosines)),
        ],
        "minimum_relative_cross_ratio_denominator": float(
            np.min(relative_denominator)
        ),
        "measured_vs_predicted_discrete_cross_ratio": _error_summary(
            predicted,
            measured,
            frequencies_hz,
        ),
    }


def _planned_time_step(
    config: FDTDReferenceConfig,
    west_model,
) -> float:
    _shape, spacing = _grid_shape_and_spacing(config)
    dx, dy, dz = spacing
    effective_cfl = min(
        float(config.cfl),
        FOURTH_ORDER_CLOSURE_CFL_LIMIT,
    )
    inverse_spacing_squared = (7.0 / 6.0) ** 2 * (
        dx**-2 + dy**-2 + dz**-2
    )
    interior = effective_cfl / (
        config.sound_speed_m_s * math.sqrt(inverse_spacing_squared)
    )
    boundary_rate = (
        config.sound_speed_m_s
        * float(west_model.maximum_normalized_admittance_bound)
        / dx
    )
    boundary = (
        0.9 / boundary_rate if boundary_rate > 0.0 else math.inf
    )
    return min(interior, boundary)


def _harmonic_amplitude(
    signal: np.ndarray,
    *,
    frequency_hz: float,
    time_step_s: float,
    start_s: float,
    stop_s: float,
) -> complex:
    start = max(0, round(float(start_s) / float(time_step_s)))
    stop = min(signal.size, round(float(stop_s) / float(time_step_s)))
    indices = np.arange(start, stop, dtype=np.float64)
    kernel = np.exp(
        -2j * math.pi * float(frequency_hz) * indices * time_step_s
    )
    return complex(2.0 * np.mean(signal[start:stop] * kernel))


def _oblique_harmonic_reflection_probe(
    model,
    *,
    frequencies_hz: tuple[float, ...],
) -> dict[str, Any]:
    measurements = []
    measured_reflections = []
    predicted_reflections = []
    for frequency_hz in frequencies_hz:
        duration_s = 1.2 if float(frequency_hz) <= 270.0 else 0.7
        common = {
            "room_dim_m": (3.0, 0.656, 0.36),
            "grid_spacing_m": 0.06,
            "duration_s": duration_s,
            "cfl": 0.92,
            "source_position_m": (2.4, 0.328, 0.18),
            "source_center_hz": float(frequency_hz),
            "source_delay_s": 0.0,
            "boundary_pressure_scheme": _SCHEME,
            "spatial_derivative_order": _DERIVATIVE_ORDER,
            "near_wall_closure": _CLOSURE,
        }
        first_config = FDTDReferenceConfig(
            receiver_position_m=(0.45, 0.328, 0.18),
            **common,
        )
        time_step = _planned_time_step(first_config, model)
        num_samples = int(math.ceil(first_config.duration_s / time_step))
        time_s = np.arange(num_samples, dtype=np.float64) * time_step
        ramp_duration_s = 0.15
        ramp = np.ones(num_samples, dtype=np.float64)
        ramp_samples = min(
            num_samples,
            max(2, round(ramp_duration_s / time_step)),
        )
        ramp[:ramp_samples] = 0.5 - 0.5 * np.cos(
            math.pi
            * np.arange(ramp_samples, dtype=np.float64)
            / float(ramp_samples - 1)
        )
        source_signal = ramp * np.sin(
            2.0 * math.pi * float(frequency_hz) * time_s
        )
        first_output, first_geometry = _simulate_plane_mode(
            model,
            config=first_config,
            tangential_mode_y=1,
            tangential_mode_z=0,
            source_signal=source_signal,
        )
        second_config = FDTDReferenceConfig(
            receiver_position_m=(1.05, 0.328, 0.18),
            **common,
        )
        second_output, second_geometry = _simulate_plane_mode(
            model,
            config=second_config,
            tangential_mode_y=1,
            tangential_mode_z=0,
            source_signal=source_signal,
        )
        first_amplitude = _harmonic_amplitude(
            first_output,
            frequency_hz=frequency_hz,
            time_step_s=time_step,
            start_s=duration_s - 0.20,
            stop_s=duration_s,
        )
        second_amplitude = _harmonic_amplitude(
            second_output,
            frequency_hz=frequency_hz,
            time_step_s=time_step,
            start_s=duration_s - 0.20,
            stop_s=duration_s,
        )
        kt_y = float(
            first_geometry["tangential_wavenumber_y_rad_m"]
        )
        wavenumber = 2.0 * math.pi * frequency_hz / 343.0
        cosine = math.sqrt(1.0 - (kt_y / wavenumber) ** 2)
        direction = (cosine, kt_y / wavenumber, 0.0)
        dx, dy, dz = first_geometry["grid_spacing_xyz_m"]
        predicted = fdtd_discrete_plane_wave_reflection(
            model,
            frequency_hz=frequency_hz,
            incident_direction_unit=direction,
            normal_axis=0,
            grid_spacing_xyz_m=(dx, dy, dz),
            time_step_s=time_step,
            boundary_pressure_scheme=_SCHEME,
            spatial_derivative_order=_DERIVATIVE_ORDER,
        )
        first_x = float(first_geometry["receiver_plane_x_m"])
        second_x = float(second_geometry["receiver_plane_x_m"])
        normal_wavenumber = predicted.normal_wavenumber_rad_m
        decomposition = np.asarray(
            [
                [
                    np.exp(1j * normal_wavenumber * first_x),
                    np.exp(-1j * normal_wavenumber * first_x),
                ],
                [
                    np.exp(1j * normal_wavenumber * second_x),
                    np.exp(-1j * normal_wavenumber * second_x),
                ],
            ],
            dtype=np.complex128,
        )
        incident, reflected = np.linalg.solve(
            decomposition,
            np.asarray(
                [first_amplitude, second_amplitude],
                dtype=np.complex128,
            ),
        )
        measured = complex(reflected / incident)
        measured_reflections.append(measured)
        predicted_reflections.append(predicted.reflection_coefficient)
        measurements.append(
            {
                "frequency_hz": float(frequency_hz),
                "duration_s": duration_s,
                "requested_incidence_cosine": cosine,
                "discrete_incidence_cosine": (
                    predicted.discrete_incidence_cosine
                ),
                "normal_wavenumber_rad_m": normal_wavenumber,
                "probe_x_m": [first_x, second_x],
                "decomposition_condition_number": float(
                    np.linalg.cond(decomposition)
                ),
                "measured_reflection": {
                    "real": float(measured.real),
                    "imag": float(measured.imag),
                },
                "predicted_reflection": {
                    "real": float(
                        predicted.reflection_coefficient.real
                    ),
                    "imag": float(
                        predicted.reflection_coefficient.imag
                    ),
                },
            }
        )
    measured_values = np.asarray(
        measured_reflections,
        dtype=np.complex128,
    )
    predicted_values = np.asarray(
        predicted_reflections,
        dtype=np.complex128,
    )
    frequency_values = np.asarray(frequencies_hz, dtype=np.float64)
    return {
        "method": (
            "steady harmonic drive; two-probe incident/reflected spatial "
            "decomposition between the west wall and source plane"
        ),
        "analysis_window": "last 0.20 seconds of each duration",
        "measurements": measurements,
        "measured_vs_predicted_discrete_reflection": _error_summary(
            predicted_values,
            measured_values,
            frequency_values,
        ),
    }


def main() -> None:
    args = _parse_args()
    time_domain_report = json.loads(
        args.time_domain_report.read_text(encoding="utf-8")
    )
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(
        args.boundary_config
    )
    model = _uniform_model(boundary_config)
    common = {
        "grid_spacing_m": 0.06,
        "sound_speed_m_s": 343.0,
        "air_density_kg_m3": 1.204,
        "cfl": 0.92,
        "boundary_pressure_scheme": _SCHEME,
        "spatial_derivative_order": _DERIVATIVE_ORDER,
        "near_wall_closure": _CLOSURE,
    }
    normal_config = FDTDReferenceConfig(
        room_dim_m=(20.0, 0.36, 0.36),
        duration_s=0.22,
        source_position_m=(15.0, 0.18, 0.18),
        receiver_position_m=(10.0, 0.18, 0.18),
        source_center_hz=240.0,
        source_delay_s=0.020,
        **common,
    )
    normal_frequencies = np.linspace(168.0, 300.0, 265)
    normal_center_s = 0.020 + 25.0 / 343.0
    normal = _plane_mode_cross_ratio(
        model,
        config=normal_config,
        tangential_mode_y=0,
        tangential_mode_z=0,
        frequencies_hz=normal_frequencies,
        window_s=(normal_center_s - 0.025, normal_center_s + 0.075),
        window_ramp_s=0.008,
    )
    oblique = _oblique_harmonic_reflection_probe(
        model,
        frequencies_hz=(270.0, 285.0, 300.0),
    )
    normal_metrics = normal[
        "measured_vs_predicted_discrete_cross_ratio"
    ]
    oblique_metrics = oblique[
        "measured_vs_predicted_discrete_reflection"
    ]
    normal_accepted = bool(
        normal_metrics["maximum_complex_error"]
        <= float(args.maximum_normal_complex_error)
        and normal_metrics["maximum_phase_error_deg"]
        <= float(args.maximum_normal_phase_error_deg)
    )
    oblique_accepted = bool(
        oblique_metrics["maximum_complex_error"]
        <= float(args.maximum_oblique_complex_error)
        and oblique_metrics["maximum_phase_error_deg"]
        <= float(args.maximum_oblique_phase_error_deg)
    )
    predecessor_accepted = bool(
        time_domain_report["acceptance"][
            "one_dimensional_time_domain_prototype_accepted"
        ]
    )
    accepted = bool(
        predecessor_accepted and normal_accepted and oblique_accepted
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "boundary_config": str(args.boundary_config),
        "boundary_reference_id": boundary_config.reference_id,
        "time_domain_report": str(args.time_domain_report),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_report"
        },
        "implementation": {
            "spatial_derivative_order": _DERIVATIVE_ORDER,
            "near_wall_closure": _CLOSURE,
            "boundary_pressure_scheme": _SCHEME,
            "three_axis_fourth_order_update": True,
            "six_face_one_sided_closure": True,
            "edge_corner_divergence_accumulation": True,
            "fourth_order_cfl_bound": True,
            "fourth_order_closure_cfl_limit": (
                FOURTH_ORDER_CLOSURE_CFL_LIMIT
            ),
            "plane_mode_source_receiver_projection": True,
            "controlled_harmonic_source_signal": True,
            "production_default_changed": False,
        },
        "normal_plane_mode": normal,
        "oblique_harmonic_plane_mode": oblique,
        "acceptance": {
            "m3_8_predecessor_accepted": predecessor_accepted,
            "maximum_normal_complex_error": float(
                args.maximum_normal_complex_error
            ),
            "maximum_normal_phase_error_deg": float(
                args.maximum_normal_phase_error_deg
            ),
            "maximum_oblique_complex_error": float(
                args.maximum_oblique_complex_error
            ),
            "maximum_oblique_phase_error_deg": float(
                args.maximum_oblique_phase_error_deg
            ),
            "normal_plane_mode_accepted": normal_accepted,
            "oblique_plane_mode_accepted": oblique_accepted,
            "opt_in_three_dimensional_reference_accepted": accepted,
            "production_default_accepted": False,
        },
        "scope": {
            "rigid_tangential_cosine_mode": True,
            "single_oblique_tangential_mode": True,
            "broadband_arbitrary_azimuth_validated": False,
            "full_room_crossover_rerun": False,
            "production_default_changed": False,
        },
        "decision": (
            "accept_opt_in_three_dimensional_reference"
            if accepted
            else "retain_as_unvalidated_three_dimensional_prototype"
        ),
        "next_action": (
            "run multi-axis/multi-azimuth oblique plane-mode holdouts and "
            "full-room attribution before changing any production default"
            if accepted
            else "inspect the 3D plane-mode closure mismatch"
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(
            report,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.output_report),
                "normal": normal_metrics,
                "oblique": oblique_metrics,
                "acceptance": report["acceptance"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
