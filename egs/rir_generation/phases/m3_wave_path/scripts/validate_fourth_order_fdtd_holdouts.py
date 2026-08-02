#!/usr/bin/env python3
"""Build the M3.10 multi-axis and edge/corner 3D FDTD holdout report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from egs.rir_generation.phases.m3_wave_path.scripts.validate_fourth_order_fdtd_3d import (
    _harmonic_amplitude,
    _rigid_model,
    _rigid_staggered_pressure_mode,
)
from egs.rir_generation.phases.m3_wave_path.scripts.validate_oblique_fdtd_boundary import (
    _error_summary,
    _uniform_model,
)
from puresound.audio.fdtd_reference import (
    BOUNDARIES,
    FOURTH_ORDER_CLOSURE_CFL_LIMIT,
    FDTDReferenceConfig,
    fdtd_discrete_plane_wave_reflection,
    simulate_reciprocal_fdtd_reference,
    simulate_fdtd_reference,
)
from puresound.audio.impedance_modes import (
    RectangularImpedanceBoundaryConfig,
)


REPORT_SCHEMA_VERSION = "puresound.fourth_order_fdtd_holdouts.v1"
_SCHEME = "face_quadratic_time_quadratic"
_CLOSURE = "third_order_one_sided"
_NEGATIVE_BOUNDARY = {0: "west", 1: "south", 2: "floor"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate fourth-order 3D FDTD across all normal axes, a "
            "two-tangential-mode azimuth, and face/edge/corner reciprocity."
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
        "--m3-9-report",
        type=Path,
        default=Path(
            "egs/rir_generation/phases/m3_wave_path/reports/"
            "fourth_order_fdtd_3d_m3_9_report.json"
        ),
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--maximum-plane-mode-complex-error",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--maximum-plane-mode-phase-error-deg",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-reciprocity-nrmse",
        type=float,
        default=1e-10,
    )
    return parser.parse_args()


def _shape_spacing(
    dimensions_xyz_m: tuple[float, float, float],
    grid_spacing_m: float,
) -> tuple[tuple[int, int, int], tuple[float, float, float]]:
    nx, ny, nz = (
        max(6, int(round(value / grid_spacing_m)))
        for value in dimensions_xyz_m
    )
    lx, ly, lz = dimensions_xyz_m
    return (nz, ny, nx), (lx / nx, ly / ny, lz / nz)


def _plane_weights(
    config: FDTDReferenceConfig,
    *,
    normal_axis: int,
    mode_indices_xyz: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shape, spacing = _shape_spacing(
        config.room_dim_m,
        config.grid_spacing_m,
    )
    nz, ny, nx = shape
    counts_xyz = (nx, ny, nz)
    vectors_xyz = []
    mode_metadata = []
    for axis, (count, delta, mode_index) in enumerate(
        zip(counts_xyz, spacing, mode_indices_xyz)
    ):
        if axis == normal_axis:
            if mode_index != 0:
                raise ValueError("normal-axis mode index must be zero")
            vector = np.ones(count, dtype=np.float64)
            metadata = {
                "equivalent_wavenumber_rad_m": 0.0,
                "dimensionless_pressure_laplacian_eigenvalue": 0.0,
            }
        else:
            vector, metadata = _rigid_staggered_pressure_mode(
                count,
                delta,
                mode_index,
                near_wall_closure=config.near_wall_closure,
            )
        vectors_xyz.append(vector)
        mode_metadata.append(metadata)
    mode_zyx = (
        vectors_xyz[2][:, np.newaxis, np.newaxis]
        * vectors_xyz[1][np.newaxis, :, np.newaxis]
        * vectors_xyz[0][np.newaxis, np.newaxis, :]
    )
    source_index_xyz = [
        int(
            np.clip(
                math.floor(
                    config.source_position_m[axis] / spacing[axis]
                ),
                0,
                counts_xyz[axis] - 1,
            )
        )
        for axis in range(3)
    ]
    receiver_index_xyz = [
        int(
            np.clip(
                math.floor(
                    config.receiver_position_m[axis] / spacing[axis]
                ),
                0,
                counts_xyz[axis] - 1,
            )
        )
        for axis in range(3)
    ]
    source = np.zeros(shape, dtype=np.float64)
    receiver = np.zeros(shape, dtype=np.float64)
    source_slice = [slice(None), slice(None), slice(None)]
    receiver_slice = [slice(None), slice(None), slice(None)]
    zyx_axis = 2 - normal_axis
    source_slice[zyx_axis] = source_index_xyz[normal_axis]
    receiver_slice[zyx_axis] = receiver_index_xyz[normal_axis]
    source[tuple(source_slice)] = mode_zyx[tuple(source_slice)]
    receiver_mode = mode_zyx[tuple(receiver_slice)]
    receiver[tuple(receiver_slice)] = receiver_mode / float(
        np.sum(receiver_mode**2)
    )
    return source, receiver, {
        "grid_shape_zyx": list(shape),
        "grid_spacing_xyz_m": list(spacing),
        "normal_axis": int(normal_axis),
        "mode_indices_xyz": list(mode_indices_xyz),
        "equivalent_wavenumber_xyz_rad_m": [
            float(item["equivalent_wavenumber_rad_m"])
            for item in mode_metadata
        ],
        "source_plane_m": (
            source_index_xyz[normal_axis] + 0.5
        )
        * spacing[normal_axis],
        "receiver_plane_m": (
            receiver_index_xyz[normal_axis] + 0.5
        )
        * spacing[normal_axis],
    }


def _planned_dt(
    config: FDTDReferenceConfig,
    model,
    normal_axis: int,
) -> float:
    _shape, spacing = _shape_spacing(
        config.room_dim_m,
        config.grid_spacing_m,
    )
    inverse_spacing_squared = (7.0 / 6.0) ** 2 * sum(
        value**-2 for value in spacing
    )
    interior = min(
        config.cfl,
        FOURTH_ORDER_CLOSURE_CFL_LIMIT,
    ) / (
        config.sound_speed_m_s * math.sqrt(inverse_spacing_squared)
    )
    boundary_rate = (
        config.sound_speed_m_s
        * float(model.maximum_normalized_admittance_bound)
        / spacing[normal_axis]
    )
    boundary = 0.9 / boundary_rate
    return min(interior, boundary)


def _simulate_mode(
    model,
    *,
    config: FDTDReferenceConfig,
    normal_axis: int,
    mode_indices_xyz: tuple[int, int, int],
    source_signal: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    source, receiver, metadata = _plane_weights(
        config,
        normal_axis=normal_axis,
        mode_indices_xyz=mode_indices_xyz,
    )
    boundaries = {boundary: _rigid_model() for boundary in BOUNDARIES}
    boundaries[_NEGATIVE_BOUNDARY[normal_axis]] = model
    result = simulate_fdtd_reference(
        config,
        boundary_admittance=boundaries,
        source_spatial_weights_zyx=source,
        receiver_spatial_weights_zyx=receiver,
        source_signal=source_signal,
        remove_output_dc=False,
    )
    return result.rir, {
        **metadata,
        "sample_rate_hz": result.sample_rate_hz,
        "time_step_s": result.time_step_s,
        "effective_interior_cfl": result.effective_interior_cfl,
    }


def _plane_mode_case(
    model,
    *,
    case_id: str,
    normal_axis: int,
    dimensions_xyz_m: tuple[float, float, float],
    mode_indices_xyz: tuple[int, int, int],
    frequency_hz: float,
    duration_s: float,
) -> dict[str, Any]:
    source_position = [
        0.5 * value for value in dimensions_xyz_m
    ]
    first_receiver = list(source_position)
    second_receiver = list(source_position)
    source_position[normal_axis] = 0.8 * dimensions_xyz_m[normal_axis]
    first_receiver[normal_axis] = 0.15 * dimensions_xyz_m[normal_axis]
    second_receiver[normal_axis] = 0.35 * dimensions_xyz_m[normal_axis]
    common = {
        "room_dim_m": dimensions_xyz_m,
        "grid_spacing_m": 0.06,
        "duration_s": duration_s,
        "cfl": 0.92,
        "source_position_m": tuple(source_position),
        "source_center_hz": frequency_hz,
        "source_delay_s": 0.0,
        "boundary_pressure_scheme": _SCHEME,
        "spatial_derivative_order": 4,
        "near_wall_closure": _CLOSURE,
    }
    first_config = FDTDReferenceConfig(
        receiver_position_m=tuple(first_receiver),
        **common,
    )
    time_step = _planned_dt(first_config, model, normal_axis)
    num_samples = int(math.ceil(duration_s / time_step))
    time_s = np.arange(num_samples, dtype=np.float64) * time_step
    ramp_samples = max(2, round(0.12 / time_step))
    ramp = np.ones(num_samples, dtype=np.float64)
    ramp[:ramp_samples] = 0.5 - 0.5 * np.cos(
        math.pi
        * np.arange(ramp_samples, dtype=np.float64)
        / float(ramp_samples - 1)
    )
    source_signal = ramp * np.sin(
        2.0 * math.pi * frequency_hz * time_s
    )
    first_output, first_geometry = _simulate_mode(
        model,
        config=first_config,
        normal_axis=normal_axis,
        mode_indices_xyz=mode_indices_xyz,
        source_signal=source_signal,
    )
    second_config = FDTDReferenceConfig(
        receiver_position_m=tuple(second_receiver),
        **common,
    )
    second_output, second_geometry = _simulate_mode(
        model,
        config=second_config,
        normal_axis=normal_axis,
        mode_indices_xyz=mode_indices_xyz,
        source_signal=source_signal,
    )
    analysis_start = duration_s - 0.20
    first_amplitude = _harmonic_amplitude(
        first_output,
        frequency_hz=frequency_hz,
        time_step_s=time_step,
        start_s=analysis_start,
        stop_s=duration_s,
    )
    second_amplitude = _harmonic_amplitude(
        second_output,
        frequency_hz=frequency_hz,
        time_step_s=time_step,
        start_s=analysis_start,
        stop_s=duration_s,
    )
    k = 2.0 * math.pi * frequency_hz / 343.0
    equivalent_k = first_geometry[
        "equivalent_wavenumber_xyz_rad_m"
    ]
    tangential_squared = sum(
        (equivalent_k[axis] / k) ** 2
        for axis in range(3)
        if axis != normal_axis
    )
    cosine = math.sqrt(1.0 - tangential_squared)
    direction = [
        equivalent_k[axis] / k for axis in range(3)
    ]
    direction[normal_axis] = cosine
    predicted = fdtd_discrete_plane_wave_reflection(
        model,
        frequency_hz=frequency_hz,
        incident_direction_unit=direction,
        normal_axis=normal_axis,
        grid_spacing_xyz_m=first_geometry["grid_spacing_xyz_m"],
        time_step_s=time_step,
        boundary_pressure_scheme=_SCHEME,
        spatial_derivative_order=4,
    )
    first_x = float(first_geometry["receiver_plane_m"])
    second_x = float(second_geometry["receiver_plane_m"])
    kn = predicted.normal_wavenumber_rad_m
    decomposition = np.asarray(
        [
            [np.exp(1j * kn * first_x), np.exp(-1j * kn * first_x)],
            [
                np.exp(1j * kn * second_x),
                np.exp(-1j * kn * second_x),
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
    metrics = _error_summary(
        np.asarray([predicted.reflection_coefficient]),
        np.asarray([measured]),
        np.asarray([frequency_hz]),
    )
    return {
        "case_id": case_id,
        "normal_axis": normal_axis,
        "mode_indices_xyz": list(mode_indices_xyz),
        "frequency_hz": frequency_hz,
        "duration_s": duration_s,
        "requested_incidence_cosine": cosine,
        "discrete_incidence_cosine": predicted.discrete_incidence_cosine,
        "geometry": first_geometry,
        "decomposition_condition_number": float(
            np.linalg.cond(decomposition)
        ),
        "measured_reflection": {
            "real": float(measured.real),
            "imag": float(measured.imag),
        },
        "predicted_reflection": {
            "real": float(predicted.reflection_coefficient.real),
            "imag": float(predicted.reflection_coefficient.imag),
        },
        "error": metrics,
    }


def _reciprocity_case(
    model,
    *,
    case_id: str,
    source_position_m: tuple[float, float, float],
    receiver_position_m: tuple[float, float, float],
) -> dict[str, Any]:
    common = {
        "room_dim_m": (1.2, 1.0, 0.8),
        "grid_spacing_m": 0.10,
        "duration_s": 0.08,
        "source_center_hz": 240.0,
        "source_delay_s": 0.010,
        "boundary_pressure_scheme": _SCHEME,
        "spatial_derivative_order": 4,
        "near_wall_closure": _CLOSURE,
    }
    forward = simulate_fdtd_reference(
        FDTDReferenceConfig(
            source_position_m=source_position_m,
            receiver_position_m=receiver_position_m,
            **common,
        ),
        boundary_admittance=model,
        remove_output_dc=False,
    )
    reverse = simulate_fdtd_reference(
        FDTDReferenceConfig(
            source_position_m=receiver_position_m,
            receiver_position_m=source_position_m,
            **common,
        ),
        boundary_admittance=model,
        remove_output_dc=False,
    )
    difference = forward.rir - reverse.rir
    raw_nrmse = float(
        np.sqrt(np.mean(difference**2))
        / max(float(np.sqrt(np.mean(forward.rir**2))), 1e-30)
    )
    reciprocal = simulate_reciprocal_fdtd_reference(
        FDTDReferenceConfig(
            source_position_m=source_position_m,
            receiver_position_m=receiver_position_m,
            **common,
        ),
        boundary_admittance=model,
        remove_output_dc=False,
    )
    expected_reciprocal = 0.5 * (forward.rir + reverse.rir)
    reciprocal_difference = reciprocal.rir - expected_reciprocal
    reciprocal_nrmse = float(
        np.sqrt(np.mean(reciprocal_difference**2))
        / max(
            float(np.sqrt(np.mean(expected_reciprocal**2))),
            1e-30,
        )
    )
    return {
        "case_id": case_id,
        "source_position_m": list(source_position_m),
        "receiver_position_m": list(receiver_position_m),
        "all_samples_finite": bool(
            np.all(np.isfinite(forward.rir))
            and np.all(np.isfinite(reverse.rir))
        ),
        "raw_maximum_absolute_error": float(
            np.max(np.abs(difference))
        ),
        "raw_nrmse": raw_nrmse,
        "reciprocalized_matches_bidirectional_average_nrmse": (
            reciprocal_nrmse
        ),
        "reciprocalized_metadata_raw_nrmse": (
            reciprocal.raw_reciprocity_nrmse
        ),
        "reciprocity_averaged": reciprocal.reciprocity_averaged,
        "effective_interior_cfl": forward.effective_interior_cfl,
    }


def main() -> None:
    args = _parse_args()
    predecessor = json.loads(
        args.m3_9_report.read_text(encoding="utf-8")
    )
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(
        args.boundary_config
    )
    model = _uniform_model(boundary_config)
    cases = [
        _plane_mode_case(
            model,
            case_id="x_normal_y_tangential",
            normal_axis=0,
            dimensions_xyz_m=(3.0, 0.656, 0.36),
            mode_indices_xyz=(0, 1, 0),
            frequency_hz=285.0,
            duration_s=0.70,
        ),
        _plane_mode_case(
            model,
            case_id="y_normal_z_tangential",
            normal_axis=1,
            dimensions_xyz_m=(0.36, 3.0, 0.656),
            mode_indices_xyz=(0, 0, 1),
            frequency_hz=285.0,
            duration_s=0.70,
        ),
        _plane_mode_case(
            model,
            case_id="z_normal_x_tangential",
            normal_axis=2,
            dimensions_xyz_m=(0.656, 0.36, 3.0),
            mode_indices_xyz=(1, 0, 0),
            frequency_hz=285.0,
            duration_s=0.70,
        ),
        _plane_mode_case(
            model,
            case_id="x_normal_yz_non_equal_azimuth",
            normal_axis=0,
            dimensions_xyz_m=(3.0, 0.8, 1.2),
            mode_indices_xyz=(0, 1, 1),
            frequency_hz=285.0,
            duration_s=0.80,
        ),
    ]
    reciprocity = [
        _reciprocity_case(
            model,
            case_id="face_near",
            source_position_m=(0.15, 0.50, 0.40),
            receiver_position_m=(0.85, 0.55, 0.45),
        ),
        _reciprocity_case(
            model,
            case_id="edge_near",
            source_position_m=(0.15, 0.15, 0.40),
            receiver_position_m=(0.85, 0.75, 0.50),
        ),
        _reciprocity_case(
            model,
            case_id="corner_near",
            source_position_m=(0.15, 0.15, 0.15),
            receiver_position_m=(0.85, 0.75, 0.65),
        ),
    ]
    maximum_complex = max(
        case["error"]["maximum_complex_error"] for case in cases
    )
    maximum_phase = max(
        case["error"]["maximum_phase_error_deg"] for case in cases
    )
    maximum_raw_reciprocity_nrmse = max(
        case["raw_nrmse"] for case in reciprocity
    )
    maximum_reciprocalized_nrmse = max(
        case["reciprocalized_matches_bidirectional_average_nrmse"]
        for case in reciprocity
    )
    plane_modes_accepted = bool(
        maximum_complex <= float(args.maximum_plane_mode_complex_error)
        and maximum_phase <= float(args.maximum_plane_mode_phase_error_deg)
    )
    reciprocity_accepted = bool(
        all(case["all_samples_finite"] for case in reciprocity)
        and all(case["reciprocity_averaged"] for case in reciprocity)
        and maximum_reciprocalized_nrmse
        <= float(args.maximum_reciprocity_nrmse)
    )
    predecessor_accepted = bool(
        predecessor["acceptance"][
            "opt_in_three_dimensional_reference_accepted"
        ]
    )
    accepted = bool(
        predecessor_accepted
        and plane_modes_accepted
        and reciprocity_accepted
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "boundary_config": str(args.boundary_config),
        "boundary_reference_id": boundary_config.reference_id,
        "m3_9_report": str(args.m3_9_report),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_report"
        },
        "plane_mode_holdouts": cases,
        "face_edge_corner_reciprocity": reciprocity,
        "summary": {
            "maximum_plane_mode_complex_error": maximum_complex,
            "maximum_plane_mode_phase_error_deg": maximum_phase,
            "maximum_raw_reciprocity_nrmse": (
                maximum_raw_reciprocity_nrmse
            ),
            "maximum_reciprocalized_nrmse": (
                maximum_reciprocalized_nrmse
            ),
        },
        "acceptance": {
            "m3_9_predecessor_accepted": predecessor_accepted,
            "multi_axis_azimuth_plane_modes_accepted": (
                plane_modes_accepted
            ),
            "face_edge_corner_reciprocity_accepted": (
                reciprocity_accepted
            ),
            "expanded_three_dimensional_reference_accepted": accepted,
            "production_default_accepted": False,
        },
        "scope": {
            "normal_axes_xyz": True,
            "single_tangential_modes": True,
            "two_tangential_non_equal_azimuth": True,
            "face_edge_corner_raw_reciprocity_audited": True,
            "face_edge_corner_reciprocalized_transfer": True,
            "full_room_crossover_rerun": False,
            "production_default_changed": False,
        },
        "decision": (
            "accept_expanded_three_dimensional_reference"
            if accepted
            else "retain_scoped_m3_9_reference"
        ),
        "next_action": (
            "rerun frozen full-room direct/early/later crossover attribution "
            "with the accepted fourth-order reference"
            if accepted
            else "inspect failed multi-axis or reciprocity holdouts"
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
                "summary": report["summary"],
                "acceptance": report["acceptance"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
