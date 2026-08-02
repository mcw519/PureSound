#!/usr/bin/env python3
"""Build the M3.5 discrete oblique-boundary reflection report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from puresound.audio.acoustic_impedance import (
    digital_locally_reacting_reflection_filter,
)
from puresound.audio.fdtd_reference import (
    fdtd_discrete_plane_wave_reflection,
)
from puresound.audio.impedance_modes import (
    RectangularImpedanceBoundaryConfig,
)
from puresound.audio.rir_path_events import (
    generate_shoebox_path_events,
    partition_path_events_by_arrival,
)


REPORT_SCHEMA_VERSION = "puresound.oblique_fdtd_boundary_validation.v1"
_BOUNDARY_AXIS = {
    "west": 0,
    "east": 0,
    "south": 1,
    "north": 1,
    "floor": 2,
    "ceiling": 2,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare continuous, PathEvent-digital, and exact implemented "
            "staggered-grid FDTD oblique reflection coefficients."
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
        "--full-room-report",
        type=Path,
        default=Path(
            "egs/rir_generation/phases/m3_wave_path/reports/"
            "full_room_crossover_m3_3_report.json"
        ),
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--path-event-sample-rate-hz", type=int, default=8000)
    parser.add_argument(
        "--incidence-cosines",
        type=float,
        nargs="+",
        default=(0.1, 0.25, 0.5, 0.75, 1.0),
    )
    parser.add_argument(
        "--tangential-azimuth-deg",
        type=float,
        nargs="+",
        default=(0.0, 45.0, 90.0),
    )
    parser.add_argument(
        "--maximum-complex-error",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--maximum-magnitude-error-db",
        type=float,
        default=0.10,
    )
    parser.add_argument(
        "--maximum-phase-error-deg",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def _uniform_model(boundary_config: RectangularImpedanceBoundaryConfig):
    models = tuple(boundary_config.boundaries.values())
    reference = models[0].metadata()
    if any(model.metadata() != reference for model in models[1:]):
        raise ValueError("M3.5 currently requires one uniform boundary model")
    return models[0]


def _direction(
    normal_axis: int,
    incidence_cosine: float,
    azimuth_deg: float,
) -> tuple[float, float, float]:
    cosine = float(incidence_cosine)
    if not 0.0 < cosine <= 1.0:
        raise ValueError("incidence cosines must lie in (0, 1]")
    axis = int(normal_axis)
    tangent_axes = [index for index in range(3) if index != axis]
    azimuth = math.radians(float(azimuth_deg))
    tangent = math.sqrt(max(0.0, 1.0 - cosine**2))
    values = np.zeros(3, dtype=np.float64)
    values[axis] = cosine
    values[tangent_axes[0]] = tangent * math.cos(azimuth)
    values[tangent_axes[1]] = tangent * math.sin(azimuth)
    return tuple(values.astype(float).tolist())


def _error_summary(
    reference: np.ndarray,
    candidate: np.ndarray,
    frequencies_hz: np.ndarray,
) -> dict[str, float]:
    reference_values = np.asarray(reference, dtype=np.complex128)
    candidate_values = np.asarray(candidate, dtype=np.complex128)
    complex_error = np.abs(candidate_values - reference_values)
    magnitude_error_db = np.abs(
        20.0
        * np.log10(
            np.maximum(np.abs(candidate_values), 1e-30)
            / np.maximum(np.abs(reference_values), 1e-30)
        )
    )
    phase_error_deg = np.abs(
        np.degrees(np.angle(candidate_values / reference_values))
    )
    complex_index = int(np.argmax(complex_error))
    magnitude_index = int(np.argmax(magnitude_error_db))
    phase_index = int(np.argmax(phase_error_deg))
    return {
        "maximum_complex_error": float(complex_error[complex_index]),
        "maximum_complex_error_frequency_hz": float(
            frequencies_hz[complex_index]
        ),
        "maximum_magnitude_error_db": float(
            magnitude_error_db[magnitude_index]
        ),
        "maximum_magnitude_error_frequency_hz": float(
            frequencies_hz[magnitude_index]
        ),
        "maximum_phase_error_deg": float(phase_error_deg[phase_index]),
        "maximum_phase_error_frequency_hz": float(
            frequencies_hz[phase_index]
        ),
        "mean_complex_error": float(np.mean(complex_error)),
        "mean_magnitude_error_db": float(np.mean(magnitude_error_db)),
        "mean_phase_error_deg": float(np.mean(phase_error_deg)),
    }


def _compare_direction(
    model,
    *,
    frequencies_hz: np.ndarray,
    direction: tuple[float, float, float],
    normal_axis: int,
    grid_spacing_xyz_m: tuple[float, float, float],
    time_step_s: float,
    path_event_sample_rate_hz: float,
    boundary_pressure_scheme: str = "cell_center",
    spatial_derivative_order: int = 2,
) -> dict[str, Any]:
    requested_cosine = abs(float(direction[normal_axis]))
    path_filter = digital_locally_reacting_reflection_filter(
        model,
        requested_cosine,
        path_event_sample_rate_hz,
    )
    continuous = []
    path_digital = []
    fdtd_discrete = []
    dispersion_matched = []
    discrete_cosines = []
    for frequency_hz in frequencies_hz:
        analog_admittance = complex(
            model.normalized_admittance(float(frequency_hz))
        )
        continuous.append(
            (requested_cosine - analog_admittance)
            / (requested_cosine + analog_admittance)
        )
        path_digital.append(
            path_filter.frequency_response(float(frequency_hz))
        )
        discrete = fdtd_discrete_plane_wave_reflection(
            model,
            frequency_hz=float(frequency_hz),
            incident_direction_unit=direction,
            normal_axis=normal_axis,
            grid_spacing_xyz_m=grid_spacing_xyz_m,
            time_step_s=time_step_s,
            boundary_pressure_scheme=boundary_pressure_scheme,
            spatial_derivative_order=spatial_derivative_order,
        )
        fdtd_discrete.append(discrete.reflection_coefficient)
        discrete_cosines.append(discrete.discrete_incidence_cosine)
        dispersion_matched.append(
            (
                discrete.discrete_incidence_cosine
                - discrete.digital_normalized_admittance
            )
            / (
                discrete.discrete_incidence_cosine
                + discrete.digital_normalized_admittance
            )
        )
    continuous_values = np.asarray(continuous, dtype=np.complex128)
    path_values = np.asarray(path_digital, dtype=np.complex128)
    fdtd_values = np.asarray(fdtd_discrete, dtype=np.complex128)
    dispersion_matched_values = np.asarray(
        dispersion_matched,
        dtype=np.complex128,
    )
    discrete_cosine_values = np.asarray(
        discrete_cosines,
        dtype=np.float64,
    )
    return {
        "direction_unit": list(direction),
        "normal_axis": int(normal_axis),
        "boundary_pressure_scheme": boundary_pressure_scheme,
        "spatial_derivative_order": int(spatial_derivative_order),
        "requested_incidence_cosine": requested_cosine,
        "discrete_incidence_cosine_range": [
            float(np.min(discrete_cosine_values)),
            float(np.max(discrete_cosine_values)),
        ],
        "maximum_discrete_incidence_cosine_error": float(
            np.max(np.abs(discrete_cosine_values - requested_cosine))
        ),
        "path_digital_vs_continuous": _error_summary(
            continuous_values,
            path_values,
            frequencies_hz,
        ),
        "fdtd_discrete_vs_continuous": _error_summary(
            continuous_values,
            fdtd_values,
            frequencies_hz,
        ),
        "fdtd_discrete_vs_path_digital": _error_summary(
            path_values,
            fdtd_values,
            frequencies_hz,
        ),
        "fdtd_discrete_vs_dispersion_matched": _error_summary(
            dispersion_matched_values,
            fdtd_values,
            frequencies_hz,
        ),
        "maximum_continuous_reflection_magnitude": float(
            np.max(np.abs(continuous_values))
        ),
        "maximum_path_digital_reflection_magnitude": float(
            np.max(np.abs(path_values))
        ),
        "maximum_fdtd_discrete_reflection_magnitude": float(
            np.max(np.abs(fdtd_values))
        ),
    }


def _worst_metrics(
    comparisons: list[dict[str, Any]],
    comparison_key: str,
) -> dict[str, float]:
    metrics = [item[comparison_key] for item in comparisons]
    output = {}
    for metric in (
        "maximum_complex_error",
        "maximum_magnitude_error_db",
        "maximum_phase_error_deg",
        "mean_complex_error",
        "mean_magnitude_error_db",
        "mean_phase_error_deg",
    ):
        reducer = max if metric.startswith("maximum_") else np.mean
        output[
            (
                f"worst_{metric}"
                if metric.startswith("maximum_")
                else f"mean_of_{metric}"
            )
        ] = float(reducer([value[metric] for value in metrics]))
    return output


def _early_geometry_samples(
    source_case: dict[str, Any],
    boundary_config: RectangularImpedanceBoundaryConfig,
    *,
    maximum_order: int,
    early_window_s: float,
) -> tuple[dict[str, Any], list[tuple[int, tuple[float, float, float]]]]:
    event_set = generate_shoebox_path_events(
        dimensions_m=source_case["case"]["room_dim_m"],
        source_position_m=source_case["fdtd"]["source_cell_center_m"],
        receiver_position_m=source_case["fdtd"]["receiver_cell_center_m"],
        sound_speed_m_s=343.0,
        scene_id=source_case["case"]["case_id"],
        max_order=maximum_order,
        edge_corner_policy="sequential_face_product_diagnostic",
        boundary_admittance_models=boundary_config.boundaries,
        reflection_frequencies_hz=(168.0, 300.0),
    )
    buckets = partition_path_events_by_arrival(
        event_set,
        early_window_s=early_window_s,
    )
    by_axis: dict[int, list[tuple[float, tuple[float, float, float]]]] = {
        0: [],
        1: [],
        2: [],
    }
    for event in buckets["early_reflections"]:
        direction = tuple(
            abs(float(value))
            for value in event.departure_direction_unit
        )
        for surface_id, incidence_cosine in zip(
            event.surface_ids,
            event.incidence_cosines,
        ):
            axis = _BOUNDARY_AXIS[surface_id]
            if not math.isclose(
                direction[axis],
                float(incidence_cosine),
                rel_tol=0.0,
                abs_tol=1e-10,
            ):
                raise RuntimeError(
                    "shoebox path direction and incidence cosine disagree"
                )
            by_axis[axis].append((float(incidence_cosine), direction))
    quantile_probabilities = (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    samples = []
    axis_reports = {}
    for axis, values in by_axis.items():
        ordered = sorted(values, key=lambda item: item[0])
        cosines = np.asarray(
            [value[0] for value in ordered],
            dtype=np.float64,
        )
        selected_indices = sorted(
            {
                int(round(probability * (len(ordered) - 1)))
                for probability in quantile_probabilities
            }
        )
        samples.extend(
            (axis, ordered[index][1]) for index in selected_indices
        )
        axis_reports[str(axis)] = {
            "interaction_count": len(ordered),
            "incidence_cosine_quantiles": {
                f"{probability:g}": float(np.quantile(cosines, probability))
                for probability in quantile_probabilities
            },
            "representative_sample_count": len(selected_indices),
        }
    return (
        {
            "early_reflection_event_count": len(
                buckets["early_reflections"]
            ),
            "later_reflection_event_count": len(
                buckets["later_reflections"]
            ),
            "axis_interactions": axis_reports,
        },
        samples,
    )


def main() -> None:
    args = _parse_args()
    cosines = tuple(sorted(set(float(value) for value in args.incidence_cosines)))
    if not cosines or cosines[0] <= 0.0 or cosines[-1] > 1.0:
        raise ValueError("incidence cosines must lie in (0, 1]")
    azimuths = tuple(
        sorted(set(float(value) for value in args.tangential_azimuth_deg))
    )
    if not azimuths or any(not math.isfinite(value) for value in azimuths):
        raise ValueError("tangential azimuths must be finite")
    full_room = json.loads(
        args.full_room_report.read_text(encoding="utf-8")
    )
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(
        args.boundary_config
    )
    model = _uniform_model(boundary_config)
    crossover_hz = float(full_room["configuration"]["crossover_hz"])
    frequencies_hz = np.linspace(
        0.7 * crossover_hz,
        min(300.0, 1.25 * crossover_hz),
        265,
    )
    maximum_order = int(
        full_room["diagnostic"]["causal_path_event"]["max_order"]
    )
    case_reports = []
    all_canonical = []
    all_early_geometry = []
    for source_case in full_room["cases"]:
        grid_spacing = tuple(
            float(value)
            for value in source_case["fdtd"]["grid_spacing_xyz_m"]
        )
        time_step = 1.0 / float(
            source_case["fdtd"]["sample_rate_hz"]
        )
        canonical = []
        for normal_axis in range(3):
            for cosine in cosines:
                selected_azimuths = (0.0,) if cosine == 1.0 else azimuths
                for azimuth in selected_azimuths:
                    comparison = _compare_direction(
                        model,
                        frequencies_hz=frequencies_hz,
                        direction=_direction(
                            normal_axis,
                            cosine,
                            azimuth,
                        ),
                        normal_axis=normal_axis,
                        grid_spacing_xyz_m=grid_spacing,
                        time_step_s=time_step,
                        path_event_sample_rate_hz=float(
                            args.path_event_sample_rate_hz
                        ),
                    )
                    comparison["tangential_azimuth_deg"] = azimuth
                    canonical.append(comparison)
        geometry_report, geometry_samples = _early_geometry_samples(
            source_case,
            boundary_config,
            maximum_order=maximum_order,
            early_window_s=0.050,
        )
        early_geometry = [
            _compare_direction(
                model,
                frequencies_hz=frequencies_hz,
                direction=direction,
                normal_axis=normal_axis,
                grid_spacing_xyz_m=grid_spacing,
                time_step_s=time_step,
                path_event_sample_rate_hz=float(
                    args.path_event_sample_rate_hz
                ),
            )
            for normal_axis, direction in geometry_samples
        ]
        all_canonical.extend(canonical)
        all_early_geometry.extend(early_geometry)
        case_reports.append(
            {
                "case_id": source_case["case"]["case_id"],
                "grid_spacing_xyz_m": list(grid_spacing),
                "fdtd_sample_rate_hz": 1.0 / time_step,
                "canonical_direction_count": len(canonical),
                "canonical_aggregate": {
                    key: _worst_metrics(canonical, key)
                    for key in (
                        "path_digital_vs_continuous",
                        "fdtd_discrete_vs_continuous",
                        "fdtd_discrete_vs_path_digital",
                    )
                },
                "early_path_geometry": geometry_report,
                "early_path_representative_direction_count": len(
                    early_geometry
                ),
                "early_path_representative_aggregate": {
                    key: _worst_metrics(early_geometry, key)
                    for key in (
                        "path_digital_vs_continuous",
                        "fdtd_discrete_vs_continuous",
                        "fdtd_discrete_vs_path_digital",
                    )
                },
            }
        )
    canonical_aggregate = {
        key: _worst_metrics(all_canonical, key)
        for key in (
            "path_digital_vs_continuous",
            "fdtd_discrete_vs_continuous",
            "fdtd_discrete_vs_path_digital",
        )
    }
    early_geometry_aggregate = {
        key: _worst_metrics(all_early_geometry, key)
        for key in (
            "path_digital_vs_continuous",
            "fdtd_discrete_vs_continuous",
            "fdtd_discrete_vs_path_digital",
        )
    }
    worst_canonical = max(
        all_canonical,
        key=lambda item: item["fdtd_discrete_vs_continuous"][
            "maximum_complex_error"
        ],
    )
    base_case = full_room["cases"][0]
    base_spacing = tuple(
        float(value)
        for value in base_case["fdtd"]["grid_spacing_xyz_m"]
    )
    base_time_step = 1.0 / float(base_case["fdtd"]["sample_rate_hz"])
    convergence = []
    worst_direction = tuple(worst_canonical["direction_unit"])
    worst_axis = int(worst_canonical["normal_axis"])
    for scale in (1.0, 0.5, 0.25, 0.125):
        comparison = _compare_direction(
            model,
            frequencies_hz=frequencies_hz,
            direction=worst_direction,
            normal_axis=worst_axis,
            grid_spacing_xyz_m=tuple(
                scale * value for value in base_spacing
            ),
            time_step_s=scale * base_time_step,
            path_event_sample_rate_hz=float(
                args.path_event_sample_rate_hz
            ),
        )
        convergence.append(
            {
                "linear_grid_scale": scale,
                "grid_spacing_xyz_m": [
                    scale * value for value in base_spacing
                ],
                "fdtd_sample_rate_hz": 1.0 / (scale * base_time_step),
                "fdtd_discrete_vs_continuous": comparison[
                    "fdtd_discrete_vs_continuous"
                ],
            }
        )
    fdtd_metrics = canonical_aggregate["fdtd_discrete_vs_continuous"]
    parity_accepted = bool(
        fdtd_metrics["worst_maximum_complex_error"]
        <= float(args.maximum_complex_error)
        and fdtd_metrics["worst_maximum_magnitude_error_db"]
        <= float(args.maximum_magnitude_error_db)
        and fdtd_metrics["worst_maximum_phase_error_deg"]
        <= float(args.maximum_phase_error_deg)
    )
    passive_accepted = all(
        item["maximum_fdtd_discrete_reflection_magnitude"]
        <= 1.0 + 1e-12
        for item in all_canonical
    )
    convergence_errors = [
        item["fdtd_discrete_vs_continuous"]["maximum_complex_error"]
        for item in convergence
    ]
    convergence_accepted = all(
        later < earlier
        for earlier, later in zip(
            convergence_errors,
            convergence_errors[1:],
        )
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "boundary_config": str(args.boundary_config),
        "boundary_reference_id": boundary_config.reference_id,
        "full_room_report": str(args.full_room_report),
        "frequency_band_hz": [
            float(frequencies_hz[0]),
            float(frequencies_hz[-1]),
        ],
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_report"
        },
        "discrete_boundary_equation": {
            "formula": (
                "Gamma_fdtd=(q_d-Y_d*exp(-j*Omega/2)"
                "*exp(+j*k_n*dx_n/2))/(q_d+Y_d*exp(-j*Omega/2)"
                "*exp(-j*k_n*dx_n/2))"
            ),
            "q_d": (
                "normal discrete characteristic admittance from the 3D "
                "staggered-grid dispersion relation"
            ),
            "half_time_factor": (
                "pressure is sampled at integer time and boundary velocity "
                "at the following half step"
            ),
            "half_cell_factors": (
                "boundary velocity lies on the wall face while boundary "
                "pressure is taken from the first cell center"
            ),
        },
        "cases": case_reports,
        "canonical_aggregate": canonical_aggregate,
        "early_path_representative_aggregate": (
            early_geometry_aggregate
        ),
        "grid_convergence_at_worst_canonical_direction": {
            "direction_unit": list(worst_direction),
            "normal_axis": worst_axis,
            "cases": convergence,
        },
        "acceptance": {
            "maximum_complex_error": float(args.maximum_complex_error),
            "maximum_magnitude_error_db": float(
                args.maximum_magnitude_error_db
            ),
            "maximum_phase_error_deg": float(
                args.maximum_phase_error_deg
            ),
            "fdtd_continuous_boundary_parity_accepted": parity_accepted,
            "fdtd_discrete_passivity_accepted": passive_accepted,
            "grid_convergence_accepted": convergence_accepted,
            "protocol_completed": True,
        },
        "scope": {
            "validated": [
                "exact harmonic reflection of the implemented FDTD update",
                "3D discrete dispersion and tangential azimuth",
                "half-time and half-cell boundary staggering",
                "canonical and actual early-path incidence distributions",
                "linear grid refinement convergence",
            ],
            "time_domain_plane_wave_simulated": False,
            "continuous_boundary_parity_claimed": parity_accepted,
            "production_path_event_phase_changed": False,
        },
        "decision": (
            "accept_current_fdtd_boundary_as_continuous_reference"
            if parity_accepted
            else (
                "reject_coarse_fdtd_boundary_phase_as_continuous_ground_truth"
            )
        ),
        "next_action": (
            "validate a face-pressure or boundary-phase-compensated update "
            "against this discrete equation and a time-domain plane-wave "
            "case; do not fit the physically continuous PathEvent reflection "
            "phase to the current coarse-grid FDTD artifact"
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.output_report),
                "fdtd_continuous_boundary_parity_accepted": parity_accepted,
                "fdtd_discrete_passivity_accepted": passive_accepted,
                "grid_convergence_accepted": convergence_accepted,
                "canonical_fdtd_vs_continuous": fdtd_metrics,
                "early_path_fdtd_vs_continuous": (
                    early_geometry_aggregate[
                        "fdtd_discrete_vs_continuous"
                    ]
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
