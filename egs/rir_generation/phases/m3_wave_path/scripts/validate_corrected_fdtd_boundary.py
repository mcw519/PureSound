#!/usr/bin/env python3
"""Build the M3.6 corrected-boundary harmonic and time-domain report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from egs.rir_generation.phases.m2_impedance.scripts.validate_full_room_crossover import (
    _sampled_frequency_response,
)
from egs.rir_generation.phases.m3_wave_path.scripts.validate_oblique_fdtd_boundary import (
    _compare_direction,
    _direction,
    _early_geometry_samples,
    _error_summary,
    _uniform_model,
    _worst_metrics,
)
from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
    digital_normalized_admittance_filter,
)
from puresound.audio.rir.physics.wave.fdtd import (
    fdtd_discrete_plane_wave_reflection,
    ricker_source,
)
from puresound.audio.rir.physics.impedance.modes import (
    RectangularImpedanceBoundaryConfig,
)


REPORT_SCHEMA_VERSION = "puresound.corrected_fdtd_boundary_validation.v1"
_SCHEMES = (
    "cell_center",
    "face_extrapolated",
    "face_time_extrapolated",
)
_TIME_DOMAIN_SCHEMES = (
    *_SCHEMES,
    "face_quadratic_time_quadratic",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare legacy and experimental wall-face pressure schemes in "
            "harmonic oblique scans and a 1D time-domain plane-wave probe."
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
    parser.add_argument(
        "--maximum-time-domain-complex-error",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--maximum-time-domain-phase-error-deg",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


class _ScalarIIR:
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray):
        self.numerator = np.asarray(numerator, dtype=np.float64)
        self.denominator = np.asarray(denominator, dtype=np.float64)
        self.input_history = np.zeros(
            max(0, self.numerator.size - 1),
            dtype=np.float64,
        )
        self.output_history = np.zeros(
            max(0, self.denominator.size - 1),
            dtype=np.float64,
        )

    def step(self, value: float) -> float:
        output = float(self.numerator[0]) * float(value)
        if self.input_history.size:
            output += float(
                np.dot(self.numerator[1:], self.input_history)
            )
        if self.output_history.size:
            output -= float(
                np.dot(self.denominator[1:], self.output_history)
            )
        if self.input_history.size:
            self.input_history[1:] = self.input_history[:-1]
            self.input_history[0] = float(value)
        if self.output_history.size:
            self.output_history[1:] = self.output_history[:-1]
            self.output_history[0] = output
        return output


def _simulate_1d_plane_wave(
    model,
    *,
    boundary_pressure_scheme: str,
    sample_rate_hz: float,
    grid_spacing_m: float,
    duration_s: float,
    room_length_m: float,
    source_position_m: float,
    receiver_position_m: float,
    source_center_hz: float,
    source_delay_s: float,
    sound_speed_m_s: float = 343.0,
    air_density_kg_m3: float = 1.204,
    spatial_derivative_order: int = 2,
    near_wall_closure: str = "second_order",
) -> tuple[np.ndarray, dict[str, Any]]:
    if boundary_pressure_scheme not in _TIME_DOMAIN_SCHEMES:
        raise ValueError("unsupported boundary pressure scheme")
    derivative_order = int(spatial_derivative_order)
    if derivative_order not in {2, 4}:
        raise ValueError("spatial_derivative_order must be 2 or 4")
    if near_wall_closure not in {
        "second_order",
        "third_order_one_sided",
    }:
        raise ValueError("unsupported near-wall closure")
    if derivative_order == 4 and near_wall_closure == "second_order":
        raise ValueError(
            "fourth-order interior requires an explicit near-wall closure"
        )
    if (
        boundary_pressure_scheme == "face_quadratic_time_quadratic"
        and derivative_order != 4
    ):
        raise ValueError(
            "quadratic face/time scheme requires fourth-order interior"
        )
    sample_rate = float(sample_rate_hz)
    time_step = 1.0 / sample_rate
    num_cells = max(6, int(round(room_length_m / grid_spacing_m)))
    spacing = float(room_length_m) / num_cells
    num_samples = round(float(duration_s) * sample_rate)
    source_cell = int(
        np.clip(math.floor(source_position_m / spacing), 0, num_cells - 1)
    )
    receiver_cell = int(
        np.clip(
            math.floor(receiver_position_m / spacing),
            0,
            num_cells - 1,
        )
    )
    source_position = (source_cell + 0.5) * spacing
    receiver_position = (receiver_cell + 0.5) * spacing
    source = ricker_source(
        num_samples,
        time_step,
        source_center_hz,
        source_delay_s,
    )
    numerator, denominator = digital_normalized_admittance_filter(
        model,
        sample_rate,
    )
    boundary_filter = _ScalarIIR(numerator, denominator)
    pressure = np.zeros(num_cells, dtype=np.float64)
    velocity = np.zeros(num_cells + 1, dtype=np.float64)
    output = np.zeros(num_samples, dtype=np.float64)
    velocity_scale = time_step / (air_density_kg_m3 * spacing)
    pressure_scale = air_density_kg_m3 * sound_speed_m_s**2 * time_step
    characteristic_impedance = air_density_kg_m3 * sound_speed_m_s
    previous_face_pressure = 0.0
    previous_face_pressure_2 = 0.0
    for sample in range(num_samples):
        if derivative_order == 2:
            velocity[1:-1] -= velocity_scale * (
                pressure[1:] - pressure[:-1]
            )
        else:
            velocity[2:-2] -= velocity_scale * (
                9.0
                / 8.0
                * (pressure[2:-1] - pressure[1:-2])
                - 1.0
                / 24.0
                * (pressure[3:] - pressure[:-3])
            )
            velocity[1] -= velocity_scale * (
                -23.0 / 24.0 * pressure[0]
                + 7.0 / 8.0 * pressure[1]
                + 1.0 / 8.0 * pressure[2]
                - 1.0 / 24.0 * pressure[3]
            )
            velocity[-2] -= velocity_scale * (
                1.0 / 24.0 * pressure[-4]
                - 1.0 / 8.0 * pressure[-3]
                - 7.0 / 8.0 * pressure[-2]
                + 23.0 / 24.0 * pressure[-1]
            )
        if boundary_pressure_scheme == "cell_center":
            boundary_pressure = float(pressure[0])
        elif boundary_pressure_scheme in {
            "face_extrapolated",
            "face_time_extrapolated",
        }:
            face_pressure = float(1.5 * pressure[0] - 0.5 * pressure[1])
            if boundary_pressure_scheme == "face_time_extrapolated":
                boundary_pressure = (
                    1.5 * face_pressure - 0.5 * previous_face_pressure
                )
                previous_face_pressure = face_pressure
            else:
                boundary_pressure = face_pressure
        else:
            face_pressure = float(
                15.0 / 8.0 * pressure[0]
                - 5.0 / 4.0 * pressure[1]
                + 3.0 / 8.0 * pressure[2]
            )
            boundary_pressure = (
                15.0 / 8.0 * face_pressure
                - 5.0 / 4.0 * previous_face_pressure
                + 3.0 / 8.0 * previous_face_pressure_2
            )
            previous_face_pressure_2 = previous_face_pressure
            previous_face_pressure = face_pressure
        velocity[0] = -boundary_filter.step(
            boundary_pressure
        ) / characteristic_impedance
        velocity[-1] = 0.0
        if derivative_order == 2:
            pressure -= pressure_scale * (
                velocity[1:] - velocity[:-1]
            ) / spacing
        else:
            pressure[1:-1] -= pressure_scale / spacing * (
                9.0
                / 8.0
                * (velocity[2:-1] - velocity[1:-2])
                - 1.0
                / 24.0
                * (velocity[3:] - velocity[:-3])
            )
            pressure[0] -= pressure_scale / spacing * (
                -23.0 / 24.0 * velocity[0]
                + 7.0 / 8.0 * velocity[1]
                + 1.0 / 8.0 * velocity[2]
                - 1.0 / 24.0 * velocity[3]
            )
            pressure[-1] -= pressure_scale / spacing * (
                1.0 / 24.0 * velocity[-4]
                - 1.0 / 8.0 * velocity[-3]
                - 7.0 / 8.0 * velocity[-2]
                + 23.0 / 24.0 * velocity[-1]
            )
        pressure[source_cell] += float(source[sample])
        output[sample] = pressure[receiver_cell]
    return output, {
        "grid_spacing_m": spacing,
        "source_cell_center_m": source_position,
        "receiver_cell_center_m": receiver_position,
        "sample_rate_hz": sample_rate,
        "time_step_s": time_step,
        "spatial_derivative_order": derivative_order,
        "near_wall_closure": near_wall_closure,
        "one_dimensional_cfl_number": (
            sound_speed_m_s
            * time_step
            / spacing
            * (7.0 / 6.0 if derivative_order == 4 else 1.0)
        ),
    }


def _raised_cosine_interval(
    *,
    num_samples: int,
    sample_rate_hz: float,
    start_s: float,
    stop_s: float,
    ramp_s: float,
) -> np.ndarray:
    time_s = np.arange(num_samples, dtype=np.float64) / sample_rate_hz
    window = np.zeros(num_samples, dtype=np.float64)
    plateau = (time_s >= start_s + ramp_s) & (
        time_s <= stop_s - ramp_s
    )
    window[plateau] = 1.0
    attack = (time_s >= start_s) & (time_s < start_s + ramp_s)
    release = (time_s > stop_s - ramp_s) & (time_s <= stop_s)
    window[attack] = 0.5 - 0.5 * np.cos(
        math.pi * (time_s[attack] - start_s) / ramp_s
    )
    window[release] = 0.5 - 0.5 * np.cos(
        math.pi * (stop_s - time_s[release]) / ramp_s
    )
    return window


def _time_domain_probe(
    model,
    *,
    scheme: str,
    frequencies_hz: np.ndarray,
    sample_rate_hz: float,
    grid_spacing_m: float,
    spatial_derivative_order: int = 2,
    near_wall_closure: str = "second_order",
) -> dict[str, Any]:
    common = {
        "boundary_pressure_scheme": scheme,
        "sample_rate_hz": sample_rate_hz,
        "grid_spacing_m": grid_spacing_m,
        "duration_s": 0.220,
        "room_length_m": 20.0,
        "source_position_m": 15.0,
        "receiver_position_m": 10.0,
        "source_center_hz": 240.0,
        "source_delay_s": 0.020,
        "spatial_derivative_order": spatial_derivative_order,
        "near_wall_closure": near_wall_closure,
    }
    rigid = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.0,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    reference = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.35,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=100.0,
    )
    target_output, geometry = _simulate_1d_plane_wave(model, **common)
    rigid_output, _rigid_geometry = _simulate_1d_plane_wave(
        rigid,
        **common,
    )
    reference_output, _reference_geometry = _simulate_1d_plane_wave(
        reference,
        **common,
    )
    first_floor_center_s = (
        common["source_delay_s"]
        + (
            geometry["source_cell_center_m"]
            + geometry["receiver_cell_center_m"]
        )
        / 343.0
    )
    window = _raised_cosine_interval(
        num_samples=target_output.size,
        sample_rate_hz=sample_rate_hz,
        start_s=first_floor_center_s - 0.025,
        stop_s=first_floor_center_s + 0.075,
        ramp_s=0.008,
    )
    target_difference = (target_output - reference_output) * window
    rigid_difference = (rigid_output - reference_output) * window
    target_response = _sampled_frequency_response(
        target_difference,
        sample_rate_hz,
        frequencies_hz,
    )
    rigid_response = _sampled_frequency_response(
        rigid_difference,
        sample_rate_hz,
        frequencies_hz,
    )
    measured_cross_ratio = target_response / rigid_response
    direction = (0.0, 0.0, 1.0)
    # Preserve the published M3.6 second-order report convention. M3.8 starts
    # a new schema and uses the realized spacing after room-cell rounding.
    harmonic_grid_spacing_m = (
        geometry["grid_spacing_m"]
        if int(spatial_derivative_order) == 4
        else grid_spacing_m
    )
    target_reflections = []
    reference_reflections = []
    for frequency_hz in frequencies_hz:
        target_reflections.append(
            fdtd_discrete_plane_wave_reflection(
                model,
                frequency_hz=float(frequency_hz),
                incident_direction_unit=direction,
                normal_axis=2,
                grid_spacing_xyz_m=(harmonic_grid_spacing_m,) * 3,
                time_step_s=1.0 / sample_rate_hz,
                boundary_pressure_scheme=scheme,
                spatial_derivative_order=spatial_derivative_order,
            ).reflection_coefficient
        )
        reference_reflections.append(
            fdtd_discrete_plane_wave_reflection(
                reference,
                frequency_hz=float(frequency_hz),
                incident_direction_unit=direction,
                normal_axis=2,
                grid_spacing_xyz_m=(harmonic_grid_spacing_m,) * 3,
                time_step_s=1.0 / sample_rate_hz,
                boundary_pressure_scheme=scheme,
                spatial_derivative_order=spatial_derivative_order,
            ).reflection_coefficient
        )
    target_reflections = np.asarray(
        target_reflections,
        dtype=np.complex128,
    )
    reference_reflections = np.asarray(
        reference_reflections,
        dtype=np.complex128,
    )
    predicted_cross_ratio = (
        target_reflections - reference_reflections
    ) / (
        1.0 - reference_reflections
    )
    relative_denominator = np.abs(rigid_response) / max(
        float(np.max(np.abs(rigid_response))),
        1e-30,
    )
    return {
        "boundary_pressure_scheme": scheme,
        "spatial_derivative_order": int(spatial_derivative_order),
        "near_wall_closure": near_wall_closure,
        "geometry": geometry,
        "first_floor_reflection_center_s": first_floor_center_s,
        "window_s": [
            first_floor_center_s - 0.025,
            first_floor_center_s + 0.075,
        ],
        "minimum_relative_cross_ratio_denominator": float(
            np.min(relative_denominator)
        ),
        "measured_vs_predicted_discrete_cross_ratio": _error_summary(
            predicted_cross_ratio,
            measured_cross_ratio,
            frequencies_hz,
        ),
    }


def _scheme_harmonic_report(
    model,
    *,
    scheme: str,
    full_room: dict[str, Any],
    boundary_config: RectangularImpedanceBoundaryConfig,
    frequencies_hz: np.ndarray,
    path_event_sample_rate_hz: float,
    spatial_derivative_order: int = 2,
) -> dict[str, Any]:
    canonical = []
    actual = []
    geometry_reports = []
    maximum_order = int(
        full_room["diagnostic"]["causal_path_event"]["max_order"]
    )
    for source_case in full_room["cases"]:
        grid_spacing = tuple(
            float(value)
            for value in source_case["fdtd"]["grid_spacing_xyz_m"]
        )
        time_step = 1.0 / float(
            source_case["fdtd"]["sample_rate_hz"]
        )
        for normal_axis in range(3):
            for cosine in (0.1, 0.25, 0.5, 0.75, 1.0):
                azimuths = (0.0,) if cosine == 1.0 else (0.0, 45.0, 90.0)
                for azimuth in azimuths:
                    canonical.append(
                        _compare_direction(
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
                            path_event_sample_rate_hz=(
                                path_event_sample_rate_hz
                            ),
                            boundary_pressure_scheme=scheme,
                            spatial_derivative_order=(
                                spatial_derivative_order
                            ),
                        )
                    )
        geometry_report, samples = _early_geometry_samples(
            source_case,
            boundary_config,
            maximum_order=maximum_order,
            early_window_s=0.050,
        )
        geometry_reports.append(
            {
                "case_id": source_case["case"]["case_id"],
                **geometry_report,
            }
        )
        actual.extend(
            _compare_direction(
                model,
                frequencies_hz=frequencies_hz,
                direction=direction,
                normal_axis=normal_axis,
                grid_spacing_xyz_m=grid_spacing,
                time_step_s=time_step,
                path_event_sample_rate_hz=path_event_sample_rate_hz,
                boundary_pressure_scheme=scheme,
                spatial_derivative_order=spatial_derivative_order,
            )
            for normal_axis, direction in samples
        )
    comparison_key = "fdtd_discrete_vs_continuous"
    return {
        "scheme": scheme,
        "spatial_derivative_order": int(spatial_derivative_order),
        "canonical_direction_count": len(canonical),
        "actual_early_representative_direction_count": len(actual),
        "canonical_fdtd_vs_continuous": _worst_metrics(
            canonical,
            comparison_key,
        ),
        "canonical_fdtd_vs_dispersion_matched": _worst_metrics(
            canonical,
            "fdtd_discrete_vs_dispersion_matched",
        ),
        "actual_early_fdtd_vs_continuous": _worst_metrics(
            actual,
            comparison_key,
        ),
        "actual_early_fdtd_vs_dispersion_matched": _worst_metrics(
            actual,
            "fdtd_discrete_vs_dispersion_matched",
        ),
        "maximum_canonical_reflection_magnitude": float(
            max(
                item["maximum_fdtd_discrete_reflection_magnitude"]
                for item in canonical
            )
        ),
        "maximum_actual_early_reflection_magnitude": float(
            max(
                item["maximum_fdtd_discrete_reflection_magnitude"]
                for item in actual
            )
        ),
        "early_path_geometry": geometry_reports,
    }


def main() -> None:
    args = _parse_args()
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
    scheme_reports = {
        scheme: _scheme_harmonic_report(
            model,
            scheme=scheme,
            full_room=full_room,
            boundary_config=boundary_config,
            frequencies_hz=frequencies_hz,
            path_event_sample_rate_hz=float(
                args.path_event_sample_rate_hz
            ),
        )
        for scheme in _SCHEMES
    }
    base_case = full_room["cases"][0]
    base_grid = tuple(
        float(value)
        for value in base_case["fdtd"]["grid_spacing_xyz_m"]
    )
    base_sample_rate = float(base_case["fdtd"]["sample_rate_hz"])
    time_domain = {
        scheme: _time_domain_probe(
            model,
            scheme=scheme,
            frequencies_hz=frequencies_hz,
            sample_rate_hz=base_sample_rate,
            grid_spacing_m=float(np.mean(base_grid)),
        )
        for scheme in _SCHEMES
    }
    scheme_acceptance = {}
    for scheme, scheme_report in scheme_reports.items():
        harmonic = scheme_report[
            "actual_early_fdtd_vs_continuous"
        ]
        time_metrics = time_domain[scheme][
            "measured_vs_predicted_discrete_cross_ratio"
        ]
        harmonic_parity = bool(
            harmonic["worst_maximum_complex_error"]
            <= float(args.maximum_complex_error)
            and harmonic["worst_maximum_magnitude_error_db"]
            <= float(args.maximum_magnitude_error_db)
            and harmonic["worst_maximum_phase_error_deg"]
            <= float(args.maximum_phase_error_deg)
        )
        passive = bool(
            scheme_report["maximum_canonical_reflection_magnitude"]
            <= 1.0 + 1e-12
            and scheme_report[
                "maximum_actual_early_reflection_magnitude"
            ]
            <= 1.0 + 1e-12
        )
        time_domain_accepted = bool(
            time_metrics["maximum_complex_error"]
            <= float(args.maximum_time_domain_complex_error)
            and time_metrics["maximum_phase_error_deg"]
            <= float(args.maximum_time_domain_phase_error_deg)
        )
        scheme_acceptance[scheme] = {
            "harmonic_continuous_parity_accepted": harmonic_parity,
            "discrete_passivity_accepted": passive,
            "time_domain_equation_accepted": time_domain_accepted,
            "reference_candidate_accepted": bool(
                harmonic_parity and passive and time_domain_accepted
            ),
        }
    accepted_candidates = [
        scheme
        for scheme, acceptance in scheme_acceptance.items()
        if acceptance["reference_candidate_accepted"]
    ]
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
        "schemes": scheme_reports,
        "time_domain_normal_incidence_cross_ratio": time_domain,
        "acceptance": {
            "maximum_complex_error": float(args.maximum_complex_error),
            "maximum_magnitude_error_db": float(
                args.maximum_magnitude_error_db
            ),
            "maximum_phase_error_deg": float(
                args.maximum_phase_error_deg
            ),
            "maximum_time_domain_complex_error": float(
                args.maximum_time_domain_complex_error
            ),
            "maximum_time_domain_phase_error_deg": float(
                args.maximum_time_domain_phase_error_deg
            ),
            "by_scheme": scheme_acceptance,
            "accepted_reference_candidates": accepted_candidates,
        },
        "scope": {
            "face_extrapolated": (
                "linear extrapolation from the first two pressure-cell centers"
            ),
            "face_time_extrapolated": (
                "face extrapolation plus causal 1.5*p[n]-0.5*p[n-1] "
                "half-step predictor"
            ),
            "time_domain_probe": (
                "1D normal-incidence FDTD cross-ratio; direct and all paths "
                "without the test wall cancel algebraically"
            ),
            "3d_time_domain_oblique_validated": False,
            "production_default_changed": False,
        },
        "decision": (
            "accept_corrected_boundary_reference"
            if accepted_candidates
            else "reject_simple_extrapolation_as_complete_boundary_fix"
        ),
        "next_action": (
            "retain the legacy production default; simple local extrapolation "
            "cannot remove grazing-angle dispersion. Evaluate a higher-order "
            "characteristic or angle-aware boundary reference before rerunning "
            "the full-room crossover"
            if not accepted_candidates
            else (
                "rerun the frozen full-room crossover with the accepted "
                "experimental boundary while retaining the legacy default"
            )
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
                "accepted_reference_candidates": accepted_candidates,
                "harmonic_actual_early": {
                    scheme: scheme_reports[scheme][
                        "actual_early_fdtd_vs_continuous"
                    ]
                    for scheme in _SCHEMES
                },
                "time_domain": {
                    scheme: time_domain[scheme][
                        "measured_vs_predicted_discrete_cross_ratio"
                    ]
                    for scheme in _SCHEMES
                },
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
