#!/usr/bin/env python3
"""Build the M3.7 higher-order staggered-grid harmonic candidate report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from egs.rir_generation.phases.m3_wave_path.scripts.validate_corrected_fdtd_boundary import (
    _scheme_harmonic_report,
)
from egs.rir_generation.phases.m3_wave_path.scripts.validate_oblique_fdtd_boundary import (
    _compare_direction,
    _direction,
    _uniform_model,
    _worst_metrics,
)
from puresound.audio.rir.physics.impedance.modes import (
    RectangularImpedanceBoundaryConfig,
)


REPORT_SCHEMA_VERSION = "puresound.higher_order_fdtd_candidate.v1"
_CANDIDATES = {
    "second_order_cell_center": {
        "boundary_pressure_scheme": "cell_center",
        "spatial_derivative_order": 2,
    },
    "second_order_linear_face_time": {
        "boundary_pressure_scheme": "face_time_extrapolated",
        "spatial_derivative_order": 2,
    },
    "fourth_order_linear_face_time": {
        "boundary_pressure_scheme": "face_time_extrapolated",
        "spatial_derivative_order": 4,
    },
    "fourth_order_quadratic_face_time": {
        "boundary_pressure_scheme": "face_quadratic_time_quadratic",
        "spatial_derivative_order": 4,
    },
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare second- and fourth-order staggered-grid dispersion with "
            "linear and quadratic wall-face/half-time predictors."
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
    parser.add_argument("--maximum-complex-error", type=float, default=0.02)
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


def _passes_error_gate(
    metrics: dict[str, float],
    *,
    maximum_complex_error: float,
    maximum_magnitude_error_db: float,
    maximum_phase_error_deg: float,
) -> bool:
    return bool(
        metrics["worst_maximum_complex_error"] <= maximum_complex_error
        and metrics["worst_maximum_magnitude_error_db"]
        <= maximum_magnitude_error_db
        and metrics["worst_maximum_phase_error_deg"]
        <= maximum_phase_error_deg
    )


def _fourth_order_cfl_number(
    *,
    grid_spacing_xyz_m: tuple[float, float, float],
    time_step_s: float,
    sound_speed_m_s: float = 343.0,
) -> float:
    return float(
        sound_speed_m_s
        * time_step_s
        * math.sqrt(
            sum(
                (7.0 / (6.0 * spacing_m)) ** 2
                for spacing_m in grid_spacing_xyz_m
            )
        )
    )


def _grid_refinement_report(
    model,
    *,
    base_grid_spacing_xyz_m: tuple[float, float, float],
    base_time_step_s: float,
    frequencies_hz: np.ndarray,
    path_event_sample_rate_hz: float,
) -> list[dict[str, Any]]:
    reports = []
    for scale in (1.0, 0.5, 0.25, 0.125):
        spacing = tuple(
            scale * value for value in base_grid_spacing_xyz_m
        )
        time_step = scale * base_time_step_s
        comparisons = []
        for normal_axis in range(3):
            for cosine in (0.1, 0.25, 0.5, 0.75, 1.0):
                azimuths = (0.0,) if cosine == 1.0 else (0.0, 45.0, 90.0)
                for azimuth in azimuths:
                    comparisons.append(
                        _compare_direction(
                            model,
                            frequencies_hz=frequencies_hz,
                            direction=_direction(
                                normal_axis,
                                cosine,
                                azimuth,
                            ),
                            normal_axis=normal_axis,
                            grid_spacing_xyz_m=spacing,
                            time_step_s=time_step,
                            path_event_sample_rate_hz=(
                                path_event_sample_rate_hz
                            ),
                            boundary_pressure_scheme=(
                                "face_quadratic_time_quadratic"
                            ),
                            spatial_derivative_order=4,
                        )
                    )
        reports.append(
            {
                "linear_scale": scale,
                "grid_spacing_xyz_m": list(spacing),
                "time_step_s": time_step,
                "fourth_order_cfl_number": _fourth_order_cfl_number(
                    grid_spacing_xyz_m=spacing,
                    time_step_s=time_step,
                ),
                "fdtd_vs_continuous": _worst_metrics(
                    comparisons,
                    "fdtd_discrete_vs_continuous",
                ),
                "maximum_discrete_incidence_cosine_error": float(
                    max(
                        item[
                            "maximum_discrete_incidence_cosine_error"
                        ]
                        for item in comparisons
                    )
                ),
                "maximum_reflection_magnitude": float(
                    max(
                        item[
                            "maximum_fdtd_discrete_reflection_magnitude"
                        ]
                        for item in comparisons
                    )
                ),
            }
        )
    return reports


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
    candidate_reports = {}
    acceptance = {}
    case_cfl = {}
    for source_case in full_room["cases"]:
        grid_spacing = tuple(
            float(value)
            for value in source_case["fdtd"]["grid_spacing_xyz_m"]
        )
        time_step = 1.0 / float(
            source_case["fdtd"]["sample_rate_hz"]
        )
        case_cfl[source_case["case"]["case_id"]] = (
            _fourth_order_cfl_number(
                grid_spacing_xyz_m=grid_spacing,
                time_step_s=time_step,
            )
        )
    for candidate_id, candidate in _CANDIDATES.items():
        candidate_report = _scheme_harmonic_report(
            model,
            scheme=str(candidate["boundary_pressure_scheme"]),
            spatial_derivative_order=int(
                candidate["spatial_derivative_order"]
            ),
            full_room=full_room,
            boundary_config=boundary_config,
            frequencies_hz=frequencies_hz,
            path_event_sample_rate_hz=float(
                args.path_event_sample_rate_hz
            ),
        )
        candidate_reports[candidate_id] = candidate_report
        canonical_accepted = _passes_error_gate(
            candidate_report["canonical_fdtd_vs_continuous"],
            maximum_complex_error=float(args.maximum_complex_error),
            maximum_magnitude_error_db=float(
                args.maximum_magnitude_error_db
            ),
            maximum_phase_error_deg=float(args.maximum_phase_error_deg),
        )
        actual_accepted = _passes_error_gate(
            candidate_report["actual_early_fdtd_vs_continuous"],
            maximum_complex_error=float(args.maximum_complex_error),
            maximum_magnitude_error_db=float(
                args.maximum_magnitude_error_db
            ),
            maximum_phase_error_deg=float(args.maximum_phase_error_deg),
        )
        passive = bool(
            candidate_report["maximum_canonical_reflection_magnitude"]
            <= 1.0 + 1e-12
            and candidate_report[
                "maximum_actual_early_reflection_magnitude"
            ]
            <= 1.0 + 1e-12
        )
        stable_cfl = bool(
            int(candidate["spatial_derivative_order"]) != 4
            or max(case_cfl.values()) < 1.0
        )
        harmonic_accepted = bool(
            canonical_accepted
            and actual_accepted
            and passive
            and stable_cfl
        )
        acceptance[candidate_id] = {
            "canonical_continuous_parity_accepted": canonical_accepted,
            "actual_early_continuous_parity_accepted": actual_accepted,
            "discrete_passivity_accepted": passive,
            "frozen_grid_cfl_accepted": stable_cfl,
            "harmonic_candidate_accepted": harmonic_accepted,
            "time_domain_reference_accepted": False,
        }
    base_case = full_room["cases"][0]
    base_grid = tuple(
        float(value)
        for value in base_case["fdtd"]["grid_spacing_xyz_m"]
    )
    base_time_step = 1.0 / float(
        base_case["fdtd"]["sample_rate_hz"]
    )
    grid_refinement = _grid_refinement_report(
        model,
        base_grid_spacing_xyz_m=base_grid,
        base_time_step_s=base_time_step,
        frequencies_hz=frequencies_hz,
        path_event_sample_rate_hz=float(
            args.path_event_sample_rate_hz
        ),
    )
    accepted_candidates = [
        candidate_id
        for candidate_id, result in acceptance.items()
        if result["harmonic_candidate_accepted"]
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
        "candidate_definitions": _CANDIDATES,
        "candidates": candidate_reports,
        "fourth_order_frozen_grid_cfl_number": case_cfl,
        "fourth_order_quadratic_grid_refinement": grid_refinement,
        "acceptance": {
            "maximum_complex_error": float(args.maximum_complex_error),
            "maximum_magnitude_error_db": float(
                args.maximum_magnitude_error_db
            ),
            "maximum_phase_error_deg": float(
                args.maximum_phase_error_deg
            ),
            "by_candidate": acceptance,
            "accepted_harmonic_candidates": accepted_candidates,
            "accepted_time_domain_references": [],
        },
        "scope": {
            "fourth_order_interior": (
                "harmonic staggered derivative symbol only"
            ),
            "quadratic_face_time": (
                "15/8*p0-5/4*p1+3/8*p2 in space and the same causal "
                "three-sample predictor in time"
            ),
            "time_domain_implemented": False,
            "production_default_changed": False,
        },
        "decision": (
            "promote_harmonic_candidate_to_time_domain_prototype"
            if accepted_candidates
            else "reject_higher_order_harmonic_candidate"
        ),
        "next_action": (
            "implement a fourth-order 1D staggered-grid plane-wave prototype "
            "with an explicit near-wall closure; validate its cross-ratio "
            "against the accepted harmonic equation before changing the 3D "
            "reference or rerunning the full-room crossover"
            if accepted_candidates
            else "evaluate a non-local characteristic boundary reference"
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report["acceptance"], indent=2, sort_keys=True))
    print(f"decision={report['decision']}")


if __name__ == "__main__":
    main()
