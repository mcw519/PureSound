#!/usr/bin/env python3
"""Build the M3.8 fourth-order 1D time-domain closure report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from egs.rir_generation.phases.m3_wave_path.scripts.validate_corrected_fdtd_boundary import (
    _simulate_1d_plane_wave,
    _time_domain_probe,
)
from egs.rir_generation.phases.m3_wave_path.scripts.validate_oblique_fdtd_boundary import (
    _uniform_model,
)
from puresound.audio.impedance_modes import (
    RectangularImpedanceBoundaryConfig,
)


REPORT_SCHEMA_VERSION = "puresound.higher_order_fdtd_time_domain.v1"
_CANDIDATE_ID = "fourth_order_quadratic_face_time"
_BOUNDARY_PRESSURE_SCHEME = "face_quadratic_time_quadratic"
_SPATIAL_DERIVATIVE_ORDER = 4
_NEAR_WALL_CLOSURE = "third_order_one_sided"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the M3.7 fourth-order/quadratic harmonic candidate in "
            "an independent 1D time-domain cross-ratio probe."
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
    parser.add_argument(
        "--harmonic-candidate-report",
        type=Path,
        default=Path(
            "egs/rir_generation/phases/m3_wave_path/reports/"
            "higher_order_fdtd_candidate_m3_7_report.json"
        ),
    )
    parser.add_argument("--output-report", type=Path, required=True)
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
    parser.add_argument(
        "--maximum-tail-to-early-rms-ratio",
        type=float,
        default=1.0,
    )
    return parser.parse_args()


def _long_stability_probe(
    model,
    *,
    sample_rate_hz: float,
    grid_spacing_m: float,
) -> dict[str, Any]:
    output, geometry = _simulate_1d_plane_wave(
        model,
        boundary_pressure_scheme=_BOUNDARY_PRESSURE_SCHEME,
        sample_rate_hz=sample_rate_hz,
        grid_spacing_m=grid_spacing_m,
        duration_s=1.0,
        room_length_m=20.0,
        source_position_m=15.0,
        receiver_position_m=10.0,
        source_center_hz=240.0,
        source_delay_s=0.020,
        spatial_derivative_order=_SPATIAL_DERIVATIVE_ORDER,
        near_wall_closure=_NEAR_WALL_CLOSURE,
    )
    early = output[
        round(0.10 * sample_rate_hz) : round(0.20 * sample_rate_hz)
    ]
    tail = output[-round(0.10 * sample_rate_hz) :]
    early_rms = float(np.sqrt(np.mean(early**2)))
    tail_rms = float(np.sqrt(np.mean(tail**2)))
    return {
        "duration_s": 1.0,
        "geometry": geometry,
        "all_samples_finite": bool(np.all(np.isfinite(output))),
        "maximum_absolute_pressure": float(np.max(np.abs(output))),
        "early_rms": early_rms,
        "tail_rms": tail_rms,
        "tail_to_early_rms_ratio": tail_rms / max(early_rms, 1e-30),
    }


def main() -> None:
    args = _parse_args()
    full_room = json.loads(
        args.full_room_report.read_text(encoding="utf-8")
    )
    harmonic_report = json.loads(
        args.harmonic_candidate_report.read_text(encoding="utf-8")
    )
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(
        args.boundary_config
    )
    model = _uniform_model(boundary_config)
    base_case = full_room["cases"][0]
    base_sample_rate_hz = float(
        base_case["fdtd"]["sample_rate_hz"]
    )
    base_grid_spacing_m = float(
        np.mean(base_case["fdtd"]["grid_spacing_xyz_m"])
    )
    frequencies_hz = np.linspace(
        float(harmonic_report["frequency_band_hz"][0]),
        float(harmonic_report["frequency_band_hz"][1]),
        265,
    )
    time_domain_refinement = []
    for linear_scale in (1.0, 0.5, 0.25):
        probe = _time_domain_probe(
            model,
            scheme=_BOUNDARY_PRESSURE_SCHEME,
            frequencies_hz=frequencies_hz,
            sample_rate_hz=base_sample_rate_hz / linear_scale,
            grid_spacing_m=base_grid_spacing_m * linear_scale,
            spatial_derivative_order=_SPATIAL_DERIVATIVE_ORDER,
            near_wall_closure=_NEAR_WALL_CLOSURE,
        )
        time_domain_refinement.append(
            {
                "linear_scale": linear_scale,
                **probe,
            }
        )
    stability = _long_stability_probe(
        model,
        sample_rate_hz=base_sample_rate_hz,
        grid_spacing_m=base_grid_spacing_m,
    )
    harmonic_candidate_accepted = bool(
        _CANDIDATE_ID
        in harmonic_report["acceptance"]["accepted_harmonic_candidates"]
    )
    time_domain_equation_accepted = bool(
        all(
            probe["measured_vs_predicted_discrete_cross_ratio"][
                "maximum_complex_error"
            ]
            <= float(args.maximum_time_domain_complex_error)
            and probe["measured_vs_predicted_discrete_cross_ratio"][
                "maximum_phase_error_deg"
            ]
            <= float(args.maximum_time_domain_phase_error_deg)
            for probe in time_domain_refinement
        )
    )
    cfl_accepted = bool(
        all(
            probe["geometry"]["one_dimensional_cfl_number"] < 1.0
            for probe in time_domain_refinement
        )
    )
    stability_accepted = bool(
        stability["all_samples_finite"]
        and stability["tail_to_early_rms_ratio"]
        <= float(args.maximum_tail_to_early_rms_ratio)
    )
    one_dimensional_prototype_accepted = bool(
        harmonic_candidate_accepted
        and time_domain_equation_accepted
        and cfl_accepted
        and stability_accepted
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "boundary_config": str(args.boundary_config),
        "boundary_reference_id": boundary_config.reference_id,
        "full_room_report": str(args.full_room_report),
        "harmonic_candidate_report": str(
            args.harmonic_candidate_report
        ),
        "candidate_id": _CANDIDATE_ID,
        "frequency_band_hz": [
            float(frequencies_hz[0]),
            float(frequencies_hz[-1]),
        ],
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_report"
        },
        "prototype": {
            "spatial_derivative_order": _SPATIAL_DERIVATIVE_ORDER,
            "interior_stencil": (
                "9/8*(f[i]-f[i-1])-1/24*(f[i+1]-f[i-2])"
            ),
            "near_wall_closure": _NEAR_WALL_CLOSURE,
            "near_wall_derivative_weights": [
                -23.0 / 24.0,
                7.0 / 8.0,
                1.0 / 8.0,
                -1.0 / 24.0,
            ],
            "boundary_pressure_scheme": _BOUNDARY_PRESSURE_SCHEME,
            "quadratic_face_time_weights": [
                15.0 / 8.0,
                -5.0 / 4.0,
                3.0 / 8.0,
            ],
        },
        "time_domain_grid_refinement": time_domain_refinement,
        "long_stability_probe": stability,
        "acceptance": {
            "maximum_time_domain_complex_error": float(
                args.maximum_time_domain_complex_error
            ),
            "maximum_time_domain_phase_error_deg": float(
                args.maximum_time_domain_phase_error_deg
            ),
            "maximum_tail_to_early_rms_ratio": float(
                args.maximum_tail_to_early_rms_ratio
            ),
            "harmonic_candidate_accepted": harmonic_candidate_accepted,
            "time_domain_equation_accepted": (
                time_domain_equation_accepted
            ),
            "one_dimensional_cfl_accepted": cfl_accepted,
            "long_stability_accepted": stability_accepted,
            "one_dimensional_time_domain_prototype_accepted": (
                one_dimensional_prototype_accepted
            ),
            "three_dimensional_reference_accepted": False,
        },
        "scope": {
            "normal_incidence_only": True,
            "one_dimensional_only": True,
            "oblique_time_domain_validated": False,
            "three_dimensional_time_domain_implemented": False,
            "production_default_changed": False,
        },
        "decision": (
            "promote_to_three_dimensional_prototype"
            if one_dimensional_prototype_accepted
            else "reject_one_dimensional_near_wall_closure"
        ),
        "next_action": (
            "implement the same fourth-order interior, one-sided closure, "
            "and quadratic boundary history in an opt-in 3D FDTD reference; "
            "then validate oblique time-domain reflection before rerunning "
            "the full-room crossover"
            if one_dimensional_prototype_accepted
            else "derive an energy-stable staggered-grid boundary closure"
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
                "time_domain_grid_refinement": [
                    {
                        "linear_scale": probe["linear_scale"],
                        **probe[
                            "measured_vs_predicted_discrete_cross_ratio"
                        ],
                    }
                    for probe in time_domain_refinement
                ],
                "long_stability_probe": stability,
                "acceptance": report["acceptance"],
                "decision": report["decision"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
