#!/usr/bin/env python3
"""Assemble the frozen M3 implementation and exit-gate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REPORT_SCHEMA_VERSION = "puresound.m3_exit.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    config = Path("egs/rir_generation/phases/m3_wave_path/reports")
    parser.add_argument(
        "--path-events-report",
        type=Path,
        default=config / "path_events_m3_1_report.json",
    )
    parser.add_argument(
        "--fdtd-holdouts-report",
        type=Path,
        default=config / "fourth_order_fdtd_holdouts_m3_10_report.json",
    )
    parser.add_argument(
        "--crossover-report",
        type=Path,
        default=config / "full_room_crossover_m3_11_report.json",
    )
    parser.add_argument(
        "--attribution-report",
        type=Path,
        default=config / "direct_early_later_m3_11_report.json",
    )
    parser.add_argument(
        "--mesh-report",
        type=Path,
        default=config / "mesh_engine_m3_12_report.json",
    )
    parser.add_argument(
        "--scene-interactions-report",
        type=Path,
        default=config / "scene_interactions_m3_13_m3_14_report.json",
    )
    parser.add_argument(
        "--measured-exit-report",
        type=Path,
        default=config / "m3_measured_exit_report.json",
    )
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read M3 evidence report {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"M3 evidence report must contain an object: {path}")
    return value


def main() -> None:
    args = _parse_args()
    path_events = _load(args.path_events_report)
    fdtd_holdouts = _load(args.fdtd_holdouts_report)
    crossover = _load(args.crossover_report)
    attribution = _load(args.attribution_report)
    mesh = _load(args.mesh_report)
    interactions = _load(args.scene_interactions_report)
    measured = _load(args.measured_exit_report)

    implementation_steps = {
        "m3_10_expanded_fdtd_holdouts": bool(
            fdtd_holdouts["acceptance"][
                "expanded_three_dimensional_reference_accepted"
            ]
            and fdtd_holdouts["acceptance"][
                "multi_axis_azimuth_plane_modes_accepted"
            ]
            and fdtd_holdouts["acceptance"][
                "face_edge_corner_reciprocity_accepted"
            ]
        ),
        "m3_11_crossover_interaction_audited": bool(
            crossover["acceptance"]["protocol_completed"]
            and attribution["acceptance"][
                "m3_4_attribution_protocol_accepted"
            ]
        ),
        "m3_12_existing_mesh_engine_evaluated": bool(
            mesh["acceptance"][
                "evaluation_completed_before_new_geometry_code"
            ]
            and mesh["acceptance"]["existing_engine_smoke_accepted"]
        ),
        "m3_13_furniture_visibility_geometry": bool(
            interactions["acceptance"]["visibility_accepted"]
        ),
        "m3_14_physical_interactions": bool(
            interactions["acceptance"]["transmission_accepted"]
            and interactions["acceptance"]["diffraction_accepted"]
            and interactions["acceptance"][
                "controlled_scattering_accepted"
            ]
            and interactions["acceptance"]["serialization_accepted"]
            and interactions["acceptance"]["rendering_accepted"]
        ),
    }
    exit_gate = {
        "path_delays_match_geometry": bool(
            path_events["acceptance"]["geometry_accepted"]
        ),
        "nearby_positions_produce_continuous_paths": bool(
            path_events["acceptance"][
                "nearby_position_continuity_accepted"
            ]
        ),
        "reciprocity_obeys_expected_scope": bool(
            path_events["acceptance"]["reciprocity_accepted"]
            and interactions["acceptance"]["reciprocity_accepted"]
            and fdtd_holdouts["acceptance"][
                "face_edge_corner_reciprocity_accepted"
            ]
        ),
        "early_reflection_timing_improves_against_measured": bool(
            measured["acceptance"]["early_timing_gap_improved"]
        ),
        "c50_improves_against_measured": bool(
            measured["acceptance"]["c50_gap_improved"]
        ),
    }
    implementation_complete = all(implementation_steps.values())
    exit_accepted = all(exit_gate.values())

    continuity_cases = path_events["nearby_position_continuity_cases"]
    geometry_cases = path_events["geometry_and_reciprocity_cases"]
    measured_gap = measured["aggregate_gap"]
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "evidence_reports": {
            key.removesuffix("_report"): str(value)
            for key, value in vars(args).items()
            if key.endswith("_report") and key != "output_report"
        },
        "implementation_steps": implementation_steps,
        "exit_gate": exit_gate,
        "evidence_summary": {
            "maximum_path_geometry_error_m": max(
                case["maximum_distance_error_m"]
                for case in geometry_cases
            ),
            "maximum_path_delay_error_s": max(
                case["maximum_delay_error_s"]
                for case in geometry_cases
            ),
            "maximum_continuity_distance_change_over_displacement": max(
                case["distance_change_over_displacement"]
                for case in continuity_cases
            ),
            "maximum_path_reciprocity_error": max(
                case["maximum_reciprocity_distance_error_m"]
                for case in geometry_cases
            ),
            "maximum_fdtd_plane_mode_complex_error": (
                fdtd_holdouts["summary"][
                    "maximum_plane_mode_complex_error"
                ]
            ),
            "maximum_reciprocalized_fdtd_nrmse": (
                fdtd_holdouts["summary"][
                    "maximum_reciprocalized_nrmse"
                ]
            ),
            "c50_mean_absolute_median_gap": {
                "m1": measured_gap["c50_db"][
                    "m1_mean_absolute_median_gap"
                ],
                "m3": measured_gap["c50_db"][
                    "m3_mean_absolute_median_gap"
                ],
                "relative_improvement": measured_gap["c50_db"][
                    "relative_improvement"
                ],
            },
            "early_energy_centroid_mean_absolute_median_gap_ms": {
                "m1": measured_gap[
                    "early_reflection_energy_centroid_ms"
                ]["m1_mean_absolute_median_gap"],
                "m3": measured_gap[
                    "early_reflection_energy_centroid_ms"
                ]["m3_mean_absolute_median_gap"],
                "relative_improvement": measured_gap[
                    "early_reflection_energy_centroid_ms"
                ]["relative_improvement"],
            },
        },
        "known_limits": {
            "full_room_low_high_complex_gate_accepted": bool(
                crossover["acceptance"][
                    "full_room_complex_gate_accepted"
                ]
            ),
            "dominant_peak_timing_is_exit_metric": False,
            "dominant_peak_timing_diagnostic": measured_gap[
                "dominant_early_reflection_delay_ms"
            ],
            "path_event_backend_is_production_default": False,
            "late_field_is_m4_scope": True,
        },
        "acceptance": {
            "all_m3_implementation_steps_completed": implementation_complete,
            "all_m3_exit_gates_accepted": exit_accepted,
            "m3_completed": bool(
                implementation_complete and exit_accepted
            ),
        },
        "decision": (
            "complete_m3_and_advance_to_m4_late_field"
            if implementation_complete and exit_accepted
            else "retain_m3_blocker"
        ),
    }
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.output_report),
                "implementation_steps": implementation_steps,
                "exit_gate": exit_gate,
                "acceptance": report["acceptance"],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    if not report["acceptance"]["m3_completed"]:
        raise RuntimeError("M3 implementation or exit gate remains incomplete")


if __name__ == "__main__":
    main()
