#!/usr/bin/env python3
"""Validate the M3.1 PathEvent geometry and scalar fractional-delay renderer."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from puresound.audio.acoustic_impedance import FirstOrderRelaxationAdmittance
from puresound.audio.rir_path_events import (
    FRACTIONAL_DELAY_POLICY,
    PATH_EVENT_SCHEMA_VERSION,
    PATH_EVENT_SET_SCHEMA_VERSION,
    PathEventSet,
    causal_fractional_delay_kernel,
    generate_shoebox_path_events,
    render_path_events,
)
from puresound.audio.rir_source_convention import (
    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
)


REPORT_SCHEMA_VERSION = "puresound.path_event_validation.v1"
BOUNDARIES = {
    "west": (0, False, np.asarray([-1.0, 0.0, 0.0])),
    "east": (0, True, np.asarray([1.0, 0.0, 0.0])),
    "south": (1, False, np.asarray([0.0, -1.0, 0.0])),
    "north": (1, True, np.asarray([0.0, 1.0, 0.0])),
    "floor": (2, False, np.asarray([0.0, 0.0, -1.0])),
    "ceiling": (2, True, np.asarray([0.0, 0.0, 1.0])),
}


@dataclass(frozen=True)
class GeometryCase:
    case_id: str
    dimensions_m: tuple[float, float, float]
    source_position_m: tuple[float, float, float]
    receiver_position_m: tuple[float, float, float]


CASES = (
    GeometryCase(
        "ordinary_room",
        (5.0, 4.0, 3.0),
        (1.1, 1.3, 1.2),
        (3.8, 2.9, 1.6),
    ),
    GeometryCase(
        "asymmetric_room",
        (7.1, 3.6, 2.7),
        (0.7, 2.8, 0.4),
        (6.2, 0.6, 2.2),
    ),
    GeometryCase(
        "near_grazing_paths",
        (4.2, 3.7, 2.5),
        (0.18, 0.25, 1.95),
        (3.91, 3.31, 2.08),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate exact shoebox PathEvent geometry, reciprocity, continuity, "
            "serialization, and causal forward-Lagrange fractional delay."
        )
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--sound-speed-m-s", type=float, default=343.0)
    parser.add_argument(
        "--maximum-geometry-error-m",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--maximum-reciprocity-error",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--maximum-fractional-delay-magnitude-error-db",
        type=float,
        default=0.11,
    )
    parser.add_argument(
        "--maximum-fractional-delay-phase-error-deg",
        type=float,
        default=0.60,
    )
    return parser.parse_args()


def _generate(
    case: GeometryCase,
    sound_speed_m_s: float,
    *,
    source_position_m: tuple[float, float, float] | np.ndarray | None = None,
    reverse: bool = False,
    boundary_model: FirstOrderRelaxationAdmittance | None = None,
) -> PathEventSet:
    source = np.asarray(
        case.source_position_m if source_position_m is None else source_position_m,
        dtype=np.float64,
    )
    receiver = np.asarray(case.receiver_position_m, dtype=np.float64)
    if reverse:
        source, receiver = receiver, source
    models = (
        {boundary: boundary_model for boundary in BOUNDARIES}
        if boundary_model is not None
        else None
    )
    return generate_shoebox_path_events(
        dimensions_m=case.dimensions_m,
        source_position_m=source,
        receiver_position_m=receiver,
        sound_speed_m_s=sound_speed_m_s,
        scene_id=case.case_id,
        source_id="receiver" if reverse else "source",
        receiver_id="source" if reverse else "receiver",
        boundary_admittance_models=models,
        reflection_frequencies_hz=(60.0, 80.0, 120.0, 160.0, 200.0, 240.0),
    )


def _path_key(event) -> str:
    return event.surface_ids[0] if event.surface_ids else "direct"


def _geometry_and_reciprocity_case(
    case: GeometryCase,
    sound_speed_m_s: float,
    boundary_model: FirstOrderRelaxationAdmittance,
) -> dict[str, object]:
    event_set = _generate(
        case,
        sound_speed_m_s,
        boundary_model=boundary_model,
    )
    reverse = _generate(
        case,
        sound_speed_m_s,
        reverse=True,
        boundary_model=boundary_model,
    )
    dimensions = np.asarray(case.dimensions_m, dtype=np.float64)
    source = np.asarray(case.source_position_m, dtype=np.float64)
    receiver = np.asarray(case.receiver_position_m, dtype=np.float64)
    distance_errors = []
    delay_errors = []
    boundary_plane_errors = []
    reflection_law_errors = []
    gain_formula_errors = []
    for event in event_set.events:
        if event.path_type == "direct":
            expected_distance = float(np.linalg.norm(receiver - source))
        else:
            boundary = event.surface_ids[0]
            axis, upper, normal = BOUNDARIES[boundary]
            plane = float(dimensions[axis]) if upper else 0.0
            image = source.copy()
            image[axis] = 2.0 * plane - image[axis]
            expected_distance = float(np.linalg.norm(receiver - image))
            point = np.asarray(event.interaction_points_m[0])
            incidence = abs(
                float(
                    np.dot(
                        np.asarray(event.departure_direction_unit),
                        normal,
                    )
                )
            )
            outgoing = abs(
                float(
                    np.dot(
                        np.asarray(event.arrival_direction_unit),
                        normal,
                    )
                )
            )
            boundary_plane_errors.append(abs(float(point[axis]) - plane))
            reflection_law_errors.append(abs(incidence - outgoing))
            expected_gain = np.asarray(
                [
                    (
                        (incidence - boundary_model.normalized_admittance(frequency))
                        / (
                            incidence
                            + boundary_model.normalized_admittance(frequency)
                        )
                        / event.distance_m
                    )
                    for frequency in event.gain_spectrum.frequencies_hz
                ],
                dtype=np.complex128,
            )
            gain_formula_errors.extend(
                np.abs(event.gain_spectrum.values - expected_gain).tolist()
            )
        distance_errors.append(abs(event.distance_m - expected_distance))
        delay_errors.append(
            abs(event.delay_s - expected_distance / sound_speed_m_s)
        )

    forward_by_key = {_path_key(event): event for event in event_set.events}
    reverse_by_key = {_path_key(event): event for event in reverse.events}
    reciprocity_distance_errors = []
    reciprocity_point_errors = []
    reciprocity_direction_errors = []
    reciprocity_gain_errors = []
    for key, event in forward_by_key.items():
        reciprocal = reverse_by_key[key]
        reciprocity_distance_errors.append(
            abs(event.distance_m - reciprocal.distance_m)
        )
        if event.interaction_points_m:
            reciprocity_point_errors.append(
                float(
                    np.max(
                        np.abs(
                            np.asarray(event.interaction_points_m)
                            - np.asarray(reciprocal.interaction_points_m)
                        )
                    )
                )
            )
        reciprocity_direction_errors.extend(
            [
                float(
                    np.max(
                        np.abs(
                            np.asarray(reciprocal.departure_direction_unit)
                            + np.asarray(event.arrival_direction_unit)
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            np.asarray(reciprocal.arrival_direction_unit)
                            + np.asarray(event.departure_direction_unit)
                        )
                    )
                ),
            ]
        )
        reciprocity_gain_errors.append(
            float(
                np.max(
                    np.abs(
                        reciprocal.gain_spectrum.values
                        - event.gain_spectrum.values
                    )
                )
            )
        )

    round_trip_equal = (
        PathEventSet.from_json(event_set.to_json()).to_dict()
        == event_set.to_dict()
    )
    return {
        "case_id": case.case_id,
        "event_count": len(event_set.events),
        "direct_event_count": sum(
            event.path_type == "direct" for event in event_set.events
        ),
        "first_order_reflection_count": sum(
            event.path_type == "specular_reflection"
            for event in event_set.events
        ),
        "maximum_distance_error_m": float(max(distance_errors)),
        "maximum_delay_error_s": float(max(delay_errors)),
        "maximum_boundary_plane_error_m": float(max(boundary_plane_errors)),
        "maximum_reflection_law_cosine_error": float(
            max(reflection_law_errors)
        ),
        "maximum_complex_gain_formula_error": float(max(gain_formula_errors)),
        "maximum_reciprocity_distance_error_m": float(
            max(reciprocity_distance_errors)
        ),
        "maximum_reciprocity_point_error_m": float(
            max(reciprocity_point_errors)
        ),
        "maximum_reciprocity_direction_component_error": float(
            max(reciprocity_direction_errors)
        ),
        "maximum_reciprocity_complex_gain_error": float(
            max(reciprocity_gain_errors)
        ),
        "maximum_reflection_coefficient_magnitude": float(
            max(
                np.max(np.abs(event.gain_spectrum.values * event.distance_m))
                for event in event_set.events[1:]
            )
        ),
        "schema_round_trip_equal": bool(round_trip_equal),
    }


def _continuity_case(
    case: GeometryCase,
    sound_speed_m_s: float,
    displacement_m: float = 1e-3,
) -> dict[str, object]:
    original = _generate(case, sound_speed_m_s)
    source = np.asarray(case.source_position_m, dtype=np.float64)
    moved_source = source + np.asarray([displacement_m, 0.0, 0.0])
    moved = _generate(
        case,
        sound_speed_m_s,
        source_position_m=moved_source,
    )
    distance_changes = [
        abs(second.distance_m - first.distance_m)
        for first, second in zip(original.events, moved.events)
    ]
    delay_changes = [
        abs(second.delay_s - first.delay_s)
        for first, second in zip(original.events, moved.events)
    ]
    event_ids_stable = [event.event_id for event in original.events] == [
        event.event_id for event in moved.events
    ]
    return {
        "case_id": case.case_id,
        "source_displacement_m": displacement_m,
        "event_ids_stable": bool(event_ids_stable),
        "maximum_path_distance_change_m": float(max(distance_changes)),
        "distance_change_over_displacement": float(
            max(distance_changes) / displacement_m
        ),
        "maximum_delay_change_s": float(max(delay_changes)),
        "delay_change_over_displacement_divided_by_c": float(
            max(delay_changes) / (displacement_m / sound_speed_m_s)
        ),
    }


def _fractional_delay_case(
    sample_rate_hz: float,
    sound_speed_m_s: float,
) -> dict[str, object]:
    frequencies_hz = np.linspace(60.0, 1000.0, 95)
    distances_m = (0.8, 1.0, 1.5, 2.0, 3.0)
    channel_reports = []
    for distance_m in distances_m:
        event_set = generate_shoebox_path_events(
            dimensions_m=(8.0, 8.0, 8.0),
            source_position_m=(4.0 - distance_m, 4.0, 4.0),
            receiver_position_m=(4.0, 4.0, 4.0),
            sound_speed_m_s=sound_speed_m_s,
            scene_id="anechoic",
            source_id=f"source_{distance_m:g}m",
            receiver_id="receiver",
            max_order=0,
        )
        event = event_set.events[0]
        rir = render_path_events(
            event_set,
            sample_rate_hz=sample_rate_hz,
            num_samples=int(math.ceil(0.05 * sample_rate_hz)),
        )
        time_s = np.arange(rir.size, dtype=np.float64) / sample_rate_hz
        response = np.asarray(
            [
                np.sum(rir * np.exp(-2j * math.pi * frequency * time_s))
                for frequency in frequencies_hz
            ],
            dtype=np.complex128,
        )
        ideal = (
            np.exp(-2j * math.pi * frequencies_hz * event.delay_s)
            / event.distance_m
        )
        ratio = response / ideal
        magnitude_error_db = 20.0 * np.log10(
            np.maximum(np.abs(ratio), 1e-15)
        )
        phase_error_deg = np.rad2deg(np.angle(ratio))
        first_sample = int(math.floor(event.delay_s * sample_rate_hz))
        channel_reports.append(
            {
                "distance_m": distance_m,
                "fractional_delay_samples": float(
                    event.delay_s * sample_rate_hz
                    - math.floor(event.delay_s * sample_rate_hz)
                ),
                "first_allowed_sample": first_sample,
                "pre_arrival_nonzero_sample_count": int(
                    np.count_nonzero(rir[:first_sample])
                ),
                "dc_gain_error": float(
                    abs(np.sum(rir) - 1.0 / event.distance_m)
                ),
                "maximum_magnitude_error_db": float(
                    np.max(np.abs(magnitude_error_db))
                ),
                "maximum_phase_error_deg": float(
                    np.max(np.abs(phase_error_deg))
                ),
            }
        )

    kernel_scan = []
    for fraction in np.linspace(0.05, 0.95, 19):
        delay_samples = 10.0 + float(fraction)
        start, kernel = causal_fractional_delay_kernel(delay_samples, order=3)
        omega = 2.0 * math.pi * frequencies_hz / sample_rate_hz
        response = np.exp(-1j * omega * start) * (
            np.exp(-1j * omega[:, None] * np.arange(kernel.size)) @ kernel
        )
        ratio = response / np.exp(-1j * omega * delay_samples)
        kernel_scan.append(
            {
                "fractional_delay_samples": float(fraction),
                "maximum_magnitude_error_db": float(
                    np.max(
                        np.abs(
                            20.0
                            * np.log10(
                                np.maximum(np.abs(ratio), 1e-15)
                            )
                        )
                    )
                ),
                "maximum_phase_error_deg": float(
                    np.max(np.abs(np.rad2deg(np.angle(ratio))))
                ),
                "dc_sum_error": float(abs(np.sum(kernel) - 1.0)),
            }
        )
    return {
        "sample_rate_hz": sample_rate_hz,
        "validation_band_hz": [60.0, 1000.0],
        "channels": channel_reports,
        "worst_channel_magnitude_error_db": float(
            max(item["maximum_magnitude_error_db"] for item in channel_reports)
        ),
        "worst_channel_phase_error_deg": float(
            max(item["maximum_phase_error_deg"] for item in channel_reports)
        ),
        "maximum_pre_arrival_nonzero_sample_count": int(
            max(
                item["pre_arrival_nonzero_sample_count"]
                for item in channel_reports
            )
        ),
        "maximum_dc_gain_error": float(
            max(item["dc_gain_error"] for item in channel_reports)
        ),
        "fraction_scan": kernel_scan,
        "worst_fraction_scan_magnitude_error_db": float(
            max(item["maximum_magnitude_error_db"] for item in kernel_scan)
        ),
        "worst_fraction_scan_phase_error_deg": float(
            max(item["maximum_phase_error_deg"] for item in kernel_scan)
        ),
        "maximum_fraction_scan_dc_sum_error": float(
            max(item["dc_sum_error"] for item in kernel_scan)
        ),
    }


def main() -> None:
    args = _parse_args()
    sound_speed = float(args.sound_speed_m_s)
    boundary_model = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=0.04,
        normalized_admittance_relaxation=0.25,
        relaxation_frequency_hz=140.0,
    )
    geometry_cases = [
        _geometry_and_reciprocity_case(
            case,
            sound_speed,
            boundary_model,
        )
        for case in CASES
    ]
    continuity_cases = [
        _continuity_case(case, sound_speed) for case in CASES
    ]
    fractional_delay_cases = [
        _fractional_delay_case(sample_rate, sound_speed)
        for sample_rate in (8000.0, 16000.0, 48000.0)
    ]

    geometry_accepted = all(
        case["event_count"] == 7
        and case["direct_event_count"] == 1
        and case["first_order_reflection_count"] == 6
        and case["maximum_distance_error_m"]
        <= float(args.maximum_geometry_error_m)
        and case["maximum_boundary_plane_error_m"]
        <= float(args.maximum_geometry_error_m)
        and case["maximum_reflection_law_cosine_error"]
        <= float(args.maximum_geometry_error_m)
        and case["maximum_complex_gain_formula_error"]
        <= float(args.maximum_reciprocity_error)
        and case["schema_round_trip_equal"]
        for case in geometry_cases
    )
    reciprocity_accepted = all(
        case["maximum_reciprocity_distance_error_m"]
        <= float(args.maximum_reciprocity_error)
        and case["maximum_reciprocity_point_error_m"]
        <= float(args.maximum_reciprocity_error)
        and case["maximum_reciprocity_direction_component_error"]
        <= float(args.maximum_reciprocity_error)
        and case["maximum_reciprocity_complex_gain_error"]
        <= float(args.maximum_reciprocity_error)
        for case in geometry_cases
    )
    continuity_accepted = all(
        case["event_ids_stable"]
        and case["distance_change_over_displacement"] <= 1.0 + 1e-10
        and case["delay_change_over_displacement_divided_by_c"]
        <= 1.0 + 1e-10
        for case in continuity_cases
    )
    fractional_delay_accepted = all(
        case["maximum_pre_arrival_nonzero_sample_count"] == 0
        and case["maximum_dc_gain_error"] <= 1e-12
        and case["maximum_fraction_scan_dc_sum_error"] <= 1e-12
        and case["worst_fraction_scan_magnitude_error_db"]
        <= float(args.maximum_fractional_delay_magnitude_error_db)
        and case["worst_fraction_scan_phase_error_deg"]
        <= float(args.maximum_fractional_delay_phase_error_deg)
        for case in fractional_delay_cases
    )
    passive_boundary_accepted = all(
        case["maximum_reflection_coefficient_magnitude"] <= 1.0 + 1e-12
        for case in geometry_cases
    )
    accepted = bool(
        geometry_accepted
        and reciprocity_accepted
        and continuity_accepted
        and fractional_delay_accepted
        and passive_boundary_accepted
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "path_event_schema_version": PATH_EVENT_SCHEMA_VERSION,
        "path_event_set_schema_version": PATH_EVENT_SET_SCHEMA_VERSION,
        "source_convention": FREE_FIELD_1_OVER_R_RIR_CONVENTION,
        "fractional_delay_policy": FRACTIONAL_DELAY_POLICY,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_report"
        },
        "geometry_and_reciprocity_cases": geometry_cases,
        "nearby_position_continuity_cases": continuity_cases,
        "fractional_delay_cases": fractional_delay_cases,
        "acceptance": {
            "maximum_geometry_error_m": float(
                args.maximum_geometry_error_m
            ),
            "maximum_reciprocity_error": float(
                args.maximum_reciprocity_error
            ),
            "fractional_delay_validation_band_hz": [60.0, 1000.0],
            "maximum_fractional_delay_magnitude_error_db": float(
                args.maximum_fractional_delay_magnitude_error_db
            ),
            "maximum_fractional_delay_phase_error_deg": float(
                args.maximum_fractional_delay_phase_error_deg
            ),
            "geometry_accepted": bool(geometry_accepted),
            "reciprocity_accepted": bool(reciprocity_accepted),
            "nearby_position_continuity_accepted": bool(continuity_accepted),
            "fractional_delay_accepted": bool(fractional_delay_accepted),
            "passive_boundary_spectrum_accepted": bool(
                passive_boundary_accepted
            ),
            "accepted": accepted,
        },
        "scope": {
            "validated": [
                "versioned PathEvent and PathEventSet JSON round trip",
                "exact shoebox direct and six first-order image paths",
                "specular reflection law and locally reacting complex gain",
                "source-receiver reciprocity",
                "one-millimeter local path continuity",
                "causal scalar forward-Lagrange fractional-delay rendering",
                "8/16/48 kHz response from 60 to 1000 Hz",
            ],
            "complex_boundary_filter_time_realization_validated": False,
            "higher_order_paths_validated": False,
            "mesh_visibility_validated": False,
            "early_c50_against_measured_rooms_validated": False,
            "m2_12_full_room_crossover_rerun": False,
            "production_hybrid_backend_integration_validated": False,
        },
        "decision": (
            "accept_m3_1_path_event_geometry_and_scalar_renderer"
            if accepted
            else "reject_m3_1_and_fix_before_backend_integration"
        ),
        "next_action": (
            "realize each angle-aware complex boundary spectrum as a passive "
            "causal digital filter, render the same PathEvents coherently, and "
            "then rerun the frozen M2.12 full-room crossover protocol"
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
                "accepted": accepted,
                "maximum_geometry_error_m": max(
                    case["maximum_distance_error_m"]
                    for case in geometry_cases
                ),
                "maximum_reciprocity_error": max(
                    max(
                        case["maximum_reciprocity_distance_error_m"],
                        case["maximum_reciprocity_point_error_m"],
                        case[
                            "maximum_reciprocity_direction_component_error"
                        ],
                        case["maximum_reciprocity_complex_gain_error"],
                    )
                    for case in geometry_cases
                ),
                "maximum_fractional_delay_magnitude_error_db": max(
                    case["worst_fraction_scan_magnitude_error_db"]
                    for case in fractional_delay_cases
                ),
                "maximum_fractional_delay_phase_error_deg": max(
                    case["worst_fraction_scan_phase_error_deg"]
                    for case in fractional_delay_cases
                ),
            },
            indent=2,
        )
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
