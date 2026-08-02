#!/usr/bin/env python3
"""Build the M3.3 higher-order coherent PathEvent convergence report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from egs.rir_generation.phases.m2_impedance.scripts.validate_full_room_crossover import (
    _complex_metrics,
    _image_source_transfer,
    _shoebox_image_geometry,
)
from puresound.audio.impedance_modes import (
    RectangularImpedanceBoundaryConfig,
)
from puresound.audio.rir_path_events import (
    generate_shoebox_path_events,
)


REPORT_SCHEMA_VERSION = "puresound.higher_order_path_event_validation.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ordered higher-order shoebox PathEvents, quantify image "
            "order convergence, and audit edge/corner diagnostic paths."
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
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=(1, 2, 4, 8, 12),
    )
    parser.add_argument(
        "--maximum-spectral-path-nrmse",
        type=float,
        default=1e-10,
    )
    parser.add_argument(
        "--maximum-time-renderer-nrmse",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--minimum-time-renderer-correlation",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--time-renderer-energy-ratio-range",
        type=float,
        nargs=2,
        default=(0.98, 1.02),
    )
    return parser.parse_args()


def _event_transfer(events, frequencies_hz: np.ndarray) -> np.ndarray:
    output = np.zeros(frequencies_hz.size, dtype=np.complex128)
    for event in events:
        output += event.gain_spectrum.values * np.exp(
            -2j * np.pi * frequencies_hz * event.delay_s
        )
    return output


def _has_simultaneous_interactions(event) -> bool:
    return any(
        current == previous
        for previous, current in zip(
            event.interaction_group_ids,
            event.interaction_group_ids[1:],
        )
    )


def _aggregate_metrics(
    cases: list[dict[str, object]],
    order_key: str,
    metric_key: str,
) -> dict[str, float]:
    metrics = [
        case["orders"][order_key][metric_key] for case in cases
    ]
    return {
        "case_count": len(metrics),
        "mean_complex_nrmse": float(
            np.mean([metric["complex_nrmse"] for metric in metrics])
        ),
        "mean_complex_correlation": float(
            np.mean(
                [metric["complex_correlation"] for metric in metrics]
            )
        ),
        "mean_transfer_energy_ratio": float(
            np.mean(
                [metric["transfer_energy_ratio"] for metric in metrics]
            )
        ),
    }


def main() -> None:
    args = _parse_args()
    orders = tuple(sorted(set(int(value) for value in args.orders)))
    if not orders or orders[0] < 1 or orders[-1] > 20:
        raise ValueError("orders must be unique integers in [1, 20]")
    maximum_order = orders[-1]
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(
        args.boundary_config
    )
    boundary_models = tuple(boundary_config.boundaries.values())
    uniform_model = boundary_models[0]
    if any(
        model.metadata() != uniform_model.metadata()
        for model in boundary_models[1:]
    ):
        raise ValueError("M3.3 convergence currently requires uniform walls")
    full_room = json.loads(
        args.full_room_report.read_text(encoding="utf-8")
    )
    source_order = full_room["diagnostic"]["causal_path_event"]["max_order"]
    if int(source_order) != maximum_order:
        raise ValueError(
            "full-room PathEvent order does not match convergence maximum"
        )
    crossover_hz = float(full_room["configuration"]["crossover_hz"])
    frequencies = np.linspace(
        0.7 * crossover_hz,
        min(300.0, 1.25 * crossover_hz),
        265,
    )
    case_reports = []
    for source_case in full_room["cases"]:
        dimensions = tuple(source_case["case"]["room_dim_m"])
        source_position = tuple(
            source_case["fdtd"]["source_cell_center_m"]
        )
        receiver_position = tuple(
            source_case["fdtd"]["receiver_cell_center_m"]
        )
        event_set = generate_shoebox_path_events(
            dimensions_m=dimensions,
            source_position_m=source_position,
            receiver_position_m=receiver_position,
            sound_speed_m_s=float(
                full_room["configuration"].get("sound_speed_m_s", 343.0)
            ),
            scene_id=source_case["case"]["case_id"],
            source_id="source_0",
            receiver_id="receiver_0",
            max_order=maximum_order,
            edge_corner_policy="sequential_face_product_diagnostic",
            boundary_admittance_models=boundary_config.boundaries,
            reflection_frequencies_hz=frequencies,
        )
        simultaneous_events = [
            event
            for event in event_set.events
            if _has_simultaneous_interactions(event)
        ]
        physical_events = [
            event
            for event in event_set.events
            if not _has_simultaneous_interactions(event)
        ]
        order_reports = {}
        transfers = {}
        analytic_transfers = {}
        for order in orders:
            selected_events = [
                event
                for event in event_set.events
                if sum(abs(value) for value in event.image_order_xyz)
                <= order
            ]
            transfer = _event_transfer(selected_events, frequencies)
            geometry = _shoebox_image_geometry(
                dimensions,
                source_position,
                receiver_position,
                order,
            )
            analytic = _image_source_transfer(
                geometry,
                frequencies,
                uniform_model,
                sound_speed_m_s=343.0,
                reflection_policy="complex_angle",
            )
            transfers[str(order)] = transfer
            analytic_transfers[str(order)] = analytic
            order_reports[str(order)] = {
                "event_count": len(selected_events),
                "analytic_image_count": int(
                    geometry["distances_m"].size
                ),
                "event_spectrum_vs_same_order_analytic": _complex_metrics(
                    analytic,
                    transfer,
                ),
            }
        maximum_analytic = analytic_transfers[str(maximum_order)]
        for order in orders:
            key = str(order)
            order_reports[key][
                "same_order_analytic_vs_max_order_analytic"
            ] = _complex_metrics(
                maximum_analytic,
                analytic_transfers[key],
            )

        direct_delay = min(event.delay_s for event in event_set.events)
        direct_events = [
            event for event in event_set.events if event.path_type == "direct"
        ]
        early_events = [
            event
            for event in event_set.events
            if event.path_type != "direct"
            and event.delay_s <= direct_delay + 0.050
        ]
        later_events = [
            event
            for event in event_set.events
            if event.delay_s > direct_delay + 0.050
        ]
        maximum_transfer = transfers[str(maximum_order)]
        bucket_report = {}
        for name, events in (
            ("direct", direct_events),
            ("reflections_through_50ms_after_direct", early_events),
            ("paths_after_50ms", later_events),
        ):
            component = _event_transfer(events, frequencies)
            bucket_report[name] = {
                "event_count": len(events),
                "component_vs_total_transfer": _complex_metrics(
                    maximum_transfer,
                    component,
                ),
            }
        physical_transfer = _event_transfer(physical_events, frequencies)
        source_realization = source_case["causal_path_event"]
        case_reports.append(
            {
                "case_id": source_case["case"]["case_id"],
                "event_count": len(event_set.events),
                "simultaneous_edge_or_corner_event_count": len(
                    simultaneous_events
                ),
                "physical_exclude_policy_event_count": len(physical_events),
                "orders": order_reports,
                "delay_buckets": bucket_report,
                "physical_exclude_vs_legacy_face_product": _complex_metrics(
                    maximum_transfer,
                    physical_transfer,
                ),
                "time_renderer_vs_max_order_analytic": (
                    source_realization[
                        "time_renderer_vs_analytic_complex_angle"
                    ]
                ),
            }
        )

    order_aggregate = {
        str(order): {
            "event_spectrum_vs_same_order_analytic": _aggregate_metrics(
                case_reports,
                str(order),
                "event_spectrum_vs_same_order_analytic",
            ),
            "same_order_analytic_vs_max_order_analytic": _aggregate_metrics(
                case_reports,
                str(order),
                "same_order_analytic_vs_max_order_analytic",
            ),
        }
        for order in orders
    }
    spectral_representation_accepted = all(
        case["orders"][str(order)][
            "event_spectrum_vs_same_order_analytic"
        ]["complex_nrmse"]
        <= float(args.maximum_spectral_path_nrmse)
        for case in case_reports
        for order in orders
    )
    energy_minimum, energy_maximum = (
        float(value) for value in args.time_renderer_energy_ratio_range
    )
    time_renderer_accepted = all(
        case["time_renderer_vs_max_order_analytic"]["complex_nrmse"]
        <= float(args.maximum_time_renderer_nrmse)
        and case["time_renderer_vs_max_order_analytic"][
            "complex_correlation"
        ]
        >= float(args.minimum_time_renderer_correlation)
        and case["time_renderer_vs_max_order_analytic"][
            "transfer_energy_ratio"
        ]
        >= energy_minimum
        and case["time_renderer_vs_max_order_analytic"][
            "transfer_energy_ratio"
        ]
        <= energy_maximum
        for case in case_reports
    )
    representation_accepted = bool(
        spectral_representation_accepted and time_renderer_accepted
    )
    full_room_gate_accepted = bool(
        full_room["acceptance"]["full_room_complex_gate_accepted"]
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "boundary_config": str(args.boundary_config),
        "boundary_reference_id": boundary_config.reference_id,
        "full_room_report": str(args.full_room_report),
        "orders": list(orders),
        "frequency_band_hz": [
            float(frequencies[0]),
            float(frequencies[-1]),
        ],
        "edge_corner_contract": {
            "production_policy": (
                "exclude simultaneous face crossings pending a physical "
                "edge/corner diffraction model"
            ),
            "legacy_comparison_policy": (
                "group coincident interactions at one point and cascade face "
                "filters only to reproduce the old analytic image product"
            ),
            "legacy_policy_promoted_to_production": False,
        },
        "cases": case_reports,
        "order_aggregate": order_aggregate,
        "full_room_crossover": {
            "causal_path_event_order_aggregate": full_room["diagnostic"][
                "causal_path_event"
            ],
            "frozen_gate_accepted": full_room_gate_accepted,
        },
        "acceptance": {
            "maximum_spectral_path_nrmse": float(
                args.maximum_spectral_path_nrmse
            ),
            "maximum_time_renderer_nrmse": float(
                args.maximum_time_renderer_nrmse
            ),
            "minimum_time_renderer_correlation": float(
                args.minimum_time_renderer_correlation
            ),
            "time_renderer_energy_ratio_range": [
                energy_minimum,
                energy_maximum,
            ],
            "higher_order_spectral_representation_accepted": bool(
                spectral_representation_accepted
            ),
            "higher_order_time_renderer_accepted": bool(
                time_renderer_accepted
            ),
            "m3_3_higher_order_representation_accepted": (
                representation_accepted
            ),
            "full_room_crossover_accepted": full_room_gate_accepted,
        },
        "scope": {
            "validated": [
                "ordered shoebox surface crossings through order 12",
                "higher-order complex-gain products",
                "source-receiver reciprocal image paths",
                "causal repeated boundary-filter cascades",
                "order convergence in three frozen M2.12 rooms",
            ],
            "physical_edge_or_corner_model_validated": False,
            "mesh_visibility_validated": False,
            "production_high_backend_replaced": False,
            "measured_c50_validated": False,
        },
        "decision": (
            "accept_m3_3_representation_reject_higher_order_as_full_room_fix"
            if representation_accepted and not full_room_gate_accepted
            else (
                "accept_m3_3_and_full_room_gate"
                if representation_accepted
                else "reject_m3_3_representation"
            )
        ),
        "next_action": (
            "audit direct/early/late complex energy against the FDTD reference "
            "and evaluate a mesh engine with explicit visibility; adding image "
            "order alone has reproduced, not fixed, the crossover failure"
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
                "m3_3_higher_order_representation_accepted": (
                    representation_accepted
                ),
                "order_convergence": {
                    order: order_aggregate[str(order)][
                        "same_order_analytic_vs_max_order_analytic"
                    ]
                    for order in orders
                },
                "maximum_time_renderer_nrmse": max(
                    case["time_renderer_vs_max_order_analytic"][
                        "complex_nrmse"
                    ]
                    for case in case_reports
                ),
                "full_room_crossover_accepted": full_room_gate_accepted,
            },
            indent=2,
        )
    )
    if not representation_accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
