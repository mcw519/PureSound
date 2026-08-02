#!/usr/bin/env python3
"""Build the M3.4 direct/early/later full-room attribution report."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from egs.rir_generation.phases.m2_impedance.scripts.validate_full_room_crossover import (
    _complex_metrics,
    _sampled_frequency_response,
)
from puresound.audio.fdtd_reference import (
    FDTDReferenceConfig,
    ricker_source,
    simulate_reciprocal_fdtd_reference,
    simulate_fdtd_reference,
)
from puresound.audio.impedance_modes import (
    RectangularImpedanceBoundaryConfig,
)
from puresound.audio.rir_attribution import (
    ATTRIBUTION_SCHEMA_VERSION,
    decompose_direct_early_later,
    reconstruction_error,
)
from puresound.audio.rir_path_events import (
    generate_shoebox_path_events,
    partition_path_events_by_arrival,
    render_path_events,
)
from puresound.audio.rir_source_convention import (
    fdtd_cell_center_position,
    fdtd_pressure_cell_to_free_field_input,
)


REPORT_SCHEMA_VERSION = "puresound.direct_early_later_validation.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Attribute the frozen full-room FDTD versus ordered PathEvent "
            "difference to direct, early-reflection, and later components."
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
    parser.add_argument("--path-event-max-order", type=int, default=12)
    parser.add_argument("--analysis-sample-rate-hz", type=int, default=8000)
    parser.add_argument("--early-window-s", type=float, default=0.050)
    parser.add_argument(
        "--transition-width-s",
        type=float,
        default=0.008,
    )
    parser.add_argument(
        "--maximum-reconstruction-nrmse",
        type=float,
        default=1e-12,
    )
    parser.add_argument(
        "--minimum-relative-source-magnitude",
        type=float,
        default=0.01,
    )
    return parser.parse_args()


def _resample_to_analysis_grid(
    values: np.ndarray,
    *,
    source_sample_rate_hz: float,
    target_sample_rate_hz: int,
    num_target_samples: int,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim != 1 or source.size < 2:
        raise ValueError("resampling input must be a 1D array of length >= 2")
    source_time_s = (
        np.arange(source.size, dtype=np.float64)
        / float(source_sample_rate_hz)
    )
    target_time_s = (
        np.arange(num_target_samples, dtype=np.float64)
        / float(target_sample_rate_hz)
    )
    return np.interp(
        target_time_s,
        source_time_s,
        source,
        left=0.0,
        right=0.0,
    )


def _output_transfer(
    output: np.ndarray,
    input_signal: np.ndarray,
    *,
    sample_rate_hz: float,
    frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, float]:
    input_response = _sampled_frequency_response(
        input_signal,
        sample_rate_hz,
        frequencies_hz,
    )
    input_scale = max(float(np.max(np.abs(input_response))), 1e-30)
    relative_source = np.abs(input_response) / input_scale
    if np.any(relative_source <= 1e-12):
        raise ValueError("source transfer contains a numerical zero")
    output_response = _sampled_frequency_response(
        output,
        sample_rate_hz,
        frequencies_hz,
    )
    return output_response / input_response, float(np.min(relative_source))


def _convolve_input(input_signal: np.ndarray, rir: np.ndarray) -> np.ndarray:
    return fftconvolve(input_signal, rir)[: input_signal.size]


def _component_transfers(
    components: dict[str, np.ndarray],
    input_signal: np.ndarray,
    *,
    sample_rate_hz: float,
    frequencies_hz: np.ndarray,
) -> tuple[dict[str, np.ndarray], float]:
    transfers = {}
    minimum_source = math.inf
    for name, output in components.items():
        transfer, relative_source = _output_transfer(
            output,
            input_signal,
            sample_rate_hz=sample_rate_hz,
            frequencies_hz=frequencies_hz,
        )
        transfers[name] = transfer
        minimum_source = min(minimum_source, relative_source)
    return transfers, float(minimum_source)


def _coherent_error_attribution(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
) -> dict[str, float]:
    early_error = (
        candidate["early_reflections"] - reference["early_reflections"]
    )
    later_error = (
        candidate["later_reflections"] - reference["later_reflections"]
    )
    full_error = candidate["full"] - reference["full"]
    full_squared = max(
        float(np.vdot(full_error, full_error).real),
        1e-30,
    )
    early_squared = float(np.vdot(early_error, early_error).real)
    later_squared = float(np.vdot(later_error, later_error).real)
    cross = float(2.0 * np.vdot(early_error, later_error).real)
    reconstruction = early_error + later_error - full_error
    return {
        "early_error_norm_over_total_error_norm": float(
            math.sqrt(early_squared / full_squared)
        ),
        "later_error_norm_over_total_error_norm": float(
            math.sqrt(later_squared / full_squared)
        ),
        "early_squared_term": early_squared / full_squared,
        "later_squared_term": later_squared / full_squared,
        "early_later_cross_term": cross / full_squared,
        "term_sum": (early_squared + later_squared + cross) / full_squared,
        "error_reconstruction_nrmse": float(
            np.linalg.norm(reconstruction)
            / max(float(np.linalg.norm(full_error)), 1e-30)
        ),
    }


def _case_report(
    source_case: dict[str, Any],
    *,
    full_room: dict[str, Any],
    boundary_config: RectangularImpedanceBoundaryConfig,
    args: argparse.Namespace,
    frequencies_hz: np.ndarray,
) -> dict[str, Any]:
    configuration = full_room["configuration"]
    case = source_case["case"]
    fdtd_reference_profile = configuration.get(
        "fdtd_reference_profile",
        "legacy_second_order",
    )
    fourth_order_reference = (
        fdtd_reference_profile == "fourth_order_reciprocal"
    )
    fdtd_config = FDTDReferenceConfig(
        room_dim_m=tuple(case["room_dim_m"]),
        grid_spacing_m=float(configuration["grid_spacing_m"]),
        duration_s=float(configuration["duration_s"]),
        source_position_m=tuple(case["source_position_m"]),
        receiver_position_m=tuple(case["receiver_position_m"]),
        source_center_hz=float(configuration["source_center_hz"]),
        source_delay_s=float(configuration["source_delay_s"]),
        **(
            {
                "boundary_pressure_scheme": (
                    "face_quadratic_time_quadratic"
                ),
                "spatial_derivative_order": 4,
                "near_wall_closure": "third_order_one_sided",
            }
            if fourth_order_reference
            else {}
        ),
    )
    fdtd_simulator = (
        simulate_reciprocal_fdtd_reference
        if fourth_order_reference
        else simulate_fdtd_reference
    )
    fdtd = fdtd_simulator(
        fdtd_config,
        boundary_admittance=boundary_config.boundaries,
    )
    source_position = fdtd_cell_center_position(
        fdtd.source_cell_zyx,
        fdtd.grid_spacing_xyz_m,
    )
    receiver_position = fdtd_cell_center_position(
        fdtd.receiver_cell_zyx,
        fdtd.grid_spacing_xyz_m,
    )
    if not np.allclose(
        source_position,
        source_case["fdtd"]["source_cell_center_m"],
        rtol=0.0,
        atol=1e-12,
    ) or not np.allclose(
        receiver_position,
        source_case["fdtd"]["receiver_cell_center_m"],
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError("M3.4 FDTD cell centers differ from frozen M3.3")
    source = ricker_source(
        fdtd.rir.size,
        fdtd.time_step_s,
        fdtd_config.source_center_hz,
        fdtd_config.source_delay_s,
    )
    equivalent_input = fdtd_pressure_cell_to_free_field_input(
        source,
        time_step_s=fdtd.time_step_s,
        cell_volume_m3=float(np.prod(fdtd.grid_spacing_xyz_m)),
        sound_speed_m_s=fdtd_config.sound_speed_m_s,
    )
    analysis_sample_rate = int(args.analysis_sample_rate_hz)
    num_samples = round(float(configuration["duration_s"]) * analysis_sample_rate)
    input_signal = _resample_to_analysis_grid(
        equivalent_input,
        source_sample_rate_hz=fdtd.sample_rate_hz,
        target_sample_rate_hz=analysis_sample_rate,
        num_target_samples=num_samples,
    )
    fdtd_output = _resample_to_analysis_grid(
        fdtd.rir,
        source_sample_rate_hz=fdtd.sample_rate_hz,
        target_sample_rate_hz=analysis_sample_rate,
        num_target_samples=num_samples,
    )
    event_set = generate_shoebox_path_events(
        dimensions_m=case["room_dim_m"],
        source_position_m=source_position,
        receiver_position_m=receiver_position,
        sound_speed_m_s=fdtd_config.sound_speed_m_s,
        scene_id=case["case_id"],
        source_id="source_0",
        receiver_id="receiver_0",
        max_order=int(args.path_event_max_order),
        edge_corner_policy="sequential_face_product_diagnostic",
        boundary_admittance_models=boundary_config.boundaries,
        reflection_frequencies_hz=frequencies_hz,
    )
    arrival_buckets = partition_path_events_by_arrival(
        event_set,
        early_window_s=float(args.early_window_s),
    )
    arrival_rirs = {
        name: render_path_events(
            events,
            sample_rate_hz=analysis_sample_rate,
            num_samples=num_samples,
            surface_admittance_models=boundary_config.boundaries,
        )
        for name, events in arrival_buckets.items()
    }
    path_full_rir = sum(
        arrival_rirs.values(),
        np.zeros(num_samples, dtype=np.float64),
    )
    arrival_outputs = {
        name: _convolve_input(input_signal, rir)
        for name, rir in arrival_rirs.items()
    }
    arrival_outputs["early_cumulative"] = (
        arrival_outputs["direct"]
        + arrival_outputs["early_reflections"]
    )
    arrival_outputs["full"] = _convolve_input(input_signal, path_full_rir)
    direct_delay_s = float(arrival_buckets["direct"][0].delay_s)
    split_center_s = (
        float(configuration["source_delay_s"])
        + direct_delay_s
        + float(args.early_window_s)
    )
    fdtd_components = decompose_direct_early_later(
        fdtd_output,
        arrival_outputs["direct"],
        sample_rate_hz=analysis_sample_rate,
        split_center_s=split_center_s,
        transition_width_s=float(args.transition_width_s),
    )
    path_window_components = decompose_direct_early_later(
        arrival_outputs["full"],
        arrival_outputs["direct"],
        sample_rate_hz=analysis_sample_rate,
        split_center_s=split_center_s,
        transition_width_s=float(args.transition_width_s),
    )
    fdtd_transfers, fdtd_minimum_source = _component_transfers(
        fdtd_components,
        input_signal,
        sample_rate_hz=analysis_sample_rate,
        frequencies_hz=frequencies_hz,
    )
    path_window_transfers, path_minimum_source = _component_transfers(
        path_window_components,
        input_signal,
        sample_rate_hz=analysis_sample_rate,
        frequencies_hz=frequencies_hz,
    )
    arrival_transfers, arrival_minimum_source = _component_transfers(
        arrival_outputs,
        input_signal,
        sample_rate_hz=analysis_sample_rate,
        frequencies_hz=frequencies_hz,
    )
    full_reference_norm = max(
        float(np.linalg.norm(fdtd_transfers["full"])),
        1e-30,
    )
    component_names = (
        "direct",
        "early_reflections",
        "later_reflections",
        "early_cumulative",
        "full",
    )
    component_metrics = {
        name: {
            **_complex_metrics(
                fdtd_transfers[name],
                path_window_transfers[name],
            ),
            "fdtd_component_norm_over_fdtd_full_norm": float(
                np.linalg.norm(fdtd_transfers[name]) / full_reference_norm
            ),
            "path_component_norm_over_fdtd_full_norm": float(
                np.linalg.norm(path_window_transfers[name])
                / full_reference_norm
            ),
            "component_error_norm_over_fdtd_full_norm": float(
                np.linalg.norm(
                    path_window_transfers[name] - fdtd_transfers[name]
                )
                / full_reference_norm
            ),
        }
        for name in component_names
    }
    arrival_vs_window = {
        name: _complex_metrics(
            path_window_transfers[name],
            arrival_transfers[name],
        )
        for name in (
            "early_reflections",
            "later_reflections",
            "early_cumulative",
            "full",
        )
    }
    return {
        "case_id": case["case_id"],
        "split": case["split"],
        "fdtd_sample_rate_hz": float(fdtd.sample_rate_hz),
        "fdtd_reference_profile": fdtd_reference_profile,
        "fdtd_reciprocity_averaged": fdtd.reciprocity_averaged,
        "fdtd_raw_reciprocity_nrmse": fdtd.raw_reciprocity_nrmse,
        "analysis_sample_rate_hz": analysis_sample_rate,
        "source_cell_center_m": list(source_position),
        "receiver_cell_center_m": list(receiver_position),
        "direct_delay_s": direct_delay_s,
        "split_center_s": split_center_s,
        "path_event_counts": {
            name: len(events) for name, events in arrival_buckets.items()
        },
        "simultaneous_edge_or_corner_path_count": len(
            event_set.metadata[
                "included_diagnostic_edge_or_corner_image_orders"
            ]
        ),
        "minimum_relative_source_magnitude": min(
            fdtd_minimum_source,
            path_minimum_source,
            arrival_minimum_source,
        ),
        "reconstruction": {
            "fdtd": reconstruction_error(fdtd_components),
            "path_time_window": reconstruction_error(
                path_window_components
            ),
            "path_arrival_bucket_time_domain_nrmse": float(
                np.linalg.norm(
                    arrival_outputs["direct"]
                    + arrival_outputs["early_reflections"]
                    + arrival_outputs["later_reflections"]
                    - arrival_outputs["full"]
                )
                / max(
                    float(np.linalg.norm(arrival_outputs["full"])),
                    1e-30,
                )
            ),
        },
        "component_metrics_path_time_window_vs_fdtd": component_metrics,
        "path_arrival_bucket_vs_path_time_window": arrival_vs_window,
        "coherent_full_error_attribution": _coherent_error_attribution(
            fdtd_transfers,
            path_window_transfers,
        ),
    }


def _aggregate_component(
    cases: list[dict[str, Any]],
    component: str,
) -> dict[str, float]:
    metrics = [
        case["component_metrics_path_time_window_vs_fdtd"][component]
        for case in cases
    ]
    keys = tuple(metrics[0])
    return {
        "case_count": len(metrics),
        **{
            f"mean_{key}": float(np.mean([metric[key] for metric in metrics]))
            for key in keys
        },
    }


def main() -> None:
    args = _parse_args()
    if not 1 <= int(args.path_event_max_order) <= 20:
        raise ValueError("path-event max order must be in [1, 20]")
    if int(args.analysis_sample_rate_hz) < 2000:
        raise ValueError("analysis sample rate must be at least 2000 Hz")
    if float(args.early_window_s) <= 0.0:
        raise ValueError("early window must be positive")
    if not 0.0 < float(args.transition_width_s) < float(
        args.early_window_s
    ):
        raise ValueError("transition width must be in (0, early window)")
    full_room = json.loads(
        args.full_room_report.read_text(encoding="utf-8")
    )
    source_order = full_room["diagnostic"]["causal_path_event"]["max_order"]
    if int(source_order) != int(args.path_event_max_order):
        raise ValueError("M3.3 report and M3.4 path orders differ")
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(
        args.boundary_config
    )
    crossover_hz = float(full_room["configuration"]["crossover_hz"])
    frequencies_hz = np.linspace(
        0.7 * crossover_hz,
        min(300.0, 1.25 * crossover_hz),
        265,
    )
    cases = [
        _case_report(
            source_case,
            full_room=full_room,
            boundary_config=boundary_config,
            args=args,
            frequencies_hz=frequencies_hz,
        )
        for source_case in full_room["cases"]
    ]
    components = (
        "direct",
        "early_reflections",
        "later_reflections",
        "early_cumulative",
        "full",
    )
    aggregate = {
        component: _aggregate_component(cases, component)
        for component in components
    }
    reconstruction_accepted = all(
        case["reconstruction"][method]["nrmse"]
        <= float(args.maximum_reconstruction_nrmse)
        for case in cases
        for method in ("fdtd", "path_time_window")
    ) and all(
        case["reconstruction"]["path_arrival_bucket_time_domain_nrmse"]
        <= float(args.maximum_reconstruction_nrmse)
        for case in cases
    )
    source_band_accepted = all(
        case["minimum_relative_source_magnitude"]
        >= float(args.minimum_relative_source_magnitude)
        for case in cases
    )
    mean_early_error = aggregate["early_reflections"][
        "mean_component_error_norm_over_fdtd_full_norm"
    ]
    mean_later_error = aggregate["later_reflections"][
        "mean_component_error_norm_over_fdtd_full_norm"
    ]
    dominant_component = (
        "early_reflections"
        if mean_early_error >= mean_later_error
        else "later_reflections"
    )
    coherent_attribution_aggregate = {
        key: float(
            np.mean(
                [
                    case["coherent_full_error_attribution"][key]
                    for case in cases
                ]
            )
        )
        for key in cases[0]["coherent_full_error_attribution"]
    }
    protocol_accepted = bool(
        reconstruction_accepted and source_band_accepted
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "attribution_schema_version": ATTRIBUTION_SCHEMA_VERSION,
        "boundary_config": str(args.boundary_config),
        "boundary_reference_id": boundary_config.reference_id,
        "full_room_report": str(args.full_room_report),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_report"
        },
        "frequency_band_hz": [
            float(frequencies_hz[0]),
            float(frequencies_hz[-1]),
        ],
        "decomposition_contract": {
            "direct_anchor": (
                "validated free-field 1/r PathEvent convolved with the exact "
                "FDTD-equivalent band-limited input"
            ),
            "why_direct_is_not_time_gated": (
                "the 180 Hz source pulse is wider than the approximately "
                "0.7-1.1 ms direct-to-first-reflection separation"
            ),
            "fdtd_early_later": (
                "full minus direct anchor, split by exactly complementary "
                "raised-cosine output masks"
            ),
            "path_primary_comparison": (
                "the complete PathEvent output is split by the same masks"
            ),
            "path_geometry_audit": (
                "events are also partitioned by arrival <= or > direct+50ms; "
                "filter tails remain owned by their generating event"
            ),
            "component_norms_are_additive_energies": False,
            "reason": (
                "complex pressure components interfere; squared self terms "
                "plus the reported cross term reconstruct total error energy"
            ),
        },
        "cases": cases,
        "aggregate": aggregate,
        "diagnostic": {
            "dominant_mean_component_error": dominant_component,
            "mean_early_component_error_norm_over_fdtd_full_norm": (
                mean_early_error
            ),
            "mean_later_component_error_norm_over_fdtd_full_norm": (
                mean_later_error
            ),
            "coherent_error_attribution_aggregate": (
                coherent_attribution_aggregate
            ),
            "interpretation": (
                "the early-reflection field has similar standalone norm and "
                "high correlation, but its remaining complex error disrupts "
                "the strong direct/early cancellation; later response has "
                "large component-relative NRMSE but a much smaller norm and "
                "contribution to the full-transfer error"
            ),
            "full_room_crossover_accepted": bool(
                full_room["acceptance"][
                    "full_room_complex_gate_accepted"
                ]
            ),
        },
        "acceptance": {
            "maximum_reconstruction_nrmse": float(
                args.maximum_reconstruction_nrmse
            ),
            "minimum_relative_source_magnitude": float(
                args.minimum_relative_source_magnitude
            ),
            "reconstruction_accepted": reconstruction_accepted,
            "source_band_accepted": source_band_accepted,
            "m3_4_attribution_protocol_accepted": protocol_accepted,
        },
        "scope": {
            "validated": [
                "exact reconstructive direct/early/later decomposition",
                "common output-window comparison between FDTD and PathEvents",
                "geometric PathEvent arrival buckets",
                "coherent early/later error cross-term accounting",
            ],
            "pure_fdtd_direct_extracted": False,
            "mesh_visibility_validated": False,
            "diffraction_validated": False,
            "production_backend_replaced": False,
        },
        "decision": (
            "accept_m3_4_attribution_and_localize_next_model_work"
            if protocol_accepted
            else "reject_m3_4_attribution_protocol"
        ),
        "next_action": (
            "validate oblique single-wall FDTD reflection magnitude and phase "
            "against the same locally reacting boundary model, then audit the "
            "modal low-pass branch in the crossover before evaluating a mesh "
            "engine or adding more late paths"
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
                "m3_4_attribution_protocol_accepted": protocol_accepted,
                "aggregate": aggregate,
                "dominant_mean_component_error": dominant_component,
            },
            indent=2,
        )
    )
    if not protocol_accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
