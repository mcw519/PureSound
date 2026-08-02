#!/usr/bin/env python3
"""Build the M2.12 full-room complex-crossover failure baseline."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import butter, sosfreqz

from puresound.audio.rir.physics.wave.fdtd import (
    FDTDReferenceConfig,
    ricker_source,
    simulate_fdtd_reference,
    simulate_reciprocal_fdtd_reference,
)
from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.render.low_frequency import ImpedanceModalLowFrequencyBackend
from puresound.audio.rir.scene.sampling import (
    HybridRIRScene,
    upgrade_hybrid_scene_to_v2,
)
from puresound.audio.rir.physics.impedance.modes import (
    RectangularImpedanceBoundaryConfig,
)
from puresound.audio.rir.physics.impedance.residues import (
    ImpedanceModalResidueCalibration,
)
from puresound.audio.rir.physics.wave.source_convention import (
    fdtd_cell_center_position,
    fdtd_pressure_cell_to_free_field_input,
)
from puresound.audio.rir.path_events import (
    generate_shoebox_path_events,
    render_path_events,
)


REPORT_SCHEMA_VERSION = "puresound.full_room_crossover_validation.v1"


@dataclass(frozen=True)
class FullRoomCrossoverCase:
    case_id: str
    split: str
    room_dim_m: tuple[float, float, float]
    source_position_m: tuple[float, float, float]
    receiver_position_m: tuple[float, float, float]


DEFAULT_CASES = (
    FullRoomCrossoverCase(
        case_id="room_a_train_0",
        split="train",
        room_dim_m=(2.0, 1.35, 1.05),
        source_position_m=(0.43, 0.37, 0.31),
        receiver_position_m=(1.47, 0.96, 0.73),
    ),
    FullRoomCrossoverCase(
        case_id="room_a_position_holdout_0",
        split="position_holdout",
        room_dim_m=(2.0, 1.35, 1.05),
        source_position_m=(0.34, 0.68, 0.76),
        receiver_position_m=(1.72, 0.67, 0.28),
    ),
    FullRoomCrossoverCase(
        case_id="room_c_room_holdout_0",
        split="room_holdout",
        room_dim_m=(1.72, 1.42, 1.12),
        source_position_m=(0.37, 0.36, 0.30),
        receiver_position_m=(1.31, 1.02, 0.79),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a common-room FDTD transfer against fixed-pole modal "
            "low response and three image-source boundary-phase hypotheses."
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
        "--residue-calibration",
        type=Path,
        default=Path(
            "egs/rir_generation/phases/m2_impedance/config/"
            "impedance_residue_calibration_"
            "glass_wool_14kgm3_100mm_m2_10.json"
        ),
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--grid-spacing-m", type=float, default=0.06)
    parser.add_argument("--duration-s", type=float, default=0.35)
    parser.add_argument("--sample-rate-hz", type=int, default=8000)
    parser.add_argument("--source-center-hz", type=float, default=180.0)
    parser.add_argument("--source-delay-s", type=float, default=0.025)
    parser.add_argument(
        "--fdtd-reference-profile",
        choices=("legacy_second_order", "fourth_order_reciprocal"),
        default="legacy_second_order",
        help=(
            "Select the frozen legacy baseline or the M3.10 fourth-order "
            "bidirectionally averaged Green transfer."
        ),
    )
    parser.add_argument("--crossover-hz", type=float, default=240.0)
    parser.add_argument(
        "--crossover-scan-hz",
        type=float,
        nargs="+",
        default=[120.0, 150.0, 180.0, 210.0, 240.0],
    )
    parser.add_argument("--image-source-max-order", type=int, default=12)
    parser.add_argument(
        "--include-causal-path-event-first-order",
        action="store_true",
        help=(
            "Add the M3.2 direct-plus-first-order causal PathEvent branch as "
            "a diagnostic; it does not replace the order-12 production proxy."
        ),
    )
    parser.add_argument(
        "--causal-path-event-max-order",
        type=int,
        default=None,
        help=(
            "Render coherent causal PathEvents through this order. When set, "
            "it supersedes --include-causal-path-event-first-order."
        ),
    )
    parser.add_argument(
        "--minimum-complex-correlation",
        type=float,
        default=0.90,
    )
    parser.add_argument(
        "--maximum-complex-nrmse",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--minimum-transfer-energy-ratio",
        type=float,
        default=0.50,
    )
    parser.add_argument(
        "--maximum-transfer-energy-ratio",
        type=float,
        default=2.0,
    )
    return parser.parse_args()


def _shoebox_image_geometry(
    room_dim_m: tuple[float, float, float],
    source_position_m: tuple[float, float, float],
    receiver_position_m: tuple[float, float, float],
    max_order: int,
) -> dict[str, np.ndarray]:
    """Enumerate exact shoebox image positions without a renderer dependency."""
    if int(max_order) < 0:
        raise ValueError("image-source max_order must be non-negative")
    room = np.asarray(room_dim_m, dtype=np.float64)
    source = np.asarray(source_position_m, dtype=np.float64)
    receiver = np.asarray(receiver_position_m, dtype=np.float64)
    if room.shape != (3,) or np.any(room <= 0.0):
        raise ValueError("room dimensions must contain three positive values")
    if (
        source.shape != (3,)
        or receiver.shape != (3,)
        or np.any(source <= 0.0)
        or np.any(source >= room)
        or np.any(receiver <= 0.0)
        or np.any(receiver >= room)
    ):
        raise ValueError("source and receiver must be strictly inside the room")
    orders = []
    images = []
    limit = int(max_order)
    for nx in range(-limit, limit + 1):
        for ny in range(-limit, limit + 1):
            for nz in range(-limit, limit + 1):
                order = np.asarray([nx, ny, nz], dtype=np.int64)
                if int(np.sum(np.abs(order))) > limit:
                    continue
                translation_index = np.floor_divide(order + 1, 2)
                parity = np.where(order % 2 == 0, 1.0, -1.0)
                image = 2.0 * translation_index * room + parity * source
                orders.append(order)
                images.append(image)
    order_array = np.asarray(orders, dtype=np.int64).T
    image_array = np.asarray(images, dtype=np.float64).T
    vectors = receiver[:, None] - image_array
    distances = np.linalg.norm(vectors, axis=0)
    return {
        "orders_xyz": order_array,
        "images_xyz_m": image_array,
        "distances_m": distances,
        "incidence_cosines_xyz": np.abs(vectors) / distances,
    }


def _image_source_transfer(
    geometry: dict[str, np.ndarray],
    frequencies_hz: np.ndarray,
    boundary_model: Any,
    *,
    sound_speed_m_s: float,
    reflection_policy: str,
) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    orders = np.abs(geometry["orders_xyz"])
    total_order = np.sum(orders, axis=0)
    distances = geometry["distances_m"]
    cosines = geometry["incidence_cosines_xyz"]
    output = np.empty(frequencies.size, dtype=np.complex128)
    for index, frequency_hz in enumerate(frequencies):
        admittance = complex(
            boundary_model.normalized_admittance(float(frequency_hz))
        )
        normal_reflection = (1.0 - admittance) / (1.0 + admittance)
        if reflection_policy == "magnitude_only_normal":
            gain = abs(normal_reflection) ** total_order
        elif reflection_policy == "complex_normal":
            gain = normal_reflection**total_order
        elif reflection_policy == "complex_angle":
            axis_reflection = (
                (cosines - admittance) / (cosines + admittance)
            )
            gain = np.prod(axis_reflection**orders, axis=0)
        else:
            raise ValueError("unsupported image-source reflection policy")
        propagation = (
            np.exp(
                -2j
                * math.pi
                * float(frequency_hz)
                * distances
                / float(sound_speed_m_s)
            )
            / distances
        )
        output[index] = np.sum(gain * propagation)
    return output


def _sampled_frequency_response(
    signal: np.ndarray,
    sample_rate_hz: float,
    frequencies_hz: np.ndarray,
) -> np.ndarray:
    values = np.asarray(signal, dtype=np.float64)
    time_s = np.arange(values.size, dtype=np.float64) / float(sample_rate_hz)
    return np.asarray(
        [
            np.sum(
                values
                * np.exp(-2j * math.pi * float(frequency_hz) * time_s)
            )
            for frequency_hz in frequencies_hz
        ],
        dtype=np.complex128,
    )


def _linkwitz_riley_branches(
    sample_rate_hz: float,
    crossover_hz: float,
    frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    lowpass = butter(
        2,
        crossover_hz,
        btype="lowpass",
        fs=sample_rate_hz,
        output="sos",
    )
    highpass = butter(
        2,
        crossover_hz,
        btype="highpass",
        fs=sample_rate_hz,
        output="sos",
    )
    _frequencies, low_once = sosfreqz(
        lowpass,
        worN=frequencies_hz,
        fs=sample_rate_hz,
    )
    _frequencies, high_once = sosfreqz(
        highpass,
        worN=frequencies_hz,
        fs=sample_rate_hz,
    )
    return low_once**2, high_once**2


def _complex_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
) -> dict[str, float]:
    reference = np.asarray(reference, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    reference_norm = float(np.linalg.norm(reference))
    candidate_norm = float(np.linalg.norm(candidate))
    denominator = max(reference_norm * candidate_norm, 1e-30)
    return {
        "complex_nrmse": float(
            np.linalg.norm(candidate - reference)
            / max(reference_norm, 1e-30)
        ),
        "complex_correlation": float(
            abs(np.vdot(reference, candidate)) / denominator
        ),
        "transfer_energy_ratio": float(
            candidate_norm / max(reference_norm, 1e-30)
        ),
    }


def _band_limited_time_response(
    transfer: np.ndarray,
    frequencies_hz: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    frequencies = np.asarray(frequencies_hz, dtype=np.float64)
    if frequencies.size < 2 or not np.allclose(
        np.diff(frequencies),
        frequencies[1] - frequencies[0],
    ):
        raise ValueError("time synthesis requires uniform frequency spacing")
    delta_hz = float(frequencies[1] - frequencies[0])
    kernel = np.exp(
        2j * math.pi * time_s[:, None] * frequencies[None, :]
    )
    return 2.0 * delta_hz * np.real(kernel @ transfer)


def _real_window_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float | None]:
    reference_window = np.asarray(reference, dtype=np.float64)[mask]
    candidate_window = np.asarray(candidate, dtype=np.float64)[mask]
    reference_norm = float(np.linalg.norm(reference_window))
    candidate_norm = float(np.linalg.norm(candidate_window))
    if reference_window.size < 3 or reference_norm <= 1e-20:
        return {
            "nrmse": None,
            "correlation": None,
            "energy_ratio": None,
        }
    correlation = float(
        np.corrcoef(reference_window, candidate_window)[0, 1]
    )
    return {
        "nrmse": float(
            np.linalg.norm(candidate_window - reference_window)
            / reference_norm
        ),
        "correlation": (
            correlation if math.isfinite(correlation) else None
        ),
        "energy_ratio": float(candidate_norm / reference_norm),
    }


def _time_window_report(
    reference_transfer: np.ndarray,
    candidate_transfers: dict[str, np.ndarray],
    frequencies_hz: np.ndarray,
    *,
    duration_s: float,
    sample_rate_hz: int,
    direct_arrival_s: float,
) -> dict[str, dict[str, dict[str, float | None]]]:
    time_s = (
        np.arange(round(duration_s * sample_rate_hz), dtype=np.float64)
        / sample_rate_hz
    )
    reference = _band_limited_time_response(
        reference_transfer,
        frequencies_hz,
        time_s,
    )
    windows = {
        "direct": (
            (time_s >= max(0.0, direct_arrival_s - 0.012))
            & (time_s < direct_arrival_s + 0.012)
        ),
        "early_50ms": (
            (time_s >= max(0.0, direct_arrival_s - 0.012))
            & (time_s < direct_arrival_s + 0.050)
        ),
        "full": np.ones(time_s.size, dtype=bool),
    }
    report = {}
    for name, transfer in candidate_transfers.items():
        candidate = _band_limited_time_response(
            transfer,
            frequencies_hz,
            time_s,
        )
        report[name] = {
            window_name: _real_window_metrics(
                reference,
                candidate,
                mask,
            )
            for window_name, mask in windows.items()
        }
    return report


def _uniform_boundary_model(
    boundary_config: RectangularImpedanceBoundaryConfig,
) -> Any:
    models = tuple(boundary_config.boundaries.values())
    reference = models[0].metadata()
    if any(model.metadata() != reference for model in models[1:]):
        raise ValueError("M2.12 baseline currently requires a uniform boundary")
    return models[0]


def _simulate_case(
    case: FullRoomCrossoverCase,
    boundary_config: RectangularImpedanceBoundaryConfig,
    residue_calibration: ImpedanceModalResidueCalibration,
    args: argparse.Namespace,
) -> dict[str, Any]:
    path_event_max_order = (
        int(args.causal_path_event_max_order)
        if args.causal_path_event_max_order is not None
        else (1 if args.include_causal_path_event_first_order else None)
    )
    if path_event_max_order is not None and not 0 <= path_event_max_order <= 20:
        raise ValueError("causal PathEvent max order must be in [0, 20]")
    path_event_variant = (
        None
        if path_event_max_order is None
        else (
            "causal_path_event_first_order"
            if path_event_max_order == 1
            else f"causal_path_event_order_{path_event_max_order}"
        )
    )
    boundary_model = _uniform_boundary_model(boundary_config)
    fourth_order_reference = (
        args.fdtd_reference_profile == "fourth_order_reciprocal"
    )
    fdtd_config = FDTDReferenceConfig(
        room_dim_m=case.room_dim_m,
        grid_spacing_m=float(args.grid_spacing_m),
        duration_s=float(args.duration_s),
        source_position_m=case.source_position_m,
        receiver_position_m=case.receiver_position_m,
        source_center_hz=float(args.source_center_hz),
        source_delay_s=float(args.source_delay_s),
        **(
            {
                "boundary_pressure_scheme": (
                    "face_quadratic_time_quadratic"
                ),
                "spatial_derivative_order": 4,
                "near_wall_closure": "third_order_one_sided",
            }
            if args.fdtd_reference_profile == "fourth_order_reciprocal"
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
    renderer_config = HybridRIRConfig(
        sample_rate=int(args.sample_rate_hz),
        duration=float(args.duration_s),
        crossover_hz=float(args.crossover_hz),
        low_fmin_hz=60.0,
        low_fmax_hz=float(args.crossover_hz),
        output_mode="calibrated",
        match_crossover_energy=False,
        num_near_sources=1,
        num_far_sources=0,
        num_obstacles_range=(0, 0),
    )
    legacy_scene = HybridRIRScene(
        room_dim=list(case.room_dim_m),
        rt60=0.5,
        mic_pos=list(receiver_position),
        source_pos=[list(source_position)],
        source_labels=["source_0"],
    )
    modal_scene = upgrade_hybrid_scene_to_v2(
        legacy_scene,
        seed=1,
        room_type="office",
    )
    modal_backend = ImpedanceModalLowFrequencyBackend(
        boundary_config=boundary_config,
        residue_calibration=residue_calibration,
        num_modes_per_axis=5,
        max_modes=128,
    )
    modal_rir = np.asarray(
        modal_backend.simulate(modal_scene, renderer_config)[0],
        dtype=np.float64,
    )
    source_position_array = np.asarray(source_position)
    receiver_position_array = np.asarray(receiver_position)
    direct_distance_m = float(
        np.linalg.norm(source_position_array - receiver_position_array)
    )
    low_frequencies_hz = np.linspace(60.0, float(args.crossover_hz), 181)
    crossover_frequencies_hz = np.linspace(
        0.7 * float(args.crossover_hz),
        min(300.0, 1.25 * float(args.crossover_hz)),
        265,
    )

    def fdtd_transfer(frequencies_hz: np.ndarray) -> tuple[np.ndarray, float]:
        output_response = _sampled_frequency_response(
            fdtd.rir,
            fdtd.sample_rate_hz,
            frequencies_hz,
        )
        input_response = _sampled_frequency_response(
            equivalent_input,
            fdtd.sample_rate_hz,
            frequencies_hz,
        )
        relative_source = np.abs(input_response) / max(
            float(np.max(np.abs(input_response))),
            1e-30,
        )
        return output_response / input_response, float(np.min(relative_source))

    low_reference, low_minimum_source = fdtd_transfer(low_frequencies_hz)
    crossover_reference, crossover_minimum_source = fdtd_transfer(
        crossover_frequencies_hz
    )
    modal_low = _sampled_frequency_response(
        modal_rir,
        renderer_config.sample_rate,
        low_frequencies_hz,
    )
    modal_crossover = _sampled_frequency_response(
        modal_rir,
        renderer_config.sample_rate,
        crossover_frequencies_hz,
    )
    geometry = _shoebox_image_geometry(
        case.room_dim_m,
        source_position,
        receiver_position,
        int(args.image_source_max_order),
    )
    high_variants = {
        policy: _image_source_transfer(
            geometry,
            crossover_frequencies_hz,
            boundary_model,
            sound_speed_m_s=renderer_config.sound_speed,
            reflection_policy=policy,
        )
        for policy in (
            "magnitude_only_normal",
            "complex_normal",
            "complex_angle",
        )
    }
    path_event_realization_metrics = None
    path_event_set = None
    if path_event_max_order is not None:
        path_event_set = generate_shoebox_path_events(
            dimensions_m=case.room_dim_m,
            source_position_m=source_position,
            receiver_position_m=receiver_position,
            sound_speed_m_s=renderer_config.sound_speed,
            scene_id=case.case_id,
            source_id="source_0",
            receiver_id="receiver_0",
            max_order=path_event_max_order,
            edge_corner_policy="sequential_face_product_diagnostic",
            boundary_admittance_models=boundary_config.boundaries,
            reflection_frequencies_hz=crossover_frequencies_hz,
        )
        path_event_rir = render_path_events(
            path_event_set,
            sample_rate_hz=renderer_config.sample_rate,
            num_samples=renderer_config.num_samples,
            surface_admittance_models=boundary_config.boundaries,
        )
        path_event_transfer = _sampled_frequency_response(
            path_event_rir,
            renderer_config.sample_rate,
            crossover_frequencies_hz,
        )
        first_order_geometry = _shoebox_image_geometry(
            case.room_dim_m,
            source_position,
            receiver_position,
            path_event_max_order,
        )
        first_order_analytic = _image_source_transfer(
            first_order_geometry,
            crossover_frequencies_hz,
            boundary_model,
            sound_speed_m_s=renderer_config.sound_speed,
            reflection_policy="complex_angle",
        )
        high_variants[path_event_variant] = path_event_transfer
        path_event_realization_metrics = _complex_metrics(
            first_order_analytic,
            path_event_transfer,
        )
    low_branch, high_branch = _linkwitz_riley_branches(
        renderer_config.sample_rate,
        renderer_config.crossover_hz,
        crossover_frequencies_hz,
    )
    crossover_candidates = {
        "modal_crossover_raw": modal_crossover,
        "modal_lowpass_branch": low_branch * modal_crossover,
    }
    crossover_candidates.update(
        {
            f"geometric_{name}": transfer
            for name, transfer in high_variants.items()
        }
    )
    crossover_candidates.update(
        {
            f"geometric_highpass_{name}": high_branch * transfer
            for name, transfer in high_variants.items()
        }
    )
    crossover_candidates.update(
        {
            f"hybrid_{name}": (
                low_branch * modal_crossover + high_branch * transfer
            )
            for name, transfer in high_variants.items()
        }
    )
    direct_index = round(
        direct_distance_m
        / renderer_config.sound_speed
        * renderer_config.sample_rate
    )
    direct_rir = np.zeros_like(modal_rir)
    if direct_index < direct_rir.size:
        direct_rir[direct_index] = 1.0 / max(direct_distance_m, 0.1)
    clipped_modal_component = modal_rir - direct_rir
    onset_scan = {}
    sample_indices = np.arange(modal_rir.size, dtype=np.float64)
    local_time_s = (
        sample_indices - float(direct_index)
    ) / renderer_config.sample_rate
    for onset_duration_ms in (0.0, 2.0, 4.0, 8.0, 12.0, 20.0):
        if onset_duration_ms <= 0.0:
            onset = (local_time_s >= 0.0).astype(np.float64)
        else:
            progress = np.clip(
                local_time_s / (onset_duration_ms / 1000.0),
                0.0,
                1.0,
            )
            onset = 0.5 - 0.5 * np.cos(math.pi * progress)
        onset_rir = direct_rir + clipped_modal_component * onset
        onset_low = _sampled_frequency_response(
            onset_rir,
            renderer_config.sample_rate,
            low_frequencies_hz,
        )
        onset_crossover = _sampled_frequency_response(
            onset_rir,
            renderer_config.sample_rate,
            crossover_frequencies_hz,
        )
        onset_scan[f"{onset_duration_ms:g}"] = {
            "onset_duration_ms": onset_duration_ms,
            "modal_low": _complex_metrics(low_reference, onset_low),
            "modal_crossover_raw": _complex_metrics(
                crossover_reference,
                onset_crossover,
            ),
            "modal_lowpass_branch": _complex_metrics(
                crossover_reference,
                low_branch * onset_crossover,
            ),
            "hybrid_magnitude_only_normal": _complex_metrics(
                crossover_reference,
                (
                    low_branch * onset_crossover
                    + high_branch
                    * high_variants["magnitude_only_normal"]
                ),
            ),
            "hybrid_complex_angle": _complex_metrics(
                crossover_reference,
                (
                    low_branch * onset_crossover
                    + high_branch * high_variants["complex_angle"]
                ),
            ),
        }
    scan_frequencies_hz = np.linspace(60.0, 300.0, 481)
    scan_reference, scan_minimum_source = fdtd_transfer(
        scan_frequencies_hz
    )
    scan_modal = _sampled_frequency_response(
        modal_rir,
        renderer_config.sample_rate,
        scan_frequencies_hz,
    )
    scan_high_variants = {
        policy: _image_source_transfer(
            geometry,
            scan_frequencies_hz,
            boundary_model,
            sound_speed_m_s=renderer_config.sound_speed,
            reflection_policy=policy,
        )
        for policy in (
            "magnitude_only_normal",
            "complex_normal",
            "complex_angle",
        )
    }
    crossover_scan = {}
    for crossover_hz in args.crossover_scan_hz:
        crossover_value = float(crossover_hz)
        if not 60.0 / 0.7 <= crossover_value <= 300.0 / 1.25:
            raise ValueError(
                "crossover scan values must keep [0.7fc, 1.25fc] "
                "inside 60-300 Hz"
            )
        scan_mask = (
            (scan_frequencies_hz >= 0.7 * crossover_value)
            & (scan_frequencies_hz <= 1.25 * crossover_value)
        )
        scan_low_branch, scan_high_branch = _linkwitz_riley_branches(
            renderer_config.sample_rate,
            crossover_value,
            scan_frequencies_hz,
        )
        crossover_scan[f"{crossover_value:g}"] = {
            policy: _complex_metrics(
                scan_reference[scan_mask],
                (
                    scan_low_branch * scan_modal
                    + scan_high_branch * transfer
                )[scan_mask],
            )
            for policy, transfer in scan_high_variants.items()
        }
    overlap_mask = (
        crossover_frequencies_hz <= float(args.crossover_hz) + 1e-9
    )
    return {
        "case": asdict(case),
        "fdtd": {
            "reference_profile": args.fdtd_reference_profile,
            "reciprocity_averaged": fdtd.reciprocity_averaged,
            "raw_reciprocity_nrmse": fdtd.raw_reciprocity_nrmse,
            "config": asdict(fdtd_config),
            "sample_rate_hz": float(fdtd.sample_rate_hz),
            "grid_spacing_xyz_m": list(fdtd.grid_spacing_xyz_m),
            "source_cell_zyx": list(fdtd.source_cell_zyx),
            "receiver_cell_zyx": list(fdtd.receiver_cell_zyx),
            "source_cell_center_m": list(source_position),
            "receiver_cell_center_m": list(receiver_position),
            "direct_distance_m": direct_distance_m,
            "minimum_relative_source_magnitude": {
                "low_band": low_minimum_source,
                "crossover_band": crossover_minimum_source,
                "scan_band": scan_minimum_source,
            },
        },
        "modal_mode_count": int(
            modal_backend.last_modal_metadata["mode_count"]
        ),
        "image_source_count": int(geometry["distances_m"].size),
        "causal_path_event": (
            None
            if path_event_realization_metrics is None
            else {
                "max_order": path_event_max_order,
                "variant": path_event_variant,
                "event_count": len(path_event_set.events),
                "excluded_edge_or_corner_path_count": len(
                    path_event_set.metadata[
                        "excluded_edge_or_corner_image_orders"
                    ]
                ),
                "diagnostic_edge_or_corner_path_count": len(
                    path_event_set.metadata[
                        "included_diagnostic_edge_or_corner_image_orders"
                    ]
                ),
                "edge_corner_policy": path_event_set.metadata[
                    "edge_corner_policy"
                ],
                "analytic_image_source_count": int(
                    first_order_geometry["distances_m"].size
                ),
                "time_renderer_vs_analytic_complex_angle": (
                    path_event_realization_metrics
                ),
            }
        ),
        "frequency_bands_hz": {
            "low": [
                float(low_frequencies_hz[0]),
                float(low_frequencies_hz[-1]),
            ],
            "crossover": [
                float(crossover_frequencies_hz[0]),
                float(crossover_frequencies_hz[-1]),
            ],
        },
        "complex_metrics": {
            "modal_low": _complex_metrics(low_reference, modal_low),
            **{
                name: _complex_metrics(crossover_reference, transfer)
                for name, transfer in crossover_candidates.items()
            },
        },
        "overlap_complex_metrics": {
            name: _complex_metrics(
                crossover_reference[overlap_mask],
                transfer[overlap_mask],
            )
            for name, transfer in crossover_candidates.items()
        },
        "crossover_scan": crossover_scan,
        "modal_onset_scan": onset_scan,
        "band_limited_time_windows": {
            "low": _time_window_report(
                low_reference,
                {"modal_low": modal_low},
                low_frequencies_hz,
                duration_s=float(args.duration_s),
                sample_rate_hz=int(args.sample_rate_hz),
                direct_arrival_s=(
                    direct_distance_m / renderer_config.sound_speed
                ),
            ),
            "crossover": _time_window_report(
                crossover_reference,
                crossover_candidates,
                crossover_frequencies_hz,
                duration_s=float(args.duration_s),
                sample_rate_hz=int(args.sample_rate_hz),
                direct_arrival_s=(
                    direct_distance_m / renderer_config.sound_speed
                ),
            ),
        },
    }


def _aggregate(
    case_reports: list[dict[str, Any]],
    candidate: str,
    metrics_key: str = "complex_metrics",
) -> dict[str, float]:
    metrics = [
        report[metrics_key][candidate] for report in case_reports
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


def _aggregate_metric_values(
    metrics: list[dict[str, float]],
) -> dict[str, float]:
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
    path_event_max_order = (
        int(args.causal_path_event_max_order)
        if args.causal_path_event_max_order is not None
        else (1 if args.include_causal_path_event_first_order else None)
    )
    path_event_variant = (
        None
        if path_event_max_order is None
        else (
            "causal_path_event_first_order"
            if path_event_max_order == 1
            else f"causal_path_event_order_{path_event_max_order}"
        )
    )
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(
        args.boundary_config
    )
    residue_calibration = ImpedanceModalResidueCalibration.from_json(
        args.residue_calibration
    )
    if (
        residue_calibration.boundary_reference_id
        != boundary_config.reference_id
    ):
        raise ValueError("residue calibration does not match boundary config")
    case_reports = [
        _simulate_case(
            case,
            boundary_config,
            residue_calibration,
            args,
        )
        for case in DEFAULT_CASES
    ]
    candidates = tuple(case_reports[0]["complex_metrics"])
    aggregate = {
        candidate: _aggregate(case_reports, candidate)
        for candidate in candidates
    }
    overlap_candidates = tuple(
        case_reports[0]["overlap_complex_metrics"]
    )
    overlap_aggregate = {
        candidate: _aggregate(
            case_reports,
            candidate,
            metrics_key="overlap_complex_metrics",
        )
        for candidate in overlap_candidates
    }
    scan_frequencies = tuple(case_reports[0]["crossover_scan"])
    scan_policies = tuple(
        case_reports[0]["crossover_scan"][scan_frequencies[0]]
    )
    crossover_scan_aggregate = {
        frequency: {
            policy: _aggregate_metric_values(
                [
                    report["crossover_scan"][frequency][policy]
                    for report in case_reports
                ]
            )
            for policy in scan_policies
        }
        for frequency in scan_frequencies
    }
    best_scan_frequency, best_scan_policy = min(
        (
            (frequency, policy)
            for frequency in scan_frequencies
            for policy in scan_policies
        ),
        key=lambda value: crossover_scan_aggregate[value[0]][value[1]][
            "mean_complex_nrmse"
        ],
    )
    onset_durations = tuple(case_reports[0]["modal_onset_scan"])
    onset_metric_names = tuple(
        case_reports[0]["modal_onset_scan"][onset_durations[0]]
    )
    onset_metric_names = tuple(
        name for name in onset_metric_names if name != "onset_duration_ms"
    )
    modal_onset_scan_aggregate = {
        duration: {
            metric_name: _aggregate_metric_values(
                [
                    report["modal_onset_scan"][duration][metric_name]
                    for report in case_reports
                ]
            )
            for metric_name in onset_metric_names
        }
        for duration in onset_durations
    }
    low_safe_onsets = [
        duration
        for duration in onset_durations
        if modal_onset_scan_aggregate[duration]["modal_low"][
            "mean_complex_correlation"
        ]
        >= float(args.minimum_complex_correlation)
        and modal_onset_scan_aggregate[duration]["modal_low"][
            "mean_complex_nrmse"
        ]
        <= float(args.maximum_complex_nrmse)
    ]
    best_safe_onset = (
        min(
            low_safe_onsets,
            key=lambda duration: modal_onset_scan_aggregate[duration][
                "hybrid_complex_angle"
            ]["mean_complex_nrmse"],
        )
        if low_safe_onsets
        else None
    )
    production_proxy = aggregate["hybrid_magnitude_only_normal"]
    per_case_acceptance = {}
    for case_report in case_reports:
        metrics = case_report["complex_metrics"][
            "hybrid_magnitude_only_normal"
        ]
        accepted = bool(
            metrics["complex_correlation"]
            >= float(args.minimum_complex_correlation)
            and metrics["complex_nrmse"]
            <= float(args.maximum_complex_nrmse)
            and metrics["transfer_energy_ratio"]
            >= float(args.minimum_transfer_energy_ratio)
            and metrics["transfer_energy_ratio"]
            <= float(args.maximum_transfer_energy_ratio)
        )
        per_case_acceptance[case_report["case"]["case_id"]] = {
            **metrics,
            "accepted": accepted,
        }
    full_room_accepted = all(
        value["accepted"] for value in per_case_acceptance.values()
    )
    hybrid_candidates = (
        "hybrid_magnitude_only_normal",
        "hybrid_complex_normal",
        "hybrid_complex_angle",
    )
    best_diagnostic = min(
        hybrid_candidates,
        key=lambda name: aggregate[name]["mean_complex_nrmse"],
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "protocol_role": (
            "expanded_fourth_order_reference_audit"
            if args.fdtd_reference_profile
            == "fourth_order_reciprocal"
            else "failure_baseline"
        ),
        "boundary_config": str(args.boundary_config),
        "boundary_reference_id": boundary_config.reference_id,
        "residue_calibration": str(args.residue_calibration),
        "residue_reference_id": residue_calibration.reference_id,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_report"
        },
        "reflection_hypotheses": {
            "magnitude_only_normal": (
                "normal-incidence abs(Gamma), no reflection phase"
            ),
            "complex_normal": (
                "normal-incidence complex Gamma applied to every path"
            ),
            "complex_angle": (
                "locally reacting Gamma(theta)=(cos(theta)-y)/(cos(theta)+y)"
            ),
            **(
                {
                    path_event_variant: (
                        f"coherent causal PathEvents through order "
                        f"{path_event_max_order}; fractional propagation "
                        "delay and bilinear passive angle-aware boundary "
                        "filters rendered in time"
                    )
                }
                if path_event_variant is not None
                else {}
            ),
        },
        "cases": case_reports,
        "aggregate": aggregate,
        "overlap_aggregate_168hz_to_crossover": overlap_aggregate,
        "crossover_scan_aggregate": crossover_scan_aggregate,
        "modal_onset_scan_aggregate": modal_onset_scan_aggregate,
        "acceptance": {
            "candidate_under_gate": "hybrid_magnitude_only_normal",
            "minimum_complex_correlation": float(
                args.minimum_complex_correlation
            ),
            "maximum_complex_nrmse": float(args.maximum_complex_nrmse),
            "transfer_energy_ratio_range": [
                float(args.minimum_transfer_energy_ratio),
                float(args.maximum_transfer_energy_ratio),
            ],
            "by_case": per_case_acceptance,
            "full_room_complex_gate_accepted": full_room_accepted,
            "protocol_completed": True,
        },
        "diagnostic": {
            "production_proxy_aggregate": production_proxy,
            "best_hybrid_reflection_hypothesis": best_diagnostic,
            "best_hybrid_aggregate": aggregate[best_diagnostic],
            "best_crossover_scan": {
                "crossover_hz": float(best_scan_frequency),
                "reflection_policy": best_scan_policy,
                "aggregate": crossover_scan_aggregate[
                    best_scan_frequency
                ][best_scan_policy],
            },
            "best_low_band_safe_onset": (
                None
                if best_safe_onset is None
                else {
                    "onset_duration_ms": float(best_safe_onset),
                    "aggregate": modal_onset_scan_aggregate[
                        best_safe_onset
                    ],
                }
            ),
            "gain_fitting_performed": False,
            "pole_or_residue_refitting_performed": False,
            "causal_path_event": (
                None
                if path_event_variant is None
                else {
                    "max_order": path_event_max_order,
                    "variant": path_event_variant,
                    "geometric_aggregate": aggregate[
                        f"geometric_{path_event_variant}"
                    ],
                    "geometric_highpass_aggregate": aggregate[
                        f"geometric_highpass_{path_event_variant}"
                    ],
                    "hybrid_aggregate": aggregate[
                        f"hybrid_{path_event_variant}"
                    ],
                    "production_backend_replacement_claimed": False,
                    "scalar_gain_fitted": False,
                }
            ),
        },
        "scope": {
            "uniform_boundary_only": True,
            "image_source_max_order": int(args.image_source_max_order),
            "production_high_backend_exactly_reproduced": False,
            "image_source_geometry_and_reflection_hypotheses_only": True,
            "causal_path_event_max_order": path_event_max_order,
            "measured_room_validated": False,
            "production_material_mapping_validated": False,
        },
        "decision": (
            "pass_full_room_complex_gate"
            if full_room_accepted
            else (
                "record_expanded_reference_failure_and_localize_error"
                if args.fdtd_reference_profile
                == "fourth_order_reciprocal"
                else "record_failure_baseline_and_do_not_fit_scalar_gain"
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
                "protocol_completed": True,
                "full_room_complex_gate_accepted": full_room_accepted,
                "production_proxy": production_proxy,
                "best_hybrid_reflection_hypothesis": best_diagnostic,
                "best_hybrid": aggregate[best_diagnostic],
                "best_crossover_scan": {
                    "crossover_hz": float(best_scan_frequency),
                    "reflection_policy": best_scan_policy,
                    "aggregate": crossover_scan_aggregate[
                        best_scan_frequency
                    ][best_scan_policy],
                },
                "best_low_band_safe_onset_ms": (
                    None
                    if best_safe_onset is None
                    else float(best_safe_onset)
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
