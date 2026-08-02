#!/usr/bin/env python3
"""Calibrate fixed-pole 3D impedance modal residues against FDTD references."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.signal import butter, fftconvolve, sosfiltfilt

from puresound.audio.fdtd_reference import (
    FDTDReferenceConfig,
    ricker_source,
    simulate_fdtd_reference,
)
from puresound.audio.impedance_modes import (
    RectangularImpedanceBoundaryConfig,
    solve_rectangular_impedance_modes,
)
from puresound.audio.impedance_residues import (
    FixedPoleResidueCase,
    fit_fixed_pole_modal_residues,
)


@dataclass(frozen=True)
class PositionPair:
    case_id: str
    source_position_m: tuple[float, float, float]
    receiver_position_m: tuple[float, float, float]
    split: str


@dataclass(frozen=True)
class RoomCalibrationSet:
    room_id: str
    room_dim_m: tuple[float, float, float]
    positions: tuple[PositionPair, ...]


DEFAULT_CALIBRATION_SETS = (
    RoomCalibrationSet(
        room_id="room_a",
        room_dim_m=(2.0, 1.35, 1.05),
        positions=(
            PositionPair(
                "room_a_train_0",
                (0.43, 0.37, 0.31),
                (1.47, 0.96, 0.73),
                "train",
            ),
            PositionPair(
                "room_a_train_1",
                (0.62, 0.91, 0.32),
                (1.61, 0.42, 0.78),
                "train",
            ),
            PositionPair(
                "room_a_holdout_0",
                (0.34, 0.68, 0.76),
                (1.72, 0.67, 0.28),
                "holdout",
            ),
        ),
    ),
    RoomCalibrationSet(
        room_id="room_b",
        # Keep every axial fundamental inside the boundary model's >=60 Hz
        # validity. A longer x dimension would create an unmodelled sub-band
        # pole whose filter transition contaminates the 60 Hz comparison.
        room_dim_m=(1.85, 1.55, 1.15),
        positions=(
            PositionPair(
                "room_b_train_0",
                (0.40, 0.42, 0.35),
                (1.48, 1.11, 0.81),
                "train",
            ),
            PositionPair(
                "room_b_train_1",
                (0.62, 1.03, 0.36),
                (1.60, 0.47, 0.84),
                "train",
            ),
            PositionPair(
                "room_b_holdout_0",
                (0.35, 0.77, 0.82),
                (1.62, 0.74, 0.31),
                "holdout",
            ),
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Hold 3D impedance poles fixed and fit a transferable complex "
            "modal-residue scale/power law to independent FDTD responses."
        )
    )
    parser.add_argument("--boundary-config", type=Path, required=True)
    parser.add_argument("--output-calibration", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--reference-id",
        default="glass_wool_14kgm3_100mm_fdtd_residue_m2_8",
    )
    parser.add_argument("--grid-spacing-m", type=float, default=0.12)
    parser.add_argument("--duration-s", type=float, default=0.52)
    parser.add_argument("--source-center-hz", type=float, default=150.0)
    parser.add_argument("--source-delay-s", type=float, default=0.025)
    parser.add_argument("--analysis-start-s", type=float, default=0.055)
    parser.add_argument("--analysis-stop-s", type=float, default=0.48)
    parser.add_argument("--minimum-frequency-hz", type=float, default=60.0)
    parser.add_argument("--maximum-frequency-hz", type=float, default=240.0)
    parser.add_argument("--reference-frequency-hz", type=float, default=100.0)
    parser.add_argument("--mode-index-limit", type=int, default=5)
    parser.add_argument("--continuation-steps", type=int, default=8)
    parser.add_argument("--exponent-minimum", type=float, default=-0.5)
    parser.add_argument("--exponent-maximum", type=float, default=2.0)
    parser.add_argument("--exponent-steps", type=int, default=51)
    return parser.parse_args()


def _candidate_mode_indices(
    room_dim_m: tuple[float, float, float],
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    mode_index_limit: int,
    sound_speed_m_s: float,
) -> tuple[tuple[int, int, int], ...]:
    candidates = []
    search_minimum = max(0.0, 0.65 * float(minimum_frequency_hz))
    search_maximum = 1.25 * float(maximum_frequency_hz)
    for nx in range(int(mode_index_limit) + 1):
        for ny in range(int(mode_index_limit) + 1):
            for nz in range(int(mode_index_limit) + 1):
                if nx == ny == nz == 0:
                    continue
                rigid_frequency = (
                    0.5
                    * float(sound_speed_m_s)
                    * math.sqrt(
                        (nx / room_dim_m[0]) ** 2
                        + (ny / room_dim_m[1]) ** 2
                        + (nz / room_dim_m[2]) ** 2
                    )
                )
                if search_minimum <= rigid_frequency <= search_maximum:
                    candidates.append((rigid_frequency, (nx, ny, nz)))
    return tuple(indices for _frequency, indices in sorted(candidates))


def _cell_center_position(
    cell_zyx: tuple[int, int, int],
    spacing_xyz_m: tuple[float, float, float],
) -> tuple[float, float, float]:
    iz, iy, ix = cell_zyx
    dx, dy, dz = spacing_xyz_m
    return (
        (float(ix) + 0.5) * dx,
        (float(iy) + 0.5) * dy,
        (float(iz) + 0.5) * dz,
    )


def _make_case(
    room_set: RoomCalibrationSet,
    position_pair: PositionPair,
    modes,
    boundary_config: RectangularImpedanceBoundaryConfig,
    args: argparse.Namespace,
    *,
    grid_spacing_m: float | None = None,
    case_id_prefix: str = "",
) -> FixedPoleResidueCase:
    nominal_grid_spacing_m = (
        float(args.grid_spacing_m)
        if grid_spacing_m is None
        else float(grid_spacing_m)
    )
    fdtd_config = FDTDReferenceConfig(
        room_dim_m=room_set.room_dim_m,
        grid_spacing_m=nominal_grid_spacing_m,
        duration_s=float(args.duration_s),
        source_position_m=position_pair.source_position_m,
        receiver_position_m=position_pair.receiver_position_m,
        source_center_hz=float(args.source_center_hz),
        source_delay_s=float(args.source_delay_s),
    )
    result = simulate_fdtd_reference(
        fdtd_config,
        boundary_admittance=boundary_config.boundaries,
    )
    source_position = _cell_center_position(
        result.source_cell_zyx,
        result.grid_spacing_xyz_m,
    )
    receiver_position = _cell_center_position(
        result.receiver_cell_zyx,
        result.grid_spacing_xyz_m,
    )
    cell_volume_m3 = float(np.prod(result.grid_spacing_xyz_m))
    time_s = (
        np.arange(result.rir.size, dtype=np.float64)
        * float(result.time_step_s)
    )
    source = ricker_source(
        result.rir.size,
        result.time_step_s,
        args.source_center_hz,
        args.source_delay_s,
    )
    real_basis = []
    imaginary_basis = []
    for mode in modes:
        # The FDTD source adds one pressure sample to one cell. Multiplication
        # by cell volume maps the continuous, volume-normalized eigenfunction
        # product to that discrete initial-pressure convention.
        coupling = (
            cell_volume_m3
            * mode.eigenfunction_at(source_position)
            * mode.eigenfunction_at(receiver_position)
        )
        free_response = coupling * np.exp(
            mode.complex_angular_frequency_rad_s * time_s
        )
        real_basis.append(
            fftconvolve(source, free_response.real)[: result.rir.size]
        )
        imaginary_basis.append(
            fftconvolve(source, free_response.imag)[: result.rir.size]
        )
    analysis_mask = (
        (time_s >= float(args.analysis_start_s))
        & (time_s < float(args.analysis_stop_s))
    )
    analysis_filter = butter(
        4,
        [
            float(args.minimum_frequency_hz),
            float(args.maximum_frequency_hz),
        ],
        btype="bandpass",
        fs=float(result.sample_rate_hz),
        output="sos",
    )
    filtered_real_basis = sosfiltfilt(
        analysis_filter,
        np.asarray(real_basis, dtype=np.float64),
        axis=-1,
    )
    filtered_imaginary_basis = sosfiltfilt(
        analysis_filter,
        np.asarray(imaginary_basis, dtype=np.float64),
        axis=-1,
    )
    filtered_target = sosfiltfilt(
        analysis_filter,
        np.asarray(result.rir, dtype=np.float64),
    )
    return FixedPoleResidueCase(
        case_id=f"{case_id_prefix}{position_pair.case_id}",
        split=position_pair.split,
        mode_frequencies_hz=np.asarray(
            [mode.frequency_hz for mode in modes],
            dtype=np.float64,
        ),
        real_mode_basis=filtered_real_basis,
        imaginary_mode_basis=filtered_imaginary_basis,
        target=filtered_target,
        analysis_mask=analysis_mask,
        metadata={
            "room_id": room_set.room_id,
            "room_dim_m": list(room_set.room_dim_m),
            "requested_source_position_m": list(
                position_pair.source_position_m
            ),
            "requested_receiver_position_m": list(
                position_pair.receiver_position_m
            ),
            "fdtd_source_cell_center_m": list(source_position),
            "fdtd_receiver_cell_center_m": list(receiver_position),
            "fdtd_grid_shape_zyx": list(result.grid_shape_zyx),
            "fdtd_nominal_grid_spacing_m": nominal_grid_spacing_m,
            "fdtd_grid_spacing_xyz_m": list(
                result.grid_spacing_xyz_m
            ),
            "fdtd_sample_rate_hz": float(result.sample_rate_hz),
            "fdtd_cell_volume_m3": cell_volume_m3,
            "analysis_window_s": [
                float(args.analysis_start_s),
                float(args.analysis_stop_s),
            ],
            "analysis_filter": {
                "type": "zero_phase_butterworth_bandpass",
                "order": 4,
                "frequency_hz": [
                    float(args.minimum_frequency_hz),
                    float(args.maximum_frequency_hz),
                ],
            },
        },
    )


def build_fdtd_residue_cases(
    calibration_sets: tuple[RoomCalibrationSet, ...],
    boundary_config: RectangularImpedanceBoundaryConfig,
    args: argparse.Namespace,
    *,
    grid_spacing_overrides_m: dict[str, float] | None = None,
    case_id_prefix: str = "",
) -> tuple[list[FixedPoleResidueCase], list[dict]]:
    """Run all FDTD position cases and build fixed-pole quadrature bases."""
    overrides = grid_spacing_overrides_m or {}
    cases = []
    mode_report = []
    for room_set in calibration_sets:
        indices = _candidate_mode_indices(
            room_set.room_dim_m,
            args.minimum_frequency_hz,
            args.maximum_frequency_hz,
            args.mode_index_limit,
            sound_speed_m_s=343.0,
        )
        solved = solve_rectangular_impedance_modes(
            room_set.room_dim_m,
            boundary_config,
            mode_indices=indices,
            continuation_steps=args.continuation_steps,
        )
        modes = tuple(
            mode
            for mode in solved
            if args.minimum_frequency_hz
            <= mode.frequency_hz
            <= args.maximum_frequency_hz
        )
        if not modes:
            raise RuntimeError(
                f"no modes in calibration band for {room_set.room_id}"
            )
        mode_report.append(
            {
                "room_id": room_set.room_id,
                "room_dim_m": list(room_set.room_dim_m),
                "modes": [mode.metadata() for mode in modes],
            }
        )
        for position_pair in room_set.positions:
            spacing = overrides.get(position_pair.case_id)
            spacing_label = (
                f", grid={spacing:g} m" if spacing is not None else ""
            )
            print(
                f"FDTD {case_id_prefix}{position_pair.case_id} "
                f"({position_pair.split}, {len(modes)} fixed poles"
                f"{spacing_label})"
            )
            cases.append(
                _make_case(
                    room_set,
                    position_pair,
                    modes,
                    boundary_config,
                    args,
                    grid_spacing_m=spacing,
                    case_id_prefix=case_id_prefix,
                )
            )
    return cases, mode_report


def main() -> None:
    args = _parse_args()
    if not 0.0 < args.minimum_frequency_hz < args.maximum_frequency_hz:
        raise ValueError("calibration frequency range must be positive and ordered")
    if not 0.0 <= args.analysis_start_s < args.analysis_stop_s <= args.duration_s:
        raise ValueError("analysis window must lie inside the FDTD duration")
    if args.mode_index_limit < 1 or args.continuation_steps < 1:
        raise ValueError("mode index limit and continuation steps must be positive")
    if args.exponent_steps < 2:
        raise ValueError("exponent steps must be at least two")
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(
        args.boundary_config
    )
    boundary_validity = boundary_config.applicability.get(
        "valid_frequency_range_hz"
    )
    if boundary_validity is not None and (
        args.minimum_frequency_hz < float(boundary_validity[0])
        or args.maximum_frequency_hz > float(boundary_validity[1])
    ):
        raise ValueError("calibration band lies outside boundary model validity")

    cases, mode_report = build_fdtd_residue_cases(
        DEFAULT_CALIBRATION_SETS,
        boundary_config,
        args,
    )

    fit = fit_fixed_pole_modal_residues(
        cases,
        reference_id=args.reference_id,
        boundary_reference_id=boundary_config.reference_id,
        valid_frequency_range_hz=(
            float(args.minimum_frequency_hz),
            float(args.maximum_frequency_hz),
        ),
        reference_frequency_hz=float(args.reference_frequency_hz),
        exponent_candidates=np.round(
            np.linspace(
                float(args.exponent_minimum),
                float(args.exponent_maximum),
                int(args.exponent_steps),
            ),
            decimals=12,
        ),
        source={
            "evidence_tier": "independent_numerical_reference",
            "reference_solver": "staggered_pressure_velocity_fdtd",
            "source_convention": (
                "unit_peak_ricker_pressure_added_to_one_cell; "
                "modal coupling multiplied by FDTD cell volume"
            ),
            "boundary_config": str(args.boundary_config),
        },
        applicability={
            "scope": "controlled_rectangular_room_position_transfer",
            "room_geometry": "shoebox",
            "boundary_reference_id": boundary_config.reference_id,
            "production_material_mapping_enabled": False,
            "measured_room_transfer_validated": False,
        },
    )
    report = {
        **fit.report,
        "boundary_config": boundary_config.metadata(),
        "fdtd_configuration": {
            key: value
            for key, value in vars(args).items()
            if key not in {"output_calibration", "output_report"}
        },
        "calibration_sets": [
            {
                "room_id": room.room_id,
                "room_dim_m": list(room.room_dim_m),
                "positions": [
                    asdict(position) for position in room.positions
                ],
            }
            for room in DEFAULT_CALIBRATION_SETS
        ],
        "fixed_modes": mode_report,
    }
    report["fdtd_configuration"]["boundary_config"] = str(
        args.boundary_config
    )
    args.output_calibration.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    fit.calibration.to_json(args.output_calibration)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    holdout = fit.report["aggregate"]["holdout"]
    print(
        json.dumps(
            {
                "calibration": str(args.output_calibration),
                "report": str(args.output_report),
                "complex_scale": fit.calibration.metadata()["complex_scale"],
                "frequency_exponent": fit.calibration.frequency_exponent,
                "holdout": holdout,
                "accepted": fit.report["acceptance"]["accepted"],
            },
            indent=2,
            allow_nan=False,
        )
    )
    if not fit.report["acceptance"]["accepted"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
