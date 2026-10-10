#!/usr/bin/env python3
"""M2.9 multi-boundary, room-holdout and grid-holdout residue protocol."""

from __future__ import annotations

import argparse
import cmath
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from egs.rir_generation.phases.m2_impedance.scripts.calibrate_impedance_modal_residues import (
    PositionPair,
    RoomCalibrationSet,
    build_fdtd_residue_cases,
)
from puresound.audio.rir.physics.impedance.modes import (
    RectangularImpedanceBoundaryConfig,
)
from puresound.audio.rir.physics.impedance.residues import (
    FixedPoleResidueCase,
    fit_fixed_pole_modal_residues,
)


PROTOCOL_SCHEMA_VERSION = "puresound.impedance_modal_residue_protocol.v1"
ACCEPTANCE_SPLITS = (
    "position_holdout",
    "room_holdout",
    "grid_holdout",
)

M2_9_CALIBRATION_SETS = (
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
                "room_a_position_holdout_0",
                (0.34, 0.68, 0.76),
                (1.72, 0.67, 0.28),
                "position_holdout",
            ),
        ),
    ),
    RoomCalibrationSet(
        room_id="room_b",
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
                "room_b_position_holdout_0",
                (0.35, 0.77, 0.82),
                (1.62, 0.74, 0.31),
                "position_holdout",
            ),
        ),
    ),
    RoomCalibrationSet(
        room_id="room_c_unseen",
        room_dim_m=(1.72, 1.42, 1.12),
        positions=(
            PositionPair(
                "room_c_room_holdout_0",
                (0.37, 0.36, 0.30),
                (1.31, 1.02, 0.79),
                "room_holdout",
            ),
            PositionPair(
                "room_c_room_holdout_1",
                (0.63, 0.97, 0.35),
                (1.44, 0.48, 0.84),
                "room_holdout",
            ),
            # Same requested geometry as room_holdout_0, but a finer FDTD
            # grid. It is never used to fit the residue coefficients.
            PositionPair(
                "room_c_grid_holdout_0",
                (0.37, 0.36, 0.30),
                (1.31, 1.02, 0.79),
                "grid_holdout",
            ),
        ),
    ),
)


DEFAULT_BOUNDARY_CASES = (
    (
        "glass_wool_14kgm3_50mm",
        Path(
            "egs/rir_generation/phases/m2_impedance/config/"
            "impedance_reference_glass_wool_14kgm3_50mm.json"
        ),
    ),
    (
        "glass_wool_14kgm3_100mm",
        Path(
            "egs/rir_generation/phases/m2_impedance/config/"
            "impedance_reference_glass_wool_14kgm3_100mm.json"
        ),
    ),
)


def _parse_boundary_case(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError(
            "boundary case must use NAME=/path/to/config.json"
        )
    name, raw_path = value.split("=", 1)
    if not name or not raw_path:
        raise argparse.ArgumentTypeError(
            "boundary case name and path must be non-empty"
        )
    if any(character not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for character in name):
        raise argparse.ArgumentTypeError(
            "boundary case name must use lowercase letters, numbers, _ or -"
        )
    return name, Path(raw_path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Fit fixed-pole residues for multiple evidenced boundary variants, "
            "then test unseen positions, an unseen room and a finer FDTD grid."
        )
    )
    parser.add_argument(
        "--boundary-case",
        action="append",
        type=_parse_boundary_case,
        help=(
            "Repeat NAME=config.json. Defaults to the controlled 50 mm and "
            "100 mm glass-wool Miki variants."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--output-report",
        type=Path,
        required=True,
    )
    parser.add_argument("--grid-spacing-m", type=float, default=0.12)
    parser.add_argument("--fine-grid-spacing-m", type=float, default=0.08)
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
    parser.add_argument(
        "--minimum-holdout-correlation",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--maximum-holdout-nrmse",
        type=float,
        default=0.35,
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not 0.0 < args.minimum_frequency_hz < args.maximum_frequency_hz:
        raise ValueError("protocol frequency range must be positive and ordered")
    if not 0.0 <= args.analysis_start_s < args.analysis_stop_s <= args.duration_s:
        raise ValueError("analysis window must lie inside FDTD duration")
    if not 0.0 < args.fine_grid_spacing_m < args.grid_spacing_m:
        raise ValueError("fine grid spacing must be positive and below coarse")
    if args.mode_index_limit < 1 or args.continuation_steps < 1:
        raise ValueError("mode and continuation limits must be positive")
    if args.exponent_steps < 2:
        raise ValueError("exponent steps must be at least two")


def _prefixed_cases(
    cases: list[FixedPoleResidueCase],
    prefix: str,
) -> list[FixedPoleResidueCase]:
    return [
        FixedPoleResidueCase(
            case_id=f"{prefix}__{case.case_id}",
            split=case.split,
            mode_frequencies_hz=case.mode_frequencies_hz,
            real_mode_basis=case.real_mode_basis,
            imaginary_mode_basis=case.imaginary_mode_basis,
            target=case.target,
            analysis_mask=case.analysis_mask,
            metadata={
                **case.metadata,
                "boundary_case": prefix,
            },
        )
        for case in cases
    ]


def _fit_boundary(
    name: str,
    path: Path,
    args: argparse.Namespace,
):
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(path)
    validity = boundary_config.applicability.get("valid_frequency_range_hz")
    if validity is None or (
        args.minimum_frequency_hz < float(validity[0])
        or args.maximum_frequency_hz > float(validity[1])
    ):
        raise ValueError(
            f"{name} boundary validity does not contain protocol band"
        )
    cases, mode_report = build_fdtd_residue_cases(
        M2_9_CALIBRATION_SETS,
        boundary_config,
        args,
        grid_spacing_overrides_m={
            "room_c_grid_holdout_0": float(args.fine_grid_spacing_m)
        },
    )
    fit = fit_fixed_pole_modal_residues(
        cases,
        reference_id=f"{name}_fdtd_residue_m2_9",
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
        acceptance_splits=ACCEPTANCE_SPLITS,
        minimum_mean_holdout_correlation=float(
            args.minimum_holdout_correlation
        ),
        maximum_mean_holdout_nrmse=float(args.maximum_holdout_nrmse),
        source={
            "evidence_tier": "independent_numerical_reference",
            "reference_solver": "staggered_pressure_velocity_fdtd",
            "boundary_config": str(path),
            "protocol": "m2_9_multi_boundary_room_grid_holdout",
        },
        applicability={
            "scope": "controlled_rectangular_room_position_room_grid_transfer",
            "room_geometry": "shoebox",
            "boundary_reference_id": boundary_config.reference_id,
            "production_material_mapping_enabled": False,
            "measured_room_transfer_validated": False,
        },
    )
    detail_report = {
        **fit.report,
        "boundary_case": name,
        "boundary_config": boundary_config.metadata(),
        "calibration_sets": [
            {
                "room_id": room.room_id,
                "room_dim_m": list(room.room_dim_m),
                "positions": [
                    asdict(position) for position in room.positions
                ],
            }
            for room in M2_9_CALIBRATION_SETS
        ],
        "fixed_modes": mode_report,
        "grid_policy": {
            "coarse_nominal_spacing_m": float(args.grid_spacing_m),
            "fine_nominal_spacing_m": float(args.fine_grid_spacing_m),
            "fine_case_id": "room_c_grid_holdout_0",
        },
    }
    return boundary_config, cases, fit, detail_report


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    boundary_cases = tuple(args.boundary_case or DEFAULT_BOUNDARY_CASES)
    if len(boundary_cases) < 2:
        raise ValueError("M2.9 protocol requires at least two boundary cases")
    if len({name for name, _path in boundary_cases}) != len(boundary_cases):
        raise ValueError("boundary case names must be unique")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)

    boundary_results = []
    all_cases = []
    for name, path in boundary_cases:
        print(f"=== boundary {name}: {path} ===")
        boundary_config, cases, fit, detail_report = _fit_boundary(
            name,
            path,
            args,
        )
        calibration_path = (
            args.output_dir / f"impedance_residue_calibration_{name}_m2_9.json"
        )
        detail_path = (
            args.output_dir
            / f"impedance_residue_calibration_{name}_m2_9_report.json"
        )
        fit.calibration.to_json(calibration_path)
        detail_path.write_text(
            json.dumps(
                detail_report,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        all_cases.extend(_prefixed_cases(cases, name))
        boundary_results.append(
            {
                "name": name,
                "path": str(path),
                "boundary_reference_id": boundary_config.reference_id,
                "calibration_path": str(calibration_path),
                "detail_report_path": str(detail_path),
                "calibration": fit.calibration.metadata(),
                "aggregate": fit.report["aggregate"],
                "acceptance": fit.report["acceptance"],
            }
        )

    shared_fit = fit_fixed_pole_modal_residues(
        all_cases,
        reference_id="multi_boundary_shared_residue_m2_9_protocol_only",
        boundary_reference_id="multi_boundary_protocol_not_renderer_compatible",
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
        acceptance_splits=ACCEPTANCE_SPLITS,
        minimum_mean_holdout_correlation=float(
            args.minimum_holdout_correlation
        ),
        maximum_mean_holdout_nrmse=float(args.maximum_holdout_nrmse),
        source={
            "evidence_tier": "independent_numerical_reference",
            "protocol": "m2_9_cross_boundary_shared_parameter_test",
        },
        applicability={
            "scope": "protocol_diagnostic_only",
            "renderer_compatible": False,
            "production_material_mapping_enabled": False,
        },
    )
    scales = [
        complex(
            result["calibration"]["complex_scale"]["real"],
            result["calibration"]["complex_scale"]["imag"],
        )
        for result in boundary_results
    ]
    magnitudes = [abs(value) for value in scales]
    phases = [cmath.phase(value) for value in scales]
    exponents = [
        float(result["calibration"]["frequency_exponent"])
        for result in boundary_results
    ]
    parameter_spread = {
        "maximum_to_minimum_scale_magnitude_ratio": float(
            max(magnitudes) / min(magnitudes)
        ),
        "complex_scale_phase_span_rad": float(max(phases) - min(phases)),
        "frequency_exponent_span": float(max(exponents) - min(exponents)),
    }
    per_boundary_accepted = all(
        result["acceptance"]["accepted"]
        for result in boundary_results
    )
    shared_accepted = bool(shared_fit.report["acceptance"]["accepted"])
    report = {
        "schema_version": PROTOCOL_SCHEMA_VERSION,
        "protocol": {
            "frequency_range_hz": [
                float(args.minimum_frequency_hz),
                float(args.maximum_frequency_hz),
            ],
            "pole_policy": "fixed_no_refit",
            "train_room_ids": ["room_a", "room_b"],
            "unseen_room_ids": ["room_c_unseen"],
            "acceptance_splits": list(ACCEPTANCE_SPLITS),
            "coarse_nominal_grid_spacing_m": float(args.grid_spacing_m),
            "fine_nominal_grid_spacing_m": float(args.fine_grid_spacing_m),
            "minimum_holdout_correlation": float(
                args.minimum_holdout_correlation
            ),
            "maximum_holdout_nrmse": float(args.maximum_holdout_nrmse),
        },
        "boundary_results": boundary_results,
        "shared_boundary_parameter_test": {
            "calibration": shared_fit.calibration.metadata(),
            "aggregate": shared_fit.report["aggregate"],
            "acceptance": shared_fit.report["acceptance"],
        },
        "per_boundary_parameter_spread": parameter_spread,
        "decision": {
            "scope": (
                "two_model_derived_rigid_backed_thickness_variants_of_one_"
                "measured_flow_resistivity"
            ),
            "per_boundary_calibrations_accepted": per_boundary_accepted,
            "shared_boundary_invariant_calibration_accepted": shared_accepted,
            "boundary_conditioning_required_within_tested_variants": (
                not shared_accepted
            ),
            "general_boundary_invariance_established": False,
            "production_material_mapping_enabled": False,
            "measured_room_transfer_validated": False,
            "protocol_accepted": per_boundary_accepted,
        },
    }
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.output_report),
                "decision": report["decision"],
                "parameter_spread": parameter_spread,
                "per_boundary": [
                    {
                        "name": result["name"],
                        "accepted": result["acceptance"]["accepted"],
                        "aggregate": result["aggregate"],
                    }
                    for result in boundary_results
                ],
            },
            indent=2,
            allow_nan=False,
        )
    )
    if not per_boundary_accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
