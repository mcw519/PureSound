#!/usr/bin/env python3
"""Validate the FDTD-to-digital-RIR source-convention conversion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from puresound.audio.rir.physics.impedance.admittance import (
    FirstOrderRelaxationAdmittance,
)
from puresound.audio.rir.physics.wave.fdtd import (
    FDTDReferenceConfig,
    ricker_source,
    simulate_fdtd_reference,
)
from puresound.audio.rir.physics.wave.source_convention import (
    FDTD_PRESSURE_CELL_SOURCE_CONVENTION,
    FREE_FIELD_1_OVER_R_RIR_CONVENTION,
    PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM,
    convert_pressure_state_modal_residue,
    fdtd_cell_center_position,
    fdtd_pressure_cell_free_field_direct,
    fdtd_pressure_cell_to_free_field_input,
)


REPORT_SCHEMA_VERSION = "puresound.rir_source_convention_validation.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate the Green's-function scale of the FDTD pressure-cell "
            "source and its conversion to a discrete 1/r RIR."
        )
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--minimum-direct-correlation", type=float, default=0.95)
    parser.add_argument(
        "--maximum-direct-amplitude-error",
        type=float,
        default=0.15,
    )
    parser.add_argument(
        "--maximum-modal-identity-nrmse",
        type=float,
        default=0.007,
    )
    parser.add_argument(
        "--maximum-sample-rate-response-error",
        type=float,
        default=0.05,
    )
    return parser.parse_args()


def _shifted(signal: np.ndarray, samples: int) -> np.ndarray:
    output = np.zeros_like(signal)
    if samples > 0:
        output[samples:] = signal[:-samples]
    elif samples < 0:
        output[:samples] = signal[-samples:]
    else:
        output[:] = signal
    return output


def _direct_path_cases(args: argparse.Namespace) -> list[dict[str, object]]:
    cases = (
        {
            "case_id": "axis_grid_50mm",
            "grid_spacing_m": 0.05,
            "source_position_m": (0.275, 0.625, 0.625),
            "receiver_position_m": (0.825, 0.625, 0.625),
        },
        {
            "case_id": "axis_grid_40mm",
            "grid_spacing_m": 0.04,
            "source_position_m": (0.30, 0.60, 0.60),
            "receiver_position_m": (0.82, 0.60, 0.60),
        },
        {
            "case_id": "diagonal_grid_30mm",
            "grid_spacing_m": 0.03,
            "source_position_m": (0.30, 0.30, 0.30),
            "receiver_position_m": (0.78, 0.70, 0.62),
        },
    )
    matched_boundary = FirstOrderRelaxationAdmittance(
        normalized_admittance_infinite=1.0,
        normalized_admittance_relaxation=0.0,
        relaxation_frequency_hz=1.0,
    )
    reports: list[dict[str, object]] = []
    for case in cases:
        config = FDTDReferenceConfig(
            room_dim_m=(1.2, 1.2, 1.2),
            grid_spacing_m=float(case["grid_spacing_m"]),
            duration_s=0.05,
            source_position_m=case["source_position_m"],
            receiver_position_m=case["receiver_position_m"],
            source_center_hz=400.0,
            source_delay_s=0.015,
        )
        result = simulate_fdtd_reference(
            config,
            boundary_admittance=matched_boundary,
        )
        source_position = np.asarray(
            fdtd_cell_center_position(
                result.source_cell_zyx,
                result.grid_spacing_xyz_m,
            )
        )
        receiver_position = np.asarray(
            fdtd_cell_center_position(
                result.receiver_cell_zyx,
                result.grid_spacing_xyz_m,
            )
        )
        distance_m = float(
            np.linalg.norm(receiver_position - source_position)
        )
        source = ricker_source(
            result.rir.size,
            result.time_step_s,
            config.source_center_hz,
            config.source_delay_s,
        )
        predicted = fdtd_pressure_cell_free_field_direct(
            source,
            time_step_s=result.time_step_s,
            cell_volume_m3=float(np.prod(result.grid_spacing_xyz_m)),
            distance_m=distance_m,
            sound_speed_m_s=config.sound_speed_m_s,
        )
        time_s = np.arange(result.rir.size) * result.time_step_s
        direct_center_s = (
            config.source_delay_s
            + distance_m / config.sound_speed_m_s
        )
        direct_window = np.abs(time_s - direct_center_s) < 0.004
        candidates = []
        for offset_samples in range(-2, 3):
            shifted = _shifted(predicted, offset_samples)
            denominator = float(
                np.dot(shifted[direct_window], shifted[direct_window])
            )
            amplitude_scale = float(
                np.dot(result.rir[direct_window], shifted[direct_window])
                / max(denominator, 1e-30)
            )
            fitted = amplitude_scale * shifted[direct_window]
            correlation = float(
                np.corrcoef(result.rir[direct_window], fitted)[0, 1]
            )
            nrmse = float(
                np.linalg.norm(result.rir[direct_window] - fitted)
                / np.linalg.norm(result.rir[direct_window])
            )
            candidates.append(
                (
                    correlation,
                    -nrmse,
                    offset_samples,
                    amplitude_scale,
                    nrmse,
                )
            )
        (
            correlation,
            _negative_nrmse,
            offset_samples,
            amplitude_scale,
            nrmse,
        ) = max(candidates)
        accepted = bool(
            correlation >= float(args.minimum_direct_correlation)
            and abs(amplitude_scale - 1.0)
            <= float(args.maximum_direct_amplitude_error)
            and abs(offset_samples) <= 1
        )
        reports.append(
            {
                "case_id": case["case_id"],
                "grid_spacing_xyz_m": list(result.grid_spacing_xyz_m),
                "sample_rate_hz": float(result.sample_rate_hz),
                "source_cell_zyx": list(result.source_cell_zyx),
                "receiver_cell_zyx": list(result.receiver_cell_zyx),
                "cell_center_distance_m": distance_m,
                "theoretical_arrival_s": (
                    distance_m / config.sound_speed_m_s
                ),
                "best_time_offset_samples": int(offset_samples),
                "fitted_over_theoretical_amplitude": amplitude_scale,
                "amplitude_relative_error": abs(amplitude_scale - 1.0),
                "correlation": correlation,
                "nrmse_after_amplitude_fit": nrmse,
                "accepted": accepted,
            }
        )
    return reports


def _modal_identity_cases() -> list[dict[str, float]]:
    reports = []
    for sample_rate_hz in (4000.0, 8000.0, 32000.0):
        time_step_s = 1.0 / sample_rate_hz
        time_s = (
            np.arange(round(0.3 * sample_rate_hz), dtype=np.float64)
            * time_step_s
        )
        sound_speed_m_s = 343.0
        cell_volume_m3 = 0.04**3
        pole = -20.0 + 1j * 2.0 * math.pi * 120.0
        coupling = 0.3 + 0.1j
        argument = math.pi * 150.0 * (time_s - 0.025)
        source = (1.0 - 2.0 * argument**2) * np.exp(-(argument**2))
        pressure_state_mode = np.real(
            cell_volume_m3 * coupling * np.exp(pole * time_s)
        )
        expected = fftconvolve(source, pressure_state_mode)[: time_s.size]
        equivalent_input = fdtd_pressure_cell_to_free_field_input(
            source,
            time_step_s=time_step_s,
            cell_volume_m3=cell_volume_m3,
            sound_speed_m_s=sound_speed_m_s,
        )
        digital_residue = convert_pressure_state_modal_residue(
            coupling,
            pole,
            sample_rate_hz=sample_rate_hz,
            sound_speed_m_s=sound_speed_m_s,
        )
        digital_mode = np.real(
            digital_residue * np.exp(pole * time_s)
        )
        actual = fftconvolve(
            equivalent_input,
            digital_mode,
        )[: time_s.size]
        comparison = time_s >= 0.05
        nrmse = float(
            np.linalg.norm(actual[comparison] - expected[comparison])
            / np.linalg.norm(expected[comparison])
        )
        correlation = float(
            np.corrcoef(actual[comparison], expected[comparison])[0, 1]
        )
        reports.append(
            {
                "sample_rate_hz": sample_rate_hz,
                "nrmse": nrmse,
                "correlation": correlation,
            }
        )
    return reports


def _sample_rate_invariance() -> dict[str, object]:
    sound_speed_m_s = 343.0
    distance_m = sound_speed_m_s * 0.0015
    frequencies_hz = np.linspace(60.0, 240.0, 37)
    poles = (
        -14.0 + 1j * 2.0 * math.pi * 90.0,
        -22.0 + 1j * 2.0 * math.pi * 160.0,
    )
    residues = (0.5 + 0.1j, -0.2 + 0.35j)

    def response(sample_rate_hz: float) -> np.ndarray:
        time_s = (
            np.arange(round(0.8 * sample_rate_hz), dtype=np.float64)
            / sample_rate_hz
        )
        rir = np.zeros(time_s.size, dtype=np.float64)
        direct_index = round(
            distance_m / sound_speed_m_s * sample_rate_hz
        )
        rir[direct_index] = 1.0 / distance_m
        for pole, residue in zip(poles, residues):
            converted = convert_pressure_state_modal_residue(
                residue,
                pole,
                sample_rate_hz=sample_rate_hz,
                sound_speed_m_s=sound_speed_m_s,
            )
            rir += np.real(converted * np.exp(pole * time_s))
        return np.asarray(
            [
                np.sum(
                    rir
                    * np.exp(
                        -2j * math.pi * frequency_hz * time_s
                    )
                )
                for frequency_hz in frequencies_hz
            ]
        )

    reference_sample_rate_hz = 32000.0
    reference = response(reference_sample_rate_hz)
    comparisons = []
    for sample_rate_hz in (4000.0, 8000.0, 16000.0):
        candidate = response(sample_rate_hz)
        relative_error = np.abs(candidate - reference) / np.maximum(
            np.abs(reference),
            1e-12,
        )
        comparisons.append(
            {
                "sample_rate_hz": sample_rate_hz,
                "mean_complex_response_relative_error": float(
                    np.mean(relative_error)
                ),
                "maximum_complex_response_relative_error": float(
                    np.max(relative_error)
                ),
            }
        )
    return {
        "frequency_range_hz": [
            float(frequencies_hz[0]),
            float(frequencies_hz[-1]),
        ],
        "frequency_count": int(frequencies_hz.size),
        "reference_sample_rate_hz": reference_sample_rate_hz,
        "comparisons": comparisons,
    }


def main() -> None:
    args = _parse_args()
    direct_cases = _direct_path_cases(args)
    modal_cases = _modal_identity_cases()
    sample_rate = _sample_rate_invariance()
    direct_accepted = all(bool(case["accepted"]) for case in direct_cases)
    modal_accepted = all(
        float(case["nrmse"]) <= float(args.maximum_modal_identity_nrmse)
        for case in modal_cases
    )
    sample_rate_accepted = all(
        float(case["maximum_complex_response_relative_error"])
        <= float(args.maximum_sample_rate_response_error)
        for case in sample_rate["comparisons"]
    )
    accepted = direct_accepted and modal_accepted and sample_rate_accepted
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "conventions": {
            "fitted_fdtd_source": FDTD_PRESSURE_CELL_SOURCE_CONVENTION,
            "rendered_rir": FREE_FIELD_1_OVER_R_RIR_CONVENTION,
            "modal_residue_transform": (
                PRESSURE_STATE_TO_FREE_FIELD_RESIDUE_TRANSFORM
            ),
        },
        "derivation": {
            "fdtd_pressure_equation_source": (
                "dp/dt + rho*c^2*div(v) = "
                "(cell_volume/dt)*q(t)*delta(x-xs)"
            ),
            "fdtd_free_field_direct": (
                "p(r,t) = cell_volume/(dt*4*pi*c^2*r) "
                "* dq(t-r/c)/dt"
            ),
            "digital_rir_direct": "h_direct(t) = delta(t-r/c)/r",
            "equivalent_digital_input": (
                "x(t) = cell_volume/(dt*4*pi*c^2) * dq(t)/dt"
            ),
            "modal_residue_conversion": (
                "R_digital = R_pressure_state * 4*pi*c^2/(fs*pole)"
            ),
        },
        "fdtd_direct_path_cases": direct_cases,
        "modal_transform_identity_cases": modal_cases,
        "sample_rate_invariance": sample_rate,
        "acceptance": {
            "minimum_direct_correlation": float(
                args.minimum_direct_correlation
            ),
            "maximum_direct_amplitude_error": float(
                args.maximum_direct_amplitude_error
            ),
            "maximum_direct_time_offset_samples": 1,
            "maximum_modal_identity_nrmse": float(
                args.maximum_modal_identity_nrmse
            ),
            "maximum_sample_rate_response_error": float(
                args.maximum_sample_rate_response_error
            ),
            "direct_path_accepted": direct_accepted,
            "modal_transform_accepted": modal_accepted,
            "sample_rate_invariance_accepted": sample_rate_accepted,
            "accepted": accepted,
        },
        "scope": {
            "fdtd_boundary": "matched_local_normal_incidence",
            "fdtd_source": "unit_peak_400_hz_ricker_pressure_cell_increment",
            "direct_path_geometry": "axis_aligned_and_diagonal",
            "modal_identity": "synthetic_single_fixed_pole_band_limited_input",
            "sample_rate_band_hz": [60.0, 240.0],
            "production_measured_room_validated": False,
            "hybrid_crossover_level_calibrated": False,
        },
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
                "minimum_direct_correlation": min(
                    float(case["correlation"]) for case in direct_cases
                ),
                "maximum_direct_amplitude_error": max(
                    float(case["amplitude_relative_error"])
                    for case in direct_cases
                ),
                "maximum_modal_identity_nrmse": max(
                    float(case["nrmse"]) for case in modal_cases
                ),
                "maximum_sample_rate_response_error": max(
                    float(case["maximum_complex_response_relative_error"])
                    for case in sample_rate["comparisons"]
                ),
            },
            indent=2,
        )
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
