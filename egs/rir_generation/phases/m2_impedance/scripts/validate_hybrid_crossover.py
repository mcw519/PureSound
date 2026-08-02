#!/usr/bin/env python3
"""Validate complex direct-path behavior through the hybrid crossover."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfreqz

from puresound.audio.hybrid_rir import (
    HybridRIRConfig,
    HybridRIRScene,
    PyroomacousticsHighFrequencyBackend,
    _hybrid_crossover_with_metadata,
    hybrid_crossover,
)


REPORT_SCHEMA_VERSION = "puresound.hybrid_crossover_validation.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate Linkwitz-Riley complex complementarity, quantify the "
            "legacy RMS-match distortion, and audit anechoic 1/r direct paths."
        )
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument("--crossover-hz", type=float, default=240.0)
    parser.add_argument(
        "--maximum-direct-magnitude-error-db",
        type=float,
        default=1.1,
    )
    parser.add_argument(
        "--maximum-direct-phase-error-deg",
        type=float,
        default=6.0,
    )
    return parser.parse_args()


def _filter_responses(
    sample_rate_hz: float,
    crossover_hz: float,
    frequencies_hz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    _frequency, lowpass_once = sosfreqz(
        lowpass,
        worN=frequencies_hz,
        fs=sample_rate_hz,
    )
    _frequency, highpass_once = sosfreqz(
        highpass,
        worN=frequencies_hz,
        fs=sample_rate_hz,
    )
    low_branch = lowpass_once**2
    high_branch = highpass_once**2
    return low_branch, high_branch, low_branch + high_branch


def _filter_complementarity(
    sample_rate_hz: float,
    crossover_hz: float,
) -> dict[str, float]:
    frequencies_hz = np.geomspace(
        20.0,
        0.45 * sample_rate_hz,
        2048,
    )
    low_branch, high_branch, combined = _filter_responses(
        sample_rate_hz,
        crossover_hz,
        frequencies_hz,
    )
    crossover_region = (
        (frequencies_hz >= 0.5 * crossover_hz)
        & (frequencies_hz <= 2.0 * crossover_hz)
    )
    phase_difference_deg = np.rad2deg(
        np.angle(low_branch[crossover_region] / high_branch[crossover_region])
    )
    crossover_index = int(
        np.argmin(np.abs(frequencies_hz - crossover_hz))
    )
    return {
        "sample_rate_hz": float(sample_rate_hz),
        "maximum_combined_magnitude_error_db": float(
            np.max(
                np.abs(
                    20.0 * np.log10(np.maximum(np.abs(combined), 1e-15))
                )
            )
        ),
        "maximum_branch_phase_difference_deg": float(
            np.max(np.abs(phase_difference_deg))
        ),
        "low_branch_magnitude_at_crossover_db": float(
            20.0
            * np.log10(max(abs(low_branch[crossover_index]), 1e-15))
        ),
        "high_branch_magnitude_at_crossover_db": float(
            20.0
            * np.log10(max(abs(high_branch[crossover_index]), 1e-15))
        ),
    }


def _common_source_energy_match_audit(
    sample_rate_hz: float,
    crossover_hz: float,
) -> dict[str, object]:
    config = HybridRIRConfig(
        sample_rate=int(sample_rate_hz),
        duration=0.1,
        crossover_hz=float(crossover_hz),
        output_mode="calibrated",
        match_crossover_energy=True,
    )
    direct = np.zeros((config.num_sources, config.num_samples))
    direct[:, round(0.01 * sample_rate_hz)] = 1.0
    _output, metadata = _hybrid_crossover_with_metadata(
        direct,
        direct,
        config,
    )
    gain = float(metadata["low_band_gain_by_channel"][0])
    frequencies_hz = np.linspace(
        max(20.0, 0.25 * crossover_hz),
        min(4.0 * crossover_hz, 0.45 * sample_rate_hz),
        1024,
    )
    low_branch, high_branch, _combined = _filter_responses(
        sample_rate_hz,
        crossover_hz,
        frequencies_hz,
    )
    rescaled_combined = gain * low_branch + high_branch
    magnitude_error_db = 20.0 * np.log10(
        np.maximum(np.abs(rescaled_combined), 1e-15)
    )
    return {
        "sample_rate_hz": float(sample_rate_hz),
        "effective_match_band_hz": metadata["effective_match_band_hz"],
        "estimated_low_gain_for_identical_inputs": gain,
        "maximum_flat_sum_magnitude_error_db": float(
            np.max(np.abs(magnitude_error_db))
        ),
        "interpretation": (
            "Any gain other than one distorts an already matched common "
            "source; therefore this estimator is not applied to a backend "
            "with a validated source convention."
        ),
    }


def _anechoic_direct_case(
    sample_rate_hz: int,
    crossover_hz: float,
) -> dict[str, object]:
    config = HybridRIRConfig(
        sample_rate=sample_rate_hz,
        duration=0.08,
        crossover_hz=crossover_hz,
        low_fmin_hz=60.0,
        low_fmax_hz=crossover_hz,
        output_mode="calibrated",
        match_crossover_energy=False,
    )
    distances_m = (0.8, 1.0, 1.5, 2.0, 3.0)
    microphone = np.asarray([4.0, 4.0, 4.0])
    source_positions = [
        [4.0 - distance, 4.0, 4.0] for distance in distances_m
    ]
    scene = HybridRIRScene(
        room_dim=[8.0, 8.0, 8.0],
        rt60=0.5,
        mic_pos=microphone.tolist(),
        source_pos=source_positions,
        source_labels=[
            f"direct_{index}" for index in range(len(distances_m))
        ],
    )
    high = PyroomacousticsHighFrequencyBackend(
        max_order=0,
        ray_tracing=False,
        air_absorption=False,
    ).simulate(scene, config)
    low = np.zeros_like(high, dtype=np.float64)
    for channel, distance_m in enumerate(distances_m):
        direct_index = round(
            distance_m / config.sound_speed * sample_rate_hz
        )
        low[channel, direct_index] = 1.0 / distance_m
    combined = np.asarray(
        hybrid_crossover(low, high, config),
        dtype=np.float64,
    )
    frequencies_hz = np.linspace(60.0, 1000.0, 95)
    low_filter, high_filter, common_filter = _filter_responses(
        sample_rate_hz,
        crossover_hz,
        frequencies_hz,
    )
    del low_filter, high_filter
    time_s = np.arange(combined.shape[-1]) / float(sample_rate_hz)
    magnitude_errors_db: list[float] = []
    phase_errors_deg: list[float] = []
    channel_reports = []
    for channel, distance_m in enumerate(distances_m):
        response = np.asarray(
            [
                np.sum(
                    combined[channel]
                    * np.exp(-2j * math.pi * frequency_hz * time_s)
                )
                for frequency_hz in frequencies_hz
            ]
        )
        ideal = (
            np.exp(
                -2j
                * math.pi
                * frequencies_hz
                * distance_m
                / config.sound_speed
            )
            / distance_m
            * common_filter
        )
        ratio = response / ideal
        channel_magnitude_error_db = 20.0 * np.log10(
            np.maximum(np.abs(ratio), 1e-15)
        )
        channel_phase_error_deg = np.rad2deg(np.angle(ratio))
        magnitude_errors_db.extend(channel_magnitude_error_db.tolist())
        phase_errors_deg.extend(channel_phase_error_deg.tolist())
        channel_reports.append(
            {
                "distance_m": distance_m,
                "maximum_magnitude_error_db": float(
                    np.max(np.abs(channel_magnitude_error_db))
                ),
                "maximum_phase_error_deg": float(
                    np.max(np.abs(channel_phase_error_deg))
                ),
            }
        )
    return {
        "sample_rate_hz": sample_rate_hz,
        "frequency_range_hz": [
            float(frequencies_hz[0]),
            float(frequencies_hz[-1]),
        ],
        "crossover_hz": crossover_hz,
        "maximum_magnitude_error_db": float(
            np.max(np.abs(magnitude_errors_db))
        ),
        "mean_absolute_magnitude_error_db": float(
            np.mean(np.abs(magnitude_errors_db))
        ),
        "maximum_phase_error_deg": float(
            np.max(np.abs(phase_errors_deg))
        ),
        "mean_absolute_phase_error_deg": float(
            np.mean(np.abs(phase_errors_deg))
        ),
        "channels": channel_reports,
    }


def main() -> None:
    args = _parse_args()
    sample_rates_hz = (8000, 16000, 48000)
    filter_cases = [
        _filter_complementarity(
            sample_rate_hz,
            float(args.crossover_hz),
        )
        for sample_rate_hz in sample_rates_hz
    ]
    energy_audits = [
        _common_source_energy_match_audit(
            sample_rate_hz,
            float(args.crossover_hz),
        )
        for sample_rate_hz in sample_rates_hz
    ]
    direct_cases = [
        _anechoic_direct_case(
            sample_rate_hz,
            float(args.crossover_hz),
        )
        for sample_rate_hz in sample_rates_hz
    ]
    filter_accepted = all(
        case["maximum_combined_magnitude_error_db"] <= 1e-8
        and case["maximum_branch_phase_difference_deg"] <= 1e-8
        for case in filter_cases
    )
    direct_accepted = all(
        case["maximum_magnitude_error_db"]
        <= float(args.maximum_direct_magnitude_error_db)
        and case["maximum_phase_error_deg"]
        <= float(args.maximum_direct_phase_error_deg)
        for case in direct_cases
    )
    accepted = bool(filter_accepted and direct_accepted)
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "crossover_hz": float(args.crossover_hz),
        "source_convention": (
            "puresound.free_field_pressure_1_over_r_discrete_rir.v1"
        ),
        "filter_complementarity": filter_cases,
        "common_source_energy_match_audit": energy_audits,
        "anechoic_direct_backend_cases": direct_cases,
        "implementation_policy": {
            "validated_source_convention": (
                "skip per-channel low-band RMS rescaling"
            ),
            "uncalibrated_legacy_backend": (
                "retain bounded RMS matching as an explicitly reported bridge"
            ),
            "automatic_match_band": "[0.7*crossover, 1.3*crossover]",
            "explicit_match_band_requirement": (
                "must contain the configured crossover"
            ),
        },
        "acceptance": {
            "maximum_filter_magnitude_error_db": 1e-8,
            "maximum_filter_branch_phase_difference_deg": 1e-8,
            "maximum_direct_magnitude_error_db": float(
                args.maximum_direct_magnitude_error_db
            ),
            "maximum_direct_phase_error_deg": float(
                args.maximum_direct_phase_error_deg
            ),
            "filter_complementarity_accepted": filter_accepted,
            "anechoic_direct_crossover_accepted": direct_accepted,
            "accepted": accepted,
        },
        "scope": {
            "validated": [
                "digital Linkwitz-Riley complex complementarity",
                "1/r low direct plus pyroomacoustics anechoic direct",
                "source-convention-preserving gain policy",
            ],
            "full_room_modal_geometric_complex_match_validated": False,
            "boundary_reflection_phase_match_validated": False,
            "measured_room_crossover_validated": False,
            "production_material_mapping_validated": False,
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
                "maximum_filter_magnitude_error_db": max(
                    case["maximum_combined_magnitude_error_db"]
                    for case in filter_cases
                ),
                "maximum_direct_magnitude_error_db": max(
                    case["maximum_magnitude_error_db"]
                    for case in direct_cases
                ),
                "maximum_direct_phase_error_deg": max(
                    case["maximum_phase_error_deg"]
                    for case in direct_cases
                ),
            },
            indent=2,
        )
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
