#!/usr/bin/env python3
"""Build the M3.2 passive causal PathEvent boundary-filter report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from puresound.audio.acoustic_impedance import (
    DIGITAL_BOUNDARY_FILTER_SCHEMA_VERSION,
    FirstOrderRelaxationAdmittance,
    PassiveMultiPoleAdmittance,
    PassiveResonantAdmittance,
    digital_locally_reacting_reflection_filter,
)
from puresound.audio.impedance_modes import (
    RectangularImpedanceBoundaryConfig,
)


REPORT_SCHEMA_VERSION = "puresound.path_event_filter_validation.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate positive-real-to-bounded-real angle filters and audit "
            "the M3.2 causal PathEvent rerun of the frozen M2.12 rooms."
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
            "full_room_crossover_m3_2_report.json"
        ),
    )
    parser.add_argument("--output-report", type=Path, required=True)
    parser.add_argument(
        "--maximum-reference-complex-error",
        type=float,
        default=0.003,
    )
    parser.add_argument(
        "--maximum-reference-magnitude-error-db",
        type=float,
        default=0.025,
    )
    parser.add_argument(
        "--maximum-reference-phase-error-deg",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--maximum-path-renderer-nrmse",
        type=float,
        default=0.01,
    )
    parser.add_argument(
        "--minimum-path-renderer-correlation",
        type=float,
        default=0.999,
    )
    parser.add_argument(
        "--path-renderer-energy-ratio-range",
        type=float,
        nargs=2,
        default=(0.99, 1.01),
    )
    return parser.parse_args()


def _angle_filter_case(
    *,
    case_id: str,
    model: Any,
    sample_rate_hz: float,
    minimum_frequency_hz: float,
    maximum_frequency_hz: float,
    incidence_cosines: tuple[float, ...],
) -> dict[str, object]:
    frequencies = np.linspace(
        minimum_frequency_hz,
        maximum_frequency_hz,
        481,
    )
    angle_reports = []
    for incidence_cosine in incidence_cosines:
        digital_filter = digital_locally_reacting_reflection_filter(
            model,
            incidence_cosine,
            sample_rate_hz,
        )
        analog = np.asarray(
            [
                (
                    incidence_cosine
                    - model.normalized_admittance(float(frequency))
                )
                / (
                    incidence_cosine
                    + model.normalized_admittance(float(frequency))
                )
                for frequency in frequencies
            ],
            dtype=np.complex128,
        )
        digital = np.asarray(
            [
                digital_filter.frequency_response(float(frequency))
                for frequency in frequencies
            ],
            dtype=np.complex128,
        )
        ratio = digital / analog
        sweep_frequencies = np.linspace(
            0.0,
            0.5 * sample_rate_hz,
            4097,
        )
        maximum_reflection = max(
            abs(digital_filter.frequency_response(float(frequency)))
            for frequency in sweep_frequencies
        )
        impulse = digital_filter.impulse_response(
            max(256, round(0.1 * sample_rate_hz))
        )
        angle_reports.append(
            {
                "incidence_cosine": incidence_cosine,
                "maximum_complex_error": float(
                    np.max(np.abs(digital - analog))
                ),
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
                    np.max(np.abs(np.angle(ratio, deg=True)))
                ),
                "maximum_all_band_reflection_magnitude": float(
                    maximum_reflection
                ),
                "maximum_pole_magnitude": (
                    digital_filter.maximum_pole_magnitude
                ),
                "impulse_is_finite": bool(np.all(np.isfinite(impulse))),
                "first_impulse_sample": float(impulse[0]),
                "filter_metadata": digital_filter.to_dict(),
            }
        )
    return {
        "case_id": case_id,
        "model": model.metadata(),
        "sample_rate_hz": sample_rate_hz,
        "comparison_band_hz": [
            minimum_frequency_hz,
            maximum_frequency_hz,
        ],
        "angles": angle_reports,
        "maximum_complex_error": float(
            max(item["maximum_complex_error"] for item in angle_reports)
        ),
        "maximum_magnitude_error_db": float(
            max(
                item["maximum_magnitude_error_db"]
                for item in angle_reports
            )
        ),
        "maximum_phase_error_deg": float(
            max(item["maximum_phase_error_deg"] for item in angle_reports)
        ),
        "maximum_all_band_reflection_magnitude": float(
            max(
                item["maximum_all_band_reflection_magnitude"]
                for item in angle_reports
            )
        ),
        "maximum_pole_magnitude": float(
            max(item["maximum_pole_magnitude"] for item in angle_reports)
        ),
        "all_impulses_finite": bool(
            all(item["impulse_is_finite"] for item in angle_reports)
        ),
    }


def _load_full_room_audit(path: Path) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    configuration = report.get("configuration", {})
    if not configuration.get("include_causal_path_event_first_order"):
        raise ValueError(
            "full-room report does not include the M3.2 PathEvent diagnostic"
        )
    cases = []
    for case in report["cases"]:
        realization = (
            case.get("causal_path_event")
            or case.get("causal_path_event_first_order")
        )
        if realization is None:
            raise ValueError("full-room case is missing PathEvent realization")
        cases.append(
            {
                "case_id": case["case"]["case_id"],
                **realization,
            }
        )
    return {
        "source_report": str(path),
        "source_report_schema_version": report["schema_version"],
        "protocol_completed": bool(
            report["acceptance"]["protocol_completed"]
        ),
        "cases": cases,
        "renderer_vs_analytic_aggregate": {
            "maximum_complex_nrmse": float(
                max(
                    case["time_renderer_vs_analytic_complex_angle"][
                        "complex_nrmse"
                    ]
                    for case in cases
                )
            ),
            "minimum_complex_correlation": float(
                min(
                    case["time_renderer_vs_analytic_complex_angle"][
                        "complex_correlation"
                    ]
                    for case in cases
                )
            ),
            "minimum_transfer_energy_ratio": float(
                min(
                    case["time_renderer_vs_analytic_complex_angle"][
                        "transfer_energy_ratio"
                    ]
                    for case in cases
                )
            ),
            "maximum_transfer_energy_ratio": float(
                max(
                    case["time_renderer_vs_analytic_complex_angle"][
                        "transfer_energy_ratio"
                    ]
                    for case in cases
                )
            ),
        },
        "first_order_full_room_diagnostic": (
            report["diagnostic"].get("causal_path_event")
            or report["diagnostic"]["causal_path_event_first_order"]
        ),
        "frozen_m2_12_full_room_gate_accepted": bool(
            report["acceptance"]["full_room_complex_gate_accepted"]
        ),
    }


def main() -> None:
    args = _parse_args()
    boundary_config = RectangularImpedanceBoundaryConfig.from_json(
        args.boundary_config
    )
    reference_models = tuple(boundary_config.boundaries.values())
    reference_model = reference_models[0]
    if any(
        model.metadata() != reference_model.metadata()
        for model in reference_models[1:]
    ):
        raise ValueError("M3.2 reference filter audit requires uniform walls")
    incidence_cosines = (0.05, 0.1, 0.25, 0.5, 1.0)
    reference_cases = [
        _angle_filter_case(
            case_id=f"m2_10_reference_{sample_rate:g}hz",
            model=reference_model,
            sample_rate_hz=sample_rate,
            minimum_frequency_hz=60.0,
            maximum_frequency_hz=300.0,
            incidence_cosines=incidence_cosines,
        )
        for sample_rate in (8000.0, 16000.0, 48000.0)
    ]
    general_model_cases = [
        _angle_filter_case(
            case_id="positive_real_multi_pole_16khz",
            model=PassiveMultiPoleAdmittance(
                normalized_admittance_static=0.02,
                pole_frequencies_hz=(80.0, 400.0),
                normalized_admittance_lowpass=(0.10, 0.03),
                normalized_admittance_highpass=(0.40, 0.70),
            ),
            sample_rate_hz=16000.0,
            minimum_frequency_hz=20.0,
            maximum_frequency_hz=3000.0,
            incidence_cosines=incidence_cosines,
        ),
        _angle_filter_case(
            case_id="passive_resonant_rlc_192khz",
            model=PassiveResonantAdmittance(
                normalized_admittance_static=0.03,
                resonance_frequencies_hz=(1600.0,),
                quality_factors=(12.0,),
                peak_normalized_admittances=(7.5,),
            ),
            sample_rate_hz=192000.0,
            minimum_frequency_hz=200.0,
            maximum_frequency_hz=6000.0,
            incidence_cosines=incidence_cosines,
        ),
        _angle_filter_case(
            case_id="positive_relaxation_reference_16khz",
            model=FirstOrderRelaxationAdmittance(
                normalized_admittance_infinite=0.18,
                normalized_admittance_relaxation=0.55,
                relaxation_frequency_hz=110.0,
            ),
            sample_rate_hz=16000.0,
            minimum_frequency_hz=20.0,
            maximum_frequency_hz=1000.0,
            incidence_cosines=incidence_cosines,
        ),
    ]
    full_room = _load_full_room_audit(args.full_room_report)
    reference_filter_accepted = all(
        case["maximum_complex_error"]
        <= float(args.maximum_reference_complex_error)
        and case["maximum_magnitude_error_db"]
        <= float(args.maximum_reference_magnitude_error_db)
        and case["maximum_phase_error_deg"]
        <= float(args.maximum_reference_phase_error_deg)
        and case["maximum_all_band_reflection_magnitude"] <= 1.0 + 1e-10
        and case["maximum_pole_magnitude"] < 1.0
        and case["all_impulses_finite"]
        for case in reference_cases
    )
    all_models_bounded_real = all(
        case["maximum_all_band_reflection_magnitude"] <= 1.0 + 1e-10
        and case["maximum_pole_magnitude"] < 1.0
        and case["all_impulses_finite"]
        for case in [*reference_cases, *general_model_cases]
    )
    realization = full_room["renderer_vs_analytic_aggregate"]
    energy_minimum, energy_maximum = (
        float(value) for value in args.path_renderer_energy_ratio_range
    )
    path_renderer_accepted = bool(
        realization["maximum_complex_nrmse"]
        <= float(args.maximum_path_renderer_nrmse)
        and realization["minimum_complex_correlation"]
        >= float(args.minimum_path_renderer_correlation)
        and realization["minimum_transfer_energy_ratio"] >= energy_minimum
        and realization["maximum_transfer_energy_ratio"] <= energy_maximum
    )
    accepted = bool(
        reference_filter_accepted
        and all_models_bounded_real
        and path_renderer_accepted
        and full_room["protocol_completed"]
    )
    report = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "digital_filter_schema_version": (
            DIGITAL_BOUNDARY_FILTER_SCHEMA_VERSION
        ),
        "boundary_config": str(args.boundary_config),
        "boundary_reference_id": boundary_config.reference_id,
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
            if key != "output_report"
        },
        "physical_contract": {
            "continuous_reflection": (
                "Gamma(theta,s)=(cos(theta)-y(s))/(cos(theta)+y(s))"
            ),
            "digital_admittance": "Y(z)=B(z)/A(z)",
            "digital_reflection": (
                "Gamma_theta(z)=(cos(theta)A(z)-B(z))/"
                "(cos(theta)A(z)+B(z))"
            ),
            "discretization": (
                "bilinear transform of a positive-real admittance followed "
                "by a Cayley transform to a bounded-real reflection filter"
            ),
            "raw_complex_spectrum_ifft_used": False,
            "scalar_gain_fitted": False,
        },
        "m2_reference_filter_cases": reference_cases,
        "general_passive_model_cases": general_model_cases,
        "full_room_rerun": full_room,
        "acceptance": {
            "maximum_reference_complex_error": float(
                args.maximum_reference_complex_error
            ),
            "maximum_reference_magnitude_error_db": float(
                args.maximum_reference_magnitude_error_db
            ),
            "maximum_reference_phase_error_deg": float(
                args.maximum_reference_phase_error_deg
            ),
            "maximum_path_renderer_nrmse": float(
                args.maximum_path_renderer_nrmse
            ),
            "minimum_path_renderer_correlation": float(
                args.minimum_path_renderer_correlation
            ),
            "path_renderer_energy_ratio_range": [
                energy_minimum,
                energy_maximum,
            ],
            "reference_filter_accepted": bool(reference_filter_accepted),
            "all_rational_models_bounded_real": bool(
                all_models_bounded_real
            ),
            "path_renderer_matches_first_order_analytic": bool(
                path_renderer_accepted
            ),
            "m3_2_filter_realization_accepted": accepted,
            "frozen_m2_12_full_room_gate_accepted": bool(
                full_room["frozen_m2_12_full_room_gate_accepted"]
            ),
        },
        "scope": {
            "validated": [
                "first-order relaxation angle filter at 8/16/48 kHz",
                "multi-pole relaxation bounded-real digital reflection",
                "resonant RLC bounded-real digital reflection",
                "causal PathEvent rendering against seven analytic image paths",
                "three frozen M2.12 train/position/room cases",
            ],
            "full_order_12_path_events_rendered": False,
            "mesh_visibility_validated": False,
            "production_hybrid_backend_replaced": False,
            "measured_room_validated": False,
        },
        "decision": (
            "accept_m3_2_causal_filter_realization_keep_full_room_gate_open"
            if accepted
            else "reject_m3_2_filter_realization"
        ),
        "next_action": (
            "increase the coherent PathEvent set beyond first order or connect "
            "a mesh tracer, then separate direct/early and late energy before "
            "repeating the full-room crossover and measured C50 gates"
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
                "m3_2_filter_realization_accepted": accepted,
                "maximum_reference_complex_error": max(
                    case["maximum_complex_error"]
                    for case in reference_cases
                ),
                "maximum_reference_magnitude_error_db": max(
                    case["maximum_magnitude_error_db"]
                    for case in reference_cases
                ),
                "maximum_reference_phase_error_deg": max(
                    case["maximum_phase_error_deg"]
                    for case in reference_cases
                ),
                "path_renderer": realization,
                "frozen_m2_12_full_room_gate_accepted": full_room[
                    "frozen_m2_12_full_room_gate_accepted"
                ],
            },
            indent=2,
        )
    )
    if not accepted:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
