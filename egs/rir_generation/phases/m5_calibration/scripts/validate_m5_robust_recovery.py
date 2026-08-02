#!/usr/bin/env python3
"""Validate M5.2b under noise, calibrated nuisance error, and model mismatch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

from puresound.audio.rir.calibration.synthetic_recovery import (
    PerturbedSyntheticMeasurement,
    RIR_ROBUST_RECOVERY_OBJECTIVE_POLICY,
    SyntheticMeasurementPerturbation,
    SyntheticRecoveryObjectiveConfig,
    SyntheticRecoveryObservation,
    SyntheticRecoveryParameters,
    build_synthetic_recovery_observation,
    evaluate_synthetic_recovery,
    fit_synthetic_recovery_parameters,
    perturb_synthetic_recovery_measurement,
    render_synthetic_recovery_rir,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_REPORT = REPO_ROOT / "egs/rir_generation/phases/m5_calibration/reports/m5_robust_recovery_report.json"
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_robust_recovery"
CENTERS_HZ = (500.0, 1000.0, 2000.0, 4000.0)


def _truth() -> SyntheticRecoveryParameters:
    return SyntheticRecoveryParameters(
        mixing_time_s=0.032,
        early_reflection_gain_db=-1.5,
        rt60_s_by_hz={
            500.0: 0.82,
            1000.0: 0.68,
            2000.0: 0.56,
            4000.0: 0.44,
        },
        late_gain_db_by_hz={
            500.0: -5.0,
            1000.0: -4.0,
            2000.0: -3.0,
            4000.0: -2.0,
        },
    )


def _initializations() -> tuple[SyntheticRecoveryParameters, ...]:
    return (
        SyntheticRecoveryParameters(
            mixing_time_s=0.054,
            early_reflection_gain_db=4.0,
            rt60_s_by_hz={
                500.0: 0.45,
                1000.0: 1.10,
                2000.0: 0.85,
                4000.0: 0.75,
            },
            late_gain_db_by_hz={
                500.0: 1.0,
                1000.0: -9.0,
                2000.0: 2.0,
                4000.0: -8.0,
            },
        ),
        SyntheticRecoveryParameters(
            mixing_time_s=0.019,
            early_reflection_gain_db=-8.0,
            rt60_s_by_hz={
                500.0: 1.40,
                1000.0: 0.38,
                2000.0: 1.20,
                4000.0: 0.30,
            },
            late_gain_db_by_hz={
                500.0: -13.0,
                1000.0: 5.0,
                2000.0: -11.0,
                4000.0: 4.0,
            },
        ),
    )


def _observations(
    prefix: str,
    distances: Sequence[float],
    seeds: Sequence[int],
    sample_rate: int,
    sample_count: int,
) -> tuple[SyntheticRecoveryObservation, ...]:
    return tuple(
        build_synthetic_recovery_observation(
            f"{prefix}-{index}",
            sample_rate,
            sample_count,
            distance,
            centers_hz=CENTERS_HZ,
            seed=seed,
        )
        for index, (distance, seed) in enumerate(zip(distances, seeds))
    )


def _measurements(
    observations: Sequence[SyntheticRecoveryObservation],
    truth: SyntheticRecoveryParameters,
    perturbations: Sequence[SyntheticMeasurementPerturbation],
) -> tuple[PerturbedSyntheticMeasurement, ...]:
    if len(observations) != len(perturbations):
        raise ValueError("observations and perturbations must have equal length")
    return tuple(
        perturb_synthetic_recovery_measurement(
            observation,
            truth,
            perturbation,
        )
        for observation, perturbation in zip(observations, perturbations)
    )


def _parameter_errors(
    truth: SyntheticRecoveryParameters,
    fitted: SyntheticRecoveryParameters,
) -> dict[str, Any]:
    rt60 = {
        f"{center:g}": float(
            abs(fitted.rt60_s_by_hz[center] - truth.rt60_s_by_hz[center])
            / truth.rt60_s_by_hz[center]
        )
        for center in truth.centers_hz
    }
    late_gain = {
        f"{center:g}": float(
            abs(fitted.late_gain_db_by_hz[center] - truth.late_gain_db_by_hz[center])
        )
        for center in truth.centers_hz
    }
    return {
        "mixing_time_absolute_error_ms": float(
            abs(fitted.mixing_time_s - truth.mixing_time_s) * 1000.0
        ),
        "early_reflection_gain_absolute_error_db": float(
            abs(fitted.early_reflection_gain_db - truth.early_reflection_gain_db)
        ),
        "rt60_relative_error_by_hz": rt60,
        "maximum_rt60_relative_error": max(rt60.values()),
        "late_gain_absolute_error_db_by_hz": late_gain,
        "maximum_late_gain_absolute_error_db": max(late_gain.values()),
    }


def _mean_total(reports: Sequence[dict[str, Any]]) -> float:
    return float(np.mean([float(item["total"]) for item in reports]))


def _mean_term(reports: Sequence[dict[str, Any]], term: str) -> float:
    return float(np.mean([float(item["terms"][term]) for item in reports]))


def _measurement_audit(
    observations: Sequence[SyntheticRecoveryObservation],
    measurements: Sequence[PerturbedSyntheticMeasurement],
) -> dict[str, Any]:
    items = []
    for observation, measurement in zip(observations, measurements):
        metadata = measurement.metadata
        items.append(
            {
                "observation_id": observation.observation_id,
                **measurement.to_dict(),
                "raw_direct_error_samples": int(
                    metadata["windowed_detected_raw_direct_sample"]
                    - metadata["expected_raw_direct_sample"]
                ),
                "corrected_direct_error_samples": int(
                    metadata["windowed_detected_corrected_direct_sample"]
                    - metadata["expected_corrected_direct_sample"]
                ),
                "global_peak_is_not_direct_diagnostic": bool(
                    metadata["global_raw_peak_sample_diagnostic"]
                    != metadata["expected_raw_direct_sample"]
                ),
            }
        )
    return {
        "items": items,
        "all_windowed_raw_arrivals_detected": all(
            item["raw_direct_error_samples"] == 0 for item in items
        ),
        "all_known_nuisance_corrections_restore_arrival": all(
            item["corrected_direct_error_samples"] == 0 for item in items
        ),
        "global_peak_failure_count": sum(
            item["global_peak_is_not_direct_diagnostic"] for item in items
        ),
    }


def build_report(
    *,
    sample_rate: int = 16000,
    duration_s: float = 0.512,
    maximum_evaluations: int = 160,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    sample_count = int(round(sample_rate * duration_s))
    truth = _truth()
    train = _observations(
        "train",
        (1.1, 2.0, 3.2),
        (10, 11, 12),
        sample_rate,
        sample_count,
    )
    holdout = _observations(
        "position-holdout",
        (1.55, 2.6),
        (98, 99),
        sample_rate,
        sample_count,
    )
    train_perturbations = (
        SyntheticMeasurementPerturbation(35.0, 1.2, 7, 0.08, 0.10, 100),
        SyntheticMeasurementPerturbation(32.0, -0.8, -4, 0.10, 0.12, 101),
        SyntheticMeasurementPerturbation(38.0, 0.5, 11, 0.06, 0.08, 102),
    )
    holdout_perturbations = (
        SyntheticMeasurementPerturbation(33.0, -1.1, 5, 0.10, 0.12, 200),
        SyntheticMeasurementPerturbation(36.0, 0.7, -6, 0.08, 0.10, 201),
    )
    train_measurements = _measurements(train, truth, train_perturbations)
    holdout_measurements = _measurements(holdout, truth, holdout_perturbations)
    train_targets = tuple(item.corrected_rir for item in train_measurements)
    holdout_targets = tuple(item.corrected_rir for item in holdout_measurements)
    objective = SyntheticRecoveryObjectiveConfig(
        mode="m4_multiterm_v2",
        waveform_weight=1.0,
        early_waveform_weight=1.0,
        broadband_decay_weight=0.25,
        octave_decay_weight=0.25,
        noise_margin_db=20.0,
    )
    initializations = _initializations()
    robust_fits = tuple(
        fit_synthetic_recovery_parameters(
            train,
            train_targets,
            initial,
            objective=objective,
            maximum_evaluations=maximum_evaluations,
        )
        for initial in initializations
    )
    best_index = int(np.argmin([fit.final_cost for fit in robust_fits]))
    best = robust_fits[best_index]
    waveform_fit = fit_synthetic_recovery_parameters(
        train,
        train_targets,
        initializations[0],
        maximum_evaluations=maximum_evaluations,
    )
    initial_holdout = evaluate_synthetic_recovery(
        holdout,
        holdout_targets,
        initializations[0],
    )
    waveform_holdout = evaluate_synthetic_recovery(
        holdout,
        holdout_targets,
        waveform_fit.parameters,
    )
    robust_holdout = evaluate_synthetic_recovery(
        holdout,
        holdout_targets,
        best.parameters,
    )
    robust_train = evaluate_synthetic_recovery(
        train,
        train_targets,
        best.parameters,
    )
    robust_errors = _parameter_errors(truth, best.parameters)
    waveform_errors = _parameter_errors(truth, waveform_fit.parameters)
    fitted_vectors = np.vstack([fit.parameters.to_vector() for fit in robust_fits])
    maximum_spread = float(np.max(np.ptp(fitted_vectors, axis=0)))
    train_audit = _measurement_audit(train, train_measurements)
    holdout_audit = _measurement_audit(holdout, holdout_measurements)
    initial_holdout_total = _mean_total(initial_holdout)
    robust_holdout_total = _mean_total(robust_holdout)
    waveform_octave = _mean_term(waveform_holdout, "octave_acoustics")
    robust_octave = _mean_term(robust_holdout, "octave_acoustics")
    checks = {
        "all_known_gain_latency_corrections_restore_arrival": bool(
            train_audit["all_known_nuisance_corrections_restore_arrival"]
            and holdout_audit["all_known_nuisance_corrections_restore_arrival"]
        ),
        "noise_and_model_mismatch_are_retained": all(
            np.linalg.norm(item.corrected_rir - item.clean_rir) > 0.0
            for item in (*train_measurements, *holdout_measurements)
        ),
        "all_robust_multistarts_converged": all(fit.success for fit in robust_fits),
        "all_robust_multistarts_locally_full_rank": all(
            fit.locally_full_rank for fit in robust_fits
        ),
        "scaled_jacobian_condition_below_100": (
            best.scaled_jacobian_condition_number < 100.0
        ),
        "mixing_time_recovered_within_0_25_ms": (
            robust_errors["mixing_time_absolute_error_ms"] <= 0.25
        ),
        "early_gain_recovered_within_0_25_db": (
            robust_errors["early_reflection_gain_absolute_error_db"] <= 0.25
        ),
        "all_rt60_recovered_within_3_percent": (
            robust_errors["maximum_rt60_relative_error"] <= 0.03
        ),
        "all_late_gains_recovered_within_0_25_db": (
            robust_errors["maximum_late_gain_absolute_error_db"] <= 0.25
        ),
        "robust_multistart_parameter_spread_below_1e_5": (maximum_spread <= 1e-5),
        "heldout_m5_1_oracle_total_reduced_50_percent": (
            robust_holdout_total <= 0.5 * initial_holdout_total
        ),
        "robust_octave_error_below_half_waveform_ablation": (
            robust_octave <= 0.5 * waveform_octave
        ),
        "robust_rt60_error_below_waveform_ablation": (
            robust_errors["maximum_rt60_relative_error"]
            < waveform_errors["maximum_rt60_relative_error"]
        ),
        "robust_late_gain_error_below_waveform_ablation": (
            robust_errors["maximum_late_gain_absolute_error_db"]
            < waveform_errors["maximum_late_gain_absolute_error_db"]
        ),
        "all_fitted_outputs_remain_causal": all(
            np.count_nonzero(
                render_synthetic_recovery_rir(item, best.parameters)[
                    : item.direct_sample
                ]
            )
            == 0
            for item in (*train, *holdout)
        ),
    }
    artifacts = {
        "holdout_clean": holdout_measurements[0].clean_rir,
        "holdout_raw_perturbed": holdout_measurements[0].raw_rir,
        "holdout_corrected_target": holdout_measurements[0].corrected_rir,
        "holdout_initial": render_synthetic_recovery_rir(
            holdout[0],
            initializations[0],
        ),
        "holdout_waveform_ablation": render_synthetic_recovery_rir(
            holdout[0],
            waveform_fit.parameters,
        ),
        "holdout_robust_recovered": render_synthetic_recovery_rir(
            holdout[0],
            best.parameters,
        ),
    }
    report = {
        "schema_version": "puresound.m5_robust_recovery_report.v1",
        "milestone": "M5.2b",
        "policy": RIR_ROBUST_RECOVERY_OBJECTIVE_POLICY,
        "scope": "controlled_noise_nuisance_and_model_mismatch",
        "objective": objective.to_dict(),
        "not_claimed": [
            "measured-room fitting",
            "complete M4 renderer inversion",
            "autograd implementation",
            "global identifiability proof",
            "learned residual",
        ],
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "train_position_count": len(train),
        "heldout_position_count": len(holdout),
        "truth": truth.to_dict(),
        "measurement_audit": {
            "train": train_audit,
            "heldout": holdout_audit,
        },
        "robust_multistart_fits": [fit.to_dict() for fit in robust_fits],
        "best_robust_fit_index": best_index,
        "waveform_v1_ablation_fit": waveform_fit.to_dict(),
        "parameter_errors": {
            "robust_multiterm": robust_errors,
            "waveform_v1_ablation": waveform_errors,
            "robust_multistart_max_parameter_spread": maximum_spread,
        },
        "independent_m5_1_oracle": {
            "initial_holdout_mean_total": initial_holdout_total,
            "waveform_ablation_holdout_mean_total": _mean_total(waveform_holdout),
            "robust_holdout_mean_total": robust_holdout_total,
            "robust_holdout_relative_reduction": float(
                1.0
                - robust_holdout_total
                / max(initial_holdout_total, np.finfo(np.float64).tiny)
            ),
            "waveform_ablation_holdout_octave": waveform_octave,
            "robust_holdout_octave": robust_octave,
            "robust_train_reports": robust_train,
            "robust_holdout_reports": robust_holdout,
        },
        "checks": checks,
        "exit": {
            "passed": bool(all(checks.values())),
            "m5_2b_robust_synthetic_recovery_complete": bool(all(checks.values())),
            "measured_inverse_fit_complete": False,
        },
        "next_stage": {
            "milestone": "M5.3",
            "blocked_until_controlled_campaign_ready": True,
            "parallel_model_work": (
                "map the robust objective onto complete M4 renderer parameters"
            ),
        },
    }
    json.dumps(report, allow_nan=False)
    return report, artifacts


def _write_artifacts(
    artifact_dir: Path,
    artifacts: Mapping[str, np.ndarray],
    sample_rate: int,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, audio in artifacts.items():
        path = artifact_dir / f"m5_2b_{name}.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        paths[name] = str(path.relative_to(REPO_ROOT))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=0.512)
    parser.add_argument("--maximum-evaluations", type=int, default=160)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    report, artifacts = build_report(
        sample_rate=args.sample_rate,
        duration_s=args.duration,
        maximum_evaluations=args.maximum_evaluations,
    )
    report["artifacts"] = _write_artifacts(
        args.artifact_dir,
        artifacts,
        args.sample_rate,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    for name, passed in report["checks"].items():
        print(f"check\t{name}\t{'PASS' if passed else 'FAIL'}")
    print(
        "# M5.2b robust synthetic recovery: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# M5.3 measured inverse fit: BLOCKED ON CONTROLLED CAMPAIGN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
