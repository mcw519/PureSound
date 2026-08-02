#!/usr/bin/env python3
"""Run the M5.2 multi-position, multi-start synthetic recovery baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import soundfile as sf

from puresound.audio.rir_inverse_calibration import (
    RIR_SYNTHETIC_RECOVERY_POLICY,
    SyntheticRecoveryObservation,
    SyntheticRecoveryParameters,
    build_synthetic_recovery_observation,
    evaluate_synthetic_recovery,
    fit_synthetic_recovery_parameters,
    render_synthetic_recovery_rir,
)


REPO_ROOT = Path(__file__).resolve().parents[5]
DEFAULT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m5_calibration/reports/m5_synthetic_recovery_report.json"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_synthetic_recovery"
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
            rt60_s_by_hz={500.0: 0.45, 1000.0: 1.10, 2000.0: 0.85, 4000.0: 0.75},
            late_gain_db_by_hz={500.0: 1.0, 1000.0: -9.0, 2000.0: 2.0, 4000.0: -8.0},
        ),
        SyntheticRecoveryParameters(
            mixing_time_s=0.019,
            early_reflection_gain_db=-8.0,
            rt60_s_by_hz={500.0: 1.40, 1000.0: 0.38, 2000.0: 1.20, 4000.0: 0.30},
            late_gain_db_by_hz={500.0: -13.0, 1000.0: 5.0, 2000.0: -11.0, 4000.0: 4.0},
        ),
        SyntheticRecoveryParameters(
            mixing_time_s=0.069,
            early_reflection_gain_db=8.0,
            rt60_s_by_hz={500.0: 1.70, 1000.0: 1.45, 2000.0: 0.32, 4000.0: 1.05},
            late_gain_db_by_hz={500.0: 7.0, 1000.0: -14.0, 2000.0: 6.0, 4000.0: -12.0},
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


def _mean_total(reports: Sequence[dict[str, Any]]) -> float:
    return float(np.mean([float(item["total"]) for item in reports]))


def _parameter_errors(
    truth: SyntheticRecoveryParameters,
    fitted: SyntheticRecoveryParameters,
) -> dict[str, Any]:
    return {
        "mixing_time_absolute_error_ms": float(
            abs(fitted.mixing_time_s - truth.mixing_time_s) * 1000.0
        ),
        "early_reflection_gain_absolute_error_db": float(
            abs(fitted.early_reflection_gain_db - truth.early_reflection_gain_db)
        ),
        "rt60_relative_error_by_hz": {
            f"{center:g}": float(
                abs(fitted.rt60_s_by_hz[center] - truth.rt60_s_by_hz[center])
                / truth.rt60_s_by_hz[center]
            )
            for center in truth.centers_hz
        },
        "late_gain_absolute_error_db_by_hz": {
            f"{center:g}": float(
                abs(
                    fitted.late_gain_db_by_hz[center] - truth.late_gain_db_by_hz[center]
                )
            )
            for center in truth.centers_hz
        },
    }


def build_report(
    *,
    sample_rate: int = 16000,
    duration_s: float = 0.512,
    maximum_evaluations: int = 200,
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
    train_targets = tuple(render_synthetic_recovery_rir(item, truth) for item in train)
    holdout_targets = tuple(
        render_synthetic_recovery_rir(item, truth) for item in holdout
    )
    initializations = _initializations()
    fits = tuple(
        fit_synthetic_recovery_parameters(
            train,
            train_targets,
            initial,
            maximum_evaluations=maximum_evaluations,
        )
        for initial in initializations
    )
    best_index = int(np.argmin([fit.final_cost for fit in fits]))
    best = fits[best_index]
    initial_train_reports = evaluate_synthetic_recovery(
        train,
        train_targets,
        initializations[0],
    )
    fitted_train_reports = evaluate_synthetic_recovery(
        train,
        train_targets,
        best.parameters,
    )
    initial_holdout_reports = evaluate_synthetic_recovery(
        holdout,
        holdout_targets,
        initializations[0],
    )
    fitted_holdout_reports = evaluate_synthetic_recovery(
        holdout,
        holdout_targets,
        best.parameters,
    )
    errors = _parameter_errors(truth, best.parameters)
    fitted_vectors = np.vstack([fit.parameters.to_vector() for fit in fits])
    multi_start_spread = np.ptp(fitted_vectors, axis=0)
    maximum_rt60_error = max(errors["rt60_relative_error_by_hz"].values())
    maximum_late_gain_error = max(errors["late_gain_absolute_error_db_by_hz"].values())
    initial_holdout_total = _mean_total(initial_holdout_reports)
    fitted_holdout_total = _mean_total(fitted_holdout_reports)
    checks = {
        "all_multistarts_converged": all(fit.success for fit in fits),
        "all_multistarts_locally_full_rank": all(fit.locally_full_rank for fit in fits),
        "scaled_jacobian_condition_below_100": (
            best.scaled_jacobian_condition_number < 100.0
        ),
        "mixing_time_recovered_within_0_1_ms": (
            errors["mixing_time_absolute_error_ms"] <= 0.1
        ),
        "early_gain_recovered_within_0_05_db": (
            errors["early_reflection_gain_absolute_error_db"] <= 0.05
        ),
        "all_rt60_recovered_within_0_5_percent": maximum_rt60_error <= 0.005,
        "all_late_gains_recovered_within_0_05_db": (maximum_late_gain_error <= 0.05),
        "multistart_parameter_spread_below_1e_6": bool(
            np.max(multi_start_spread) <= 1e-6
        ),
        "heldout_position_oracle_loss_reduced_99_percent": bool(
            fitted_holdout_total <= 0.01 * initial_holdout_total
        ),
        "all_outputs_causal": all(
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
        "holdout_target": holdout_targets[0],
        "holdout_initial": render_synthetic_recovery_rir(
            holdout[0],
            initializations[0],
        ),
        "holdout_recovered": render_synthetic_recovery_rir(
            holdout[0],
            best.parameters,
        ),
    }
    report = {
        "schema_version": "puresound.m5_synthetic_recovery_report.v1",
        "milestone": "M5.2",
        "policy": RIR_SYNTHETIC_RECOVERY_POLICY,
        "scope": "noise_free_approximate_renderer_identifiability_baseline",
        "not_claimed": [
            "measured-room fitting",
            "full M4 renderer inversion",
            "global identifiability proof",
            "learned residual",
        ],
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "train_position_count": len(train),
        "heldout_position_count": len(holdout),
        "truth": truth.to_dict(),
        "multistart_fits": [fit.to_dict() for fit in fits],
        "best_fit_index": best_index,
        "parameter_errors": errors,
        "multistart_max_parameter_spread": float(np.max(multi_start_spread)),
        "independent_m5_1_oracle": {
            "initial_train_mean_total": _mean_total(initial_train_reports),
            "fitted_train_mean_total": _mean_total(fitted_train_reports),
            "initial_holdout_mean_total": initial_holdout_total,
            "fitted_holdout_mean_total": fitted_holdout_total,
            "holdout_relative_reduction": float(
                1.0
                - fitted_holdout_total
                / max(initial_holdout_total, np.finfo(np.float64).tiny)
            ),
            "fitted_holdout_reports": fitted_holdout_reports,
        },
        "checks": checks,
        "exit": {
            "passed": bool(all(checks.values())),
            "synthetic_recovery_complete": bool(all(checks.values())),
            "measured_inverse_fit_complete": False,
        },
        "next_stage": {
            "milestone": "M5.3",
            "blocked_until_controlled_campaign_ready": True,
            "parallel_model_work": (
                "replace surrogate residual with differentiable M4-consistent terms"
            ),
        },
    }
    json.dumps(report, allow_nan=False)
    return report, artifacts


def _write_artifacts(
    artifact_dir: Path,
    artifacts: dict[str, np.ndarray],
    sample_rate: int,
) -> dict[str, str]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, audio in artifacts.items():
        path = artifact_dir / f"m5_{name}.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        paths[name] = str(path.relative_to(REPO_ROOT))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=0.512)
    parser.add_argument("--maximum-evaluations", type=int, default=200)
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
        f"# M5.2 synthetic recovery: {'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# M5.3 measured inverse fit: BLOCKED ON CONTROLLED CAMPAIGN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
