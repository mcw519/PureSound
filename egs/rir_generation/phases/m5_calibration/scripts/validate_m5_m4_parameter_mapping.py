#!/usr/bin/env python3
"""Validate M5.2c inverse mapping through the actual M4 renderer coupling."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egs.rir_generation.generate_hybrid_rir import _make_high_backend
from puresound.audio.hybrid_rir import (
    HybridRIRConfig,
    PathEventHighFrequencyBackend,
    PyroomacousticsHighFrequencyBackend,
    sample_material_first_rir_scene,
)
from puresound.audio.rir_calibration import (
    CalibrationLossWeights,
    analyze_rir_calibration_loss,
)
from puresound.audio.rir_late_coupling import PATH_EVENT_FDN_COUPLING_POLICY
from puresound.audio.rir_m4_inverse_calibration import (
    M4_PARAMETER_PROFILE_INVERSE_POLICY,
    M4InverseObservation,
    M4InverseParameters,
    M4ProfileObjectiveConfig,
    fit_m4_parameter_profile,
    render_m4_inverse_observation,
)


DEFAULT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m5_calibration/reports/m5_m4_parameter_mapping_report.json"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_m4_parameter_mapping"
CENTERS_HZ = (500.0, 1000.0, 2000.0)


def _truth() -> M4InverseParameters:
    return M4InverseParameters(
        mixing_time_s=0.024,
        coherent_reflection_gain_db=-1.5,
        target_rt60_s_by_hz={500.0: 0.68, 1000.0: 0.54, 2000.0: 0.42},
    )


def _initial() -> M4InverseParameters:
    return M4InverseParameters(
        mixing_time_s=0.028,
        coherent_reflection_gain_db=4.0,
        target_rt60_s_by_hz={500.0: 0.35, 1000.0: 0.95, 2000.0: 0.80},
    )


def _perturbed_target(
    observation: M4InverseObservation,
    truth: M4InverseParameters,
    *,
    alternate_seed: int,
    noise_seed: int,
    nominal_fraction: float = 0.92,
    snr_db: float = 42.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Retain deterministic topology mismatch and broadband measurement noise."""

    nominal = render_m4_inverse_observation(observation, truth).rir
    alternate_observation = replace(observation, fdn_seed=int(alternate_seed))
    alternate = render_m4_inverse_observation(alternate_observation, truth).rir
    mismatched = nominal_fraction * nominal + (1.0 - nominal_fraction) * alternate
    rng = np.random.default_rng(int(noise_seed))
    noise = rng.standard_normal(mismatched.size)
    signal_rms = float(np.sqrt(np.mean(np.square(mismatched))))
    noise *= (
        signal_rms
        * 10.0 ** (-float(snr_db) / 20.0)
        / max(
            float(np.sqrt(np.mean(np.square(noise)))),
            np.finfo(np.float64).tiny,
        )
    )
    target = mismatched + noise
    return target, {
        "nominal_fraction": float(nominal_fraction),
        "alternate_fdn_seed": int(alternate_seed),
        "noise_seed": int(noise_seed),
        "snr_db": float(snr_db),
        "mismatch_relative_norm": float(
            np.linalg.norm(mismatched - nominal)
            / max(np.linalg.norm(nominal), np.finfo(np.float64).tiny)
        ),
        "noise_relative_norm": float(
            np.linalg.norm(noise)
            / max(np.linalg.norm(mismatched), np.finfo(np.float64).tiny)
        ),
    }


def _observations(
    coherent_rirs: np.ndarray,
    distances_m: Sequence[float],
    sample_rate: int,
    sound_speed_m_s: float,
    source_indices: Sequence[int],
    *,
    seed_offset: int,
    delay_line_count: int,
) -> tuple[M4InverseObservation, ...]:
    return tuple(
        M4InverseObservation(
            observation_id=f"source-{source_index}",
            path_event_rir=coherent_rirs[source_index],
            sample_rate=int(sample_rate),
            direct_sample=int(
                round(distances_m[source_index] / float(sound_speed_m_s) * sample_rate)
            ),
            fdn_seed=int(seed_offset + source_index),
            delay_line_count=int(delay_line_count),
        )
        for source_index in source_indices
    )


def _oracle(
    targets: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
    observations: Sequence[M4InverseObservation],
    distances_m: Sequence[float],
    sound_speed_m_s: float,
) -> dict[str, Any]:
    direct = [observation.direct_sample for observation in observations]
    physical = [
        int(math.floor(distance / float(sound_speed_m_s) * observation.sample_rate))
        for distance, observation in zip(distances_m, observations)
    ]
    report = analyze_rir_calibration_loss(
        np.asarray(targets),
        np.asarray(candidates),
        observations[0].sample_rate,
        measured_direct_samples=direct,
        synthetic_direct_samples=direct,
        physical_first_samples=physical,
        weights=CalibrationLossWeights(spatial_coherence=0.0),
        fft_sizes=(256, 512),
        octave_centers_hz=CENTERS_HZ,
    )
    return report.to_dict()


def _parameter_errors(
    truth: M4InverseParameters,
    fitted: M4InverseParameters,
) -> dict[str, Any]:
    rt60 = {
        f"{center:g}": float(
            abs(fitted.target_rt60_s_by_hz[center] - truth.target_rt60_s_by_hz[center])
            / truth.target_rt60_s_by_hz[center]
        )
        for center in truth.centers_hz
    }
    return {
        "mixing_time_absolute_error_ms": float(
            abs(fitted.mixing_time_s - truth.mixing_time_s) * 1000.0
        ),
        "coherent_reflection_gain_absolute_error_db": float(
            abs(fitted.coherent_reflection_gain_db - truth.coherent_reflection_gain_db)
        ),
        "rt60_relative_error_by_hz": rt60,
        "maximum_rt60_relative_error": max(rt60.values()),
    }


def build_report(
    *,
    sample_rate: int = 8000,
    duration_s: float = 0.4,
    max_order: int = 4,
    scene_seed: int = 20260801,
    fdn_seed: int = 70,
    delay_line_count: int = 4,
    maximum_evaluations: int = 60,
    objective: M4ProfileObjectiveConfig | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = HybridRIRConfig(
        sample_rate=int(sample_rate),
        duration=float(duration_s),
        crossover_hz=1000.0,
        output_mode="calibrated",
        match_crossover_energy=False,
        num_obstacles_range=(1, 1),
    )
    scene = sample_material_first_rir_scene(
        config,
        seed=int(scene_seed),
        room_type="office",
        scene_id="m5-2c-m4-parameter-mapping",
    )
    config = replace(config, sound_speed=float(scene.environment.sound_speed_m_s))
    coherent_backend = PathEventHighFrequencyBackend(max_order=int(max_order))
    coherent = coherent_backend.simulate(scene, config).astype(np.float64)
    distances = scene.source_distances()
    train_indices = (0, 2)
    holdout_indices = (1, 3)
    train = _observations(
        coherent,
        distances,
        sample_rate,
        config.sound_speed,
        train_indices,
        seed_offset=fdn_seed,
        delay_line_count=delay_line_count,
    )
    holdout = _observations(
        coherent,
        distances,
        sample_rate,
        config.sound_speed,
        holdout_indices,
        seed_offset=fdn_seed,
        delay_line_count=delay_line_count,
    )
    truth = _truth()
    initial = _initial()
    train_pairs = tuple(
        _perturbed_target(
            observation,
            truth,
            alternate_seed=fdn_seed + 1000 + index,
            noise_seed=fdn_seed + 2000 + index,
        )
        for index, observation in enumerate(train)
    )
    holdout_pairs = tuple(
        _perturbed_target(
            observation,
            truth,
            alternate_seed=fdn_seed + 1100 + index,
            noise_seed=fdn_seed + 2100 + index,
        )
        for index, observation in enumerate(holdout)
    )
    train_targets = tuple(pair[0] for pair in train_pairs)
    holdout_targets = tuple(pair[0] for pair in holdout_pairs)
    active_objective = objective or M4ProfileObjectiveConfig()
    fit = fit_m4_parameter_profile(
        train,
        train_targets,
        initial,
        (0.020, 0.024, 0.028),
        objective=active_objective,
        maximum_evaluations=int(maximum_evaluations),
    )
    fitted = fit.best.parameters
    initial_holdout = tuple(
        render_m4_inverse_observation(observation, initial).rir
        for observation in holdout
    )
    fitted_holdout = tuple(
        render_m4_inverse_observation(observation, fitted).rir
        for observation in holdout
    )
    initial_oracle = _oracle(
        holdout_targets,
        initial_holdout,
        holdout,
        [distances[index] for index in holdout_indices],
        config.sound_speed,
    )
    fitted_oracle = _oracle(
        holdout_targets,
        fitted_holdout,
        holdout,
        [distances[index] for index in holdout_indices],
        config.sound_speed,
    )
    errors = _parameter_errors(truth, fitted)
    fit_dict = fit.to_dict()
    profile_delays = {
        tuple(
            delay
            for values in point.delay_lengths_by_observation.values()
            for delay in values
        )
        for point in fit.profiles
    }
    fitted_results = tuple(
        render_m4_inverse_observation(observation, fitted)
        for observation in (*train, *holdout)
    )
    fitted_distances = [
        *(distances[index] for index in train_indices),
        *(distances[index] for index in holdout_indices),
    ]
    physical_first = [
        int(math.floor(distance / config.sound_speed * sample_rate))
        for distance in fitted_distances
    ]
    default_backend = _make_high_backend(
        {
            "high_backend": "pyroomacoustics",
            "pra_max_order": 4,
            "pra_n_rays": 100,
        }
    )
    best_to_second = float(fit_dict["best_to_second_cost_ratio"])
    oracle_reduction = float(
        1.0
        - fitted_oracle["total"]
        / max(initial_oracle["total"], np.finfo(np.float64).tiny)
    )
    checks = {
        "actual_path_event_backend_used": isinstance(
            coherent_backend, PathEventHighFrequencyBackend
        ),
        "actual_m4_coupling_policy_used": all(
            result.metadata["policy"] == PATH_EVENT_FDN_COUPLING_POLICY
            for result in fitted_results
        ),
        "m4_pre_transition_samples_exactly_preserved": all(
            result.metadata["pre_transition_max_abs_error"] == 0.0
            for result in fitted_results
        ),
        "true_mixing_profile_selected": fitted.mixing_time_s == truth.mixing_time_s,
        "best_profile_cost_below_one_fifth_second": best_to_second <= 0.20,
        "all_profile_inner_solves_converged": all(
            point.success for point in fit.profiles
        ),
        "all_profile_jacobians_locally_full_rank": all(
            point.locally_full_rank for point in fit.profiles
        ),
        "best_scaled_jacobian_condition_below_100": (
            fit.best.scaled_jacobian_condition_number < 100.0
        ),
        "coherent_gain_recovered_within_0_75_db": (
            errors["coherent_reflection_gain_absolute_error_db"] <= 0.75
        ),
        "all_rt60_recovered_within_5_percent": (
            errors["maximum_rt60_relative_error"] <= 0.05
        ),
        "heldout_m5_1_oracle_total_reduced_50_percent": oracle_reduction >= 0.50,
        "mixing_candidates_produce_distinct_delay_topologies": (
            len(profile_delays) == len(fit.profiles)
        ),
        "all_fitted_outputs_are_physical_arrival_causal": all(
            np.count_nonzero(result.rir[: physical_first[index]]) == 0
            for index, result in enumerate(fitted_results)
        ),
        "production_default_remains_pyroomacoustics": isinstance(
            default_backend, PyroomacousticsHighFrequencyBackend
        ),
    }
    artifacts = {
        "holdout_target": holdout_targets[0],
        "holdout_initial": initial_holdout[0],
        "holdout_recovered": fitted_holdout[0],
    }
    report = {
        "schema_version": "puresound.m5_m4_parameter_mapping_report.v1",
        "milestone": "M5.2c",
        "policy": M4_PARAMETER_PROFILE_INVERSE_POLICY,
        "scope": "actual_path_event_plus_multiband_fdn_parameter_profile",
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "scene": {
            "scene_id": scene.scene_id,
            "room_type": scene.room_type,
            "room_dimensions_m": list(scene.dimensions_m),
            "path_event_max_order": int(max_order),
            "train_source_indices": list(train_indices),
            "holdout_source_indices": list(holdout_indices),
        },
        "model_mapping": {
            "discrete_outer_parameter": "mixing_time_s -> prime FDN delay topology",
            "continuous_inner_parameters": [
                "coherent_reflection_gain_db proxy",
                "octave target_rt60_s_by_hz",
            ],
            "coupling_policy": PATH_EVENT_FDN_COUPLING_POLICY,
            "production_default_changed": False,
        },
        "objective": active_objective.to_dict(),
        "truth": truth.to_dict(),
        "fit": fit_dict,
        "parameter_errors": errors,
        "perturbations": {
            "train": [pair[1] for pair in train_pairs],
            "holdout": [pair[1] for pair in holdout_pairs],
        },
        "independent_m5_1_oracle": {
            "spatial_weight_disabled_reason": (
                "channels represent different source positions, not synchronized receivers"
            ),
            "initial_holdout": initial_oracle,
            "recovered_holdout": fitted_oracle,
            "relative_total_reduction": oracle_reduction,
        },
        "checks": checks,
        "exit": {
            "passed": bool(all(checks.values())),
            "m5_2c_actual_m4_parameter_mapping_complete": bool(all(checks.values())),
            "measured_inverse_fit_complete": False,
        },
        "not_claimed": [
            "measured-room fitting",
            "individual surface absorption or scattering identification",
            "continuous differentiability across FDN delay-topology changes",
            "global identifiability outside the supplied mixing-time grid",
            "production-default replacement",
        ],
        "next_stage": {
            "milestone": "M5.2d/M5.3",
            "parallel_model_work": (
                "separate material/path groups and test their identifiability"
            ),
            "measured_fit_blocked_until_controlled_campaign_ready": True,
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
        path = artifact_dir / f"m5_2c_{name}.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        paths[name] = str(path.relative_to(REPO_ROOT))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=0.4)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--maximum-evaluations", type=int, default=60)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    report, artifacts = build_report(
        sample_rate=args.sample_rate,
        duration_s=args.duration,
        max_order=args.max_order,
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
        "# M5.2c actual M4 parameter mapping: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# M5.3 measured inverse fit: BLOCKED ON CONTROLLED CAMPAIGN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
