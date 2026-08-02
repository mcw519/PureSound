#!/usr/bin/env python3
"""Validate M5.2d grouped PathEvent losses and reject mono ambiguities."""

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

from puresound.audio.hybrid_rir import (  # noqa: E402
    HybridRIRConfig,
    sample_material_first_rir_scene,
)
from puresound.audio.rir_calibration import (  # noqa: E402
    CalibrationLossWeights,
    analyze_rir_calibration_loss,
)
from puresound.audio.rir_m5_pipeline import (  # noqa: E402
    M5_GROUPED_PATH_INVERSE_POLICY,
    GroupedPathObservation,
    GroupedPathParameters,
    analyze_local_identifiability,
    fit_grouped_path_gains,
    render_grouped_path_observation,
)
from puresound.audio.rir_path_events import (  # noqa: E402
    generate_scene_shoebox_path_events,
)


DEFAULT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m5_calibration/reports/m5_group_identifiability_report.json"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_group_identifiability"
CENTERS_HZ = (500.0, 1000.0, 2000.0)


def _truth() -> GroupedPathParameters:
    return GroupedPathParameters(
        mixing_time_s=0.024,
        reflection_adjustment_db_by_group={
            "ceiling": -0.8,
            "east": -2.0,
            "floor": -3.0,
            "north": -2.4,
            "south": -1.4,
            "west": -1.0,
        },
        target_rt60_s_by_hz={500.0: 0.65, 1000.0: 0.52, 2000.0: 0.40},
    )


def _initial(value_db: float) -> GroupedPathParameters:
    truth = _truth()
    return replace(
        truth,
        reflection_adjustment_db_by_group={
            name: float(value_db) for name in truth.group_names
        },
    )


def _observations(
    scene,
    source_indices: Sequence[int],
    sample_rate: int,
    sample_count: int,
    max_order: int,
    fdn_seed: int,
) -> tuple[GroupedPathObservation, ...]:
    surface_groups = {
        surface.surface_id: surface.boundary for surface in scene.surfaces
    }
    observations = []
    for source_index in source_indices:
        event_set = generate_scene_shoebox_path_events(
            scene,
            source_index=int(source_index),
            max_order=int(max_order),
            edge_corner_policy="exclude",
            resolve_object_visibility=True,
            include_scene_interactions=False,
        )
        direct = next(
            event for event in event_set.events if event.path_type == "direct"
        )
        observations.append(
            GroupedPathObservation(
                observation_id=f"source-{source_index}",
                event_set=event_set,
                surface_group_by_id=surface_groups,
                sample_rate=int(sample_rate),
                sample_count=int(sample_count),
                direct_sample=int(round(direct.delay_s * sample_rate)),
                fdn_seed=int(fdn_seed + source_index),
                delay_line_count=4,
            )
        )
    return tuple(observations)


def _targets(
    observations: Sequence[GroupedPathObservation],
    truth: GroupedPathParameters,
    *,
    snr_db: float,
    seed: int,
) -> tuple[np.ndarray, ...]:
    values = []
    for index, observation in enumerate(observations):
        clean = render_grouped_path_observation(observation, truth).rir
        rng = np.random.default_rng(int(seed + index))
        noise = rng.standard_normal(clean.size)
        noise *= (
            math.sqrt(float(np.mean(np.square(clean))))
            * 10.0 ** (-float(snr_db) / 20.0)
            / max(
                math.sqrt(float(np.mean(np.square(noise)))),
                np.finfo(np.float64).tiny,
            )
        )
        values.append(clean + noise)
    return tuple(values)


def _oracle(
    targets: Sequence[np.ndarray],
    candidates: Sequence[np.ndarray],
    observations: Sequence[GroupedPathObservation],
) -> dict[str, Any]:
    direct = [observation.direct_sample for observation in observations]
    report = analyze_rir_calibration_loss(
        np.asarray(targets),
        np.asarray(candidates),
        observations[0].sample_rate,
        measured_direct_samples=direct,
        synthetic_direct_samples=direct,
        physical_first_samples=[max(0, value - 1) for value in direct],
        weights=CalibrationLossWeights(spatial_coherence=0.0),
        fft_sizes=(256, 512),
        octave_centers_hz=CENTERS_HZ,
    )
    return report.to_dict()


def _errors(
    truth: GroupedPathParameters,
    fitted: GroupedPathParameters,
) -> dict[str, Any]:
    values = {
        name: float(
            abs(
                fitted.reflection_adjustment_db_by_group[name]
                - truth.reflection_adjustment_db_by_group[name]
            )
        )
        for name in truth.group_names
    }
    return {
        "absolute_error_db_by_group": values,
        "maximum_absolute_error_db": max(values.values()),
    }


def build_report(
    *,
    sample_rate: int = 8000,
    duration_s: float = 0.4,
    max_order: int = 4,
    scene_seed: int = 20260802,
    fdn_seed: int = 120,
    snr_db: float = 48.0,
    maximum_evaluations: int = 60,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = HybridRIRConfig(
        sample_rate=int(sample_rate),
        duration=float(duration_s),
        crossover_hz=1000.0,
        output_mode="calibrated",
        match_crossover_energy=False,
        num_obstacles_range=(0, 0),
    )
    scene = sample_material_first_rir_scene(
        config,
        seed=int(scene_seed),
        room_type="office",
        scene_id="m5-2d-group-identifiability",
    )
    sample_count = config.num_samples
    train_indices = (0, 2, 4)
    holdout_indices = (1, 3)
    train = _observations(
        scene,
        train_indices,
        sample_rate,
        sample_count,
        max_order,
        fdn_seed,
    )
    holdout = _observations(
        scene,
        holdout_indices,
        sample_rate,
        sample_count,
        max_order,
        fdn_seed,
    )
    truth = _truth()
    train_targets = _targets(train, truth, snr_db=snr_db, seed=300)
    holdout_targets = _targets(holdout, truth, snr_db=snr_db, seed=400)
    initializations = (_initial(0.0), _initial(-7.0))
    fits = tuple(
        fit_grouped_path_gains(
            train,
            train_targets,
            initial,
            maximum_evaluations=int(maximum_evaluations),
        )
        for initial in initializations
    )
    best_index = int(np.argmin([fit.final_cost for fit in fits]))
    best = fits[best_index]
    fitted_vectors = np.vstack([fit.parameters.gain_vector() for fit in fits])
    maximum_spread_db = float(np.max(np.ptp(fitted_vectors, axis=0)))
    errors = _errors(truth, best.parameters)

    # In a mono coherent model, absorption loss and specular scattering loss
    # multiply the same path amplitude. Their local sensitivity columns are
    # therefore exactly duplicated; retain effective reflection loss and defer
    # scattering to synchronized spatial evidence in M5.4.
    base_jacobian = best.scaled_jacobian
    split_names = tuple(
        [f"effective_reflection:{name}" for name in truth.group_names]
        + [f"scattering_specular_loss:{name}" for name in truth.group_names]
    )
    absorption_scattering = analyze_local_identifiability(
        np.column_stack((base_jacobian, base_jacobian)),
        split_names,
    )
    initial_holdout = tuple(
        render_grouped_path_observation(observation, initializations[0]).rir
        for observation in holdout
    )
    fitted_holdout = tuple(
        render_grouped_path_observation(observation, best.parameters).rir
        for observation in holdout
    )
    initial_oracle = _oracle(holdout_targets, initial_holdout, holdout)
    fitted_oracle = _oracle(holdout_targets, fitted_holdout, holdout)
    oracle_reduction = float(
        1.0
        - fitted_oracle["total"]
        / max(initial_oracle["total"], np.finfo(np.float64).tiny)
    )
    fitted_renders = tuple(
        render_grouped_path_observation(observation, best.parameters)
        for observation in (*train, *holdout)
    )
    checks = {
        "all_grouped_multistarts_converged": all(fit.success for fit in fits),
        "six_effective_boundary_groups_locally_full_rank": (
            best.identifiability.full_column_rank
        ),
        "six_effective_boundary_groups_all_accepted": (
            len(best.identifiability.accepted_parameters) == 6
            and not best.identifiability.rejected_parameters
        ),
        "effective_group_condition_below_100": (
            best.identifiability.normalized_condition_number < 100.0
        ),
        "all_effective_group_gains_recovered_within_0_25_db": (
            errors["maximum_absolute_error_db"] <= 0.25
        ),
        "grouped_multistart_spread_below_1e_4_db": maximum_spread_db <= 1e-4,
        "absorption_scattering_split_is_rank_deficient": (
            not absorption_scattering.full_column_rank
            and absorption_scattering.numerical_rank == 6
        ),
        "all_six_scattering_duplicates_rejected": (
            set(absorption_scattering.rejected_parameters)
            == {f"scattering_specular_loss:{name}" for name in truth.group_names}
        ),
        "heldout_m5_1_total_reduced_50_percent": oracle_reduction >= 0.50,
        "all_actual_m4_outputs_preserve_pre_transition": all(
            result.metadata["pre_transition_max_abs_error"] == 0.0
            for result in fitted_renders
        ),
        "all_outputs_remain_physical_arrival_causal": all(
            np.count_nonzero(result.rir[: max(0, observation.direct_sample - 1)]) == 0
            for observation, result in zip((*train, *holdout), fitted_renders)
        ),
    }
    artifacts = {
        "holdout_target": holdout_targets[0],
        "holdout_initial": initial_holdout[0],
        "holdout_recovered": fitted_holdout[0],
    }
    report = {
        "schema_version": "puresound.m5_group_identifiability_report.v1",
        "milestone": "M5.2d",
        "policy": M5_GROUPED_PATH_INVERSE_POLICY,
        "scope": "actual_path_event_m4_effective_boundary_identifiability",
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "path_event_max_order": int(max_order),
        "train_source_indices": list(train_indices),
        "holdout_source_indices": list(holdout_indices),
        "target_snr_db": float(snr_db),
        "truth": truth.to_dict(),
        "fits": [fit.to_dict() for fit in fits],
        "best_fit_index": best_index,
        "parameter_errors": errors,
        "maximum_multistart_spread_db": maximum_spread_db,
        "absorption_scattering_split_diagnostic": (absorption_scattering.to_dict()),
        "selection_decision": {
            "accepted_for_m5_3": [
                f"effective_reflection:{name}" for name in truth.group_names
            ],
            "deferred_to_synchronized_m5_4": [
                f"scattering:{name}" for name in truth.group_names
            ],
            "reason": (
                "mono coherent amplitudes identify only the combined effective "
                "reflection loss; absorption and specular scattering columns "
                "are locally identical"
            ),
        },
        "independent_m5_1_oracle": {
            "initial_holdout": initial_oracle,
            "recovered_holdout": fitted_oracle,
            "relative_total_reduction": oracle_reduction,
        },
        "checks": checks,
        "exit": {
            "passed": bool(all(checks.values())),
            "m5_2d_group_identifiability_complete": bool(all(checks.values())),
            "individual_absorption_scattering_separation_claimed": False,
        },
        "next_stage": {
            "milestone": "M5.3/M5.4",
            "measured_fit_requires_ready_controlled_campaign": True,
            "scattering_requires_synchronized_receivers": True,
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
        path = artifact_dir / f"m5_2d_{name}.wav"
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
        "# M5.2d grouped material/path identifiability: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# absorption/scattering mono separation: REJECTED")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
