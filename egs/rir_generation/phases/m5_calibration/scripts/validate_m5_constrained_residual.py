#!/usr/bin/env python3
"""Validate M5.5 constrained residual and all required ablations."""

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
    PathEventHighFrequencyBackend,
    sample_material_first_rir_scene,
)
from puresound.audio.rir_constrained_residual import (  # noqa: E402
    M5_CONSTRAINED_RESIDUAL_POLICY,
    evaluate_residual_ablation,
    fit_causal_decay_residual,
)
from puresound.audio.rir_m4_inverse_calibration import (  # noqa: E402
    M4InverseObservation,
    M4InverseParameters,
    render_m4_inverse_observation,
)


DEFAULT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m5_calibration/reports/m5_constrained_residual_report.json"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_constrained_residual"


def _physical_room(
    room_index: int,
    source_indices: Sequence[int],
    sample_rate: int,
    duration_s: float,
) -> tuple[tuple[np.ndarray, ...], tuple[int, ...], tuple[int, ...]]:
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
        seed=20260810 + int(room_index),
        room_type=("office", "meeting_room", "living_room")[room_index],
        scene_id=f"m5-5-room-{room_index}",
    )
    config = replace(config, sound_speed=scene.environment.sound_speed_m_s)
    coherent = PathEventHighFrequencyBackend(max_order=4).simulate(
        scene,
        config,
    )
    truth = M4InverseParameters(
        mixing_time_s=0.024,
        coherent_reflection_gain_db=-1.5,
        target_rt60_s_by_hz={500.0: 0.66, 1000.0: 0.52, 2000.0: 0.40},
    )
    rirs = []
    direct_samples = []
    physical_first = []
    distances = scene.source_distances()
    for source_index in source_indices:
        direct = int(round(distances[source_index] / config.sound_speed * sample_rate))
        observation = M4InverseObservation(
            observation_id=f"room-{room_index}:source-{source_index}",
            path_event_rir=coherent[source_index],
            sample_rate=int(sample_rate),
            direct_sample=direct,
            fdn_seed=500 + 10 * room_index + source_index,
            delay_line_count=4,
        )
        rirs.append(render_m4_inverse_observation(observation, truth).rir)
        direct_samples.append(direct)
        physical_first.append(
            int(math.floor(distances[source_index] / config.sound_speed * sample_rate))
        )
    return tuple(rirs), tuple(direct_samples), tuple(physical_first)


def _systematic_template(sample_count: int, sample_rate: int) -> np.ndarray:
    local_time = np.arange(sample_count, dtype=np.float64) / sample_rate
    template = np.zeros(sample_count, dtype=np.float64)
    start = int(round(0.006 * sample_rate))
    time = local_time[: sample_count - start]
    template[start:] = (
        0.7 * np.sin(2.0 * np.pi * 730.0 * time + 0.2)
        + 0.4 * np.sin(2.0 * np.pi * 1470.0 * time + 0.7)
    ) * np.exp(-6.91 * time / 0.42)
    for delay_ms, amplitude in ((11.0, 1.0), (19.0, -0.7), (31.0, 0.45)):
        index = int(round(delay_ms * 1e-3 * sample_rate))
        template[index] += amplitude
    template /= max(np.linalg.norm(template), np.finfo(np.float64).tiny)
    template *= math.sqrt(0.08)
    return template


def _targets(
    physical: Sequence[np.ndarray],
    direct_samples: Sequence[int],
    sample_rate: int,
    *,
    noise_seed: int,
    snr_db: float,
) -> tuple[np.ndarray, ...]:
    result = []
    for index, (rir, direct) in enumerate(zip(physical, direct_samples)):
        tail_length = rir.size - direct
        normalized = _systematic_template(tail_length, sample_rate)
        residual = np.zeros_like(rir)
        residual[direct:] = np.linalg.norm(rir[direct:]) * normalized
        target = rir + residual
        rng = np.random.default_rng(noise_seed + index)
        noise = rng.standard_normal(rir.size)
        noise *= (
            math.sqrt(float(np.mean(np.square(target))))
            * 10.0 ** (-float(snr_db) / 20.0)
            / max(
                math.sqrt(float(np.mean(np.square(noise)))),
                np.finfo(np.float64).tiny,
            )
        )
        result.append(target + noise)
    return tuple(result)


def _decay_constraint_holds(model) -> bool:
    values = model.normalized_template
    block = model.decay_block_samples
    previous = None
    factor = 10.0 ** (-3.0 * (block / model.sample_rate) / model.maximum_rt60_s)
    for start in range(model.decay_start_sample, values.size, block):
        current = math.sqrt(float(np.mean(np.square(values[start : start + block]))))
        if previous is not None and current > previous * factor + 1e-12:
            return False
        previous = current
    return True


def build_report(
    *,
    sample_rate: int = 8000,
    duration_s: float = 0.45,
    target_snr_db: float = 52.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    train_room_ids = ("synthetic-room-0", "synthetic-room-1")
    heldout_room_ids = ("synthetic-room-2",)
    train_physical = []
    train_direct = []
    train_first = []
    for room_index in (0, 1):
        physical, direct, first = _physical_room(
            room_index,
            (0, 2),
            sample_rate,
            duration_s,
        )
        train_physical.extend(physical)
        train_direct.extend(direct)
        train_first.extend(first)
    heldout_physical, heldout_direct, heldout_first = _physical_room(
        2,
        (1, 3),
        sample_rate,
        duration_s,
    )
    train_targets = _targets(
        train_physical,
        train_direct,
        sample_rate,
        noise_seed=700,
        snr_db=target_snr_db,
    )
    heldout_targets = _targets(
        heldout_physical,
        heldout_direct,
        sample_rate,
        noise_seed=800,
        snr_db=target_snr_db,
    )
    fit = fit_causal_decay_residual(
        train_targets,
        train_physical,
        train_direct,
        sample_rate,
        decay_start_ms=50.0,
        decay_block_ms=10.0,
        maximum_rt60_s=0.8,
        maximum_residual_to_physical_energy_ratio=0.15,
    )
    ablation = evaluate_residual_ablation(
        heldout_targets,
        heldout_physical,
        heldout_direct,
        sample_rate,
        fit.model,
        physical_first_samples=heldout_first,
    )
    residual_rirs = ablation.pop("residual_rirs")
    combined_rirs = ablation.pop("combined_rirs")
    physical_total = float(ablation["physical_only"]["total"])
    residual_total = float(ablation["residual_only"]["total"])
    combined_total = float(ablation["combined"]["total"])
    reduction = float(
        1.0 - combined_total / max(physical_total, np.finfo(np.float64).tiny)
    )
    interpolated_physical = 0.5 * heldout_physical[0] + 0.5 * heldout_physical[1]
    interpolation_direct = min(heldout_direct)
    interpolated_combined = fit.model.apply(
        interpolated_physical,
        interpolation_direct,
    )
    checks = {
        "training_and_heldout_rooms_are_disjoint": not (
            set(train_room_ids) & set(heldout_room_ids)
        ),
        "residual_model_uses_multiple_train_rooms": (
            fit.model.training_observation_count == 4
        ),
        "heldout_combined_total_reduced_60_percent": reduction >= 0.60,
        "combined_beats_physical_only": combined_total < physical_total,
        "combined_beats_residual_only": combined_total < residual_total,
        "physical_residual_combined_ablation_all_reported": set(ablation)
        >= {"physical_only", "residual_only", "combined"},
        "residual_template_energy_within_budget": (
            fit.constrained_normalized_template_energy
            <= fit.model.maximum_residual_to_physical_energy_ratio + 1e-12
        ),
        "residual_template_obeys_decay_constraint": _decay_constraint_holds(fit.model),
        "all_residual_outputs_are_direct_relative_causal": all(
            np.count_nonzero(residual[:direct]) == 0
            for residual, direct in zip(residual_rirs, heldout_direct)
        ),
        "all_combined_outputs_are_physical_arrival_causal": all(
            np.count_nonzero(combined[:first]) == 0
            for combined, first in zip(combined_rirs, heldout_first)
        ),
        "interpolated_output_is_finite_and_causal": bool(
            np.all(np.isfinite(interpolated_combined))
            and np.count_nonzero(interpolated_combined[: min(heldout_first)]) == 0
        ),
        "noise_and_unmodelled_residual_retained_in_targets": all(
            not np.array_equal(target, physical)
            for target, physical in zip(heldout_targets, heldout_physical)
        ),
    }
    artifacts = {
        "heldout_target": heldout_targets[0],
        "heldout_physical_only": heldout_physical[0],
        "heldout_residual_only": residual_rirs[0],
        "heldout_combined": combined_rirs[0],
        "interpolated_combined": interpolated_combined,
    }
    report = {
        "schema_version": "puresound.m5_constrained_residual_report.v1",
        "milestone": "M5.5",
        "policy": M5_CONSTRAINED_RESIDUAL_POLICY,
        "scope": "room_disjoint_synthetic_systematic_residual_reference",
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "target_snr_db": float(target_snr_db),
        "room_split": {
            "train_room_ids": list(train_room_ids),
            "heldout_room_ids": list(heldout_room_ids),
            "room_disjoint": True,
        },
        "fit": fit.to_dict(),
        "heldout_ablation": ablation,
        "heldout_total_summary": {
            "physical_only": physical_total,
            "residual_only": residual_total,
            "combined": combined_total,
            "combined_relative_reduction_from_physical": reduction,
        },
        "checks": checks,
        "exit": {
            "passed": bool(all(checks.values())),
            "m5_5_constrained_residual_implementation_complete": bool(
                all(checks.values())
            ),
            "measured_residual_training_complete": False,
        },
        "not_claimed": [
            "training on controlled measured rooms",
            "neural residual architecture selection",
            "downstream speech-task improvement",
            "production enablement",
        ],
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
        path = artifact_dir / f"m5_5_{name}.wav"
        sf.write(path, audio, sample_rate, subtype="FLOAT")
        paths[name] = str(path.relative_to(REPO_ROOT))
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=8000)
    parser.add_argument("--duration", type=float, default=0.45)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()
    report, artifacts = build_report(
        sample_rate=args.sample_rate,
        duration_s=args.duration,
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
        "# M5.5 constrained learned residual: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# measured/downstream residual exit: OPEN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
