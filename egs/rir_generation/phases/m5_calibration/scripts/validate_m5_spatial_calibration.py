#!/usr/bin/env python3
"""Validate M5.4 synchronized scattering/directivity candidate profiling."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.sampling import sample_material_first_rir_scene
from puresound.audio.rir.calibration.inverse_m5 import (
    M5_SPATIAL_CANDIDATE_PROFILE_POLICY,
    select_spatial_calibration_candidate,
)
from puresound.audio.rir.scene.schema import MaterialSpectrum
from puresound.audio.rir.render.spatial import (
    SPATIAL_ROOM_RIR_POLICY,
    render_room_scene_spatial_rir,
)


DEFAULT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m5_calibration/reports/m5_spatial_calibration_report.json"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m5/rir_m5_spatial_calibration"


def _array_scene(scene, *, directivity: str, scattering_scale: float):
    original = scene.receivers[0]
    position = np.asarray(original.pose.position_m, dtype=np.float64)
    left = replace(
        original,
        transducer_id="receiver-left",
        directivity_id=directivity,
        pose=replace(
            original.pose,
            position_m=(position + [-0.06, 0.0, 0.0]).tolist(),
            orientation_ypr_deg=[0.0, 0.0, 0.0],
        ),
        array_id="m5-synchronized-pair",
        channel_index=0,
    )
    right = replace(
        original,
        transducer_id="receiver-right",
        directivity_id=directivity,
        pose=replace(
            original.pose,
            position_m=(position + [0.06, 0.0, 0.0]).tolist(),
            orientation_ypr_deg=[180.0, 0.0, 0.0],
        ),
        array_id="m5-synchronized-pair",
        channel_index=1,
    )
    materials = {}
    for material_id, material in scene.materials.items():
        scattering = material.scattering
        materials[material_id] = replace(
            material,
            scattering=MaterialSpectrum(
                center_frequencies_hz=list(scattering.center_frequencies_hz),
                values=[
                    float(np.clip(scattering_scale * value, 0.0, 1.0))
                    for value in scattering.values
                ],
                uncertainty_std=[
                    float(scattering_scale * value)
                    for value in scattering.uncertainty_std
                ],
            ),
        )
    return replace(scene, receivers=[left, right], materials=materials)


def _render(scene, sample_rate: int, duration_s: float, seed: int):
    return render_room_scene_spatial_rir(
        scene,
        sample_rate=int(sample_rate),
        duration_s=float(duration_s),
        source_index=2,
        max_order=2,
        mixing_time_s=0.024,
        transition_duration_s=0.016,
        delay_line_count=4,
        plane_wave_count=24,
        seed=int(seed),
    )


def build_report(
    *,
    sample_rate: int = 8000,
    duration_s: float = 0.45,
    scene_seed: int = 20260803,
    render_seed: int = 20260803,
    target_snr_db: float = 50.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    config = HybridRIRConfig(
        sample_rate=int(sample_rate),
        duration=float(duration_s),
        room_dim_range=((5.2, 5.2), (4.3, 4.3), (2.9, 2.9)),
        num_obstacles_range=(0, 0),
    )
    base_scene = sample_material_first_rir_scene(
        config,
        seed=int(scene_seed),
        room_type="office",
        scene_id="m5-4-spatial-calibration",
    )
    scenes = {
        "truth_scattering_cardioid": _array_scene(
            base_scene,
            directivity="cardioid",
            scattering_scale=1.0,
        ),
        "zero_scattering_cardioid": _array_scene(
            base_scene,
            directivity="cardioid",
            scattering_scale=0.0,
        ),
        "truth_scattering_omni": _array_scene(
            base_scene,
            directivity="omnidirectional",
            scattering_scale=1.0,
        ),
        "zero_scattering_omni": _array_scene(
            base_scene,
            directivity="omnidirectional",
            scattering_scale=0.0,
        ),
    }
    renders = {
        candidate_id: _render(scene, sample_rate, duration_s, render_seed)
        for candidate_id, scene in scenes.items()
    }
    truth_id = "truth_scattering_cardioid"
    clean_target = renders[truth_id].receiver_rirs
    rng = np.random.default_rng(20260804)
    noise = rng.standard_normal(clean_target.shape)
    noise *= (
        np.sqrt(np.mean(np.square(clean_target), axis=1, keepdims=True))
        * 10.0 ** (-float(target_snr_db) / 20.0)
        / np.maximum(
            np.sqrt(np.mean(np.square(noise), axis=1, keepdims=True)),
            np.finfo(np.float64).tiny,
        )
    )
    target = clean_target + noise
    direct = renders[truth_id].metadata["direct_samples"]
    physical = [max(0, int(value) - 1) for value in direct]
    selection = select_spatial_calibration_candidate(
        target,
        {
            candidate_id: render.receiver_rirs
            for candidate_id, render in renders.items()
        },
        sample_rate,
        direct,
        synthetic_direct_samples_by_candidate={
            candidate_id: render.metadata["direct_samples"]
            for candidate_id, render in renders.items()
        },
        physical_first_samples=physical,
    )
    selection_dict = selection.to_dict()
    best_report = selection.candidate_reports[selection.best_candidate_id]
    no_scattering_report = selection.candidate_reports["zero_scattering_cardioid"]
    omni_report = selection.candidate_reports["truth_scattering_omni"]
    mono_rejected = False
    try:
        select_spatial_calibration_candidate(
            target[:1],
            {truth_id: renders[truth_id].receiver_rirs[:1]},
            sample_rate,
            direct[:1],
        )
    except ValueError as exc:
        mono_rejected = "synchronized receivers" in str(exc)
    checks = {
        "actual_m4_spatial_renderer_used": all(
            render.metadata["policy"] == SPATIAL_ROOM_RIR_POLICY
            for render in renders.values()
        ),
        "truth_scattering_directivity_candidate_selected": (
            selection.best_candidate_id == truth_id
        ),
        "best_to_second_total_ratio_below_0_9": (
            selection_dict["best_to_second_total_ratio"] <= 0.9
        ),
        "spatial_term_is_evaluable": bool(
            best_report["diagnostics"]["spatial_coherence"]["evaluable"]
        ),
        "truth_total_below_zero_scattering": (
            best_report["total"] < no_scattering_report["total"]
        ),
        "truth_total_below_wrong_directivity": (
            best_report["total"] < omni_report["total"]
        ),
        "mono_candidate_profile_is_rejected": mono_rejected,
        "noise_is_retained_in_target": not np.array_equal(target, clean_target),
        "all_candidates_finite": all(
            np.all(np.isfinite(render.receiver_rirs)) for render in renders.values()
        ),
        "all_candidates_remain_causal": all(
            all(
                np.count_nonzero(channel[: physical[index]]) == 0
                for index, channel in enumerate(render.receiver_rirs)
            )
            for render in renders.values()
        ),
        "production_default_unchanged": all(
            render.metadata["production_default_changed"] is False
            for render in renders.values()
        ),
    }
    artifacts = {
        "target_synchronized_pair": target,
        "selected_candidate": renders[truth_id].receiver_rirs,
        "wrong_zero_scattering": renders["zero_scattering_cardioid"].receiver_rirs,
        "wrong_omni_directivity": renders["truth_scattering_omni"].receiver_rirs,
    }
    report = {
        "schema_version": "puresound.m5_spatial_calibration_report.v1",
        "milestone": "M5.4",
        "policy": M5_SPATIAL_CANDIDATE_PROFILE_POLICY,
        "scope": "synchronized_actual_m4_scattering_directivity_profile",
        "sample_rate": int(sample_rate),
        "duration_s": float(duration_s),
        "target_snr_db": float(target_snr_db),
        "candidate_metadata": {
            candidate_id: {
                "scattering_scale": (0.0 if candidate_id.startswith("zero") else 1.0),
                "receiver_directivity": (
                    "omnidirectional"
                    if candidate_id.endswith("omni")
                    else "opposed_cardioid"
                ),
            }
            for candidate_id in renders
        },
        "scattering_evidence_boundary": {
            "selected_by": (
                "synchronized multichannel early/spectral/octave and total loss"
            ),
            "not_selected_by": "late spatial coherence alone",
            "reason": (
                "current M4 scattering partitions coherent first-order paths; "
                "the post-80-ms FDN field is scattering-independent"
            ),
        },
        "selection": selection_dict,
        "checks": checks,
        "exit": {
            "passed": bool(all(checks.values())),
            "m5_4_spatial_calibration_implementation_complete": bool(
                all(checks.values())
            ),
            "measured_spatial_calibration_complete": False,
        },
        "not_claimed": [
            "measured synchronized receiver calibration",
            "continuous scattering coefficient recovery",
            "licensed HRTF validation",
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
        path = artifact_dir / f"m5_4_{name}.wav"
        sf.write(path, np.asarray(audio).T, sample_rate, subtype="FLOAT")
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
        "# M5.4 synchronized spatial calibration: "
        f"{'PASS' if report['exit']['passed'] else 'FAIL'}"
    )
    print("# measured synchronized calibration: OPEN")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
