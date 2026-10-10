#!/usr/bin/env python
"""Validate M4.5 synchronized receiver-array and Ambisonic RIR synthesis."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.contracts import HybridRIRConfig
from puresound.audio.rir.scene.sampling import sample_material_first_rir_scene
from puresound.audio.rir.metrics import (
    analyze_array_spatial_coherence,
    analyze_binaural_iacc,
)
from puresound.audio.rir.render.spatial import render_room_scene_spatial_rir


DEFAULT_OUTPUT_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m4_spatial_late_field/reports/m4_spatial_rir_report.json"
)
DEFAULT_OUTPUT_DIR = REPO_ROOT / "egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial"


def build_binaural_scene(
    *,
    sample_rate: int,
    duration_s: float,
    scene_seed: int,
    spacing_m: float,
):
    """Build one deterministic material-first scene with a two-channel array."""

    config = HybridRIRConfig(
        sample_rate=int(sample_rate),
        duration=float(duration_s),
        output_mode="calibrated",
        match_crossover_energy=False,
        room_dim_range=((5.0, 5.0), (4.0, 4.0), (2.8, 2.8)),
        num_obstacles_range=(0, 0),
    )
    scene = sample_material_first_rir_scene(
        config,
        seed=int(scene_seed),
        room_type="office",
        scene_id="m4-5-spatial-validator",
    )
    original = scene.receivers[0]
    center = np.asarray(original.pose.position_m, dtype=np.float64)
    half_spacing = 0.5 * float(spacing_m)
    left = replace(
        original,
        transducer_id="binaural-left",
        array_id="binaural-0",
        channel_index=0,
        directivity_id="omnidirectional",
        pose=replace(
            original.pose,
            position_m=(center + [-half_spacing, 0.0, 0.0]).tolist(),
        ),
    )
    right = replace(
        original,
        transducer_id="binaural-right",
        array_id="binaural-0",
        channel_index=1,
        directivity_id="omnidirectional",
        pose=replace(
            original.pose,
            position_m=(center + [half_spacing, 0.0, 0.0]).tolist(),
        ),
    )
    return replace(scene, receivers=[left, right])


def build_report(
    *,
    sample_rate: int = 16000,
    duration_s: float = 1.2,
    max_order: int = 4,
    scene_seed: int = 20260731,
    spatial_seed: int = 20260731,
    spacing_m: float = 0.17,
    plane_wave_count: int = 256,
    delay_line_count: int = 16,
    nperseg: int = 48,
    maximum_coherence_rmse: float = 0.20,
    maximum_band_coherence_rmse: float = 0.23,
    maximum_imaginary_rms: float = 0.15,
    maximum_late_iacc_l4: float = 0.50,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Render twice and return M4.5 structural and spatial evidence."""

    scene = build_binaural_scene(
        sample_rate=sample_rate,
        duration_s=duration_s,
        scene_seed=scene_seed,
        spacing_m=spacing_m,
    )
    rendered = render_room_scene_spatial_rir(
        scene,
        sample_rate=sample_rate,
        duration_s=duration_s,
        source_index=0,
        max_order=max_order,
        delay_line_count=delay_line_count,
        plane_wave_count=plane_wave_count,
        seed=spatial_seed,
    )
    repeated = render_room_scene_spatial_rir(
        scene,
        sample_rate=sample_rate,
        duration_s=duration_s,
        source_index=0,
        max_order=max_order,
        delay_line_count=delay_line_count,
        plane_wave_count=plane_wave_count,
        seed=spatial_seed,
    )
    direct_sample = min(rendered.metadata["direct_samples"])
    first_physical_sample = min(
        int(
            math.floor(
                np.linalg.norm(
                    np.asarray(scene.sources[0].pose.position_m)
                    - np.asarray(receiver.pose.position_m)
                )
                / float(scene.environment.sound_speed_m_s)
                * sample_rate
            )
        )
        for receiver in scene.receivers
    )
    coherence = analyze_array_spatial_coherence(
        rendered.receiver_rirs[0],
        rendered.receiver_rirs[1],
        sample_rate,
        microphone_spacing_m=spacing_m,
        sound_speed_m_s=float(scene.environment.sound_speed_m_s),
        direct_index=direct_sample,
        start_ms=80.0,
        nperseg=nperseg,
        centers_hz=(500.0, 1000.0, 2000.0, 4000.0),
    )
    iacc = analyze_binaural_iacc(
        rendered.receiver_rirs[0],
        rendered.receiver_rirs[1],
        sample_rate,
        direct_index=direct_sample,
        early_end_ms=80.0,
        centers_hz=(500.0, 1000.0, 2000.0, 4000.0),
    )
    receiver_channels = rendered.receiver_coupling.metadata["channels"]
    ambisonic_channels = rendered.ambisonic_coupling.metadata["channels"]
    structural_checks = {
        "receiver_shape_is_two_by_samples": bool(
            rendered.receiver_rirs.shape == (2, round(duration_s * sample_rate))
        ),
        "ambisonic_shape_is_four_by_samples": bool(
            rendered.ambisonic_acn_sn3d.shape
            == (4, round(duration_s * sample_rate))
        ),
        "same_seed_exact_receiver_determinism": bool(
            np.array_equal(rendered.receiver_rirs, repeated.receiver_rirs)
        ),
        "same_seed_exact_ambisonic_determinism": bool(
            np.array_equal(
                rendered.ambisonic_acn_sn3d,
                repeated.ambisonic_acn_sn3d,
            )
        ),
        "finite_receiver_output": bool(
            np.all(np.isfinite(rendered.receiver_rirs))
        ),
        "finite_ambisonic_output": bool(
            np.all(np.isfinite(rendered.ambisonic_acn_sn3d))
        ),
        "causal_before_earliest_physical_arrival": bool(
            np.all(rendered.receiver_rirs[:, :first_physical_sample] == 0.0)
        ),
        "receiver_early_exact_through_transition_start": bool(
            max(
                channel["pre_transition_max_abs_error"]
                for channel in receiver_channels
            )
            <= 1e-12
        ),
        "ambisonic_early_exact_through_transition_start": bool(
            max(
                channel["pre_transition_max_abs_error"]
                for channel in ambisonic_channels
            )
            <= 1e-12
        ),
        "receiver_post_transition_energy_exact": bool(
            rendered.receiver_coupling.metadata["aggregate_energy"][
                "relative_energy_error"
            ]
            <= 1e-10
        ),
        "ambisonic_post_transition_energy_exact": bool(
            rendered.ambisonic_coupling.metadata["aggregate_energy"][
                "relative_energy_error"
            ]
            <= 1e-10
        ),
        "one_shared_gain_preserves_spatial_channel_ratios": bool(
            rendered.receiver_coupling.metadata["energy_policy"]
            == "one_shared_array_gain_preserves_spatial_ratios"
            and rendered.ambisonic_coupling.metadata["energy_policy"]
            == "one_shared_array_gain_preserves_spatial_ratios"
        ),
        "ambisonic_channel_order_is_acn_sn3d_wyzx": bool(
            rendered.spatial_late_field.metadata["ambisonic"]
            ["channel_labels"]
            == ["W", "Y", "Z", "X"]
        ),
        "production_default_unchanged": bool(
            rendered.metadata["production_default_changed"] is False
        ),
    }
    band_gates = {
        key: bool(band["complex_rmse"] <= maximum_band_coherence_rmse)
        for key, band in coherence.get("bands", {}).items()
    }
    qualified_frequency_bin_count = sum(
        band["frequency_bin_count"]
        for band in coherence.get("bands", {}).values()
    )
    qualified_complex_rmse = (
        math.sqrt(
            sum(
                band["frequency_bin_count"] * band["complex_rmse"] ** 2
                for band in coherence.get("bands", {}).values()
            )
            / qualified_frequency_bin_count
        )
        if qualified_frequency_bin_count > 0
        else None
    )
    late_iacc_l4 = iacc["iacc_l4"]
    spatial_checks = {
        "coherence_estimate_valid": bool(coherence.get("valid", False)),
        "qualified_octave_complex_coherence_rmse_within_tolerance": bool(
            qualified_complex_rmse is not None
            and qualified_complex_rmse <= maximum_coherence_rmse
        ),
        "imaginary_coherence_rms_within_tolerance": bool(
            coherence.get("valid", False)
            and coherence["measured_imaginary_rms"] <= maximum_imaginary_rms
        ),
        "all_reported_octave_coherence_bands_within_tolerance": bool(
            band_gates and all(band_gates.values())
        ),
        "late_iacc_l4_within_model_gate": bool(
            late_iacc_l4 is not None and late_iacc_l4 <= maximum_late_iacc_l4
        ),
    }
    passed = bool(
        all(structural_checks.values()) and all(spatial_checks.values())
    )
    report = {
        "schema_version": "puresound.m4_spatial_rir_validation.v1",
        "stage": "M4.5",
        "evidence_scope": "model_derived_deterministic_validation",
        "configuration": {
            "sample_rate": int(sample_rate),
            "duration_s": float(duration_s),
            "max_path_event_order": int(max_order),
            "scene_seed": int(scene_seed),
            "spatial_seed": int(spatial_seed),
            "receiver_spacing_m": float(spacing_m),
            "plane_wave_count": int(plane_wave_count),
            "delay_line_count": int(delay_line_count),
            "coherence_estimator": {
                "method": "Welch complex coherence on decaying late RIR",
                "start_ms_after_direct": 80.0,
                "nperseg": int(nperseg),
                "reason_for_short_segment": (
                    "reduce variance from nonstationary exponential decay"
                ),
            },
        },
        "scene": scene.to_metadata(),
        "renderer": rendered.metadata,
        "structural_checks": structural_checks,
        "spatial_checks": spatial_checks,
        "coherence_band_gates": band_gates,
        "qualified_octave_coherence": {
            "centers_hz": [
                float(key) for key in coherence.get("bands", {})
            ],
            "frequency_bin_count": int(qualified_frequency_bin_count),
            "complex_rmse": qualified_complex_rmse,
            "scope": (
                "500 Hz through 4 kHz nominal octave bands frozen by M4.2"
            ),
        },
        "coherence": coherence,
        "iacc": iacc,
        "thresholds": {
            "maximum_coherence_rmse": float(maximum_coherence_rmse),
            "maximum_band_coherence_rmse": float(
                maximum_band_coherence_rmse
            ),
            "maximum_imaginary_rms": float(maximum_imaginary_rms),
            "maximum_late_iacc_l4": float(maximum_late_iacc_l4),
        },
        "exit": {
            "passed": passed,
            "implementation_complete": passed,
            "production_default_enabled": False,
            "measured_multi_receiver_validation_complete": False,
        },
        "limitations": [
            "Diffuse coherence is validated against the isotropic plane-wave model, not a measured multi-receiver room dataset.",
            "The finite plane-wave quadrature and decaying-RIR Welch estimator have deterministic approximation error.",
            "FOA coherent and late paths use the receiver-array centroid as the Ambisonic origin.",
        ],
    }
    artifacts = {
        "m4_spatial_receiver_rir_2ch": rendered.receiver_rirs,
        "m4_spatial_ambisonic_rir_4ch": rendered.ambisonic_acn_sn3d,
        "m4_spatial_coherent_receiver_2ch": (
            rendered.coherent_receiver_rirs
        ),
        "m4_spatial_late_receiver_2ch": (
            rendered.spatial_late_field.receiver_rirs
        ),
    }
    return report, artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--scene-seed", type=int, default=20260731)
    parser.add_argument("--spatial-seed", type=int, default=20260731)
    parser.add_argument("--spacing-m", type=float, default=0.17)
    parser.add_argument("--plane-waves", type=int, default=256)
    parser.add_argument("--delay-lines", type=int, default=16)
    parser.add_argument("--nperseg", type=int, default=48)
    parser.add_argument("--output-report", type=Path, default=DEFAULT_OUTPUT_REPORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    try:
        report, artifacts = build_report(
            sample_rate=args.sample_rate,
            duration_s=args.duration,
            max_order=args.max_order,
            scene_seed=args.scene_seed,
            spatial_seed=args.spatial_seed,
            spacing_m=args.spacing_m,
            plane_wave_count=args.plane_waves,
            delay_line_count=args.delay_lines,
            nperseg=args.nperseg,
        )
    except (ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    for key, value in report["structural_checks"].items():
        print(f"structure\t{key}\t{'PASS' if value else 'FAIL'}")
    for key, value in report["spatial_checks"].items():
        print(f"spatial\t{key}\t{'PASS' if value else 'FAIL'}")
    print(
        "coherence\tqualified_octave_complex_rmse"
        f"\t{report['qualified_octave_coherence']['complex_rmse']:.6f}"
    )
    print(
        "coherence\tfull_spectrum_diagnostic_rmse"
        f"\t{report['coherence']['complex_rmse']:.6f}"
    )
    print(f"iacc\tlate_l4\t{report['iacc']['iacc_l4']:.6f}")
    print(f"# M4.5 spatial RIR: {'PASS' if report['exit']['passed'] else 'FAIL'}")

    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, audio in artifacts.items():
        path = args.output_dir / f"{name}.wav"
        sf.write(path, np.asarray(audio).T, args.sample_rate, subtype="FLOAT")
        print(f"# wrote {path}")
    print(f"# wrote {args.output_report}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
