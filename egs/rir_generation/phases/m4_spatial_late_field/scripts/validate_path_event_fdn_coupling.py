#!/usr/bin/env python
"""Validate M4.4 causal PathEvent-early / multiband-FDN-late coupling."""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.hybrid_rir import (  # noqa: E402
    AnalyticModalLowFrequencyBackend,
    HybridRIRConfig,
    PathEventFDNHighFrequencyBackend,
    PathEventHighFrequencyBackend,
    hybrid_crossover,
    sample_material_first_rir_scene,
)
from puresound.audio.rir_metrics import (  # noqa: E402
    analyze_multiband_late_field,
    clarity_db,
    valid_octave_centers,
)


DEFAULT_TARGET_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json"
)
DEFAULT_QUALIFIED_CENTERS_HZ = (500.0, 1000.0, 2000.0, 4000.0)


def _measured_overall(
    target_report: dict[str, Any],
    reference_tag: str,
) -> dict[str, Any]:
    try:
        return target_report["banks"][reference_tag]["summary"]["overall"]
    except KeyError as exc:
        raise ValueError(
            f"reference tag missing from M4.2 report: {reference_tag}"
        ) from exc


def _finite_median(values: Iterable[float | None]) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def _early_reflection_energy_centroid_ms(
    rir: np.ndarray,
    sample_rate: int,
    direct_sample: int,
    *,
    exclusion_ms: float = 2.5,
    maximum_delay_ms: float = 50.0,
) -> float | None:
    start = direct_sample + max(1, round(exclusion_ms * 1e-3 * sample_rate))
    end = min(
        rir.size,
        direct_sample + round(maximum_delay_ms * 1e-3 * sample_rate),
    )
    if end <= start:
        return None
    energy = np.square(np.asarray(rir[start:end], dtype=np.float64))
    total = float(np.sum(energy))
    if total <= np.finfo(np.float64).tiny:
        return None
    delay_ms = (
        np.arange(start, end, dtype=np.float64) - direct_sample
    ) / sample_rate * 1000.0
    return float(np.sum(delay_ms * energy) / total)


def _band_analyses(
    rirs: np.ndarray,
    direct_samples: list[int],
    sample_rate: int,
    centers_hz: Iterable[float],
) -> list[dict[str, Any]]:
    return [
        analyze_multiband_late_field(
            channel,
            sample_rate,
            direct_index=direct,
            centers_hz=centers_hz,
            probe_times_ms=(50.0, 100.0, 200.0, 400.0),
        )
        for channel, direct in zip(rirs, direct_samples)
    ]


def build_report(
    target_report: dict[str, Any],
    *,
    reference_tag: str = "measured",
    sample_rate: int = 16000,
    duration_s: float = 1.2,
    max_order: int = 4,
    scene_seed: int = 20260731,
    fdn_seed: int = 20260731,
    mixing_time_s: float = 0.024,
    transition_duration_s: float = 0.016,
    delay_line_count: int = 16,
    qualified_centers_hz: Iterable[float] = DEFAULT_QUALIFIED_CENTERS_HZ,
    maximum_rt60_relative_error: float = 0.12,
    maximum_late_density_absolute_error: float = 0.08,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    measured = _measured_overall(target_report, reference_tag)
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
        scene_id="m4-4-coupling-validator",
    )
    config = replace(
        config,
        sound_speed=float(scene.environment.sound_speed_m_s),
    )
    coherent_backend = PathEventHighFrequencyBackend(max_order=int(max_order))
    coupled_backend = PathEventFDNHighFrequencyBackend(
        max_order=int(max_order),
        mixing_time_s=float(mixing_time_s),
        transition_duration_s=float(transition_duration_s),
        delay_line_count=int(delay_line_count),
        fdn_seed=int(fdn_seed),
    )
    repeated_backend = PathEventFDNHighFrequencyBackend(
        max_order=int(max_order),
        mixing_time_s=float(mixing_time_s),
        transition_duration_s=float(transition_duration_s),
        delay_line_count=int(delay_line_count),
        fdn_seed=int(fdn_seed),
    )
    coherent = coherent_backend.simulate(scene, config).astype(np.float64)
    coupled = coupled_backend.simulate(scene, config).astype(np.float64)
    repeated = repeated_backend.simulate(scene, config).astype(np.float64)
    coupling = coupled_backend.last_late_field_metadata
    if coupling is None:
        raise RuntimeError("M4 backend did not expose coupling metadata")

    distances = scene.source_distances()
    direct_samples = [
        int(round(distance / config.sound_speed * config.sample_rate))
        for distance in distances
    ]
    first_physical_samples = [
        int(math.floor(distance / config.sound_speed * config.sample_rate))
        for distance in distances
    ]
    channel_metadata = coupling["channels"]
    pre_transition_errors = [
        float(item["pre_transition_max_abs_error"])
        for item in channel_metadata
    ]
    energy_errors = [
        float(item["energy"]["relative_energy_error"])
        for item in channel_metadata
    ]
    channel_seeds = [int(item["seed"]) for item in channel_metadata]
    structural_checks = {
        "shape_preserved": bool(coherent.shape == coupled.shape),
        "same_seed_exact_determinism": bool(np.array_equal(coupled, repeated)),
        "finite_output": bool(np.all(np.isfinite(coupled))),
        "causal_no_prearrival": bool(
            all(
                np.count_nonzero(np.abs(channel[:first]) > 1e-12) == 0
                for channel, first in zip(coupled, first_physical_samples)
            )
        ),
        "pre_transition_exact_preservation": bool(
            max(pre_transition_errors, default=math.inf) == 0.0
        ),
        "post_transition_energy_preserved": bool(
            max(energy_errors, default=math.inf) <= 1e-10
        ),
        "channel_seeds_are_distinct": bool(
            len(set(channel_seeds)) == len(channel_seeds)
        ),
        "production_default_unchanged": bool(
            coupling["production_default_changed"] is False
        ),
    }

    target_rt60 = {
        float(center): float(value)
        for center, value in coupling["target_rt60_s_by_hz"].items()
    }
    qualified = sorted(
        set(float(center) for center in qualified_centers_hz)
        & set(valid_octave_centers(sample_rate, target_rt60))
        & set(target_rt60)
    )
    if not qualified:
        raise ValueError("no qualified M4.4 octave center remains valid")
    missing_measured = [
        center for center in qualified if f"{center:g}" not in measured["bands"]
    ]
    if missing_measured:
        raise ValueError(
            f"qualified centers missing from M4.2 report: {missing_measured}"
        )

    coherent_analysis = _band_analyses(
        coherent,
        direct_samples,
        sample_rate,
        qualified,
    )
    coupled_analysis = _band_analyses(
        coupled,
        direct_samples,
        sample_rate,
        qualified,
    )
    band_checks: dict[str, Any] = {}
    acoustic_passes: list[bool] = []
    for center in qualified:
        key = f"{center:g}"
        m3_bands = [item["bands"][key] for item in coherent_analysis]
        m4_bands = [item["bands"][key] for item in coupled_analysis]
        m4_t20 = _finite_median(item["t20_s"] for item in m4_bands)
        m3_mixing = _finite_median(
            item["echo_density"]["mixing_time_s"] for item in m3_bands
        )
        m4_mixing = _finite_median(
            item["echo_density"]["mixing_time_s"] for item in m4_bands
        )
        m3_density = _finite_median(
            item["echo_density"]["late_median_normalized_density"]
            for item in m3_bands
        )
        m4_density = _finite_median(
            item["echo_density"]["late_median_normalized_density"]
            for item in m4_bands
        )
        measured_band = measured["bands"][key]
        measured_density = float(
            measured_band["late_median_normalized_density"]["median"]
        )
        t20_error = (
            abs(m4_t20 - target_rt60[center]) / target_rt60[center]
            if m4_t20 is not None
            else None
        )
        m3_density_error = (
            abs(m3_density - measured_density)
            if m3_density is not None
            else None
        )
        m4_density_error = (
            abs(m4_density - measured_density)
            if m4_density is not None
            else None
        )
        t20_pass = bool(
            t20_error is not None
            and t20_error <= maximum_rt60_relative_error
        )
        mixing_pass = bool(
            m4_mixing is not None
            and measured_band["mixing_time_s"]["p10"]
            <= m4_mixing
            <= measured_band["mixing_time_s"]["p90"]
        )
        density_pass = bool(
            m4_density_error is not None
            and m4_density_error <= maximum_late_density_absolute_error
        )
        density_improved = bool(
            m3_density_error is not None
            and m4_density_error is not None
            and m4_density_error < m3_density_error
        )
        acoustic_passes.extend(
            (t20_pass, mixing_pass, density_pass, density_improved)
        )
        band_checks[key] = {
            "center_hz": center,
            "material_target_rt60_s": target_rt60[center],
            "m4_median_t20_s": m4_t20,
            "m4_t20_relative_error": t20_error,
            "t20_passed": t20_pass,
            "m3_median_mixing_time_s": m3_mixing,
            "m4_median_mixing_time_s": m4_mixing,
            "measured_mixing_time_p10_s": measured_band["mixing_time_s"]["p10"],
            "measured_mixing_time_p90_s": measured_band["mixing_time_s"]["p90"],
            "mixing_time_passed": mixing_pass,
            "measured_median_late_density": measured_density,
            "m3_median_late_density": m3_density,
            "m4_median_late_density": m4_density,
            "m3_late_density_absolute_error": m3_density_error,
            "m4_late_density_absolute_error": m4_density_error,
            "late_density_tolerance_passed": density_pass,
            "late_density_improved_over_m3": density_improved,
        }

    low_backend = AnalyticModalLowFrequencyBackend(
        num_modes_per_axis=3,
        max_modes=64,
        material_modal_damping=True,
    )
    low = low_backend.simulate(scene, config)
    full_m3 = hybrid_crossover(low, coherent, config).astype(np.float64)
    full_m4 = hybrid_crossover(low, coupled, config).astype(np.float64)
    full_pre_errors = [
        float(
            np.max(
                np.abs(
                    full_m4[index, : item["transition_start_sample"] + 1]
                    - full_m3[index, : item["transition_start_sample"] + 1]
                )
            )
        )
        for index, item in enumerate(channel_metadata)
    ]
    c50_m3 = [
        clarity_db(channel, sample_rate, direct_index=direct)
        for channel, direct in zip(full_m3, direct_samples)
    ]
    c50_m4 = [
        clarity_db(channel, sample_rate, direct_index=direct)
        for channel, direct in zip(full_m4, direct_samples)
    ]
    c50_delta = [abs(after - before) for before, after in zip(c50_m3, c50_m4)]
    centroid_m3 = [
        _early_reflection_energy_centroid_ms(channel, sample_rate, direct)
        for channel, direct in zip(full_m3, direct_samples)
    ]
    centroid_m4 = [
        _early_reflection_energy_centroid_ms(channel, sample_rate, direct)
        for channel, direct in zip(full_m4, direct_samples)
    ]
    centroid_delta = [
        abs(after - before)
        for before, after in zip(centroid_m3, centroid_m4)
        if before is not None and after is not None
    ]
    full_hybrid_checks = {
        "exact_before_transition": bool(max(full_pre_errors) == 0.0),
        "median_absolute_c50_change_at_most_1db": bool(
            float(np.median(c50_delta)) <= 1.0
        ),
        "maximum_absolute_c50_change_at_most_3db": bool(max(c50_delta) <= 3.0),
        "median_early_centroid_change_at_most_1ms": bool(
            centroid_delta and float(np.median(centroid_delta)) <= 1.0
        ),
        "maximum_early_centroid_change_at_most_2_5ms": bool(
            centroid_delta and max(centroid_delta) <= 2.5
        ),
    }

    all_structural = all(structural_checks.values())
    all_acoustic = bool(acoustic_passes) and all(acoustic_passes)
    all_full_hybrid = all(full_hybrid_checks.values())
    report = {
        "schema_version": 1,
        "milestone": "M4.4",
        "scope": "opt_in_path_event_early_multiband_fdn_late_coupling",
        "reference_tag": reference_tag,
        "config": {
            "sample_rate": int(sample_rate),
            "duration_s": float(duration_s),
            "max_path_order": int(max_order),
            "scene_seed": int(scene_seed),
            "fdn_seed": int(fdn_seed),
            "mixing_time_s": float(mixing_time_s),
            "transition_duration_s": float(transition_duration_s),
            "delay_line_count": int(delay_line_count),
            "qualified_centers_hz": qualified,
            "maximum_rt60_relative_error": float(maximum_rt60_relative_error),
            "maximum_late_density_absolute_error": float(
                maximum_late_density_absolute_error
            ),
        },
        "scene": {
            "scene_id": scene.scene_id,
            "room_type": scene.room_type,
            "room_dim_m": scene.room_dim,
            "source_distances_m": distances,
            "material_target_rt60_s_by_hz": {
                f"{center:g}": value for center, value in target_rt60.items()
            },
        },
        "coupling": coupling,
        "structural_checks": structural_checks,
        "band_checks": band_checks,
        "full_hybrid": {
            "checks": full_hybrid_checks,
            "pre_transition_max_abs_error_by_channel": full_pre_errors,
            "m3_c50_db_by_channel": c50_m3,
            "m4_c50_db_by_channel": c50_m4,
            "absolute_c50_change_db_by_channel": c50_delta,
            "m3_early_centroid_ms_by_channel": centroid_m3,
            "m4_early_centroid_ms_by_channel": centroid_m4,
            "absolute_early_centroid_change_ms_by_channel": centroid_delta,
        },
        "exit": {
            "structural_checks_passed": all_structural,
            "qualified_acoustic_checks_passed": all_acoustic,
            "full_hybrid_compatibility_checks_passed": all_full_hybrid,
            "passed": bool(all_structural and all_acoustic and all_full_hybrid),
            "limitations": [
                "This is one deterministic material-first room fixture, not a measured-room exit.",
                "The late field remains monophonic per source channel; M4.5 owns receiver-array spatial output.",
                "Metallic coloration still requires controlled listening tests.",
                "Pyroomacoustics remains the production default; path-events-m4 is explicit opt-in.",
            ],
        },
    }
    artifacts = {
        "m3_coherent_high": coherent,
        "m4_coupled_high": coupled,
        "m3_full_hybrid": full_m3,
        "m4_full_hybrid": full_m4,
    }
    return report, artifacts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-report", type=Path, default=DEFAULT_TARGET_REPORT)
    parser.add_argument("--reference-tag", default="measured")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.2)
    parser.add_argument("--max-order", type=int, default=4)
    parser.add_argument("--scene-seed", type=int, default=20260731)
    parser.add_argument("--fdn-seed", type=int, default=20260731)
    parser.add_argument("--mixing-time-ms", type=float, default=24.0)
    parser.add_argument("--transition-ms", type=float, default=16.0)
    parser.add_argument("--delay-lines", type=int, default=16)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    try:
        target_report = json.loads(args.target_report.read_text(encoding="utf-8"))
        report, artifacts = build_report(
            target_report,
            reference_tag=args.reference_tag,
            sample_rate=args.sample_rate,
            duration_s=args.duration,
            max_order=args.max_order,
            scene_seed=args.scene_seed,
            fdn_seed=args.fdn_seed,
            mixing_time_s=1e-3 * args.mixing_time_ms,
            transition_duration_s=1e-3 * args.transition_ms,
            delay_line_count=args.delay_lines,
        )
    except (OSError, json.JSONDecodeError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    for key, value in report["structural_checks"].items():
        print(f"structure\t{key}\t{'PASS' if value else 'FAIL'}")
    print("band_hz\tT20_error%\tM3_mix_ms\tM4_mix_ms\tM3_NED\tM4_NED\tgate")
    for key, band in report["band_checks"].items():
        m3_mix = band["m3_median_mixing_time_s"]
        print(
            f"{key}\t{100.0 * band['m4_t20_relative_error']:.2f}"
            f"\t{1000.0 * m3_mix if m3_mix is not None else float('nan'):.2f}"
            f"\t{1000.0 * band['m4_median_mixing_time_s']:.2f}"
            f"\t{band['m3_median_late_density']:.3f}"
            f"\t{band['m4_median_late_density']:.3f}"
            f"\t{'PASS' if all((band['t20_passed'], band['mixing_time_passed'], band['late_density_tolerance_passed'], band['late_density_improved_over_m3'])) else 'FAIL'}"
        )
    for key, value in report["full_hybrid"]["checks"].items():
        print(f"full_hybrid\t{key}\t{'PASS' if value else 'FAIL'}")
    print(f"# M4.4 coupling: {'PASS' if report['exit']['passed'] else 'FAIL'}")

    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            json.dumps(report, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        print(f"# wrote {args.output_report}")
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        for name, audio in artifacts.items():
            path = args.output_dir / f"{name}_5ch.wav"
            sf.write(path, np.asarray(audio).T, args.sample_rate, subtype="FLOAT")
            print(f"# wrote {path}")
    return 0 if report["exit"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
