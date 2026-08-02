#!/usr/bin/env python
"""Validate the isolated M4.3 deterministic passive multiband FDN core.

The target RT60 and echo-density envelopes are read from the M4.2 measured
report. This validator does not splice the FDN into a production hybrid RIR;
that direct/early/late coupling remains M4.4.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.multiband_fdn import (  # noqa: E402
    MULTIBAND_FDN_POLICY,
    analyze_fdn_coloration,
    design_multiband_fdn,
    render_multiband_fdn_impulse,
)
from puresound.audio.rir_metrics import analyze_multiband_late_field  # noqa: E402


DEFAULT_TARGET_REPORT = (
    REPO_ROOT / "egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json"
)
DEFAULT_QUALIFIED_CENTERS_HZ = (500.0, 1000.0, 2000.0, 4000.0)


def _measured_targets(
    target_report: dict[str, Any],
    reference_tag: str,
) -> tuple[dict[float, float], dict[str, Any]]:
    try:
        measured = target_report["banks"][reference_tag]["summary"]["overall"]
    except KeyError as exc:
        raise ValueError(f"reference tag missing from M4.2 report: {reference_tag}") from exc
    targets: dict[float, float] = {}
    for key, band in measured["bands"].items():
        target = band["decay"]["t20"]["rt60_s"]["median"]
        if target is not None:
            targets[float(key)] = float(target)
    if not targets:
        raise ValueError("M4.2 report contains no measured T20 targets")
    return targets, measured


def _block_energy_decay(rir: np.ndarray, sample_rate: int) -> dict[str, float]:
    block_size = max(1, int(round(0.05 * sample_rate)))
    energies = np.asarray(
        [
            np.square(rir[start : start + block_size], dtype=np.float64).sum()
            for start in range(0, rir.size, block_size)
        ],
        dtype=np.float64,
    )
    maximum = float(np.max(energies)) if energies.size else 0.0
    final = float(energies[-1]) if energies.size else 0.0
    return {
        "block_ms": 50.0,
        "maximum_block_energy": maximum,
        "final_block_energy": final,
        "final_to_maximum_ratio": float(final / maximum) if maximum > 0.0 else 0.0,
    }


def _inside(value: float | None, lower: float | None, upper: float | None) -> bool:
    return (
        value is not None
        and lower is not None
        and upper is not None
        and lower <= value <= upper
    )


def build_report(
    target_report: dict[str, Any],
    *,
    reference_tag: str,
    sample_rate: int,
    duration_s: float,
    delay_line_count: int,
    seed: int,
    qualified_centers_hz: Iterable[float],
    maximum_rt60_relative_error: float,
) -> tuple[dict[str, Any], np.ndarray]:
    targets, measured = _measured_targets(target_report, reference_tag)
    qualified = {float(center) for center in qualified_centers_hz}
    missing = sorted(qualified - set(targets))
    if missing:
        raise ValueError(f"qualified centers missing measured targets: {missing}")
    target_mixing_time_s = max(
        float(measured["bands"][f"{center:g}"]["mixing_time_s"]["median"])
        for center in qualified
    )
    design = design_multiband_fdn(
        sample_rate,
        targets,
        target_mixing_time_s=target_mixing_time_s,
        delay_line_count=delay_line_count,
        seed=seed,
    )
    repeated_design = design_multiband_fdn(
        sample_rate,
        targets,
        target_mixing_time_s=target_mixing_time_s,
        delay_line_count=delay_line_count,
        seed=seed,
    )
    rendered = render_multiband_fdn_impulse(design, duration_s)
    repeated_render = render_multiband_fdn_impulse(repeated_design, duration_s)
    analysis = analyze_multiband_late_field(
        rendered.rir,
        sample_rate,
        direct_index=0,
        centers_hz=targets,
        base_window_ms=20.0,
        minimum_window_cycles=4.0,
        hop_ms=2.0,
        threshold=0.9,
        minimum_sustain_ms=10.0,
        probe_times_ms=(50.0, 100.0, 200.0, 400.0),
    )
    coloration = analyze_fdn_coloration(rendered.rir, sample_rate, targets)
    design_metadata = design.to_dict(include_coefficients=False)
    first_nonzero = np.flatnonzero(np.abs(rendered.rir) > 1e-14)
    first_nonzero_sample = int(first_nonzero[0]) if first_nonzero.size else None
    deterministic = bool(np.array_equal(rendered.rir, repeated_render.rir))
    finite = bool(np.all(np.isfinite(rendered.rir)))
    causal = bool(
        first_nonzero_sample is not None
        and first_nonzero_sample >= min(design.delay_lengths_samples)
    )
    block_decay = _block_energy_decay(rendered.rir, sample_rate)
    structural_checks = {
        "distinct_prime_delays": bool(design_metadata["delays_are_distinct_primes"]),
        "orthogonal_feedback": bool(
            design_metadata["feedback_orthogonality_error_2norm"] <= 1e-12
        ),
        "strict_feedback_contraction": bool(
            all(
                band["feedback_operator_2norm"] < 1.0
                for band in design_metadata["bands"].values()
            )
        ),
        "unit_band_weight_energy": bool(
            abs(design_metadata["band_weight_energy_sum"] - 1.0) <= 1e-12
        ),
        "deterministic_same_seed": deterministic,
        "finite_output": finite,
        "causal_delayed_onset": causal,
        "decays_before_render_end": bool(
            block_decay["final_to_maximum_ratio"] <= 1e-8
        ),
    }

    band_checks: dict[str, Any] = {}
    qualified_passes: list[bool] = []
    for center_hz, target_rt60_s in targets.items():
        key = f"{center_hz:g}"
        measured_band = measured["bands"][key]
        rendered_band = analysis["bands"][key]
        rendered_t20 = rendered_band["t20_s"]
        relative_error = (
            abs(float(rendered_t20) - target_rt60_s) / target_rt60_s
            if rendered_t20 is not None
            else None
        )
        mixing_time = rendered_band["echo_density"]["mixing_time_s"]
        late_density = rendered_band["echo_density"][
            "late_median_normalized_density"
        ]
        rt60_pass = (
            relative_error is not None
            and relative_error <= maximum_rt60_relative_error
        )
        mixing_pass = _inside(
            mixing_time,
            measured_band["mixing_time_s"]["p10"],
            measured_band["mixing_time_s"]["p90"],
        )
        density_pass = _inside(
            late_density,
            measured_band["late_median_normalized_density"]["p10"],
            measured_band["late_median_normalized_density"]["p90"],
        )
        is_qualified = center_hz in qualified
        if is_qualified:
            qualified_passes.extend((rt60_pass, mixing_pass, density_pass))
        band_checks[key] = {
            "center_hz": center_hz,
            "qualified_high_band_gate": is_qualified,
            "target_t20_s": target_rt60_s,
            "rendered_t20_s": rendered_t20,
            "t20_relative_error": relative_error,
            "t20_relative_error_passed": rt60_pass,
            "rendered_mixing_time_s": mixing_time,
            "measured_mixing_time_p10_s": measured_band["mixing_time_s"]["p10"],
            "measured_mixing_time_p90_s": measured_band["mixing_time_s"]["p90"],
            "mixing_time_envelope_passed": mixing_pass,
            "rendered_late_median_normalized_density": late_density,
            "measured_late_density_p10": measured_band[
                "late_median_normalized_density"
            ]["p10"],
            "measured_late_density_p90": measured_band[
                "late_median_normalized_density"
            ]["p90"],
            "late_density_envelope_passed": density_pass,
            "coloration": coloration["bands"].get(key),
        }

    all_structural = all(structural_checks.values())
    high_band_passed = bool(qualified_passes) and all(qualified_passes)
    report = {
        "schema_version": 1,
        "milestone": "M4.3",
        "policy": MULTIBAND_FDN_POLICY,
        "scope": "isolated_late_field_core_not_production_hybrid_coupling",
        "target_report_milestone": target_report.get("milestone"),
        "reference_tag": reference_tag,
        "config": {
            "sample_rate": int(sample_rate),
            "duration_s": float(duration_s),
            "delay_line_count": int(delay_line_count),
            "seed": int(seed),
            "qualified_centers_hz": sorted(qualified),
            "maximum_rt60_relative_error": float(maximum_rt60_relative_error),
            "target_mixing_time_s": float(target_mixing_time_s),
        },
        "design": design_metadata,
        "render": {
            "sample_count": int(rendered.rir.size),
            "peak_abs": float(np.max(np.abs(rendered.rir))),
            "energy": float(np.sum(np.square(rendered.rir))),
            "first_nonzero_sample": first_nonzero_sample,
            "first_nonzero_ms": (
                float(1000.0 * first_nonzero_sample / sample_rate)
                if first_nonzero_sample is not None
                else None
            ),
            "block_energy_decay": block_decay,
        },
        "structural_checks": structural_checks,
        "band_checks": band_checks,
        "coloration_scope_note": (
            "Spectral flatness and p95/median ripple are diagnostics only; "
            "M4 listening tests remain required for metallic coloration."
        ),
        "exit": {
            "structural_checks_passed": bool(all_structural),
            "qualified_high_band_checks_passed": bool(high_band_passed),
            "passed": bool(all_structural and high_band_passed),
            "limitations": [
                "125/250 Hz remain diagnostic because the hybrid low/modal branch "
                "and parallel octave crossover affect fitted decay.",
                "The FDN tail is not yet coupled to PathEvent direct/early energy.",
                "No measured multi-receiver spatial output is available for IACC/coherence exit.",
                "Coloration requires listening validation in addition to spectral diagnostics.",
            ],
        },
    }
    return report, rendered.rir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--target-report", type=Path, default=DEFAULT_TARGET_REPORT)
    parser.add_argument("--reference-tag", default="measured")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=1.5)
    parser.add_argument("--delay-lines", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument(
        "--qualified-centers-hz",
        nargs="+",
        type=float,
        default=list(DEFAULT_QUALIFIED_CENTERS_HZ),
    )
    parser.add_argument("--maximum-rt60-relative-error", type=float, default=0.1)
    parser.add_argument("--output-report", type=Path)
    parser.add_argument("--output-wav", type=Path)
    args = parser.parse_args()
    try:
        target_report = json.loads(args.target_report.read_text(encoding="utf-8"))
        report, rir = build_report(
            target_report,
            reference_tag=args.reference_tag,
            sample_rate=args.sample_rate,
            duration_s=args.duration,
            delay_line_count=args.delay_lines,
            seed=args.seed,
            qualified_centers_hz=args.qualified_centers_hz,
            maximum_rt60_relative_error=args.maximum_rt60_relative_error,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    for key, value in report["structural_checks"].items():
        print(f"structure\t{key}\t{'PASS' if value else 'FAIL'}")
    print("band_hz\tgate\tT20_target\tT20_render\terror%\tmixing_ms\tlate_NED")
    for key, band in report["band_checks"].items():
        error = band["t20_relative_error"]
        print(
            f"{key}\t{'qualified' if band['qualified_high_band_gate'] else 'diagnostic'}"
            f"\t{band['target_t20_s']:.3f}"
            f"\t{band['rendered_t20_s']:.3f}"
            f"\t{100.0 * error:.2f}"
            f"\t{1000.0 * band['rendered_mixing_time_s']:.2f}"
            f"\t{band['rendered_late_median_normalized_density']:.3f}"
        )
    print(f"# M4.3 isolated core: {'PASS' if report['exit']['passed'] else 'FAIL'}")

    if args.output_report is not None:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"# wrote {args.output_report}")
    if args.output_wav is not None:
        args.output_wav.parent.mkdir(parents=True, exist_ok=True)
        sf.write(args.output_wav, rir.astype(np.float32), args.sample_rate, subtype="FLOAT")
        print(f"# wrote {args.output_wav}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
