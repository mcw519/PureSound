#!/usr/bin/env python
"""Build the M4.2 noise-aware octave-band late-field target report.

The legacy RIR bank stores several source-to-one-receiver responses. Those WAV
channels are not microphone-array channels, so this validator deliberately
does not compute IACC or inter-microphone coherence from them. It freezes the
monophonic multiband target and records the separate spatial input contract.

Example:
  .venv/bin/python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_multiband_late_field.py \
      measured=/work/any_exp_link/puresound_exp/real_rir_16k_train_view/items \
      m1=egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100 m3=egs/rir_generation/exp/rir_realism/m3/rir_m3_late_baseline \
      --reference-tag measured --per-bank 50 --seed 20260731 \
      --json-output egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from egs.rir_generation.phases.m4_spatial_late_field.scripts.validate_late_field_baseline import (  # noqa: E402
    discover_items,
    distance_bucket,
)
from puresound.audio.rir.metrics import (
    DIFFUSE_FIELD_COHERENCE_POLICY,
    IACC_POLICY,
    MULTIBAND_LATE_FIELD_POLICY,
    analyze_multiband_late_field,
    direct_sample,
)


M4_LATE_FIELD_CENTERS_HZ = (125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0)
PROBE_TIMES_MS = (50.0, 100.0, 200.0, 400.0)


def distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
    finite = np.asarray(
        [
            float(value)
            for value in values
            if value is not None and np.isfinite(float(value))
        ],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {"count": 0, "p10": None, "median": None, "p90": None}
    return {
        "count": int(finite.size),
        "p10": float(np.percentile(finite, 10.0)),
        "median": float(np.median(finite)),
        "p90": float(np.percentile(finite, 90.0)),
    }


def _qualified_decay(
    band: dict[str, Any],
    name: str,
    minimum_r_squared: float,
) -> float | None:
    estimate = band.get(name)
    if estimate is None or float(estimate["r_squared"]) < minimum_r_squared:
        return None
    return float(estimate["rt60_s"])


def sample_bank(
    bank_root: Path,
    *,
    count: int,
    rng: random.Random,
    centers_hz: Iterable[float],
    base_window_ms: float,
    minimum_window_cycles: float,
    hop_ms: float,
    threshold: float,
    minimum_sustain_ms: float,
) -> tuple[list[dict[str, Any]], int]:
    pairs = discover_items(bank_root)
    rng.shuffle(pairs)
    rows: list[dict[str, Any]] = []
    for metadata_path, wav_path in pairs:
        if len(rows) >= count:
            break
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            scene = metadata["scene"]
            wav, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
            channels = [
                channel
                for channel in scene["channel_map"]
                if channel.get("distance_m") is not None
                and 0 <= int(channel.get("channel", -1)) < wav.shape[1]
            ]
        except (KeyError, OSError, RuntimeError, json.JSONDecodeError):
            continue
        if not channels:
            continue

        channel = rng.choice(channels)
        channel_index = int(channel["channel"])
        response = wav[:, channel_index]
        anchor = direct_sample(response)
        late_field = analyze_multiband_late_field(
            response,
            int(sample_rate),
            direct_index=anchor,
            centers_hz=centers_hz,
            base_window_ms=base_window_ms,
            minimum_window_cycles=minimum_window_cycles,
            hop_ms=hop_ms,
            threshold=threshold,
            minimum_sustain_ms=minimum_sustain_ms,
            probe_times_ms=PROBE_TIMES_MS,
        )
        distance_m = float(channel["distance_m"])
        rows.append(
            {
                "item": metadata_path.stem,
                "room_id": scene.get("room_id", metadata_path.parent.name),
                "room_type": scene.get("room_type"),
                "origin": scene.get("origin"),
                "channel": channel_index,
                "label": channel.get("label"),
                "distance_m": distance_m,
                "distance_bucket": distance_bucket(distance_m),
                "sample_rate": int(sample_rate),
                "direct_sample": int(anchor),
                "late_field": late_field,
            }
        )
    return rows, len(pairs)


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    centers_hz: Iterable[float],
    minimum_decay_r_squared: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {"channels": len(rows), "bands": {}}
    for center_hz in centers_hz:
        key = f"{float(center_hz):g}"
        band_rows = [
            row["late_field"]["bands"][key]
            for row in rows
            if key in row["late_field"]["bands"]
        ]
        if not band_rows:
            continue
        mixing_times = [
            band["echo_density"]["mixing_time_s"] for band in band_rows
        ]
        qualified_decay = {
            name: [
                _qualified_decay(band, name, minimum_decay_r_squared)
                for band in band_rows
            ]
            for name in ("edt", "t20", "t30")
        }
        summary["bands"][key] = {
            "center_hz": float(center_hz),
            "channels": len(band_rows),
            "analysis_window_ms": float(band_rows[0]["analysis_window_ms"]),
            "noise_intersection_fraction": sum(
                band["echo_density_truncated_at_noise_intersection"]
                for band in band_rows
            )
            / len(band_rows),
            "mixing_time_coverage": sum(value is not None for value in mixing_times)
            / len(band_rows),
            "mixing_time_s": distribution(mixing_times),
            "late_median_normalized_density": distribution(
                band["echo_density"]["late_median_normalized_density"]
                for band in band_rows
            ),
            "normalized_density_at_ms": {
                f"{probe:g}": distribution(
                    band["echo_density"]["normalized_density_at_ms"].get(
                        f"{probe:g}"
                    )
                    for band in band_rows
                )
                for probe in PROBE_TIMES_MS
            },
            "decay": {
                name: {
                    "valid_fit_fraction": sum(
                        value is not None for value in qualified_decay[name]
                    )
                    / len(band_rows),
                    "rt60_s": distribution(qualified_decay[name]),
                }
                for name in ("edt", "t20", "t30")
            },
        }
    return summary


def summarize_bank(
    rows: list[dict[str, Any]],
    *,
    centers_hz: Iterable[float],
    minimum_decay_r_squared: float,
) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["distance_bucket"]].append(row)
    return {
        "overall": summarize_rows(
            rows,
            centers_hz=centers_hz,
            minimum_decay_r_squared=minimum_decay_r_squared,
        ),
        "by_distance": {
            bucket: summarize_rows(
                bucket_rows,
                centers_hz=centers_hz,
                minimum_decay_r_squared=minimum_decay_r_squared,
            )
            for bucket, bucket_rows in sorted(grouped.items())
        },
    }


def _band_metric_distributions(band: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = {
        "mixing_time_s": band["mixing_time_s"],
        "late_median_normalized_density": band[
            "late_median_normalized_density"
        ],
        "t20_s": band["decay"]["t20"]["rt60_s"],
        "t30_s": band["decay"]["t30"]["rt60_s"],
    }
    metrics.update(
        {
            f"normalized_density_at_{probe}ms": value
            for probe, value in band["normalized_density_at_ms"].items()
        }
    )
    return metrics


def compare_to_reference(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    checks: dict[str, Any] = {}
    for band_key, reference_band in reference["bands"].items():
        candidate_band = candidate["bands"].get(band_key)
        if candidate_band is None:
            continue
        reference_metrics = _band_metric_distributions(reference_band)
        candidate_metrics = _band_metric_distributions(candidate_band)
        for metric, reference_distribution in reference_metrics.items():
            candidate_median = candidate_metrics[metric]["median"]
            lower = reference_distribution["p10"]
            upper = reference_distribution["p90"]
            evaluable = (
                candidate_median is not None
                and lower is not None
                and upper is not None
            )
            check_key = f"{band_key}Hz/{metric}"
            checks[check_key] = {
                "candidate_median": candidate_median,
                "reference_p10": lower,
                "reference_p90": upper,
                "inside_reference_central_80pct": (
                    bool(lower <= candidate_median <= upper) if evaluable else None
                ),
            }
    evaluated = [
        check["inside_reference_central_80pct"]
        for check in checks.values()
        if check["inside_reference_central_80pct"] is not None
    ]
    return {
        "policy": "multiband_candidate_median_inside_measured_central_80pct.v1",
        "checks": checks,
        "evaluated_checks": len(evaluated),
        "passed_checks": sum(evaluated),
        "all_evaluated_checks_passed": bool(evaluated) and all(evaluated),
        "scope_note": (
            "Diagnostic monophonic multiband checks only; the spatial and "
            "perceptual M4 gates remain independent."
        ),
    }


def build_report(
    bank_specs: list[tuple[str, Path]],
    *,
    per_bank: int,
    seed: int,
    reference_tag: str | None,
    centers_hz: Iterable[float],
    minimum_decay_r_squared: float,
    base_window_ms: float,
    minimum_window_cycles: float,
    hop_ms: float,
    threshold: float,
    minimum_sustain_ms: float,
) -> dict[str, Any]:
    centers = tuple(float(center) for center in centers_hz)
    report: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "M4.2",
        "metric_policy": MULTIBAND_LATE_FIELD_POLICY,
        "sampling": {
            "seed": int(seed),
            "channels_per_bank": int(per_bank),
            "maximum_channels_per_item": 1,
            "channel_semantics": "different_sources_to_one_receiver",
        },
        "config": {
            "octave_centers_hz": list(centers),
            "minimum_decay_fit_r_squared": float(minimum_decay_r_squared),
            "base_window_ms": float(base_window_ms),
            "minimum_window_cycles": float(minimum_window_cycles),
            "hop_ms": float(hop_ms),
            "threshold": float(threshold),
            "minimum_sustain_ms": float(minimum_sustain_ms),
            "probe_times_ms": list(PROBE_TIMES_MS),
        },
        "spatial_contract": {
            "evaluable_from_these_banks": False,
            "reason": (
                "Each WAV channel is a different source observed by one receiver; "
                "cross-channel IACC/coherence would not represent a microphone pair."
            ),
            "required_input": (
                "Matched RIRs from one source to at least two receivers with "
                "serialized receiver positions and channel semantics."
            ),
            "binaural_policy": IACC_POLICY,
            "array_policy": DIFFUSE_FIELD_COHERENCE_POLICY,
        },
        "reference_tag": reference_tag,
        "banks": {},
        "reference_comparisons": {},
    }
    for tag, bank_root in bank_specs:
        rows, candidates = sample_bank(
            bank_root,
            count=per_bank,
            rng=random.Random(f"{seed}:{tag}"),
            centers_hz=centers,
            base_window_ms=base_window_ms,
            minimum_window_cycles=minimum_window_cycles,
            hop_ms=hop_ms,
            threshold=threshold,
            minimum_sustain_ms=minimum_sustain_ms,
        )
        if not rows:
            raise ValueError(f"no valid RIR channels found under {bank_root}")
        report["banks"][tag] = {
            "path": str(bank_root.resolve()),
            "metadata_candidates": candidates,
            "sampled_channels": len(rows),
            "summary": summarize_bank(
                rows,
                centers_hz=centers,
                minimum_decay_r_squared=minimum_decay_r_squared,
            ),
            "channels": rows,
        }

    if reference_tag is not None:
        if reference_tag not in report["banks"]:
            raise ValueError(f"reference tag not found: {reference_tag}")
        reference = report["banks"][reference_tag]["summary"]["overall"]
        report["reference_comparisons"] = {
            tag: compare_to_reference(bank["summary"]["overall"], reference)
            for tag, bank in report["banks"].items()
            if tag != reference_tag
        }
    return report


def _format(value: float | None, scale: float = 1.0) -> str:
    return "n/a" if value is None else f"{scale * float(value):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("banks", nargs="+", metavar="TAG=PATH")
    parser.add_argument("--per-bank", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260731)
    parser.add_argument("--reference-tag")
    parser.add_argument(
        "--octave-centers-hz",
        nargs="+",
        type=float,
        default=list(M4_LATE_FIELD_CENTERS_HZ),
    )
    parser.add_argument("--min-decay-r2", type=float, default=0.9)
    parser.add_argument("--base-window-ms", type=float, default=20.0)
    parser.add_argument("--minimum-window-cycles", type=float, default=4.0)
    parser.add_argument("--hop-ms", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--minimum-sustain-ms", type=float, default=10.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.per_bank <= 0:
        parser.error("--per-bank must be positive")
    if not 0.0 <= args.min_decay_r2 <= 1.0:
        parser.error("--min-decay-r2 must be between zero and one")

    specs: list[tuple[str, Path]] = []
    tags: set[str] = set()
    for spec in args.banks:
        tag, separator, path = spec.partition("=")
        if not separator or not tag or not path:
            parser.error(f"bad bank spec (want TAG=PATH): {spec}")
        if tag in tags:
            parser.error(f"duplicate bank tag: {tag}")
        tags.add(tag)
        specs.append((tag, Path(path)))
    try:
        report = build_report(
            specs,
            per_bank=args.per_bank,
            seed=args.seed,
            reference_tag=args.reference_tag,
            centers_hz=args.octave_centers_hz,
            minimum_decay_r_squared=args.min_decay_r2,
            base_window_ms=args.base_window_ms,
            minimum_window_cycles=args.minimum_window_cycles,
            hop_ms=args.hop_ms,
            threshold=args.threshold,
            minimum_sustain_ms=args.minimum_sustain_ms,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print("bank\tband_hz\tn\tmix%\tmix_ms\tlate_NED\tT20_s\tT30_s\tnoise%")
    for tag, bank in report["banks"].items():
        for key, band in bank["summary"]["overall"]["bands"].items():
            print(
                f"{tag}\t{key}\t{band['channels']}"
                f"\t{100.0 * band['mixing_time_coverage']:.1f}"
                f"\t{_format(band['mixing_time_s']['median'], 1000.0)}"
                f"\t{_format(band['late_median_normalized_density']['median'])}"
                f"\t{_format(band['decay']['t20']['rt60_s']['median'])}"
                f"\t{_format(band['decay']['t30']['rt60_s']['median'])}"
                f"\t{100.0 * band['noise_intersection_fraction']:.1f}"
            )
    for tag, comparison in report["reference_comparisons"].items():
        print(
            f"# {tag} vs {report['reference_tag']}: "
            f"{comparison['passed_checks']}/{comparison['evaluated_checks']} "
            "multiband medians inside measured central-80% envelopes"
        )
    print("# spatial: not evaluated; bank channels are sources, not receivers")

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"# wrote {args.json_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
