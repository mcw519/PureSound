#!/usr/bin/env python
"""Benchmark late-field echo density against a measured-RIR reference bank.

This is the M4.1 measurement gate. It does not declare a renderer perceptually
realistic by itself; it establishes one reproducible monophonic target before
M4 changes the late-field synthesis algorithm.

Example:
  uv run python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_late_field_baseline.py \
      measured=/path/to/puresound_exp/real_rir_16k_train_view/items \
      m1=egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100 \
      m3=egs/rir_generation/exp/rir_realism/m3/rir_m3_late_baseline \
      --reference-tag measured --per-bank 100 \
      --json-output egs/rir_generation/phases/m4_spatial_late_field/reports/m4_late_field_baseline.json
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

from puresound.audio.rir.metrics import (
    ABEL_ECHO_DENSITY_POLICY,
    analyze_echo_density,
    direct_sample,
)


DISTANCE_BUCKETS = (
    (0.0, 1.0),
    (1.0, 2.0),
    (2.0, 3.5),
    (3.5, 6.0),
    (6.0, float("inf")),
)
PROBE_TIMES_MS = (20.0, 50.0, 100.0, 200.0)


def distance_bucket(distance_m: float) -> str:
    for low, high in DISTANCE_BUCKETS:
        if low <= distance_m < high:
            return f"{low:g}-{high:g}m" if np.isfinite(high) else f"{low:g}m+"
    return "unknown"


def _paired_wav(metadata_path: Path) -> Path:
    if metadata_path.name == "metadata.json":
        return metadata_path.with_name("rir_5ch.wav")
    return metadata_path.with_suffix(".wav")


def discover_items(bank_root: Path) -> list[tuple[Path, Path]]:
    if not bank_root.is_dir():
        raise FileNotFoundError(f"bank not found: {bank_root}")
    metadata_paths = sorted(
        {
            *bank_root.glob("*.json"),
            *bank_root.glob("*/*.json"),
        }
    )
    return [
        (metadata_path, _paired_wav(metadata_path))
        for metadata_path in metadata_paths
    ]


def _distribution(values: Iterable[float | None]) -> dict[str, float | int | None]:
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


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "channels": 0,
            "mixing_time_coverage": 0.0,
            "mixing_time_s": _distribution([]),
            "late_median_normalized_density": _distribution([]),
            "normalized_density_at_ms": {},
        }
    return {
        "channels": len(rows),
        "mixing_time_coverage": sum(
            row["echo_density"]["mixing_time_s"] is not None for row in rows
        )
        / len(rows),
        "mixing_time_s": _distribution(
            row["echo_density"]["mixing_time_s"] for row in rows
        ),
        "late_median_normalized_density": _distribution(
            row["echo_density"]["late_median_normalized_density"] for row in rows
        ),
        "normalized_density_at_ms": {
            f"{probe:g}": _distribution(
                row["echo_density"]["normalized_density_at_ms"].get(f"{probe:g}")
                for row in rows
            )
            for probe in PROBE_TIMES_MS
        },
    }


def sample_bank(
    bank_root: Path,
    *,
    count: int,
    rng: random.Random,
    window_ms: float,
    hop_ms: float,
    threshold: float,
    minimum_sustain_ms: float,
) -> tuple[list[dict[str, Any]], int]:
    """Sample at most one physical source channel from each bank item."""

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
        direct_index = direct_sample(response)
        echo_density = analyze_echo_density(
            response,
            int(sample_rate),
            direct_index=direct_index,
            window_ms=window_ms,
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
                "origin": scene.get("origin"),
                "channel": channel_index,
                "label": channel.get("label"),
                "distance_m": distance_m,
                "distance_bucket": distance_bucket(distance_m),
                "sample_rate": int(sample_rate),
                "direct_sample": int(direct_index),
                "direct_policy": "absolute_peak",
                "echo_density": echo_density,
            }
        )
    return rows, len(pairs)


def summarize_bank(rows: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["distance_bucket"]].append(row)
    return {
        "overall": summarize_rows(rows),
        "by_distance": {
            bucket: summarize_rows(bucket_rows)
            for bucket, bucket_rows in sorted(grouped.items())
        },
    }


def _metric_distributions(summary: dict[str, Any]) -> dict[str, dict[str, Any]]:
    metrics = {
        "mixing_time_s": summary["mixing_time_s"],
        "late_median_normalized_density": summary[
            "late_median_normalized_density"
        ],
    }
    metrics.update(
        {
            f"normalized_density_at_{probe}ms": distribution
            for probe, distribution in summary[
                "normalized_density_at_ms"
            ].items()
        }
    )
    return metrics


def compare_to_reference(
    candidate: dict[str, Any],
    reference: dict[str, Any],
) -> dict[str, Any]:
    """Check candidate medians against the measured central-80% envelope."""

    checks: dict[str, Any] = {}
    candidate_metrics = _metric_distributions(candidate)
    reference_metrics = _metric_distributions(reference)
    for name, reference_distribution in reference_metrics.items():
        candidate_median = candidate_metrics[name]["median"]
        lower = reference_distribution["p10"]
        upper = reference_distribution["p90"]
        evaluable = (
            candidate_median is not None and lower is not None and upper is not None
        )
        checks[name] = {
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
        "policy": "candidate_median_inside_measured_central_80pct.v1",
        "mixing_time_coverage_delta": (
            float(candidate["mixing_time_coverage"])
            - float(reference["mixing_time_coverage"])
        ),
        "checks": checks,
        "evaluated_checks": len(evaluated),
        "passed_checks": sum(evaluated),
        "all_evaluated_checks_passed": bool(evaluated) and all(evaluated),
        "scope_note": (
            "Diagnostic distribution-envelope checks only; passing does not "
            "close M4 spatial or perceptual exit criteria."
        ),
    }


def build_report(
    bank_specs: list[tuple[str, Path]],
    *,
    per_bank: int,
    seed: int,
    reference_tag: str | None,
    window_ms: float,
    hop_ms: float,
    threshold: float,
    minimum_sustain_ms: float,
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "schema_version": 1,
        "milestone": "M4.1",
        "metric_policy": ABEL_ECHO_DENSITY_POLICY,
        "sampling": {
            "seed": int(seed),
            "channels_per_bank": int(per_bank),
            "maximum_channels_per_item": 1,
            "direct_policy": "absolute_peak",
        },
        "echo_density_config": {
            "window_ms": float(window_ms),
            "hop_ms": float(hop_ms),
            "threshold": float(threshold),
            "minimum_sustain_ms": float(minimum_sustain_ms),
            "probe_times_ms": list(PROBE_TIMES_MS),
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
            window_ms=window_ms,
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
            "summary": summarize_bank(rows),
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


def _format_optional(value: float | None, scale: float = 1.0) -> str:
    return "n/a" if value is None else f"{scale * float(value):.3f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("banks", nargs="+", metavar="TAG=PATH")
    parser.add_argument("--per-bank", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--reference-tag")
    parser.add_argument("--window-ms", type=float, default=20.0)
    parser.add_argument("--hop-ms", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--minimum-sustain-ms", type=float, default=10.0)
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.per_bank <= 0:
        parser.error("--per-bank must be positive")

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
            window_ms=args.window_ms,
            hop_ms=args.hop_ms,
            threshold=args.threshold,
            minimum_sustain_ms=args.minimum_sustain_ms,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    print("bank\tchannels\tmixing%\tmixing_ms\tNED20\tNED50\tNED100\tNED200\tlate_NED")
    for tag, bank in report["banks"].items():
        summary = bank["summary"]["overall"]
        probes = summary["normalized_density_at_ms"]
        print(
            f"{tag}\t{summary['channels']}"
            f"\t{100.0 * summary['mixing_time_coverage']:.1f}"
            f"\t{_format_optional(summary['mixing_time_s']['median'], 1000.0)}"
            f"\t{_format_optional(probes['20']['median'])}"
            f"\t{_format_optional(probes['50']['median'])}"
            f"\t{_format_optional(probes['100']['median'])}"
            f"\t{_format_optional(probes['200']['median'])}"
            f"\t{_format_optional(summary['late_median_normalized_density']['median'])}"
        )
    for tag, comparison in report["reference_comparisons"].items():
        print(
            f"# {tag} vs {report['reference_tag']}: "
            f"{comparison['passed_checks']}/{comparison['evaluated_checks']} "
            "candidate medians inside measured central-80% envelopes"
        )

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
