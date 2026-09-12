#!/usr/bin/env python
"""Compare low-frequency modal peak and Q distributions across RIR banks.

This complements ``compare_bank_acoustics.py``: broadband decay, DRR, and
spectral tilt can agree while the low-frequency resonances are still too
regular, too sharp, or too damped. Each sampled channel is gated after its
direct arrival, then analyzed with the same peak/Q estimator.

Usage:
  uv run python egs/rir_generation/compare_modal_acoustics.py \
      synthetic=egs/rir_generation/exp/rir_realism/m1/hybrid_rir \\
      measured=/path/to/puresound_exp/real_rir_16k_train_view --per-bank 100
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.stats import wasserstein_distance

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.physics.wave.low_frequency import estimate_low_frequency_modes


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


def _finite_percentiles(values: list[float]) -> dict[str, float] | None:
    finite = np.asarray(
        [float(value) for value in values if np.isfinite(value)],
        dtype=np.float64,
    )
    if finite.size == 0:
        return None
    percentiles = np.percentile(finite, [10.0, 25.0, 50.0, 75.0, 90.0])
    return {
        name: float(value)
        for name, value in zip(
            ("p10", "p25", "median", "p75", "p90"),
            percentiles,
        )
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    peaks = [peak for row in rows for peak in row["modal_analysis"]["peaks"]]
    q_values = [
        float(peak["q_factor"])
        for peak in peaks
        if peak["q_factor"] is not None
    ]
    spacing_hz = [
        float(second["frequency_hz"] - first["frequency_hz"])
        for row in rows
        for first, second in zip(
            row["modal_analysis"]["peaks"],
            row["modal_analysis"]["peaks"][1:],
        )
    ]
    return {
        "channels": len(rows),
        "total_peaks": len(peaks),
        "resolved_q_count": len(q_values),
        "resolved_q_fraction": len(q_values) / len(peaks) if peaks else 0.0,
        "channel_peak_count": _finite_percentiles(
            [len(row["modal_analysis"]["peaks"]) for row in rows]
        ),
        "peak_frequency_hz": _finite_percentiles(
            [float(peak["frequency_hz"]) for peak in peaks]
        ),
        "peak_prominence_db": _finite_percentiles(
            [float(peak["prominence_db"]) for peak in peaks]
        ),
        "peak_spacing_hz": _finite_percentiles(spacing_hz),
        "q_factor": _finite_percentiles(q_values),
        "bandwidth_hz": _finite_percentiles(
            [
                float(peak["bandwidth_hz"])
                for peak in peaks
                if peak["bandwidth_hz"] is not None
            ]
        ),
    }


def _analysis_group(
    scene: dict[str, Any],
    metadata_path: Path,
) -> str:
    room_type = scene.get("room_type")
    if room_type:
        return str(room_type)
    stem = metadata_path.stem
    parts = stem.split("_")
    if stem.startswith("diffrir_") and len(parts) >= 2:
        return "_".join(parts[:2])
    return parts[0] if parts else "unknown"


def _distribution_vectors(
    rows: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    peaks = [peak for row in rows for peak in row["modal_analysis"]["peaks"]]
    return {
        "channel_peak_count": np.asarray(
            [len(row["modal_analysis"]["peaks"]) for row in rows],
            dtype=np.float64,
        ),
        "peak_frequency_hz": np.asarray(
            [float(peak["frequency_hz"]) for peak in peaks],
            dtype=np.float64,
        ),
        "peak_prominence_db": np.asarray(
            [float(peak["prominence_db"]) for peak in peaks],
            dtype=np.float64,
        ),
        "peak_spacing_hz": np.asarray(
            [
                float(second["frequency_hz"] - first["frequency_hz"])
                for row in rows
                for first, second in zip(
                    row["modal_analysis"]["peaks"],
                    row["modal_analysis"]["peaks"][1:],
                )
            ],
            dtype=np.float64,
        ),
        "q_factor": np.asarray(
            [
                float(peak["q_factor"])
                for peak in peaks
                if peak["q_factor"] is not None
            ],
            dtype=np.float64,
        ),
        "bandwidth_hz": np.asarray(
            [
                float(peak["bandwidth_hz"])
                for peak in peaks
                if peak["bandwidth_hz"] is not None
            ],
            dtype=np.float64,
        ),
    }


def distribution_distance(
    rows: list[dict[str, Any]],
    reference_rows: list[dict[str, Any]],
) -> dict[str, dict[str, float] | None]:
    values = _distribution_vectors(rows)
    reference = _distribution_vectors(reference_rows)
    result: dict[str, dict[str, float] | None] = {}
    for name in values:
        current_values = values[name]
        reference_values = reference[name]
        if current_values.size == 0 or reference_values.size == 0:
            result[name] = None
            continue
        reference_q25, reference_q75 = np.percentile(
            reference_values,
            [25.0, 75.0],
        )
        reference_iqr = float(reference_q75 - reference_q25)
        distance = float(
            wasserstein_distance(current_values, reference_values)
        )
        result[name] = {
            "wasserstein": distance,
            "normalized_by_reference_iqr": (
                distance / reference_iqr if reference_iqr > 0.0 else distance
            ),
            "absolute_median_gap": abs(
                float(np.median(current_values))
                - float(np.median(reference_values))
            ),
        }
    return result


def sample_bank_channels(
    bank_root: Path,
    count: int,
    rng: random.Random,
    *,
    min_frequency_hz: float,
    max_frequency_hz: float,
    post_direct_delay_ms: float,
    analysis_duration_s: float,
    min_prominence_db: float,
    min_separation_hz: float,
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
            channel_map = scene["channel_map"]
            wav, sample_rate = sf.read(
                wav_path,
                dtype="float32",
                always_2d=True,
            )
        except (KeyError, OSError, RuntimeError, json.JSONDecodeError):
            continue
        valid_channels = [
            channel
            for channel in channel_map
            if 0 <= int(channel.get("channel", -1)) < wav.shape[1]
        ]
        if not valid_channels:
            continue
        channel = rng.choice(valid_channels)
        channel_index = int(channel["channel"])
        rir = np.asarray(wav[:, channel_index], dtype=np.float64)
        direct_index = int(np.argmax(np.abs(rir)))
        analysis_start_s = (
            direct_index / float(sample_rate)
            + float(post_direct_delay_ms) * 1e-3
        )
        try:
            analysis = estimate_low_frequency_modes(
                rir,
                float(sample_rate),
                min_frequency_hz=min_frequency_hz,
                max_frequency_hz=max_frequency_hz,
                analysis_start_s=analysis_start_s,
                analysis_duration_s=analysis_duration_s,
                min_prominence_db=min_prominence_db,
                min_separation_hz=min_separation_hz,
            )
        except ValueError:
            continue
        rows.append(
            {
                "item": str(metadata_path.relative_to(bank_root)),
                "room_id": scene.get("room_id", metadata_path.stem),
                "origin": scene.get("origin"),
                "analysis_group": _analysis_group(scene, metadata_path),
                "label": channel.get("label"),
                "distance_m": channel.get("distance_m"),
                "channel": channel_index,
                "direct_sample": direct_index,
                "modal_analysis": analysis.to_dict(),
            }
        )
    return rows, len(pairs)


def _median(summary: dict[str, Any], key: str) -> float:
    distribution = summary.get(key)
    return (
        float("nan")
        if distribution is None
        else float(distribution["median"])
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("banks", nargs="+", metavar="TAG=PATH")
    parser.add_argument("--per-bank", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-frequency-hz", type=float, default=35.0)
    parser.add_argument("--max-frequency-hz", type=float, default=300.0)
    parser.add_argument("--post-direct-delay-ms", type=float, default=20.0)
    parser.add_argument("--analysis-duration-s", type=float, default=1.2)
    parser.add_argument("--min-prominence-db", type=float, default=6.0)
    parser.add_argument("--min-separation-hz", type=float, default=3.0)
    parser.add_argument(
        "--reference-tag",
        help="also report distribution distances to this bank tag",
    )
    parser.add_argument("--json-output", type=Path)
    args = parser.parse_args()
    if args.per_bank <= 0:
        raise SystemExit("--per-bank must be positive")

    report: dict[str, Any] = {
        "schema_version": 1,
        "seed": int(args.seed),
        "sampled_channels_per_bank": int(args.per_bank),
        "analysis": {
            "min_frequency_hz": float(args.min_frequency_hz),
            "max_frequency_hz": float(args.max_frequency_hz),
            "post_direct_delay_ms": float(args.post_direct_delay_ms),
            "duration_s": float(args.analysis_duration_s),
            "min_prominence_db": float(args.min_prominence_db),
            "min_separation_hz": float(args.min_separation_hz),
            "q_definition": "peak_frequency / spectral_half_power_bandwidth",
        },
        "banks": {},
    }
    for spec in args.banks:
        tag, separator, raw_path = spec.partition("=")
        if not tag or not separator or not raw_path:
            raise SystemExit(f"bad bank spec (want TAG=PATH): {spec}")
        bank_root = Path(raw_path).resolve()
        rows, candidates = sample_bank_channels(
            bank_root,
            args.per_bank,
            random.Random(args.seed),
            min_frequency_hz=args.min_frequency_hz,
            max_frequency_hz=args.max_frequency_hz,
            post_direct_delay_ms=args.post_direct_delay_ms,
            analysis_duration_s=args.analysis_duration_s,
            min_prominence_db=args.min_prominence_db,
            min_separation_hz=args.min_separation_hz,
        )
        if not rows:
            raise SystemExit(f"no analyzable RIR channels under {bank_root}")
        report["banks"][tag] = {
            "path": str(bank_root),
            "metadata_candidates": candidates,
            "sampled_channels": len(rows),
            "summary": summarize(rows),
            "group_summary": {
                group: summarize(
                    [row for row in rows if row["analysis_group"] == group]
                )
                for group in sorted(
                    {row["analysis_group"] for row in rows}
                )
            },
            "channels": rows,
        }

    if args.reference_tag is not None:
        if args.reference_tag not in report["banks"]:
            raise SystemExit(
                f"--reference-tag={args.reference_tag!r} is not one of "
                f"{sorted(report['banks'])}"
            )
        reference_rows = report["banks"][args.reference_tag]["channels"]
        report["reference_tag"] = args.reference_tag
        for tag, bank in report["banks"].items():
            bank["distance_to_reference"] = distribution_distance(
                bank["channels"],
                reference_rows,
            )

    print("bank\tchannels\tpeaks/ch\tQ\tbandwidth_Hz\tspacing_Hz\tprominence_dB")
    for tag, bank in report["banks"].items():
        summary = bank["summary"]
        print(
            f"{tag}\t{bank['sampled_channels']}"
            f"\t{_median(summary, 'channel_peak_count'):.1f}"
            f"\t{_median(summary, 'q_factor'):.2f}"
            f"\t{_median(summary, 'bandwidth_hz'):.2f}"
            f"\t{_median(summary, 'peak_spacing_hz'):.2f}"
            f"\t{_median(summary, 'peak_prominence_db'):.2f}"
        )
    print("# Q is spectral peak frequency divided by -3 dB bandwidth.")
    print("# The fixed post-direct gate and duration make banks directly comparable.")
    if args.reference_tag is not None:
        print()
        print(
            "bank\tQ_W1/IQR\tspacing_W1/IQR\tbandwidth_W1/IQR"
            "\tpeak_count_W1/IQR"
        )
        for tag, bank in report["banks"].items():
            distances = bank["distance_to_reference"]

            def normalized(name: str) -> float:
                value = distances[name]
                return (
                    float("nan")
                    if value is None
                    else float(value["normalized_by_reference_iqr"])
                )

            print(
                f"{tag}"
                f"\t{normalized('q_factor'):.3f}"
                f"\t{normalized('peak_spacing_hz'):.3f}"
                f"\t{normalized('bandwidth_hz'):.3f}"
                f"\t{normalized('channel_peak_count'):.3f}"
            )
        print(
            f"# W1 distances are normalized by the {args.reference_tag!r} "
            "interquartile range; lower is closer."
        )

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"# wrote {args.json_output}")


if __name__ == "__main__":
    main()
