#!/usr/bin/env python
"""Measure and compare the acoustics of RIR banks, side by side.

`inspect_bank.py` reports what the metadata CLAIMS; this measures what the WAVs
actually contain, so simulated and measured banks can be compared on the same
footing before a bank goes into training. Per sampled channel:

    DRR    direct(2.5 ms)-to-reverberant ratio (the training pipeline's cue)
    C50    early(50 ms)/late energy ratio (the 'early' target boundary)
    T30    RT60 from a Schroeder-integration linear fit over [-5, -35] dB
    tilt   RIR magnitude slope 200 Hz - 4 kHz, dB/octave (coloration proxy)

Buckets by distance and prints one block per bank, so a distribution shift
between banks (or between a bank and the measured-RIR bank built from real
rooms) is visible as a table instead of an opinion.

Usage:
  uv run python egs/rir_generation/compare_bank_acoustics.py \\
      synthetic=egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_levels/wide \\
      measured=/work/any_exp_link/puresound_exp/real_rir_16k_train_view/all --per-bank 300
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir_metrics import (  # noqa: E402
    DEFAULT_OCTAVE_CENTERS_HZ,
    analyze_rir,
)

BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 6.0), (6.0, 99.0)]


def bucket_label(d: float) -> str:
    for lo, hi in BUCKETS:
        if lo <= d < hi:
            return f"{lo:g}-{hi:g}m" if hi < 99 else f"{lo:g}m+"
    return "?"


def finite_median(values) -> float | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return float(np.median(finite)) if finite else None


def qualified_decay(
    row: dict,
    name: str,
    min_r_squared: float,
) -> float | None:
    estimate = row["metrics"].get(name)
    if estimate is None or float(estimate["r_squared"]) < min_r_squared:
        return None
    return float(estimate["rt60_s"])


def json_summary(rows: list[dict], min_decay_r_squared: float) -> dict:
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_bucket[row["bucket"]].append(row)
    summary = {}
    for bucket, bucket_rows in sorted(by_bucket.items()):
        qualified = {
            name: [
                value
                for row in bucket_rows
                if (
                    value := qualified_decay(
                        row,
                        name=name,
                        min_r_squared=min_decay_r_squared,
                    )
                )
                is not None
            ]
            for name in ("edt", "t20", "t30")
        }
        summary[bucket] = {
            "channels": len(bucket_rows),
            "noise_floor_corrections": sum(
                bool(row["metrics"]["noise_floor"]["correction_applied"])
                for row in bucket_rows
            ),
            "noise_floor_correction_fraction": sum(
                bool(row["metrics"]["noise_floor"]["correction_applied"])
                for row in bucket_rows
            )
            / len(bucket_rows),
            "valid_decay_fits": {
                name: len(values) for name, values in qualified.items()
            },
            "valid_decay_fraction": {
                name: len(values) / len(bucket_rows)
                for name, values in qualified.items()
            },
            "median": {
                "drr_db": finite_median(row["metrics"]["drr_db"] for row in bucket_rows),
                "c50_db": finite_median(row["metrics"]["c50_db"] for row in bucket_rows),
                "c80_db": finite_median(row["metrics"]["c80_db"] for row in bucket_rows),
                "edt_s": finite_median(qualified["edt"]),
                "t20_s": finite_median(qualified["t20"]),
                "t30_s": finite_median(qualified["t30"]),
                "spectral_tilt_db_per_octave": finite_median(
                    row["metrics"]["spectral_tilt_db_per_octave"]
                    for row in bucket_rows
                ),
                "rt60_metadata_s": finite_median(
                    row["rt60_metadata_s"] for row in bucket_rows
                ),
                "decay_dynamic_range_db": finite_median(
                    row["metrics"]["noise_floor"]["dynamic_range_db"]
                    for row in bucket_rows
                ),
                "noise_intersection_s": finite_median(
                    row["metrics"]["noise_floor"]["intersection_time_s"]
                    for row in bucket_rows
                    if row["metrics"]["noise_floor"]["correction_applied"]
                ),
            },
        }
    return summary


def _paired_wav(metadata_path: Path) -> Path:
    if metadata_path.name == "metadata.json":
        return metadata_path.with_name("rir_5ch.wav")
    return metadata_path.with_suffix(".wav")


def discover_items(bank_root: Path) -> list[tuple[Path, Path]]:
    """List candidate metadata pairs without stat-ing every WAV in a large bank."""
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


def sample_bank_channels(
    bank_root: Path,
    count: int,
    rng: random.Random,
    direct_window_ms: float,
    octave_bands: bool,
) -> tuple[list[dict], int]:
    """Sample at most one channel per item without parsing the whole bank."""
    pairs = discover_items(bank_root)
    rng.shuffle(pairs)
    rows: list[dict] = []
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
            if channel.get("distance_m") is not None
            and 0 <= int(channel.get("channel", -1)) < wav.shape[1]
        ]
        if not valid_channels:
            continue
        channel = rng.choice(valid_channels)
        channel_index = int(channel["channel"])
        distance = float(channel["distance_m"])
        metrics = analyze_rir(
            wav[:, channel_index],
            int(sample_rate),
            direct_window_ms=direct_window_ms,
            octave_centers_hz=(
                DEFAULT_OCTAVE_CENTERS_HZ if octave_bands else None
            ),
        )
        rows.append(
            {
                "room_id": scene.get("room_id", metadata_path.stem),
                "origin": scene.get("origin"),
                "label": channel.get("label"),
                "distance_m": distance,
                "bucket": bucket_label(distance),
                "rt60_metadata_s": scene.get("rt60"),
                "metrics": metrics,
            }
        )
    return rows, len(pairs)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("banks", nargs="+", metavar="TAG=PATH",
                    help="banks/views to compare, as tag=path")
    ap.add_argument("--per-bank", type=int, default=300, help="channels sampled per bank")
    ap.add_argument("--drr-window-ms", type=float, default=2.5)
    ap.add_argument(
        "--min-decay-r2",
        type=float,
        default=0.9,
        help="minimum decay-fit R-squared included in bucket medians (default: 0.9)",
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--octave-bands",
        action="store_true",
        help="also compute valid nominal octave bands from 63 Hz through 8 kHz",
    )
    ap.add_argument(
        "--json-output",
        type=Path,
        help="write deterministic per-channel metrics and bucket summaries to JSON",
    )
    args = ap.parse_args()
    if args.per_bank <= 0:
        raise SystemExit("--per-bank must be positive")
    if not 0.0 <= args.min_decay_r2 <= 1.0:
        raise SystemExit("--min-decay-r2 must be between 0 and 1")
    rng = random.Random(args.seed)

    stats: dict[str, dict[str, list[dict]]] = {}
    report = {
        "schema_version": 1,
        "seed": int(args.seed),
        "sampled_channels_per_bank": int(args.per_bank),
        "drr_window_ms": float(args.drr_window_ms),
        "min_decay_fit_r_squared": float(args.min_decay_r2),
        "octave_bands_hz": (
            list(DEFAULT_OCTAVE_CENTERS_HZ) if args.octave_bands else None
        ),
        "banks": {},
    }
    for spec in args.banks:
        tag, _, path = spec.partition("=")
        if not tag or not path:
            raise SystemExit(f"bad bank spec (want TAG=PATH): {spec}")
        bank_root = Path(path).resolve()
        try:
            rows, discovered_items = sample_bank_channels(
                bank_root=bank_root,
                count=args.per_bank,
                rng=rng,
                direct_window_ms=args.drr_window_ms,
                octave_bands=args.octave_bands,
            )
        except FileNotFoundError as exc:
            raise SystemExit(str(exc)) from exc
        if not rows:
            raise SystemExit(f"no valid RIR channels found under {bank_root}")
        by_bucket: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            by_bucket[r["bucket"]].append(r)
        stats[tag] = by_bucket
        report["banks"][tag] = {
            "path": str(bank_root),
            "metadata_candidates": discovered_items,
            "sampled_channels": len(rows),
            "summary": json_summary(rows, min_decay_r_squared=args.min_decay_r2),
            "channels": rows,
        }
        print(
            f"# {tag}: {len(rows)} channels from {path} "
            f"({discovered_items} metadata candidates)"
        )

    def med(vals):
        value = finite_median(vals)
        return float("nan") if value is None else value

    print()
    header = (
        "bank\tbucket\tn\tDRR_dB\tC50_dB\tT30_s\tT30_fit%"
        "\tnoise%\tdyn_dB\ttilt_dB/oct\trt60_meta"
    )
    print(header)
    for tag, by_bucket in stats.items():
        for lo, hi in BUCKETS:
            b = f"{lo:g}-{hi:g}m" if hi < 99 else f"{lo:g}m+"
            rs = by_bucket.get(b, [])
            if not rs:
                continue
            rt = med([r["rt60_metadata_s"] for r in rs])
            qualified_t30 = [
                value
                for row in rs
                if (
                    value := qualified_decay(
                        row,
                        name="t30",
                        min_r_squared=args.min_decay_r2,
                    )
                )
                is not None
            ]
            valid_t30_percent = 100.0 * len(qualified_t30) / len(rs)
            corrected_noise_percent = 100.0 * sum(
                bool(row["metrics"]["noise_floor"]["correction_applied"])
                for row in rs
            ) / len(rs)
            print(
                f"{tag}\t{b}\t{len(rs)}"
                f"\t{med([r['metrics']['drr_db'] for r in rs]):6.2f}"
                f"\t{med([r['metrics']['c50_db'] for r in rs]):6.2f}"
                f"\t{med(qualified_t30):5.2f}"
                f"\t{valid_t30_percent:6.1f}"
                f"\t{corrected_noise_percent:6.1f}"
                f"\t{med([r['metrics']['noise_floor']['dynamic_range_db'] for r in rs]):6.1f}"
                f"\t{med([r['metrics']['spectral_tilt_db_per_octave'] for r in rs]):6.2f}"
                f"\t{rt:5.2f}"
            )
    print(
        f"\n# Decay medians include only fits with R^2 >= {args.min_decay_r2:g}; "
        "T30_fit% reports coverage."
    )
    print("# noise% is the fraction using Lundeby-style intersection/truncation;")
    print("# dyn_dB is the median direct-block to stationary-noise dynamic range.")
    print("# T30 far above rt60_meta = the metadata undersells the tail (or the fit failed);")
    print("# a DRR/C50/tilt shift between banks at the same distance is a domain shift the")
    print("# training pipeline will inherit.")
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"# wrote {args.json_output}")


if __name__ == "__main__":
    main()
