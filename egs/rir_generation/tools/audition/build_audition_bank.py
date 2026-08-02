#!/usr/bin/env python
"""Validate an M1 RIR bank and create deterministic dry/wet audition previews."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve, resample_poly


SCHEMA_VERSION = "puresound.rir_audition.v1"
EXPECTED_LABELS = ("near_0", "near_1", "far_0", "far_1", "far_2")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--dry-wav",
        type=Path,
        action="append",
        required=True,
        help="Dry speech/audio fixture. Repeat for multiple preview sources.",
    )
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--expected-channels", type=int, default=5)
    parser.add_argument("--min-items", type=int, default=50)
    parser.add_argument("--num-previews", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--min-near-far-drr-gap-db", type=float, default=6.0)
    parser.add_argument("--min-valid-t20-fraction", type=float, default=0.65)
    parser.add_argument("--min-decay-r2", type=float, default=0.9)
    parser.add_argument("--min-tail-energy-fraction", type=float, default=1e-5)
    parser.add_argument("--max-rir-peak", type=float, default=1.0)
    return parser.parse_args()


def _paired_wav(metadata_path: Path) -> Path:
    if metadata_path.name == "metadata.json":
        return metadata_path.with_name("rir_5ch.wav")
    return metadata_path.with_suffix(".wav")


def discover_items(bank_root: Path) -> list[tuple[Path, Path]]:
    metadata_paths = sorted(
        {
            *bank_root.glob("*.json"),
            *bank_root.glob("*/*.json"),
        }
    )
    return [
        (metadata_path, _paired_wav(metadata_path))
        for metadata_path in metadata_paths
        if _paired_wav(metadata_path).is_file()
    ]


def _finite_summary(values: list[float]) -> dict[str, float | None]:
    finite = np.asarray(
        [float(value) for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if finite.size == 0:
        return {
            "minimum": None,
            "p05": None,
            "median": None,
            "p95": None,
            "maximum": None,
        }
    return {
        "minimum": float(np.min(finite)),
        "p05": float(np.quantile(finite, 0.05)),
        "median": float(np.median(finite)),
        "p95": float(np.quantile(finite, 0.95)),
        "maximum": float(np.max(finite)),
    }


def _valid_decay(
    metrics: dict[str, Any],
    name: str,
    minimum_r_squared: float,
) -> float | None:
    estimate = metrics.get(name)
    if not estimate:
        return None
    value = float(estimate.get("rt60_s", float("nan")))
    r_squared = float(estimate.get("r_squared", 0.0))
    if not math.isfinite(value) or r_squared < minimum_r_squared:
        return None
    return value


def inspect_bank(
    bank_root: Path,
    *,
    expected_sample_rate: int,
    expected_channels: int,
    maximum_rir_peak: float,
    minimum_tail_energy_fraction: float,
    minimum_decay_r_squared: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pairs = discover_items(bank_root)
    rows: list[dict[str, Any]] = []
    failure_counts: Counter[str] = Counter()
    channel_drr: dict[str, list[float]] = {"near": [], "far": []}
    decay_values: dict[str, list[float]] = {
        "edt": [],
        "t20": [],
        "t30": [],
    }
    all_peaks: list[float] = []
    all_tail_fractions: list[float] = []
    all_prearrival_peaks: list[float] = []
    total_channels = 0

    for metadata_path, wav_path in pairs:
        item_failures: list[str] = []
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            rir, sample_rate = sf.read(
                wav_path,
                dtype="float32",
                always_2d=True,
            )
        except (OSError, RuntimeError, json.JSONDecodeError) as error:
            failure_counts["unreadable_item"] += 1
            rows.append(
                {
                    "metadata_path": str(metadata_path.resolve()),
                    "wav_path": str(wav_path.resolve()),
                    "failures": ["unreadable_item"],
                    "error": str(error),
                }
            )
            continue

        scene = metadata.get("scene", {})
        channel_map = scene.get("channel_map", [])
        realized = metadata.get("realized_acoustics", {}).get("channels", [])
        if int(sample_rate) != int(expected_sample_rate):
            item_failures.append("unexpected_sample_rate")
        if (
            rir.ndim != 2
            or rir.shape[1] != expected_channels
            or len(channel_map) != expected_channels
        ):
            item_failures.append("unexpected_channel_layout")
        labels = tuple(str(item.get("label")) for item in channel_map)
        if expected_channels == len(EXPECTED_LABELS) and labels != EXPECTED_LABELS:
            item_failures.append("unexpected_channel_labels")
        finite_audio = bool(np.isfinite(rir).all())
        if not finite_audio:
            item_failures.append("non_finite_audio")
        safe_rir = np.nan_to_num(
            rir,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )

        item_peak = float(np.max(np.abs(safe_rir))) if safe_rir.size else 0.0
        all_peaks.append(item_peak)
        if item_peak <= 1e-8:
            item_failures.append("silent_rir")
        if item_peak > float(maximum_rir_peak) + 1e-6:
            item_failures.append("rir_peak_above_limit")

        try:
            sound_speed = float(
                scene.get("environment", {}).get(
                    "sound_speed_m_s",
                    343.0,
                )
            )
        except (TypeError, ValueError):
            sound_speed = float("nan")
        if not math.isfinite(sound_speed) or sound_speed <= 0.0:
            item_failures.append("invalid_sound_speed")
            sound_speed = 343.0
        channel_rows: list[dict[str, Any]] = []
        if (
            safe_rir.ndim == 2
            and safe_rir.shape[1] == expected_channels
            and len(channel_map) == expected_channels
        ):
            realized_by_channel: dict[int, dict[str, Any]] = {}
            for item in realized:
                try:
                    realized_by_channel[int(item.get("channel", -1))] = item
                except (TypeError, ValueError):
                    item_failures.append("invalid_realized_metric_channel")
            for channel in channel_map:
                try:
                    channel_index = int(channel.get("channel", -1))
                except (TypeError, ValueError):
                    item_failures.append("invalid_channel_index")
                    continue
                if not 0 <= channel_index < safe_rir.shape[1]:
                    item_failures.append("invalid_channel_index")
                    continue
                total_channels += 1
                label = str(channel.get("label"))
                try:
                    distance_m = float(
                        channel.get("distance_m", float("nan"))
                    )
                except (TypeError, ValueError):
                    distance_m = float("nan")
                if not math.isfinite(distance_m) or distance_m < 0.0:
                    item_failures.append("invalid_source_receiver_distance")
                    continue
                signal = np.asarray(
                    safe_rir[:, channel_index],
                    dtype=np.float64,
                )
                magnitude = np.abs(signal)
                peak = float(np.max(magnitude))
                first_physical_sample = int(
                    math.floor(
                        max(distance_m, 0.0)
                        / max(sound_speed, 1e-9)
                        * sample_rate
                    )
                )
                prearrival_peak = (
                    float(np.max(magnitude[:first_physical_sample]))
                    if first_physical_sample > 0
                    else 0.0
                )
                causal_tolerance = max(1e-12, peak * 1e-8)
                causal = prearrival_peak <= causal_tolerance
                if not causal:
                    item_failures.append("prearrival_energy")
                all_prearrival_peaks.append(prearrival_peak)

                nonzero = np.flatnonzero(magnitude > max(1e-12, peak * 1e-8))
                onset_sample = int(nonzero[0]) if nonzero.size else None
                tail_start = min(
                    signal.size,
                    first_physical_sample
                    + int(round(0.05 * float(sample_rate))),
                )
                total_energy = float(np.sum(np.square(signal)))
                tail_energy_fraction = float(
                    np.sum(np.square(signal[tail_start:]))
                    / max(total_energy, 1e-30)
                )
                all_tail_fractions.append(tail_energy_fraction)
                if tail_energy_fraction < minimum_tail_energy_fraction:
                    item_failures.append("insufficient_tail_energy")

                metric_record = realized_by_channel.get(channel_index, {})
                metrics = metric_record.get("metrics", {})
                drr = metrics.get("drr_db")
                drr_value = (
                    float(drr)
                    if drr is not None and math.isfinite(float(drr))
                    else None
                )
                if drr_value is not None:
                    group = "near" if label.startswith("near") else "far"
                    channel_drr[group].append(drr_value)
                for decay_name in decay_values:
                    value = _valid_decay(
                        metrics,
                        decay_name,
                        minimum_decay_r_squared,
                    )
                    if value is not None:
                        decay_values[decay_name].append(value)
                channel_rows.append(
                    {
                        "channel": channel_index,
                        "label": label,
                        "distance_m": distance_m,
                        "first_physical_sample": first_physical_sample,
                        "onset_sample": onset_sample,
                        "prearrival_peak_abs": prearrival_peak,
                        "tail_energy_fraction": tail_energy_fraction,
                        "peak_abs": peak,
                        "drr_db": drr_value,
                    }
                )

        unique_failures = sorted(set(item_failures))
        failure_counts.update(unique_failures)
        rows.append(
            {
                "sample_id": metadata.get("sample_id", metadata_path.stem),
                "room_id": metadata.get(
                    "room_id",
                    scene.get("room_id", metadata_path.parent.name),
                ),
                "room_type": scene.get("room_type"),
                "scene_rt60_s": scene.get("rt60"),
                "metadata_path": str(metadata_path.resolve()),
                "wav_path": str(wav_path.resolve()),
                "sample_rate": int(sample_rate),
                "num_samples": int(rir.shape[0]) if rir.ndim == 2 else 0,
                "num_channels": int(rir.shape[1]) if rir.ndim == 2 else 0,
                "peak_abs": item_peak,
                "channels": channel_rows,
                "failures": unique_failures,
            }
        )

    near_median = (
        float(np.median(channel_drr["near"]))
        if channel_drr["near"]
        else None
    )
    far_median = (
        float(np.median(channel_drr["far"]))
        if channel_drr["far"]
        else None
    )
    drr_gap = (
        near_median - far_median
        if near_median is not None and far_median is not None
        else None
    )
    summary = {
        "discovered_items": len(pairs),
        "readable_items": sum("error" not in row for row in rows),
        "channels": total_channels,
        "failure_counts": dict(sorted(failure_counts.items())),
        "rir_peak_abs": _finite_summary(all_peaks),
        "prearrival_peak_abs": _finite_summary(all_prearrival_peaks),
        "tail_energy_fraction": _finite_summary(all_tail_fractions),
        "near_drr_db": _finite_summary(channel_drr["near"]),
        "far_drr_db": _finite_summary(channel_drr["far"]),
        "median_near_far_drr_gap_db": drr_gap,
        "valid_decay_fits": {
            name: len(values) for name, values in decay_values.items()
        },
        "valid_decay_fraction": {
            name: (
                len(values) / total_channels if total_channels else 0.0
            )
            for name, values in decay_values.items()
        },
        "decay_rt60_s": {
            name: _finite_summary(values)
            for name, values in decay_values.items()
        },
        "room_types": dict(
            sorted(
                Counter(
                    str(row["room_type"])
                    for row in rows
                    if row.get("room_type") is not None
                ).items()
            )
        ),
    }
    return rows, summary


def _select_preview_rows(
    rows: list[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    candidates = [
        row
        for row in rows
        if not row.get("failures")
        and row.get("scene_rt60_s") is not None
    ]
    candidates.sort(
        key=lambda row: (
            float(row["scene_rt60_s"]),
            str(row.get("sample_id")),
        )
    )
    if not candidates or count <= 0:
        return []
    target_indices = np.linspace(
        0,
        len(candidates) - 1,
        min(count, len(candidates)),
    )
    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for raw_index in target_indices:
        index = int(round(float(raw_index)))
        if index not in used:
            selected.append(candidates[index])
            used.add(index)
    return selected


def _load_dry_audio(path: Path, sample_rate: int) -> np.ndarray:
    audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
    mono = np.asarray(np.mean(audio, axis=1), dtype=np.float64)
    if int(source_rate) != int(sample_rate):
        divisor = math.gcd(int(source_rate), int(sample_rate))
        mono = resample_poly(
            mono,
            int(sample_rate) // divisor,
            int(source_rate) // divisor,
        )
    peak = float(np.max(np.abs(mono))) if mono.size else 0.0
    if peak <= 1e-12:
        raise ValueError(f"dry WAV is silent: {path}")
    return mono / peak * (10.0 ** (-3.0 / 20.0))


def _write_relative_symlink(source: Path, destination: Path) -> None:
    destination.symlink_to(os.path.relpath(source.resolve(), destination.parent))


def _build_item_view(
    rows: list[dict[str, Any]],
    destination: Path,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for row in rows:
        if row.get("error"):
            continue
        sample_id = str(row["sample_id"])
        _write_relative_symlink(
            Path(row["wav_path"]),
            destination / f"{sample_id}.wav",
        )
        _write_relative_symlink(
            Path(row["metadata_path"]),
            destination / f"{sample_id}.json",
        )


def _build_previews(
    selected_rows: list[dict[str, Any]],
    dry_paths: list[Path],
    output_dir: Path,
    sample_rate: int,
) -> list[dict[str, Any]]:
    dry_dir = output_dir / "dry"
    preview_dir = output_dir / "previews"
    dry_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)
    dry_audio = [
        _load_dry_audio(path, sample_rate) for path in dry_paths
    ]
    saved_dry: dict[int, Path] = {}
    previews: list[dict[str, Any]] = []

    for preview_index, row in enumerate(selected_rows):
        dry_index = preview_index % len(dry_paths)
        dry = dry_audio[dry_index]
        if dry_index not in saved_dry:
            dry_output = dry_dir / (
                f"dry_{dry_index:02d}_{dry_paths[dry_index].stem}.wav"
            )
            sf.write(dry_output, dry, sample_rate, subtype="FLOAT")
            saved_dry[dry_index] = dry_output

        rir, rir_rate = sf.read(
            row["wav_path"],
            dtype="float32",
            always_2d=True,
        )
        if int(rir_rate) != int(sample_rate):
            raise ValueError("validated RIR sample rate changed during preview build")
        channels_by_label = {
            channel["label"]: int(channel["channel"])
            for channel in row["channels"]
        }
        selected_labels = ("near_0", "far_0")
        wet = {
            label: fftconvolve(
                dry,
                np.asarray(rir[:, channels_by_label[label]], dtype=np.float64),
                mode="full",
            )
            for label in selected_labels
        }
        dry_rms = float(np.sqrt(np.mean(np.square(dry))))
        near_rms = float(
            np.sqrt(np.mean(np.square(wet["near_0"])))
        )
        common_gain = dry_rms / max(near_rms, 1e-12)
        wet_peak = max(float(np.max(np.abs(value))) for value in wet.values())
        common_gain = min(common_gain, 0.98 / max(wet_peak, 1e-12))

        preview_id = (
            f"{preview_index:02d}_{row['sample_id']}"
        )
        output_files: dict[str, str] = {}
        for label, signal in wet.items():
            output_path = preview_dir / f"{preview_id}_{label}_full.wav"
            sf.write(
                output_path,
                signal * common_gain,
                sample_rate,
                subtype="FLOAT",
            )
            output_files[label] = str(output_path.relative_to(output_dir))
        previews.append(
            {
                "preview_id": preview_id,
                "sample_id": row["sample_id"],
                "room_id": row["room_id"],
                "room_type": row["room_type"],
                "scene_rt60_s": float(row["scene_rt60_s"]),
                "dry_source": str(dry_paths[dry_index].resolve()),
                "dry_file": str(saved_dry[dry_index].relative_to(output_dir)),
                "rir_file": str(Path(row["wav_path"]).resolve()),
                "wet_files": output_files,
                "shared_wet_gain_db": float(
                    20.0 * math.log10(max(common_gain, 1e-30))
                ),
                "level_policy": (
                    "near_0 RMS matched to dry unless shared anti-clipping "
                    "gain is lower; near/far relative level is preserved"
                ),
            }
        )
    return previews


def _gate(
    summary: dict[str, Any],
    *,
    minimum_items: int,
    minimum_near_far_drr_gap_db: float,
    minimum_valid_t20_fraction: float,
) -> list[dict[str, Any]]:
    checks = [
        {
            "name": "minimum_items",
            "value": int(summary["readable_items"]),
            "threshold": int(minimum_items),
            "operator": ">=",
            "passed": int(summary["readable_items"]) >= int(minimum_items),
        },
        {
            "name": "no_item_failures",
            "value": int(sum(summary["failure_counts"].values())),
            "threshold": 0,
            "operator": "==",
            "passed": not summary["failure_counts"],
        },
        {
            "name": "median_near_far_drr_gap_db",
            "value": summary["median_near_far_drr_gap_db"],
            "threshold": float(minimum_near_far_drr_gap_db),
            "operator": ">=",
            "passed": (
                summary["median_near_far_drr_gap_db"] is not None
                and float(summary["median_near_far_drr_gap_db"])
                >= float(minimum_near_far_drr_gap_db)
            ),
        },
        {
            "name": "valid_t20_fraction",
            "value": float(summary["valid_decay_fraction"]["t20"]),
            "threshold": float(minimum_valid_t20_fraction),
            "operator": ">=",
            "passed": (
                float(summary["valid_decay_fraction"]["t20"])
                >= float(minimum_valid_t20_fraction)
            ),
        },
    ]
    return checks


def _write_readme(
    output_dir: Path,
    source_bank: Path,
    summary: dict[str, Any],
    gates: list[dict[str, Any]],
    previews: list[dict[str, Any]],
) -> None:
    status = "PASS" if all(item["passed"] for item in gates) else "FAIL"
    gap = summary["median_near_far_drr_gap_db"]
    gap_text = "n/a" if gap is None else f"{gap:.2f} dB"
    lines = [
        "# M1 RIR audition bank",
        "",
        f"品質狀態：**{status}**",
        "",
        f"- 來源 bank：`{source_bank.resolve()}`",
        f"- items：{summary['readable_items']}",
        f"- RIR channels：{summary['channels']}",
        f"- near/far median DRR gap：{gap_text}",
        (
            "- valid T20 fraction："
            f"{summary['valid_decay_fraction']['t20']:.1%}"
        ),
        f"- audition previews：{len(previews)} 組",
        "",
        "`items/` 是指向原始 RIR/JSON 的相對 symlink；`dry/` 與",
        "`previews/` 是可直接播放的 FLOAT WAV。每組 preview 使用同一個",
        "wet gain，因此 near/far 相對音量不會被各自 peak normalization",
        "抹掉。完整門檻、分布與逐 item 結果見 `report.json`，播放索引見",
        "`manifest.json`。audition root 可直接交給 `PreGeneratedRoomBank`；",
        "它會索引 `items/` 內的 50 組 RIR。",
        "",
        "## Gates",
        "",
    ]
    lines.extend(
        (
            f"- [{'x' if gate['passed'] else ' '}] {gate['name']}: "
            f"{gate['value']} {gate['operator']} {gate['threshold']}"
        )
        for gate in gates
    )
    (output_dir / "README.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    if not args.bank.is_dir():
        raise SystemExit(f"bank not found: {args.bank}")
    if args.output_dir.exists():
        raise SystemExit(f"output directory already exists: {args.output_dir}")
    if args.min_items < 1 or args.num_previews < 1:
        raise SystemExit("--min-items and --num-previews must be positive")
    for dry_path in args.dry_wav:
        if not dry_path.is_file():
            raise SystemExit(f"dry WAV not found: {dry_path}")

    rows, summary = inspect_bank(
        args.bank,
        expected_sample_rate=args.sample_rate,
        expected_channels=args.expected_channels,
        maximum_rir_peak=args.max_rir_peak,
        minimum_tail_energy_fraction=args.min_tail_energy_fraction,
        minimum_decay_r_squared=args.min_decay_r2,
    )
    gates = _gate(
        summary,
        minimum_items=args.min_items,
        minimum_near_far_drr_gap_db=args.min_near_far_drr_gap_db,
        minimum_valid_t20_fraction=args.min_valid_t20_fraction,
    )
    selected = _select_preview_rows(rows, args.num_previews)

    args.output_dir.mkdir(parents=True)
    _build_item_view(rows, args.output_dir / "items")
    previews = _build_previews(
        selected,
        args.dry_wav,
        args.output_dir,
        args.sample_rate,
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "source_bank": str(args.bank.resolve()),
        "quality_passed": all(gate["passed"] for gate in gates),
        "thresholds": {
            "expected_sample_rate": int(args.sample_rate),
            "expected_channels": int(args.expected_channels),
            "minimum_items": int(args.min_items),
            "maximum_rir_peak": float(args.max_rir_peak),
            "minimum_tail_energy_fraction": float(
                args.min_tail_energy_fraction
            ),
            "minimum_decay_r_squared": float(args.min_decay_r2),
            "minimum_valid_t20_fraction": float(
                args.min_valid_t20_fraction
            ),
            "minimum_near_far_drr_gap_db": float(
                args.min_near_far_drr_gap_db
            ),
        },
        "gates": gates,
        "summary": summary,
        "items": rows,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "source_bank": str(args.bank.resolve()),
        "selection": "scene_rt60_quantiles",
        "seed": int(args.seed),
        "previews": previews,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(
            manifest,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    _write_readme(
        args.output_dir,
        args.bank,
        summary,
        gates,
        previews,
    )

    status = "PASS" if report["quality_passed"] else "FAIL"
    print(
        f"audition bank {status}: {summary['readable_items']} items, "
        f"{summary['channels']} channels, {len(previews)} previews"
    )
    for gate in gates:
        print(
            f"  [{'PASS' if gate['passed'] else 'FAIL'}] "
            f"{gate['name']}: {gate['value']} "
            f"{gate['operator']} {gate['threshold']}"
        )
    return 0 if report["quality_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
