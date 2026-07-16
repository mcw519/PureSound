#!/usr/bin/env python
"""Build curriculum-ready RIR-bank views using RT60 and measured DRR separation.

The generated folders contain relative symlinks only: no RIR audio is copied.
Each output has an ``items/`` child holding same-stem ``.wav`` and ``.json``
pairs, which ``PreGeneratedRoomBank`` already indexes.

Levels are cumulative for curriculum training:

* ``core``: RT60 0.20--0.45 s and worst-case near/far DRR gap >= 6 dB.
* ``expand``: RT60 0.20--0.65 s and worst-case near/far DRR gap >= 3 dB.
* ``wide``: RT60 0.20--0.85 s and worst-case near/far DRR gap >= 3 dB (the
  dpcrn_wide_antisup deployment domain; re-added here after originally being
  built from an uncommitted variant of this script).
* ``stress``: valid items in none of core/expand/wide (extreme RT60 and/or
  weak DRR gap).
* ``all``: every valid item, no filtering -- the view to use for banks whose
  whole point is unfiltered coverage (e.g. the boundary-distance bank, where
  low DRR gap is desired, or the high-reverb bank probed above rt60 0.85).

The conservative DRR gap is ``min(DRR_near) - max(DRR_far)``. Thus every
near/far channel pair in a core item meets the printed lower bound.

Examples:
  uv run python egs/rir_generation/filter_rir_levels.py \
      egs/rir_generation/exp/hybrid_rir_16k \
      egs/rir_generation/exp/hybrid_rir_16k_levels --dry-run

  uv run python egs/rir_generation/filter_rir_levels.py \
      egs/rir_generation/exp/hybrid_rir_16k \
      egs/rir_generation/exp/hybrid_rir_16k_levels
"""
from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class Item:
    wav: Path
    meta: Path
    rt60: float
    near_drr_min: float
    far_drr_max: float

    @property
    def drr_gap(self) -> float:
        return self.near_drr_min - self.far_drr_max


def compute_drr_db(rir: np.ndarray, sample_rate: int, window_ms: float) -> float:
    peak = int(np.argmax(np.abs(rir)))
    end = min(peak + max(1, round(window_ms * 1e-3 * sample_rate)), rir.size)
    direct = float(np.square(rir[peak:end], dtype=np.float64).sum())
    tail = float(np.square(rir[end:], dtype=np.float64).sum())
    if tail <= 0.0:
        return float("inf")
    return 10.0 * math.log10(max(direct, 1e-12) / tail)


def inspect_item(pair: tuple[Path, float]) -> Item | None:
    meta_path, window_ms = pair
    wav_path = meta_path.with_suffix(".wav")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        scene = meta["scene"]
        channel_map = scene["channel_map"]
        wav, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
    except (KeyError, OSError, json.JSONDecodeError, RuntimeError):
        return None

    near, far = [], []
    for channel in channel_map:
        index = int(channel["channel"])
        label = str(channel.get("label", ""))
        if not 0 <= index < wav.shape[1]:
            return None
        drr = compute_drr_db(wav[:, index], sample_rate, window_ms)
        if label.startswith("near"):
            near.append(drr)
        elif label.startswith("far"):
            far.append(drr)
    if not near or not far:
        return None
    return Item(
        wav=wav_path,
        meta=meta_path,
        rt60=float(scene["rt60"]),
        near_drr_min=min(near),
        far_drr_max=max(far),
    )


def percentiles(values: Iterable[float]) -> dict[str, float | None]:
    a = np.asarray(list(values), dtype=np.float64)
    if not a.size:
        return {key: None for key in ("min", "p10", "p25", "p50", "p75", "p90", "max")}
    return {
        "min": float(a.min()), "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)), "p50": float(np.percentile(a, 50)),
        "p75": float(np.percentile(a, 75)), "p90": float(np.percentile(a, 90)),
        "max": float(a.max()),
    }


def in_range(value: float, low: float, high: float) -> bool:
    return low <= value <= high


def memberships(item: Item) -> tuple[str, ...]:
    labels = []
    if in_range(item.rt60, 0.20, 0.85) and item.drr_gap >= 3.0:
        labels.append("wide")
    if in_range(item.rt60, 0.20, 0.65) and item.drr_gap >= 3.0:
        labels.append("expand")
    if in_range(item.rt60, 0.20, 0.45) and item.drr_gap >= 6.0:
        labels.append("core")
    if not labels:
        labels.append("stress")
    labels.append("all")
    return tuple(labels)


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(target)


def build_level(root: Path, name: str, items: list[Item]) -> None:
    level = root / name
    item_dir = level / "items"
    item_dir.mkdir(parents=True)
    manifest = []
    for item in items:
        relative_symlink(item.wav, item_dir / item.wav.name)
        relative_symlink(item.meta, item_dir / item.meta.name)
        manifest.append({
            "id": item.wav.stem,
            "source_wav": str(item.wav),
            "rt60_s": item.rt60,
            "near_drr_min_db": item.near_drr_min,
            "far_drr_max_db": item.far_drr_max,
            "drr_gap_db": item.drr_gap,
        })
    (level / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("source", type=Path, help="existing hybrid RIR bank")
    parser.add_argument("output", type=Path, help="new parent directory for level folders")
    parser.add_argument("--drr-window-ms", type=float, default=2.5,
                        help="direct-path DRR window; match recipe config (default: 2.5)")
    parser.add_argument("--workers", type=int, default=8,
                        help="parallel WAV readers (default: 8)")
    parser.add_argument("--limit", type=int, default=None,
                        help="inspect only the first N items (diagnostic/testing only)")
    parser.add_argument("--dry-run", action="store_true", help="measure and report without creating symlinks")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if not source.is_dir():
        raise SystemExit(f"source bank not found: {source}")
    metadata = sorted(path for path in source.glob("*/*.json") if path.with_suffix(".wav").exists())
    if args.limit is not None:
        metadata = metadata[:max(0, args.limit)]
    if not metadata:
        raise SystemExit(f"no same-stem WAV/JSON items found under {source}")

    print(f"scanning {len(metadata)} RIR items with {args.workers} reader(s)...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        inspected = []
        for index, item in enumerate(
            pool.map(inspect_item, ((path, args.drr_window_ms) for path in metadata)),
            start=1,
        ):
            inspected.append(item)
            if index % 1000 == 0 or index == len(metadata):
                print(f"  inspected {index}/{len(metadata)}", flush=True)
    items = [item for item in inspected if item is not None]
    if not items:
        raise SystemExit("no valid RIR items with both near_* and far_* channels")

    levels = {name: [] for name in ("core", "expand", "wide", "stress", "all")}
    for item in items:
        for name in memberships(item):
            levels[name].append(item)
    summary = {
        "source": str(source),
        "drr_window_ms": args.drr_window_ms,
        "valid_items": len(items),
        "thresholds": {
            "core": "0.20 <= RT60 <= 0.45 and worst-case DRR gap >= 6 dB",
            "expand": "0.20 <= RT60 <= 0.65 and worst-case DRR gap >= 3 dB",
            "wide": "0.20 <= RT60 <= 0.85 and worst-case DRR gap >= 3 dB",
            "stress": "valid items in none of core/expand/wide",
            "all": "every valid RIR item, no filtering",
        },
        "all": {
            "rt60_s": percentiles(item.rt60 for item in items),
            "drr_gap_db": percentiles(item.drr_gap for item in items),
        },
        "levels": {
            name: {
                "items": len(level_items),
                "rt60_s": percentiles(item.rt60 for item in level_items),
                "drr_gap_db": percentiles(item.drr_gap for item in level_items),
            }
            for name, level_items in levels.items()
        },
    }
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        return
    if output.exists():
        raise SystemExit(f"output already exists: {output} (use a new path; no overwrite option by design)")
    output.mkdir(parents=True)
    for name, level_items in levels.items():
        build_level(output, name, level_items)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"\ncreated symlink-only RIR views under {output}")


if __name__ == "__main__":
    main()
