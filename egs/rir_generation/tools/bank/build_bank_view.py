#!/usr/bin/env python
"""Build symlink-only RIR-bank views that ``PreGeneratedRoomBank`` can index.

A view is a directory with an ``items/`` child holding same-stem ``.wav``/``.json``
pairs. Nothing is copied -- every entry is a relative symlink into the source bank --
so views are cheap to create and to keep around.

Two ways to build one:

  levels  Slice ONE generated bank into difficulty levels by RT60 and measured
          near/far DRR separation. Levels are cumulative, so a curriculum can walk
          them in order:
            core    RT60 0.20-0.45 s, worst-case near/far DRR gap >= 6 dB
            expand  RT60 0.20-0.65 s, worst-case near/far DRR gap >= 3 dB
            wide    RT60 0.20-0.85 s, worst-case near/far DRR gap >= 3 dB
            stress  valid items in none of the above (extreme RT60 / weak gap)
            all     every valid item, no filtering -- for banks whose whole point
                    is unfiltered coverage (e.g. a boundary-distance fill, where a
                    low DRR gap is the intent, or a high-reverb bank)
          The DRR gap is conservative: ``min(DRR_near) - max(DRR_far)``, so every
          near/far channel pair in an item meets the printed bound.

  merge   Union SEVERAL existing views into one training view. Banks reuse the same
          ``room_XXXXXX_YYYYYY`` stems (each generation run counts rooms from zero),
          so a naive union would collide; every symlink gets a per-source tag
          prefix (wav and json renamed together) and ``merged.json`` records where
          each group came from.

Examples:
  # measure only, no symlinks written
  uv run python egs/rir_generation/tools/bank/build_bank_view.py levels \\
      egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k \
      egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_levels --dry-run

  # cut the level views
  uv run python egs/rir_generation/tools/bank/build_bank_view.py levels \\
      egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k \
      egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_levels

  # union a base view with a boundary-distance fill
  uv run python egs/rir_generation/tools/bank/build_bank_view.py merge \\
      --source wide=egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_levels/wide \\
      --source bnd=egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_boundary_levels/all \\
      --output egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_merged
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.metrics import compute_drr_db


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(target)


# --------------------------------------------------------------------------- #
# levels
# --------------------------------------------------------------------------- #
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
        drr = compute_drr_db(
            wav[:, index],
            sample_rate=sample_rate,
            direct_window_ms=window_ms,
        )
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


def cmd_levels(args) -> None:
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

    levels: dict[str, list[Item]] = {name: [] for name in ("core", "expand", "wide", "stress", "all")}
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


# --------------------------------------------------------------------------- #
# merge
# --------------------------------------------------------------------------- #
def cmd_merge(args) -> None:
    sources: list[tuple[str, Path]] = []
    for spec in args.source:
        tag, _, path = spec.partition("=")
        if not tag or not path:
            raise SystemExit(f"bad --source spec (want TAG=PATH): {spec}")
        view = Path(path).resolve()
        if not (view / "items").is_dir():
            raise SystemExit(f"source view has no items/: {view}")
        sources.append((tag, view))

    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"output already exists: {output} (no overwrite by design)")
    item_dir = output / "items"
    item_dir.mkdir(parents=True)

    provenance = {}
    total = 0
    for tag, view in sources:
        count = 0
        for wav in sorted((view / "items").glob("*.wav")):
            meta = wav.with_suffix(".json")
            if not meta.exists():
                continue
            # resolve() follows the source view's own symlinks so the merged view
            # points straight at the bank files (one hop, not chains).
            relative_symlink(wav.resolve(), item_dir / f"{tag}_{wav.name}")
            relative_symlink(meta.resolve(), item_dir / f"{tag}_{meta.name}")
            count += 1
        provenance[tag] = {"view": str(view), "items": count}
        total += count
        print(f"  {tag}: {count} items from {view}")

    (output / "merged.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"merged {total} items -> {item_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("levels", help="slice one bank into cumulative difficulty levels",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp.add_argument("source", type=Path, help="existing generated RIR bank")
    sp.add_argument("output", type=Path, help="new parent directory for the level folders")
    sp.add_argument("--drr-window-ms", type=float, default=2.5,
                    help="direct-path DRR window; match the recipe config (default: 2.5)")
    sp.add_argument("--workers", type=int, default=8, help="parallel WAV readers (default: 8)")
    sp.add_argument("--limit", type=int, default=None,
                    help="inspect only the first N items (diagnostic/testing only)")
    sp.add_argument("--dry-run", action="store_true",
                    help="measure and report without creating symlinks")
    sp.set_defaults(func=cmd_levels)

    sp = sub.add_parser("merge", help="union several views into one training view",
                        formatter_class=argparse.RawDescriptionHelpFormatter)
    sp.add_argument("--source", action="append", required=True, metavar="TAG=PATH",
                    help="view to merge, as tag=path (tag becomes the symlink prefix); "
                         "repeat per view")
    sp.add_argument("--output", type=Path, required=True,
                    help="new merged view directory (must not exist)")
    sp.set_defaults(func=cmd_merge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
