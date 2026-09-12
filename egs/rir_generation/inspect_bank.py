#!/usr/bin/env python
"""Summarize the statistics of a folder of generated RIRs — to the terminal only.

Walks a directory for `*.json` RIR sidecars and reports distributions of room
dimensions, RT60, source distances, obstacles, and mic placement. Nothing is
written to disk.

Room-level quantities (dimensions, RT60, obstacles) are de-duplicated by room id
so a room with many RIRs is not over-counted; per-item quantities (distances,
mic height) use every RIR.

Usage:
  python inspect_bank.py egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k
  python inspect_bank.py egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k --bins 30
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def room_id_of(json_path: Path) -> str:
    # room_000000_000000.json -> room_000000
    stem = json_path.stem
    return stem.rsplit("_", 1)[0] if "_" in stem else stem


def fmt_stats(name: str, values) -> str:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return f"  {name:<16} (no data)"
    return (
        f"  {name:<16} n={a.size:<6d} "
        f"mean={a.mean():8.3f}  std={a.std():7.3f}  "
        f"min={a.min():8.3f}  p25={np.percentile(a,25):8.3f}  "
        f"p50={np.percentile(a,50):8.3f}  p75={np.percentile(a,75):8.3f}  "
        f"max={a.max():8.3f}"
    )


def histogram(values, bins: int, width: int = 40, unit: str = "") -> str:
    a = np.asarray(values, dtype=np.float64)
    if a.size == 0:
        return "    (no data)"
    counts, edges = np.histogram(a, bins=bins)
    peak = counts.max() or 1
    lines = []
    for i, c in enumerate(counts):
        bar = "█" * int(round(width * c / peak))
        lines.append(f"    [{edges[i]:7.3f}, {edges[i+1]:7.3f}){unit:<2} | "
                     f"{bar:<{width}} {c}")
    return "\n".join(lines)


def section(title: str):
    print(f"\n\033[1m{title}\033[0m")
    print("─" * len(title))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path, help="folder containing RIR *.json sidecars")
    ap.add_argument("--bins", type=int, default=20, help="histogram bins")
    args = ap.parse_args()

    json_paths = sorted(args.folder.rglob("*.json"))
    if not json_paths:
        raise SystemExit(f"no *.json sidecars found under {args.folder}")

    # Accumulators.
    rooms = {}                       # room_id -> scene (first seen)
    sample_rates, durations = [], []
    dist_by_label = {}               # label -> list of distances
    dist_near, dist_far, dist_all = [], [], []
    mic_heights = []
    sources_per_item = []
    n_items = 0

    for jp in json_paths:
        try:
            meta = json.loads(jp.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        scene = meta.get("scene")
        if scene is None:
            continue
        n_items += 1
        cfg = meta.get("config", {})
        if "sample_rate" in cfg:
            sample_rates.append(cfg["sample_rate"])
        if "duration" in cfg:
            durations.append(cfg["duration"])

        rooms.setdefault(room_id_of(jp), scene)
        mic_heights.append(scene["mic_pos"][2])
        cmap = scene.get("channel_map", [])
        sources_per_item.append(len(cmap))
        for ch in cmap:
            label = ch.get("label", "")
            d = ch.get("distance_m")
            if d is None:
                continue
            dist_by_label.setdefault(label, []).append(d)
            dist_all.append(d)
            (dist_near if label.startswith("near") else dist_far).append(d)

    # Room-level arrays (de-duplicated).
    dims = np.array([r["room_dim"] for r in rooms.values()], dtype=np.float64)
    rt60 = np.array([r["rt60"] for r in rooms.values()], dtype=np.float64)
    volumes = dims.prod(axis=1)
    floor_area = dims[:, 0] * dims[:, 1]
    n_obstacles = [len(r.get("obstacles", [])) for r in rooms.values()]
    materials = Counter(o.get("material", "?")
                        for r in rooms.values() for o in r.get("obstacles", []))

    # ---------------------------------------------------------------- report
    print(f"\033[1mRIR statistics — {args.folder}\033[0m")
    print(f"  {n_items} RIR items across {len(rooms)} unique rooms "
          f"({len(json_paths)} json files scanned)")
    sr_set = sorted(set(sample_rates))
    dur_set = sorted(set(durations))
    print(f"  sample_rate(s): {sr_set}    duration(s): {dur_set}")

    section("Room dimensions (per unique room)")
    print(fmt_stats("x / width (m)", dims[:, 0]))
    print(fmt_stats("y / depth (m)", dims[:, 1]))
    print(fmt_stats("z / height (m)", dims[:, 2]))
    print(fmt_stats("floor area (m²)", floor_area))
    print(fmt_stats("volume (m³)", volumes))

    section("RT60 (per unique room)")
    print(fmt_stats("rt60 (s)", rt60))
    print(histogram(rt60, args.bins, unit="s"))

    section("Room volume (per unique room)")
    print(histogram(volumes, args.bins))

    section("Source distances (per RIR item)")
    print(fmt_stats("all sources", dist_all))
    print(fmt_stats("near_*", dist_near))
    print(fmt_stats("far_*", dist_far))
    for label in sorted(dist_by_label):
        print(fmt_stats(label, dist_by_label[label]))
    print("\n  distribution (all sources):")
    print(histogram(dist_all, args.bins, unit="m"))

    section("Obstacles (per unique room)")
    print(fmt_stats("count / room", n_obstacles))
    total_obs = sum(n_obstacles)
    if total_obs:
        print("  materials:")
        for mat, c in materials.most_common():
            print(f"    {mat:<12} {c:5d}  ({100*c/total_obs:5.1f}%)")

    section("Microphone (per RIR item)")
    print(fmt_stats("mic height (m)", mic_heights))
    print(fmt_stats("sources / item", sources_per_item))
    print()


if __name__ == "__main__":
    main()
