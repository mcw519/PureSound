#!/usr/bin/env python
"""Merge multiple RIR-bank level views into one symlink-only training view.

``PreGeneratedRoomBank`` indexes a single ``<view>/items/`` folder of same-stem
``.wav``/``.json`` pairs. Different banks reuse the same ``room_XXXXXX_YYYYYY``
stems (each generation run counts rooms from zero), so a naive union would
collide. This script prefixes every symlink with a per-source tag (wav and json
renamed together, keeping the same stem) and writes a ``merged.json`` recording
the provenance.

Example (Phase-1 boundary experiment -- wide base + boundary-distance fill):
  uv run python egs/rir_generation/merge_rir_views.py \
      --source wide=/work/any_exp_link/puresound_exp/hybrid_rir_16k_levels/wide \
      --source bnd=/work/any_exp_link/puresound_exp/hybrid_rir_16k_boundary_levels/all \
      --output /work/any_exp_link/puresound_exp/hybrid_rir_16k_phase1
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


def relative_symlink(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    target = os.path.relpath(source.resolve(), destination.parent.resolve())
    destination.symlink_to(target)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--source",
        action="append",
        required=True,
        metavar="TAG=PATH",
        help="level view to merge, as tag=path (tag becomes the symlink prefix); "
        "repeat per view",
    )
    parser.add_argument("--output", type=Path, required=True,
                        help="new merged view directory (must not exist)")
    args = parser.parse_args()

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
        wavs = sorted((view / "items").glob("*.wav"))
        count = 0
        for wav in wavs:
            meta = wav.with_suffix(".json")
            if not meta.exists():
                continue
            # resolve() follows the source view's own symlinks so the merged
            # view points straight at the bank files (one hop, not chains).
            relative_symlink(wav.resolve(), item_dir / f"{tag}_{wav.name}")
            relative_symlink(meta.resolve(), item_dir / f"{tag}_{meta.name}")
            count += 1
        provenance[tag] = {"view": str(view), "items": count}
        total += count
        print(f"  {tag}: {count} items from {view}")

    (output / "merged.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(f"merged {total} items -> {item_dir}")


if __name__ == "__main__":
    main()
