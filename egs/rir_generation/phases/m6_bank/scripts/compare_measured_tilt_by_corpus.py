#!/usr/bin/env python
"""Check whether a measured-RIR reference is a valid spectral-tilt target.

Published RIRs carry the measurement loudspeaker's response unless the corpus
deconvolved it, so a pooled median over several corpora can look like room
physics while actually being a mixture of chains. Matching a renderer to that
number bakes somebody else's loudspeaker into the training data.

The test is a variance decomposition. Rooms inside one corpus differ in size,
materials and furnishing, so genuine room physics shows up as spread *within* a
corpus. A measurement chain is constant per corpus, so it shows up as an offset
*between* corpora. If between-corpus spread dominates, the pooled median is not
a physical target.

Run this before treating any measured tilt as something to match, and re-run it
whenever the reference corpus mixture changes.

Exit code 0 if the corpora agree closely enough to pool, 1 if they do not.
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import random
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.audio.rir.metrics import spectral_tilt_db_per_octave

DEFAULT_ROOT = Path(
    "/work/any_exp_link/puresound_exp/real_rir_16k_train_view/items"
)

#: Between-corpus spread above this many dB/oct means the corpora are measuring
#: different things and their pooled median is not a room property.  0.75 is
#: deliberately loose: it is already larger than the within-corpus spread seen
#: on every corpus in this reference.
POOLING_TOLERANCE_DB_PER_OCTAVE = 0.75

#: Minimum rooms before a corpus contributes to the decomposition.
MINIMUM_ROOMS = 3


def _corpus_of(path: Path) -> str:
    return path.name.split("_")[0]


def _room_of(path: Path) -> str:
    return path.name.rsplit("_", 1)[0]


def collect(root: Path, per_corpus: int, sample_rate: int, seed: int):
    by_corpus: dict[str, list[Path]] = collections.defaultdict(list)
    for meta in glob.glob(str(root / "*.json")):
        by_corpus[_corpus_of(Path(meta))].append(Path(meta))

    rng = random.Random(seed)
    rows: list[tuple[str, str, float, float]] = []
    for corpus, paths in sorted(by_corpus.items()):
        rng.shuffle(paths)
        used = 0
        for meta_path in paths:
            if used >= per_corpus:
                break
            try:
                metadata = json.loads(meta_path.read_text(encoding="utf-8"))
                audio, file_rate = sf.read(
                    meta_path.with_suffix(".wav"), always_2d=True
                )
            except (OSError, ValueError, json.JSONDecodeError):
                continue
            if file_rate != sample_rate:
                continue
            for entry in metadata.get("scene", {}).get("channel_map", []):
                index = int(entry["channel"])
                if index >= audio.shape[1]:
                    continue
                tilt = spectral_tilt_db_per_octave(
                    audio[:, index].astype(np.float64), sample_rate
                )
                if tilt is None or not np.isfinite(tilt):
                    continue
                rows.append((
                    corpus,
                    _room_of(meta_path),
                    float(tilt),
                    float(entry.get("distance_m", float("nan"))),
                ))
            used += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--per-corpus", type=int, default=220)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--output-report", type=Path)
    args = parser.parse_args()

    rows = collect(args.root, args.per_corpus, args.sample_rate, args.seed)
    if not rows:
        print(f"no usable RIRs under {args.root}", file=sys.stderr)
        return 1

    per_room: dict[tuple[str, str], list[float]] = collections.defaultdict(list)
    for corpus, room, tilt, _distance in rows:
        per_room[(corpus, room)].append(tilt)
    room_median = {key: float(np.median(v)) for key, v in per_room.items()}

    corpora = sorted({corpus for corpus, _ in room_median})
    print(f"{len(rows)} channels, {len(room_median)} rooms, {len(corpora)} corpora\n")
    print(f"{'corpus':<12} {'rooms':>6} {'ch':>6} {'median':>8} {'room sd':>8}")
    summary = {}
    for corpus in corpora:
        room_values = [v for (c, _), v in room_median.items() if c == corpus]
        channels = [t for c, _, t, _ in rows if c == corpus]
        sd = float(np.std(room_values, ddof=1)) if len(room_values) > 1 else float("nan")
        summary[corpus] = {
            "rooms": len(room_values),
            "channels": len(channels),
            "median_tilt_db_per_octave": float(np.median(room_values)),
            "room_sd_db_per_octave": sd,
        }
        print(f"{corpus:<12} {len(room_values):>6} {len(channels):>6} "
              f"{np.median(room_values):>+8.2f} {sd:>8.2f}")

    eligible = [c for c in corpora if summary[c]["rooms"] >= MINIMUM_ROOMS]
    within = float(np.mean([summary[c]["room_sd_db_per_octave"] for c in eligible]))
    medians = [summary[c]["median_tilt_db_per_octave"] for c in eligible]
    between = float(np.std(medians, ddof=1)) if len(medians) > 1 else 0.0
    pooled = float(np.median([t for _, _, t, _ in rows]))

    print(f"\npooled median              {pooled:+.2f} dB/oct")
    print(f"within-corpus room sd      {within:.2f} dB/oct")
    print(f"between-corpus sd          {between:.2f} dB/oct  ({between / within:.2f}x)")
    print(f"corpus median range        {min(medians):+.2f} to {max(medians):+.2f} "
          f"({max(medians) - min(medians):.2f} dB/oct apart)")

    poolable = between <= POOLING_TOLERANCE_DB_PER_OCTAVE
    if args.output_report:
        args.output_report.parent.mkdir(parents=True, exist_ok=True)
        args.output_report.write_text(
            json.dumps(
                {
                    "schema_version": "puresound.measured_tilt_by_corpus.v1",
                    "root": str(args.root),
                    "pooled_median_db_per_octave": pooled,
                    "within_corpus_room_sd_db_per_octave": within,
                    "between_corpus_sd_db_per_octave": between,
                    "pooling_tolerance_db_per_octave": (
                        POOLING_TOLERANCE_DB_PER_OCTAVE
                    ),
                    "corpora": summary,
                    "poolable": poolable,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"\nreport: {args.output_report}")

    if poolable:
        print("\nPOOLABLE — the corpora agree; the pooled median can serve as a target.")
        return 0
    print(
        "\nNOT POOLABLE — between-corpus spread exceeds the within-corpus spread, "
        "so the pooled median reflects the mixture of measurement chains rather "
        "than room physics. Do not tune a renderer to it."
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
