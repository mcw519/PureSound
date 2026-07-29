"""Build a real end-to-end far-field POOL manifest from VOiCES distant recordings.

Unlike ``egs/rir_generation/real_rir_to_bank.py`` (which emits RIRs to CONVOLVE
with clean speech), this indexes FINISHED far-field waveforms -- loudspeaker ->
air -> mic recordings that already carry a whole capture chain, including the
parts an RIR convolution cannot reproduce (transducer non-linearity, directivity,
noise floor, level and spectral tilt). The training pipeline drops these in
directly as far interferers, with no convolution.

Each pool entry is one real distant recording of a SINGLE speaker (distractor=none
by default, so the far channel is a clean lone far voice, not babble/music/tele).
The manifest is a thin index; near/far filtering and split live here, sampling
lives in ns.py.

Manifest line schema (one JSON object per line):
    {
      "wav_path":   "/abs/path.wav",   # finished far-field recording
      "room":       "rm1",             # VOiCES room id
      "mic":        6,                 # mic channel number
      "distance_m": 3.02,              # loudspeaker->mic distance (m)
      "speaker":    "sp2156",          # source speaker id
      "loc":        "far",             # VOiCES mic location token
      "distractor": "none",            # competing-sound condition
      "split":      "train"            # which VOiCES split this came from
    }

Usage (from repo root):
    uv run python egs/voice_isolate/scripts/build_realfar_pool.py \
        --voices-root /work/any_exp_link/puresound_exp/real_e2e_corpora/voices/VOiCES_rebuilt \
        --split train --min-distance 1.0 \
        --out egs/voice_isolate/data/realfar_pool/voices.train.jsonl

Run again with ``--split test`` for a held-out pool (VOiCES test/ has different
speakers) used as a generalization probe, never mixed into training.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

# Reuse the probe's filename parser (room / speaker / mic / loc) so the parsing
# can never drift between the measurement (probe) and the training pool.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from probe_retransmitted_farfield import FNAME_RE  # noqa: E402

DIST_BUCKETS = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 5.0), (5.0, 99.0)]
INCHES_TO_M = 0.0254


def load_foreground_distances(distances_csv: Path) -> dict[str, float]:
    """query_name -> loudspeaker->mic distance (m), from the official distances.csv.

    The ``foreground`` column (inches) is per-file and covers ALL mics -- including
    rm3/rm4 mc13-20 that the devkit's 12-mic table omits -- and reflects the actual
    geometry (mic position + loudspeaker rotation), so it is authoritative over the
    filename ``loc`` token (e.g. rm4-mc18 is tagged ``clo`` but sits at 1.98 m).
    """
    dist: dict[str, float] = {}
    with open(distances_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fg = row.get("foreground")
            qn = row.get("query_name")
            if not fg or not qn:
                continue
            try:
                dist[qn] = float(fg) * INCHES_TO_M
            except ValueError:
                continue
    return dist


def bucket_label(dist: float) -> str:
    for lo, hi in DIST_BUCKETS:
        if lo <= dist < hi:
            return f"{lo:.0f}-{hi:.0f}m"
    return "?"


def scan(
    voices_root: Path,
    split: str,
    distractor: str,
    min_distance: float,
    max_distance: float | None,
    dist_map: dict[str, float],
) -> list[dict]:
    speech_root = voices_root / "distant-16k" / "speech" / split
    if not speech_root.is_dir():
        sys.exit(f"not found: {speech_root}")

    entries: list[dict] = []
    skipped_unparsed = skipped_nodist = skipped_near = 0
    pattern = f"*-{distractor}-*.wav"
    for wav_path in speech_root.rglob(pattern):
        m = FNAME_RE.search(wav_path.name)
        if m is None:
            skipped_unparsed += 1
            continue
        dist = dist_map.get(wav_path.stem)
        if dist is None:
            skipped_nodist += 1
            continue
        if dist < min_distance or (max_distance is not None and dist >= max_distance):
            skipped_near += 1
            continue
        entries.append(
            {
                "wav_path": str(wav_path),
                "room": m.group("room"),
                "mic": int(m.group("mic")),
                "distance_m": round(dist, 3),
                "speaker": f"sp{m.group('spk')}",
                "loc": m.group("loc"),
                "distractor": distractor,
                "split": split,
            }
        )
    if skipped_unparsed or skipped_nodist:
        print(
            f"# skipped: {skipped_unparsed} unparsed names, {skipped_nodist} no distance in csv, "
            f"{skipped_near} below min-distance {min_distance}m",
            file=sys.stderr,
        )
    return entries


def print_stats(entries: list[dict]) -> None:
    by_room: dict[str, list[dict]] = defaultdict(list)
    by_bucket: dict[str, list[dict]] = defaultdict(list)
    for e in entries:
        by_room[e["room"]].append(e)
        by_bucket[bucket_label(e["distance_m"])].append(e)

    print(f"\n# total pool entries: {len(entries)}")
    print(f"# unique speakers: {len({e['speaker'] for e in entries})}")

    print("\n# by room:")
    print("room\tn\tspeakers\tmics\tdist_range_m")
    for room in sorted(by_room):
        rs = by_room[room]
        dists = sorted({r["distance_m"] for r in rs})
        print(
            f"{room}\t{len(rs)}\t{len({r['speaker'] for r in rs})}\t"
            f"{len({r['mic'] for r in rs})}\t{dists[0]:.2f}-{dists[-1]:.2f}"
        )

    print("\n# by distance bucket:")
    print("bucket\tn\trooms")
    for lo, hi in DIST_BUCKETS:
        key = f"{lo:.0f}-{hi:.0f}m"
        rs = by_bucket.get(key, [])
        if rs:
            print(f"{key}\t{len(rs)}\t{len({r['room'] for r in rs})}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--voices-root", required=True, help="VOiCES_rebuilt root (contains distant-16k/)")
    parser.add_argument(
        "--distances-csv",
        default=None,
        help="official distances.csv (default: <voices-root>/../distances.csv)",
    )
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument(
        "--distractor",
        default="none",
        choices=["none", "babb", "musi", "tele"],
        help="none = clean lone far speaker (the right interferer for near-isolation)",
    )
    parser.add_argument(
        "--min-distance",
        type=float,
        default=1.0,
        help="keep only recordings at/beyond this loudspeaker->mic distance (m). "
        "Default 1.0 = far interferers only (product spec: >1m suppress).",
    )
    parser.add_argument(
        "--max-distance",
        type=float,
        default=None,
        help="keep only recordings below this distance (m). Use with "
        "--min-distance 0 --max-distance 1.0 to build the real NEAR keep pool "
        "(VOiCES 0.74/0.97m mics), for use as real near-field keep rows.",
    )
    parser.add_argument("--out", help="output manifest.jsonl path (required unless --stats-only)")
    parser.add_argument("--stats-only", action="store_true", help="print stats, do not write")
    args = parser.parse_args()
    if not args.stats_only and not args.out:
        parser.error("--out is required unless --stats-only")

    voices_root = Path(args.voices_root)
    distances_csv = Path(args.distances_csv) if args.distances_csv else voices_root.parent / "distances.csv"
    if not distances_csv.is_file():
        sys.exit(f"distances.csv not found: {distances_csv} (pass --distances-csv)")
    dist_map = load_foreground_distances(distances_csv)
    print(f"# loaded {len(dist_map)} per-file distances from {distances_csv}", file=sys.stderr)

    entries = scan(voices_root, args.split, args.distractor, args.min_distance,
                   args.max_distance, dist_map)
    if not entries:
        sys.exit("no pool entries produced -- check --voices-root / --split / filters")

    print_stats(entries)

    if not args.stats_only:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            "".join(json.dumps(e, ensure_ascii=False) + "\n" for e in entries),
            encoding="utf-8",
        )
        print(f"\n# wrote {len(entries)} entries -> {out_path}")


if __name__ == "__main__":
    main()
