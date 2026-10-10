"""Choose, file by file, whether a training target is the cleaned copy or the original.

`clean_speech` writes a cleaned copy of every file and says what the cleaner did to
it. Whether to *use* that copy is a separate decision, made here, because the
cleaner is not transparent: even speech that is already clean comes back
measurably changed. Replacing a file that had no noise to remove buys nothing and
writes the cleaner's colouring into the target. Policies:

``all``        every file cleaned (DPDFNet's recipe)
``none``       every file original (the control)
``floor-drop`` cleaned only where the cleaner lowered the noise floor (quietest
               20 % of frames) by at least ``--min-floor-drop`` dB -- where there
               was something to clean

Run::

    python -m puresound.dataset.corpus.target_policy \\
        --original egs/noise_suppression/data/dns5_all/dnsde_train.csv \\
        --cleaned egs/noise_suppression/data/dns5_all/dnsde_train.v2clean.csv \\
        --policy floor-drop --min-floor-drop 3 \\
        --out egs/noise_suppression/data/pool_data1/dnsde_train.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .records import AudioRecord, read_metafile, write_metafile

POLICIES = ("all", "none", "floor-drop")


def choose(
    original: Sequence[AudioRecord],
    cleaned: Sequence[AudioRecord],
    stats: dict[str, dict],
    *,
    policy: str,
    min_floor_drop: float = 3.0,
) -> tuple[list[AudioRecord], int]:
    """``(records, n_cleaned)``, one record per original uttid.

    A file with no cleaned copy (or no stats, under ``floor-drop``) keeps its
    original -- missing cleaning is never a reason to lose a file.
    """
    if policy not in POLICIES:
        raise ValueError(f"policy must be one of {POLICIES}, got {policy!r}")
    by_uttid = {record.uttid: record for record in cleaned}
    chosen: list[AudioRecord] = []
    n_cleaned = 0
    for record in original:
        replacement = by_uttid.get(record.uttid)
        use = False
        if replacement is not None and policy == "all":
            use = True
        elif replacement is not None and policy == "floor-drop":
            row = stats.get(record.uttid)
            use = row is not None and (row["floor_in_db"] - row["floor_out_db"]) >= min_floor_drop
        chosen.append(replacement if use else record)
        n_cleaned += int(use)
    return chosen, n_cleaned


def read_stats(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["uttid"]] = row
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.dataset.corpus.target_policy", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--original", required=True)
    parser.add_argument(
        "--cleaned", required=True, action="append",
        help="Cleaned metafile; repeatable when the originals were cleaned in several runs.",
    )
    parser.add_argument(
        "--stats", default=None, action="append",
        help="Defaults to <cleaned>.stats.jsonl for each --cleaned.",
    )
    parser.add_argument("--policy", choices=POLICIES, required=True)
    parser.add_argument("--min-floor-drop", type=float, default=3.0)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    cleaned_paths = [Path(path) for path in args.cleaned]
    stats_paths = [Path(path) for path in args.stats] if args.stats else [
        path.with_suffix(".stats.jsonl") for path in cleaned_paths
    ]
    original = read_metafile(args.original)
    cleaned: list[AudioRecord] = []
    stats: dict[str, dict] = {}
    for path in cleaned_paths:
        if path.is_file():
            cleaned += read_metafile(path)
        elif args.policy != "none":
            print(f"{path} is missing; nothing to choose from.", file=sys.stderr)
            return 1
    for path in stats_paths:
        if path.is_file():
            stats.update(read_stats(path))
    records, n_cleaned = choose(
        original, cleaned, stats, policy=args.policy, min_floor_drop=args.min_floor_drop
    )
    write_metafile(Path(args.out), records)
    share = 100.0 * n_cleaned / max(len(records), 1)
    print(f"{Path(args.original).name}: {n_cleaned}/{len(records)} target(s) cleaned ({share:.1f}%), "
          f"policy={args.policy}" + (f" >= {args.min_floor_drop} dB" if args.policy == "floor-drop" else "")
          + f" -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
