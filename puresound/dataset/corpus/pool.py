"""Combine several corpora into one training pool, and say what the mix will be.

A recipe names a single ``train_metafile``, so training on more than one corpus
means merging their metafiles. That part is concatenation. The part worth a module
is the reporting, because **the sampler draws speakers uniformly**: what decides
how much of each corpus a model sees is each corpus's *speaker count*, not its
hours. A corpus with few speakers gets a small share of the batches however many
hours it brings, and nothing in the metafile says so. This prints it.

``--max-speakers`` is the knob that follows from that: it caps a source's share by
dropping whole speakers, which is the unit the sampler works in. Capping rows
instead would change the hours and leave the batch share exactly where it was.

Every row's audio is also read once and dropped if it is all zeros, empty, or
unreadable. The coverage sampler names the utterance the dataset must load, and
the dataset treats such a file as an error rather than drawing another, so one of
them in the pool stops a run hours in. ``--skip-audio-check`` skips the read.

Run::

    python -m puresound.dataset.corpus.pool merge \\
        --source egs/noise_suppression/data/dns5/dns5_train.csv \\
        --source egs/noise_suppression/data/vctk/vctk_train.csv \\
        --out egs/noise_suppression/data/pool/pool_train.csv
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import soundfile as sf

from .records import AudioRecord, read_metafile, write_metafile


class SourceSummary:
    """One source's contribution to a pool, in the units that decide the mix."""

    def __init__(self, name: str, records: Sequence[AudioRecord]):
        self.name = name
        self.records = list(records)

    @property
    def speakers(self) -> set[str]:
        return {record.spkid for record in self.records}

    @property
    def hours(self) -> float:
        return sum(record.duration for record in self.records) / 3600.0

    def row(self, total_speakers: int) -> str:
        share = len(self.speakers) / total_speakers if total_speakers else 0.0
        return (
            f"  {self.name:28s} {len(self.speakers):6d} spk  "
            f"{len(self.records):8d} utt  {self.hours:8.1f} h  "
            f"-> {share:5.1%} of batches"
        )


def cap_speakers(
    records: Sequence[AudioRecord], max_speakers: int | None, seed: int
) -> list[AudioRecord]:
    """Keep at most ``max_speakers`` whole speakers, chosen deterministically.

    Whole speakers, because a speaker split in half is still one draw for the
    sampler -- capping rows would change the hours and leave the batch share
    exactly where it was.
    """
    if not max_speakers:
        return list(records)
    speakers = sorted({record.spkid for record in records})
    if len(speakers) <= max_speakers:
        return list(records)
    keep = set(random.Random(seed).sample(speakers, max_speakers))
    return [record for record in records if record.spkid in keep]


def _audio_status(path: str) -> str:
    """``ok``, ``silent`` (no samples, or every sample zero) or ``unreadable``."""
    try:
        samples, _ = sf.read(path, dtype="float32")
    except Exception:  # whatever the file is, the dataset cannot open it either
        return "unreadable"
    return "ok" if samples.size and np.any(samples) else "silent"


def screen_audio(
    records: Sequence[AudioRecord], jobs: int = 1
) -> tuple[list[AudioRecord], list[AudioRecord], list[AudioRecord]]:
    """Split records into (usable, silent, unreadable) by reading each file once.

    Processes, not threads: decoding holds the GIL for part of the work, and a
    pool of a million files is minutes on a pool of processes.
    """
    paths = [str(record.path) for record in records]
    if jobs <= 1:
        statuses = [_audio_status(path) for path in paths]
    else:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            statuses = list(pool.map(_audio_status, paths, chunksize=256))
    split: dict[str, list[AudioRecord]] = {"ok": [], "silent": [], "unreadable": []}
    for record, status in zip(records, statuses):
        split[status].append(record)
    return split["ok"], split["silent"], split["unreadable"]


def merge_records(sources: Iterable[tuple[str, Sequence[AudioRecord]]]) -> list[AudioRecord]:
    """Concatenate, refusing an id that means two different files.

    A duplicate ``uttid`` across corpora is not a merge conflict to resolve
    quietly: the dataset keys on it, so one of the two files would silently never
    be drawn. A duplicate ``spkid`` is the worse one -- two corpora's speakers
    fused into a single sampler class -- and it is what the ``--id-prefix`` on
    each corpus module exists to prevent.
    """
    merged: list[AudioRecord] = []
    seen_utt: dict[str, str] = {}
    speaker_owner: dict[str, str] = {}
    for name, records in sources:
        for record in records:
            previous = seen_utt.get(record.uttid)
            if previous is not None:
                raise ValueError(
                    f"uttid {record.uttid!r} appears in both {previous!r} and "
                    f"{name!r}. Give one of them a different --id-prefix; the "
                    "dataset keys on this id and would only ever draw one file."
                )
            seen_utt[record.uttid] = name
            owner = speaker_owner.setdefault(record.spkid, name)
            if owner != name:
                raise ValueError(
                    f"spkid {record.spkid!r} appears in both {owner!r} and "
                    f"{name!r}, which would fuse two corpora's speakers into one "
                    "sampler class. Give one of them a different --id-prefix."
                )
            merged.append(record)
    return merged


def run_merge(args: argparse.Namespace) -> int:
    if len(args.source) < 2:
        print("Pass at least two --source metafiles to merge.", file=sys.stderr)
        return 1

    caps: dict[str, str] = {}
    for pair in args.max_speakers:
        if "=" not in pair:
            print(f"--max-speakers expects STEM=N, got {pair!r}", file=sys.stderr)
            return 1
        stem, count = pair.split("=", 1)
        caps[stem] = count
    excluded: set[str] = set()
    for listing in args.exclude_speakers:
        excluded |= {line.strip() for line in Path(listing).read_text().splitlines() if line.strip()}
    summaries: list[SourceSummary] = []
    for source in args.source:
        path = Path(source).expanduser().resolve()
        records = read_metafile(path)
        if not records:
            print(f"{path} has no rows.", file=sys.stderr)
            return 1
        name = path.stem
        if excluded:
            kept = [record for record in records if record.spkid not in excluded]
            gone = {record.spkid for record in records} - {record.spkid for record in kept}
            if gone:
                print(f"  {name}: excluded {len(gone)} speaker(s), {len(records) - len(kept)} utt")
            records = kept
        cap = caps.get(name)
        if cap is not None:
            records = cap_speakers(records, int(cap), args.seed)
        if not args.skip_audio_check:
            records, silent, unreadable = screen_audio(records, jobs=args.jobs)
            for kind, dropped in (("silent", silent), ("unreadable", unreadable)):
                if dropped:
                    print(f"  {name}: dropped {len(dropped)} {kind} utt (e.g. {dropped[0].path})")
        summaries.append(SourceSummary(name, records))

    merged = merge_records((summary.name, summary.records) for summary in summaries)
    if not merged:
        print("No usable audio remains after filtering; output was not written.", file=sys.stderr)
        return 1
    total_speakers = len({record.spkid for record in merged})

    print("pool:")
    for summary in summaries:
        print(summary.row(total_speakers))
    total_hours = sum(summary.hours for summary in summaries)
    print(
        f"  {'TOTAL':28s} {total_speakers:6d} spk  {len(merged):8d} utt  "
        f"{total_hours:8.1f} h"
    )
    print(
        "  (batch share is by speaker count: the sampler draws speakers uniformly, "
        "so hours do not decide it)"
    )

    out = Path(args.out).expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    write_metafile(out, merged)
    print(f"wrote {len(merged)} row(s) -> {out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.dataset.corpus.pool",
        description=__doc__.splitlines()[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    merge = sub.add_parser("merge", help="several metafiles -> one training pool")
    merge.add_argument("--source", action="append", required=True)
    merge.add_argument("--out", required=True)
    merge.add_argument(
        "--max-speakers",
        action="append",
        default=[],
        metavar="STEM=N",
        help="Cap one source at N speakers, named by its metafile stem "
        "(e.g. vctk_train=12). Repeatable.",
    )
    merge.add_argument(
        "--exclude-speakers",
        action="append",
        default=[],
        metavar="FILE",
        help="File of spkids (one per line) that no source may contribute -- e.g. "
        "readers matched by voice to a test set. Repeatable.",
    )
    merge.add_argument("--seed", type=int, default=1234)
    merge.add_argument(
        "--skip-audio-check",
        action="store_true",
        help="Do not read the audio. By default every row is read once and dropped "
        "if it is silent, empty or unreadable.",
    )
    merge.add_argument(
        "--jobs",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Processes for the audio check (default: half the cores).",
    )
    merge.set_defaults(func=run_merge)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
