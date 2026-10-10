"""LibriLight -> segmented training speech: pick the cleanest hours, cut them to size.

LibriLight is LibriVox audiobooks as 16 kHz FLAC, one file per chapter (often 20
minutes), each with a JSON beside it carrying the reader, a voice-activity list and
an estimated SNR. That makes two jobs:

``select``
    Rank every chapter by its SNR and take the cleanest until an hour budget is
    spent, with a per-reader cap so a few prolific readers do not become most of
    the corpus. Readers of any held-out set named by ``--exclude-speakers-in`` (a
    directory of ``<speaker>/`` folders, e.g. LibriTTS ``test-clean``) are refused
    first: LibriLight shares LibriVox's reader ids with LibriSpeech and LibriTTS,
    and our hard WER set is LibriTTS ``test-clean``.

``segment``
    Cut each chosen chapter along its voice activity into ``--min-seconds`` ..
    ``--max-seconds`` pieces -- whole-chapter files would be decoded in full for
    every few-second training crop -- and write a metafile over them.

Multilingual LibriSpeech's English split is the same LibriVox audio, but the
published copy is Opus at about 35 kbps on top of LibriVox's 64 kbps MP3; this one
is one lossy generation fewer.

Run::

    python -m puresound.dataset.corpus.librilight select /path/to/audio/LibriLight \\
        --hours 4000 --max-hours-per-speaker 4 \\
        --exclude-speakers-in /path/to/audio/LibriTTS/test-clean \\
        --out egs/noise_suppression/data/librilight/selection.jsonl
    python -m puresound.dataset.corpus.librilight segment \\
        egs/noise_suppression/data/librilight/selection.jsonl \\
        --source-root /path/to/audio/LibriLight \\
        --dest-root /path/to/training_set/ns_speech_source/librilight_segments \\
        --out egs/noise_suppression/data/librilight/ll_train.csv
"""

from __future__ import annotations

import argparse
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Iterator, Sequence

from .records import AudioRecord, write_metafile
from .scan import sanitize_id


SUBSETS = ("small", "medium", "large")


def iter_chapters(root: Path, subsets: Sequence[str] = SUBSETS) -> Iterator[dict[str, Any]]:
    """One dict per chapter: ``{flac, speaker, snr, voice_activity, subset}``."""
    for subset in subsets:
        base = root / subset
        if not base.is_dir():
            continue
        for dirpath, _, files in os.walk(base):
            for name in sorted(files):
                if not name.endswith(".json"):
                    continue
                meta_path = Path(dirpath) / name
                flac = meta_path.with_suffix(".flac")
                if not flac.is_file():
                    continue
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                yield {
                    "flac": str(flac),
                    "speaker": str(meta["speaker"]),
                    "snr": meta.get("snr"),
                    "voice_activity": meta.get("voice_activity") or [],
                    "subset": subset,
                }


def voiced_seconds(chapter: dict[str, Any]) -> float:
    return float(sum(end - start for start, end in chapter["voice_activity"]))


def speakers_in(folders: Sequence[str | Path]) -> set[str]:
    """Reader ids of held-out sets laid out as ``<folder>/<speaker>/...``."""
    found: set[str] = set()
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            raise FileNotFoundError(f"--exclude-speakers-in {folder} is not a directory")
        found |= {entry.name for entry in folder.iterdir() if entry.is_dir()}
    return found


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def select_chapters(
    chapters: Sequence[dict[str, Any]],
    *,
    hours: float,
    max_hours_per_speaker: float | None,
    excluded: set[str],
    min_snr: float | None = None,
) -> list[dict[str, Any]]:
    """Cleanest-first under the two budgets. Chapters without an SNR are skipped.

    Including NaN ones: LibriLight has a few dozen, and a NaN in the sort key does
    not sort last -- it silently scrambles the whole ranking.
    """
    ranked = sorted(
        (c for c in chapters if _finite(c["snr"]) and c["speaker"] not in excluded),
        key=lambda c: (-float(c["snr"]), c["flac"]),
    )
    budget = hours * 3600.0
    cap = max_hours_per_speaker * 3600.0 if max_hours_per_speaker else None
    per_speaker: dict[str, float] = {}
    chosen: list[dict[str, Any]] = []
    total = 0.0
    for chapter in ranked:
        if min_snr is not None and float(chapter["snr"]) < min_snr:
            break
        seconds = voiced_seconds(chapter)
        if cap is not None and per_speaker.get(chapter["speaker"], 0.0) + seconds > cap:
            continue
        chosen.append(chapter)
        per_speaker[chapter["speaker"]] = per_speaker.get(chapter["speaker"], 0.0) + seconds
        total += seconds
        if total >= budget:
            break
    return chosen


def plan_segments(
    voice_activity: Sequence[Sequence[float]],
    *,
    min_seconds: float,
    max_seconds: float,
    max_gap: float,
    context: float,
    duration: float | None = None,
) -> list[tuple[float, float]]:
    """Merge voiced regions into pieces no longer than ``max_seconds``.

    A gap longer than ``max_gap`` always ends a piece, so pieces are continuous
    reading rather than two paragraphs glued across a pause. ``context`` seconds
    of the surrounding audio are kept on each side (clipped to the file), because
    a piece that starts on the first sample of speech has no onset for the model
    to learn to keep.
    """
    pieces: list[tuple[float, float]] = []
    start = end = None
    for a, b in voice_activity:
        a, b = float(a), float(b)
        if b - a > max_seconds:  # one very long region: split it evenly
            if start is not None:
                pieces.append((start, end))
                start = end = None
            count = int(-(-(b - a) // max_seconds))
            step = (b - a) / count
            pieces += [(a + i * step, a + (i + 1) * step) for i in range(count)]
            continue
        if start is None:
            start, end = a, b
        elif a - end <= max_gap and b - start <= max_seconds:
            end = b
        else:
            pieces.append((start, end))
            start, end = a, b
    if start is not None:
        pieces.append((start, end))

    out = []
    for a, b in pieces:
        if b - a < min_seconds:
            continue
        lo = max(0.0, a - context)
        hi = b + context if duration is None else min(duration, b + context)
        out.append((round(lo, 3), round(hi, 3)))
    return out


def _segment_one(job: tuple) -> list[AudioRecord]:
    import soundfile as sf

    chapter, dest_root, source_root, knobs = job
    flac = Path(chapter["flac"])
    info = sf.info(str(flac))
    if info.samplerate != 16000:
        raise ValueError(f"{flac} is {info.samplerate} Hz; LibriLight ships 16 kHz")
    pieces = plan_segments(chapter["voice_activity"], duration=info.duration, **knobs)
    if not pieces:
        return []
    audio, rate = sf.read(str(flac), dtype="int16")
    relative = flac.relative_to(source_root).with_suffix("")
    # A few LibriLight chapter names carry commas, which the metafile format
    # cannot hold; the piece is named without them.
    relative = Path(*[part.replace(",", "_") for part in relative.parts])
    speaker = sanitize_id(chapter["speaker"])
    records = []
    for index, (a, b) in enumerate(pieces):
        dest = Path(dest_root) / relative.parent / f"{relative.name}_{index:04d}.flac"
        clip = audio[int(a * rate) : int(b * rate)]
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(".tmp.flac")
            sf.write(str(tmp), clip, rate, subtype="PCM_16")
            os.replace(tmp, dest)
        # A piece already on disk is kept as it is, so the record is built from its
        # own header: a rerun with other knobs would otherwise describe audio that
        # is not the audio on disk.
        written = sf.info(str(dest))
        records.append(
            AudioRecord(
                uttid=f"ll_{sanitize_id(relative.as_posix().replace('/', '_'))}_{index:04d}",
                spkid=f"ll_{speaker}",
                gender="None",
                path=dest,
                length=int(written.frames),
                sample_rate=int(written.samplerate),
                channels=1,
                tags={"snr": chapter["snr"], "subset": chapter["subset"]},
            )
        )
    return records


def run_select(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve()
    excluded = speakers_in(args.exclude_speakers_in or [])
    excluded |= set(args.exclude_speaker or [])
    chapters = list(iter_chapters(root, args.subset or SUBSETS))
    print(f"{len(chapters)} chapter(s), {sum(map(voiced_seconds, chapters)) / 3600:.0f} voiced hours; "
          f"{len(excluded)} excluded reader id(s)")
    refused = sum(1 for c in chapters if c["speaker"] in excluded)
    chosen = select_chapters(
        chapters, hours=args.hours, max_hours_per_speaker=args.max_hours_per_speaker,
        excluded=excluded, min_snr=args.min_snr,
    )
    hours = sum(map(voiced_seconds, chosen)) / 3600
    speakers = {c["speaker"] for c in chosen}
    print(f"refused {refused} chapter(s) of excluded readers")
    print(f"chose {len(chosen)} chapter(s), {hours:.0f} voiced hours, {len(speakers)} reader(s), "
          f"SNR >= {min(float(c['snr']) for c in chosen):.1f}" if chosen else "chose nothing")
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as handle:
        for chapter in chosen:
            handle.write(json.dumps(chapter) + "\n")
    print(f"selection -> {out}")
    return 0 if chosen else 1


def run_segment(args: argparse.Namespace) -> int:
    chosen = [json.loads(line) for line in Path(args.selection).read_text(encoding="utf-8").splitlines() if line.strip()]
    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()
    knobs = {
        "min_seconds": args.min_seconds, "max_seconds": args.max_seconds,
        "max_gap": args.max_gap, "context": args.context,
    }
    jobs = [(chapter, dest_root, source_root, knobs) for chapter in chosen]
    records: list[AudioRecord] = []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        for done, found in enumerate(pool.map(_segment_one, jobs, chunksize=4), start=1):
            records.extend(found)
            if done % 500 == 0:
                print(f"  {done}/{len(jobs)} chapter(s), {len(records)} piece(s)", flush=True)
    write_metafile(Path(args.out).expanduser(), sorted(records, key=lambda r: r.uttid))
    hours = sum(r.length for r in records) / 16000 / 3600
    print(f"wrote {len(records)} piece(s), {hours:.0f} h, "
          f"{len({r.spkid for r in records})} speaker(s) -> {args.out}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.dataset.corpus.librilight",
        description=__doc__.splitlines()[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    select = sub.add_parser("select", help="rank chapters by SNR and take an hour budget")
    select.add_argument("root", help="LibriLight root with small/ medium/ large/")
    select.add_argument("--subset", action="append", default=None, choices=SUBSETS)
    select.add_argument("--hours", type=float, required=True)
    select.add_argument("--max-hours-per-speaker", type=float, default=None)
    select.add_argument("--min-snr", type=float, default=None)
    select.add_argument("--exclude-speakers-in", action="append", default=None, metavar="DIR")
    select.add_argument("--exclude-speaker", action="append", default=None, metavar="ID")
    select.add_argument("--out", required=True)
    select.set_defaults(func=run_select)

    segment = sub.add_parser("segment", help="cut the selection into training pieces")
    segment.add_argument("selection")
    segment.add_argument("--source-root", required=True, help="the LibriLight root the selection was made from")
    segment.add_argument("--dest-root", required=True)
    segment.add_argument("--out", required=True)
    segment.add_argument("--min-seconds", type=float, default=3.0)
    segment.add_argument("--max-seconds", type=float, default=20.0)
    segment.add_argument("--max-gap", type=float, default=1.0)
    segment.add_argument("--context", type=float, default=0.25)
    segment.add_argument("--jobs", type=int, default=8)
    segment.set_defaults(func=run_segment)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
