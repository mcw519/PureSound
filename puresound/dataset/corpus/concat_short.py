"""Join a speaker's short utterances into pieces long enough to train on.

The NS recipes drop every utterance shorter than ``filter_min_utterance_length``
(4 s). That is right for DNS read speech, which is cut in 10 s segments, and it
silently deletes most of a corpus made of single sentences, such as CREMA-D
(acted sentences of about 2.5 s) or VCTK. Lowering the filter would change every
source's rows; joining the short utterances changes only the corpora that need
it.

Utterances of one speaker are taken in id order and joined with ``--gap`` seconds
of silence until a piece reaches ``--min-seconds`` (never past ``--max-seconds``).
Utterances already long enough are passed through untouched. A leftover shorter
than the minimum is kept as it is -- the recipe's filter decides about it, as it
would have anyway.

Run::

    python -m puresound.dataset.corpus.concat_short \\
        egs/noise_suppression/data/dns5_all/dnsvctk_train.csv \\
        --source-root /path/to/audio/dns-5/datasets_fullband_16k \\
        --dest-root /path/to/training_set/ns_speech_source/dns5_concat \\
        --out egs/noise_suppression/data/dns5_all/dnsvctk_concat_train.csv
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

from .records import AudioRecord, read_metafile, write_metafile


def plan_pieces(
    records: Sequence[AudioRecord], *, min_seconds: float, max_seconds: float, gap: float
) -> list[list[AudioRecord]]:
    """Groups of records to join; a group of one is passed through."""
    by_speaker: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        by_speaker[record.spkid].append(record)
    groups: list[list[AudioRecord]] = []
    for speaker in sorted(by_speaker):
        current: list[AudioRecord] = []
        seconds = 0.0
        for record in sorted(by_speaker[speaker], key=lambda r: r.uttid):
            length = record.length / record.sample_rate
            if length >= min_seconds:
                groups.append([record])
                continue
            added = length + (gap if current else 0.0)
            if current and seconds + added > max_seconds:
                groups.append(current)
                current, seconds, added = [], 0.0, length
            current.append(record)
            seconds += added
            if seconds >= min_seconds:
                groups.append(current)
                current, seconds = [], 0.0
        if current:
            groups.append(current)
    return groups


def _join(job: tuple) -> AudioRecord:
    import numpy as np
    import soundfile as sf

    group, source_root, dest_root, gap = job
    if len(group) == 1:
        return group[0]
    first = group[0]
    rate = first.sample_rate
    pieces = []
    for index, record in enumerate(group):
        audio, file_rate = sf.read(str(record.path), dtype="int16")
        if file_rate != rate:
            raise ValueError(f"{record.path} is {file_rate} Hz, the group is {rate} Hz")
        if index:
            pieces.append(np.zeros(int(gap * rate), dtype=np.int16))
        pieces.append(audio if audio.ndim == 1 else audio[:, 0])
    joined = np.concatenate(pieces)
    relative = first.path.relative_to(source_root)
    dest = Path(dest_root) / relative.parent / f"{relative.stem}_cat{len(group)}.wav"
    if not dest.is_file():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".tmp.wav")
        sf.write(str(tmp), joined, rate, subtype="PCM_16")
        tmp.replace(dest)
    # An existing file is kept as it is, so the record is built from its own
    # header: a rerun with another --gap would otherwise describe audio that is
    # not the audio on disk.
    info = sf.info(str(dest))
    return AudioRecord(
        uttid=f"{first.uttid}_cat{len(group)}", spkid=first.spkid, gender=first.gender,
        path=dest, length=int(info.frames), sample_rate=int(info.samplerate), channels=1,
        tags={**first.tags, "joined": [r.uttid for r in group]},
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m puresound.dataset.corpus.concat_short",
                                     description=__doc__.splitlines()[0])
    parser.add_argument("metafile")
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--min-seconds", type=float, default=4.5)
    parser.add_argument("--max-seconds", type=float, default=12.0)
    parser.add_argument("--gap", type=float, default=0.25)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    records = read_metafile(args.metafile)
    groups = plan_pieces(records, min_seconds=args.min_seconds, max_seconds=args.max_seconds, gap=args.gap)
    source_root = Path(args.source_root).expanduser().resolve()
    dest_root = Path(args.dest_root).expanduser().resolve()
    jobs = [(group, source_root, dest_root, args.gap) for group in groups]
    with ProcessPoolExecutor(args.jobs) as pool:
        out = list(pool.map(_join, jobs, chunksize=64))
    write_metafile(Path(args.out), sorted(out, key=lambda r: r.uttid))
    joined = sum(1 for g in groups if len(g) > 1)
    long_enough = sum(1 for r in out if r.length / r.sample_rate >= 4.0)
    print(json.dumps({
        "in_utts": len(records), "out_rows": len(out), "joined_rows": joined,
        "rows_at_least_4s": long_enough, "hours": round(sum(r.length / r.sample_rate for r in out) / 3600, 1),
    }))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
