"""Mirror a corpus into a fixed-sample-rate tree, once, before training.

Resampling inside the dataloader costs every epoch what this costs once, and for
a 48 kHz corpus that is a large share of the whole pipeline's time.

The conversion goes through :class:`~puresound.audio.io.AudioIO`, which is the same
resampler the synthesis pipeline uses. That matters more than speed -- a corpus
converted by some other resampler is a slightly different corpus, and the
difference lands in the model rather than in a log.

Records are rebuilt from the header of the file that was actually written, never by
scaling the source numbers. Dividing a sample count by the rate ratio is right until
one file is 44.1 kHz, and then the metafile lies about a length nothing rereads.
"""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple, Sequence

import torch

from puresound.audio.io import AudioIO

from .records import AudioRecord
from .scan import audio_info


@dataclass
class ResampleReport:
    """What a conversion run did. ``failed`` is a list of (source, reason)."""

    converted: int = 0
    reused: int = 0
    failed: list[tuple[Path, str]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.converted + self.reused + len(self.failed)

    def summary(self) -> str:
        return (
            f"{self.total} file(s): {self.converted} converted, {self.reused} reused, "
            f"{len(self.failed)} failed"
        )


def mirrored_path(source: Path, source_root: Path, dest_root: Path, suffix: str = ".wav") -> Path:
    """Where ``source`` lands under ``dest_root``, keeping its relative layout."""
    return (dest_root / source.relative_to(source_root)).with_suffix(suffix)


def ensure_distinct_destinations(
    records: Sequence[AudioRecord],
    source_root: Path,
    dest_root: Path,
    suffix: str = ".wav",
) -> None:
    """Raise if two records would be written to the same mirrored file.

    ``a.wav`` and ``a.flac`` in one folder both mirror to ``a<suffix>``: the second
    would find the first's output already there, take it as its own converted copy,
    and the metafile would silently point two recordings at one file.
    """
    owner: dict[Path, Path] = {}
    for record in records:
        dest = mirrored_path(record.path, source_root, dest_root, suffix)
        if dest in owner and owner[dest] != record.path:
            raise ValueError(
                f"{owner[dest]} and {record.path} both mirror to {dest}; "
                "rename one or convert them separately."
            )
        owner[dest] = record.path


class _Outcome(NamedTuple):
    index: int
    record: AudioRecord | None
    reused: bool
    error: str | None


#: libsndfile subtype per signed-PCM width this converter can write.
_PCM_SUBTYPE = {8: "PCM_S8", 16: "PCM_16", 24: "PCM_24", 32: "PCM_32"}


def _convert_one(
    source: Path,
    dest: Path,
    target_sample_rate: int,
    mono: bool,
    bits_per_sample: int | None,
) -> None:
    wav, sample_rate = AudioIO.open(str(source), resample_to=target_sample_rate)
    if mono and wav.dim() > 1 and wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save_kwargs = {}
    if bits_per_sample is not None:
        if int(bits_per_sample) not in _PCM_SUBTYPE:
            raise ValueError(
                f"bits_per_sample={bits_per_sample} is not a signed-PCM width "
                f"this converter can write; choose from {sorted(_PCM_SUBTYPE)}."
            )
        save_kwargs = {"subtype": _PCM_SUBTYPE[int(bits_per_sample)]}
    AudioIO.save(wav, str(dest), sample_rate, **save_kwargs)


def _work(job: tuple) -> _Outcome:
    """One file, in whichever process picked it up.

    Module level and argument-only on purpose: this has to be picklable, because
    the conversion runs in processes rather than threads -- resampling is
    CPU-bound and threads would serialise on the GIL.
    """
    index, record, dest, target_sample_rate, mono, bits_per_sample, overwrite = job
    # One thread per worker.  The pool is forked from a parent that has usually
    # already run torch ops, so its OpenMP team exists in the parent but not in
    # the child; the first parallel op in the child then waits forever for
    # workers that were never re-created.  Parallelism here is the pool's job.
    torch.set_num_threads(1)
    try:
        reused = False
        info = None
        if not overwrite and dest.exists() and dest.stat().st_size > 0:
            try:
                info = audio_info(dest)
                reused = info[0] == target_sample_rate and (not mono or info[3] == 1)
            except Exception:
                # A partial or otherwise unreadable destination is not reusable.
                info = None
        if not reused:
            _convert_one(record.path, dest, target_sample_rate, mono, bits_per_sample)
            info = audio_info(dest)
        sample_rate, samples, _, channels = info
    except Exception as error:  # noqa: BLE001 -- reported, not swallowed
        if dest.exists():
            dest.unlink(missing_ok=True)
        return _Outcome(index, None, False, f"{type(error).__name__}: {error}")

    return _Outcome(
        index,
        AudioRecord(
            uttid=record.uttid,
            spkid=record.spkid,
            gender=record.gender,
            path=dest,
            length=int(samples),
            sample_rate=int(sample_rate),
            channels=int(channels),
            tags=record.tags,
        ),
        reused,
        None,
    )


def build_resampled_tree(
    records: Sequence[AudioRecord],
    *,
    source_root: str | Path,
    dest_root: str | Path,
    target_sample_rate: int,
    mono: bool = True,
    bits_per_sample: int | None = 16,
    jobs: int | None = None,
    overwrite: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[AudioRecord], ResampleReport]:
    """Convert every record's audio into ``dest_root`` and return repointed records.

    Resumable: an existing non-empty destination is reused, its header re-read. A
    file that fails to convert is dropped from the returned records and listed in
    the report -- one unreadable source out of 200k should not end a conversion
    that takes an hour, but it must not silently become a metafile row pointing at
    a file that is not there either.

    ``jobs=1`` runs inline, without a pool, which is what you want under a
    debugger and what the tests use.
    """
    source_root = Path(source_root).expanduser().resolve()
    dest_root = Path(dest_root).expanduser().resolve()
    if jobs is None:
        jobs = min(16, os.cpu_count() or 1)
    jobs = max(1, jobs)
    ensure_distinct_destinations(records, source_root, dest_root)

    jobs_list = [
        (
            index,
            record,
            mirrored_path(record.path, source_root, dest_root),
            target_sample_rate,
            mono,
            bits_per_sample,
            overwrite,
        )
        for index, record in enumerate(records)
    ]

    report = ResampleReport()
    converted: list[AudioRecord | None] = [None] * len(records)

    def absorb(outcome: _Outcome) -> None:
        if outcome.error is not None:
            report.failed.append((records[outcome.index].path, outcome.error))
            return
        converted[outcome.index] = outcome.record
        if outcome.reused:
            report.reused += 1
        else:
            report.converted += 1

    if jobs == 1:
        results = map(_work, jobs_list)
    else:
        pool = ProcessPoolExecutor(max_workers=jobs)
        results = pool.map(_work, jobs_list, chunksize=16)

    try:
        for done, outcome in enumerate(results, start=1):
            absorb(outcome)
            if progress is not None and done % 500 == 0:
                progress(done, len(records))
    finally:
        if jobs != 1:
            pool.shutdown()

    if progress is not None and records:
        progress(len(records), len(records))

    return [record for record in converted if record is not None], report


def default_resample_root(source_root: Path, target_sample_rate: int) -> Path:
    """``.../datasets_fullband/clean_fullband`` -> ``.../datasets_fullband_16k/clean_fullband``.

    A sibling tree rather than a subdirectory, so a later scan of the original
    root does not walk the converted copies as if they were more corpus.
    """
    container = source_root.parent
    return (
        container.parent
        / f"{container.name}_{target_sample_rate // 1000}k"
        / source_root.name
    )


def add_resample_arguments(parser) -> None:
    """The ``--resample-*`` flags, so every corpus module spells them the same.

    Pre-converting is not only about disk: pointing the recipe at a 16 kHz tree
    instead of resampling 48 kHz files in the data loader makes the whole
    training pipeline noticeably faster.
    """
    parser.add_argument(
        "--resample-to",
        type=int,
        default=None,
        help="Convert into a mirrored tree at this rate first, and point the "
        "output at the converted files.",
    )
    parser.add_argument("--resample-root", type=str, default=None)
    parser.add_argument("--jobs", type=int, default=None)


def resample_if_requested(
    records: Sequence[AudioRecord],
    *,
    source_root: Path,
    args,
) -> list[AudioRecord]:
    """Honour ``--resample-to`` from :func:`add_resample_arguments`, or pass through."""
    if not getattr(args, "resample_to", None):
        return list(records)

    dest_root = (
        Path(args.resample_root).expanduser().resolve()
        if args.resample_root
        else default_resample_root(source_root, args.resample_to)
    )
    print(f"resampling {len(records)} file(s) -> {dest_root} @ {args.resample_to} Hz")
    records, report = build_resampled_tree(
        records,
        source_root=source_root,
        dest_root=dest_root,
        target_sample_rate=args.resample_to,
        jobs=getattr(args, "jobs", None),
        progress=lambda done, total: print(f"  {done}/{total}", flush=True),
    )
    print(f"  {report.summary()}")
    for source, reason in report.failed[:10]:
        print(f"  FAILED {source}: {reason}")
    if len(report.failed) > 10:
        print(f"  ... and {len(report.failed) - 10} more")
    return records
