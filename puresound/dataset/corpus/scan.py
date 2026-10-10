"""Turn a folder of audio into records, and split those records without leakage.

Corpus-agnostic on purpose: what changes between corpora is where the speaker
identity hides in the path and what extra metadata the filenames carry, and both
are parameters here rather than a new copy of this file per corpus.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Mapping, Sequence

from puresound.audio.io import AudioIO

from .records import AudioRecord


DEFAULT_AUDIO_SUFFIXES: tuple[str, ...] = (".wav", ".flac")

#: Where the speaker identity sits relative to the scanned root.
SPEAKER_ID_STRATEGIES: tuple[str, ...] = (
    "parent",           # .../<speaker>/utt.wav
    "grandparent",      # .../<speaker>/<chapter>/utt.wav
    "path-prefix",      # the first two directory levels, joined
    "filename-prefix",  # <speaker>_<rest>.wav
    "per-file",         # no speaker structure at all: every file is its own speaker
)

#: Corpora that flatten every utterance into one directory still name the speaker
#: somewhere in the filename. ``speaker_pattern`` searches the path relative to the
#: scanned root and takes the first group that matched, which is the only way to get
#: a speaker-disjoint split out of a tree the directory layout says is one speaker.
#: It may also be several patterns, tried in order: one DNS subset can bundle
#: corpora that each hide the speaker somewhere else (M-AILABS in a directory,
#: OpenSLR in the filename).

#: How an utterance id is built.
#:
#: ``digest`` is collision-proof across corpora, but it appends a hash, so anything
#: that parses structure out of the id (``..._seg_2`` -> segment 2) sees the hash and
#: not the structure. ``stem`` keeps the corpus's own naming, which is what such a
#: consumer needs -- at the cost of colliding when two files share a basename.
UTT_ID_STYLES: tuple[str, ...] = ("digest", "stem")

#: A callable that reads extra metadata off a path. Return an empty mapping when
#: the path does not carry any -- never a guess, because a wrong category is worse
#: than a missing one when the point of the tag is per-category diagnosis.
Tagger = Callable[[Path], Mapping[str, Any]]


def sanitize_id(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.-")
    return value or "unknown"


def parse_suffixes(value: str | Sequence[str]) -> tuple[str, ...]:
    items = value.split(",") if isinstance(value, str) else list(value)
    suffixes = []
    for item in items:
        item = item.strip().lower()
        if item:
            suffixes.append(item if item.startswith(".") else f".{item}")
    return tuple(suffixes) or DEFAULT_AUDIO_SUFFIXES


def iter_audio_files(folder: Path, suffixes: Sequence[str] = DEFAULT_AUDIO_SUFFIXES) -> Iterator[Path]:
    wanted = {suffix.lower() for suffix in suffixes}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in wanted:
            yield path


def audio_info(path: Path) -> tuple[int, int, float, int]:
    """``(sample_rate, samples, seconds, channels)``, header-only where possible."""
    try:
        return AudioIO.audio_info(str(path))
    except AttributeError:
        wav, sample_rate = AudioIO.open(str(path))
        samples = wav.shape[-1]
        channels = wav.shape[0] if wav.dim() > 1 else 1
        return sample_rate, samples, round(samples / sample_rate, 2), channels


def speaker_id(
    path: Path,
    root: Path,
    strategy: str,
    prefix: str,
    pattern: "re.Pattern[str] | Sequence[re.Pattern[str]] | None" = None,
) -> str:
    if pattern is not None:
        patterns = [pattern] if isinstance(pattern, re.Pattern) else list(pattern)
        relative = path.relative_to(root).as_posix()
        match = next(
            (found for found in (p.search(relative) for p in patterns) if found), None
        )
        if match is None:
            shown = [p.pattern for p in patterns]
            raise ValueError(
                f"speaker_pattern {shown if len(shown) > 1 else shown[0]!r} did not "
                f"match {path.name!r}. A path the pattern misses would silently "
                "become its own speaker and leak across the split."
            )
        # First group that took part in the match, so alternatives can each
        # capture their own group.
        value = next((group for group in match.groups() if group is not None), None)
        value = match.group(0) if value is None else value
        return f"{prefix}_{sanitize_id(value)}" if prefix else sanitize_id(value)

    if strategy not in SPEAKER_ID_STRATEGIES:
        raise ValueError(
            f"Unsupported speaker id strategy {strategy!r}; expected one of {SPEAKER_ID_STRATEGIES}"
        )

    parts = path.relative_to(root).parent.parts

    # A directory strategy on a tree that has no such directory must not fall back
    # to the filename: that turns every utterance into its own speaker -- and a
    # speaker-disjoint split over one-utterance speakers is disjoint in name only.
    # Say the layout does not match instead; `per-file` is how you ask for that on
    # purpose, and `speaker_pattern` is how a flat tree names its speakers.
    needed = {"parent": 1, "grandparent": 2, "path-prefix": 1}.get(strategy, 0)
    if len(parts) < needed:
        raise ValueError(
            f"speaker_id_strategy={strategy!r} needs at least {needed} directory "
            f"level(s) under the scanned root, but {path.name!r} has {len(parts)}. "
            "Use speaker_pattern= for a flat tree that names its speaker in the "
            "filename, or speaker_id_strategy='per-file' if every file really is "
            "its own speaker."
        )

    if strategy == "parent":
        value = parts[-1]
    elif strategy == "grandparent":
        value = parts[-2]
    elif strategy == "path-prefix":
        value = "_".join(parts[:2])
    elif strategy == "filename-prefix":
        value = re.split(r"[-_.]", path.stem, maxsplit=1)[0]
    else:  # per-file
        value = path.stem

    return f"{prefix}_{sanitize_id(value)}" if prefix else sanitize_id(value)


def utt_id(path: Path, root: Path, prefix: str, style: str = "digest") -> str:
    """Build an utterance id -- see :data:`UTT_ID_STYLES` for the trade-off."""
    if style not in UTT_ID_STYLES:
        raise ValueError(f"utt_id_style must be one of {UTT_ID_STYLES}, got {style!r}")

    if style == "stem":
        return path.stem

    relative = path.relative_to(root)
    readable = sanitize_id(relative.with_suffix("").as_posix().replace("/", "_"))
    digest = hashlib.sha1(relative.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{readable}_{digest}" if prefix else f"{readable}_{digest}"


def scan_folder(
    root: str | Path,
    *,
    id_prefix: str = "",
    speaker_id_strategy: str = "parent",
    speaker_pattern: str | Sequence[str] | None = None,
    utt_id_style: str = "digest",
    gender: str = "None",
    suffixes: Sequence[str] = DEFAULT_AUDIO_SUFFIXES,
    min_duration: float = 0.0,
    max_duration: float | None = None,
    max_files: int | None = None,
    skip_unreadable: bool = True,
    tagger: Tagger | None = None,
) -> list[AudioRecord]:
    """Scan ``root`` recursively into records, in sorted path order.

    ``max_files`` caps how many files are *considered*, not how many survive the
    duration filters, so a capped scan stays a prefix of the full one and a smoke
    test sees the same files every run.

    ``utt_id_style="stem"`` keeps the corpus's own file naming as the utterance id.
    Use it when something downstream parses that naming; it raises on a collision
    rather than letting two files quietly share an id.
    """
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Not a directory: {root}")
    if not speaker_pattern:
        compiled = None
    elif isinstance(speaker_pattern, str):
        compiled = re.compile(speaker_pattern)
    else:
        compiled = [re.compile(item) for item in speaker_pattern]

    records: list[AudioRecord] = []
    for index, path in enumerate(iter_audio_files(root, suffixes)):
        if max_files is not None and index >= max_files:
            break

        try:
            sample_rate, samples, seconds, channels = audio_info(path)
        except Exception:
            if skip_unreadable:
                continue
            raise

        if seconds < min_duration:
            continue
        if max_duration is not None and seconds > max_duration:
            continue

        records.append(
            AudioRecord(
                uttid=utt_id(path, root, id_prefix, utt_id_style),
                spkid=speaker_id(path, root, speaker_id_strategy, id_prefix, compiled),
                gender=gender,
                path=path,
                length=int(samples),
                sample_rate=int(sample_rate),
                channels=int(channels),
                tags=dict(tagger(path)) if tagger is not None else {},
            )
        )

    seen: dict[str, Path] = {}
    for record in records:
        if record.uttid in seen:
            raise ValueError(
                f"Duplicate utterance id {record.uttid!r} from {seen[record.uttid]} "
                f"and {record.path}. utt_id_style={utt_id_style!r} does not "
                "distinguish them; use utt_id_style='digest'."
            )
        seen[record.uttid] = record.path

    return records


def drop_duplicate_files(
    records: Sequence[AudioRecord],
) -> tuple[list[AudioRecord], list[AudioRecord]]:
    """Keep the first of every group of files with the same name and size.

    The DNS clean tree ships whole copies of itself: ``read_speech/read_speech/``
    repeats files from the level above it, and the Spanish, emotional and German
    Wikipedia subsets each carry a second flat copy. Every copy gets its own
    ``uttid`` (the id includes the relative path), so nothing downstream notices --
    the hours double and the duplicated speakers are drawn twice as often. Name
    *and* size, because two different recordings that happen to share a basename
    are not duplicates. Returns ``(kept, dropped)``; order is preserved.
    """
    kept: list[AudioRecord] = []
    dropped: list[AudioRecord] = []
    seen: set[tuple[str, int]] = set()
    for record in records:
        key = (record.path.name, record.path.stat().st_size)
        if key in seen:
            dropped.append(record)
            continue
        seen.add(key)
        kept.append(record)
    return kept, dropped


def split_records(
    records: Sequence[AudioRecord],
    *,
    valid_ratio: float,
    seed: int = 1337,
    split_by: str = "speaker",
) -> tuple[list[AudioRecord], list[AudioRecord]]:
    """Split into (train, valid).

    ``split_by="speaker"`` is the default because it is the only mode that keeps
    the two sides disjoint in the way a dev set has to be: with ``"utterance"``
    every dev speaker is also a train speaker, and the dev loss then reports how
    well the model memorised voices it was trained on.
    """
    if not 0 <= valid_ratio < 1:
        raise ValueError("valid_ratio must satisfy 0 <= valid_ratio < 1.")
    if not records:
        return [], []
    if valid_ratio == 0:
        return list(records), []

    rng = random.Random(seed)
    by_uttid = lambda item: item.uttid  # noqa: E731

    if split_by == "utterance":
        shuffled = list(records)
        rng.shuffle(shuffled)
        count = min(max(1, round(len(shuffled) * valid_ratio)), len(shuffled) - 1)
        return sorted(shuffled[count:], key=by_uttid), sorted(shuffled[:count], key=by_uttid)

    if split_by != "speaker":
        raise ValueError(f"Unsupported split mode: {split_by!r}")

    by_speaker: dict[str, list[AudioRecord]] = defaultdict(list)
    for record in records:
        by_speaker[record.spkid].append(record)

    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        # One speaker cannot be split speaker-disjointly. Say so rather than
        # silently returning an utterance split that looks speaker-disjoint.
        raise ValueError(
            "Cannot make a speaker-disjoint split from a single speaker. Use a "
            "different speaker_id_strategy, or split_by='utterance' knowing the "
            "dev set will not be speaker-disjoint."
        )

    rng.shuffle(speakers)
    count = min(max(1, round(len(speakers) * valid_ratio)), len(speakers) - 1)
    valid_speakers = set(speakers[:count])

    train = [record for record in records if record.spkid not in valid_speakers]
    valid = [record for record in records if record.spkid in valid_speakers]
    return sorted(train, key=by_uttid), sorted(valid, key=by_uttid)


def assert_disjoint(
    train: Iterable[AudioRecord], valid: Iterable[AudioRecord], *, by: str = "spkid"
) -> None:
    """Raise if the two sides share a key. Call it; do not assume the split held."""
    left = {getattr(record, by) for record in train}
    right = {getattr(record, by) for record in valid}
    shared = left & right
    if shared:
        sample = sorted(shared)[:5]
        raise AssertionError(
            f"train and valid share {len(shared)} {by} value(s), e.g. {sample}"
        )
