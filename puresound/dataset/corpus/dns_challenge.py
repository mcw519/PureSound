"""DNS Challenge corpus preparation: metafiles, a fixed-rate tree, and the dev set.

Three subcommands, because the three parts of the corpus are used differently:

``speech``
    Clean speech becomes the train/valid metafiles a recipe reads. The split is
    speaker-disjoint, and for the flattened ``read_speech`` tree that needs the
    speaker read out of the filename -- see :data:`SUBSET_SPEAKER_PATTERNS`.

``noise``
    Noise is never listed in a metafile; the synthesis pipeline draws it from a
    folder. So this subcommand exists to produce that folder at the training
    sample rate, and an inventory beside it.

``devset``
    The shipped dev set has no clean reference -- it is scored no-reference. What
    it does have is a **noise category and a capture device in every filename**,
    which is the only per-category axis this corpus gives us for free. Diagnosis
    needs that axis: noise suppression fails per class, and a mean over classes
    hides which one.

The training noise itself carries no labels -- ``noise_fullband`` files are named
after AudioSet clip ids. Nothing here invents one; ``tagger`` is the hook for
a label source when there is one.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from .records import AudioRecord, tag_histogram, write_inventory, write_metafile
from .resample import add_resample_arguments, resample_if_requested
from .scan import (
    assert_disjoint,
    drop_duplicate_files,
    parse_suffixes,
    sanitize_id,
    scan_folder,
    split_records,
)


DEFAULT_CLEAN_CANDIDATES = (
    "datasets_fullband/clean_fullband",
    "clean_fullband",
    "datasets/clean",
    "clean",
)
DEFAULT_NOISE_CANDIDATES = (
    "datasets_fullband/noise_fullband",
    "noise_fullband",
    "datasets/noise",
    "noise",
)
DEFAULT_RIR_CANDIDATES = (
    "datasets_fullband/impulse_responses",
    "impulse_responses",
    "datasets/impulse_responses",
    "rir",
)
DEFAULT_DEVSET_CANDIDATES = (
    "datasets_fullband/dev_testset/noisy_testclips",
    "dev_testset/noisy_testclips",
    "dev_testset",
)

#: M-AILABS keeps ``<locale>/<female|male>/<speaker>/<book>/wavs/``; its ``mix``
#: books are read by several people and name none of them, so the book stands in
#: for the speaker there -- a split by book is still disjoint, just coarser.
_M_AILABS = (
    r"M-AILABS_Speech_Dataset/[^/]+/(?:fe)?male/([^/]+)/",
    r"M-AILABS_Speech_Dataset/[^/]+/(mix/[^/]+)/",
)

#: Where the speaker identity lives in each clean-speech subset. ``read_speech``
#: is one flat directory of ``book_..._reader_<id>_..._seg_N.wav``, so without the
#: pattern every utterance looks like its own speaker and a "speaker-disjoint"
#: split is disjoint in name only. A tuple is tried in order: most non-English
#: subsets bundle corpora that each put the speaker somewhere else.
SUBSET_SPEAKER_PATTERNS: dict[str, str | tuple[str, ...]] = {
    "read_speech": r"reader_(\d+)",
    "french_speech": _M_AILABS,
    "italian_speech": _M_AILABS,
    "russian_speech": _M_AILABS,
    # The Spoken Wikipedia copy names the article, not the reader (far more
    # articles than readers; long ones split into ``audio2``, ``audio3``). One
    # article is one reader, so the article is a sound unit to split on, but it
    # is not a speaker count -- weight this source by share, never by how many
    # "speakers" it appears to have.
    "german_speech": _M_AILABS
    + (r"(?:^|/)German_Wikipedia_([^/]+?)_audio\d*_48kHz(?:_seg_\d+)?\.wav$",),
    "spanish_speech": _M_AILABS
    + (
        # OpenSLR 39 in two namings: <speaker>_<utt>, and a descriptor string
        # that ends in the speaker code.
        r"(?:^|/)(SLR39_\d+)_\d+_48kHz(?:_seg_\d+)?\.wav$",
        r"(?:^|/)SLR39_(?:non)?native-[^/]*-([a-z]{3}\d+)_[a-z]\d+_48kHz(?:_seg_\d+)?\.wav$",
        # The crowdsourced OpenSLR 61/71/73/74/75 sets: <region><sex>_<id>.
        r"(?:^|/)SLR\d+_es[-_][a-z]{2}_(?:(?:fe)?male_)?([a-z]{3}_\d{5})_\d+_48kHz",
    ),
    "emotional_speech": r"(?:^|/)(\d{4})_[A-Z]{3}_[A-Z]{3}_[A-Z0-9]{2}\.wav$",
    "VocalSet_48kHz_mono": r"(?:^|/)vocalset_((?:fe)?male\d+)_",
}

#: Subsets whose directory layout already is one directory per speaker.
SUBSET_SPEAKER_STRATEGIES: dict[str, str] = {
    "vctk_wav48_silence_trimmed": "parent",
}

#: Speakers a subset must never contribute, by their id before ``--id-prefix``.
#: p232 and p257 are the two VoiceBank-DEMAND test speakers, and DNS ships their
#: whole VCTK recordings -- training on them would leak that test set into
#: training.
SUBSET_EXCLUDED_SPEAKERS: dict[str, tuple[str, ...]] = {
    "vctk_wav48_silence_trimmed": ("p232", "p257"),
}


def drop_excluded_speakers(
    records: Sequence[AudioRecord], excluded: Sequence[str], id_prefix: str
) -> tuple[list[AudioRecord], list[AudioRecord]]:
    """Split ``records`` into (kept, dropped) by speaker id before the prefix."""
    unwanted = {f"{id_prefix}_{sanitize_id(s)}" if id_prefix else sanitize_id(s) for s in excluded}
    kept = [record for record in records if record.spkid not in unwanted]
    dropped = [record for record in records if record.spkid in unwanted]
    return kept, dropped


# --------------------------------------------------------------------------- #
# Layout resolution
# --------------------------------------------------------------------------- #


def resolve_subdir(
    root: Path, override: str | None, candidates: Sequence[str]
) -> Path | None:
    """Find one part of the corpus, preferring an explicit override."""
    if override:
        path = Path(override).expanduser()
        if not path.is_absolute():
            path = root / path
        return path.resolve()

    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path.resolve()
    return None


# --------------------------------------------------------------------------- #
# Dev-set filename metadata
# --------------------------------------------------------------------------- #

# Names look like
#   ms_pns_HPdesktop_A1QUQ0TV9KVD4C_copy_machine_primarywithnoise_fileid_3.wav
#   ms_emotional_Happy_pns_Lenevo_Laptop_A8B4AE3QMECV_traffic_noise_fileid_7.wav
#   ms_pns_ASUS_X205TA_clattering_primarywithnoise_fileid_1.wav
# The category is what sits between the last identifier-looking token (a worker id,
# or failing that a device model number) and the trailing flags.
_WORKER_ID = re.compile(r"^A[A-Z0-9]{8,}$")
_DEVICE_MODEL = re.compile(r"^[A-Z]{1,4}\d{2,}[A-Z]*$")
_CHAIN_FLAGS = frozenset({"ms", "pns", "realrec", "emotional"})
_EMOTIONS = frozenset(
    {"sad", "happy", "angry", "anger", "crying", "neutral", "fear", "disgust", "surprise"}
)
_TRAILING = frozenset(
    {"primarywithnoise", "withnoise", "fileid", "far", "near", "clean", "noise", "sound"}
) | _EMOTIONS
_GENERIC_TAIL = frozenset({"noise", "noises", "sound", "sounds", "noice"})

#: DNS-5 dev-set spellings that name the same source. The corpus was labelled by
#: hand, so the same fan appears as ``fan``, ``fan_noise`` and ``fannoice``.
#: Counting those as three classes makes every class too small to read. Anything
#: not listed keeps its own normalised spelling rather than being forced into a
#: neighbour -- a wrong merge is worse than a long tail.
CATEGORY_ALIASES: dict[str, str] = {
    "fannoises": "fan", "fannoice": "fan", "ceilingfan": "fan",
    "typingnoises": "typing", "runningtyping": "typing",
    "waterrunning": "runningwater",
    "shuttingdoor": "doorshutting", "doorshut": "doorshutting", "door": "doorshutting",
    "doort": "doorshutting", "doorshunting": "doorshutting",
    "doorshuttering": "doorshutting",
    "dogbark": "dogbarking", "dog": "dogbarking", "runningdogbarking": "dogbarking",
    "baby": "babycrying", "runningbaby": "babycrying",
    "kitchennoise": "kitchen", "kitchennoice": "kitchen", "utensils": "kitchen",
    "untensils": "kitchen", "dishscrubbing": "kitchen",
    "mouseclick": "mouseclicks", "mouseclicking": "mouseclicks",
    "mouceclicks": "mouseclicks", "mouseclickingnoises": "mouseclicks",
    "mousescroll": "mouseclicks", "mousescrollwheel": "mouseclicks",
    "touchpadclicks": "mouseclicks", "runningmouseclicks": "mouseclicks",
    "mouseclicksmousescrollwheeltouchpadclicks": "mouseclicks",
    "airconditioning": "airconditioner", "aircondition": "airconditioner",
    "clattering": "clatter", "clatternoise": "clatter", "clatteringnoise": "clatter",
    "creakingchairs": "creakingchair", "chaircreaking": "creakingchair",
    "creekingchair": "creakingchair", "runningcreakingchair": "creakingchair",
    "openingchipbag": "openingchipspacket", "bagofchips": "openingchipspacket",
    "openingpacketofchips": "openingchipspacket",
    "openinigchipspacket": "openingchipspacket",
    "openingchippacket": "openingchipspacket",
    "munchingoreating": "munching", "munchingoreatingwith": "munching",
    "eating": "munching", "eatingwith": "munching",
    "insidecar": "car", "outsidecar": "car", "passingcars": "car",
    "carnoise": "car", "carrunning": "car", "busystreet": "traffic",
    "breathing": "heavybreathing",
}


def parse_devset_name(name: str) -> dict[str, str]:
    """Pull ``{device, worker, category, category_raw}`` out of a dev-set filename.

    Returns an empty dict when the name has no identifier token to anchor on,
    rather than guessing where the device stops and the noise starts.
    """
    stem = name
    while stem.lower().endswith(".wav"):
        stem = stem[: -len(".wav")]

    tokens = [token for token in re.split(r"[_.]+", stem) if token]
    anchors = [
        index
        for index, token in enumerate(tokens)
        if _WORKER_ID.match(token) or _DEVICE_MODEL.match(token)
    ]
    if not anchors:
        return {}

    anchor = anchors[-1]
    device = [
        token
        for token in tokens[:anchor]
        if token.lower() not in _CHAIN_FLAGS and token.lower() not in _EMOTIONS
    ]

    category: list[str] = []
    for token in tokens[anchor + 1 :]:
        if token.lower() in _TRAILING or token.isdigit():
            break
        category.append(token)
    while category and category[-1].lower() in _GENERIC_TAIL:
        category.pop()

    raw = "_".join(category).lower()
    normalised = re.sub(r"[^a-z0-9]", "", raw)
    return {
        "device": "_".join(device).lower() or "unknown",
        "worker": tokens[anchor],
        "category_raw": raw or "unknown",
        "category": CATEGORY_ALIASES.get(normalised, normalised or "unknown"),
    }


def devset_tagger(path: Path) -> Mapping[str, Any]:
    return parse_devset_name(path.name)


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #


def run_speech(args: argparse.Namespace) -> int:
    root = Path(args.dns_root).expanduser().resolve()
    clean_root = resolve_subdir(root, args.clean_dir, DEFAULT_CLEAN_CANDIDATES)
    if clean_root is None or not clean_root.is_dir():
        print(
            "Cannot find the clean-speech folder. Pass --clean-dir, or use a DNS "
            "root containing datasets_fullband/clean_fullband.",
            file=sys.stderr,
        )
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    subsets = args.subset or [""]
    records: list[AudioRecord] = []
    for subset in subsets:
        subset_root = clean_root / subset if subset else clean_root
        if not subset_root.is_dir():
            print(f"No such subset: {subset_root}", file=sys.stderr)
            return 1
        pattern = args.speaker_pattern or SUBSET_SPEAKER_PATTERNS.get(subset)
        strategy = args.speaker_id_strategy or SUBSET_SPEAKER_STRATEGIES.get(subset, "parent")
        print(f"scanning {subset_root} (speaker: {pattern or strategy})")
        found = scan_folder(
            subset_root,
            id_prefix=args.id_prefix,
            speaker_id_strategy=strategy,
            speaker_pattern=pattern,
            utt_id_style=args.utt_id_style,
            gender=args.gender,
            suffixes=parse_suffixes(args.audio_suffixes),
            min_duration=args.min_duration,
            max_files=args.max_files,
        )
        if not args.keep_duplicates:
            found, duplicates = drop_duplicate_files(found)
            if duplicates:
                print(f"  dropped {len(duplicates)} duplicate file(s) (same name and size)")
        excluded = tuple(args.exclude_speaker or ()) + SUBSET_EXCLUDED_SPEAKERS.get(subset, ())
        if excluded:
            found, gone = drop_excluded_speakers(found, excluded, args.id_prefix)
            print(f"  excluded {len(gone)} utterance(s) of speaker(s) {sorted(set(excluded))}")
        print(f"  {len(found)} utterance(s), {len({r.spkid for r in found})} speaker(s)")
        records.extend(found)

    if not records:
        print("Found no audio.", file=sys.stderr)
        return 1

    records = resample_if_requested(
        records, source_root=clean_root, args=args
    )

    train, valid = split_records(
        records, valid_ratio=args.valid_ratio, seed=args.seed, split_by=args.split_by
    )
    if args.split_by == "speaker":
        assert_disjoint(train, valid, by="spkid")

    train_path = (
        Path(args.train_metafile).expanduser()
        if args.train_metafile
        else output_dir / f"{args.id_prefix}_train.csv"
    )
    valid_path = (
        Path(args.valid_metafile).expanduser()
        if args.valid_metafile
        else output_dir / f"{args.id_prefix}_valid.csv"
    )
    write_metafile(train_path, train)
    write_metafile(valid_path, valid)
    print(f"train_metafile: {train_path} ({len(train)} rows, {len({r.spkid for r in train})} speakers)")
    print(f"valid_metafile: {valid_path} ({len(valid)} rows, {len({r.spkid for r in valid})} speakers)")
    return 0


def run_noise(args: argparse.Namespace) -> int:
    root = Path(args.dns_root).expanduser().resolve()
    noise_root = resolve_subdir(root, args.noise_dir, DEFAULT_NOISE_CANDIDATES)
    if noise_root is None or not noise_root.is_dir():
        print("Cannot find the noise folder. Pass --noise-dir.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    print(f"scanning {noise_root}")
    records = scan_folder(
        noise_root,
        id_prefix=f"{args.id_prefix}_noise",
        speaker_id_strategy="per-file",
        utt_id_style=args.utt_id_style,
        gender="None",
        suffixes=parse_suffixes(args.audio_suffixes),
        min_duration=args.min_duration,
        max_files=args.max_files,
    )
    print(f"  {len(records)} noise file(s)")
    records = resample_if_requested(
        records, source_root=noise_root, args=args
    )

    inventory = output_dir / f"{args.id_prefix}_noise.jsonl"
    write_inventory(inventory, records)
    folders = sorted({record.path.parent for record in records})
    print(f"noise_inventory: {inventory} ({len(records)} files)")
    print("point augmentation_noise.noise_folder at:")
    for folder in folders[:5]:
        print(f"  {folder}")
    if len(folders) > 5:
        print(f"  ... and {len(folders) - 5} more")
    return 0


def run_devset(args: argparse.Namespace) -> int:
    root = Path(args.dns_root).expanduser().resolve()
    devset_root = resolve_subdir(root, args.devset_dir, DEFAULT_DEVSET_CANDIDATES)
    if devset_root is None or not devset_root.is_dir():
        print("Cannot find the dev test set. Pass --devset-dir.", file=sys.stderr)
        return 1

    output_dir = Path(args.output_dir).expanduser().resolve()
    print(f"scanning {devset_root}")
    records = scan_folder(
        devset_root,
        id_prefix=f"{args.id_prefix}_dev",
        speaker_id_strategy="per-file",
        utt_id_style=args.utt_id_style,
        gender="None",
        suffixes=parse_suffixes(args.audio_suffixes),
        max_files=args.max_files,
        tagger=devset_tagger,
    )
    unparsed = [record for record in records if not record.tags]
    inventory = output_dir / f"{args.id_prefix}_devset.jsonl"
    write_inventory(inventory, records)

    print(f"devset_inventory: {inventory} ({len(records)} clips, {len(unparsed)} unparsed)")
    for label, key in (("categories", "category"), ("devices", "device")):
        histogram = tag_histogram(records, key)
        print(f"-- {len(histogram)} {label} --")
        for name, count in list(histogram.items())[: args.top]:
            print(f"  {count:5d}  {name}")
        if len(histogram) > args.top:
            print(f"  ... and {len(histogram) - args.top} more")
    return 0


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("dns_root", type=str, help="Extracted DNS Challenge root.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--id-prefix", type=str, default="dns5")
    parser.add_argument("--audio-suffixes", type=str, default=".wav,.flac")
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument(
        "--utt-id-style",
        choices=["digest", "stem"],
        default="digest",
        help="'stem' keeps the corpus's own file naming as the utterance id -- "
        "needed when something downstream parses it (e.g. '..._seg_2' -> segment 2).",
    )
    parser.add_argument("--max-files", type=int, default=None)
    add_resample_arguments(parser)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.dataset.corpus.dns_challenge",
        description=__doc__.splitlines()[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    speech = sub.add_parser("speech", help="clean speech -> train/valid metafiles")
    _add_common(speech)
    speech.add_argument("--clean-dir", type=str, default=None)
    speech.add_argument(
        "--subset",
        action="append",
        default=None,
        help="Subset under clean_fullband, e.g. read_speech. Repeatable; "
        "omitted means the whole clean folder.",
    )
    speech.add_argument("--speaker-id-strategy", type=str, default=None)
    speech.add_argument("--speaker-pattern", type=str, default=None)
    speech.add_argument(
        "--exclude-speaker",
        action="append",
        default=None,
        help="Speaker id (before --id-prefix) to leave out, on top of the subset's "
        "built-in exclusions. Repeatable.",
    )
    speech.add_argument(
        "--keep-duplicates",
        action="store_true",
        help="Keep files whose name and size repeat elsewhere in the subset "
        "(dropped by default: DNS ships whole second copies of several subsets).",
    )
    speech.add_argument("--gender", type=str, default="None")
    speech.add_argument(
        "--train-metafile",
        type=str,
        default=None,
        help="Output path; defaults to <output-dir>/<id-prefix>_train.csv.",
    )
    speech.add_argument("--valid-metafile", type=str, default=None)
    speech.add_argument("--valid-ratio", type=float, default=0.05)
    speech.add_argument("--seed", type=int, default=1337)
    speech.add_argument("--split-by", choices=["speaker", "utterance"], default="speaker")
    speech.set_defaults(func=run_speech)

    noise = sub.add_parser("noise", help="noise -> fixed-rate folder + inventory")
    _add_common(noise)
    noise.add_argument("--noise-dir", type=str, default=None)
    noise.set_defaults(func=run_noise)

    devset = sub.add_parser("devset", help="dev test set -> inventory with categories")
    _add_common(devset)
    devset.add_argument("--devset-dir", type=str, default=None)
    devset.add_argument("--top", type=int, default=20)
    devset.set_defaults(func=run_devset)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
