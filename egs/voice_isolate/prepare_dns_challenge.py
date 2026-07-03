import argparse
import hashlib
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from puresound.audio.io import AudioIO


DEFAULT_AUDIO_SUFFIXES = (".wav", ".flac")
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


@dataclass(frozen=True)
class SpeechRecord:
    uttid: str
    spkid: str
    gender: str
    path: Path
    length: int
    sample_rate: int
    channels: int


def _audio_info(path: Path) -> tuple[int, int, float, int]:
    try:
        return AudioIO.audio_info(str(path))
    except AttributeError:
        wav, sample_rate = AudioIO.open(str(path))
        total_samples = wav.shape[-1]
        total_seconds = round(total_samples / sample_rate, 2)
        channels = wav.shape[0] if wav.dim() > 1 else 1
        return sample_rate, total_samples, total_seconds, channels


def _sanitize_id(value: str) -> str:
    value = value.strip().replace(" ", "_")
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_.-")
    return value or "unknown"


def _parse_suffixes(value: str) -> tuple[str, ...]:
    suffixes = []
    for item in value.split(","):
        item = item.strip().lower()
        if not item:
            continue
        suffixes.append(item if item.startswith(".") else f".{item}")
    return tuple(suffixes) or DEFAULT_AUDIO_SUFFIXES


def _resolve_subdir(root: Path, override: str | None, candidates: Sequence[str]) -> Path | None:
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


def _iter_audio_files(folder: Path, suffixes: Sequence[str]) -> Iterable[Path]:
    suffix_set = {suffix.lower() for suffix in suffixes}
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in suffix_set:
            yield path


def _speaker_id(path: Path, clean_root: Path, strategy: str) -> str:
    rel = path.relative_to(clean_root)
    parent_parts = rel.parent.parts

    if strategy == "parent":
        value = parent_parts[-1] if parent_parts else path.stem
    elif strategy == "grandparent":
        value = parent_parts[-2] if len(parent_parts) >= 2 else (
            parent_parts[-1] if parent_parts else path.stem
        )
    elif strategy == "path-prefix":
        value = "_".join(parent_parts[:2]) if parent_parts else path.stem
    elif strategy == "filename-prefix":
        value = re.split(r"[-_.]", path.stem, maxsplit=1)[0]
    else:
        raise ValueError(f"Unsupported speaker id strategy: {strategy}")

    return f"dns4_{_sanitize_id(value)}"


def _utt_id(path: Path, clean_root: Path, prefix: str) -> str:
    rel = path.relative_to(clean_root)
    rel_without_suffix = rel.with_suffix("").as_posix()
    readable = _sanitize_id(rel_without_suffix.replace("/", "_"))
    digest = hashlib.sha1(rel.as_posix().encode("utf-8")).hexdigest()[:10]
    return f"{prefix}_{readable}_{digest}"


def scan_clean_speech(
    clean_root: Path,
    *,
    suffixes: Sequence[str] = DEFAULT_AUDIO_SUFFIXES,
    speaker_id_strategy: str = "parent",
    gender: str = "other",
    utt_prefix: str = "dns4",
    min_duration: float = 0.0,
    max_files: int | None = None,
    skip_bad_files: bool = True,
) -> list[SpeechRecord]:
    records: list[SpeechRecord] = []
    for idx, path in enumerate(_iter_audio_files(clean_root, suffixes)):
        if max_files is not None and idx >= max_files:
            break

        try:
            sample_rate, total_samples, total_seconds, num_channels = _audio_info(path)
        except Exception:
            if skip_bad_files:
                continue
            raise

        if total_seconds < min_duration:
            continue

        records.append(
            SpeechRecord(
                uttid=_utt_id(path, clean_root, utt_prefix),
                spkid=_speaker_id(path, clean_root, speaker_id_strategy),
                gender=gender,
                path=path.resolve(),
                length=int(total_samples),
                sample_rate=int(sample_rate),
                channels=int(num_channels),
            )
        )

    return records


def split_records(
    records: Sequence[SpeechRecord],
    *,
    valid_ratio: float,
    seed: int,
    split_by: str,
) -> tuple[list[SpeechRecord], list[SpeechRecord]]:
    if not 0 <= valid_ratio < 1:
        raise ValueError("--valid-ratio must be >= 0 and < 1.")
    if not records:
        return [], []

    rng = random.Random(seed)
    if valid_ratio == 0:
        return list(records), []

    if split_by == "utterance":
        shuffled = list(records)
        rng.shuffle(shuffled)
        valid_count = max(1, round(len(shuffled) * valid_ratio))
        valid_count = min(valid_count, len(shuffled) - 1)
        return sorted(shuffled[valid_count:], key=lambda item: item.uttid), sorted(
            shuffled[:valid_count], key=lambda item: item.uttid
        )

    if split_by != "speaker":
        raise ValueError(f"Unsupported split mode: {split_by}")

    by_speaker: dict[str, list[SpeechRecord]] = defaultdict(list)
    for record in records:
        by_speaker[record.spkid].append(record)

    speakers = sorted(by_speaker)
    if len(speakers) < 2:
        return split_records(
            records, valid_ratio=valid_ratio, seed=seed, split_by="utterance"
        )

    rng.shuffle(speakers)
    valid_speaker_count = max(1, round(len(speakers) * valid_ratio))
    valid_speaker_count = min(valid_speaker_count, len(speakers) - 1)
    valid_speakers = set(speakers[:valid_speaker_count])

    train_records = []
    valid_records = []
    for record in records:
        if record.spkid in valid_speakers:
            valid_records.append(record)
        else:
            train_records.append(record)

    return sorted(train_records, key=lambda item: item.uttid), sorted(
        valid_records, key=lambda item: item.uttid
    )


def write_metafile(path: Path, records: Sequence[SpeechRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("uttid, spkid, gender, path, length, sample rate, channels\n")
        for item in records:
            f.write(
                f"{item.uttid}, {item.spkid}, {item.gender}, {item.path}, "
                f"{item.length}, {item.sample_rate}, {item.channels}\n"
            )


def write_config_hint(
    path: Path,
    *,
    train_metafile: Path,
    valid_metafile: Path,
    noise_folder: Path | None,
    rir_folder: Path | None,
) -> None:
    lines = [
        "dataset:",
        f"  train_metafile: {train_metafile}",
        f"  valid_metafile: {valid_metafile}",
    ]
    if noise_folder is not None:
        lines.extend(["augmentation_noise:", "  used: True", f"  noise_folder: {noise_folder}"])
    if rir_folder is not None:
        lines.extend(["augmentation_reverb:", f"  rir_folder: {rir_folder}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def prepare_dns_challenge_metafiles(args: argparse.Namespace) -> tuple[Path, Path]:
    dns_root = Path(args.dns_root).expanduser().resolve()
    clean_root = _resolve_subdir(dns_root, args.clean_dir, DEFAULT_CLEAN_CANDIDATES)
    if clean_root is None or not clean_root.is_dir():
        raise FileNotFoundError(
            "Cannot find DNS clean speech folder. Pass --clean-dir explicitly, "
            "or use a DNS root containing datasets_fullband/clean_fullband."
        )

    output_dir = Path(args.output_dir).expanduser().resolve()
    train_metafile = (
        Path(args.train_metafile).expanduser().resolve()
        if args.train_metafile
        else output_dir / "voice_isolate_dns4_train.csv"
    )
    valid_metafile = (
        Path(args.valid_metafile).expanduser().resolve()
        if args.valid_metafile
        else output_dir / "voice_isolate_dns4_valid.csv"
    )

    records = scan_clean_speech(
        clean_root,
        suffixes=_parse_suffixes(args.audio_suffixes),
        speaker_id_strategy=args.speaker_id_strategy,
        gender=args.gender,
        utt_prefix=args.utt_prefix,
        min_duration=args.min_duration,
        max_files=args.max_files,
        skip_bad_files=args.skip_bad_files,
    )
    train_records, valid_records = split_records(
        records,
        valid_ratio=args.valid_ratio,
        seed=args.seed,
        split_by=args.split_by,
    )
    write_metafile(train_metafile, train_records)
    write_metafile(valid_metafile, valid_records)

    noise_folder = _resolve_subdir(dns_root, args.noise_dir, DEFAULT_NOISE_CANDIDATES)
    rir_folder = _resolve_subdir(dns_root, args.rir_dir, DEFAULT_RIR_CANDIDATES)
    if args.write_config_hint:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_config_hint(
            output_dir / "voice_isolate_dns4_config_hint.yaml",
            train_metafile=train_metafile,
            valid_metafile=valid_metafile,
            noise_folder=noise_folder if noise_folder and noise_folder.is_dir() else None,
            rir_folder=rir_folder if rir_folder and rir_folder.is_dir() else None,
        )

    print(f"clean_root: {clean_root}")
    print(f"train_metafile: {train_metafile} ({len(train_records)} utterances)")
    print(f"valid_metafile: {valid_metafile} ({len(valid_records)} utterances)")
    if noise_folder and noise_folder.is_dir():
        print(f"noise_folder: {noise_folder}")
    if rir_folder and rir_folder.is_dir():
        print(f"rir_folder: {rir_folder}")

    return train_metafile, valid_metafile


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create PureSound voice_isolate metafiles from DNS-Challenge-4."
    )
    parser.add_argument("dns_root", type=str, help="DNS-Challenge-4 extracted root.")
    parser.add_argument("--output-dir", type=str, default="./data/voice_isolate_dns4")
    parser.add_argument("--train-metafile", type=str, default=None)
    parser.add_argument("--valid-metafile", type=str, default=None)
    parser.add_argument("--clean-dir", type=str, default=None)
    parser.add_argument("--noise-dir", type=str, default=None)
    parser.add_argument("--rir-dir", type=str, default=None)
    parser.add_argument("--audio-suffixes", type=str, default=".wav,.flac")
    parser.add_argument(
        "--speaker-id-strategy",
        choices=["parent", "grandparent", "path-prefix", "filename-prefix"],
        default="parent",
    )
    parser.add_argument("--gender", type=str, default="other")
    parser.add_argument("--utt-prefix", type=str, default="dns4")
    parser.add_argument("--min-duration", type=float, default=0.0)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--split-by", choices=["speaker", "utterance"], default="speaker")
    parser.add_argument(
        "--skip-bad-files",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--write-config-hint",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    return parser


if __name__ == "__main__":
    prepare_dns_challenge_metafiles(build_parser().parse_args())
