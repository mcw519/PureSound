"""VCTK-DEMAND: the standard noise-suppression benchmark, as a known corpus.

Valentini's VCTK + DEMAND set is what most of the noise-suppression literature
reports PESQ on, so a result on it can be read against published work rather than
only against our own previous runs. It ships both sides already paired, so its
test set is imported rather than synthesised -- and it ships transcripts, which is
what lets one set serve both the quality stages and the WER guardrail.

Two subcommands, because the two halves are used differently:

``testset``
    The 824 paired utterances, brought to the training sample rate and described
    by the manifest the scoring tools read.

``speech``
    The clean training speech as train/valid metafiles, for using VCTK as a
    training corpus. Speakers are named in the filename (``p232_001.wav``), so the
    split is speaker-disjoint for real rather than by directory accident.

The noise side is DEMAND, already mixed into the noisy half; this corpus has no
separate noise folder to point ``augmentation_noise`` at. But both halves are
sample-aligned, so ``noisy - clean`` *is* the DEMAND noise, and ``noise`` writes it
back out as one track per noise type. That matters because a model trained on
DNS-5 noise has never heard any of DEMAND's fifteen types -- it is the one noise
pool on disk that is held out for the whole training line, which is what a
harder WER set needs (``evaluation.tools.mix_paired_set``).

Run::

    python -m puresound.dataset.corpus.vctk_demand testset /path/to/audio/vctk_demand \\
        --out-dir egs/noise_suppression/data_report/vctk_demand_test
    python -m puresound.dataset.corpus.vctk_demand noise /path/to/audio/vctk_demand \\
        --out-dir egs/noise_suppression/data/demand_noise
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import torch

from puresound.audio.io import AudioIO

from .paired import write_paired_set
from .records import write_metafile
from .resample import add_resample_arguments, resample_if_requested
from .scan import assert_disjoint, parse_suffixes, scan_folder, split_records


DEFAULT_NOISY_TEST = ("noisy_testset_wav",)
DEFAULT_CLEAN_TEST = ("clean_testset_wav",)
DEFAULT_TEST_TEXT = ("testset_txt",)

#: The corpus ships a 28-speaker and a 56-speaker training split. The 28-speaker
#: one is the split the published baselines train on.
DEFAULT_CLEAN_TRAIN = (
    "clean_trainset_28spk_wav",
    "clean_trainset_56spk_wav",
)
DEFAULT_TRAIN_TEXT = ("trainset_28spk_txt", "trainset_56spk_txt")

#: Each split ships ``log_<split>.txt`` with one ``<utterance> <noise type> <snr>``
#: line per file. It is the only place the noise type is written down; the
#: filenames carry the speaker and nothing else.
NOISE_SPLITS = {
    "testset": ("noisy_testset_wav", "clean_testset_wav", "log_testset.txt"),
    "trainset_28spk": (
        "noisy_trainset_28spk_wav",
        "clean_trainset_28spk_wav",
        "log_trainset_28spk.txt",
    ),
}

#: ``p232_001.wav`` -> speaker ``p232``. The files sit in one flat directory, so
#: without this every utterance would look like its own speaker and a
#: "speaker-disjoint" split would be disjoint in name only.
SPEAKER_PATTERN = r"^(p\d+)_"


def resolve_subdir(root: Path, override: str | None, candidates: Sequence[str]) -> Path | None:
    if override:
        path = Path(override).expanduser()
        return (path if path.is_absolute() else root / path).resolve()
    for candidate in candidates:
        path = root / candidate
        if path.is_dir():
            return path.resolve()
    return None


def read_noise_log(path: Path) -> dict[str, str]:
    """``log_*.txt`` -> ``{utterance stem: noise type}``; the SNR column is dropped
    because ``noisy - clean`` will say what it actually was."""
    types: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 2:
            types[parts[0]] = parts[1]
    return types


@dataclass
class NoiseExtractReport:
    tracks: dict[str, float] = field(default_factory=dict)  # type -> seconds
    files: int = 0
    skipped_unpaired: list[str] = field(default_factory=list)
    skipped_unlogged: list[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"{len(self.tracks)} track(s) from {self.files} file(s)"]
        if self.skipped_unpaired:
            parts.append(f"{len(self.skipped_unpaired)} unpaired skipped")
        if self.skipped_unlogged:
            parts.append(f"{len(self.skipped_unlogged)} not in the log skipped")
        return ", ".join(parts)


def extract_noise(
    root: str | Path,
    out_dir: str | Path,
    *,
    splits: Sequence[str] = ("testset", "trainset_28spk"),
    sample_rate: int = 16000,
    limit_per_type: int | None = None,
) -> NoiseExtractReport:
    """Write ``noisy - clean`` as one ``<type>.wav`` track per DEMAND noise type.

    The subtraction happens after both sides are resampled; resampling is linear,
    so the result is the resampled noise. A pair whose lengths differ is trimmed
    to the shorter -- the corpus does not contain one, and a guard is cheaper
    than an assumption. A type appearing in two splits goes into one track.
    """
    root, out_dir = Path(root).expanduser().resolve(), Path(out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report = NoiseExtractReport()
    pieces: dict[str, list[torch.Tensor]] = defaultdict(list)
    counts: dict[str, int] = defaultdict(int)

    for split in splits:
        if split not in NOISE_SPLITS:
            raise ValueError(f"unknown split {split!r}; choose from {sorted(NOISE_SPLITS)}")
        noisy_name, clean_name, log_name = NOISE_SPLITS[split]
        noisy_dir, clean_dir, log = root / noisy_name, root / clean_name, root / log_name
        if not (noisy_dir.is_dir() and clean_dir.is_dir() and log.is_file()):
            raise FileNotFoundError(
                f"{split}: expected {noisy_name}/, {clean_name}/ and {log_name} under {root}"
            )
        types = read_noise_log(log)
        noisy_files = {p.stem: p for p in noisy_dir.glob("*.wav")}
        clean_files = {p.stem: p for p in clean_dir.glob("*.wav")}
        report.skipped_unpaired += sorted((set(noisy_files) ^ set(clean_files)))
        for stem in sorted(set(noisy_files) & set(clean_files)):
            noise_type = types.get(stem)
            if noise_type is None:
                report.skipped_unlogged.append(stem)
                continue
            if limit_per_type and counts[noise_type] >= limit_per_type:
                continue
            noisy, _ = AudioIO.open(str(noisy_files[stem]), resample_to=sample_rate)
            clean, _ = AudioIO.open(str(clean_files[stem]), resample_to=sample_rate)
            noisy, clean = noisy.reshape(1, -1), clean.reshape(1, -1)
            length = min(noisy.shape[-1], clean.shape[-1])
            pieces[noise_type].append(noisy[..., :length] - clean[..., :length])
            counts[noise_type] += 1
            report.files += 1

    index: dict[str, object] = {}
    for noise_type in sorted(pieces):
        track = torch.cat(pieces[noise_type], dim=-1)
        AudioIO.save(track, str(out_dir / f"{noise_type}.wav"), sample_rate)
        seconds = round(track.shape[-1] / sample_rate, 2)
        report.tracks[noise_type] = seconds
        index[noise_type] = {"files": counts[noise_type], "seconds": seconds}
    (out_dir / "index.json").write_text(
        json.dumps(
            {
                "source": "vctk_demand",
                "root": str(root),
                "splits": list(splits),
                "sample_rate": sample_rate,
                "derivation": "noisy - clean, per utterance, concatenated per noise type",
                "tracks": index,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return report


def run_noise(args: argparse.Namespace) -> int:
    try:
        report = extract_noise(
            args.corpus_root,
            args.out_dir,
            splits=[s.strip() for s in args.splits.split(",") if s.strip()],
            sample_rate=args.sample_rate,
            limit_per_type=args.limit_per_type,
        )
    except (FileNotFoundError, ValueError) as error:
        print(error, file=sys.stderr)
        return 1
    print(f"wrote {report.summary()} -> {Path(args.out_dir).resolve()}")
    for noise_type, seconds in report.tracks.items():
        print(f"  {noise_type:12s} {seconds:8.1f} s")
    return 0


def run_testset(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root).expanduser().resolve()
    noisy = resolve_subdir(root, args.noisy_dir, DEFAULT_NOISY_TEST)
    clean = resolve_subdir(root, args.clean_dir, DEFAULT_CLEAN_TEST)
    if noisy is None or clean is None:
        print(
            "Cannot find noisy_testset_wav / clean_testset_wav. Pass --noisy-dir "
            "and --clean-dir.",
            file=sys.stderr,
        )
        return 1

    transcripts = None
    if not args.no_transcripts:
        transcripts = resolve_subdir(root, args.transcript_dir, DEFAULT_TEST_TEXT)
        if transcripts is None:
            print("No testset_txt found; the WER stage will not be available.")

    print(f"noisy: {noisy}\nclean: {clean}")
    report = write_paired_set(
        noisy, clean, args.out_dir,
        sample_rate=args.sample_rate,
        transcript_dir=transcripts,
        source="vctk_demand",
        limit=args.limit,
        progress=lambda done, total: print(f"  {done}/{total}", flush=True),
    )
    print(f"wrote {report.summary()} -> {Path(args.out_dir).resolve()}")
    print(f"resampled to {args.sample_rate} Hz")
    if transcripts is not None and not report.missing_transcript:
        print("every utterance has a transcript -- the WER stage is available")
    return 0


def run_speech(args: argparse.Namespace) -> int:
    root = Path(args.corpus_root).expanduser().resolve()
    clean = resolve_subdir(root, args.clean_dir, DEFAULT_CLEAN_TRAIN)
    if clean is None:
        print("Cannot find a clean training folder. Pass --clean-dir.", file=sys.stderr)
        return 1

    print(f"scanning {clean}")
    records = scan_folder(
        clean,
        id_prefix=args.id_prefix,
        speaker_pattern=SPEAKER_PATTERN,
        utt_id_style=args.utt_id_style,
        suffixes=parse_suffixes(args.audio_suffixes),
        min_duration=args.min_duration,
        max_files=args.max_files,
    )
    print(f"  {len(records)} utterance(s), {len({r.spkid for r in records})} speaker(s)")
    # VCTK ships at 48 kHz and the recipes train at 16; converting once here keeps
    # the data loader from resampling every row of every epoch.
    records = resample_if_requested(records, source_root=clean, args=args)

    train, valid = split_records(
        records, valid_ratio=args.valid_ratio, seed=args.seed, split_by="speaker"
    )
    assert_disjoint(train, valid, by="spkid")

    out_dir = Path(args.out_dir).expanduser().resolve()
    train_path = out_dir / f"{args.id_prefix}_train.csv"
    valid_path = out_dir / f"{args.id_prefix}_valid.csv"
    write_metafile(train_path, train)
    write_metafile(valid_path, valid)
    print(f"train_metafile: {train_path} ({len(train)} rows, {len({r.spkid for r in train})} speakers)")
    print(f"valid_metafile: {valid_path} ({len(valid)} rows, {len({r.spkid for r in valid})} speakers)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.dataset.corpus.vctk_demand",
        description=__doc__.splitlines()[0],
    )
    sub = parser.add_subparsers(dest="command", required=True)

    testset = sub.add_parser("testset", help="the 824 paired utterances -> a scored set")
    testset.add_argument("corpus_root")
    testset.add_argument("--out-dir", required=True)
    testset.add_argument("--sample-rate", type=int, default=16000)
    testset.add_argument("--noisy-dir", default=None)
    testset.add_argument("--clean-dir", default=None)
    testset.add_argument("--transcript-dir", default=None)
    testset.add_argument("--no-transcripts", action="store_true")
    testset.add_argument("--limit", type=int, default=None)
    testset.set_defaults(func=run_testset)

    noise = sub.add_parser(
        "noise", help="noisy - clean -> one held-out DEMAND noise track per type"
    )
    noise.add_argument("corpus_root")
    noise.add_argument("--out-dir", required=True)
    noise.add_argument("--sample-rate", type=int, default=16000)
    noise.add_argument(
        "--splits",
        default="testset,trainset_28spk",
        help="Comma-separated; the test split has 5 types, the 28-speaker training "
        "split the other 10. Neither is heard by a model trained on DNS-5 noise.",
    )
    noise.add_argument("--limit-per-type", type=int, default=None)
    noise.set_defaults(func=run_noise)

    speech = sub.add_parser("speech", help="clean training speech -> train/valid metafiles")
    speech.add_argument("corpus_root")
    speech.add_argument("--out-dir", required=True)
    speech.add_argument("--clean-dir", default=None)
    speech.add_argument("--id-prefix", default="vctk")
    speech.add_argument("--utt-id-style", choices=["digest", "stem"], default="stem")
    speech.add_argument("--audio-suffixes", default=".wav,.flac")
    speech.add_argument("--min-duration", type=float, default=0.0)
    speech.add_argument("--max-files", type=int, default=None)
    speech.add_argument("--valid-ratio", type=float, default=0.05)
    speech.add_argument("--seed", type=int, default=1337)
    add_resample_arguments(speech)
    speech.set_defaults(func=run_speech)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
