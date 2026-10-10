"""Mix a transcript-bearing speech corpus with a held-out noise pool into a WER set.

The WER stage needs three things at once: a corpus transcript for every
utterance (the recogniser on clean audio is not a reference), utterances long
enough for a deletion to be more than one word out of ten, and mixtures hard
enough that recognisers disagree. VCTK-DEMAND has the transcripts but short cuts
at mild SNR, where differences between models sit inside the recogniser's own
run-to-run noise. `build_eval_set` has the difficulty and no transcripts,
because the training chain draws speech from metafiles that carry none.

This tool takes the third route: a folder of speech with a text file beside each
utterance (LibriTTS ships ``<id>.normalized.txt``), a folder of noise tracks, an
SNR window, and a minimum duration. Nothing here is corpus-specific; the two
things that are -- where the noise comes from and where the transcripts sit --
are arguments.

Noise is the choice that decides what the set measures. A pool the model trained
on measures fit; a pool it never heard measures the thing a deployment meets.
``corpus.vctk_demand noise`` writes DEMAND's fifteen types out as tracks, and a
model trained on DNS-5 noise has heard none of them.

Mixing goes through ``puresound.audio.noise.add_bg_noise`` -- the same operator
the training chain uses -- so "SNR" means the same thing here as in a recipe.
There is no reverb: the point is a set where the SNR axis alone is the difficulty,
so a measured band of ``-10..-5`` is that band and not that band plus a room.

Run::

    python -m puresound.evaluation.tools.mix_paired_set \\
        --speech-dir /path/to/audio/LibriTTS/test-clean --transcript-suffix .normalized.txt \\
        --noise-dir egs/noise_suppression/data/demand_noise \\
        --snr -10 5 --min-duration 8 --n 500 \\
        --out-dir egs/noise_suppression/data_report/libritts_demand_hard
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch

from puresound.audio.io import AudioIO
from puresound.audio.noise import add_bg_noise
from puresound.dataset.corpus.paired import read_transcript, write_pair
from puresound.evaluation.records import chain_commit

AUDIO_SUFFIXES = (".wav", ".flac")


@dataclass(frozen=True)
class Utterance:
    audio: Path
    transcript: Path
    seconds: float


def scan_speech(
    speech_dir: Path,
    transcript_suffix: str,
    min_duration: float,
    max_duration: float | None,
) -> tuple[list[Utterance], int, int]:
    """Every utterance with a transcript beside it and a duration in range.

    Returns the candidates and two counts that go in the provenance: how many
    audio files had no transcript, and how many fell outside the duration window.
    A set is described by what it left out as much as by what it kept.
    """
    candidates: list[Utterance] = []
    without_text = outside_window = 0
    for audio in sorted(p for p in speech_dir.rglob("*") if p.suffix.lower() in AUDIO_SUFFIXES):
        text = audio.with_name(audio.stem + transcript_suffix)
        if not text.is_file():
            without_text += 1
            continue
        _, _, seconds, _ = AudioIO.audio_info(str(audio))
        if seconds < min_duration or (max_duration is not None and seconds > max_duration):
            outside_window += 1
            continue
        candidates.append(Utterance(audio, text, seconds))
    return candidates, without_text, outside_window


def scan_noise(noise_dir: Path) -> list[Path]:
    tracks = sorted(p for p in noise_dir.iterdir() if p.suffix.lower() in AUDIO_SUFFIXES)
    if not tracks:
        raise ValueError(f"{noise_dir} holds no {'/'.join(AUDIO_SUFFIXES)} noise tracks.")
    return tracks


def crop_noise(track: torch.Tensor, length: int, rng: random.Random) -> torch.Tensor:
    """A random window of ``length`` samples; a track shorter than that is tiled.

    Cropping here rather than inside ``add_bg_noise`` keeps the draw on this
    module's seeded generator, so the same seed writes the same set.
    """
    if track.shape[-1] >= length:
        start = rng.randrange(0, track.shape[-1] - length + 1)
        return track[..., start : start + length]
    repeats = length // track.shape[-1] + 1
    return track.repeat(1, repeats)[..., :length]


def mix_set(
    speech_dir: str | Path,
    noise_dir: str | Path,
    out_dir: str | Path,
    *,
    n: int,
    snr_range: tuple[float, float],
    min_duration: float,
    max_duration: float | None = None,
    transcript_suffix: str = ".txt",
    sample_rate: int = 16000,
    seed: int = 1234,
    source: str | None = None,
) -> int:
    """Write ``n`` mixtures with transcripts, and a provenance that can rebuild them."""
    speech_dir = Path(speech_dir).expanduser().resolve()
    noise_dir = Path(noise_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    low, high = snr_range
    if low > high:
        raise ValueError(f"--snr LOW HIGH needs LOW <= HIGH, got {low} {high}")

    candidates, without_text, outside_window = scan_speech(
        speech_dir, transcript_suffix, min_duration, max_duration
    )
    if not candidates:
        raise ValueError(
            f"{speech_dir}: no utterance has a `{transcript_suffix}` beside it and a "
            f"duration in the requested window ({without_text} without a transcript, "
            f"{outside_window} outside the window)."
        )
    tracks = scan_noise(noise_dir)

    rng = random.Random(seed)
    chosen = rng.sample(candidates, min(n, len(candidates)))
    noise_cache: dict[Path, torch.Tensor] = {}

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "provenance.json").unlink(missing_ok=True)
    source = source or out_dir.name
    with (out_dir / "manifest.jsonl").open("w", encoding="utf-8") as manifest:
        for index, utterance in enumerate(chosen):
            clean, _ = AudioIO.open(str(utterance.audio), resample_to=sample_rate)
            clean = clean.reshape(1, -1)
            track_path = rng.choice(tracks)
            if track_path not in noise_cache:
                track, _ = AudioIO.open(str(track_path), resample_to=sample_rate)
                noise_cache[track_path] = track.reshape(1, -1)
            noise = crop_noise(noise_cache[track_path], clean.shape[-1], rng)
            snr = rng.uniform(low, high)
            (noisy,), _ = add_bg_noise(clean, [noise], [snr])

            item_id = f"mx{index:05d}"
            entry = write_pair(out_dir, item_id, noisy, clean, sample_rate, source)
            entry["transcript"] = read_transcript(utterance.transcript)
            entry["speech"] = str(utterance.audio.relative_to(speech_dir))
            entry["noise"] = track_path.stem
            entry["snr_target_db"] = round(snr, 3)
            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")

    (out_dir / "provenance.json").write_text(
        json.dumps(
            {
                "kind": "mixed",
                "source": source,
                "chain_commit": chain_commit(),
                "speech_dir": str(speech_dir),
                "transcript_suffix": transcript_suffix,
                "noise_dir": str(noise_dir),
                "noise_tracks": [p.stem for p in tracks],
                "snr_range_db": [low, high],
                "min_duration_s": min_duration,
                "max_duration_s": max_duration,
                "sample_rate": sample_rate,
                "seed": seed,
                "items": len(chosen),
                "candidates": len(candidates),
                "skipped_without_transcript": without_text,
                "skipped_outside_duration": outside_window,
                "manifest_sha256": hashlib.sha256(
                    (out_dir / "manifest.jsonl").read_bytes()
                ).hexdigest(),
                "mixed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(chosen)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.evaluation.tools.mix_paired_set",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("--speech-dir", required=True, help="Searched recursively.")
    parser.add_argument(
        "--transcript-suffix",
        default=".txt",
        help="Text file beside each utterance, same stem (LibriTTS: .normalized.txt).",
    )
    parser.add_argument("--noise-dir", required=True, help="One track per file, not recursive.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--snr", type=float, nargs=2, default=(-10.0, 5.0), metavar=("LOW", "HIGH")
    )
    parser.add_argument("--min-duration", type=float, default=8.0, help="seconds")
    parser.add_argument("--max-duration", type=float, default=None, help="seconds")
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--name", default=None, help="Defaults to the output folder name.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        written = mix_set(
            args.speech_dir,
            args.noise_dir,
            args.out_dir,
            n=args.n,
            snr_range=(args.snr[0], args.snr[1]),
            min_duration=args.min_duration,
            max_duration=args.max_duration,
            transcript_suffix=args.transcript_suffix,
            sample_rate=args.sample_rate,
            seed=args.seed,
            source=args.name,
        )
    except (ValueError, FileNotFoundError) as error:
        print(error, file=sys.stderr)
        return 1
    out = Path(args.out_dir).resolve()
    print(f"wrote {written} mixture(s) with transcripts -> {out}")
    if written < args.n:
        print(f"only {written} utterances met the duration window; asked for {args.n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
