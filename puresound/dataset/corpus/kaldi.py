"""Kaldi-style ``wav.scp`` / ``utt2spk`` -> the seven-column metafile.

For corpora that already ship a Kaldi-form listing, this is the whole preparation
step: the lists name the utterances and speakers, and the only thing missing from
the metafile is what the audio headers say.

Every recipe uses this one conversion, so the same corpus prepared through two
recipes produces the same metafile.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from puresound.audio.io import AudioIO
from puresound.utils import load_text_as_dict

from .records import METAFILE_HEADER


def convert_kaldi_metafile(
    output_path: str | Path,
    wav2scp_path: str | Path,
    utt2spk_path: str | Path,
    *,
    utt2gender_path: str | Path | None = None,
    separator: str = " ",
    insert_root_path: str | None = None,
    on_error: str = "skip",
    progress: bool = True,
) -> int:
    """Write the metafile and return how many rows it has.

    ``on_error="skip"`` drops an utterance whose audio cannot be read and says so;
    ``"raise"`` stops on the first one. Skipping is the default because a corpus
    of a few hundred thousand files usually has a couple of truncated ones, and
    losing the run to one of them is worse than losing the file.
    """
    if on_error not in {"skip", "raise"}:
        raise ValueError(f"on_error must be 'skip' or 'raise', got {on_error!r}")

    wav2scp = load_text_as_dict(file_path=str(wav2scp_path), separator=separator)
    utt2spk = load_text_as_dict(file_path=str(utt2spk_path), separator=separator)
    utt2gender = (
        load_text_as_dict(file_path=str(utt2gender_path), separator=separator)
        if utt2gender_path is not None
        else None
    )

    keys: Sequence[str] = sorted(wav2scp.keys())
    if progress:
        try:
            from tqdm import tqdm

            keys = tqdm(keys, desc="Reading audio headers")
        except ImportError:
            pass

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    missing_speaker = 0
    missing_gender = 0
    unreadable = 0

    with output_path.open("w", encoding="utf-8") as metafile:
        metafile.write(METAFILE_HEADER + "\n")
        for utt_key in keys:
            if utt_key not in utt2spk:
                missing_speaker += 1
                continue

            gender = None
            if utt2gender is not None:
                if utt_key not in utt2gender:
                    missing_gender += 1
                    continue
                gender = utt2gender[utt_key][0]

            audio_path = wav2scp[utt_key][0]
            if insert_root_path is not None:
                audio_path = os.path.join(insert_root_path, audio_path)

            try:
                sample_rate, total_samples, _, channels = AudioIO.audio_info(f_path=audio_path)
            except Exception:
                if on_error == "raise":
                    raise
                unreadable += 1
                continue

            metafile.write(
                f"{utt_key}, {utt2spk[utt_key][0]}, {gender}, {audio_path}, "
                f"{total_samples}, {sample_rate}, {channels}\n"
            )
            written += 1

    print(f"{output_path}: {written} row(s)")
    for count, reason in (
        (missing_speaker, f"not in {utt2spk_path}"),
        (missing_gender, f"not in {utt2gender_path}"),
        (unreadable, "audio could not be read"),
    ):
        if count:
            print(f"  dropped {count}: {reason}")
    return written


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.dataset.corpus.kaldi",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("output_path", type=str)
    parser.add_argument("wav2scp_path", type=str)
    parser.add_argument("utt2spk_path", type=str)
    parser.add_argument("--utt2gender_path", type=str, default=None)
    parser.add_argument(
        "--separator",
        type=str,
        default=" ",
        help="Column separator of the INPUT files; the output is always comma-separated.",
    )
    parser.add_argument("--insert_root_path", type=str, default=None)
    parser.add_argument("--on-error", choices=["skip", "raise"], default="skip")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    convert_kaldi_metafile(
        args.output_path,
        args.wav2scp_path,
        args.utt2spk_path,
        utt2gender_path=args.utt2gender_path,
        separator=args.separator,
        insert_root_path=args.insert_root_path,
        on_error=args.on_error,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
