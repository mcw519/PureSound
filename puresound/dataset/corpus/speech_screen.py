"""Screen a noise corpus for intelligible speech before it is used as noise.

Noise suppression here keeps every voice in the input: speech is the thing to
preserve, whoever is talking. A noise clip with a person talking in it therefore
teaches the opposite -- "this voice is noise, remove it" -- and deletion is the
failure this task is gated on. Babble is different: many overlapping voices with
no words to recover is exactly the restaurant/station noise we are short of.

So the line is drawn at *intelligible*, in two passes:

1. Silero VAD marks where anything voice-like is (cheap, runs on everything).
2. faster-whisper transcribes only those regions. A region counts as speech when
   whisper both believes there is speech (``no_speech_prob`` low) and is
   confident in the words it heard (``avg_logprob`` high) and heard at least a
   few of them. Babble fails the confidence test; a clear talker passes it.

Output is one JSONL row per file with the numbers, so the threshold can be moved
after the fact; ``--apply`` then moves the files judged intelligible into a
parallel ``rejected`` tree -- moved, not deleted, so the call can be audited.

Run::

    python -m puresound.dataset.corpus.speech_screen /path/to/training_set/ns_noise/cochlscene \\
        --out /path/to/training_set/ns_noise/cochlscene.screen.jsonl --device cuda
    python -m puresound.dataset.corpus.speech_screen /path/to/training_set/ns_noise/cochlscene \\
        --out /path/to/training_set/ns_noise/cochlscene.screen.jsonl \\
        --apply --rejected-root /path/to/training_set/ns_noise_rejected/cochlscene
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Sequence

import numpy as np


SAMPLE_RATE = 16000
#: Whisper-side thresholds for "a person is saying words here".
MIN_WORDS = 3
MAX_NO_SPEECH_PROB = 0.5
MIN_AVG_LOGPROB = -0.8


def _vad_worker_init() -> None:
    import torch

    torch.set_num_threads(1)
    global _VAD
    from silero_vad import load_silero_vad

    _VAD = load_silero_vad()


def voiced_regions(path: str, min_region_s: float = 0.4) -> dict[str, Any]:
    """Silero regions for one file: ``{path, seconds, voiced_s, regions}``."""
    from silero_vad import get_speech_timestamps

    from puresound.audio.io import AudioIO

    wav, _ = AudioIO.open(path, resample_to=SAMPLE_RATE)
    wav = wav.reshape(-1)
    stamps = get_speech_timestamps(wav, _VAD, sampling_rate=SAMPLE_RATE, return_seconds=True)
    regions = [(float(s["start"]), float(s["end"])) for s in stamps if s["end"] - s["start"] >= min_region_s]
    return {
        "path": path,
        "seconds": round(wav.shape[-1] / SAMPLE_RATE, 3),
        "voiced_s": round(sum(b - a for a, b in regions), 3),
        "regions": [[round(a, 2), round(b, 2)] for a, b in regions],
    }


def judge(segments: Sequence[Any]) -> dict[str, Any]:
    """Turn whisper segments into the intelligibility numbers and a verdict."""
    confident = [
        s for s in segments
        if s.no_speech_prob <= MAX_NO_SPEECH_PROB and s.avg_logprob >= MIN_AVG_LOGPROB
    ]
    words = sum(len(s.text.split()) for s in confident)
    text = " ".join(s.text.strip() for s in confident)
    # CJK scripts have no spaces; count characters as words there.
    if words < MIN_WORDS and text:
        words = max(words, sum(1 for ch in text if "぀" <= ch <= "鿿" or "가" <= ch <= "힯"))
    return {
        "words": int(words),
        "text": text[:200],
        "best_logprob": round(max((s.avg_logprob for s in segments), default=-99.0), 3),
        "min_no_speech": round(min((s.no_speech_prob for s in segments), default=1.0), 3),
        "intelligible": bool(words >= MIN_WORDS),
    }


def screen(
    files: Sequence[str],
    out: Path,
    *,
    device: str = "cuda",
    whisper_model: str = "large-v3",
    jobs: int = 8,
) -> int:
    """Screen ``files`` into ``out`` (JSONL). Resumable by path."""
    done: set[str] = set()
    if out.is_file():
        done = {json.loads(line)["path"] for line in out.read_text().splitlines() if line.strip()}
    todo = [f for f in files if f not in done]
    print(f"{len(files)} file(s), {len(done)} already screened, {len(todo)} to go")
    if not todo:
        return 0

    from faster_whisper import WhisperModel

    from puresound.audio.io import AudioIO

    asr = WhisperModel(whisper_model, device=device, compute_type="float16" if device == "cuda" else "int8")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "a", encoding="utf-8") as handle, ProcessPoolExecutor(jobs, initializer=_vad_worker_init) as pool:
        for count, row in enumerate(pool.map(voiced_regions, todo, chunksize=16), start=1):
            if row["regions"]:
                wav, _ = AudioIO.open(row["path"], resample_to=SAMPLE_RATE)
                wav = wav.reshape(-1).numpy()
                pieces = [wav[int(a * SAMPLE_RATE): int(b * SAMPLE_RATE)] for a, b in row["regions"]]
                gap = np.zeros(int(0.3 * SAMPLE_RATE), dtype=np.float32)
                joined = np.concatenate([p for piece in pieces for p in (piece, gap)]).astype(np.float32)
                segments, _ = asr.transcribe(
                    joined, beam_size=1, temperature=0.0, condition_on_previous_text=False,
                    vad_filter=False, without_timestamps=True,
                )
                row.update(judge(list(segments)))
            else:
                row.update({"words": 0, "text": "", "best_logprob": -99.0, "min_no_speech": 1.0, "intelligible": False})
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if count % 1000 == 0:
                handle.flush()
                print(f"  {count}/{len(todo)}", flush=True)
    return 0


def apply(out: Path, root: Path, rejected_root: Path) -> int:
    """Move every file judged intelligible from ``root`` into ``rejected_root``."""
    rows = [json.loads(line) for line in out.read_text().splitlines() if line.strip()]
    moved = 0
    for row in rows:
        if not row["intelligible"]:
            continue
        source = Path(row["path"])
        if not source.is_file():
            continue
        dest = rejected_root / source.relative_to(root)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(source), str(dest))
        moved += 1
    total = len(rows)
    print(f"moved {moved} of {total} file(s) ({100.0 * moved / max(total, 1):.1f}%) -> {rejected_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m puresound.dataset.corpus.speech_screen",
        description=__doc__.splitlines()[0],
    )
    parser.add_argument("root", help="Folder of 16 kHz noise audio to screen (recursive).")
    parser.add_argument("--out", required=True, help="JSONL of per-file results.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--whisper-model", default="large-v3")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--apply", action="store_true", help="Move intelligible files out, using --out.")
    parser.add_argument("--rejected-root", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    if args.apply:
        if not args.rejected_root:
            print("--apply needs --rejected-root.", file=sys.stderr)
            return 2
        return apply(out, root, Path(args.rejected_root).expanduser().resolve())
    files = sorted(str(p) for p in root.rglob("*") if p.suffix.lower() in {".wav", ".flac"})
    if args.limit:
        files = files[: args.limit]
    return screen(files, out, device=args.device, whisper_model=args.whisper_model, jobs=args.jobs)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
