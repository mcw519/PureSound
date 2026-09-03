"""Stitch DNS-5 read-speech segments back into whole-chapter utterances.

WHY. The deployed model's real-recording suppression is a long-context effect:
processing the field benchmark's 90D session in 10 s windows costs v8 9 dB against
processing it whole (benchmarks/probes -- window sweep, 2026-08-26). Training never
sees that context. Every row in `data/dns5-read.*.list` is a 10.3333 s DNS-5 segment,
and `align_audio_list` ZERO-PADS anything shorter than the target length, so raising
`training_length_seconds` past 10.33 buys silence, not context.

The corpus can pay for itself: DNS-5 ships each chapter pre-cut into exactly three
consecutive segments (`..._seg_{0,1,2}`), and they are contiguous -- the sample step
across a join measures 1.32x the local median step, against 6.29x for a random splice.
Concatenating them restores ~31 s of continuous single-speaker audio, no new corpus and
no crossfade needed.

    uv run python scripts/build_chapter_corpus.py \
        --metafile data/dns5-read.train.list \
        --out-dir /work/any_exp_link/puresound_exp/dns5_read_16k_chapters \
        --out-metafile data/dns5-read.chapters.train.list

Idempotent: an existing output of the right length is reused. Chapters that are not a
complete {0,1,2} set are dropped and counted, never silently padded.
"""
import argparse
import collections
import pathlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import soundfile as sf

HEADER = "uttid, spkid, gender, path, length, sample rate, channels"


def read_metafile(path):
    rows = [l.rstrip("\n").split(", ") for l in open(path)][1:]
    chapters = collections.defaultdict(dict)
    meta = {}
    for uttid, spkid, gender, wav, length, sr, ch in rows:
        base, _, idx = uttid.rpartition("_seg_")
        if not base:
            continue
        chapters[base][int(idx)] = wav
        meta[base] = (spkid, gender, int(sr), int(ch))
    return chapters, meta


def stitch(job):
    base, parts, out_path = job
    out = pathlib.Path(out_path)
    if out.is_file():
        try:
            return base, sf.info(str(out)).frames
        except Exception:
            out.unlink()
    chunks, subtype = [], None
    for _, wav in sorted(parts.items()):
        data, _ = sf.read(wav, dtype="float32", always_2d=False)
        subtype = subtype or sf.info(wav).subtype
        chunks.append(data if data.ndim == 1 else data[:, 0])
    joined = np.concatenate(chunks)
    tmp = out.with_suffix(".tmp.wav")           # never leave a half-written file behind
    sf.write(str(tmp), joined, 16000, subtype=subtype or "PCM_16")
    tmp.rename(out)
    return base, len(joined)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--metafile", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-metafile", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--expect-segments", type=int, default=3)
    args = ap.parse_args()

    chapters, meta = read_metafile(args.metafile)
    out_dir = pathlib.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    jobs, dropped = [], 0
    for base, parts in sorted(chapters.items()):
        if sorted(parts) != list(range(args.expect_segments)):
            dropped += 1
            continue
        jobs.append((base, parts, str(out_dir / f"{base}.wav")))
    print(f"{len(jobs)} chapters to stitch, {dropped} incomplete and dropped")

    lengths = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for i, (base, n) in enumerate(pool.map(stitch, jobs, chunksize=8), 1):
            lengths[base] = n
            if i % 2000 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    with open(args.out_metafile, "w") as f:
        f.write(HEADER + "\n")
        for base, _, out_path in jobs:
            spkid, gender, sr, ch = meta[base]
            f.write(f"{base}, {spkid}, {gender}, {out_path}, {lengths[base]}, {sr}, {ch}\n")
    secs = np.array(list(lengths.values())) / 16000.0
    print(f"wrote {args.out_metafile}: n={len(lengths)} "
          f"median {np.median(secs):.2f} s  min {secs.min():.2f}  max {secs.max():.2f}")


if __name__ == "__main__":
    main()
