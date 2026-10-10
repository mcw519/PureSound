"""Speech-shaped noise, synthesised from training speech.

Speech-shaped noise (SSN) is noise with the long-term spectrum of speech: it
overlaps speech everywhere a mask could separate it, which makes it one of the
hardest noise types for a mask-based model, and no recording corpus supplies it
-- VoiceBank-DEMAND's SSN is test material. So it is made here,
from the TRAINING speech only:

* the long-term average spectrum of ``--utts-per-clip`` random utterances (a
  different draw per clip, so clips differ in talker mix and tilt), applied to
  white Gaussian noise in the frequency domain;
* for half the clips, multiplied by the smoothed envelope of one more utterance
  ("speech-modulated" SSN), which adds the syllable-rate fluctuation a stationary
  mask handles too easily.

Run::

    python -m puresound.dataset.corpus.ssn egs/noise_suppression/data/dns5/dns5_train.csv \\
        --dest /path/to/training_set/ns_noise/ssn --clips 300 --seconds 30
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Sequence

import numpy as np

from .records import read_metafile

SAMPLE_RATE = 16000
N_FFT = 1024


def long_term_spectrum(waves: Sequence[np.ndarray]) -> np.ndarray:
    """Mean power spectrum (``N_FFT // 2 + 1`` bins) over voiced frames of ``waves``."""
    hop = N_FFT // 2
    window = np.hanning(N_FFT)
    total = np.zeros(N_FFT // 2 + 1)
    count = 0
    for wav in waves:
        if len(wav) < N_FFT:
            continue
        frames = np.lib.stride_tricks.sliding_window_view(wav, N_FFT)[::hop] * window
        power = np.abs(np.fft.rfft(frames, axis=-1)) ** 2
        energy = power.sum(axis=-1)
        voiced = energy > np.percentile(energy, 40)  # skip pauses: they are not speech's spectrum
        total += power[voiced].sum(axis=0)
        count += int(voiced.sum())
    if count == 0:
        raise ValueError("no voiced frames to take a spectrum from")
    return total / count


def shaped_noise(spectrum: np.ndarray, samples: int, rng: np.random.Generator) -> np.ndarray:
    """White noise filtered to ``spectrum`` (power), unit RMS."""
    white = rng.standard_normal(samples)
    freq = np.fft.rfft(white)
    gain = np.sqrt(np.interp(np.linspace(0, 1, len(freq)), np.linspace(0, 1, len(spectrum)), spectrum))
    noise = np.fft.irfft(freq * gain, n=samples)
    return noise / (np.sqrt(np.mean(noise**2)) + 1e-12)


def envelope(wav: np.ndarray, samples: int, smooth_s: float = 0.02) -> np.ndarray:
    """Smoothed magnitude envelope of ``wav``, tiled to ``samples``, mean 1."""
    width = max(1, int(smooth_s * SAMPLE_RATE))
    env = np.convolve(np.abs(wav), np.ones(width) / width, mode="same")
    env = np.tile(env, int(np.ceil(samples / len(env))))[:samples]
    return env / (env.mean() + 1e-12)


def _make_clip(job: tuple) -> dict:
    import soundfile as sf

    from puresound.audio.io import AudioIO

    index, paths, dest, seconds, modulated, seed = job
    rng = np.random.default_rng(seed)
    waves = []
    for path in paths:
        wav, _ = AudioIO.open(str(path), resample_to=SAMPLE_RATE)
        waves.append(wav.reshape(-1).numpy().astype(np.float64))
    samples = int(seconds * SAMPLE_RATE)
    noise = shaped_noise(long_term_spectrum(waves[:-1]), samples, rng)
    if modulated:
        noise = noise * envelope(waves[-1], samples)
        noise = noise / (np.sqrt(np.mean(noise**2)) + 1e-12)
    out = Path(dest) / f"ssn_{'mod' if modulated else 'stat'}_{index:05d}.wav"
    sf.write(str(out), (0.1 * noise / max(1.0, 0.1 * np.abs(noise).max() / 0.99)).astype(np.float32),
             SAMPLE_RATE, subtype="PCM_16")
    return {"file": out.name, "modulated": modulated, "seed": seed, "sources": [str(p) for p in paths]}


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m puresound.dataset.corpus.ssn", description=__doc__.splitlines()[0])
    parser.add_argument("metafile", help="TRAINING speech metafile; never a test set.")
    parser.add_argument("--dest", required=True)
    parser.add_argument("--clips", type=int, default=300)
    parser.add_argument("--seconds", type=float, default=30.0)
    parser.add_argument("--utts-per-clip", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260926)
    parser.add_argument("--jobs", type=int, default=8)
    args = parser.parse_args(argv)

    records = read_metafile(args.metafile)
    rng = random.Random(args.seed)
    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    jobs = [
        (i, [r.path for r in rng.sample(records, args.utts_per_clip + 1)], dest, args.seconds, i % 2 == 1, args.seed + i)
        for i in range(args.clips)
    ]
    with ProcessPoolExecutor(args.jobs) as pool, open(dest.parent / f"{dest.name}.manifest.jsonl", "w") as handle:
        for row in pool.map(_make_clip, jobs):
            handle.write(json.dumps(row) + "\n")
    print(f"wrote {args.clips} clip(s), {args.clips * args.seconds / 3600:.1f} h -> {dest}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
