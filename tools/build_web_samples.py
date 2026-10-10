"""Build the web playground's sample clips from the repository's test audio.

The clips are what a first-time visitor can try without their own recording:
two talkers, a near talker over a far one in a reverberant room, speech in
noise, and a speaker-verification pair.  Speech comes from the LibriSpeech
utterances already under ``test/test_case/`` (CC BY 4.0); the room and the
noise are synthesised here from a fixed seed, so no other recording is used
and the output is reproducible.

    uv run python tools/build_web_samples.py

writes ``puresound/web/static/samples/`` (WAVs, ``index.json``, ``README.md``).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "test" / "test_case"
TARGET = ROOT / "puresound" / "web" / "static" / "samples"
RATE = 16_000
RNG = np.random.default_rng(20260925)


def load(name: str) -> np.ndarray:
    audio, rate = sf.read(SOURCE / name, dtype="float32", always_2d=True)
    if rate != RATE:
        raise SystemExit(f"{name}: expected {RATE} Hz, got {rate}")
    return audio.mean(axis=1)


def rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(signal)) + 1e-12))


def at_level(signal: np.ndarray, dbfs: float) -> np.ndarray:
    return signal * (10 ** (dbfs / 20) / rms(signal))


def room_response(rt60: float, drr_db: float, length_s: float = 0.9) -> np.ndarray:
    """A synthetic impulse response: a direct path, a few early reflections and
    an exponentially decaying diffuse tail, scaled to the requested DRR."""

    length = int(length_s * RATE)
    time = np.arange(length) / RATE
    tail = RNG.standard_normal(length) * np.exp(-6.9 * time / rt60)
    tail[: int(0.004 * RATE)] = 0.0
    for delay_ms, gain in ((7.0, 0.5), (11.0, -0.35), (17.0, 0.3), (23.0, -0.22)):
        tail[int(delay_ms * RATE / 1000)] += gain * np.max(np.abs(tail))
    direct = np.zeros(length)
    direct[0] = 1.0
    tail *= np.sqrt(10 ** (-drr_db / 10) / np.sum(np.square(tail)))
    return direct + tail


def pink_noise(length: int) -> np.ndarray:
    spectrum = np.fft.rfft(RNG.standard_normal(length))
    frequencies = np.fft.rfftfreq(length, 1 / RATE)
    spectrum[1:] /= np.sqrt(frequencies[1:])
    spectrum[0] = 0.0
    return np.fft.irfft(spectrum, n=length)


def place(length: int, *parts: tuple[np.ndarray, float]) -> np.ndarray:
    out = np.zeros(length)
    for signal, start_s in parts:
        start = int(start_s * RATE)
        end = min(length, start + signal.size)
        out[start:end] += signal[: end - start]
    return out


def finish(signal: np.ndarray, peak_dbfs: float = -3.0) -> np.ndarray:
    return (signal * (10 ** (peak_dbfs / 20) / np.max(np.abs(signal)))).astype(np.float32)


def main() -> None:
    TARGET.mkdir(parents=True, exist_ok=True)
    near = load("61-70970-0040.flac")
    far = load("1272-141231-0008.flac")
    mix = load("1272-128104-0000_2035-147961-0014.wav")

    # Near talker over a far one: the far talker reverberant and 9 dB down,
    # already talking when the near one starts, over a quiet room floor.
    far_room = np.convolve(far, room_response(rt60=0.7, drr_db=-4.0))[: far.size + RATE // 2]
    length = int(6.0 * RATE)
    near_far = place(length, (at_level(far_room, -35.0), 0.2), (at_level(near, -26.0), 1.6))
    near_far += at_level(pink_noise(length), -58.0)

    # Speech in noise: a fan-like band, mains hum and pink noise at 5 dB SNR.
    speech = place(int(5.0 * RATE), (at_level(near, -26.0), 0.4))
    length = speech.size
    time = np.arange(length) / RATE
    hum = sum(np.sin(2 * np.pi * 120 * harmonic * time) / harmonic for harmonic in (1, 2, 3, 5))
    fan = np.fft.irfft(np.fft.rfft(RNG.standard_normal(length)) * np.exp(-((np.fft.rfftfreq(length, 1 / RATE) - 900) / 500) ** 2), n=length)
    noise = at_level(pink_noise(length), -40.0) + at_level(hum, -42.0) + at_level(fan, -36.0)
    noisy = speech + at_level(noise, -26.0 - 5.0)

    split = far.size // 2
    clips = [
        {"id": "two-talkers", "file": "two-talkers.wav", "task": "enhancement", "title": "Two talkers",
         "description": "Two LibriSpeech readers mixed at similar levels.", "audio": mix},
        {"id": "near-and-far", "file": "near-and-far.wav", "task": "enhancement", "title": "Near talker, far talker",
         "description": "A dry near talker; a reverberant far talker (RT60 0.7 s) 9 dB down, starting first.", "audio": near_far},
        {"id": "noisy-speech", "file": "noisy-speech.wav", "task": "enhancement", "title": "Speech in noise",
         "description": "One reader over a fan band, mains hum and pink noise at 5 dB SNR.", "audio": noisy},
        {"id": "speaker-a-1", "file": "speaker-a-1.wav", "task": "speaker", "title": "Speaker A, first half",
         "description": "LibriSpeech reader 1272, first half of an utterance.", "audio": far[:split]},
        {"id": "speaker-a-2", "file": "speaker-a-2.wav", "task": "speaker", "title": "Speaker A, second half",
         "description": "The same reader, the other half of the utterance.", "audio": far[split:]},
        {"id": "speaker-b", "file": "speaker-b.wav", "task": "speaker", "title": "Speaker B",
         "description": "LibriSpeech reader 61.", "audio": near},
        {"id": "speaker-a-full", "file": "speaker-a-full.wav", "task": "world", "title": "Speaker A, full utterance",
         "description": "LibriSpeech reader 1272, the complete source utterance for acoustic scenes.", "audio": far},
    ]
    index = []
    for clip in clips:
        audio = finish(clip.pop("audio"))
        sf.write(TARGET / clip["file"], audio, RATE, subtype="PCM_16")
        index.append({**clip, "seconds": round(audio.size / RATE, 2)})
    pairs = [
        {"id": "same-speaker", "title": "Same speaker", "enrollment": "speaker-a-1", "test": "speaker-a-2"},
        {"id": "different-speakers", "title": "Different speakers", "enrollment": "speaker-a-1", "test": "speaker-b"},
    ]
    (TARGET / "index.json").write_text(json.dumps({"clips": index, "speaker_pairs": pairs}, indent=2) + "\n")
    (TARGET / "README.md").write_text(
        "# Web playground samples\n\n"
        "Built by `tools/build_web_samples.py`; do not edit by hand.\n\n"
        "Speech: LibriSpeech (Panayotov et al., 2015), CC BY 4.0, via the utterances in\n"
        "`test/test_case/`. Rooms and noise are synthesised by the build script from a\n"
        "fixed seed. No other recording is included.\n"
    )
    print(f"wrote {len(index)} clips to {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
