"""Build the pipeline inspector's sample audio.

The inspector builds rows from the user's own audio; these are what it offers
when there is none. Talkers are the LibriSpeech readers the playground already
ships (``samples/*.wav``, CC BY 4.0) and are referenced, not copied. Noise is
synthesised here from a fixed seed. Rooms are copied from one of our own
synthetic RIR banks (generated in-house by the hybrid renderer), chosen to
spread over reverberation time; the bank is read once, here, and never by the
inspector.

    uv run python tools/build_pipeline_samples.py --bank /path/to/bank/items

writes ``puresound/web/static/samples/pipeline/``. Without ``--bank`` the rooms
already there are kept and only the noise and the index are rebuilt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import fftconvolve

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "puresound" / "web" / "static" / "samples" / "pipeline"
SPEECH = ROOT / "test" / "test_case"
RATE = 16_000
SECONDS = 5.0
LEVEL_DBFS = -26.0
ROOM_RT60S = (0.25, 0.45, 0.65, 0.85)
RNG = np.random.default_rng(20260927)

TALKERS = [
    {"id": "reader-61", "title": "Reader 61", "description": "LibriSpeech reader 61, one utterance.",
     "files": ["../speaker-b.wav"]},
    {"id": "reader-1272", "title": "Reader 1272", "description": "LibriSpeech reader 1272, two halves of one utterance.",
     "files": ["../speaker-a-1.wav", "../speaker-a-2.wav"]},
]


def at_level(signal: np.ndarray, dbfs: float = LEVEL_DBFS) -> np.ndarray:
    signal = signal - np.mean(signal)
    signal = signal * (10 ** (dbfs / 20) / np.sqrt(np.mean(np.square(signal)) + 1e-12))
    peak = np.max(np.abs(signal))
    return (signal * (0.95 / peak if peak > 0.95 else 1.0)).astype(np.float32)


def shaped(length: int, gain) -> np.ndarray:
    """White noise with the magnitude response ``gain(frequencies)``."""
    spectrum = np.fft.rfft(RNG.standard_normal(length))
    frequencies = np.fft.rfftfreq(length, 1 / RATE)
    return np.fft.irfft(spectrum * gain(frequencies), n=length)


def pink(length: int) -> np.ndarray:
    return shaped(length, lambda f: 1 / np.sqrt(np.maximum(f, 20.0)))


def rumble(length: int) -> np.ndarray:
    """Traffic-like: brown-ish below 250 Hz with a slow swell."""
    low = shaped(length, lambda f: (1 / np.maximum(f, 20.0)) / (1 + (f / 250.0) ** 4))
    swell = 0.6 + 0.4 * np.sin(2 * np.pi * 0.23 * np.arange(length) / RATE + 1.0)
    return low * swell


def hum(length: int) -> np.ndarray:
    time = np.arange(length) / RATE
    tone = sum(np.sin(2 * np.pi * 60 * h * time + h) / h for h in range(1, 8))
    return tone + 0.05 * np.std(tone) * pink(length) / np.std(pink(length))


def fan(length: int) -> np.ndarray:
    band = shaped(length, lambda f: np.exp(-(((f - 900) / 450) ** 2)))
    bed = shaped(length, lambda f: 1 / (1 + (f / 300.0) ** 2))
    return band / np.std(band) + 0.5 * bed / np.std(bed)


def diffuse_room(rt60: float, length_s: float = 0.8) -> np.ndarray:
    time = np.arange(int(length_s * RATE)) / RATE
    tail = RNG.standard_normal(time.size) * np.exp(-6.9 * time / rt60)
    tail[0] = 3.0 * np.max(np.abs(tail))
    return tail


def babble() -> np.ndarray:
    """Six overlapping copies of the two readers, time-reversed so no word is
    intelligible, in a diffuse room: speech's spectrum and rhythm, no content."""
    readers = [sf.read(SPEECH / name, dtype="float32")[0] for name in ("61-70970-0040.flac", "1272-141231-0008.flac")]
    length = int(SECONDS * RATE)
    mix = np.zeros(length)
    for copy in range(6):
        voice = readers[copy % 2][::-1]
        voice = voice / (np.sqrt(np.mean(voice**2)) + 1e-12)
        start = int(RNG.uniform(0, voice.size))
        looped = np.roll(np.tile(voice, int(np.ceil(length / voice.size)) + 1), -start)[:length]
        mix += looped * RNG.uniform(0.6, 1.0)
    return fftconvolve(mix, diffuse_room(0.6))[:length]


def clatter(length: int) -> np.ndarray:
    """Dishes and keys: short band-limited bursts at random times."""
    out = np.zeros(length)
    for _ in range(28):
        size = int(RNG.uniform(0.02, 0.09) * RATE)
        centre = RNG.uniform(1500, 6000)
        burst = shaped(size, lambda f: np.exp(-(((f - centre) / (0.4 * centre)) ** 2)))
        burst *= np.exp(-np.arange(size) / (0.25 * size)) * RNG.uniform(0.3, 1.0)
        start = int(RNG.uniform(0, length - size))
        out[start : start + size] += burst / (np.max(np.abs(burst)) + 1e-12)
    return out + 0.01 * pink(length)


NOISES = [
    ("pink", "Pink noise", "Steady broadband noise, power falling 3 dB per octave.", pink),
    ("rumble", "Traffic rumble", "Low-frequency rumble below 250 Hz with a slow swell.", rumble),
    ("hum", "Mains hum", "60 Hz and its harmonics over a faint noise bed.", hum),
    ("fan", "Fan", "A band around 900 Hz over low broadband noise.", fan),
    ("babble", "Babble", "The two talker samples, six overlapping copies time-reversed so no word is intelligible, in a diffuse room. With a reader as the foreground, its own reversed voice is in this noise.", None),
    ("clatter", "Clatter", "Short bright bursts at random times: dishes, keys.", clatter),
]


def build_noises() -> list[dict]:
    folder = TARGET / "noise"
    folder.mkdir(parents=True, exist_ok=True)
    length = int(SECONDS * RATE)
    entries = []
    for key, title, description, make in NOISES:
        audio = at_level(babble() if make is None else make(length))
        sf.write(folder / f"{key}.wav", audio, RATE, subtype="PCM_16")
        entries.append({"id": key, "file": f"noise/{key}.wav", "title": title,
                        "description": description, "seconds": round(audio.size / RATE, 2)})
    return entries


def copy_rooms(bank: Path, scan: int) -> None:
    candidates = []
    for sidecar in sorted(bank.glob("*.json"))[:scan]:
        meta = json.loads(sidecar.read_text())
        scene = meta.get("scene") or {}
        channels = scene.get("channel_map") or []
        if scene.get("room_dim") and scene.get("mic_pos") and channels and all("source_pos" in c for c in channels):
            candidates.append((float(scene["rt60"]), sidecar, meta))
    if len(candidates) < len(ROOM_RT60S):
        raise SystemExit(f"{bank}: only {len(candidates)} rooms with geometry in the first {scan}")
    folder = TARGET / "rooms" / "items"
    folder.mkdir(parents=True, exist_ok=True)
    for stale in folder.glob("*"):
        stale.unlink()
    used = set()
    for target in ROOM_RT60S:
        rt60, sidecar, meta = min((c for c in candidates if c[1] not in used), key=lambda c: abs(c[0] - target))
        used.add(sidecar)
        audio, rate = sf.read(sidecar.with_suffix(".wav"), dtype="float32", always_2d=True)
        sf.write(folder / sidecar.with_suffix(".wav").name, audio, rate, subtype="PCM_24")
        meta.pop("gpu_device", None)
        (folder / sidecar.name).write_text(json.dumps(meta, indent=1) + "\n")


def room_entries() -> list[dict]:
    entries = []
    for sidecar in sorted((TARGET / "rooms" / "items").glob("*.json")):
        scene = json.loads(sidecar.read_text())["scene"]
        dims = [round(float(value), 2) for value in scene["room_dim"]]
        entries.append({
            "id": sidecar.stem,
            "file": f"rooms/items/{sidecar.stem}.wav",
            "title": f"{dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f} m, RT60 {float(scene['rt60']):.2f} s",
            "rt60": round(float(scene["rt60"]), 3),
            "room_dim": dims,
            "sources": len(scene["channel_map"]),
            "obstacles": len(scene.get("obstacles") or []),
        })
    return sorted(entries, key=lambda entry: entry["rt60"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bank", type=Path, help="a room bank's items folder (room_*.wav + room_*.json)")
    parser.add_argument("--scan", type=int, default=600, help="rooms to consider, in name order")
    args = parser.parse_args()
    if args.bank:
        copy_rooms(args.bank, args.scan)
    rooms = room_entries()
    if len(rooms) != len(ROOM_RT60S):
        raise SystemExit("no rooms yet: run once with --bank")
    index = {"talkers": TALKERS, "noises": build_noises(), "rooms": rooms}
    (TARGET / "index.json").write_text(json.dumps(index, indent=2, ensure_ascii=False) + "\n")
    (TARGET / "README.md").write_text(
        "# Pipeline inspector samples\n\n"
        "Built by `tools/build_pipeline_samples.py`; do not edit by hand.\n\n"
        "Talkers: the playground's LibriSpeech clips one folder up (Panayotov et al., 2015),\n"
        "CC BY 4.0. Noise: synthesised by the build script from a fixed seed; the babble is\n"
        "those two readers, time-reversed. Rooms: impulse responses and geometry from one of\n"
        "PureSound's own synthetic RIR banks. No other recording is included.\n"
    )
    print(f"wrote {len(index['noises'])} noises and {len(rooms)} rooms to {TARGET.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
