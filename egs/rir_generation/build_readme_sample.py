#!/usr/bin/env python
"""Build the listenable M4 sample that ships with the README.

A bare RIR is a click; what a reader wants to hear is speech through it. This
convolves one dry LibriSpeech utterance with a near and a far channel of the
same M4 item, so the pair is a distance comparison inside one room rather than
two unrelated renders.

The two convolved files share one gain, because the near/far level difference
is part of what the sample shows and normalizing each separately would erase
it. The dry reference is scaled on its own: the bank is rendered in
``calibrated`` mode, so a RIR's absolute amplitude encodes a reference SPL
rather than a listening level, and forcing the dry file onto the same scale
leaves the convolved pair some 30 dB down and inaudible.

Note that most of the distance cue here is in the direct-to-reverberant ratio,
not the overall level: in a room this reverberant the diffuse field is nearly
distance-independent, so near and far end up within about a dB of each other in
RMS while sounding very different.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_SPEECH = REPO_ROOT / "test/test_case/61-70970-0040.flac"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "egs/rir_generation/assets"

#: Headroom below full scale for the loudest of the rendered files.
PEAK_TARGET = 0.89


def _channel_by_label(metadata: dict) -> dict[str, tuple[int, float]]:
    channels = metadata.get("realized_acoustics", {}).get("channels")
    if not channels:
        labels = metadata["scene"]["source_labels"]
        distances = metadata["scene"].get("source_distances_m", [])
        return {
            str(label): (index, float(distances[index]) if distances else float("nan"))
            for index, label in enumerate(labels)
        }
    return {
        str(entry["label"]): (int(entry["channel"]), float(entry["distance_m"]))
        for entry in channels
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rir", type=Path, required=True, help="M4 RIR WAV")
    parser.add_argument(
        "--json", type=Path, help="metadata JSON (default: RIR with .json)"
    )
    parser.add_argument("--speech", type=Path, default=DEFAULT_SPEECH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--near-label", default="near_0", help="channel rendered as the near example"
    )
    parser.add_argument(
        "--far-label", default="far_2", help="channel rendered as the far example"
    )
    args = parser.parse_args()

    metadata_path = args.json or args.rir.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    rir, rir_sr = sf.read(args.rir, always_2d=True)
    speech, speech_sr = sf.read(args.speech, always_2d=True)
    if rir_sr != speech_sr:
        parser.error(f"sample-rate mismatch: RIR {rir_sr} Hz, speech {speech_sr} Hz")
    dry = np.asarray(speech[:, 0], dtype=np.float64)

    by_label = _channel_by_label(metadata)
    missing = [
        label for label in (args.near_label, args.far_label) if label not in by_label
    ]
    if missing:
        parser.error(f"channels {missing} absent; have {sorted(by_label)}")

    wet: dict[str, np.ndarray] = {}
    for role, label in (("near", args.near_label), ("far", args.far_label)):
        index, distance = by_label[label]
        wet[role] = np.convolve(dry, np.asarray(rir[:, index], dtype=np.float64))
        print(f"{role:>5}: channel {index} ({label}) at {distance:.2f} m")

    # One gain across the convolved pair, so their level relationship survives;
    # the dry reference gets its own, for the reason in the module docstring.
    wet_peak = max(float(np.max(np.abs(signal))) for signal in wet.values())
    wet_gain = PEAK_TARGET / wet_peak if wet_peak > 0 else 1.0
    dry_peak = float(np.max(np.abs(dry)))
    dry_gain = PEAK_TARGET / dry_peak if dry_peak > 0 else 1.0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, signal, gain in (
        ("m4_sample_dry.wav", dry, dry_gain),
        (f"m4_sample_near_{args.near_label}.wav", wet["near"], wet_gain),
        (f"m4_sample_far_{args.far_label}.wav", wet["far"], wet_gain),
    ):
        path = args.output_dir / name
        sf.write(path, (signal * gain).astype(np.float32), rir_sr, subtype="PCM_16")
        written.append(path)

    rir_copy = args.output_dir / "m4_sample_rir.wav"
    sf.write(rir_copy, rir, rir_sr, subtype="FLOAT")
    written.append(rir_copy)

    print(f"\nconvolved-pair gain {wet_gain:.4f}, dry gain {dry_gain:.4f} "
          f"(peak target {PEAK_TARGET})")
    for path in written:
        print(f"  wrote {path.relative_to(REPO_ROOT)} "
              f"({path.stat().st_size / 2**10:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
