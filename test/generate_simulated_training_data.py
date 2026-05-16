import argparse
import math
import subprocess
import sys
from pathlib import Path

import torch

from puresound.audio.io import AudioIO


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT / "test" / "test_case" / "outputs" / "manual_voice_isolate_data"
)


def _speech_like_tone(
    sample_rate: int,
    duration: float,
    base_freq: float,
    phase: float = 0.0,
) -> torch.Tensor:
    time = torch.arange(int(sample_rate * duration), dtype=torch.float32) / sample_rate
    harmonics = (
        0.55 * torch.sin(2 * math.pi * base_freq * time + phase)
        + 0.30 * torch.sin(2 * math.pi * base_freq * 2.01 * time)
        + 0.15 * torch.sin(2 * math.pi * base_freq * 3.03 * time)
    )
    envelope = 0.55 + 0.45 * torch.sin(2 * math.pi * 3.0 * time).clamp_min(0)
    gate = (torch.sin(2 * math.pi * 1.7 * time + phase) > -0.45).float()
    return (0.2 * harmonics * envelope * gate).view(1, -1)


def _write_inputs(output_dir: Path, sample_rate: int, duration: float) -> tuple[Path, list[Path]]:
    input_dir = output_dir / "_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)

    target_path = input_dir / "synthetic_target.wav"
    interferer_paths = [
        input_dir / "synthetic_interferer_00.wav",
        input_dir / "synthetic_interferer_01.wav",
    ]
    AudioIO.save(_speech_like_tone(sample_rate, duration, 170.0), str(target_path), sample_rate)
    AudioIO.save(
        _speech_like_tone(sample_rate, duration, 260.0, phase=0.4),
        str(interferer_paths[0]),
        sample_rate,
    )
    AudioIO.save(
        _speech_like_tone(sample_rate, duration, 330.0, phase=1.1),
        str(interferer_paths[1]),
        sample_rate,
    )
    return target_path, interferer_paths


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate easy-to-audition voice_isolate training samples."
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--target", type=Path, default=None)
    parser.add_argument("--interferer", type=Path, nargs="*", default=None)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--num-interferers", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--target-rir-type", choices=["anechoic", "direct", "early", "full"], default="early")
    parser.add_argument("--sir-range", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--foreground-distance", type=float, nargs=2, default=[0.4, 1.0])
    parser.add_argument("--interferer-distance", type=float, nargs=2, default=[2.0, 4.0])
    parser.add_argument("--rt60", type=float, nargs=2, default=[0.35, 0.35])
    parser.add_argument("--rir-nsample", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    target = args.target
    interferers = args.interferer
    if target is None or not interferers:
        target, interferers = _write_inputs(args.output_dir, args.sample_rate, args.duration)

    command = [
        sys.executable,
        str(REPO_ROOT / "test" / "simulate_room_scene.py"),
        "--target",
        str(target),
        "--interferer",
        *[str(path) for path in interferers],
        "--output-dir",
        str(args.output_dir),
        "--num-samples",
        str(args.num_samples),
        "--num-interferers",
        str(args.num_interferers),
        "--sample-rate",
        str(args.sample_rate),
        "--duration",
        str(args.duration),
        "--seed",
        str(args.seed),
        "--target-rir-type",
        args.target_rir_type,
        "--sir-range",
        str(args.sir_range[0]),
        str(args.sir_range[1]),
        "--foreground-distance",
        str(args.foreground_distance[0]),
        str(args.foreground_distance[1]),
        "--interferer-distance",
        str(args.interferer_distance[0]),
        str(args.interferer_distance[1]),
        "--rt60",
        str(args.rt60[0]),
        str(args.rt60[1]),
        "--rir-nsample",
        str(args.rir_nsample),
    ]
    print(" ".join(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)
        print(f"Open comparison_channels.wav files under: {args.output_dir}")


if __name__ == "__main__":
    main()
