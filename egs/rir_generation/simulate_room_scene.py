import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from puresound.audio.augmentation import AudioEffectAugmentor
from puresound.audio.io import AudioIO
from puresound.audio.volume import calculate_rms, normalize_waveform
from puresound.utils import create_folder


REPO_ROOT = Path(__file__).resolve().parents[2]
TEST_CASE_DIR = REPO_ROOT / "test" / "test_case"
DEFAULT_TARGET = TEST_CASE_DIR / "1272-141231-0008.flac"
DEFAULT_INTERFERER = TEST_CASE_DIR / "noise" / "zzpQAtOmMhQ.wav"
DEFAULT_OUTPUT_DIR = TEST_CASE_DIR / "outputs" / "room_scene_training_data"


def _load_audio(path: Path, sample_rate: int, target_level: float) -> torch.Tensor:
    wav, _ = AudioIO.open(
        str(path),
        normalized=False,
        target_lvl=target_level,
        resample_to=sample_rate,
    )
    if wav.shape[0] != 1:
        wav = wav[:1]
    return wav


def _random_crop_or_pad(wav: torch.Tensor, length: int) -> torch.Tensor:
    if wav.shape[-1] >= length:
        start = random.randint(0, wav.shape[-1] - length)
        return wav[..., start : start + length]
    return torch.nn.functional.pad(wav, (0, length - wav.shape[-1]))


def _scale_to_sir(reference: torch.Tensor, noise: torch.Tensor, sir_db: float):
    noise = normalize_waveform(wav=noise, amp_type="rms")
    reference_rms = calculate_rms(reference)
    sir = 10 ** (torch.as_tensor(sir_db, dtype=reference.dtype) / 20)
    scaled_noise = noise * (reference_rms / sir).view(-1, 1)
    return scaled_noise


def _save_wav(wav: torch.Tensor, path: Path, sample_rate: int):
    AudioIO.save(wav=wav.detach().cpu(), f_path=str(path), sr=sample_rate)


def _write_json(path: Path, data: dict):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _build_augmentor(args) -> AudioEffectAugmentor:
    augmentor = AudioEffectAugmentor()
    augmentor.init_room_simulator(
        {
            "room_dim_range": [
                [args.room_x[0], args.room_x[1]],
                [args.room_y[0], args.room_y[1]],
                [args.room_z[0], args.room_z[1]],
            ],
            "rt60_range": [args.rt60[0], args.rt60[1]],
            "source_receiver_distance_range": [
                args.source_distance[0],
                args.source_distance[1],
            ],
            "foreground_distance_range": [
                args.foreground_distance[0],
                args.foreground_distance[1],
            ],
            "interferer_distance_range": [
                args.interferer_distance[0],
                args.interferer_distance[1],
            ],
            "receiver_margin": args.receiver_margin,
            "source_margin": args.source_margin,
            "nsample": args.rir_nsample,
            "order": args.rir_order,
            "hp_filter": args.rir_hp_filter,
        }
    )
    return augmentor


def _simulate_one_sample(
    sample_id: str,
    target_path: Path,
    interferer_paths: list[Path],
    output_dir: Path,
    augmentor: AudioEffectAugmentor,
    args,
) -> dict:
    sample_dir = output_dir / sample_id
    create_folder(str(sample_dir))
    sample_rate = args.sample_rate
    sample_length = int(sample_rate * args.duration)

    target_dry = _random_crop_or_pad(
        _load_audio(target_path, sample_rate, args.target_level), sample_length
    )
    selected_interferers = random.sample(
        interferer_paths, k=min(args.num_interferers, len(interferer_paths))
    )
    interferer_dry = [
        _random_crop_or_pad(
            _load_audio(path, sample_rate, args.target_level), sample_length
        )
        for path in selected_interferers
    ]

    scene = augmentor.sample_room_scene()
    target_full, (target_rir_id, target_info) = augmentor.apply_rir(
        wav=target_dry,
        rir_mode="full",
        sr=sample_rate,
        room_scene=scene,
        source_role="foreground",
    )
    if args.target_rir_type == "anechoic":
        clean_speech = target_dry
    else:
        clean_speech, _ = augmentor.apply_rir(
            wav=target_dry,
            rir_id=target_rir_id,
            rir_mode=args.target_rir_type,
            sr=sample_rate,
        )

    interferer_records = []
    scaled_interferers = []
    for idx, (path, dry_wav) in enumerate(zip(selected_interferers, interferer_dry)):
        reverb_wav, (_, rir_info) = augmentor.apply_rir(
            wav=dry_wav,
            rir_mode="full",
            sr=sample_rate,
            room_scene=scene,
            source_role="interferer",
        )
        sir_db = random.uniform(args.sir_range[0], args.sir_range[1])
        scaled_wav = _scale_to_sir(target_full, reverb_wav, sir_db)
        scaled_interferers.append(scaled_wav)
        interferer_records.append(
            {
                "index": idx,
                "input_path": str(path),
                "sir_db": sir_db,
                "rir": rir_info["metadata"],
                "files": {
                    "dry": f"interferer_{idx:02d}_dry.wav",
                    "reverb_full": f"interferer_{idx:02d}_far_full.wav",
                    "scaled": f"interferer_{idx:02d}_far_scaled.wav",
                },
            }
        )
        _save_wav(dry_wav, sample_dir / f"interferer_{idx:02d}_dry.wav", sample_rate)
        _save_wav(
            reverb_wav,
            sample_dir / f"interferer_{idx:02d}_far_full.wav",
            sample_rate,
        )
        _save_wav(
            scaled_wav,
            sample_dir / f"interferer_{idx:02d}_far_scaled.wav",
            sample_rate,
        )

    interference_speech = torch.stack(scaled_interferers, dim=0).sum(dim=0)
    noisy_speech = target_full + interference_speech
    max_sample = noisy_speech.abs().max().clamp_min(1.0)
    noisy_speech = noisy_speech / max_sample
    clean_speech = clean_speech / max_sample
    target_full = target_full / max_sample
    interference_speech = interference_speech / max_sample
    scaled_interferers = [wav / max_sample for wav in scaled_interferers]
    consistency_noise = noisy_speech - clean_speech

    comparison_channels = [
        ("target_dry", target_dry),
        ("target_near_full", target_full),
        (f"target_{args.target_rir_type}_clean", clean_speech),
        ("interference_speech", interference_speech),
        ("noisy_speech", noisy_speech),
        ("consistency_noise", consistency_noise),
    ]
    comparison = torch.cat([wav for _, wav in comparison_channels], dim=0)

    files = {
        "noisy_speech": "noisy_speech.wav",
        "clean_speech": "clean_speech.wav",
        "consistency_noise": "consistency_noise.wav",
        "target_dry": "target_dry.wav",
        "target_reverb_full": "target_near_full.wav",
        "interference_speech": "interference_speech.wav",
        "comparison_channels": "comparison_channels.wav",
    }
    _save_wav(noisy_speech, sample_dir / files["noisy_speech"], sample_rate)
    _save_wav(clean_speech, sample_dir / files["clean_speech"], sample_rate)
    _save_wav(
        consistency_noise, sample_dir / files["consistency_noise"], sample_rate
    )
    _save_wav(target_dry, sample_dir / files["target_dry"], sample_rate)
    _save_wav(target_full, sample_dir / files["target_reverb_full"], sample_rate)
    _save_wav(
        interference_speech, sample_dir / files["interference_speech"], sample_rate
    )
    _save_wav(comparison, sample_dir / files["comparison_channels"], sample_rate)

    metadata = {
        "sample_id": sample_id,
        "sample_rate": sample_rate,
        "duration": args.duration,
        "target_input_path": str(target_path),
        "target_rir_type": args.target_rir_type,
        "target_rir": target_info["metadata"],
        "interferers": interferer_records,
        "comparison_channels": [name for name, _ in comparison_channels],
        "files": {key: str(sample_dir / filename) for key, filename in files.items()},
    }
    _write_json(sample_dir / "metadata.json", metadata)
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate same-room target/interferer training data."
    )
    parser.add_argument("--target", type=Path, default=DEFAULT_TARGET)
    parser.add_argument(
        "--interferer",
        type=Path,
        nargs="+",
        default=[DEFAULT_INTERFERER],
        help="One or more candidate interference speaker/audio files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--num-samples", type=int, default=1)
    parser.add_argument("--num-interferers", type=int, default=1)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--duration", type=float, default=6.0)
    parser.add_argument("--target-level", type=float, default=-28.0)
    parser.add_argument("--target-rir-type", choices=["anechoic", "direct", "early", "full"], default="early")
    parser.add_argument("--sir-range", type=float, nargs=2, default=[0.0, 0.0])
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--room-x", type=float, nargs=2, default=[6.0, 6.0])
    parser.add_argument("--room-y", type=float, nargs=2, default=[5.0, 5.0])
    parser.add_argument("--room-z", type=float, nargs=2, default=[2.8, 2.8])
    parser.add_argument("--rt60", type=float, nargs=2, default=[0.35, 0.35])
    parser.add_argument("--source-distance", type=float, nargs=2, default=[0.3, 4.0])
    parser.add_argument("--foreground-distance", type=float, nargs=2, default=[0.4, 1.0])
    parser.add_argument("--interferer-distance", type=float, nargs=2, default=[2.0, 4.0])
    parser.add_argument("--receiver-margin", type=float, default=0.5)
    parser.add_argument("--source-margin", type=float, default=0.5)
    parser.add_argument("--rir-nsample", type=int, default=4096)
    parser.add_argument("--rir-order", type=int, default=-1)
    parser.add_argument("--rir-hp-filter", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    create_folder(str(args.output_dir))

    augmentor = _build_augmentor(args)
    manifest = []
    for idx in range(args.num_samples):
        sample_metadata = _simulate_one_sample(
            sample_id=f"sample_{idx:04d}",
            target_path=args.target,
            interferer_paths=args.interferer,
            output_dir=args.output_dir,
            augmentor=augmentor,
            args=args,
        )
        manifest.append(sample_metadata)

    manifest_path = args.output_dir / "manifest.jsonl"
    manifest_path.write_text(
        "\n".join(json.dumps(item, ensure_ascii=False) for item in manifest) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(manifest)} simulated training sample(s) to {args.output_dir}")
    print(f"Wrote manifest to {manifest_path}")


if __name__ == "__main__":
    main()
