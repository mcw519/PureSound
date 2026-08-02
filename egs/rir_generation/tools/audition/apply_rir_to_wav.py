import argparse
import json
from pathlib import Path

import torch
import torchaudio
from torchaudio.functional import resample

from puresound.utils import fftconvolve


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Convolve a dry WAV with a generated RIR WAV."
    )
    parser.add_argument("--wav", type=Path, required=True, help="Input dry WAV.")
    parser.add_argument("--rir", type=Path, required=True, help="Input RIR WAV.")
    parser.add_argument("--output", type=Path, required=True, help="Output wet WAV.")
    parser.add_argument(
        "--wav-channel",
        choices=["first", "mixdown"],
        default="mixdown",
        help="How to convert multi-channel dry WAV to mono before convolution.",
    )
    parser.add_argument(
        "--rir-mode",
        choices=["full", "early", "direct"],
        default="full",
        help="Use full RIR, direct path only, or early reflections.",
    )
    parser.add_argument(
        "--length-mode",
        choices=["same", "full"],
        default="same",
        help="'same' returns dry WAV length aligned to each RIR direct peak.",
    )
    parser.add_argument(
        "--output-layout",
        choices=["rir-channels", "mono-sum"],
        default="rir-channels",
        help="Keep one output channel per RIR channel or sum them to mono.",
    )
    parser.add_argument(
        "--dry-wet",
        type=float,
        default=1.0,
        help="Wet amount. 1.0 means fully convolved; 0.0 means dry only.",
    )
    parser.add_argument(
        "--peak-normalize",
        action="store_true",
        help="Peak-normalize output to --target-peak.",
    )
    parser.add_argument("--target-peak", type=float, default=0.98)
    parser.add_argument(
        "--output-low-gain-db",
        type=float,
        default=0.0,
        help="Gain applied to the wet output below --output-low-cutoff-hz.",
    )
    parser.add_argument("--output-low-cutoff-hz", type=float, default=1000.0)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=None,
        help="Optional JSON path to write convolution metadata.",
    )
    return parser.parse_args()


def _to_mono(wav: torch.Tensor, mode: str) -> torch.Tensor:
    if wav.ndim != 2:
        raise ValueError(f"Expected WAV tensor [channels, samples], got {wav.shape}")
    if wav.shape[0] == 1:
        return wav
    if mode == "first":
        return wav[:1]
    if mode == "mixdown":
        return wav.mean(dim=0, keepdim=True)
    raise ValueError(f"Unknown wav-channel mode: {mode}")


def _trim_rir(rir: torch.Tensor, sample_rate: int, rir_mode: str) -> torch.Tensor:
    if rir_mode == "full":
        return rir
    window_ms = 6.0 if rir_mode == "direct" else 50.0
    window = max(1, int(round(window_ms * 1e-3 * sample_rate)))
    peaks = rir.abs().argmax(dim=-1)
    ends = torch.clamp(peaks + window, max=rir.shape[-1])
    out = torch.zeros_like(rir)
    for channel, end in enumerate(ends.tolist()):
        out[channel, :end] = rir[channel, :end]
    return out[:, : int(ends.max().item())]


def apply_rir_to_wav(
    wav: torch.Tensor,
    rir: torch.Tensor,
    sample_rate: int,
    rir_mode: str = "full",
    length_mode: str = "same",
    output_layout: str = "rir-channels",
    dry_wet: float = 1.0,
    peak_normalize: bool = False,
    target_peak: float = 0.98,
    output_low_gain_db: float = 0.0,
    output_low_cutoff_hz: float = 1000.0,
) -> torch.Tensor:
    if wav.ndim != 2 or wav.shape[0] != 1:
        raise ValueError(f"Expected mono WAV tensor [1, samples], got {wav.shape}")
    if rir.ndim != 2:
        raise ValueError(f"Expected RIR tensor [channels, samples], got {rir.shape}")
    rir = _trim_rir(rir.float(), sample_rate, rir_mode)
    wav = wav.float()

    wet_channels = []
    for channel in range(rir.shape[0]):
        full = fftconvolve(wav, rir[channel : channel + 1], mode="full")
        if length_mode == "same":
            direct = int(rir[channel].abs().argmax().item())
            wet = full[..., direct : direct + wav.shape[-1]]
        elif length_mode == "full":
            wet = full
        else:
            raise ValueError(f"Unknown length mode: {length_mode}")
        wet_channels.append(wet)
    wet = torch.cat(wet_channels, dim=0)

    if dry_wet < 1.0:
        dry = wav
        if length_mode == "full":
            dry = torch.nn.functional.pad(dry, (0, wet.shape[-1] - dry.shape[-1]))
        dry = dry.expand_as(wet)
        wet = float(dry_wet) * wet + (1.0 - float(dry_wet)) * dry

    if output_layout == "mono-sum":
        wet = wet.sum(dim=0, keepdim=True)
    elif output_layout != "rir-channels":
        raise ValueError(f"Unknown output layout: {output_layout}")

    if abs(float(output_low_gain_db)) > 1e-6:
        from torchaudio.functional import highpass_biquad, lowpass_biquad

        low = lowpass_biquad(wet, sample_rate, float(output_low_cutoff_hz))
        high = highpass_biquad(wet, sample_rate, float(output_low_cutoff_hz))
        low_gain = 10.0 ** (float(output_low_gain_db) / 20.0)
        wet = low * low_gain + high

    if peak_normalize:
        peak = wet.abs().max()
        if peak > 1e-12:
            wet = wet / peak * float(target_peak)
    return wet


def main():
    args = _parse_args()
    wav, wav_sr = torchaudio.load(args.wav)
    rir, rir_sr = torchaudio.load(args.rir)
    if wav_sr != rir_sr:
        wav = resample(wav, wav_sr, rir_sr)
    wav = _to_mono(wav, args.wav_channel)
    wet = apply_rir_to_wav(
        wav=wav,
        rir=rir,
        sample_rate=rir_sr,
        rir_mode=args.rir_mode,
        length_mode=args.length_mode,
        output_layout=args.output_layout,
        dry_wet=args.dry_wet,
        peak_normalize=args.peak_normalize,
        target_peak=args.target_peak,
        output_low_gain_db=args.output_low_gain_db,
        output_low_cutoff_hz=args.output_low_cutoff_hz,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(str(args.output), wet.cpu(), rir_sr, encoding="PCM_F")
    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "input_wav": str(args.wav),
            "input_rir": str(args.rir),
            "output": str(args.output),
            "sample_rate": int(rir_sr),
            "rir_channels": int(rir.shape[0]),
            "output_channels": int(wet.shape[0]),
            "output_samples": int(wet.shape[-1]),
            "wav_channel": args.wav_channel,
            "rir_mode": args.rir_mode,
            "length_mode": args.length_mode,
            "output_layout": args.output_layout,
            "dry_wet": float(args.dry_wet),
            "peak_normalize": bool(args.peak_normalize),
            "target_peak": float(args.target_peak),
            "output_low_gain_db": float(args.output_low_gain_db),
            "output_low_cutoff_hz": float(args.output_low_cutoff_hz),
        }
        args.metadata.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
