"""DPCRN streaming ONNX tools for voice_isolate.

export    config + checkpoint -> per-frame streaming ONNX (+ JSON manifest)
infer     run the streaming ORT runtime on an audio file
benchmark measure real-time factor of the streaming runtime
verify    offline (full-utterance) vs ORT streaming parity, aligned + trimmed

Look-ahead note: a recipe with delay>0 (e.g. train_dpcrn_wide_antisup.yaml, delay=[1,1,1])
streams bit-exactly but the output carries a fixed algorithmic latency of
`streaming_delay_frames` frames (see manifest). Any offline<->streaming comparison
MUST align by that latency and trim the edges, else the delay reads as error --
the `verify` command does this automatically.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.streaming import StreamingDpcrnOrt, export_streaming_dpcrn_onnx  # noqa: E402


def export(args) -> None:
    manifest = export_streaming_dpcrn_onnx(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        onnx_path=args.onnx_path,
        manifest_path=args.manifest_path,
        opset_version=args.opset,
    )
    print(json.dumps(manifest, indent=2))


def infer(args) -> None:
    import torch

    from puresound.audio.io import AudioIO
    from puresound.utils import create_folder

    runtime = StreamingDpcrnOrt(
        onnx_path=args.onnx_path,
        manifest_path=args.manifest_path,
        provider=args.provider,
    )
    wav, sr = AudioIO.open(str(args.input_audio), target_lvl=None, resample_to=runtime.sample_rate)
    samples = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    enhanced = runtime.process_samples(samples)
    enhanced = np.concatenate([enhanced, runtime.flush()])
    output_path = Path(args.output_audio)
    create_folder(str(output_path.parent))
    AudioIO.save(torch.from_numpy(enhanced).view(1, -1), str(output_path), runtime.sample_rate)
    print(f"saved: {output_path}")
    print(f"provider: {runtime.providers}")
    print(f"input_sr: {sr}, output_sr: {runtime.sample_rate}, samples: {enhanced.shape[0]}")
    print(f"algorithmic latency: {runtime.manifest.get('streaming_delay_frames', 0)} frames "
          f"(~{runtime.manifest.get('streaming_delay_frames', 0) * runtime.hop_length / 16:.0f} ms)")


def benchmark(args) -> None:
    runtime = StreamingDpcrnOrt(onnx_path=args.onnx_path, manifest_path=args.manifest_path, provider=args.provider)
    rng = np.random.default_rng(args.seed)
    samples = rng.standard_normal(runtime.sample_rate * args.seconds).astype(np.float32) * 0.01
    start = time.perf_counter()
    enhanced = runtime.process_samples(samples)
    enhanced = np.concatenate([enhanced, runtime.flush()])
    elapsed = time.perf_counter() - start
    audio_seconds = samples.shape[0] / runtime.sample_rate
    print(f"provider: {runtime.providers}")
    print(f"audio_seconds: {audio_seconds:.3f}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"rtf: {elapsed / audio_seconds:.4f}")
    print(f"output_samples: {enhanced.shape[0]}")


def _sisdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(len(ref), len(est))
    ref = ref[:n] - ref[:n].mean()
    est = est[:n] - est[:n].mean()
    a = np.dot(est, ref) / (np.dot(ref, ref) + 1e-9)
    s = a * ref
    noise = est - s
    return float(10 * np.log10((np.dot(s, s) + 1e-9) / (np.dot(noise, noise) + 1e-9)))


def _align_sisdr(offline: np.ndarray, streaming: np.ndarray, max_lag: int, trim: int):
    """Best SI-SDR over integer lags (streaming is delayed by the look-ahead
    latency), edges trimmed so warmup / flush transients do not skew the metric."""
    best = None
    for lag in range(-max_lag, max_lag):
        if lag >= 0:
            x, y = offline[lag:], streaming
        else:
            x, y = offline, streaming[-lag:]
        n = min(len(x), len(y))
        if n < 2 * trim + 3000:
            continue
        sd = _sisdr(x[trim : n - trim], y[trim : n - trim])
        if best is None or sd > best[1]:
            best = (lag, sd)
    return best


def verify(args) -> None:
    import torch

    from puresound.streaming import load_streaming_dpcrn_model

    frame_model = load_streaming_dpcrn_model(args.config_path, args.checkpoint_path).eval()
    system_model = frame_model.system_model.eval()
    runtime = StreamingDpcrnOrt(onnx_path=args.onnx_path, manifest_path=args.manifest_path, provider=args.provider)

    if args.input_audio is not None:
        from puresound.audio.io import AudioIO

        wav, _ = AudioIO.open(str(args.input_audio), target_lvl=None, resample_to=runtime.sample_rate)
        wav = wav[:, : args.seconds * runtime.sample_rate]
    else:
        rng = np.random.default_rng(args.seed)
        wav = torch.from_numpy((rng.standard_normal(args.seconds * runtime.sample_rate) * 0.2).astype(np.float32)).view(1, -1)

    with torch.no_grad():
        offline = system_model(wav).squeeze().cpu().numpy()

    L = wav.shape[1]
    chunks = [runtime.process_samples(wav[0, i : i + 1024].numpy()) for i in range(0, L, 1024)]
    chunks.append(runtime.flush())
    streaming = np.concatenate(chunks)

    delay_frames = int(runtime.manifest.get("streaming_delay_frames", 0))
    max_lag = max(1200, (delay_frames + 4) * runtime.hop_length)
    lag, sisdr = _align_sisdr(offline, streaming, max_lag=max_lag, trim=args.trim)
    print(f"algorithmic latency: {delay_frames} frames (~{delay_frames * runtime.hop_length / 16:.0f} ms)")
    print(f"measured alignment lag: {lag} samples ({lag / runtime.sample_rate * 1000:.1f} ms)")
    print(f"offline vs ORT streaming SI-SDR (aligned + trimmed): {sisdr:.1f} dB")
    print("PASS" if sisdr >= 40.0 else "LOW (check STFT/latency alignment)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPCRN streaming ONNX tools (voice_isolate)")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("export", help="export config/checkpoint to streaming ONNX")
    p.add_argument("config_path", type=Path)
    p.add_argument("checkpoint_path", type=Path)
    p.add_argument("onnx_path", type=Path)
    p.add_argument("--manifest_path", type=Path, default=None)
    p.add_argument("--opset", type=int, default=17)
    p.set_defaults(func=export)

    p = sub.add_parser("infer", help="run streaming ONNX inference on an audio file")
    p.add_argument("onnx_path", type=Path)
    p.add_argument("input_audio", type=Path)
    p.add_argument("output_audio", type=Path)
    p.add_argument("--manifest_path", type=Path, default=None)
    p.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    p.set_defaults(func=infer)

    p = sub.add_parser("benchmark", help="benchmark streaming ONNX runtime")
    p.add_argument("onnx_path", type=Path)
    p.add_argument("--manifest_path", type=Path, default=None)
    p.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--seconds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=benchmark)

    p = sub.add_parser("verify", help="offline vs ORT streaming parity (aligned + trimmed)")
    p.add_argument("config_path", type=Path)
    p.add_argument("checkpoint_path", type=Path)
    p.add_argument("onnx_path", type=Path)
    p.add_argument("--manifest_path", type=Path, default=None)
    p.add_argument("--input_audio", type=Path, default=None, help="if omitted, uses synthetic noise")
    p.add_argument("--provider", choices=["auto", "cpu", "cuda"], default="cpu")
    p.add_argument("--seconds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trim", type=int, default=1200, help="edge samples to drop before scoring")
    p.set_defaults(func=verify)
    return parser


if __name__ == "__main__":
    cli = build_parser()
    parsed = cli.parse_args()
    parsed.func(parsed)
