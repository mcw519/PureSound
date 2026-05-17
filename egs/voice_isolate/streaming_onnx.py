import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from puresound.streaming import StreamingDparnOrt, export_streaming_dparn_onnx  # noqa: E402


def export(args) -> None:
    manifest = export_streaming_dparn_onnx(
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

    runtime = StreamingDparnOrt(
        onnx_path=args.onnx_path,
        manifest_path=args.manifest_path,
        provider=args.provider,
    )
    wav, sr = AudioIO.open(
        str(args.input_audio), target_lvl=None, resample_to=runtime.sample_rate
    )
    samples = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    enhanced = runtime.process_samples(samples)
    enhanced = np.concatenate([enhanced, runtime.flush()])
    output_path = Path(args.output_audio)
    create_folder(str(output_path.parent))
    AudioIO.save(
        torch.from_numpy(enhanced).view(1, -1), str(output_path), runtime.sample_rate
    )
    print(f"saved: {output_path}")
    print(f"provider: {runtime.providers}")
    print(
        f"input_sr: {sr}, output_sr: {runtime.sample_rate}, samples: {enhanced.shape[0]}"
    )


def benchmark(args) -> None:
    runtime = StreamingDparnOrt(
        onnx_path=args.onnx_path,
        manifest_path=args.manifest_path,
        provider=args.provider,
    )
    rng = np.random.default_rng(args.seed)
    samples = (
        rng.standard_normal(runtime.sample_rate * args.seconds).astype(np.float32)
        * 0.01
    )
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="DPARN streaming ONNX tools")
    sub = parser.add_subparsers(dest="command", required=True)

    export_parser = sub.add_parser(
        "export", help="export config/checkpoint to streaming ONNX"
    )
    export_parser.add_argument("config_path", type=Path)
    export_parser.add_argument("checkpoint_path", type=Path)
    export_parser.add_argument("onnx_path", type=Path)
    export_parser.add_argument("--manifest_path", type=Path, default=None)
    export_parser.add_argument("--opset", type=int, default=17)
    export_parser.set_defaults(func=export)

    infer_parser = sub.add_parser(
        "infer", help="run streaming ONNX inference on an audio file"
    )
    infer_parser.add_argument("onnx_path", type=Path)
    infer_parser.add_argument("input_audio", type=Path)
    infer_parser.add_argument("output_audio", type=Path)
    infer_parser.add_argument("--manifest_path", type=Path, default=None)
    infer_parser.add_argument(
        "--provider", choices=["auto", "cpu", "cuda"], default="auto"
    )
    infer_parser.set_defaults(func=infer)

    bench_parser = sub.add_parser("benchmark", help="benchmark streaming ONNX runtime")
    bench_parser.add_argument("onnx_path", type=Path)
    bench_parser.add_argument("--manifest_path", type=Path, default=None)
    bench_parser.add_argument(
        "--provider", choices=["auto", "cpu", "cuda"], default="auto"
    )
    bench_parser.add_argument("--seconds", type=int, default=5)
    bench_parser.add_argument("--seed", type=int, default=0)
    bench_parser.set_defaults(func=benchmark)
    return parser


if __name__ == "__main__":
    cli = build_parser()
    parsed = cli.parse_args()
    parsed.func(parsed)
