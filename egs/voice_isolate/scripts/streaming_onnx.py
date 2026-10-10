"""DPCRN streaming ONNX tools for voice_isolate.

export    config + checkpoint -> per-frame streaming ONNX (+ JSON manifest)
infer     run the streaming ORT runtime on an audio file
benchmark measure real-time factor of the streaming runtime
verify    offline (full-utterance) vs ORT streaming parity, aligned + trimmed

Look-ahead note: a recipe with delay>0 (e.g. delay=[1,1,1])
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
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from _gate_flags import add_onset_guard_arg, build_onset_guard  # noqa: E402
from puresound.inference import PROVIDER_CHOICES  # noqa: E402
from puresound.streaming import StreamingDpcrnOrt, export_streaming_dpcrn_onnx  # noqa: E402
from puresound.system.postprocess import Postprocessor  # noqa: E402


#: Per-request refusal of a guard the manifest records. Spelled once because
#: `infer` and `verify` have to refuse it the same way -- `verify`'s offline
#: reference has to drop it too, or the comparison is between two systems.
_GUARD_OFF = {"enabled": False}


def _guard_overrides(args):
    return _GUARD_OFF if getattr(args, "no_onset_guard", False) else None


def _cpu_options(args):
    return {
        "native_ssm": getattr(args, "native_ssm", "auto"),
        "native_library": getattr(args, "native_library", None),
        "intra_op_num_threads": getattr(args, "threads", None),
    }


def _add_cpu_options(parser):
    parser.add_argument("--native-ssm", choices=("auto", "off", "required"), default="auto")
    parser.add_argument("--native-library", type=Path, default=None,
                        help="locally built CPU SSM library (or PURESOUND_ORT_SSM_LIBRARY)")
    parser.add_argument("--threads", type=int, default=None, help="ORT intra-op threads")


def export(args) -> None:
    manifest = export_streaming_dpcrn_onnx(
        config_path=args.config_path,
        checkpoint_path=args.checkpoint_path,
        onnx_path=args.onnx_path,
        manifest_path=args.manifest_path,
        opset_version=args.opset,
        # Recorded in the manifest, applied by the runtime. The released default
        # is 0.9 -- see egs/voice_isolate/README.md -- so an export at 1.0 ships
        # something the scorecards never measured.
        postprocess=Postprocessor(dry_blend=args.dry_blend),
        # Same contract, different stage: recorded here, applied by the runtime,
        # absent unless --onset-guard is passed.
        onset_guard=build_onset_guard(args),
        optimization=args.optimization,
        native_library=args.native_library,
        quantization=args.quantize,
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
        onset_guard_overrides=_guard_overrides(args),
        **_cpu_options(args),
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
    print(f"execution_graph: {runtime.execution_path}, native_ssm: {runtime.native_ssm_enabled}")
    print(f"input_sr: {sr}, output_sr: {runtime.sample_rate}, samples: {enhanced.shape[0]}")
    print(f"algorithmic latency: {runtime.manifest.get('streaming_delay_frames', 0)} frames "
          f"(~{runtime.manifest.get('streaming_delay_frames', 0) * runtime.hop_length / 16:.0f} ms)")
    print(f"dry_blend: {runtime.dry_blend}")
    print(f"onset_guard: {runtime.onset_guard}")
    # Whole-clip level of the output against the input, which is where an onset
    # guard is visible at all: it hands spans back untouched, so a run with it on
    # sits closer to 0 dB than the same run without.
    n = min(enhanced.shape[0], samples.shape[0])
    span_db = 10.0 * np.log10(
        (float(np.mean(enhanced[:n] ** 2)) + 1e-12)
        / (float(np.mean(samples[:n] ** 2)) + 1e-12)
    )
    print(f"span level (output vs input, whole clip): {span_db:+.2f} dB")


def benchmark(args) -> None:
    runtime = StreamingDpcrnOrt(onnx_path=args.onnx_path, manifest_path=args.manifest_path,
                                provider=args.provider, **_cpu_options(args))
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    if args.input_audio is None:
        rng = np.random.default_rng(args.seed)
        samples = rng.standard_normal(runtime.sample_rate * args.seconds).astype(np.float32) * 0.01
    else:
        from puresound.audio.io import AudioIO

        wav, _ = AudioIO.open(str(args.input_audio), target_lvl=None, resample_to=runtime.sample_rate)
        samples = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    if len(samples) == 0:
        raise ValueError("benchmark audio must not be empty")
    timings = []
    for repeat in range(args.warmup + args.repeats):
        runtime.reset()
        start = time.perf_counter()
        enhanced = np.concatenate([runtime.process_samples(samples), runtime.flush()])
        elapsed = time.perf_counter() - start
        if repeat >= args.warmup:
            timings.append(elapsed)
    elapsed = float(np.median(timings))
    audio_seconds = samples.shape[0] / runtime.sample_rate
    print(f"provider: {runtime.providers}")
    print(f"execution_graph: {runtime.execution_path}, native_ssm: {runtime.native_ssm_enabled}")
    print(f"audio_seconds: {audio_seconds:.3f}")
    print(f"elapsed_seconds: {elapsed:.3f}")
    print(f"rtf: {elapsed / audio_seconds:.4f}")
    print(f"rtf_repeats: {[round(value / audio_seconds, 6) for value in timings]}")
    # ORT reports 0 when the count was left to it (one thread per physical core).
    threads = runtime.session.get_session_options().intra_op_num_threads
    print(f"warmup: {args.warmup}, repeats: {args.repeats}, intra_op_threads: "
          f"{threads or 'ORT default (physical cores)'}")
    print("timing: STFT + model + overlap-add + flush; excludes session load and warmup")
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


def _aligned_max_abs(offline: np.ndarray, streaming: np.ndarray, lag: int, trim: int) -> float:
    """Worst-case sample error at the lag SI-SDR chose.

    SI-SDR is scale-invariant and averages; a post-graph stage applied to the
    wrong span shows up here first, because it is a handful of samples that are
    completely different rather than a whole signal that is slightly off.
    """
    x, y = (offline[lag:], streaming) if lag >= 0 else (offline, streaming[-lag:])
    n = min(len(x), len(y))
    if n <= 2 * trim:
        return float("nan")
    return float(np.abs(x[trim : n - trim] - y[trim : n - trim]).max())


def verify(args) -> None:
    import torch

    from puresound.streaming import load_streaming_dpcrn_model

    frame_model = load_streaming_dpcrn_model(args.config_path, args.checkpoint_path).eval()
    system_model = frame_model.system_model.eval()
    runtime = StreamingDpcrnOrt(
        onnx_path=args.onnx_path,
        manifest_path=args.manifest_path,
        provider=args.provider,
        onset_guard_overrides=_guard_overrides(args),
        **_cpu_options(args),
    )

    if args.input_audio is not None:
        from puresound.audio.io import AudioIO

        wav, _ = AudioIO.open(str(args.input_audio), target_lvl=None, resample_to=runtime.sample_rate)
        wav = wav[:, : args.seconds * runtime.sample_rate]
    else:
        rng = np.random.default_rng(args.seed)
        wav = torch.from_numpy((rng.standard_normal(args.seconds * runtime.sample_rate) * 0.2).astype(np.float32)).view(1, -1)

    # The runtime applies the manifest's dry_blend and onset guard after the
    # graph, so the offline reference has to apply the same ones -- otherwise this
    # compares two different systems and reports the difference as an alignment
    # error: the blend alone makes a bit-exact export read as a large parity error.
    blend = runtime.dry_blend
    guard = runtime.onset_guard
    with torch.no_grad():
        offline = (
            system_model(wav, dry_blend=blend, onset_guard=guard)
            .squeeze()
            .cpu()
            .numpy()
        )

    L = wav.shape[1]
    chunks = [runtime.process_samples(wav[0, i : i + 1024].numpy()) for i in range(0, L, 1024)]
    chunks.append(runtime.flush())
    streaming = np.concatenate(chunks)

    delay_frames = int(runtime.manifest.get("streaming_delay_frames", 0))
    max_lag = max(1200, (delay_frames + 4) * runtime.hop_length)
    lag, sisdr = _align_sisdr(offline, streaming, max_lag=max_lag, trim=args.trim)
    max_abs = _aligned_max_abs(offline, streaming, lag=lag, trim=args.trim)
    print(f"algorithmic latency: {delay_frames} frames (~{delay_frames * runtime.hop_length / 16:.0f} ms)")
    print(f"measured alignment lag: {lag} samples ({lag / runtime.sample_rate * 1000:.1f} ms)")
    print(f"offline reference dry_blend: {blend} (matched to the manifest)")
    print(f"offline reference onset_guard: {guard}")
    print(f"offline vs ORT streaming SI-SDR (aligned + trimmed): {sisdr:.1f} dB")
    print(f"offline vs ORT streaming max abs diff (aligned + trimmed): {max_abs:.3e}")
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
    p.add_argument("--optimization", choices=("none", "portable", "cpu"), default="none",
                   help="portable: fixed batch + frequency-major + single-step LSTM; cpu: also export Mamba SSM companion")
    p.add_argument("--native-library", type=Path, default=None,
                   help="required to validate the native companion for --optimization cpu")
    p.add_argument("--quantize", choices=("none", "int8"), default="none",
                   help="int8: dynamic int8 LSTM/MatMul/Gemm weights for CPU; lossy, so gate the result")
    p.add_argument(
        "--dry-blend",
        dest="dry_blend",
        type=float,
        default=0.9,
        help="over-suppression relief the RUNTIME applies after the graph "
        "(default 0.9, the released setting; 1.0 disables it). Caps suppression "
        "at 20*log10(1 - dry_blend) -- 0.9 means -20 dB.",
    )
    # The same four knobs every eval stage spells, so an export and a scorecard
    # can be the same operating point. Recorded in the manifest under
    # `onset_guard`; omitted, the export carries no guard at all.
    add_onset_guard_arg(p)
    p.set_defaults(func=export)

    p = sub.add_parser("infer", help="run streaming ONNX inference on an audio file")
    p.add_argument("onnx_path", type=Path)
    p.add_argument("input_audio", type=Path)
    p.add_argument("output_audio", type=Path)
    p.add_argument("--manifest_path", type=Path, default=None)
    p.add_argument("--provider", choices=PROVIDER_CHOICES, default="auto")
    p.add_argument(
        "--no-onset-guard",
        dest="no_onset_guard",
        action="store_true",
        help="ignore the onset guard the manifest records (A/B against it)",
    )
    p.set_defaults(func=infer)
    _add_cpu_options(p)

    p = sub.add_parser("benchmark", help="benchmark streaming ONNX runtime")
    p.add_argument("onnx_path", type=Path)
    p.add_argument("--manifest_path", type=Path, default=None)
    p.add_argument("--provider", choices=PROVIDER_CHOICES, default="auto")
    p.add_argument("--seconds", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--input_audio", type=Path, default=None, help="benchmark a complete real audio file")
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--repeats", type=int, default=3)
    p.set_defaults(func=benchmark)
    _add_cpu_options(p)

    p = sub.add_parser("verify", help="offline vs ORT streaming parity (aligned + trimmed)")
    p.add_argument("config_path", type=Path)
    p.add_argument("checkpoint_path", type=Path)
    p.add_argument("onnx_path", type=Path)
    p.add_argument("--manifest_path", type=Path, default=None)
    p.add_argument("--input_audio", type=Path, default=None, help="if omitted, uses synthetic noise")
    p.add_argument("--provider", choices=PROVIDER_CHOICES, default="cpu")
    p.add_argument("--seconds", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trim", type=int, default=1200, help="edge samples to drop before scoring")
    p.add_argument(
        "--no-onset-guard",
        dest="no_onset_guard",
        action="store_true",
        help="drop the manifest's onset guard from BOTH sides of the comparison",
    )
    p.set_defaults(func=verify)
    _add_cpu_options(p)
    return parser


if __name__ == "__main__":
    cli = build_parser()
    parsed = cli.parse_args()
    parsed.func(parsed)
