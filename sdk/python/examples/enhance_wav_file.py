"""Enhance a WAV file with an exported PureSound streaming ONNX model.

Offline batch example: read a file, stream it through the runtime frame by
frame, write the enhanced result. The same runtime works for realtime input --
call ``process_samples`` / ``process_int16`` repeatedly with whatever chunk size
your audio source delivers (see ``livekit_frame_processor.py``).

Verified against the training repo's own ORT inference path
(``puresound.streaming.StreamingDpcrnOrt``): byte-identical output for the
DPCRN ``dpcrn_wide_antisup`` export.

This example is not imported by tests because ``soundfile`` is an example-only
dependency (the SDK itself needs only numpy + onnxruntime).

Usage:
    python enhance_wav_file.py model.onnx input.wav output.wav
    python enhance_wav_file.py model.onnx input.wav output.wav \
        --manifest model.json --provider cuda

Input must be 16 kHz mono float32 (or any depth soundfile reads as float32).
The SDK does NOT resample -- convert other sample rates before calling it, e.g.
    sox input48k.wav -r 16000 -c 1 input.wav
"""

import argparse

import numpy as np
import soundfile as sf

from puresound_streaming import PureSoundStreamingRuntime


def enhance_file(
    onnx_path: str,
    input_wav: str,
    output_wav: str,
    manifest_path: str | None = None,
    provider: str = "auto",
) -> None:
    runtime = PureSoundStreamingRuntime(onnx_path, manifest_path, provider=provider)

    wav, sr = sf.read(input_wav, dtype="float32")
    if wav.ndim > 1:  # downmix to mono
        wav = wav.mean(axis=1)
    if sr != runtime.sample_rate:
        raise ValueError(
            f"input is {sr} Hz but the model expects {runtime.sample_rate} Hz; "
            "resample first (the SDK does not resample)."
        )

    runtime.reset()
    # process_samples buffers internally, so any chunk size works -- feeding the
    # whole file at once and feeding 20 ms blocks give the same result. flush()
    # emits the final overlap-add tail once the input ends.
    enhanced = runtime.process_samples(wav)
    enhanced = np.concatenate([enhanced, runtime.flush()])

    sf.write(output_wav, enhanced, runtime.sample_rate)
    print(f"provider: {runtime.providers}")
    print(f"wrote {output_wav} ({enhanced.shape[0]} samples @ {runtime.sample_rate} Hz)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("onnx_path", help="exported streaming .onnx model")
    parser.add_argument("input_wav", help="16 kHz mono input audio")
    parser.add_argument("output_wav", help="path to write the enhanced audio")
    parser.add_argument(
        "--manifest",
        default=None,
        help="model .json manifest (defaults to the .onnx path with a .json suffix)",
    )
    parser.add_argument(
        "--provider",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="ONNX Runtime execution provider (default: auto)",
    )
    args = parser.parse_args()
    enhance_file(
        args.onnx_path,
        args.input_wav,
        args.output_wav,
        manifest_path=args.manifest,
        provider=args.provider,
    )


if __name__ == "__main__":
    main()
