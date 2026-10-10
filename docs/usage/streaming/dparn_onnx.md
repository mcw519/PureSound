# DPARN streaming ONNX

Traditional Chinese: [dparn_onnx.zh-TW.md](dparn_onnx.zh-TW.md)

> **Status: legacy** -- kept working and frozen: no new features, no rewrites.
> No released checkpoint uses it; the deployment path is [DPCRN](dpcrn_onnx.md).

DPARN streaming inference uses a feature-frame ONNX model, with the same split as
DPCRN: Python owns audio buffering, the fixed Hann STFT, overlap-add iSTFT and the
state tensors; ONNX Runtime runs one DPARN feature frame at a time. Both
backbones share the frame-model base, the exporter, the manifest format and the
runtime (`puresound/streaming/base.py`).

## Supported configuration

`validate_streaming_dparn_config` checks the recipe before export. These are
hard requirements -- a violation raises `ValueError`:

- `dataset.target_sample_rate: 16000`
- `ConvEncDec` encoder with a Hann window, `sr: 16000`, `fmax: 8000`,
  `trainable: False`, `win_length <= fft_length`, `hop_length > 0`
- `features.feats_type: complex`, `drop_stft_first_bin: True`,
  `trainable: False`, no `include_specaug`
- `DPARN` backbone with `input_dim: 256`, `norm_type` one of `cLN` / `iLN` /
  `bN2d`, `skip_conv: False`, and `stride_t: 1` and `dilation_t: 1` on every down
  layer

Two checks only warn, and export still succeeds:

- `transpose_delay: True` -- the exported model is per-frame but not causal
- a non-zero `delay` -- down layers look at future frames

Unlike the DPCRN path, **the DPARN exporter has no look-ahead compensation**: it
does not buffer future frames or gate recurrent state during a warm-up. A
non-causal `delay` / `transpose_delay` config still exports and still passes the
export-time frame check, but the result is not real streaming -- that check
compares one frame against the PyTorch frame model, not behaviour under a live
buffer that only has the past.

**The repository's DPARN example is not a streaming recipe.**
`egs/noise_suppression/config/dparn.yaml` sets `encoder_args.trainable: True`
and adds a `freq_eq` block: it is an offline recipe, and a learned front-end
cannot be exported as the fixed STFT this runtime relies on. Exporting it fails
on `encoder.trainable`, which is the check working as intended. To export a DPARN
checkpoint for streaming, train it with `encoder_args.trainable: False` and a
fixed Hann front-end.

## Export

There is no command-line wrapper for DPARN; call the library:

```python
from puresound.streaming import export_streaming_dparn_onnx
from puresound.system.postprocess import Postprocessor

manifest = export_streaming_dparn_onnx(
    "path/to/dparn_recipe.yaml",
    "path/to/model.ckpt",
    "path/to/model.onnx",                      # manifest: model.json beside it
    postprocess=Postprocessor(dry_blend=1.0),  # recorded, applied by the runtime
)
```

The exporter writes the same manifest as DPCRN's: audio settings, input, output
and state names and shapes, preferred providers, `streaming_delay_frames`, and
the post-graph `recommended_inference` section. It does not take an onset guard;
a DPARN manifest that carries an `onset_guard` section is still honoured by the
runtime. The post-graph stages are documented once, in
[dpcrn_onnx.md](dpcrn_onnx.md#post-graph-stages-the-runtime-applies).

During export the ONNX frame output is compared with the PyTorch frame model;
export fails if they do not match within tolerance.

## Inference

The runtime is the shared `StreamingOrt`, re-exported as `StreamingDparnOrt`:

```python
from puresound.streaming import StreamingDparnOrt

runtime = StreamingDparnOrt("/path/to/model.onnx", provider="auto")
runtime.reset()

enhanced_0 = runtime.process_samples(chunk_0)
enhanced_1 = runtime.process_samples(chunk_1)
tail = runtime.flush()
```

`process_samples()` accepts arbitrary chunk sizes and emits whatever overlap-add
output is complete. `flush()` pads and drains the remaining audio. Providers are
`auto` (CUDA, then CoreML on macOS, otherwise CPU), `cuda`, `coreml`, `mps` (an
alias for CoreML) and `cpu`. The DPCRN command-line tool's `infer` and
`benchmark` subcommands read only the ONNX file and its manifest, so they run a
DPARN export too:

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer model.onnx in.wav out.wav
uv run python egs/voice_isolate/scripts/streaming_onnx.py benchmark model.onnx --seconds 10
```

For lower-level testing or custom export flows:

```python
from puresound.streaming import load_streaming_dparn_model

model = load_streaming_dparn_model("path/to/dparn_recipe.yaml", "path/to/model.ckpt")
state = model.initial_state(batch_size=1)
enhanced_frame, next_state = model.forward_frame(noisy_frame, state)
```

`noisy_frame` and `enhanced_frame` have shape `[batch, 257, 2]`, the last axis
holding real and imaginary parts.

## Portable SDK

The standalone SDK in [`sdk/python`](../../../sdk/python/README.md) loads DPARN
exports the same way as DPCRN ones (`processor: stft_frame_ort`):

```python
from puresound_streaming import PureSoundStreamingRuntime

runtime = PureSoundStreamingRuntime("model.onnx", "model.json", provider="auto")
enhanced = runtime.process_samples(audio_float32)
tail = runtime.flush()
out_pcm = runtime.process_int16(in_pcm)     # int16 helpers for real-time frameworks
tail_pcm = runtime.flush_int16()
```

## Notes

- The ONNX model is feature-frame only; audio STFT and iSTFT are intentionally
  outside the graph.
- Learned STFT kernels are not part of the streaming contract; train or fine-tune
  with the fixed Hann front-end settings above.
- The runtime supports waveform streaming with batch size 1.
