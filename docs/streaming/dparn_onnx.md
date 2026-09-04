# DPARN Streaming ONNX Runtime

繁體中文版本：[dparn_onnx.zh-TW.md](dparn_onnx.zh-TW.md)

> **Status: legacy** — kept working and frozen: no new features, no rewrites.

DPARN streaming inference uses a feature-frame ONNX model. Python owns audio
buffering, fixed Hann STFT, overlap-add iSTFT, and ONNX Runtime state
management. ONNX Runtime runs one DPARN feature frame at a time.

This path is intended for the voice-isolate DPARN recipe configured for
16 kHz audio.

## Supported Configuration

`validate_streaming_dparn_config` checks the recipe before export. Most
checks are **hard requirements** — any violation raises `ValueError`
immediately:

- `dataset.target_sample_rate: 16000`
- `ConvEncDec` frontend with a Hann window, `encoder_args.sr: 16000`,
  `fmax: 8000`, `win_length <= fft_length`, `hop_length > 0`
- `encoder_args.trainable: False`
- `features.feats_type: complex`, `drop_stft_first_bin: True`,
  `features.trainable: False`, no `include_specaug`
- `DPARN` backbone with `input_dim: 256`
- `norm_type` one of `cLN` / `iLN` / `bN2d` (`BatchNorm2d` in `eval()` mode
  applies fixed running stats and is therefore frame-independent)
- `skip_conv: False`
- `stride_t: 1` and `dilation_t: 1` on every down layer

Two more checks are **soft** — they only emit `warnings.warn`; export still
succeeds:

- `transpose_delay` not `True` — warns that the exported ONNX is per-frame
  but not truly causal (live audio will leak future frames)
- any non-zero `delay` — warns that down layers peek at future frames

Unlike the [DPCRN streaming path](dpcrn_onnx.md), **DPARN's exporter has no
look-ahead compensation machinery**: it does not buffer future frames or
gate RNN state during a warmup period. A non-causal `delay`/`transpose_delay`
config still exports (with a warning) and still passes the offline-vs-ONNX
comparison check performed at export time, but the resulting model is not
real streaming — that check only verifies a whole-utterance (offline)
comparison, not behavior under a live buffer that only has the past.

**The repo's DPARN recipe is not a streaming recipe.** The one real DPARN
config, `egs/noise_suppression/config/dparn.yaml`, sets
`encoder_args.trainable: True` and adds a `freq_eq` block — it is an offline
enhancement recipe by design, and a learned frontend cannot be exported as
the fixed STFT this runtime relies on (Python owns STFT/iSTFT; ORT only runs
the per-frame backbone). Exporting it fails on `encoder.trainable`, which is
the check working as intended, not a defect. To export a DPARN checkpoint for
streaming, train one with `encoder_args.trainable: False` and a fixed Hann
frontend; its `norm_type: bN2d` is fine as-is.

## Export

Export a trained checkpoint to a streaming feature-frame ONNX model:

```bash
uv run python <your_recipe>/streaming_onnx.py export \
  <your_recipe>/config/dparn.yaml \
  /path/to/model.ckpt \
  /path/to/model.onnx
```

The exporter also writes `/path/to/model.json`. The manifest contains:

- audio settings: sample rate, FFT length, window length, hop length
- ONNX input and output names
- explicit streaming state tensor names and shapes
- preferred ONNX Runtime providers
- DPARN streaming delay in frames
- post-graph stages the RUNTIME applies, because the graph contains the model
  and nothing after it: `recommended_inference` (`dry_blend`) and, if the
  export was given one, `onset_guard`. Both keys are read by the shared
  `StreamingOrt` and by the portable SDK, so a DPARN manifest carrying them
  behaves the same way; an absent key means that stage is off. This legacy
  exporter does not itself write `onset_guard` (`export_streaming_dpcrn_onnx`
  does) — the mechanism, its cost, and the one-hop analysis-lag rule are
  documented once, in [dpcrn_onnx.md](dpcrn_onnx.md#post-graph-stage-two-the-onset-guard-onset_guard).

During export, PureSound compares the ONNX frame output with the PyTorch
frame wrapper output. Export fails if they do not match within tolerance.

## Inference

Run streaming inference on an audio file:

```bash
uv run python <your_recipe>/streaming_onnx.py infer \
  /path/to/model.onnx \
  input.wav \
  output.wav \
  --provider auto
```

Provider options:

- `auto`: use CUDA when available, then CoreML on macOS, otherwise CPU
- `cuda`: request CUDA with CPU fallback
- `coreml`: request Apple's CoreML execution provider (macOS)
- `mps`: alias for `coreml`; ONNX Runtime has no native MPS provider
- `cpu`: force CPU

Benchmark realtime factor:

```bash
uv run python <your_recipe>/streaming_onnx.py benchmark \
  /path/to/model.onnx \
  --provider cuda \
  --seconds 10
```

## Library API

```python
from puresound.streaming import StreamingDparnOrt

runtime = StreamingDparnOrt("/path/to/model.onnx", provider="auto")
runtime.reset()

enhanced_0 = runtime.process_samples(chunk_0)
enhanced_1 = runtime.process_samples(chunk_1)
tail = runtime.flush()
```

`process_samples()` accepts arbitrary chunk sizes and emits any complete
overlap-add output that is ready. `flush()` pads and drains the remaining
audio.

For lower-level testing or custom export flows:

```python
from puresound.streaming import load_streaming_dparn_model

model = load_streaming_dparn_model(
    "<your_recipe>/config/dparn.yaml",
    "/path/to/model.ckpt",
)
state = model.initial_state(batch_size=1)
enhanced_frame, next_state = model.forward_frame(noisy_frame, state)
```

`noisy_frame` and `enhanced_frame` have shape `[batch, 257, 2]`, where the
last axis is real and imaginary parts.

## Portable SDK

Projects that only need inference do not need to install the full PureSound
training package. Use the standalone SDK in [sdk/python](../../sdk/python):

```bash
python -m pip install sdk/python
```

Then load exported artifacts:

```python
from puresound_streaming import PureSoundStreamingRuntime

runtime = PureSoundStreamingRuntime("model.onnx", "model.json", provider="auto")
enhanced = runtime.process_samples(audio_float32)
tail = runtime.flush()
```

For realtime systems such as LiveKit agents, use the int16 helpers:

```python
out_pcm = runtime.process_int16(in_pcm)
tail_pcm = runtime.flush_int16()
```

The SDK depends only on NumPy and ONNX Runtime. It does not import
`puresound`, PyTorch, torchaudio, Lightning, or the training recipe stack.
DPARN is the first export that uses the SDK's `stft_frame_ort` processor
profile. Future PureSound streaming exports should add or reuse
manifest-driven processor profiles rather than introducing model-specific
runtime classes.

## Gradio Demo

The voice-isolate demo supports two backends:

- `PyTorch offline`: load a `.ckpt`, `.pt`, or `.pth` checkpoint and run the
  original offline model path.
- `ORT streaming`: select a `.onnx` model and run the streaming ONNX Runtime
  path.

Start the demo:

```bash
uv run python <your_recipe>/demo.py \
  --config_path <your_recipe>/config/dparn.yaml
```

Use **Refresh checkpoints** to scan the configured `work_folder` and `exp`
directory. Then choose backend `ORT streaming` and select an exported
`.onnx` model.

## Notes

- The ONNX model is feature-frame only; audio STFT and iSTFT are
  intentionally outside the graph.
- Existing checkpoints should be trained or fine-tuned with the fixed Hann
  frontend settings above. Learned STFT kernels are not part of the v1
  streaming contract.
- The runtime currently supports waveform streaming with batch size 1.
