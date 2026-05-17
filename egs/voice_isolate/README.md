# Voice Isolate

Train a near-field voice isolation model. The foreground speaker is simulated
as a near-field source, while other sampled speakers are placed farther away in
the same room and treated as interference to suppress.

This recipe reuses the noise suppression training loop with:

- source-level shoebox room simulation
- near-field foreground distance range
- far-field interferer distance range
- dynamic Silero VAD labels
- SkiM baseline backbone
- DPARN streaming ONNX Runtime deployment

## Prepare metadata

The metadata format is the same as the noise suppression recipe.

```bash
uv run python egs/voice_isolate/prepare_metafile.py \
  data/voice_isolate_train \
  /path/to/wav.scp \
  /path/to/utt2spk
```

For DNS-Challenge-4, use the adapter to scan `clean_fullband` and create the
PureSound train/valid metafiles:

```bash
uv run python egs/voice_isolate/prepare_dns_challenge.py \
  /path/to/DNS-Challenge-4 \
  --output-dir ./data/voice_isolate_dns4 \
  --speaker-id-strategy parent \
  --valid-ratio 0.05
```

The adapter writes:

- `voice_isolate_dns4_train.csv`
- `voice_isolate_dns4_valid.csv`
- `voice_isolate_dns4_config_hint.yaml`

Use the generated config hint to update `train_metafile`, `valid_metafile`,
`augmentation_noise.noise_folder`, and `augmentation_reverb.rir_folder` in
`config/skim.yaml`. If your DNS clean speech directory groups speakers one
level above the waveform folders, use `--speaker-id-strategy grandparent`.

## Inspect training samples

```bash
uv run python egs/voice_isolate/main.py \
  egs/voice_isolate/config/skim.yaml \
  --dump_training_samples True
```

The generated dummy samples contain three channels:

1. `noisy_speech`: near-field target plus far-field interferers
2. `clean_speech`: near-field target reference
3. `consistency_noise`: residual noise/interference

## Train

```bash
uv run python egs/voice_isolate/main.py \
  egs/voice_isolate/config/skim.yaml \
  --training True
```

Before training, update `train_metafile`, `valid_metafile`, `test_folder`, and
worker/GPU counts in [config/skim.yaml](config/skim.yaml).

## DPARN streaming ONNX export and runtime

The DPARN recipe in [config/dparn.yaml](config/dparn.yaml) is the realtime
baseline for streaming ONNX inference. It uses a fixed Hann frontend:

- 16 kHz audio
- 512-point FFT and 512-sample window
- 128-sample hop
- `fmax: 8000`
- non-trainable encoder and feature frontend

The streaming deployment artifact is two files:

- `model.onnx`: one-frame ONNX Runtime graph
- `model.json`: manifest that describes audio settings, state tensor names,
  state shapes, and the runtime processor profile

The exported ONNX model is a feature-frame model. Python or the portable SDK
handles audio buffering, STFT, iSTFT, overlap-add, and state management.

### Find a checkpoint

Training writes checkpoints under the configured `trainer.work_folder`. For the
default DPARN config this is:

```bash
egs/voice_isolate/exp/dparn_near_field_16k_v1/lightning_logs/version_0/checkpoints/
```

Example checkpoint:

```bash
egs/voice_isolate/exp/dparn_near_field_16k_v1/lightning_logs/version_0/checkpoints/epoch=0-step=250.ckpt
```

If `egs/voice_isolate/exp` is a symlink, use that path rather than a repo-root
`exp/` path.

### Export ONNX and manifest

Export a trained DPARN checkpoint:

```bash
uv run python egs/voice_isolate/streaming_onnx.py export \
  egs/voice_isolate/config/dparn.yaml \
  egs/voice_isolate/exp/dparn_near_field_16k_v1/lightning_logs/version_0/checkpoints/epoch=0-step=250.ckpt \
  /tmp/puresound_dparn_streaming.onnx \
  --manifest_path /tmp/puresound_dparn_streaming.json
```

During export, PureSound compares the PyTorch frame wrapper with the exported
ONNX Runtime graph. Export fails if the frame output does not match within the
configured tolerance.

The manifest includes fields such as:

```json
{
  "model_type": "dparn_streaming_frame",
  "processor": "stft_frame_ort",
  "sample_rate": 16000,
  "fft_length": 512,
  "win_length": 512,
  "hop_length": 128,
  "freq_bins": 257,
  "feature_bins": 256,
  "streaming_delay_frames": 5,
  "state_input_names": ["down_cache_0", "h_0", "c_0"],
  "state_output_names": ["next_down_cache_0", "next_h_0", "next_c_0"],
  "state_shapes": {
    "down_cache_0": [1, 2, 256, 1],
    "h_0": [1, 64, 128],
    "c_0": [1, 64, 128]
  }
}
```

`freq_bins: 257` is the full STFT frame input to the ONNX model. The training
feature frontend drops the first bin inside the graph, so `feature_bins: 256`.

### Benchmark runtime

Benchmark realtime factor:

```bash
uv run python egs/voice_isolate/streaming_onnx.py benchmark \
  /tmp/puresound_dparn_streaming.onnx \
  --manifest_path /tmp/puresound_dparn_streaming.json \
  --provider cpu \
  --seconds 10
```

Provider options are:

- `auto`: use CUDA when available, otherwise CPU
- `cuda`: request CUDA with CPU fallback
- `cpu`: force CPU

The current environment may only expose `CPUExecutionProvider`; the command
prints the provider actually used.

### Run file inference

Run streaming inference:

```bash
uv run python egs/voice_isolate/streaming_onnx.py infer \
  /tmp/puresound_dparn_streaming.onnx \
  input.wav \
  output.wav \
  --manifest_path /tmp/puresound_dparn_streaming.json \
  --provider auto
```

The input file is resampled to the manifest sample rate. The output length may
include streaming/STFT flush tail samples.

### Use the portable SDK in another project

For a project that should not depend on the full PureSound repository, install
or copy the portable SDK:

```bash
python -m pip install sdk/python
```

Then load the exported artifacts:

```python
import numpy as np

from puresound_streaming import PureSoundStreamingRuntime

runtime = PureSoundStreamingRuntime(
    "/tmp/puresound_dparn_streaming.onnx",
    "/tmp/puresound_dparn_streaming.json",
    provider="auto",
)

audio = np.zeros(16000, dtype=np.float32)
enhanced = runtime.process_samples(audio)
enhanced = np.concatenate([enhanced, runtime.flush()])
```

Realtime systems that use PCM int16 can use:

```python
out_pcm = runtime.process_int16(in_pcm)
tail_pcm = runtime.flush_int16()
```

The SDK is manifest-driven. It chooses a runtime processor from the manifest
`processor` field. DPARN currently exports `processor: "stft_frame_ort"`, but
the public SDK API remains model-neutral:

```python
PureSoundStreamingRuntime(...)
```

### Realtime integration notes

- Keep one runtime instance per audio stream because the runtime owns streaming
  state and overlap-add buffers.
- Call `reset()` when a new call/session starts.
- Feed arbitrary chunk sizes to `process_samples()`; the runtime emits any
  complete output that is ready.
- Call `flush()` when the stream ends.
- For LiveKit-style audio frames, convert `AudioFrame` PCM int16 data to a NumPy
  array, call `process_int16()`, then build a new frame from the returned PCM.

### Validation checklist

After export, run:

```bash
uv run python egs/voice_isolate/streaming_onnx.py benchmark \
  /tmp/puresound_dparn_streaming.onnx \
  --manifest_path /tmp/puresound_dparn_streaming.json \
  --provider cpu \
  --seconds 2
```

Then check file inference:

```bash
uv run python egs/voice_isolate/streaming_onnx.py infer \
  /tmp/puresound_dparn_streaming.onnx \
  /path/to/input.wav \
  /tmp/puresound_streaming_output.wav \
  --manifest_path /tmp/puresound_dparn_streaming.json \
  --provider cpu
```

For early checkpoints, the first few output samples may contain a large startup
transient. Use a more mature checkpoint for quality evaluation, or add a
call-level warmup/drop-initial-output policy in the host application.

## Gradio demo

Start the demo:

```bash
uv run python egs/voice_isolate/demo.py \
  --config_path egs/voice_isolate/config/dparn.yaml
```

Use backend `PyTorch offline` for normal checkpoints, or `ORT streaming` for an
exported `.onnx` model. The checkpoint dropdown also scans `.onnx` files under
the configured `work_folder` and `./exp`.

See [../../docs/streaming/dparn_onnx.md](../../docs/streaming/dparn_onnx.md)
for the full streaming API and deployment notes.
