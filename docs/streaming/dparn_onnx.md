# DPARN Streaming ONNX Runtime

DPARN streaming inference uses a feature-frame ONNX model. Python owns audio
buffering, fixed Hann STFT, overlap-add iSTFT, and ONNX Runtime state
management. ONNX Runtime runs one DPARN feature frame at a time.

This path is intended for the voice-isolate DPARN recipe configured for 16 kHz
audio.

## Supported Configuration

The streaming exporter validates the config before exporting. The current v1
runtime supports:

- `dataset.target_sample_rate: 16000`
- `ConvEncDec` frontend with `fft_length: 512`, `win_length: 512`, `hop_length: 128`
- `encoder_args.sr: 16000`, `fmax: 8000`, `win_type: hann`
- `encoder_args.trainable: False`
- `features.feats_type: complex`
- `features.drop_stft_first_bin: True`
- `features.trainable: False`
- DPARN backbone with `delay: 0`, `stride_t: 1`, `dilation_t: 1`
- `transpose_delay: True`
- `norm_type: cLN` or `iLN`

Unsupported configs fail fast with a `ValueError`.

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

During export, PureSound compares the ONNX frame output with the PyTorch frame
wrapper output. Export fails if they do not match within tolerance.

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

- `auto`: use CUDA when available, otherwise CPU
- `cuda`: request CUDA with CPU fallback
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
overlap-add output that is ready. `flush()` pads and drains the remaining audio.

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

`noisy_frame` and `enhanced_frame` have shape `[batch, 257, 2]`, where the last
axis is real and imaginary parts.

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

The SDK depends only on NumPy and ONNX Runtime. It does not import `puresound`,
PyTorch, torchaudio, Lightning, or the training recipe stack. DPARN is the first
export that uses the SDK's `stft_frame_ort` processor profile. Future PureSound
streaming exports should add or reuse manifest-driven processor profiles rather
than introducing model-specific runtime classes.

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
directory. Then choose backend `ORT streaming` and select an exported `.onnx`
model.

## Notes

- The ONNX model is feature-frame only; audio STFT and iSTFT are intentionally
  outside the graph.
- Existing checkpoints should be trained or fine-tuned with the fixed Hann
  frontend settings above. Learned STFT kernels are not part of the v1
  streaming contract.
- The runtime currently supports waveform streaming with batch size 1.
