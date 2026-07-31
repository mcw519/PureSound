# DPCRN Streaming ONNX Runtime

DPCRN streaming inference uses a per-frame ONNX model. Python owns audio
buffering, the fixed Hann STFT, overlap-add iSTFT, and ONNX Runtime state
management; ONNX Runtime runs one DPCRN feature frame at a time. This is the
deployment path of the released voice-isolate checkpoints (see
`egs/voice_isolate/pretrained_ckpt/streaming/`).

## Supported Configuration

`validate_streaming_dpcrn_config` checks the recipe before export and fails fast
with a `ValueError` on anything outside:

- `dataset.target_sample_rate: 16000`, `ConvEncDec` frontend (`sr: 16000`,
  Hann window, `win_length <= fft_length`), frozen encoder/features
- `features.feats_type: complex`, `drop_stft_first_bin: True`
- DPCRN backbone with `stride_t: 1` and `dilation_t: 1` on every down layer;
  `norm_type` one of `cLN` / `iLN` / `bN2d`

**Look-ahead**: `delay=[0,0,0]` streams bit-exact with zero latency.
A look-ahead down-path (`delay > 0`, e.g. the released `[1,1,1]`) is also
bit-exact, but the streamed output carries a fixed algorithmic latency of
`sum(delay)` frames (3 frames = 30 ms at hop 160). The exporter bakes the
future-buffering into the graph as extra state — inter-LSTM warmup gating,
per-skip delay lines, and a noisy-spectrum delay — so no runtime code changes
are needed. **Any offline↔streaming comparison must align by that latency and
trim the edges**, or the delay reads as error; the `verify` command below does
this automatically.

## Export / Verify / Run

```bash
# export a checkpoint to per-frame ONNX + JSON manifest
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt /path/to/model.onnx

# offline vs ORT streaming parity (aligned + trimmed)
uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.onnx \
    --manifest_path egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_v8.json

# file-to-file streaming inference / RTF benchmark
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer  <onnx> in.wav out.wav
uv run python egs/voice_isolate/scripts/streaming_onnx.py benchmark <onnx>
```

## Library API

```python
from puresound.streaming import (
    StreamingDpcrnOrt,              # ORT runtime: process_chunk()/flush()
    export_streaming_dpcrn_onnx,    # config + ckpt -> onnx + manifest
    load_streaming_dpcrn_model,     # torch per-frame model (StreamingDpcrnFrameModel)
    validate_streaming_dpcrn_config,
)
```

The JSON manifest records fft/hop, state names/shapes, the algorithmic latency
(`streaming_delay_frames`), and — for released checkpoints — a
`recommended_inference` block (e.g. `dry_blend: 0.9`, applied by blending the
enhanced frame with the latency-aligned input frame; zero added latency).

The manifest-driven portable SDK (`sdk/python/puresound_streaming`,
`processor: stft_frame_ort`) loads these exports directly.
