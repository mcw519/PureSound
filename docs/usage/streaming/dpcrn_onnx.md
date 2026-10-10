# DPCRN streaming ONNX

Traditional Chinese: [dpcrn_onnx.zh-TW.md](dpcrn_onnx.zh-TW.md)

DPCRN streaming inference uses a per-frame ONNX model. Python (or the portable
SDK) owns audio buffering, the fixed Hann STFT, overlap-add iSTFT and the state
tensors; ONNX Runtime runs one DPCRN feature frame at a time. This is the
deployment path of every released checkpoint -- voice isolation
(`egs/voice_isolate/pretrained_ckpt/streaming/`) and noise suppression
(`egs/noise_suppression/pretrained_ckpt/streaming/`).

DPCRN and DPARN subclass the same `Unet` base, so the encoder/decoder stepping,
state layout, feature/mask/iSTFT front-end, export and manifest are shared
(`puresound/streaming/base.py`). What differs is the bottleneck: DPCRN's intra
path runs over frequency and carries no cross-frame state; the inter path over
time is the only operator whose state persists across frames.

## Export, verify, run

`egs/voice_isolate/scripts/streaming_onnx.py` drives the library for any DPCRN
checkpoint, of either task:

```bash
# export a checkpoint to per-frame ONNX + JSON manifest
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_curriculum_v1.ckpt /path/to/model.onnx

# offline vs ORT streaming parity (aligned by the latency, edges trimmed)
uv run python egs/voice_isolate/scripts/streaming_onnx.py verify \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    egs/voice_isolate/pretrained_ckpt/dpcrn_curriculum_v1.ckpt \
    egs/voice_isolate/pretrained_ckpt/streaming/dpcrn_curriculum_v1.onnx \
    --input_audio speech.wav

# file-to-file streaming inference, and the streaming real-time factor
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer <onnx> in.wav out.wav
uv run python egs/voice_isolate/scripts/streaming_onnx.py benchmark <onnx> --seconds 10
```

| Subcommand | Flags worth knowing |
| --- | --- |
| `export` | `--manifest_path` (default: beside the ONNX), `--opset`, `--dry-blend` (default 0.9, the voice-isolation release setting), `--onset-guard` with `--guard-t-arm`, `--guard-t-forget`, `--guard-tau-dn`, `--guard-margin-db` |
| `verify` | `--input_audio` (default: synthetic noise), `--seconds`, `--trim`, `--provider`, `--no-onset-guard` |
| `infer` | `--manifest_path`, `--provider`, `--no-onset-guard` |
| `benchmark` | `--seconds`, `--provider` |

**Noise-suppression checkpoints are exported at `--dry-blend 1.0`**, the
operating point their records were scored at; the default 0.9 would ship a
system the gate never measured:

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/noise_suppression/config/infer_dpcrn.yaml \
    egs/noise_suppression/pretrained_ckpt/dpcrn_mamba_v2.ckpt \
    /path/to/dpcrn_mamba_v2.onnx --dry-blend 1.0
```

Judge a new export on real speech (`verify --input_audio`). The default
white-noise probe is a stress signal: its parity figure tracks how aggressively a
version modulates its mask, not whether the streaming graph is correct.

## CPU optimization for DPCRN and DPCRN-Mamba

`export --optimization portable` fixes batch size to one and keeps the recurrent
blocks in frequency-major layout, using standard ONNX operators. `cpu` also
exports a sibling `model.native.onnx` with a fused FP32 Mamba state-update/readout
operator. Both reuse the checkpoint and state ports and add no buffering delay.
The portable path supports an intra LSTM with either a one-layer inter LSTM,
Mamba, or Mamba-context. The inter LSTM advances all frequency positions together
with one combined input/hidden projection and standard ONNX gate operations;
the intra frequency LSTM retains its trained bidirectional recurrence.
The native companion option requires a Mamba inter path.

Build the optional library on Linux, then export:

```bash
uv run python -m puresound.streaming.native.build \
    --output /path/to/libpuresound_ssm.so
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/noise_suppression/config/infer_dpcrn_mamba_wide.yaml \
    egs/noise_suppression/pretrained_ckpt/dpcrn_mamba_v3.ckpt \
    /path/to/v3.onnx --dry-blend 1.0 --optimization cpu \
    --native-library /path/to/libpuresound_ssm.so
uv run python egs/voice_isolate/scripts/streaming_onnx.py benchmark \
    /path/to/v3.onnx --provider cpu --native-ssm required \
    --native-library /path/to/libpuresound_ssm.so \
    --input_audio puresound/web/static/samples/noisy-speech.wav \
    --warmup 2 --repeats 3
```

The build requires a C++17 compiler (`g++` or `CXX`) and performs no downloads.
It is compiled for the baseline x86-64 ISA and picks its AVX2 or AVX-512 kernel
when loaded, so one build runs on any x86-64 Linux host with a compatible C
library. Rebuilding replaces the file atomically, so running sessions keep the
copy they loaded.
Only the fused kernel uses reassociated FP32 arithmetic, so small numerical
differences are expected. The primary ONNX is portable and needs no compiler or
custom library. Browser/WASM deployments use that primary graph; use the normal
web asset builder to prepare its manifest.

```python
from puresound.streaming import StreamingOrt

runtime = StreamingOrt("/path/to/v3.onnx", provider="cpu",
                       native_library="/path/to/libpuresound_ssm.so",
                       native_ssm="required")
print(runtime.execution_path, runtime.native_ssm_enabled)
```

`native_ssm="auto"` (default) uses the companion only for a CPU session with a
specified library, falling back to the primary ONNX if native loading fails.
`off` always uses the primary; `required` raises instead of falling back.
`PURESOUND_ORT_SSM_LIBRARY` supplies the library to Python facade/web-server
sessions as well. There is no compilation during inference. Deploy the primary
ONNX, its JSON, and its sibling native ONNX together; the manifest records no
machine-specific library path. Optimized CPU manifests recommend one intra-op
thread, overridable with `intra_op_num_threads` or CLI `--threads`.

The benchmark reports the actual selected graph, native status, thread count,
median RTF and repeat values. It measures STFT, model, overlap-add and flush,
excluding session load and warmup. Algorithmic latency remains 52–62 ms for v3.
All six DPCRN streaming model-zoo releases use the portable optimization by
default. The three NS Mamba releases also include native companions. `load_model`
uses the portable graph without a library, and selects native CPU execution when
`PURESOUND_ORT_SSM_LIBRARY` points to a compatible local build.

Re-export and validate the entire catalog, updating its SHA256 values in place:

```bash
uv run python model_zoo/reexport.py --native-library /path/to/libpuresound_ssm.so
uv run python sdk/web/tools/build_assets.py
```

### int8 quantization

`export --quantize int8` (with `--optimization portable` or `cpu`) stores the
LSTM and matrix-product weights as int8 and quantizes their activations at run
time, in both the primary graph and the native companion. Export still validates
the float graph strictly first. It then checks only that the int8 graph is finite
and within a sanity bound, and records the measured error in the manifest's
`quantization` section. That bound does not decide quality: int8 output shifts
with any change to the graph or kernel, so gate every int8 export again.
NS v3 ships this as the `flash` variant (`--variant flash`).

## Supported configuration

`validate_streaming_dpcrn_config` checks the recipe before export and raises
`ValueError` on anything outside:

- `dataset.target_sample_rate: 16000`
- `ConvEncDec` encoder with a Hann window, `sr: 16000`, `fmax: 8000`,
  `trainable: False`, `win_length <= fft_length`, `hop_length > 0`
- `features.feats_type: complex`, `drop_stft_first_bin: True`,
  `trainable: False`, no `include_specaug`
- `DPCRN` backbone with `input_dim == fft_length // 2`, `norm_type` one of `cLN` /
  `iLN` / `bN2d`, `skip_conv: False`, and `stride_t: 1` and `dilation_t: 1` on
  every down layer. `bN2d` is allowed because `BatchNorm2d` in `eval()` applies
  fixed running statistics per (frequency, time) location, so it carries no
  cross-frame state.

The frame model also requires a complex mask. It supports the inter path as an
LSTM or as a Mamba block (`inter_type: mamba` or `mamba_context`), perceptual
banding of the bottleneck, attention intra paths, and the deep-filter residual
head. It refuses `inter_type: lstm+mamba`, whose parallel branch needs state
ports the manifest layout does not have, rather than export the LSTM branch
alone.

Two conditions only warn: a non-zero `delay` (look-ahead, supported -- see below)
and `transpose_delay: True`, which makes the decoder use future frames and breaks
streaming parity. Keep `transpose_delay: False`.

## Look-ahead: how future-buffering works

**Causal (`delay=[0,0,0]`)**: the per-frame graph matches the offline model with
**zero** added latency -- no state beyond the down/up conv caches and the inter
path's state.

**Look-ahead (`delay > 0` on some down layers, e.g. `[1,1,1]`)**: offline DPCRN
lets each down layer see `delay[i]` future frames through right-padding. A causal
per-frame model cannot look forward, so it **delays its own output** by the same
number of frames and keeps what the offline path saw "in the future" as extra
state. The streamed output is the offline result, delayed.

Three extra pieces of state realise this, on `DpcrnStreamingState`, beside the
causal `down_caches` / `up_caches` / `h_states` / `c_states`:

- **`skip_caches`** -- one FIFO per U-Net skip connection that needs it. Down
  layer `k`'s cumulative look-ahead is `cum[k] = sum(delay[:k+1])`, and the main
  path ends up delayed by `D = cum[-1]` frames. A skip from layer `k` is only
  delayed by `cum[k]` where it branches off, so it passes through an extra
  `D - cum[k]`-frame FIFO before it meets the main path; otherwise the two would
  refer to different offline frames. Layers where `D - cum[k] == 0` need none
  (`skip_delay_layers` lists the ones that do).
- **`noisy_cache`** -- a `D`-frame delay line for the noisy spectrum the mask is
  applied to, so the mask (which emerges `D` frames late) and the spectrum it
  multiplies refer to the same frame.
- **`counter`** -- a frame index that gates the inter-path state during the first
  `D` frames (`warmup_frames = D`). The causal down path's first `D` output frames
  are start-up transients with no offline equivalent, so the model zeros the next
  inter state until the pipeline has flushed them; without the gate the first real
  update would start from a contaminated state and corrupt the whole stream.

With the deep-filter residual head, a `df_cache` holds the most recent frames of
the aligned noisy low band the filter reads.

The algorithmic latency is `model.streaming_delay` (equal to
`model.bottleneck_delay`) frames, recorded in the manifest as
`streaming_delay_frames`: 3 frames, 30 ms at hop 160, for the released `[1,1,1]`
look-ahead. **Any offline-versus-streaming comparison must realign by that many
frames and trim both ends**, or the latency reads as error; `verify` does both.
`test/streaming/test_dpcrn_streaming.py` holds the parity tests for the causal,
look-ahead, Mamba, banded, attention and deep-filter variants.

That figure is the look-ahead the graph adds. A sample also waits for the
analysis window around it before its hop is emitted, so the frame runtime's
input-to-output delay runs from look-ahead + window − hop to look-ahead + window:
52–62 ms for the released geometry (30 ms look-ahead, 32 ms window, 10 ms hop).
Budget a device against the upper figure.

## The ORT runtime

`StreamingDpcrnOrt` is the shared `puresound.streaming.StreamingOrt`: the runtime
is manifest-driven (`processor: stft_frame_ort`, every state tensor read back by
name), so one class serves both backbones and every inter-path type.

```python
from puresound.streaming import StreamingOrt

runtime = StreamingOrt("model.onnx", provider="auto")   # manifest: model.json beside it
runtime.reset()
out = runtime.process_samples(chunk)      # any chunk size
tail = runtime.flush()
```

- **`reset(batch_size=1)`** zero-fills every state tensor from
  `manifest["state_shapes"]` and clears the input buffer and the overlap-add
  accumulators. Waveform streaming supports batch size 1 only.
- **`process_samples(samples)`** appends to the input buffer and, while a full
  window is available, takes an STFT frame, runs it with the state through the
  session, stores the new state, iSTFTs the frame and overlap-adds it, emitting
  one `hop_length` chunk per frame. Arbitrary chunk sizes give the same output as
  one call. A non-finite input sample is read as silence: the state is recurrent,
  so one NaN would otherwise corrupt every later sample until `reset`.
- **`flush()`** feeds zero frames until the graph's look-ahead has drained, so
  the stream ends with exactly `input length + streaming_delay_frames *
  hop_length` samples: the whole input, that many samples late. The partially
  covered overlap-add remainder past it maps to padding and is dropped.
- `provider` is `auto` (CUDA, then CoreML, then CPU), `cpu`, `cuda`, `coreml`, or
  `mps` (an alias for CoreML).

## Post-graph stages the runtime applies

The traced graph contains the model and nothing after it. Two stages that shape
the deployed output are therefore **recorded in the manifest and applied by the
runtime**, so a deployment that runs the graph runs the system that was
evaluated. An absent key means the stage is off.

### `recommended_inference`: the dry blend

```json
"recommended_inference": {
  "dry_blend": 0.9,
  "suppression_ceiling_db": -20.0,
  "note": "out = 0.9*enhanced + 0.1*input, with the input latency-aligned ..."
}
```

`out = dry_blend * enhanced + (1 - dry_blend) * input`, with the input delayed by
`streaming_delay_frames` so both refer to the same frame. It adds no latency, and
it caps attenuation at `20*log10(1 - dry_blend)` -- -20 dB at 0.9 -- so a residual
reported near that figure is measuring the blend, not the model.
`StreamingOrt(..., postprocess_overrides={...})` changes it per request.

### `onset_guard`: protect the first second of a talker

```json
"onset_guard": {
  "t_arm_s": 1.0, "t_forget_s": 5.0, "tau_up_s": 0.05, "tau_dn_s": 2.0,
  "margin_db": 8.0, "floor_win_s": 2.0, "floor_rise_db_per_s": 3.0,
  "init_s": 0.2, "hangover_s": 0.2, "min_run_s": 0.1, "snap": 0.001,
  "note": "out = input, bit for bit, until a talker has been heard for 1 s ..."
}
```

Written by `export --onset-guard`. The streaming state treats whoever it heard
last as the foreground, so the first second of the first talker -- and of the next
near talker after a pause -- can be attenuated as a foreground change. Until a
talker has been heard for `t_arm_s` of sustained speech (`margin_db` above a
tracked noise floor), the output is `g*input + (1 - g)*enhanced` with `g = 1`,
the input bit for bit; then it releases toward the model with time constant
`tau_dn_s`, and `t_forget_s` of quiet re-arms it for the next onset. It trades
suppression depth in the first second for fewer deleted words; where to sit on
that trade is a product choice (`puresound/system/onset_guard.py`).

At runtime it costs, per hop, one frame energy, a running-minimum floor and one
first-order integrator step: no FFT, no model state, nothing learned.

**The one-hop lag rule.** Frame energy spans two hops, so the guard's frame `t` is
decided once input hop `t + 1` has arrived, and output hop `m` needs the gain of
input frame `m - streaming_delay_frames`. When output hop `m` is emitted the
runtime has received `m*hop_length + win_length` samples, so the gain applied is
the exact one the offline guard computes, with no added latency, whenever

```
streaming_delay_frames + win_length//hop_length - 2 >= 0
```

which the runtime exposes as `runtime.onset_guard_lookahead_hops`. At the 512/160
geometry this holds for both look-ahead and causal exports -- the analysis window
alone covers the lag. A window shorter than two hops on a zero-latency graph is
refused with `ValueError` rather than served a gain from the wrong frame.

The guard is applied **after** the dry blend, because it has to be able to
restore the whole input rather than `1 - dry_blend` of it -- the order
`SISO.forward` uses offline. A request can refuse a recorded guard
(`StreamingOrt(..., onset_guard_overrides={"enabled": False})`, or
`--no-onset-guard` on `infer` / `verify`) or replace individual knobs. `verify`
applies the same guard to its offline reference, so its parity figure stays the
graph's own.

## Auxiliary heads

A backbone with `vad_head` and/or `background_vad_head` exports them as side
outputs (`vad_logit`, `background_vad_logit`), listed in the manifest under
`extra_output_names`; each head adds its own state ports. Enabling them does not
change the audio. The logits describe the bottleneck frame they were computed
from, which **leads** the emitted audio by `streaming_delay_frames`; a consumer
aligning them to the output has to account for that. `StreamingOrt(...,
collect_extras=True)` keeps them, and `drain_extras()` returns and clears them so
a long stream stays bounded. The recipe
`egs/voice_isolate/config/infer_dpcrn_heads.yaml` loads a checkpoint with these
heads.

`dist_head` is deliberately not exported: it is utterance-pooled and has no
per-frame meaning. It stays a training-only auxiliary.

## Library API

```python
from puresound.streaming import (
    StreamingDpcrnOrt,              # the shared ORT runtime
    export_streaming_dpcrn_onnx,    # config + ckpt -> onnx + manifest
    load_streaming_dpcrn_model,     # torch per-frame model (StreamingDpcrnFrameModel)
    validate_streaming_dpcrn_config,
)
```

`export_streaming_dpcrn_onnx(config, ckpt, onnx, manifest=None, opset_version=17,
postprocess=IDENTITY, onset_guard=None)` traces the frame model with the
TorchScript exporter, checks the ONNX output (and each side output) against the
PyTorch frame model, and writes the manifest: sample rate, FFT/window/hop, input,
output and state names, state shapes, `streaming_delay_frames`,
`extra_output_names`, preferred providers, and the post-graph sections above.

## Portable SDK

A deployment that only needs inference uses the standalone SDK in
[`sdk/python`](../../../sdk/python/README.md), which depends only on NumPy and
ONNX Runtime and loads these exports directly (`processor: stft_frame_ort`):

```python
from puresound_streaming import PureSoundStreamingRuntime

runtime = PureSoundStreamingRuntime("model.onnx", "model.json", provider="auto")
enhanced = runtime.process_samples(audio_float32)
tail = runtime.flush()
```

It carries a NumPy transcription of the dry blend and the onset guard;
`test/streaming/test_sdk_runtime.py` pins it bit-identical to the library
runtime.

## Notes

- The ONNX model is feature-frame only; audio STFT and iSTFT are intentionally
  outside the graph.
- The runtime supports waveform streaming with batch size 1.
