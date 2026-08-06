# DPCRN Streaming ONNX Runtime

繁體中文版本：[dpcrn_onnx.zh-TW.md](dpcrn_onnx.zh-TW.md)

DPCRN streaming inference uses a per-frame ONNX model. Python owns audio
buffering, the fixed Hann STFT, overlap-add iSTFT, and ONNX Runtime state
management; ONNX Runtime runs one DPCRN feature frame at a time. This is the
deployment path of the released voice-isolate checkpoints (see
`egs/voice_isolate/pretrained_ckpt/streaming/`) — **the production path**,
not a legacy one, so this doc goes deeper than its DPARN counterpart.

`puresound/streaming/dpcrn.py`'s own module docstring states the relationship
between the two backbones plainly: DPCRN and DPARN both subclass the same
`Unet` base, so the encoder/decoder stepping, state layout, feature/mask/
iSTFT front-end, export, and manifest are identical between the two
streaming modules. The **only** structural difference is the bottleneck
block: DPCRN's intra path is a bidirectional LSTM over the frequency axis
(time-independent, carries no cross-frame state), whereas DPARN uses
self-attention. The inter path (a unidirectional LSTM over time) is the same
`SingleRNN` in both, and is the only operator whose `(h, c)` state must
persist across frames.

## Supported Configuration

`validate_streaming_dpcrn_config` checks the recipe before export and fails
fast with `ValueError` on anything outside:

- `dataset.target_sample_rate: 16000`, `ConvEncDec` frontend (`sr: 16000`,
  Hann window, `win_length <= fft_length`), frozen encoder/features
- `features.feats_type: complex`, `drop_stft_first_bin: True`
- DPCRN backbone with `stride_t: 1` and `dilation_t: 1` on every down layer;
  `norm_type` one of `cLN` / `iLN` / `bN2d` (`bN2d` is allowed here — unlike
  [DPARN's validator](dparn_onnx.md) — because `BatchNorm2d` in `eval()`
  mode applies fixed running stats per `(freq, time)` location, so it
  carries no cross-frame state)

`transpose_delay` must stay `False`: `transpose_delay=True` would make the
decoder anti-causal (it would need future frames) and breaks streaming
parity. This check only raises a `warnings.warn`, not a hard failure —
double-check it in your own recipes.

## Look-ahead: how future-buffering actually works

**Causal (`delay=[0,0,0]`)**: the per-frame graph streams bit-exact with
**zero** added latency — no extra state beyond the down/up conv caches and
the inter-LSTM `(h, c)`.

**Look-ahead (`delay > 0` on some down layers, e.g. the released
`[1,1,1]`)**: offline DPCRN lets each down layer peek `delay[i]` future
frames via right-padding. A causal per-frame model cannot peek forward, so
it instead **delays its own output** by the same number of frames,
buffering what the offline path saw "in the future" as extra persistent
state. This is bit-exact with offline DPCRN — just delayed — verified by
`test/test_utils/test_dpcrn_streaming.py`'s full-utterance parity test
(relative error < 1e-3 once the streaming output is realigned by the
bottleneck delay).

Three extra pieces of state realize this, tracked on `DpcrnStreamingState`
in addition to the causal `down_caches` / `up_caches` / `h_states` /
`c_states`:

- **`skip_caches`** — one FIFO delay line per U-Net skip connection that
  needs it. Down layer `k`'s cumulative look-ahead is `cum[k] =
  sum(delay[:k+1])`; the bottleneck (and hence the whole main path) ends up
  delayed by `D = cum[-1]` frames. A skip from down layer `k` is *itself*
  only delayed by `cum[k]` frames at the point it branched off, so it must
  be pushed through an extra `D - cum[k]`-frame FIFO before being
  concatenated with the main path at the matching up layer — otherwise the
  skip and the main path would refer to different offline time steps. Skip
  layers where `D - cum[k] == 0` need no delay line at all
  (`skip_delay_layers` only lists the ones that do).
- **`noisy_cache`** — a `D`-frame delay line for the noisy spectrum
  (`features_for_enhanced`) that the mask is applied to, so the mask (which
  emerges already delayed by `D` frames) and the spectrum it multiplies
  refer to the same offline frame.
- **`counter`** — a frame index used only to gate the inter-LSTM state
  during the first `D` frames of a stream (`warmup_frames = D`): the causal
  down-path's first `D` output frames are phantom startup transients with
  no offline equivalent (there's no "past" yet to look ahead from), so the
  model zeros `next_h`/`next_c` (`keep = counter >= D`) until the pipeline
  has flushed them. Without this gate, the inter-LSTM's first real update
  would start from a contaminated state instead of a clean one, corrupting
  the whole stream, not just the startup transient.

The algorithmic latency is `model.streaming_delay` (== `model.
bottleneck_delay`) frames — `3` frames (~30 ms at hop 160, i.e. the released
config) for the shipped checkpoints. **Any offline↔streaming comparison must
realign by this many frames and trim both ends** before comparing, or the
latency itself reads as error. The `verify` CLI command below does this
automatically; see `_offline_vs_streaming_rel` in
`test/test_utils/test_dpcrn_streaming.py` for the alignment search it
performs (`test_dpcrn_streaming_matches_offline_for_lookahead_model` asserts
the best-matching delay equals `model.bottleneck_delay` exactly).

Two ready-to-use recipes if you want to inspect either shape directly:
`egs/voice_isolate/config/exp/train_dpcrn_wide_causal.yaml` (`delay=[0,0,0]`)
and `egs/voice_isolate/config/exp/train_dpcrn_wide_antisup.yaml`
(`delay=[1,1,1]`, look-ahead) — the same two configs the streaming test
suite itself loads.

## ORT streaming state handling

`StreamingDpcrnOrt` is not a separate implementation — it is literally
`puresound.streaming.dparn.StreamingDparnOrt` re-exported under a DPCRN name
(`from puresound.streaming.dparn import StreamingDparnOrt as
StreamingDpcrnOrt`). The runtime is entirely manifest-driven (dispatched by
`manifest["processor"] == "stft_frame_ort"`, every state tensor read back by
name from the JSON), so the one ORT class serves both backbones unchanged —
there is nothing DPCRN-specific in the runtime at all.

What that runtime does per call:

- **`reset(batch_size=1)`** — zero-fills every state tensor from
  `manifest["state_shapes"]`, and clears the internal `input_buffer` and the
  `ola`/`ola_norm` overlap-add accumulators. Only `batch_size=1` is
  supported for waveform streaming (raises `ValueError` otherwise).
- **`process_samples(samples)`** — appends `samples` to `input_buffer`,
  then, while a full `win_length`-sample window is available: takes an STFT
  frame, runs it plus the full state dict through the ONNX session
  (`session.run(...)`), overwrites `self.state` from the outputs, iSTFTs
  the enhanced frame, and overlap-adds it into `ola`/`ola_norm`
  (normalizing by accumulated squared-window energy, floor `1e-8`),
  emitting exactly one `hop_length`-sample chunk per input frame consumed.
  Arbitrary chunk sizes are accepted —
  `test_streaming_ort_is_chunk_invariant_with_identity_session` checks that
  feeding samples in 3 arbitrarily-sized pieces vs. all at once produces
  identical output.
- **`flush()`** — zero-pads and drains whatever is left in `input_buffer`,
  plus whatever tail remains in the `ola` accumulator (normalized the same
  way).

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
    StreamingDpcrnOrt,              # ORT runtime: process_samples()/flush()
    export_streaming_dpcrn_onnx,    # config + ckpt -> onnx + manifest
    load_streaming_dpcrn_model,     # torch per-frame model (StreamingDpcrnFrameModel)
    validate_streaming_dpcrn_config,
)
```

The JSON manifest records fft/hop, state names/shapes, the algorithmic
latency (`streaming_delay_frames`), and — for released checkpoints — a
`recommended_inference` block. The shipped `dpcrn_v8.json`'s block, for
example:

```json
"recommended_inference": {
  "dry_blend": 0.9,
  "note": "out = 0.9*enhanced + 0.1*input, with the input latency-aligned to the enhanced stream (algorithmic latency 3 frames / 30 ms). Bounds attenuation at any point to -20 dB, which trades a little residual interferer for far fewer deletions on capture chains the model was not trained on."
}
```

`dry_blend` is applied by the *caller*, not baked into the exported graph:
blend the enhanced frame with the latency-aligned input frame at inference
time (zero added latency, since the input is already available once the
enhanced frame is). It is a deployment-time safety knob, unrelated to the
look-ahead state described above.

The manifest-driven portable SDK (`sdk/python/puresound_streaming`,
`processor: stft_frame_ort`) loads these exports directly.

## Gap: `vad_head` / `dist_head` are silently dropped

`DPCRN` (`puresound/nnet/dpcrn.py`) supports two optional auxiliary heads off
the bottleneck, each gated by its own config block
(`backbone_args.vad_head.enabled` / `backbone_args.dist_head.enabled`):
`self.vad_head`/`self.dist_head`, populating `self.last_vad_logits`/
`self.last_dist_preds` as a side effect of the offline `forward()` call. The
production recipe `egs/voice_isolate/config/train_dpcrn.yaml` currently
ships with `dist_head.enabled: True` (an utterance-level distance/DRR
regression auxiliary, trained with `DistHeadRegressionLoss`) — and that
config's own comment already flags exactly the gap this section documents:
*"Training-only: inference and the streaming export never read it."*

`StreamingDpcrnFrameModel._forward_feature_frame` (and its DPARN
counterpart) never calls the offline `backbone.forward()` at all — it
reimplements the down/bottleneck/up path frame-by-frame directly against
`backbone.cnn_down` / the DPRNN blocks / `backbone.cnn_up`, and stops there.
There is no code path anywhere in `puresound/streaming/dpcrn.py` that
references `vad_head`, `dist_head`, `last_vad_logits`, or
`last_dist_preds`.

**Practically**: if you export a checkpoint whose backbone config enables
`vad_head` and/or `dist_head`, the streaming ONNX graph still exports
successfully and still produces a correct enhanced audio frame — it just
silently has no VAD or distance output at all, even though the checkpoint's
weights include a trained head. There is no warning at export time. If a
downstream consumer needs those signals during streaming, they currently
have to be computed some other way (e.g. running the full offline model on
buffered audio) — the per-frame ONNX path does not expose them.

## Notes

- The ONNX model is feature-frame only; audio STFT and iSTFT are
  intentionally outside the graph.
- The runtime currently supports waveform streaming with batch size 1.
