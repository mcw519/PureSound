# puresound.nnet.lobe.heads

繁體中文版本：[heads.zh-TW.md](heads.zh-TW.md)

Auxiliary readouts from a backbone's bottleneck feature map `[N, C, F, T]`.
They do not change the enhanced output: the backbone stores each result on an
attribute after its forward, and a loss or an evaluation reads it from there.
Any backbone with a `[N, C, F, T]` bottleneck can attach them; `DPCRN` does,
reading the map after its dual-path blocks.

| head | `backbone_args` key | output attribute | shape | trained by |
| --- | --- | --- | --- | --- |
| `VADHead` | `vad_head` | `last_vad_logits` | `[N, T]` | [`VADHeadBCELoss`](../../losses/vad.md) |
| `VADHead` | `background_vad_head` | `last_background_vad_logits` | `[N, T]` | [`BackgroundVADHeadBCELoss`](../../losses/vad.md) |
| `DistHead` | `dist_head` | `last_dist_preds` | `[N, 3]` | [`DistHeadRegressionLoss`](../../losses/dist.md) |
| `IdentityHead` | `identity_head` | `last_identity_emb` | `[N, T, dim]` | [`IdentityContrastiveLoss`](../../losses/identity.md) |
| `ProximityHead` | `proximity_head` | `last_proximity` | `[N, T]` | [`RelativeProximityLoss`](../../losses/proximity.md) |

## Configuration

Each head has a config model (`VADHeadConfig`, `DistHeadConfig`,
`IdentityHeadConfig`, `ProximityHeadConfig`) and a
`from_config(config, *, enc_channels)` classmethod that returns `None` when
the block is absent or `enabled` is false. A disabled head adds no parameters
and no operations. The models reject unknown keys, so a misspelled key fails
instead of silently taking a default.

```yaml
model:
  backbone:
    type: DPCRN
    backbone_args:
      vad_head:
        enabled: True
        hidden: 64
        kernel_t: 5
        ema_taus_s: [0.05, 0.25, 1.0, 4.0]   # optional; omit for the plain head
      background_vad_head: {enabled: True}
      dist_head: {enabled: True, hidden: 128}
      identity_head: {enabled: True, dim: 64, kernel_t: 5}
      proximity_head: {enabled: True, hidden: 64}
      expose_bottleneck: True                 # needed by IdentityContrastiveLoss
```

Checkpoint keys follow the attribute the backbone stores a head under (for
example `backbone.vad_head.*`), not this module's path.

## Class: `VADHead`

Frame-level speech-activity logits, causal in time.

```python
VADHead(
    enc_channels: int,                           # bottleneck channels C
    hidden: int,
    kernel_t: int,                               # causal conv window, frames
    ema_taus_s: Optional[Sequence[float]] = None,  # EMA time constants, seconds
    frame_rate: float = 100.0,                   # bottleneck frames per second
)
```

Config keys: `enabled` (False), `hidden` (None = `enc_channels`), `kernel_t`
(5), `ema_taus_s` (None; a non-empty list of positive values), `frame_rate`
(100.0 = 16000 / hop 160).

```
h = mean over F of x                                   # [N, C, T]
h = [h, ema_1(h), ..., ema_K(h)]                       # if ema_taus_s: [N, (K+1)C, T]
h = Linear(-> hidden)(h)
h = SiLU(Conv1d(hidden, hidden, kernel_t)(left_pad(h, kernel_t - 1)))
logit = Conv1d(hidden, 1, 1)(h)                        # [N, T]
```

Each EMA uses `a = 1 - exp(-1 / (tau * frame_rate))`,
`s_t = s_{t-1} + a (h_t - s_{t-1})` from `s_{-1} = 0`, and is divided by
`1 - (1 - a)^(t+1)` so it is the average of the frames seen so far from the
first frame on.

Streaming: `initial_stream_state(batch_size=1, device="cpu", dtype=None)`
returns `(ema [N, K, C], conv_cache [N, hidden, kernel_t - 1], count [N, 1])`,
and `step(x [N, C, F, 1], state) -> (logit [N, 1], new_state)` reproduces
`forward` one frame at a time. The streaming DPCRN export publishes
`vad_head` and `background_vad_head` as per-frame outputs `vad_logit` and
`background_vad_logit`, with state ports `<name>_ema`, `<name>_conv_cache`
and `<name>_count`.

The model never multiplies its output by this head. A consumer that gates on
it applies `sigmoid` and a threshold it calibrates itself.

## Class: `DistHead`

Utterance-level regression of proximity labels.

```python
DistHead(enc_channels: int, hidden: int = 128, n_out: int = 3)
# [N, C, F, T] -> mean over F and T -> Linear -> SiLU -> Linear -> [N, n_out]
```

With `n_out=3` the outputs are
`[foreground_drr / drr_scale, log10(foreground_distance), log10(nearest_interferer_distance)]`;
`drr_scale` belongs to the loss. Training-only: it is pooled over the whole
utterance, so the streaming export leaves it out.

## Class: `IdentityHead`

Per-frame speaker-identity embedding, causal in time.

```python
IdentityHead(enc_channels: int, dim: int = 64, kernel_t: int = 5)
```

```
h = mean over F of x                                  # skipped if x is already [N, C, T]
h = LayerNorm(DepthwiseConv1d(kernel_t)(left_pad(h, kernel_t - 1)))   # [N, T, C]
e = L2-normalise(Linear(C -> dim)(h))                 # [N, T, dim]
```

`IdentityContrastiveLoss` mean-pools the embedding over each single-talker
turn and contrasts it with the turn embeddings of a stop-gradient EMA copy of
the head. The copy runs on the graph-carrying bottleneck, so the backbone
needs `expose_bottleneck: true`. Training-only.

## Class: `ProximityHead`

Per-frame relative-proximity scalar, causal in time.

```python
ProximityHead(enc_channels: int, hidden: int = 64, kernel_t: int = 5)
```

Same trunk as `IdentityHead` (causal depthwise conv and LayerNorm), then
`Linear(C -> hidden) -> SiLU -> Linear(hidden -> 1)`, giving `[N, T]`. The only
config key besides `enabled` is `hidden`; `kernel_t` stays at 5 so both
per-frame heads share one time scale. Training-only.

`RelativeProximityLoss` supervises only differences: turns are ordered by
their rendered distance with a margin, and the ordering must also hold on a
second rendering of the same rows through another capture chain.

## Related

[`multiframe.DeepFilterResidualHead`](multiframe.md) is also attached to
`DPCRN` (`df_head`), but it changes the output and reads the decoder, not the
bottleneck.

## Design notes

- `VADHead`'s plain window is `kernel_t` frames (50 ms at the default), while a
  presence decision needs on the order of a second of speech. The EMA bank
  lets the head read several time scales at once, and the ratio of a fast to
  a slow average measures envelope modulation depth, a cue that separates near
  from far talkers where per-frame features do not. The decay rates are fixed:
  a learned rate can drift to single-frame or to constant, and a fixed one
  keeps the time scale an explicit hyperparameter.
- The EMA recurrence runs in float32 with autocast disabled: at `tau = 4 s`
  the step is `a = 0.0025`, which a bf16 state loses entirely.
  `torchaudio.functional.lfilter` is called with `clamp=False`, since
  bottleneck features are not bounded by 1.
- Debiasing removes the start-up transient of a zero-initialised average, and
  gives streaming the same semantics by carrying `(state, count)`.
- `DistHead` puts pressure on the bottleneck to encode physical proximity cues
  (DRR, source distance) rather than the capture-chain signature of the
  training data.
- `IdentityHead` normalises its output so the contrastive cosine is a dot
  product and no turn can win by growing its norm. It is not asked to be
  discriminative frame by frame, only turn by turn.
- `ProximityHead` has no absolute scale because a fixed threshold on a readout
  that drifts across capture chains and checkpoints does not hold; only
  within-recording differences are meaningful.
