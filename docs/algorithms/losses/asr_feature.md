# puresound.nnet.loss.asr_feature

繁體中文版本：[asr_feature.zh-TW.md](asr_feature.zh-TW.md)

Feature-matching loss through a frozen self-supervised speech encoder
(torchaudio wav2vec2 / HuBERT / WavLM). It compares the enhanced output with the
clean target in the encoder's intermediate layers, a differentiable proxy for
intelligibility.

## Class: `ASRFeatureLoss`

### What it computes

```
F_l(x) = output of transformer layer l of the frozen encoder
loss   = mean over l in layers of  d(F_l(enhanced), F_l(target))
d      = L1 | MSE | mean(1 - cosine similarity over the feature axis)
```

The target's features are computed under `no_grad`; the gradient flows back to
`enhanced` only.

### Constructor

```python
ASRFeatureLoss(
    bundle: str = "HUBERT_BASE",       # torchaudio.pipelines bundle name
    layers=(6, 9),                     # transformer-layer indices whose outputs are matched
    loss: str = "l1",                  # "l1" | "mse" | "cosine"
    crop_seconds: float | None = None, # score a random window of this length; None = whole row
)
```

- `bundle`: e.g. `HUBERT_BASE`, `WAV2VEC2_BASE`, `WAV2VEC2_ASR_BASE_960H`,
  `WAVLM_BASE_PLUS` (all 16 kHz). The loss does not resample; the input must be
  at the bundle's sample rate.
- `layers`: the encoder runs up to `max(layers) + 1` layers.
- `loss`: anything else raises `NotImplementedError`.
- `crop_seconds`: when the row is longer, one random window of this length
  (the same window for both signals) is scored per call.

### Inputs

`forward(enhanced, target) -> Tensor` (scalar). Waveforms `[B, T]`, `[B, 1, T]`
or `[T]`; the two are cut to the shorter length. Declares no `required_inputs`,
so it receives `("enhanced", "target")`.

### Config usage

```yaml
loss_func:
  - type: ASRFeatureLoss          # general phonetic content
    weighted: 1.0
    args: {bundle: HUBERT_BASE, layers: [6, 9], loss: cosine, crop_seconds: 6.}
  - type: ASRFeatureLoss          # noise/overlap-robust features
    weighted: 1.0
    args: {bundle: WAVLM_BASE_PLUS, layers: [6, 9], loss: cosine, crop_seconds: 6.}
```

### Design notes

- Signal losses (SDR, MR-STFT) reward suppressing interference but do not
  penalise destroying intelligibility, so a model trained on them alone can
  over-suppress speech. SSL encoders trained on large amounts of real speech
  encode phonetic content robustly across conditions; matching their features
  targets the content a recogniser needs. WER itself is not differentiable.
- Mid layers carry the most phonetic information, hence the default `(6, 9)`.
- Two bundles can be stacked because they fail differently: HuBERT carries
  general phonetic content; WavLM, pretrained with denoising and simulated
  overlap, reacts to noise and overlap conditions. `cosine` is scale-invariant,
  so stacked terms with equal `weighted` have equal influence; with `l1` the
  weights need rebalancing per bundle because feature magnitudes differ.
- The encoder is frozen and kept out of the module registry (held in a list),
  so it is not saved in checkpoints, not synced by DDP and not seen by the
  optimiser. It is moved to the input's device on first use and run in fp32
  with autocast disabled.
- `crop_seconds` exists because the encoders are transformers: memory grows with
  the square of the row length while every other term grows linearly. The loss
  is a per-frame distance, so a random window means the same as the whole row
  and covers it over an epoch.
