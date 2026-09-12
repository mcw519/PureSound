# puresound.nnet.loss.asr_feature

繁體中文版本：[asr_feature.zh-TW.md](asr_feature.zh-TW.md)

ASR-aware perceptual loss. Signal losses (SI-SDR / MR-STFT) reward suppressing
interference but do not penalise destroying intelligibility, so a model tuned on
them alone can over-suppress real speech and raise WER. This loss matches the
enhanced output's features to the clean target's features inside a **frozen,
self-supervised speech encoder** (torchaudio wav2vec2 / HuBERT / WavLM, 16 kHz,
differentiable), a differentiable proxy for the non-differentiable WER.

## Class: `ASRFeatureLoss`

### Constructor

```python
ASRFeatureLoss(
    bundle: str = "HUBERT_BASE",   # any torchaudio.pipelines bundle name
    layers=(6, 9),                 # encoder layers whose features are matched
    loss: str = "l1",              # "l1" | "mse" | "cosine"
)
```

**Parameters:**
- `bundle` – torchaudio pipeline bundle (`HUBERT_BASE`, `WAVLM_BASE_PLUS`, ...).
  Stacking two terms with different bundles covers different failure modes:
  HuBERT carries general phonetic content, WavLM (pretrained with denoising and
  simulated overlap) reacts to the noise/overlap conditions where deletion
  appears.
- `layers` – which transformer layers' features enter the loss.
- `loss` – feature distance. `cosine` is scale-invariant, so multiple stacked
  terms with equal weights get equal influence; with `l1` the weights need
  rebalancing per bundle.

### `forward(enhanced, target) -> Tensor`

Waveforms in, scalar loss out. The encoder is frozen, kept **out of the module
registry** (list-wrapped) so it is neither saved into checkpoints nor synced by
DDP, moved to the input device lazily, and run in fp32 (autocast disabled).

### Config usage

```yaml
loss_func:
  - type: ASRFeatureLoss
    weighted: 1.0
    args: {bundle: HUBERT_BASE, layers: [6, 9], loss: cosine}
  - type: ASRFeatureLoss
    weighted: 1.0
    args: {bundle: WAVLM_BASE_PLUS, layers: [6, 9], loss: cosine}
```
