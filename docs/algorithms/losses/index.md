# puresound.nnet.loss

繁體中文版本：[index.zh-TW.md](index.zh-TW.md)

The loss library. A recipe lists its losses under `loss_func`; each entry's
`type` is a class name resolved with `getattr` on `puresound.nnet.loss` by
`puresound.recipes.init_loss_func`, and `args` are its constructor arguments.
A class must be listed in `puresound.nnet.loss.__all__` to be reachable.

```yaml
loss_func:
  - type: SDRLoss
    weighted: 1.0
    args: {...}
```

A loss declares what it is called with in `required_inputs`, a tuple of
provider names in the order of its `forward` arguments; the training module
(`puresound.system.base.invoke_loss`) looks each name up in its provider table
and fails loudly when one is missing. A loss that declares nothing gets
`("enhanced", "target")`. Providers include the waveforms, the batch dict, the
VAD targets and the backbone's side outputs (`vad_logits`, `dist_preds`,
`identity_emb`, `proximity`, ...), which exist only when the matching head is
enabled on the backbone.

| Page | Classes | Status | What it computes |
| --- | --- | --- | --- |
| [sdr](sdr.md) | `SDRLoss` | active | time-domain SDR family (SI-SNR, SD-SDR, SA-SDR, t-SDR, ...) with a separate path for target-absent rows |
| [stft_loss](stft_loss.md) | `MultiResolutionSTFTLoss`, `SpectralLoss`, `OverSuppressionLoss` | active | spectral distances; `OverSuppressionLoss` is a one-sided penalty on removed target energy |
| [active_bins](active_bins.md) | `ActiveBinLogMagLoss` | active | per-bin log-magnitude error counted only on bins where the clean target has speech |
| [asr_feature](asr_feature.md) | `ASRFeatureLoss` | active | feature matching through a frozen self-supervised speech model, a differentiable proxy for word errors |
| [residual](residual.md) | `ResidualReferenceLoss` | active | supervises `noisy - enhanced` against a reference residual |
| [dist](dist.md) | `DistHeadRegressionLoss` | active | NaN-masked distance / DRR regression for the `DistHead` |
| [vad](vad.md) | `VADActivityLoss`, `VADHeadBCELoss`, `BackgroundVADHeadBCELoss`, `F1_loss` | active | frame-activity supervision for the enhanced output and the VAD heads |
| [identity](identity.md) | `IdentityContrastiveLoss` | library | turn-level speaker contrast on the `IdentityHead` against an EMA teacher |
| [proximity](proximity.md) | `RelativeProximityLoss` | active | orders user and bystander turns by rendered distance on the `ProximityHead` readout |
| [inherit](inherit.md) | `AnchorInheritanceLoss` | library | hinge keeping the user's onset gain above the gain applied to what preceded the user |
| [spk](spk.md) | `AAMsoftmax`, `SphereFace2`, `GE2ELoss`, `TripletLoss` | frozen legacy | speaker-embedding losses for the speaker-verification and TSE recipes |
| — (`loss/__init__.py`) | `TimeDomainBasicLoss` | library | plain L1 or MSE between waveforms (`name: l1 \| mse`, `reduction`) |

- **active** — used by a maintained recipe (at least one class of the module).
- **library** — importable and tested, not used by a maintained recipe.
- **frozen legacy** — kept for existing recipes, not developed further.
