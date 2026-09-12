# puresound.nnet.loss

繁體中文版本：[index.zh-TW.md](index.zh-TW.md)

Loss library. Every class here is reachable from a recipe config's
`loss_func[].type` (resolved with `getattr` on this package by
`puresound.recipes.init_loss_func`).

| Module | Classes | Purpose |
|--------|---------|---------|
| [loss/sdr](sdr.md) | `SDRLoss` | time-domain SDR family (SI-SNR / SD-SDR / SA-SDR / t-SDR / ...), with a dedicated `inactive_sdr_loss` path for target-absent rows |
| [loss/stft_loss](stft_loss.md) | `MultiResolutionSTFTLoss`, `SpectralLoss`, `OverSuppressionLoss` | spectral losses; `OverSuppressionLoss` is the one-sided anti-deletion pressure |
| [loss/asr_feature](asr_feature.md) | `ASRFeatureLoss` | frozen-SSL feature matching, a differentiable WER proxy against over-suppression |
| [loss/residual](residual.md) | `ResidualReferenceLoss` | supervise `noisy - enhanced` against a reference residual |
| [loss/dist](dist.md) | `DistHeadRegressionLoss` | NaN-masked distance/DRR regression for the `DistHead` auxiliary |
| [loss/vad](vad.md) | `VADActivityLoss`, `VADHeadBCELoss`, `BackgroundVADHeadBCELoss`, `F1_loss` | frame-activity supervision (gate head training) |
| [loss/spk](spk.md) | `AAMsoftmax`, `SphereFace2`, `GE2ELoss`, `TripletLoss` | speaker-embedding losses |
| `loss/__init__` | `TimeDomainBasicLoss` | plain L1/MSE on waveforms |
