# puresound.nnet.loss

English version: [index.md](index.md)

Loss 函式庫。這裡的每個 class 都能透過 recipe config 的
`loss_func[].type` 取用（由 `puresound.recipes.init_loss_func` 以
`getattr` 在這個 package 上解析出來）。

| Module | Classes | Purpose |
|--------|---------|---------|
| [loss/sdr](sdr.md) | `SDRLoss` | 時域 SDR 系列 loss（SI-SNR / SD-SDR / SA-SDR / t-SDR / ...），另有專門處理 target-absent rows 的 `inactive_sdr_loss` 路徑 |
| [loss/stft_loss](stft_loss.md) | `MultiResolutionSTFTLoss`、`SpectralLoss`、`OverSuppressionLoss` | 頻譜類 loss；`OverSuppressionLoss` 是單邊的 anti-deletion 壓力 |
| [loss/asr_feature](asr_feature.md) | `ASRFeatureLoss` | frozen-SSL feature matching，作為對抗 over-suppression 的可微分 WER proxy |
| [loss/residual](residual.md) | `ResidualReferenceLoss` | 把 `noisy - enhanced` 監督到一個參考殘差上 |
| [loss/dist](dist.md) | `DistHeadRegressionLoss` | 給 `DistHead` 輔助任務用、NaN-masked 的 distance/DRR regression |
| [loss/vad](vad.md) | `VADActivityLoss`、`VADHeadBCELoss`、`BackgroundVADHeadBCELoss`、`F1_loss` | frame-activity 監督訊號（gate head 訓練用） |
| [loss/spk](spk.md) | `AAMsoftmax`、`SphereFace2`、`GE2ELoss`、`TripletLoss` | speaker-embedding losses |
| `loss/__init__` | `TimeDomainBasicLoss` | 對 waveform 做的單純 L1/MSE |
