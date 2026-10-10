# 神經網路

English version: [index.md](index.md)

`puresound.nnet` 收錄模型 backbone、組成它們的積木（`puresound.nnet.lobe`）、
前後的 feature 與 mask 層，以及 loss 函式庫（`puresound.nnet.loss`）。Recipe 在
`backbone.type` 以 class 名稱指定模型，名稱由 `getattr` 在 `puresound.nnet` 上
解析，所以 class 必須先列進 `puresound.nnet.__all__`，YAML 才能引用它。

## Backbones

| 頁面 | Classes | 狀態 | 內容 |
| --- | --- | --- | --- |
| [DPCRN](dpcrn.zh-TW.md) | `DPCRN` | active | 沿頻率軸的 U-Net CNN encoder/decoder，bottleneck 為 dual-path（intra-frequency / inter-time）recurrent 結構；voice-isolation 與 noise-suppression 的 backbone |
| [DPARN](dparn.zh-TW.md) | `DPARN` | active | 同一骨架，以 self-attention 取代 intra-frequency RNN |
| [Conv-TasNet](conv_tasnet.zh-TW.md) | `ConvTasNet` | library | 以堆疊的 dilated temporal convolution network 做時域分離 |
| [DPRNN](dprnn.zh-TW.md) | `DPRNN` | library | 在切塊序列上做 dual-path RNN |
| [SkiM](skim.zh-TW.md) | `SkiM` | library | 以 skipping-memory LSTM 串接各 segment LSTM |
| [TF-GridNet](tfgridnet.zh-TW.md) | `TFGridNet` | library | 在時頻格上做 intra-spectral、sub-band temporal 與 full-band attention |
| [U-Net](unet.zh-TW.md) | `Unet`、`UnetTcn`、`UnetFsmn` | library | 共用的卷積 encoder-decoder 骨架，可接 TCN 或 FSMN bottleneck |
| [ECAPA-TDNN](ecapa_tdnn.zh-TW.md) | `EcapaTdnnExtractor` | 凍結的 legacy | speaker-verification 與 TSE recipe 用的語者 embedding 抽取器 |

- **active**——有維護中的 recipe 在用。
- **library**——可 import、有 forward smoke test，但沒有維護中的 recipe 使用；
  作為模型庫的一部分保留。
- **凍結的 legacy**——為既有 recipe 保留，不再開發。

## 前端與 mask 套用

| 頁面 | Classes | 內容 |
| --- | --- | --- |
| [Feature layers](features.zh-TW.md) | `FeatureEncoder`、`MelBank`、`WeightedSum` | 把 encoder 輸出轉成 backbone 讀取的 feature |
| [Mask 套用](masker.zh-TW.md) | `Masker` | 把 backbone 的輸出（mask、filter taps 或 filter 統計量）套到混音的時頻表示上 |
| [Encoders 與 decoders](lobe/encoder.zh-TW.md) | `ConvEncDec`、`FreeEncDec`、`ConvSTFT`、`UnifiedConvEncDec` | 可學習 filterbank 與以卷積實作的 STFT analysis / synthesis |
| [Learnable DSP](lobe/dsp.zh-TW.md) | `FrequencyEQLayer` | 參數式 EQ，以 STFT 上逐 bin 的（可選擇可訓練）增益實作 |

## 積木（`puresound.nnet.lobe`）

| 頁面 | 內容 |
| --- | --- |
| [Activation functions](lobe/activation.zh-TW.md) | `get_activation`，依名稱查找 activation |
| [Attention](lobe/attention.zh-TW.md) | positional encoding、可加 mask 的 multi-head attention 與 Transformer encoder layer |
| [頻帶分組](lobe/banding.zh-TW.md) | ERB / mel 頻帶邊界、三角頻帶矩陣與 `BandBottleneck` |
| [CNN blocks](lobe/cnn.zh-TW.md) | depthwise-separable convolution 與 fast Fourier convolution |
| [Grouped operations](lobe/group_op.zh-TW.md) | transform-average-concatenate、grouped linear / GRU 層與 `SqueezedGRU` |
| [Auxiliary heads](lobe/heads.zh-TW.md) | 讀取 backbone bottleneck 的 VAD、距離、身分與 proximity head |
| [Metric discriminator](lobe/metric_discriminator.zh-TW.md) | metric-adversarial 微調用的可學習 PESQ 預測器 |
| [Multi-frame filters](lobe/multiframe.zh-TW.md) | deep filtering、multi-frame Wiener / MVDR filter 與 deep-filter residual head |
| [Normalization](lobe/norm.zh-TW.md) | global、channel-wise、instant 與 2-D layer norm，以及 `get_norm` |
| [Pooling](lobe/pooling.zh-TW.md) | attentive statistics pooling |
| [RNN blocks](lobe/rnn.zh-TW.md) | `SingleRNN`、FSMN 與 conditioned FSMN |
| [狀態空間區塊（Mamba / S6）](lobe/ssm.zh-TW.md) | `MambaInter`，用於 inter-time 路徑的 selective-scan 區塊 |
| [STFT helpers](lobe/stft.zh-TW.md) | Fourier kernel、overlap-add、mel filterbank |
| [Utility layers](lobe/trivial.zh-TW.md) | function wrapper、magnitude、`Gate` / `FiLM` conditioning、`SplitMerge` 切段、moving average、spectral compression、SpecAugment |

## Losses

見 [loss 函式庫](../losses/index.zh-TW.md)。
