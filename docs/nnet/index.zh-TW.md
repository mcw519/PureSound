# 神經網路

English: [index.md](index.md)

`puresound.nnet` 包含模型 backbone、共用 layers、mask 操作與訓練 losses。

## Backbones

| 模型 | 狀態 | 用途 |
| --- | --- | --- |
| [DPCRN](algorithms/dpcrn.zh-TW.md) | active | 已發布的 voice-isolation 模型 |
| [DPARN](algorithms/dparn.zh-TW.md) | active | Attention-based enhancement |
| [ECAPA-TDNN](algorithms/ecapa_tdnn.zh-TW.md) | active | Speaker embeddings |
| [Conv-TasNet](algorithms/conv_tasnet.zh-TW.md) | library | 時域分離 |
| [DPRNN](algorithms/dprnn.zh-TW.md) | library | Dual-path recurrent separation |
| [SkiM](algorithms/skim.zh-TW.md) | library | Skipping-memory recurrent model |
| [TF-GridNet](algorithms/tfgridnet.zh-TW.md) | library | 時頻分離 |
| [U-Net variants](algorithms/unet.zh-TW.md) | library | 共用 encoder-decoder 模型 |

Active 模型由現行 recipe 使用。Library 模型仍可 import 並有 forward smoke test，
但不是預設 release 路徑。

## 元件

- [Feature layers](features.zh-TW.md)
- [Mask 操作](masker.zh-TW.md)
- [Loss functions](loss/index.zh-TW.md)
- [Activation functions](lobe/activation.zh-TW.md)
- [Attention](lobe/attention.zh-TW.md)
- [CNN blocks](lobe/cnn.zh-TW.md)
- [Learnable DSP](lobe/dsp.zh-TW.md)
- [Encoders 與 decoders](lobe/encoder.zh-TW.md)
- [Grouped operations](lobe/group_op.zh-TW.md)
- [Auxiliary heads](lobe/heads.zh-TW.md)
- [Multi-frame filters](lobe/multiframe.zh-TW.md)
- [Normalization](lobe/norm.zh-TW.md)
- [Pooling](lobe/pooling.zh-TW.md)
- [RNN blocks](lobe/rnn.zh-TW.md)
- [STFT layers](lobe/stft.zh-TW.md)
- [Utility layers](lobe/trivial.zh-TW.md)

Recipe 的模型名稱由 `puresound.nnet.__all__` 解析。YAML 使用新 class 前，必須先
將它加入該清單。
