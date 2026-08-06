# puresound.nnet

English version: [index.md](index.md)

用於語音增強、聲紋辨識（speaker verification）與目標語者抽取（target speaker extraction）的神經網路模型與building blocks。

## Sub-modules

puresound.nnet 是一個**模型庫（model library）**：每個 backbone 即使目前沒有任何 recipe 使用，仍會保留在庫中且維持 config-reachable（可透過 `getattr(nnet, type)` 取得）。
Status 標籤:*active* = 目前有維護中的 recipe 在用;*library* = 保留作為可選用資產，並附一個 forward smoke test。
對於 *library* 頁面，內文描述的是架構;具權威性的 constructor 簽章一律以原始檔案中的 class docstring 為準。

### Model Architectures

| Module | Status | Description |
|--------|--------|-------------|
| [algorithms/dpcrn](algorithms/dpcrn.zh-TW.md) | active | Dual-Path Convolutional RNN (DPCRN) — 已發佈的 voice-isolate backbone |
| [algorithms/dparn](algorithms/dparn.zh-TW.md) | active | Dual-Path Attention RNN (DPARN) |
| [algorithms/ecapa_tdnn](algorithms/ecapa_tdnn.zh-TW.md) | active | ECAPA-TDNN speaker embedder |
| [algorithms/conv_tasnet](algorithms/conv_tasnet.zh-TW.md) | library | Conv-TasNet |
| [algorithms/dprnn](algorithms/dprnn.zh-TW.md) | library | Dual-Path RNN (DPRNN) |
| [algorithms/skim](algorithms/skim.zh-TW.md) | library | Skipping Memory LSTM (SkiM) |
| [algorithms/tfgridnet](algorithms/tfgridnet.zh-TW.md) | library | TF-GridNet |
| [algorithms/unet](algorithms/unet.zh-TW.md) | library | U-Net time-frequency models —— `Unet`（同時也是 DPCRN/DPARN 的 chassis）、`UnetTcn`、`UnetFsmn` |
| [nnet.features](features.zh-TW.md) | active | Feature processing layers |
| [nnet.masker](masker.zh-TW.md) | active | Mask application utilities |

### Building Blocks (`nnet.lobe`)

| Module | Description |
|--------|-------------|
| [lobe/activation](lobe/activation.md) | Activation function factory |
| [lobe/attention](lobe/attention.md) | Multi-head attention 與 positional encoding |
| [lobe/cnn](lobe/cnn.md) | CNN building blocks |
| [lobe/dsp](lobe/dsp.md) | 可訓練的 DSP layers（parametric EQ） |
| [lobe/encoder](lobe/encoder.md) | 音訊 encoder/decoder 模組 |
| [lobe/group_op](lobe/group_op.md) | Group operations（TAC、grouped GRU） |
| [lobe/heads](lobe/heads.md) | 輔助性的 bottleneck heads（frame VAD gate、distance/DRR regression） |
| [lobe/multiframe](lobe/multiframe.md) | Multi-frame 語音增強 filters |
| [lobe/norm](lobe/norm.md) | Normalization layers |
| [lobe/pooling](lobe/pooling.md) | 用於 speaker embeddings 的 pooling layers |
| [lobe/rnn](lobe/rnn.md) | RNN 變體（LSTM、GRU、FSMN） |
| [lobe/stft](lobe/stft.md) | STFT kernels 與工具函式 |
| [lobe/trivial](lobe/trivial.md) | 工具型 layers（FiLM、Gate、SpecAugment 等） |

### Loss Functions (`nnet.loss`)

| Module | Description |
|--------|-------------|
| [loss/sdr](loss/sdr.zh-TW.md) | SDR 系列 losses（SI-SNR、t-SDR、SA-SDR） |
| [loss/stft_loss](loss/stft_loss.zh-TW.md) | STFT-domain losses（multi-resolution、spectral、over-suppression） |
| [loss/asr_feature](loss/asr_feature.zh-TW.md) | frozen-SSL feature matching（可微分的 WER proxy） |
| [loss/residual](loss/residual.zh-TW.md) | residual（`noisy - enhanced`）監督訊號 |
| [loss/dist](loss/dist.zh-TW.md) | 給 DistHead 輔助任務用的 distance/DRR regression |
| [loss/vad](loss/vad.zh-TW.md) | frame-activity losses（VAD gate 訓練用） |
| [loss/spk](loss/spk.zh-TW.md) | speaker-embedding losses（AAM-softmax、SphereFace2、GE2E、Triplet） |

## Top-level Exports

以下 class 可直接從 `puresound.nnet` import（這就是 `puresound/nnet/__init__.py` 裡完整的
`__all__` 清單 —— config loader 的 `getattr(nnet, backbone["type"])` 只能解析這份清單上的名稱，
不在清單內的模型即使原始碼存在，也無法被任何 recipe config 指名使用）:

```python
from puresound.nnet import (
    ConvEncDec,
    ConvTasNet,
    DPARN,
    DPCRN,
    DPRNN,
    EcapaTdnnExtractor,
    FeatureEncoder,
    FreeEncDec,
    FrequencyEQLayer,
    SkiM,
    TFGridNet,
    Unet,
    UnetFsmn,
    UnetTcn,
)
```
