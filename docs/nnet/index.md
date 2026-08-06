# puresound.nnet

繁體中文版本：[index.zh-TW.md](index.zh-TW.md)

Neural network models and building blocks for speech enhancement, speaker verification, and target speaker extraction.

## Sub-modules

puresound.nnet is a **model library**: every backbone stays available and
config-reachable (`getattr(nnet, type)`) even when no current recipe uses it.
Status: *active* = used by a maintained recipe today; *library* = kept as a
selectable asset with a forward smoke test.
For *library* pages the prose describes the architecture; the authoritative
constructor signature is always the class docstring in the source file.

### Model Architectures

| Module | Status | Description |
|--------|--------|-------------|
| [algorithms/dpcrn](algorithms/dpcrn.md) | active | Dual-Path Convolutional RNN (DPCRN) — the released voice-isolate backbone |
| [algorithms/dparn](algorithms/dparn.md) | active | Dual-Path Attention RNN (DPARN) |
| [algorithms/ecapa_tdnn](algorithms/ecapa_tdnn.md) | active | ECAPA-TDNN speaker embedder |
| [algorithms/conv_tasnet](algorithms/conv_tasnet.md) | library | Conv-TasNet |
| [algorithms/dprnn](algorithms/dprnn.md) | library | Dual-Path RNN (DPRNN) |
| [algorithms/skim](algorithms/skim.md) | library | Skipping Memory LSTM (SkiM) |
| [algorithms/tfgridnet](algorithms/tfgridnet.md) | library | TF-GridNet |
| [algorithms/unet](algorithms/unet.md) | library | U-Net time-frequency models — `Unet` (also the DPCRN/DPARN chassis), `UnetTcn`, `UnetFsmn` |
| [nnet.features](features.md) | active | Feature processing layers |
| [nnet.masker](masker.md) | active | Mask application utilities |

### Building Blocks (`nnet.lobe`)

| Module | Description |
|--------|-------------|
| [lobe/activation](lobe/activation.md) | Activation function factory |
| [lobe/attention](lobe/attention.md) | Multi-head attention and positional encoding |
| [lobe/cnn](lobe/cnn.md) | CNN building blocks |
| [lobe/dsp](lobe/dsp.md) | Learnable DSP layers (parametric EQ) |
| [lobe/encoder](lobe/encoder.md) | Audio encoder/decoder modules |
| [lobe/group_op](lobe/group_op.md) | Group operations (TAC, grouped GRU) |
| [lobe/heads](lobe/heads.md) | Auxiliary bottleneck heads (frame VAD gate, distance/DRR regression) |
| [lobe/multiframe](lobe/multiframe.md) | Multi-frame speech enhancement filters |
| [lobe/norm](lobe/norm.md) | Normalization layers |
| [lobe/pooling](lobe/pooling.md) | Pooling layers for speaker embeddings |
| [lobe/rnn](lobe/rnn.md) | RNN variants (LSTM, GRU, FSMN) |
| [lobe/stft](lobe/stft.md) | STFT kernels and utilities |
| [lobe/trivial](lobe/trivial.md) | Utility layers (FiLM, Gate, SpecAugment, etc.) |

### Loss Functions (`nnet.loss`)

| Module | Description |
|--------|-------------|
| [loss/sdr](loss/sdr.md) | SDR-family losses (SI-SNR, t-SDR, SA-SDR) |
| [loss/stft_loss](loss/stft_loss.md) | STFT-domain losses (multi-resolution, spectral, over-suppression) |
| [loss/asr_feature](loss/asr_feature.md) | frozen-SSL feature matching (differentiable WER proxy) |
| [loss/residual](loss/residual.md) | residual (`noisy - enhanced`) supervision |
| [loss/dist](loss/dist.md) | distance/DRR regression for the DistHead auxiliary |
| [loss/vad](loss/vad.md) | frame-activity losses (VAD gate training) |
| [loss/spk](loss/spk.md) | speaker-embedding losses (AAM-softmax, SphereFace2, GE2E, Triplet) |

## Top-level Exports

The following classes are directly importable from `puresound.nnet` (this is
the complete `__all__` list in `puresound/nnet/__init__.py` — the config
loader's `getattr(nnet, backbone["type"])` can only resolve names on this
list, so a model missing here exists on disk but is not config-reachable):

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
