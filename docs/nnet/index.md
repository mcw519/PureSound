# puresound.nnet

Neural network models and building blocks for speech enhancement, speaker verification, and target speaker extraction.

## Sub-modules

### Model Architectures

| Module | Description |
|--------|-------------|
| [nnet.base_nn](base_nn.md) | Base model classes for encoder-decoder pipelines |
| [nnet.features](features.md) | Feature processing layers (Mel-filterbank, weighted sum) |
| [nnet.masker](masker.md) | Mask application utilities |
| [algorithms/conv_tasnet](algorithms/conv_tasnet.md) | Conv-TasNet temporal convolution blocks |
| [algorithms/dparn](algorithms/dparn.md) | Dual-Path Attention RNN (DPARN) |
| [algorithms/dpcrn](algorithms/dpcrn.md) | Dual-Path Conditional RNN (DPCRN) |
| [algorithms/dprnn](algorithms/dprnn.md) | Dual-Path RNN (DPRNN) |
| [algorithms/ecapa_tdnn](algorithms/ecapa_tdnn.md) | ECAPA-TDNN speaker embedder |
| [algorithms/skim](algorithms/skim.md) | Speech-aware Memory LSTM (SkiM) |
| [algorithms/tfgridnet](algorithms/tfgridnet.md) | TF-GridNet context module |
| [algorithms/unet](algorithms/unet.md) | U-Net time-frequency model |

### Building Blocks (`nnet.lobe`)

| Module | Description |
|--------|-------------|
| [lobe/activation](lobe/activation.md) | Activation function factory |
| [lobe/attention](lobe/attention.md) | Multi-head attention and positional encoding |
| [lobe/cnn](lobe/cnn.md) | CNN building blocks |
| [lobe/dsp](lobe/dsp.md) | Learnable DSP layers (parametric EQ) |
| [lobe/encoder](lobe/encoder.md) | Audio encoder/decoder modules |
| [lobe/group_op](lobe/group_op.md) | Group operations (TAC, grouped GRU) |
| [lobe/multiframe](lobe/multiframe.md) | Multi-frame speech enhancement filters |
| [lobe/norm](lobe/norm.md) | Normalization layers |
| [lobe/pooling](lobe/pooling.md) | Pooling layers for speaker embeddings |
| [lobe/rnn](lobe/rnn.md) | RNN variants (LSTM, GRU, FSMN) |
| [lobe/stft](lobe/stft.md) | STFT kernels and utilities |
| [lobe/trivial](lobe/trivial.md) | Utility layers (FiLM, Gate, SpecAugment, etc.) |

### Loss Functions (`nnet.loss`)

| Module | Description |
|--------|-------------|
| [loss/metrics](loss/metrics.md) | GE2E and time-domain losses |
| [loss/sdr](loss/sdr.md) | SDR-family losses (SI-SNR, t-SDR, SA-SDR) |
| [loss/spk](loss/spk.md) | Speaker classification losses (AAM-softmax, SphereFace2) |
| [loss/stft_loss](loss/stft_loss.md) | STFT-domain losses (multi-resolution, spectral) |
| [loss/vad](loss/vad.md) | Voice activity detection loss |

## Top-level Exports

The following classes are directly importable from `puresound.nnet`:

```python
from puresound.nnet import (
    DPARN,
    DPCRN,
    EcapaTdnnExtractor,
    FeatureEncoder,
    FrequencyEQLayer,
    ConvEncDec,
    FreeEncDec,
    SkiM,
    TFGridNet,
)
```
