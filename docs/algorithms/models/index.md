# Neural networks

繁體中文版本：[index.zh-TW.md](index.zh-TW.md)

`puresound.nnet` holds the model backbones, the building blocks they are made
of (`puresound.nnet.lobe`), the feature and mask layers around them, and the
loss library (`puresound.nnet.loss`). A recipe names a model by class name under
`backbone.type`; the name is resolved with `getattr` on `puresound.nnet`, so a
class must be listed in `puresound.nnet.__all__` before a YAML file can refer
to it.

## Backbones

| Page | Classes | Status | What it is |
| --- | --- | --- | --- |
| [DPCRN](dpcrn.md) | `DPCRN` | active | U-Net CNN encoder/decoder over frequency with a dual-path (intra-frequency / inter-time) recurrent bottleneck; the voice-isolation and noise-suppression backbone |
| [DPARN](dparn.md) | `DPARN` | active | the same chassis with self-attention in place of the intra-frequency RNN |
| [Conv-TasNet](conv_tasnet.md) | `ConvTasNet` | library | time-domain separation with a stacked dilated temporal convolution network |
| [DPRNN](dprnn.md) | `DPRNN` | library | dual-path RNN over chunked sequences |
| [SkiM](skim.md) | `SkiM` | library | segment LSTMs linked by a skipping-memory LSTM |
| [TF-GridNet](tfgridnet.md) | `TFGridNet` | library | intra-spectral, sub-band temporal and full-band attention blocks on the time-frequency grid |
| [U-Net](unet.md) | `Unet`, `UnetTcn`, `UnetFsmn` | library | the shared convolutional encoder-decoder chassis, with TCN or FSMN bottlenecks |
| [ECAPA-TDNN](ecapa_tdnn.md) | `EcapaTdnnExtractor` | frozen legacy | speaker-embedding extractor for the speaker-verification and TSE recipes |

- **active** — used by a maintained recipe.
- **library** — importable and covered by forward smoke tests, not used by a
  maintained recipe; kept as part of the model library.
- **frozen legacy** — kept for existing recipes, not developed further.

## Front ends and mask application

| Page | Classes | What it is |
| --- | --- | --- |
| [Feature layers](features.md) | `FeatureEncoder`, `MelBank`, `WeightedSum` | turns an encoder output into the features a backbone reads |
| [Mask application](masker.md) | `Masker` | applies a backbone's output (a mask, filter taps or filter statistics) to the mixture's time-frequency representation |
| [Encoders and decoders](lobe/encoder.md) | `ConvEncDec`, `FreeEncDec`, `ConvSTFT`, `UnifiedConvEncDec` | learned filterbank and convolutional STFT analysis / synthesis |
| [Learnable DSP](lobe/dsp.md) | `FrequencyEQLayer` | parametric EQ as an (optionally trainable) per-bin gain on an STFT |

## Building blocks (`puresound.nnet.lobe`)

| Page | What it holds |
| --- | --- |
| [Activation functions](lobe/activation.md) | `get_activation`, activation lookup by name |
| [Attention](lobe/attention.md) | positional encoding, masked multi-head attention and a Transformer encoder layer |
| [Frequency banding](lobe/banding.md) | ERB / mel band edges, triangular band matrices and `BandBottleneck` |
| [CNN blocks](lobe/cnn.md) | depthwise-separable convolution and fast Fourier convolution |
| [Grouped operations](lobe/group_op.md) | transform-average-concatenate, grouped linear / GRU layers and `SqueezedGRU` |
| [Auxiliary heads](lobe/heads.md) | VAD, distance, identity and proximity heads that read a backbone's bottleneck |
| [Metric discriminator](lobe/metric_discriminator.md) | learned PESQ predictor for metric-adversarial fine-tuning |
| [Multi-frame filters](lobe/multiframe.md) | deep filtering, multi-frame Wiener / MVDR filters and the deep-filter residual head |
| [Normalization](lobe/norm.md) | global, channel-wise, instant and 2-D layer norms and `get_norm` |
| [Pooling](lobe/pooling.md) | attentive statistics pooling |
| [RNN blocks](lobe/rnn.md) | `SingleRNN`, FSMN and conditioned FSMN |
| [State-space blocks (Mamba / S6)](lobe/ssm.md) | `MambaInter`, a selective-scan block for the inter-time path |
| [STFT helpers](lobe/stft.md) | Fourier kernels, overlap-add, mel filterbanks |
| [Utility layers](lobe/trivial.md) | function wrapper, magnitude, `Gate` / `FiLM` conditioning, `SplitMerge` segmentation, moving average, spectral compression, SpecAugment |

## Losses

See the [loss library](../losses/index.md).
