# Neural networks

Traditional Chinese: [index.zh-TW.md](index.zh-TW.md)

`puresound.nnet` contains model backbones, reusable layers, mask operations,
and training losses.

## Backbones

| Model | Status | Use |
| --- | --- | --- |
| [DPCRN](algorithms/dpcrn.md) | active | Released voice-isolation model |
| [DPARN](algorithms/dparn.md) | active | Attention-based enhancement |
| [ECAPA-TDNN](algorithms/ecapa_tdnn.md) | active | Speaker embeddings |
| [Conv-TasNet](algorithms/conv_tasnet.md) | library | Time-domain separation |
| [DPRNN](algorithms/dprnn.md) | library | Dual-path recurrent separation |
| [SkiM](algorithms/skim.md) | library | Skipping-memory recurrent model |
| [TF-GridNet](algorithms/tfgridnet.md) | library | Time-frequency separation |
| [U-Net variants](algorithms/unet.md) | library | Shared encoder-decoder models |

Active models are used by maintained recipes. Library models remain importable
and have forward smoke tests, but are not the default release path.

## Components

- [Feature layers](features.md)
- [Mask application](masker.md)
- [Loss functions](loss/index.md)
- [Activation functions](lobe/activation.md)
- [Attention](lobe/attention.md)
- [CNN blocks](lobe/cnn.md)
- [Learnable DSP](lobe/dsp.md)
- [Encoders and decoders](lobe/encoder.md)
- [Grouped operations](lobe/group_op.md)
- [Auxiliary heads](lobe/heads.md)
- [Multi-frame filters](lobe/multiframe.md)
- [Normalization](lobe/norm.md)
- [Pooling](lobe/pooling.md)
- [RNN blocks](lobe/rnn.md)
- [STFT layers](lobe/stft.md)
- [Utility layers](lobe/trivial.md)

Recipe model names are resolved from `puresound.nnet.__all__`. Add a class
there before referring to it from YAML.
