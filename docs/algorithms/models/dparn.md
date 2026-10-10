# puresound.nnet.dparn

繁體中文版本：[dparn.zh-TW.md](dparn.zh-TW.md)

DPARN, the dual-path attention recurrent network: the [`Unet`](unet.md)
chassis of [DPCRN](dpcrn.md) with a bottleneck whose intra-frequency path is
two self-attention layers instead of a bidirectional LSTM. The inter-time path
stays a unidirectional LSTM. The same intra path is available inside DPCRN as
`intra_type: attention`; DPARN is the standalone model with a configurable
number of blocks, used by `egs/noise_suppression/config/dparn.yaml` and the
DPARN streaming export.

## Class: `DPARNblock2D`

```python
DPARNblock2D(
    input_size: int,       # bottleneck channels CH (channels[-1])
    hidden_size: int,      # attention feed-forward width and inter-LSTM hidden width
    nhead: int,            # attention heads; must divide input_size
    dropout: float = 0.0,  # shared by the attention layers and the LSTM
)
```

```python
forward(x: Tensor, intra_skip: bool = True, inter_skip: bool = True) -> Tensor
# x: [N, CH, C, T] -> [N, CH, C, T]
```

- **Intra (frequency, per frame):** reshape to `[N*T, C, CH]`; two
  `MhaSelfAttenLayer`s ([lobe/attention](lobe/attention.md)), the first with
  sinusoidal position encoding and the second without, each a Transformer
  encoder layer (self-attention and a ReLU feed-forward of width
  `hidden_size`, each with residual and LayerNorm); then `Linear` and
  `LayerNorm`; add the block input when `intra_skip`.
- **Inter (time, per frequency position):** reshape to `[N*C, T, CH]`; one
  unidirectional `SingleRNN("LSTM")` ([lobe/rnn](lobe/rnn.md)) and
  `LayerNorm`; add the intra output when `inter_skip`.

Attention is always called with `causal=False`, which is safe because it runs
across frequency within one frame. The layers are built with
`improved=False`, so `bidirectional=False` on them has no effect.

## Class: `DPARN`

### Constructor

```python
DPARN(
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 32, 32, 32, 64, 128),
    transpose_t_size: int = 2,
    transpose_delay: bool = False,
    skip_conv: bool = False,
    kernel_t: Tuple = (2, 2, 2, 2, 2),
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (5, 3, 3, 3, 3),
    stride_f: Tuple = (2, 2, 1, 1, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 0, 0, 0),
    n_dparn_block: int = 2,          # stacked DPARNblock2D at the bottleneck
    rnn_hidden: int = 128,           # hidden_size of every block
    nhead: int = 1,                  # nhead of every block
    spectral_compress: bool = False, # |X|^0.3 with phase kept, before input_norm
)
```

All arguments from `input_dim` to `delay` except `transpose_delay` are passed
to `Unet.__init__`; see [unet](unet.md). `transpose_delay` chooses which end of
the transpose convolution's extra `transpose_t_size - 1` frames is cropped;
`False` crops the end and keeps the decoder causal. `spectral_compress`
applies `spectral_compression(x, alpha=0.3, dim=1)`
([lobe/trivial](lobe/trivial.md)).

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, CH, C, T] or [N, C, T] (unsqueezed to [N, 1, C, T])
# returns the mask, [N, CH, C, T]
```

`spectral_compress` (optional), `input_norm`, the CNN down path (keeping each
output as a skip), `n_dparn_block` blocks, then the CNN up path with skip
concat or `skip_conv`, as in [unet](unet.md).

### `get_args`

Returns every constructor argument, so `DPARN(**model.get_args)` rebuilds the
same architecture.

### Config usage

```yaml
model:
  backbone:
    type: DPARN
    backbone_args:
      input_dim: 256
      norm_type: bN2d
      dropout: 0.1
      channels: [2, 32, 32, 32, 64, 128]
      kernel_t: [2, 2, 2, 2, 2]
      stride_t: [1, 1, 1, 1, 1]
      dilation_t: [1, 1, 1, 1, 1]
      kernel_f: [5, 3, 3, 3, 3]
      stride_f: [2, 2, 1, 1, 1]
      dilation_f: [1, 1, 1, 1, 1]
      delay: [1, 1, 1, 1, 1]
      n_dparn_block: 2
      rnn_hidden: 128
      nhead: 8
```

## Streaming

DPARN exports as a per-frame ONNX model. The wrapper keeps the offline model
unchanged and carries the frame state explicitly: temporal caches of the CNN
down path, pending outputs of the transpose convolutions, and the LSTM `(h, c)`
of each block. The attention path needs no state across frames.

```python
enhanced_frame, next_state = forward_frame(noisy_frame, state)
# noisy_frame, enhanced_frame: [batch, fft_length // 2 + 1, 2]
```

Audio buffering, the Hann STFT, the iSTFT and overlap-add run outside ONNX in
`puresound.streaming.StreamingDparnOrt`. See
[DPARN streaming ONNX runtime](../../usage/streaming/dparn_onnx.md) for the
supported config, export command and runtime API.

## Design notes

- Frequency has no causal constraint within a frame, so the intra path can
  attend over all positions at once; attention replaces the bidirectional
  LSTM's sequential steps with one matrix product per layer.
- Time is the only axis that carries state, and it stays a unidirectional LSTM,
  so the streaming state has the same layout as DPCRN's with
  `inter_type: lstm`.
