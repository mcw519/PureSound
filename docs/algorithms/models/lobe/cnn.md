# puresound.nnet.lobe.cnn

繁體中文版本：[cnn.zh-TW.md](cnn.zh-TW.md)

A 1-D depthwise-separable convolution (the Conv-TasNet TCN core) and the two
Fast Fourier Convolution blocks for `[N, CH, F, T]` time-frequency maps.

## Class: `DepthwiseSeparableConv1d`

```python
DepthwiseSeparableConv1d(
    in_channels: int,
    out_channels: int,
    hid_channels: Optional[int] = None,  # if set: 1x1 conv + norm + PReLU to this width first
    norm_cls: str = "gGN",               # name for norm.get_norm
    kernel: int = 3,
    stride: int = 1,
    dilation: int = 1,
    skip: bool = False,                  # add Conv1d(in_channels, out_channels, 1)(x)
    causal: bool = False,
)
```

`forward(x [N, in_channels, T]) -> [N, out_channels, T']`, with `T' = T` at
stride 1 and an odd kernel.

```
h = in_conv(x)                          if hid_channels, else x
h = PReLU(norm(depthwise_conv(h)))      # groups = width, kernel, stride, dilation
h = PReLU(norm(pointwise_conv(h)))      # 1x1, width -> out_channels
h = h[..., :-(kernel-1)*dilation]       if causal
h = h + skip_conv(x)                    if skip
```

- Padding is `(kernel - 1) * dilation` per side when causal, and the trailing
  frames are cut, so no output frame reads the future. Non-causal padding is
  `((kernel - 1) // 2) * dilation` per side.
- `causal=True` asserts `norm_cls` is not `gLN` or `gGN`, which normalise over
  the whole sequence. With `kernel == 1` the padding is zero and nothing is
  trimmed.
- `skip` assumes stride 1, since the skip path is not strided.

## Class: `SpectralTransform`

The global branch of an FFC: a real FFT along the frequency axis gives every
output position a view of the whole spectrum.

```python
SpectralTransform(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),  # (kernel_f, kernel_t)
    stride: Tuple[int, int] = (1, 1),       # (stride_f, stride_t)
    causal: bool = True,                    # time padding left-only; frequency padding is symmetric
)
```

`forward(x [N, in_channels, F, T]) -> [N, out_channels, F', T']`.

```
h = ReLU(BN(conv2d(pad(x))))
H = rfft(h, dim=F);  H = [Re H, Im H] stacked on channels
H = ReLU(BN(conv1x1(H)));  g = irfft(H, dim=F)
y = conv1x1(h + g)
```

`irfft` is asked for the frequency size `h` has after the first conv, so an odd
size round-trips like an even one.

## Class: `FFC`

Fast Fourier Convolution: channels are split into a local branch (ordinary
convolution) and a global branch (`SpectralTransform`), with cross paths in
both directions.

```python
FFC(
    in_channels: int,
    out_channels: int,
    alpha: float = 0.3,                     # global share: int(channels * alpha)
    kernel_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    causal: bool = True,
)
```

`forward(x [N, in_channels, F, T]) -> [N, out_channels, F', T']`, where the
input is split as `global_in = x[:, :int(in_channels * alpha)]`,
`local_in = the rest`, and

```
global_out = ReLU(BN(SpectralTransform(global_in) + conv(local_in)))
local_out  = ReLU(BN(conv(global_in) + conv(local_in)))
return cat([local_out, global_out], dim=1)
```

The output puts local channels first while the input split reads global
channels first, so stacking two `FFC`s sends the first layer's local
channels into the second layer's global branch.

Reference: Shchekotov et al., "FFC-SE: Fast Fourier Convolution for Speech
Enhancement", Interspeech 2022; Chi, Jiang, Mu, "Fast Fourier Convolution",
NeurIPS 2020.

## Use

`ConvTasNet`'s `TCN` layer wraps `DepthwiseSeparableConv1d` with
`hid_channels=None`, `skip=False` and `norm_cls=dconv_norm` (default
`"gGN"`); its own 1x1 input and output convs surround it. See
[Conv-TasNet](../conv_tasnet.md). `SpectralTransform` and `FFC` have no caller
in the backbones.

## Design notes

- Splitting a convolution into depthwise and pointwise parts costs
  `C * k + C * C_out` weights instead of `C * C_out * k`, which is what lets a
  TCN stack many dilated layers.
- The FFT over frequency in `SpectralTransform` gives a receptive field over the
  whole spectrum in one layer, which a stack of small frequency kernels only
  reaches after many layers.
