# puresound.nnet.lobe.trivial

繁體中文版本：[trivial.zh-TW.md](trivial.zh-TW.md)

Small utility layers: a function wrapper, complex-to-magnitude conversion, two
conditioning layers (`Gate`, `FiLM`), dual-path segmentation (`SplitMerge`), a
moving average, power-law spectral compression, and SpecAugment masking.

## Class: `LambdaLayer`

Wraps a function as an `nn.Module`, for parameter-free transforms inside an
`nn.Sequential`.

```python
LambdaLayer(lambda_func: LambdaType)
```

`forward(x, **kwargs)` returns `lambda_func(x, **kwargs)`. `FeatureEncoder`
uses it for axis permutes and squeezes.

## Class: `Magnitude`

```
mag = sqrt(re² + im² + 1e-8)        (then log1p(mag) if log1p)
```

```python
Magnitude(
    drop_first: bool = True,   # drop frequency bin 0 (DC)
    log1p: bool = False,       # return log1p(mag)
)
```

`forward(x)` accepts `[N, C, T, 2]` (re/im on the last axis) or `[N, 2C, T]`
(re/im halves on the channel axis); other ranks raise `TypeError`. Returns
`[N, C, T]`, or `[N, C - 1, T]` with `drop_first`. The `1e-8` keeps the
gradient finite at zero magnitude.

## Class: `Gate`

Residual gated conditioning: a content branch from `x` multiplied by a sigmoid
gate that sees both `x` and the conditioning vector.

```
h   = Conv1d_1x1(x)                                   # input_size -> hidden_size
g   = right_conv([h; broadcast(condition)])           # Conv1d -> ChanLN -> PReLU -> Dropout -> Sigmoid
out = x + Conv1d_1x1(left_conv(h) * g)                # left_conv: Conv1d -> ChanLN -> PReLU -> Dropout
```

```python
Gate(input_size: int, hidden_size: int, embed_size: int, dropout: float = 0.0)
```

`forward(x [N, input_size, T], condition [N, embed_size]) -> [N, input_size, T]`.
Used by `SkiM` for speaker conditioning.

## Class: `FiLM`

Feature-wise linear modulation (Perez et al., "FiLM: Visual Reasoning with a
General Conditioning Layer", AAAI 2018), with the scale and bias computed from
**both** the features and the conditioning vector:

```
x     = LayerNorm(x)                     if input_norm
c     = [x; broadcast(condition)]        # feats_size + embed_size channels
out   = Conv1d_1x1(c) * x + Conv1d_1x1'(c)
```

```python
FiLM(feats_size: int, embed_size: int, input_norm: bool = True)
```

`forward(x [N, feats_size, T], condition [N, embed_size]) -> [N, feats_size, T]`.
Used by `DPRNN`, `SkiM` and `DPCRN`'s `DPRNNblock2D` (when `dvec_dim` is set).
Computing the affine parameters from the features as well as the embedding
lets the modulation depend on the content at each frame, not only on who the
target is.

## Class: `SplitMerge`

Time-axis segmentation for dual-path models (DPRNN, SkiM): cut `[N, C, T]` into
50%-overlapping chunks of `seg_size` frames, and stitch them back by
overlap-averaging.

```python
SplitMerge(seg_size: int, seg_overlap: bool = True)
```

`split` and `merge` are static methods and are called on the class;
`seg_overlap` is stored but not read — the overlap is always 50%
(`seg_stride = seg_size // 2`).

- `SplitMerge.split(x [N, C, T], seg_size) -> (segments [N, S, K, C], rest)` —
  zero-pads the end by `rest` frames and both ends by `seg_stride`, then
  interleaves two half-shifted views. `K = seg_size`, `S` = number of segments.
- `SplitMerge.merge(x [N, S, K, C], rest) -> [N, C, T]` — averages the two
  overlapping halves and trims `rest`.

## Class: `MovingAverage1D`

Moving average over a one-value-per-frame track `[N, T]` (for example a gain or
VAD probability curve), via `nn.AvgPool1d`.

```python
MovingAverage1D(
    kernel_size: int,
    stride: int,
    add_padding: bool = False,   # zero-pad so the output keeps (about) T frames
    causal: bool = True,         # with padding: kernel_size - 1 zeros on the left;
                                 # otherwise kernel_size // 2 on each side
)
```

`forward(x [N, T]) -> [N, T']`.

## Function: `spectral_compression`

```python
spectral_compression(x: Tensor, alpha: float = 0.3, dim: int = 1, eps: float = 1e-8) -> Tensor
```

Power-law magnitude compression with the phase kept,
`|X|^alpha · exp(j·angle(X))`, on real/imag halves stacked along `dim`:

```python
_re, _im = torch.chunk(x, 2, dim=dim)
mag = (_re.pow(2) + _im.pow(2) + eps).sqrt()
scale = mag.pow(alpha - 1.0)
return torch.cat([_re * scale, _im * scale], dim=dim)
```

Returns a real tensor with the shape and dtype of `x`. `alpha = 1` is the
identity; `eps` keeps `mag ** (alpha - 1)` finite at the origin.

Scaling both parts by `|X|^(alpha-1)` is the same identity as rebuilding them
from `atan2`/`cos`/`sin`, without the trigonometric round trip, and it returns
exactly zero on silent bins (where `atan2(0, 0) = 0` would produce a spurious
real part). Keeping the stacked real layout lets real-valued Conv2d backbones
use it as a drop-in pre-processing step.

`DPCRN` and `DPARN` apply it to their input with `alpha = 0.3` when built with
`spectral_compress: True` (default `False`). The DPCRN streaming runner rejects
`spectral_compress=True`.

## Class: `SpecAugment`

Random time and frequency masking for training (Park et al., "SpecAugment: A
Simple Data Augmentation Method for Automatic Speech Recognition", Interspeech
2019).

```python
SpecAugment(
    freq_mask_length: int,   # max mask width in bins; width drawn from [0, length]
    time_mask_length: int,   # max mask width in frames
    fill_value: float,       # value written into masked cells; no default
    n_freq_mask: int = 1,
    n_time_mask: int = 1,
    prob: float = 0.5,       # per call and per axis
)
```

`forward(x [N, C, F, T])` masks only in training mode (`axis=2` frequency,
`axis=3` time, via `torchaudio.functional.mask_along_axis`); frequency and time
each draw their own `torch.rand(1) < prob`. In eval mode `x` is returned
unchanged. `FeatureEncoder` builds it from `specaug_args`.

## Example

```python
from puresound.nnet.lobe.trivial import FiLM, SpecAugment

film = FiLM(feats_size=256, embed_size=192)
x_cond = film(features, speaker_embedding)             # [N, 256, T]

spec_aug = SpecAugment(freq_mask_length=27, time_mask_length=100,
                       fill_value=0.0, n_freq_mask=2)
x_aug = spec_aug(mel_features)                         # [N, C, F, T]
```
