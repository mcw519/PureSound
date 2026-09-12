# puresound.nnet.lobe.trivial

繁體中文版本：`trivial.zh-TW.md`

Small utility layers: a function-as-module wrapper, complex→magnitude
conversion, two different conditioning mechanisms (`Gate`, `FiLM`), chunked
segmentation/merge for dual-path models, a scalar moving average, spectral
power-law compression, and SpecAugment-style masking.

## Class: `LambdaLayer`

Wraps an arbitrary function as an `nn.Module`, for dropping non-parameterized
transforms into an `nn.Sequential`.

```python
LambdaLayer(lambda_func: LambdaType)
```

Note the constructor argument is named `lambda_func`, not `fn`.

### `forward(x: Tensor, **kwargs) -> Any`

Returns `lambda_func(x, **kwargs)` — extra keyword arguments pass straight
through, so `lambda_func` can itself take more than one argument.

---

## Class: `Magnitude`

Converts a complex spectrum to a magnitude spectrum. Accepts **either** of
two input layouts:

```python
Magnitude(drop_first: bool = True, log1p: bool = False)
```

**Parameters:**
- `drop_first` – if `True`, drops the first frequency bin (DC) from the output
- `log1p` – if `True`, applies `log1p` to the magnitude (log-magnitude compression)

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x`, either:
- `[N, C, T, 2]` (4D — real/imag stacked on the last axis, e.g. straight off [`encoder.ConvSTFT`](encoder.md)), or
- `[N, 2*C, T]` (3D — real/imag concatenated on the channel axis, split via `torch.chunk(x, 2, dim=1)`)

Any other rank raises `TypeError`. Computes
`sqrt(re**2 + im**2 + 1e-8)` (the `1e-8` keeps the gradient finite at zero
magnitude), then optionally drops the first bin and/or applies `log1p`.

**Returns:** `[N, C(-1), T]` (frequency-bin count reduced by 1 if `drop_first`).

---

## Class: `Gate`

A GLU-style gate: a content branch (from `x` alone) multiplied elementwise by
a sigmoid gate branch that sees **both** `x` and the conditioning vector.

```python
Gate(input_size: int, hidden_size: int, embed_size: int, dropout: float = 0.0)
```

**Parameters:**
- `input_size` – feature dimension of `x` (also the output dimension — the block is residual/shape-preserving)
- `hidden_size` – internal projection width
- `embed_size` – conditioning embedding dimension
- `dropout` – applied inside both the content and gate branches

### `forward(x, condition) -> Tensor`

```
res  = x
h    = in_conv(x)                                # Conv1d 1x1, input_size -> hidden_size
cond = broadcast(condition) over time
h_c  = concat([h, cond], channel axis)            # hidden_size + embed_size
out  = left_conv(h) * right_conv(h_c)             # left: no direct view of `condition`
out  = out_conv(out)                              # hidden_size -> input_size
return out + res
```

`left_conv` is `Conv1d → ChanLN → PReLU → Dropout` (content only); `right_conv`
is `Conv1d → ChanLN → PReLU → Dropout → Sigmoid` (the actual gate, conditioned
on both `x` and `condition`).

**Parameters:**
- `x` – `[N, input_size, T]`
- `condition` – `[N, embed_size]`

**Returns:** `[N, input_size, T]`.

---

## Class: `FiLM`

**Feature-wise Linear Modulation.** Unlike "textbook" FiLM (where the
scale/bias come from the conditioning vector alone), here `cond_scale` /
`cond_bias` are computed from the **concatenation** of `x` and the broadcast
conditioning vector — so the affine parameters are content-aware, not purely
a function of `condition`.

```python
FiLM(feats_size: int, embed_size: int, input_norm: bool = True)
```

**Parameters:**
- `feats_size` – feature dimension of `x`
- `embed_size` – conditioning embedding dimension
- `input_norm` – if `True` (default), applies `nn.LayerNorm(feats_size)` to `x` (over the channel axis) before computing the modulation

**Reference:** Perez et al., "FiLM: Visual Reasoning with a General
Conditioning Layer," AAAI 2018.

### `forward(x, condition) -> Tensor`

```
x        = LayerNorm(x)  if input_norm else x
cond_cat = concat([x, broadcast(condition)], channel axis)
scale    = cond_scale(cond_cat)     # Conv1d 1x1, feats_size+embed_size -> feats_size
bias     = cond_bias(cond_cat)      # Conv1d 1x1, feats_size+embed_size -> feats_size
return scale * x + bias
```

**Parameters:**
- `x` – `[N, feats_size, T]`
- `condition` – `[N, embed_size]`

**Returns:** `[N, feats_size, T]`.

---

## Class: `SplitMerge`

**Not a channel split.** This is the DPRNN/SkiM-style **time-axis**,
50%-overlapping chunking scheme: cut a long sequence into fixed-size,
half-overlapping segments for dual-path (intra-chunk / inter-chunk)
processing, and stitch them back together later by overlap-averaging.

```python
SplitMerge(seg_size: int, seg_overlap: bool = True)
```

**Parameters:**
- `seg_size` – chunk length in frames
- `seg_overlap` – accepted and stored on the instance, but currently has **no effect**: `split`/`merge` are `@staticmethod`s that never read `self` (or `self.seg_overlap`) at all, and always use a hardcoded 50% stride (`seg_stride = seg_size // 2`). Every current caller (`test/test_lobe.py`, `puresound/nnet/dprnn.py`) calls `SplitMerge.split(...)` / `SplitMerge.merge(...)` directly on the class, without ever constructing an instance — so the constructor and `seg_overlap` are exercised by nothing today.

> **Fixed in this pass:** `__init__` previously never called `super().__init__()`,
> leaving the instance without `nn.Module`'s internal `_parameters`/`_buffers`/
> `_modules` bookkeeping — harmless for plain-int attribute storage, but any
> `.to(device)`, `.state_dict()`, `repr()`, or module-tree traversal
> (`.parameters()`, etc.) on an instance would have raised `AttributeError`.
> Since `__init__` only ever stores plain `int`/`bool` attributes (no
> `nn.Parameter`/submodule), adding `super().__init__()` is a pure hygiene fix
> with no behavior change for the static-method call pattern actually in use.

### `split(x: Tensor, seg_size: int) -> Tuple[Tensor, int]` (staticmethod)

**Parameters:** `x` – `[N, C, T]`

Zero-pads `T` so it tiles evenly into `seg_size`-wide, 50%-overlapping
segments, then assembles them from two half-segment-shifted views (the
standard DPRNN segmentation trick).

**Returns:** `(segments, rest)` — `segments`: `[N, S, K, C]` (`S` = number of
segments, `K = seg_size`); `rest` = padding length added, needed by `merge`.

### `merge(x: Tensor, rest: int) -> Tensor` (staticmethod)

**Parameters:** `x` – `[N, S, K, C]`, `rest` — from the matching `split` call

Reverses the segmentation by overlap-averaging the two halves back together
and trimming off `rest`. **Returns:** `[N, C, T]`.

---

## Class: `MovingAverage1D`

A simple moving average over a **scalar-per-timestep** signal (not
multi-channel feature maps) — e.g. smoothing a gain curve or VAD probability
track.

```python
MovingAverage1D(
    kernel_size: int,
    stride: int,
    add_padding: bool = False,
    causal: bool = True,
)
```

**Parameters:**
- `kernel_size` / `stride` – forwarded to the internal `nn.AvgPool1d`
- `add_padding` – if `True`, zero-pads before pooling so the output length is preserved (approximately)
- `causal` – when `add_padding=True`, pad only on the left (`kernel_size - 1` zeros) if `True`; otherwise pad symmetrically (`kernel_size // 2` each side)

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, T]`

**Returns:** `[N, T']` (`T' = T` if `add_padding=True`, shorter otherwise).

---

## Function: `spectral_compression`

```python
spectral_compression(x: Tensor, alpha: float = 0.3, dim: int = 1, eps: float = 1e-8) -> Tensor
```

Power-law magnitude compression with phase preserved, i.e.
`|X|**alpha * exp(j*angle(X))`, computed and returned in the stacked real/imag
layout the real-valued backbones use:

```python
_re, _im = torch.chunk(x, 2, dim=dim)
mag = (_re.pow(2) + _im.pow(2) + eps).sqrt()
scale = mag.pow(alpha - 1.0)
return torch.cat([_re * scale, _im * scale], dim=dim)
```

Phase is preserved by scaling both parts by `|X|**(alpha-1)` rather than
rebuilding them via `atan2`/`cos`/`sin`. The two are the same identity —
`cos(angle) == re/|X|` — but the direct form is roughly 2x faster (no
transcendentals), avoids a round trip through angle space, and is correct on
silent bins: `atan2(0, 0)` is `0`, so the trigonometric form emits a spurious
`|X|**alpha` real part for every all-zero bin, which the direct form returns
as exactly zero.

Returning stacked real/imag rather than a `torch.complex64` tensor is what
keeps it usable by the Conv2d backbones that call it — the output has the same
shape *and* the same real dtype as the input, so it can be dropped in as a
pre-processing step without changing anything downstream.

**Parameters:**
- `x` – real/imag stacked along `dim` (e.g. `[N, 2*C, T, ...]` if `dim=1`)
- `alpha` – compression exponent. `alpha=1.0` is the identity (magnitude is
  raised to the first power and the original phase is restored exactly)
- `dim` – axis the real/imag halves are stacked on
- `eps` – magnitude floor, keeping `mag.pow(alpha - 1.0)` finite at the origin

**Returns:** a real tensor with the same shape and dtype as `x`. Its magnitude
follows `|X|**alpha` and its phase is unchanged.

> **Still gated off everywhere.** `DPARNblock2D` / `DPRNNblock2D` both gate
> this call behind a `spectral_compress: bool = False` constructor flag, and
> **every** recipe config in `egs/` that sets it does so explicitly as
> `spectral_compress: False`. No trained or deployed model in this repo goes
> through this path.

---

## Class: `SpecAugment`

Random time/frequency masking for training-time spectrogram augmentation.

```python
SpecAugment(
    freq_mask_length: int,
    time_mask_length: int,
    fill_value: float,
    n_freq_mask: int = 1,
    n_time_mask: int = 1,
    prob: float = 0.5,
)
```

**Parameters:**
- `freq_mask_length` / `time_mask_length` – **required**; maximum mask width along each axis (actual width per mask is uniformly sampled from `[0, length]`)
- `fill_value` – **required**; value written into masked bins/frames — there is no default, callers must pick one explicitly (e.g. the feature's mean or `0.0`)
- `n_freq_mask` / `n_time_mask` – how many independent masks to apply per axis
- `prob` – **per-call, per-axis** gate — masking along an axis fires only if `torch.rand(1) < prob`; frequency and time each roll their own independent draw, so on a given call you may get neither, either, or both

**Reference:** Park et al., "SpecAugment: A Simple Data Augmentation Method
for ASR," Interspeech 2019.

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, C, F, T]` (masking uses `torchaudio.functional.mask_along_axis`
with `axis=2` for frequency and `axis=3` for time — **4D**, not the `[N, F, T]`
3D shape a naive reading might suggest)

Only masks while `self.training` is `True`; in `.eval()` mode, returns `x`
unchanged.

**Returns:** masked (or unchanged) tensor, same shape as input.

## Wiring

`puresound/nnet/features.py`'s `FeatureEncoder` uses `LambdaLayer` (for ad
hoc permutes), `Magnitude`, and `SpecAugment` (constructed from a
`specaug_args` config dict). `FiLM`/`SplitMerge` drive `DPRNN`'s dual-path
chunking (`puresound/nnet/dprnn.py`); `FiLM`/`Gate` drive `SkiM`'s
conditioning (`puresound/nnet/skim.py`); `spectral_compression` is wired into
`DPARN`/`DPCRN` but switched off in every config (see the gating note above).

## Example

```python
from puresound.nnet.lobe.trivial import FiLM, SpecAugment, LambdaLayer

film = FiLM(feats_size=256, embed_size=192)
x_cond = film(features, speaker_embedding)

spec_aug = SpecAugment(freq_mask_length=27, time_mask_length=100, fill_value=0.0, n_freq_mask=2)
x_aug = spec_aug(mel_features)  # [N, C, F, T], masked only during training
```
