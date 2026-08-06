# puresound.nnet.lobe.cnn

繁體中文版本：`cnn.zh-TW.md`

1D depthwise-separable convolution (Conv-TasNet-style TCN block), plus Fast
Fourier Convolution (FFC-SE) building blocks for 2D time-frequency features.

## Class: `DepthwiseSeparableConv1d`

Depthwise-separable 1D convolution with an optional dimension-changing input
projection, optional causal padding, and an optional skip connection.

```python
DepthwiseSeparableConv1d(
    in_channels: int,
    out_channels: int,
    hid_channels: Optional[int] = None,
    norm_cls: str = "gGN",
    kernel: int = 3,
    stride: int = 1,
    dilation: int = 1,
    skip: bool = False,
    causal: bool = False,
)
```

**Parameters:**
- `in_channels` – input channel count
- `out_channels` – output channel count — **can differ from `in_channels`**, it is not merely preserved
- `hid_channels` – if given, a `Conv1d(1×1) → norm → PReLU` first projects `in_channels → hid_channels` (a "dense transform") before the depthwise stage; if `None`, the depthwise stage runs directly on `in_channels`
- `norm_cls` – a short code string resolved through [`norm.get_norm`](norm.md) — e.g. `"gLN"`, `"gGN"`, `"cLN"`, `"iLN"`, `"bN1d"` — **not** long-form names like `"global_layer_norm"`
- `kernel` – depthwise kernel size
- `stride` – depthwise stride
- `dilation` – depthwise dilation
- `skip` – if `True`, adds a separate `Conv1d(in_channels, out_channels, 1)` projection of the raw input straight to the output, independent of the main depthwise/pointwise path
- `causal` – if `True`, left-pads the depthwise conv by `(kernel - 1) * dilation` and trims that many trailing frames back off afterward, so no future frame ever leaks in. Asserts `norm_cls not in ["gLN", "gGN"]`, since those normalize over the whole sequence and are not causal-safe

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – `[N, in_channels, T]`

**Returns:** `[N, out_channels, T']` (`T' == T` unless `stride > 1`).

Flow: optional `in_conv` (only if `hid_channels` given) → depthwise conv +
norm + PReLU → pointwise `1×1` conv + norm + PReLU → (if `causal`, trim the
trailing padding) → (if `skip`, add `skip_conv(x)`).

---

## Class: `SpectralTransform`

Part of `FFC` (below). Applies a real FFT along the **frequency axis** of a
`[N, CH, C, T]` feature map — "the same concept as cepstrum-space
transformation" per the source docstring — processes the real/imag parts with
a `1×1` conv in that transformed domain, then inverts the FFT and residual-adds
back onto the spatial-domain branch.

```python
SpectralTransform(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    causal: bool = True,
)
```

**Parameters:**
- `in_channels` / `out_channels` – channel counts
- `kernel_size` – `(kernel_f, kernel_t)`
- `stride` – `(stride_f, stride_t)`
- `causal` – left-only time padding if `True`; symmetric time padding if `False`. Frequency padding is always symmetric (`kernel_f // 2` both sides)

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, CH, C, T]` (`C` here is the **frequency axis**, not
a channel count; `CH` is the channel count)

Flow: `Conv2d + BN + ReLU` → `rfft` along the frequency axis (`dim=2`) → stack
real/imag parts onto the channel axis → `1×1 Conv2d + BN + ReLU` in FFT-space
→ un-stack, `irfft` back → residual-add onto the pre-FFT branch → `1×1 Conv2d`
output projection.

**Returns:** `[N, CH, out_channels, T]`.

No current caller in the repository — a library building block, not yet wired
into any backbone.

---

## Class: `FFC`

**Fast Fourier Convolution.** Splits channels into a "local" (ordinary
spatial conv) branch and a "global" (FFT, whole-frequency-axis receptive
field) branch, cross-feeds both directions, and concatenates the result.

```python
FFC(
    in_channels: int,
    out_channels: int,
    alpha: float = 0.3,
    kernel_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    causal: bool = True,
)
```

**Parameters:**
- `in_channels` / `out_channels` – total channel counts, split between the local/global branches
- `alpha` – global-branch channel fraction: `fft_in_ch = int(in_channels * alpha)`, `fft_out_ch = int(out_channels * alpha)`; the remainder on each side is the local branch (`local_in_ch`, `local_out_ch`)
- `kernel_size`, `stride` – as in `SpectralTransform`
- `causal` – as in `SpectralTransform`, applied identically to all four internal conv paths

**Reference:** "FFC-SE: Fast Fourier Convolution for Speech Enhancement".

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, CH, C, T]`, split along `CH` into
`global_in = x[:, :fft_in_ch]` and `local_in = x[:, fft_in_ch:]`.

Four conv paths combine into two branch outputs (variable names match source):

| Output | = | FFT/global path | + | cross/local path |
|---|---|---|---|---|
| `global_out` | = | `global_spec_trans(global_in)` (a `SpectralTransform`) | + | `local_global_conv(local_in)` (plain `Conv2d`, local→global) |
| `local_out` | = | `global_conv(global_in)` (plain `Conv2d`, global→local) | + | `local_local_conv(local_in)` (plain `Conv2d`, local→local) |

Each branch output then passes through its own `BatchNorm2d + ReLU`.

**Returns:** `torch.cat([local_out, global_out], dim=1)` → `[N, out_channels, C, T]` (local channels first, then global channels).

No current caller in the repository — a library building block, same status
as `SpectralTransform`.

## Wiring

`ConvTasNet`'s TCN block (`puresound/nnet/conv_tasnet.py`) wraps
`DepthwiseSeparableConv1d` with `hid_channels=None, skip=False` and a
config-driven `dconv_norm` (default `"gGN"`); the dimension-changing
projection and channel `in_conv`/`out_conv` are handled by `conv_tasnet.py`'s
own surrounding layers in that backbone, not by `hid_channels` here.

## Example

```python
from puresound.nnet.lobe.cnn import DepthwiseSeparableConv1d

conv = DepthwiseSeparableConv1d(
    in_channels=256,
    out_channels=256,
    kernel=3,
    dilation=4,
    causal=False,
    norm_cls="gLN",
)
out = conv(features)  # [N, 256, T]
```
