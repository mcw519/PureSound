# puresound.nnet.unet

繁體中文版本：[unet.zh-TW.md](unet.zh-TW.md)

Status: *library* (see [nnet index](../index.md)) for direct use, but this
module is also the **chassis** [DPCRN](dpcrn.md) and [DPARN](dparn.md)
subclass — both active — and `UnetTcn` is config-reachable
(`getattr(nnet, "UnetTcn")`) with its own forward smoke test
(`test/test_backbone.py::test_unet_tcn_backbone`). The file has three
classes; only `Unet` and `UnetTcn` are exported.

## Class: `Unet`

A 2D CNN encoder/decoder over `[frequency, time]` feature maps, with skip
connections between matching down/up stages. Every structural knob is a
per-layer tuple (one entry per down-conv stage), and frequency/time axes are
configured **independently** — there is no single `encoder_kernel: List[Tuple[int,int]]`.

### Constructor

```python
Unet(
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 1, 8, 8, 16, 16),   # length n_cnn+1; channels[i] -> channels[i+1] per stage
    transpose_t_size: int = 2,                 # up-path's ConvTranspose2d time-kernel (shared by every stage)
    skip_conv: bool = False,
    kernel_t: Tuple = (5, 1, 9, 1, 1),
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (1, 5, 1, 5, 1),
    stride_f: Tuple = (1, 4, 1, 4, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 1, 0, 0),            # per down-layer look-ahead frames
    multi_output: int = 1,                     # channel *multiplier* on the final up-layer, not a bool
)
```

**Parameters:**
- `input_dim` – input frequency-bin count
- `channels` – CNN channel progression; `len(channels) == n_cnn + 1` where
  `n_cnn = len(kernel_t)`. `channels[0]` is the input channel count fed to
  `input_norm`
- `kernel_t` / `stride_t` / `dilation_t` and `kernel_f` / `stride_f` /
  `dilation_f` – one entry per down-conv stage, all six tuples must be the
  same length (asserted). There is no combined `(freq, time)` pair in the
  public API; internally they're zipped into `self.kernel = list(zip(kernel_f,
  kernel_t))`
- `delay` – per-layer look-ahead frame count. `0` = fully causal for that
  layer (all padding placed on the past side); the down-conv's time padding
  is `((kernel_t[i]-1)*dilation_t[i] - delay[i], delay[i])` — as `delay[i]`
  grows, padding shifts from the causal (left) side to the look-ahead
  (right) side, so `delay[i]` **is** that layer's added latency in frames.
  Frequency padding is always symmetric (`kernel_f[i]//2 * dilation_f[i]` on
  both sides) — only time has a causality dimension.
- `transpose_t_size` – every up-conv stage's `ConvTranspose2d` uses this as
  its time-axis kernel size (not `kernel_t`, which only applies going down);
  a `ConvTranspose2d` with `stride=1` on time lengthens the sequence by
  `transpose_t_size - 1` frames, which `forward` crops back off
- `skip_conv` – `False` (default): concatenate the skip connection onto the
  up-path input (up-conv's input channels double, `channels[i+1] * 2`).
  `True`: project the skip connection through a `1x1 Conv2d + activation`
  (`self.skip_cnn`) and **add** it instead (up-conv's input channels stay
  `channels[i+1]`)
- `multi_output` – a channel-count **multiplier** applied only to the last
  (`i == 0`) up-conv layer's output channels (`channels[0] * multi_output`).
  Not a bool, and not "return a list of per-stage outputs" — `forward`
  always returns one `Tensor`; a `multi_output > 1` just widens that one
  tensor's final channel count (e.g. to pack multiple mask heads into a
  single conv). **None of this module's four subclasses** (`DPCRN`,
  `DPARN`, `UnetTcn`, `UnetFsmn`) forward this argument through their own
  `super().__init__()` call, so it is effectively fixed at `1` — a
  multi-output final layer is only reachable by instantiating `Unet`
  directly.

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, CH, C, T] or [N, C, T] (unsqueezed to [N, 1, C, T])
# returns: [N, channels[0] * multi_output, C, T]
```

`input_norm` (an `iLN` over the flattened `channels[0] * input_dim`, see
[lobe/norm](../lobe/norm.md)) → CNN-down (each stage appends its output to a
`skip` list) → CNN-up in reverse, concatenating or adding the matching skip
connection at each stage, cropping the trailing `transpose_t_size - 1`
frames the transpose-conv adds. There is no `transpose_delay` option at this
base-class level — the crop always removes the *trailing* frames (the
causal-safe direction); `transpose_delay` is a feature the subclasses below
add themselves by re-implementing `forward`.

### `shape_info()` and `get_args`

`shape_info() -> (down_shape, up_shape)` walks the configured strides to
report the frequency-bin count at every stage (a sanity-check helper for a
given `channels`/`stride_f` configuration, not used by `forward`).
`get_args` returns a complete `Dict` of all 15 constructor arguments — every
one of them, unlike [`DPARN.get_args`](dparn.md).

## Class: `UnetTcn`

`Unet` with the bottleneck replaced by a stack of [`TCN`/`GatedTCN`](conv_tasnet.md)
blocks operating on the flattened `(channel * frequency)` axis, instead of a
recurrent or attention bottleneck.

### Constructor

```python
UnetTcn(
    embed_dim: int = 0,
    embed_norm: bool = False,
    input_type: str = "RI",           # accepted but never read anywhere -- dead parameter
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 1, 8, 8, 16, 16),
    transpose_t_size: int = 2,
    transpose_delay: bool = False,
    skip_conv: bool = False,
    kernel_t: Tuple = (5, 1, 9, 1, 1),
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (1, 5, 1, 5, 1),
    stride_f: Tuple = (1, 4, 1, 4, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 1, 0, 0),
    tcn_layer: str = "normal",         # "normal" -> TCN, "gated" -> GatedTCN
    tcn_kernel: int = 3,
    tcn_dim: int = 256,
    tcn_dilated_basic: int = 2,
    per_tcn_stack: int = 5,
    repeat_tcn: int = 4,
    tcn_with_embed: List = [1, 0, 0, 0, 0],
    tcn_use_film: bool = False,
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",           # ignored when tcn_layer == "gated" (a warning is printed)
    causal: bool = False,
)
```

`input_dim` through `delay` are forwarded to `Unet.__init__` exactly as in
the base class (`multi_output` is not exposed — always `1`, see above);
`tcn_layer` through `causal` build the TCN stack exactly as in
[`ConvTasNet`](conv_tasnet.md) (same assert that `len(tcn_with_embed) ==
per_tcn_stack`, same dilation schedule), operating on
`temporal_input_dim = (input_dim after all stride_f downsampling) *
channels[-1]` channels. Two parameters worth flagging:
- `input_type` is accepted but **never stored or read anywhere** in the
  class — passing anything here has zero effect.
- **`UnetTcn.forward` never calls `self.input_norm`** — the `iLN` layer
  `Unet.__init__` builds is constructed and holds parameters, but this
  subclass's own `forward` override skips straight from the input
  unsqueeze to the CNN-down loop. `DPCRN` and `DPARN` (RNN-bottleneck
  subclasses) do call `input_norm`; `UnetTcn` and `UnetFsmn` (TCN/FSMN
  bottleneck subclasses) both do not. Whether that's intentional is not
  documented anywhere in source — treat `UnetTcn`'s (and `UnetFsmn`'s)
  input as effectively un-normalized.

`transpose_delay` (unlike base `Unet`) is a real, working option here:
`forward` crops the *leading* `transpose_t_size - 1` frames when `True`,
the trailing ones when `False` (default) — same convention as
[`DPCRN`](dpcrn.md)'s.

### `forward(x, dvec=None) -> Tensor`

```python
forward(x: Tensor, dvec: Optional[Tensor] = None) -> Tensor
# x:    [N, CH, C, T] or [N, C, T]
# dvec: [N, embed_dim], required only if any tcn_with_embed[i] == 1
# returns: [N, CH, C, T]
```

CNN-down → flatten `(CH, C)` into one channel axis → `repeat_tcn` stacks of
`per_tcn_stack` TCN/GatedTCN blocks (conditioned on `dvec` per
`tcn_with_embed`) → unflatten back to `(CH, C)` → CNN-up (skip-concat or
`skip_conv`, `transpose_delay`-aware cropping).

### `get_args` property

Complete for everything **except** `input_type` (never stored, so it
couldn't be recovered) and `multi_output` (not exposed by this subclass at
all, see `Unet` above).

### Example (mirrors `test/test_backbone.py::test_unet_tcn_backbone`)

```python
from puresound.nnet import UnetTcn

model = UnetTcn(
    embed_dim=192, embed_norm=True, input_dim=256,
    activation_type="PReLU", norm_type="gLN",
    channels=(2, 32, 64, 128, 128, 128, 128),
    transpose_t_size=2, transpose_delay=True, skip_conv=False,
    kernel_t=(2, 2, 2, 2, 2, 2), kernel_f=(5, 5, 5, 5, 5, 5),
    stride_t=(1, 1, 1, 1, 1, 1), stride_f=(2, 2, 2, 2, 2, 2),
    dilation_t=(1, 1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1, 1),
    delay=(0, 0, 0, 0, 0, 0),
    tcn_layer="gated", tcn_kernel=3, tcn_dim=256, tcn_dilated_basic=2,
    per_tcn_stack=5, repeat_tcn=3, tcn_with_embed=[1, 0, 0, 0, 0],
    tcn_norm="gLN", dconv_norm=None, causal=False,
)
y = model(torch.rand(1, 2, 256, 100), torch.rand(1, 192))  # [1, 2, 256, 100]
```

## Class: `UnetFsmn`

A third subclass exists in this file — `Unet` with the bottleneck replaced
by a stack of `FSMN`/`ConditionFSMN` layers (see [lobe/rnn](../lobe/rnn.md))
instead of TCN blocks. It is **currently unexported** —
`puresound/nnet/__init__.py` imports only `Unet` and `UnetTcn` from this
module, so `UnetFsmn` is not config-reachable (`getattr(nnet, "UnetFsmn")`
fails) and has no test coverage. A separate cleanup task is tracking
whether to export it or delete it; no further documentation is provided
here pending that decision.
