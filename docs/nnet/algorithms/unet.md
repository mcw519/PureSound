# puresound.nnet.unet

繁體中文版本：[unet.zh-TW.md](unet.zh-TW.md)

Status: *library* (see [nnet index](../index.md)) for direct use, but this
module is also the **chassis** [DPCRN](dpcrn.md) and [DPARN](dparn.md)
subclass — both active. All three classes here (`Unet`, `UnetTcn`,
`UnetFsmn`) are exported from `puresound/nnet/__init__.py` and therefore
config-reachable the way recipes resolve backbones
(`getattr(nnet, "UnetTcn")`), which
`test/test_backbone.py::test_backbone_reachable_from_config` asserts for each
of them; `UnetTcn` additionally has its own forward smoke test
(`test/test_backbone.py::test_unet_tcn_backbone`).

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
channels[-1]` channels. One parameter worth flagging:
- `input_type` is accepted but **never stored or read anywhere** in the
  class — passing anything here has zero effect.

> **Fixed in this pass:** `UnetTcn.forward` (and `UnetFsmn.forward`) used to
> skip straight from the input unsqueeze to the CNN-down loop, never calling
> the `self.input_norm` `iLN` that `Unet.__init__` builds — so the layer sat
> in checkpoints holding parameters that were never applied, unlike `Unet` /
> `DPCRN` / `DPARN`, which all call it. Both subclasses now apply it, matching
> the rest of the family. No config in this repo uses any Unet variant, so no
> checkpoint depended on the old behavior.

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

`input_norm` → CNN-down → flatten `(CH, C)` into one channel axis →
`repeat_tcn` stacks of `per_tcn_stack` TCN/GatedTCN blocks (conditioned on
`dvec` per `tcn_with_embed`) → unflatten back to `(CH, C)` → CNN-up
(skip-concat or `skip_conv`, `transpose_delay`-aware cropping).

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

`Unet` with the bottleneck replaced by a stack of `FSMN`/`ConditionFSMN`
layers (see [lobe/rnn](../lobe/rnn.md)) instead of TCN blocks — the same
chassis as `UnetTcn`, swapping the temporal-modeling core for FSMN's
feedforward memory blocks. Status *library*: exported and config-reachable,
but no recipe in this repo currently builds one.

```python
UnetFsmn(
    embed_dim: int = 0,
    embed_norm: bool = False,
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
    fsmn_l_context: int = 3,
    fsmn_r_context: int = 0,
    fsmn_dim: int = 256,
    num_fsmn: int = 8,
    fsmn_with_embed: List = [1, 1, 1, 1, 1, 1, 1, 1],
    fsmn_norm: str = "gLN",
    use_film: bool = True,
)
```

Everything from `input_dim` through `delay` behaves exactly as in `Unet`.
The FSMN-specific parameters:
- `num_fsmn` – how many FSMN blocks the bottleneck stacks; asserted equal to
  `len(fsmn_with_embed)`
- `fsmn_with_embed` – per-block flag; `1` builds a `ConditionFSMN` (takes
  `dvec`), `0` builds a plain `FSMN`
- `fsmn_l_context` / `fsmn_r_context` – left/right memory taps. `fsmn_r_context=0`
  (the default) keeps the block causal
- `fsmn_dim` – the FSMN projection width; input/output width is
  `temporal_input_dim`, derived like `UnetTcn`'s from the post-`stride_f`
  frequency resolution times `channels[-1]`
- `fsmn_norm` – norm type inside the FSMN blocks
- `use_film` – `ConditionFSMN` conditioning style: FiLM when `True`, otherwise
  the concat-based path (see [lobe/rnn](../lobe/rnn.md))

### `forward(x, dvec=None) -> Tensor`

```python
forward(x: Tensor, dvec: Optional[Tensor] = None) -> Tensor
# x:    [N, CH, C, T] or [N, C, T]
# dvec: [N, embed_dim], required only if any fsmn_with_embed[i] == 1
# returns: [N, CH, C, T]
```

Note the default `fsmn_with_embed` is all ones, so **a `dvec` is required
unless you override it** — this is a conditional (speaker-aware) backbone by
default, like `UnetTcn` with `embed_dim > 0`. Calling `forward(x)` with no
`dvec` fails inside `ConditionFSMN` rather than at the call site.

`input_norm` → CNN-down → flatten `(CH, C)` into one channel axis → the FSMN
stack, threading a `memory` tensor from block to block → unflatten back to
`(CH, C)` → CNN-up (skip-concat or `skip_conv`, `transpose_delay`-aware
cropping). As in `UnetTcn`, `transpose_delay=True` crops the *leading*
`transpose_t_size - 1` frames instead of the trailing ones.

### `get_args` property

Complete — every constructor argument is stored and returned.
