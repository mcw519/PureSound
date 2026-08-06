# puresound.nnet.conv_tasnet

繁體中文版本：[conv_tasnet.zh-TW.md](conv_tasnet.zh-TW.md)

Status: *library* (see [nnet index](../index.md)) — config-reachable via
`getattr(nnet, "ConvTasNet")`, covered by a forward smoke test
(`test/test_backbone.py::test_conv_tasnet_backbone` /
`test_conv_tasnet_dvec_backbone`), not used by a maintained recipe today.

**Reference:** Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal
Time–Frequency Magnitude Masking for Speech Separation," IEEE/ACM TASLP,
2019.

This module holds only the temporal-convolution mask estimator — **not** the
learned waveform encoder/decoder from the original paper. Like every other
backbone in this library it is a pluggable middle stage in
`Encoder -> Features -> Backbone -> Masker -> Decoder`
(see [nnet.features](../features.md) / [nnet.masker](../masker.md)); pick any
front end (`ConvEncDec`, `FreeEncDec`, ...) and let this module predict the
mask from whatever features that front end produces.

## Class: `TCN`

A single non-gated temporal convolution block: bottleneck-project, depthwise
dilated conv, project back out, residual-add.

### Constructor

```python
TCN(
    in_channels: int,
    hid_channels: int,
    kernel: int,
    dilation: int,
    dropout: float = 0.0,
    emb_dim: int = 0,
    causal: bool = False,
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",
)
```

**Parameters:**
- `in_channels` – input/output feature dimension (residual add requires them equal)
- `hid_channels` – bottleneck width inside the block
- `kernel` / `dilation` – depthwise conv kernel size and dilation rate
- `dropout` – dropout after the depthwise conv (`0.0` disables it)
- `emb_dim` – if not zero, `forward`'s `embed` argument is concatenated onto
  `x` **before** the input `1x1 Conv` (`in_conv`); `TCN` has no `right_conv`
  (that submodule only exists on `GatedTCN`, see below)
- `causal` – forwarded to the inner `DepthwiseSeparableConv1d`
  ([lobe/cnn](../lobe/cnn.md)): left-only padding when `True`, centered
  padding to preserve length when `False`
- `tcn_norm` – norm layer around `in_conv` (via [`get_norm`](../lobe/norm.md))
- `dconv_norm` – norm layer used *inside* the depthwise separable conv

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x:     [N, in_channels, T]
# embed: [N, emb_dim], broadcast over T and concatenated onto x
# returns: [N, in_channels, T] -- always a single Tensor (no skip-path tuple)
```

`in_conv -> dconv -> out_conv`, then `+ x` (residual). There is no
`skip_connection` option and no tuple return — every call, conditioned or
not, returns one residual-summed tensor.

## Class: `GatedTCN`

A gated variant: two parallel branches (`left_conv`, `right_conv`) multiply
together instead of a single conv path, and conditioning can optionally be
injected as FiLM instead of concatenation.

### Constructor

```python
GatedTCN(
    in_channels: int,
    hid_channels: int,
    kernel: int,
    dilation: int,
    dropout: float = 0.0,
    emb_dim: int = 0,
    causal: bool = False,
    tcn_norm: str = "gLN",
    use_film: bool = False,
)
```

**Parameters:** same as `TCN` plus `use_film` — **off by default**. FiLM
conditioning is not this block's headline feature, just one of two wiring
modes for `embed`:
- `use_film=False` (default): `embed` is broadcast over time and
  concatenated onto `right_conv`'s input (`right_conv`'s in-channels are
  `hid_channels + emb_dim`) — this is the one case where the copy-pasted
  "concate in right_conv's input" docstring line is actually accurate.
- `use_film=True`: `embed` instead drives two `1x1 Conv`s
  (`cond_scale`, `cond_bias`) that FiLM-modulate `left_conv`'s output
  (`right_conv`'s in-channels stay `hid_channels`, no concatenation).

Unlike `TCN`, `causal=True` here also crops the output
(`x[..., :-self.padd]`) rather than relying on symmetric padding, since the
left/right convs pad with `(kernel-1)*dilation` on the causal side.

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x: [N, in_channels, T] -> returns [N, in_channels, T]
```

`x = left_conv(in_conv(x)) * right_conv(x_r)` (`x_r` = concat- or
FiLM-conditioned), then `out_conv`, then residual-add (cropped if causal).

## Class: `ConvTasNet`

The exported backbone (`from puresound.nnet import ConvTasNet`) — a stack of
`repeat_tcn` groups of `per_tcn_stack` `TCN`/`GatedTCN` blocks with
exponentially increasing dilation, optionally conditioned on a speaker
embedding at chosen layers (target-speaker extraction).

### Constructor

```python
ConvTasNet(
    input_dim: int = 512,
    embed_dim: int = 256,
    embed_norm: bool = False,
    tcn_layer: str = "normal",       # "normal" -> TCN, "gated" -> GatedTCN
    tcn_kernel: int = 3,
    tcn_dim: int = 256,
    tcn_dilated_basic: int = 2,
    per_tcn_stack: int = 5,          # blocks per stack; dilation = tcn_dilated_basic ** i
    repeat_tcn: int = 4,             # number of stacks
    tcn_with_embed: List = [1, 0, 0, 0, 0],  # len must == per_tcn_stack
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",         # ignored when tcn_layer == "gated"
    causal: bool = False,
)
```

**Parameters:**
- `input_dim` – feature (channel) dimension in and out; unchanged throughout
  the stack (every `TCN`/`GatedTCN` is shape-preserving)
- `embed_dim` – speaker-embedding dimension for target-speaker conditioning;
  irrelevant if no `tcn_with_embed` entry is `1`
- `embed_norm` – L2-normalize `dvec` before conditioning
- `tcn_dilated_basic` – dilation base; block `i` in a stack gets dilation
  `tcn_dilated_basic ** i`, so `per_tcn_stack=5` with the default base `2`
  covers dilations `1, 2, 4, 8, 16` before the next stack restarts at `1`
- `tcn_with_embed` – per-block flag (length `per_tcn_stack`, asserted) marking
  which blocks in *every* stack receive `dvec`
- `dconv_norm` – only meaningful for `tcn_layer="normal"`; `GatedTCN` has no
  such knob (`UnetTcn`, which reuses this same block, prints a warning if you
  set it while `tcn_layer="gated"`)

### `forward(x, dvec=None) -> Tensor`

```python
forward(x: Tensor, dvec: Optional[Tensor] = None) -> Tensor
# x:    [N, input_dim, T]
# dvec: [N, embed_dim], required only if any tcn_with_embed[i] == 1
# returns: [N, input_dim, T] -- a mask in the feature domain (Masker applies it)
```

### `get_args` property

Returns every constructor argument as a `Dict` — the standard
checkpoint-reconstruction pattern used across this library (compare
[`DPCRN.get_args`](dpcrn.md), [`Unet.get_args`](unet.md)).

### Example (mirrors `test/test_backbone.py`)

```python
from puresound.nnet import ConvTasNet

# Plain separation/enhancement, no conditioning
model = ConvTasNet(
    input_dim=512, embed_dim=0, embed_norm=True,
    tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2,
    per_tcn_stack=8, tcn_with_embed=[0] * 8,
    tcn_norm="gLN", dconv_norm="gGN", causal=False, tcn_layer="normal",
)
mask = model(torch.rand(1, 512, 100))  # [1, 512, 100]

# Target-speaker extraction: first 3 of 8 blocks per stack see dvec
model = ConvTasNet(
    input_dim=512, embed_dim=192, embed_norm=True,
    tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2,
    per_tcn_stack=8, tcn_with_embed=[1, 1, 1, 0, 0, 0, 0, 0],
    tcn_norm="gLN", dconv_norm="gGN", causal=False, tcn_layer="normal",
)
mask = model(torch.rand(1, 512, 100), torch.rand(1, 192))
```
