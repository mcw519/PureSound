# puresound.nnet.unet

繁體中文版本：[unet.zh-TW.md](unet.zh-TW.md)

Status: library — `Unet`, `UnetTcn` and `UnetFsmn` are reachable from a config
(`type: Unet` etc.) and `UnetTcn` has a forward test in
`test/nnet/test_backbone.py`; no maintained recipe uses them directly.
`Unet` is also the base class of the active backbones [DPCRN](dpcrn.md) and
[DPARN](dparn.md).

A 2-D convolutional encoder/decoder over `[frequency, time]` feature maps with
skip connections. The down path downsamples frequency only (time stride is 1
in every recipe); the subclasses insert a temporal model at the bottleneck.

## Class: `Unet`

```python
Unet(
    input_dim: int = 512,                    # frequency bins
    activation_type: str = "PReLU",          # lobe.activation.get_activation
    norm_type: str = "bN2d",                 # lobe.norm.get_norm
    dropout: float = 0.05,                   # after every down stage
    channels: Tuple = (1, 1, 8, 8, 16, 16),  # len == n_cnn + 1; stage i maps channels[i] -> channels[i+1]
    transpose_t_size: int = 2,               # time kernel of every up-stage ConvTranspose2d
    skip_conv: bool = False,                 # False: concat skips; True: 1x1 conv + add
    kernel_t: Tuple = (5, 1, 9, 1, 1),       # per down stage, time axis
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (1, 5, 1, 5, 1),       # per down stage, frequency axis
    stride_f: Tuple = (1, 4, 1, 4, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 1, 0, 0),          # per down stage look-ahead frames
    multi_output: int = 1,                   # channel multiplier of the last up stage
)
```

The six kernel/stride/dilation tuples must have the same length `n_cnn`
(asserted).

```python
forward(x: Tensor) -> Tensor
# x: [N, CH, F, T] or [N, F, T] (unsqueezed to [N, 1, F, T])
# returns [N, channels[0] * multi_output, F, T]
```

Structure:

1. `input_norm`: `iLN(channels[0] * input_dim)`, a per-frame layer norm over
   channel and frequency ([lobe/norm](lobe/norm.md)).
2. Down stage `i`: `ZeroPad2d` -> `Conv2d(channels[i], channels[i+1],
   (kernel_f[i], kernel_t[i]), stride, dilation)` -> norm -> activation ->
   dropout. Time padding is
   `((kernel_t[i] - 1) * dilation_t[i] - delay[i], delay[i])`, so `delay[i]`
   future frames and the rest past frames; frequency padding is
   `kernel_f[i] // 2 * dilation_f[i]` on both sides.
3. Up stages in reverse order: the skip from the matching down stage is
   concatenated (the up conv then takes `2 * channels[i+1]` channels) or, with
   `skip_conv=True`, passed through `1x1 Conv2d + activation` and added.
   `ConvTranspose2d(kernel=(kernel_f[i], transpose_t_size), stride=(stride_f[i],
   stride_t[i]), dilation=1)` restores the frequency size; every stage but the
   last is followed by norm and activation, the last is linear.
4. A transpose conv with time kernel `K` adds `K - 1` frames; `Unet` crops them
   from the end, which keeps the up path causal.

With `transpose_delay` off (the only mode `Unet` has), the model looks
`sum(delay)` frames ahead.

`multi_output` widens only the final up stage's output channels; `forward`
still returns one tensor. The subclasses in this module and `DPCRN`/`DPARN` do
not pass it on, so it is always 1 for them.

`shape_info() -> (down_shape, up_shape)` lists the frequency size at every
stage for a configuration; `forward` does not use it. `get_args` returns all 15
constructor arguments.

## Class: `UnetTcn`

`Unet` with a stack of [`TCN`/`GatedTCN`](conv_tasnet.md) blocks at the
bottleneck. The bottleneck `[N, channels[-1], F', T]` is flattened to
`[N, channels[-1] * F', T]` for the TCN stack and reshaped back, where `F'` is
`input_dim` after all `stride_f` divisions (rounded up).

```python
UnetTcn(
    embed_dim: int = 0,                # speaker-embedding width
    embed_norm: bool = False,          # L2-normalize dvec first
    input_type: str = "RI",            # accepted and ignored
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 1, 8, 8, 16, 16),
    transpose_t_size: int = 2,
    transpose_delay: bool = False,     # crop the leading instead of the trailing frames
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
    tcn_with_embed: List = [1, 0, 0, 0, 0],  # len == per_tcn_stack (asserted)
    tcn_use_film: bool = False,        # FiLM conditioning in conditioned GatedTCN blocks
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",           # TCN only
    causal: bool = False,              # causal TCN blocks
)
# forward(x [N, CH, F, T] | [N, F, T], dvec [N, embed_dim] | None) -> [N, CH, F, T]
```

- The `input_dim` ... `delay` arguments go to `Unet`; the `tcn_*` arguments
  build the stack exactly as in [`ConvTasNet`](conv_tasnet.md) (dilation
  `tcn_dilated_basic ** i`, `tcn_with_embed` per block).
- `tcn_layer="gated"` logs a warning that `dconv_norm` is ignored; any other
  value than `normal`/`gated` raises `ValueError`.
- `tcn_use_film` only affects `GatedTCN` blocks that take `dvec`.
- `transpose_delay=True` keeps the later frames of each transpose conv, which
  adds `transpose_t_size - 1` frames of look-ahead per up stage on top of
  `sum(delay)`.
- `get_args` returns every argument except `input_type`.

### Example

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

`Unet` with a stack of `FSMN` / `ConditionFSMN` blocks
([lobe/rnn](lobe/rnn.md)) at the bottleneck, flattened the same way as
`UnetTcn`.

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
    fsmn_l_context: int = 3,           # past taps of the memory filter
    fsmn_r_context: int = 0,           # future taps; 0 keeps the stack causal
    fsmn_dim: int = 256,               # projection (memory) width
    num_fsmn: int = 8,                 # == len(fsmn_with_embed) (asserted)
    fsmn_with_embed: List = [1, 1, 1, 1, 1, 1, 1, 1],  # 1 -> ConditionFSMN, 0 -> FSMN
    fsmn_norm: str = "gLN",
    use_film: bool = True,             # ConditionFSMN: FiLM instead of concat
)
# forward(x [N, CH, F, T] | [N, F, T], dvec [N, embed_dim] | None) -> [N, CH, F, T]
```

Each FSMN block projects its input to `fsmn_dim`, adds a depthwise filter over
`fsmn_l_context` past and `fsmn_r_context` future frames, adds the previous
block's projection (the memory passed from block to block), and projects back
through norm and ReLU. The default `fsmn_with_embed` conditions every block,
so `forward` needs `dvec` unless the flags are overridden; without it the
first `ConditionFSMN` fails. `transpose_delay` behaves as in `UnetTcn`.
`get_args` returns every argument.

## Design notes

- Frequency and time have separate kernel, stride and dilation tuples because
  only time has a causality constraint: frequency padding is symmetric, time
  padding is placed by `delay`.
- Look-ahead is set per down stage, so the total latency `sum(delay)` frames is
  explicit in the config.
- The per-frame `iLN` input norm keeps the model usable frame by frame, unlike
  an utterance-level norm.
