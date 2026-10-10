# puresound.nnet.lobe.norm

繁體中文版本：[norm.zh-TW.md](norm.zh-TW.md)

Normalization layers for 1D (`[N, C, T]`) and 2D (`[N, C, F, T]`) features, and
`get_norm`, which maps a config string to a normalization class.

All layer norms here compute `(x - mean) / sqrt(var + eps)` with `eps = 1e-8`
(`LayerNorm2D`: `/ (std + 1e-8)`) and then apply a learned affine transform.
They differ in which axes the statistics are taken over, which decides whether
the layer is causal: a layer is causal when it never pools over time.

## Base class: `_LayerNorm`

Private. Holds `gamma` (ones) and `beta` (zeros), each `[channel_size]`, and
`apply_gain_and_bias`, which moves the channel axis last, applies them, and
moves it back. Subclasses implement only the statistics.

```python
_LayerNorm(channel_size: int)
```

## Class: `GlobLN` (alias `gLN`)

Global layer norm: one mean and variance per batch item, over every non-batch
axis (channels and time together). Input `[N, C, *]`. **Not causal** — the
statistics include future frames.

```python
GlobLN(channel_size: int)
```

## Class: `ChanLN` (alias `cLN`)

Channel-wise layer norm: one mean and variance per `(batch, time)` position,
over the channel axis only. Input `[N, C, *]`. Causal.

```python
ChanLN(channel_size: int)
```

## Class: `InstantLN` (alias `iLN`)

Instant layer norm for 2D maps: reshapes `[N, CH, C, T]` to `[N, CH*C, T]` and
normalizes each frame over the flattened channel-frequency axis. Causal.
`channel_size` must be `CH * C`, because the affine parameters are per
channel-frequency position.

```python
InstantLN(channel_size: int)
```

## Class: `LayerNorm2D` (alias `LN2D`)

Layer norm over channel and frequency jointly (`dims=[1, 2]`) at each frame,
with affine parameters per `(channel, frequency)` pair — `w` and `b` are
`[1, ch, f, 1]`, so the frequency width is fixed at construction. Input
`[N, ch, f, T]`. Causal.

```python
LayerNorm2D(ch: int, f: int)
```

`get_norm` does not accept `"LN2D"`; construct `LayerNorm2D` directly (as
`TFGridNet` does).

## Aliases

```python
gLN  = GlobLN
cLN  = ChanLN
iLN  = InstantLN
bN1d = nn.BatchNorm1d
bN2d = nn.BatchNorm2d
gGN  = lambda x: nn.GroupNorm(1, x, 1e-8)
LN2D = LayerNorm2D
```

## Function: `get_norm`

```python
get_norm(name: str)   # returns a class (or the gGN factory), not an instance
```

Call the result with the channel count to build the layer. Accepted names —
exactly these six; anything else raises `NameError`:

| name | layer | causal |
|---|---|:-:|
| `"gLN"` | `GlobLN` | no |
| `"cLN"` | `ChanLN` | yes |
| `"iLN"` | `InstantLN` | yes |
| `"bN1d"` | `nn.BatchNorm1d` | yes at inference (running statistics) |
| `"bN2d"` | `nn.BatchNorm2d` | yes at inference (running statistics) |
| `"gGN"` | `nn.GroupNorm(1, C, eps=1e-8)` | no |

## Where each is used

| layer | used by |
|---|---|
| `get_norm(...)` | [`cnn.DepthwiseSeparableConv1d`](cnn.md), [`rnn.FSMN`](rnn.md) / `ConditionFSMN`, `ConvTasNet` (`tcn_norm`), `Unet` and its subclasses such as `DPCRN` (`norm_type`, default `"bN2d"` in DPCRN) |
| `InstantLN` | `Unet`'s input norm, `iLN(channels[0] * input_dim)` |
| `ChanLN`, `LayerNorm2D`, `GlobLN` | `TFGridNet` (intra/inter norms, attention norms, `input_norm` = `LayerNorm2D` or `gLN`) |

## Design notes

Streaming models need a causal norm: `ChanLN`, `InstantLN`, `LayerNorm2D`, or
batch norm (fixed statistics at inference). `GlobLN` and `gGN` pool over the
whole sequence and suit offline models only.

## Example

```python
from puresound.nnet.lobe.norm import get_norm

NormCls = get_norm("cLN")
norm = NormCls(256)
out = norm(features)  # [N, 256, T]
```
