# puresound.nnet.lobe.norm

繁體中文版本：`norm.zh-TW.md`

Normalization layers for 1D (`[N, C, T]`) and 2D (`[N, C, F, T]`) speech
features, plus a `get_norm` name→class factory used throughout the backbone
library to pick a normalization strategy from a config string.

## Base Class: `_LayerNorm`

Not part of the public API (no leading re-export, name-mangled with a leading
underscore). Holds the learnable `gamma`/`beta` affine parameters, each shape
`[channel_size]`, and `apply_gain_and_bias`, which applies them by
transposing the channel axis to the end, scaling, then transposing back — so
subclasses only need to implement the actual mean/var computation in
`forward`.

```python
_LayerNorm(channel_size: int)
```

---

## Class: `GlobLN`  (alias `gLN`)

**Global Layer Normalization** — normalizes over *every* non-batch dimension
at once (all channels and all time steps together, one mean/var per batch
item).

```python
GlobLN(channel_size: int)
```

### `forward(x: Tensor) -> Tensor`

**Input:** `[N, C, *]` (works for any rank above 2D, not just `[N, C, T]`).
Not causal-safe — statistics see the entire sequence, including future frames.

---

## Class: `ChanLN`  (alias `cLN`)

**Channel-wise Layer Normalization** — one mean/var per `(batch, time)` pair,
computed across the channel axis only.

```python
ChanLN(channel_size: int)
```

### `forward(x: Tensor) -> Tensor`

**Input:** `[N, C, *]`. Causal-safe (each time step normalizes independently
of neighboring time steps) — used e.g. by `TFGridNet`'s intra/inter norms.

---

## Class: `InstantLN`  (alias `iLN`)

**Instant Layer Normalization** — normalizes over the flattened
`(channel, frequency)` axes at each time step independently; suitable for
causal/streaming 2D models.

```python
InstantLN(channel_size: int)
```

### `forward(x: Tensor) -> Tensor`

**Input:** `[N, CH, C, T]` — reshapes to `[N, CH*C, T]`, normalizes per time
step across the flattened channel/frequency axis, reshapes back.

---

## Class: `LayerNorm2D`  (alias `LN2D`)

**Channel- and frequency-wise Layer Normalization.** Unlike the three classes
above, this one needs the frequency-axis width up front, because its affine
parameters are shaped per-`(channel, frequency)` pair, not just per-channel:

```python
LayerNorm2D(ch: int, f: int)
```

**Parameters:**
- `ch` – channel count
- `f` – frequency-axis width — `self.w`/`self.b` are `[1, ch, f, 1]`, **not** `[ch]`

### `forward(x: Tensor) -> Tensor`

**Input:** `[N, ch, C, T]` (`C` must equal `f`). Normalizes jointly over the
`(channel, freq)` axes (`dims=[1,2]`) at each time step, then applies the
per-`(ch, f)` affine transform.

> **Known gap:** `LN2D` is a real alias in this module (`LN2D = LayerNorm2D`),
> but it is **not reachable through the `get_norm` factory** below — the
> factory's whitelist never includes `"LN2D"`. Callers that want
> `LayerNorm2D` must import and construct it directly (as `TFGridNet` does —
> see Wiring). This is a known, unfixed gap in `get_norm`, not addressed here.

---

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
get_norm(name: str) -> nn.Module
```

Factory that resolves a **short code string** to a normalization **class**
(not an instance — call the result with the channel count to build the
layer). Single parameter; there is no separate `channel_size` argument here —
you instantiate the returned class yourself.

**Supported values** (exactly these 6 strings — long-form names like
`"global_layer_norm"` are not accepted):

| Code | Class |
|------|-------|
| `"gLN"` | `GlobLN` |
| `"cLN"` | `ChanLN` |
| `"iLN"` | `InstantLN` |
| `"bN1d"` | `nn.BatchNorm1d` |
| `"bN2d"` | `nn.BatchNorm2d` |
| `"gGN"` | `nn.GroupNorm(1, ·, 1e-8)` |

**Raises:** `NameError` for any other string (including `"LN2D"` — see the
known gap above).

## Choosing a Normalization Strategy

| Class | Causal-safe | Needs frequency width | Typical use |
|-------|:-----------:|:----------------------:|-------------|
| `GlobLN` | No | No | non-causal 1D (Conv-TasNet TCN) |
| `ChanLN` | Yes | No | causal 1D, or `TFGridNet`'s intra/inter norm |
| `InstantLN` | Yes | No | causal/streaming 2D (`Unet`'s input norm) |
| `LayerNorm2D` | No | Yes | 2D time-frequency attention (`TFGridNet`) |

## Wiring

`get_norm` is used by [`cnn.DepthwiseSeparableConv1d`](cnn.md),
[`rnn.FSMN`](rnn.md)/`ConditionFSMN`, and directly by `ConvTasNet`/`Unet` to
pick their normalization layer from a config string (`tcn_norm`, `norm_type`,
etc.). `TFGridNet` imports `LayerNorm2D`, `cLN`, `gLN` directly — bypassing
`get_norm` for its own `input_norm in ["LayerNorm2D", "gLN"]` config switch,
exactly because `get_norm` cannot produce a `LayerNorm2D` (see the known gap
above).

## Example

```python
from puresound.nnet.lobe.norm import get_norm

NormCls = get_norm("gLN")
norm = NormCls(256)
out = norm(features)  # [N, 256, T]
```
