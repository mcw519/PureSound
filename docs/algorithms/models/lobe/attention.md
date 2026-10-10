# puresound.nnet.lobe.attention

繁體中文版本：[attention.zh-TW.md](attention.zh-TW.md)

Sinusoidal positional encoding, a multi-head attention wrapper whose causal or
local-window mask is chosen per call, and a post-LN Transformer encoder layer
built from them.

## Class: `PositionalEncoding`

Adds the Transformer sinusoidal position code
`PE[t, 2i] = sin(t / 10000^(2i/d))`, `PE[t, 2i+1] = cos(t / 10000^(2i/d))`,
then applies dropout.

```python
PositionalEncoding(
    d_model: int,          # feature width; must be even, else ValueError
    dropout: float = 0.1,  # applied after the addition
    max_len: int = 5000,   # longest sequence the precomputed table covers
)
```

`forward(x [N, T, C]) -> [N, T, C]`.

## Class: `MHA`

`nn.MultiheadAttention` (`batch_first=True`, no projection bias, no attention
dropout) plus mask construction.

```python
MHA(embed_dim: int, heads: int = 1)   # embed_dim = head_dim * heads
```

`forward(query, key, value, causal: bool = True, context_range: int = None)`
takes `[N, T, C]` tensors and returns `(output [N, T, C], weights)` from
`nn.MultiheadAttention`. The mask is built from the query length, so it
assumes self-attention (key length equal to query length). Frame `i` may
attend to frame `j` when:

| `causal` | `context_range` | allowed |
| --- | --- | --- |
| `True` | `None` | `j <= i` |
| `True` | `k` | `i - k < j <= i` (the current frame and `k - 1` past frames) |
| `False` | `k` | `abs(i - j) <= k - 2`; use `k >= 2` |
| `False` | `None` | all frames |

## Class: `MhaSelfAttenLayer`

One encoder layer: self-attention, residual, LayerNorm; then a feed-forward
sublayer, residual, LayerNorm (post-LN).

```python
MhaSelfAttenLayer(
    feats_dim: int,                  # model width (attention embed_dim)
    hidden_dim: int,                 # feed-forward width, or LSTM hidden size when improved
    nhead: int,
    dropout: float = 0.0,            # attention output, feed-forward, positional encoding
    improved: bool = False,          # LSTM before the feed-forward output layer [1]
    bidirectional: bool = False,     # LSTM direction; only used when improved
    position_encoding: bool = True,  # add PositionalEncoding before attention
)
```

`forward(x, causal=False, context_range=None, return_atten_weight=False)`
takes channel-first `x [N, C, T]` and returns `[N, C, T]`, or
`([N, C, T], weights)` when `return_atten_weight=True`. `causal` and
`context_range` go to `MHA.forward` unchanged.

```
src = x
x   = pos(x)                    if position_encoding
x   = norm1(src + dropout(attention(x, x, x)))
src = x
x   = lstm(x)                   if improved
x   = feedforward(x)            # Linear-ReLU-Dropout-Linear-Dropout,
                                # or ReLU-Dropout-Linear-Dropout when improved
x   = norm2(src + x)
```

With `improved=True` the LSTM carries position, so `position_encoding=True`
is ignored with a logged warning and no encoding is built or applied.
`bidirectional` without `improved` is ignored with a logged warning.

[1] Chen, Mao, Liu, "Dual-Path Transformer Network: Direct Context-Aware
Modeling for End-to-End Monaural Speech Separation", Interspeech 2020.

## Use

`DPARN` and `DPCRN` with `intra_type: attention` run two `MhaSelfAttenLayer`s
over the frequency positions of each frame (the bottleneck channels are the
feature axis), non-causal, `improved=False`. The first adds the positional
encoding, the second does not, so position is injected once. See
[DPARN](../dparn.md) and [DPCRN](../dpcrn.md).

```yaml
backbone:
  type: DPCRN
  backbone_args:
    intra_type: attention
    intra_nhead: 4
```

## Design notes

- Masking is a call argument rather than a constructor argument, so one layer
  can run causal over time and unmasked over frequency.
- The residual is taken before the positional encoding, so the skip path
  carries only the features and position reaches the output only through what
  attention derives from it.
- Frequency has no causal constraint, so attention over frequency sees every
  position in one matrix product, where an intra-frequency LSTM walks them one
  step at a time.
