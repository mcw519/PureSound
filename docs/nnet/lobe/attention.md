# puresound.nnet.lobe.attention

繁體中文版本：`attention.zh-TW.md`

Sinusoidal positional encoding, a thin multi-head-attention wrapper with
forward-time causal / local-context masking, and a full self-attention encoder
block that can swap its feed-forward sublayer for an LSTM.

## Class: `PositionalEncoding`

Adds the standard sinusoidal Transformer positional encoding to a sequence.

```python
PositionalEncoding(d_model: int, dropout: float = 0.1, max_len: int = 5000)
```

**Parameters:**
- `d_model` – feature dimension; must be even (interleaved `sin`/`cos` pairs), otherwise raises `ValueError`
- `dropout` – dropout applied after adding the positional term — **default is `0.1`**, not `0.0`
- `max_len` – longest sequence length the precomputed `pe` buffer supports

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – `[N, T, C]`

Internally permutes to `[T, N, C]` to add the `pe` buffer, applies dropout,
then permutes back. **Returns:** `[N, T, C]`, same shape as input.

---

## Class: `MHA`

A thin wrapper around `nn.MultiheadAttention` that adds causal / local-context
window masking as **forward-time** arguments. The constructor itself takes
only two arguments — dropout and the in/out-projection bias are hardcoded,
not configurable:

```python
MHA(embed_dim: int, heads: int = 1)
```

```python
self.atten = nn.MultiheadAttention(
    embed_dim=embed_dim, num_heads=heads, dropout=0, batch_first=True, bias=False,
)
```

**Parameters:**
- `embed_dim` – total attention dimension (`embed_dim = head_dim * heads`)
- `heads` – number of attention heads

### `forward(query, key, value, causal: bool = True, context_range: int = None) -> Tuple[Tensor, Tensor]`

Masking is entirely a `forward`-time choice. `causal` and `context_range`
combine into four attention patterns:

| `causal` | `context_range` | Attends to |
|:---:|:---:|---|
| `True` | `None` | standard causal mask — current + all past frames |
| `True` | `k` | causal **and** windowed — only the trailing `k` frames (excludes older history) |
| `False` | `k` | non-causal local window — `±(k-1)` frames around each position |
| `False` | `None` | no mask — full bidirectional attention |

**Parameters:**
- `query`, `key`, `value` – `[N, T, C]` (`batch_first=True`)
- `causal` – **defaults to `True`**; callers wanting bidirectional attention must pass `causal=False` explicitly
- `context_range` – window length; combines with `causal` per the table above

**Returns:** `(attn_output, attn_weights)`, passed straight through from
`nn.MultiheadAttention.forward`; `attn_output` is `[N, T, C]`.

---

## Class: `MhaSelfAttenLayer`

A complete self-attention encoder block: self-attention + residual + norm,
then a feed-forward (or LSTM) sublayer + residual + norm. Normalization
happens **after** each residual add (post-LN, as in the original Transformer
paper) — not pre-LN.

```python
MhaSelfAttenLayer(
    feats_dim: int,
    hidden_dim: int,
    nhead: int,
    dropout: float = 0.0,
    improved: bool = False,
    bidirectional: bool = False,
    position_encoding: bool = True,
)
```

**Parameters:**
- `feats_dim` – model feature dimension (attention `embed_dim` and residual-stream width)
- `hidden_dim` – feed-forward hidden size, or the internal `nn.LSTM`'s hidden size when `improved=True`
- `nhead` – attention heads, forwarded to `MHA(feats_dim, heads=nhead)`
- `dropout` – applied to the attention output, inside the feed-forward path, and (if `position_encoding`) inside `PositionalEncoding`
- `improved` – replaces the linear feed-forward sublayer with an `nn.LSTM` (see reference below). There is **no `causal` constructor parameter** on this class — see `forward` below
- `bidirectional` – only meaningful when `improved=True` (controls the internal LSTM); ignored, with a printed warning ("Ignored bidirectional option since no LSTM here."), when `improved=False`
- `position_encoding` – add a `PositionalEncoding` before attention; ignored, with a printed warning ("Ignored position_encoding option here replaced by LSTM modeling."), when `improved=True`, since the LSTM already encodes order

**Reference:** Dual-Path Transformer Network — Direct Context-Aware Modeling
for End-to-End Monaural Speech Separation (the `improved` FF↔LSTM swap).

### `forward(x, causal=False, context_range=None, return_atten_weight=False)`

**Parameters:**
- `x` – `[N, C, T]` — **channel-first**, the opposite of `MHA`'s own `[N, T, C]`; this layer permutes internally
- `causal` – forwarded to the internal `MHA.forward` call. **Default `False`** here, overriding `MHA.forward`'s own default of `True` — this layer always passes `causal` explicitly, so `MHA`'s internal default never actually applies
- `context_range` – forwarded to `MHA.forward` unchanged
- `return_atten_weight` – if `True`, also return the attention weights

Block structure:

```
src = x                                          # pre-positional-encoding residual
x   = pos(x)   if position_encoding else x       # positional term feeds only the attention branch
x   = self_atten(x, x, x, causal, context_range)
x   = norm1(src + dropout(x))
src = x
x   = recurrent(x)   if improved else x          # LSTM only — no linear layer ahead of it
x   = feedforward(x)                             # Linear-ReLU-Drop-Linear-Drop, or ReLU-Drop-Linear-Drop if improved
x   = norm2(src + x)
```

Note the residual (`src`) captured before `pos(x)` means the skip connection
never carries positional information directly — only whatever the attention
sublayer derives from it.

**Returns:** `x` shaped `[N, C, T]` (same as input), or `(x, w)` if
`return_atten_weight=True`.

## Wiring

`DPARN`'s dual-path block (`puresound/nnet/dparn.py`) stacks two
`MhaSelfAttenLayer`s for intra-chunk modeling — the first with
`position_encoding=True`, the second with `position_encoding=False` (so order
is injected once, not twice), both non-causal and both `improved=False`.
Inter-chunk modeling in the same block uses [`rnn.SingleRNN`](rnn.md) instead
of attention.

## Example

```python
from puresound.nnet.lobe.attention import MhaSelfAttenLayer, PositionalEncoding

pos_enc = PositionalEncoding(d_model=256, dropout=0.1)
atten   = MhaSelfAttenLayer(feats_dim=256, hidden_dim=1024, nhead=8)

x = torch.randn(4, 256, 100)   # [N, C, T]
x_out = atten(x, causal=False)
```
