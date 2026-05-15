# puresound.nnet.lobe.attention

Attention mechanisms including positional encoding, multi-head attention, and self-attention layers.

## Class: `PositionalEncoding`

Adds sinusoidal positional encoding to a sequence of features, following the standard Transformer formulation.

### Constructor

```python
PositionalEncoding(d_model: int, dropout: float = 0.0, max_len: int = 5000)
```

**Parameters:**
- `d_model` – Feature dimension (must match input embedding dimension)
- `dropout` – Dropout rate applied after adding positional encoding
- `max_len` – Maximum sequence length supported

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – Input tensor `[batch, T, d_model]`

**Returns:** Positionally encoded tensor of the same shape.

---

## Class: `MHA`

Multi-head attention module with support for causal masking and context window constraints.

### Constructor

```python
MHA(
    d_model: int,
    num_heads: int,
    dropout: float = 0.0,
    causal: bool = False,
    context_size: Optional[int] = None,
)
```

**Parameters:**
- `d_model` – Total attention dimension
- `num_heads` – Number of attention heads
- `dropout` – Attention weight dropout rate
- `causal` – If `True`, applies a causal (lower triangular) attention mask
- `context_size` – If set, limits attention to a fixed local context window (past `context_size` frames only)

### `forward(query: Tensor, key: Tensor, value: Tensor, mask: Optional[Tensor] = None) -> Tensor`

**Parameters:**
- `query`, `key`, `value` – Attention inputs `[batch, T, d_model]`
- `mask` – Optional boolean mask `[batch, T, T]`

**Returns:** Attended output tensor `[batch, T, d_model]`.

---

## Class: `MhaSelfAttenLayer`

A complete self-attention layer with pre-layer normalization, multi-head attention, residual connection, and feed-forward projection.

### Constructor

```python
MhaSelfAttenLayer(
    d_model: int,
    num_heads: int,
    ffn_dim: int,
    dropout: float = 0.0,
    causal: bool = False,
)
```

**Parameters:**
- `d_model` – Feature dimension
- `num_heads` – Number of attention heads
- `ffn_dim` – Feed-forward hidden dimension
- `dropout` – Dropout rate
- `causal` – If `True`, enables causal attention masking

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – Input sequence `[batch, T, d_model]`

**Returns:** Output sequence of the same shape.

## Example

```python
from puresound.nnet.lobe.attention import MhaSelfAttenLayer, PositionalEncoding

pos_enc = PositionalEncoding(d_model=256, dropout=0.1)
atten   = MhaSelfAttenLayer(d_model=256, num_heads=8, ffn_dim=1024, causal=False)

x_pe  = pos_enc(x)
x_out = atten(x_pe)
```
