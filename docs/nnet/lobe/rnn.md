# puresound.nnet.lobe.rnn

Recurrent neural network variants for sequential audio processing.

## Class: `SingleRNN`

A unified wrapper around PyTorch's RNN, LSTM, and GRU modules with optional output projection.

### Constructor

```python
SingleRNN(
    rnn_type: str,
    in_size: int,
    hid_size: int,
    dropout: float = 0.0,
    bidirectional: bool = False,
    proj_size: Optional[int] = None,
)
```

**Parameters:**
- `rnn_type` – One of `"RNN"`, `"LSTM"`, `"GRU"`
- `in_size` – Input feature dimension
- `hid_size` – Hidden state dimension
- `dropout` – Dropout between layers (0 = no dropout)
- `bidirectional` – If `True`, use bidirectional RNN
- `proj_size` – If set, applies a linear projection from `2*hid_size` (or `hid_size`) to `proj_size`

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – Input sequence `[batch, T, in_size]`

**Returns:** Output `[batch, T, proj_size or hid_size*(1+bidirectional)]`.

---

## Class: `FSMN`

**Feedforward Sequential Memory Network** — an efficient alternative to RNNs that uses dilated 1D convolutions to look at both past and future frames without recurrence.

**Reference:** Zhang et al., "Feedforward Sequential Memory Neural Network without Recurrent Feedback," arXiv 2015.

### Constructor

```python
FSMN(
    in_size: int,
    hid_size: int,
    memory_size: int,
    lookahead: int = 0,
    lorder: int = 20,
)
```

**Parameters:**
- `in_size` – Input feature dimension
- `hid_size` – Hidden/output dimension
- `memory_size` – Memory order (number of past frames to aggregate)
- `lookahead` – Number of future frames to look ahead (0 = causal)
- `lorder` – Left order (past context length)

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – Input sequence `[batch, T, in_size]`

**Returns:** Output sequence `[batch, T, hid_size]`.

---

## Class: `ConditionFSMN`

An FSMN variant that supports FiLM conditioning from an external embedding vector.

### Constructor

```python
ConditionFSMN(
    in_size: int,
    hid_size: int,
    memory_size: int,
    embed_dim: int,
    lookahead: int = 0,
    lorder: int = 20,
)
```

**Parameters:**
- `embed_dim` – Conditioning embedding dimension
- All other parameters as in `FSMN`

### `forward(x: Tensor, embed: Tensor) -> Tensor`

**Parameters:**
- `x` – Input sequence `[batch, T, in_size]`
- `embed` – Conditioning embedding `[batch, embed_dim]`

**Returns:** Conditioned output sequence `[batch, T, hid_size]`.

## Comparison

| Class | Recurrent | Causal Option | Conditioning | Notes |
|-------|:---------:|:------------:|:------------:|-------|
| `SingleRNN` | ✅ | ✅ (unidirectional) | ❌ | General RNN/LSTM/GRU wrapper |
| `FSMN` | ❌ | ✅ | ❌ | Efficient memory network |
| `ConditionFSMN` | ❌ | ✅ | ✅ | FSMN + FiLM conditioning |
