# puresound.nnet.dprnn

Dual-Path RNN (DPRNN) for time-domain speech source separation and enhancement.

**Reference:** Luo et al., "Dual-Path RNN: Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech Separation," ICASSP 2020.

## Class: `DPRNN`

Processes sequential audio features using alternating intra-chunk and inter-chunk LSTM layers, enabling efficient long-sequence modeling.

### Architecture

```
Input Features [batch, channel, time]
  └─ Segment into overlapping chunks [batch, channel, chunk_size, num_chunks]
       └─ For each DPRNN layer:
            ├─ Intra-chunk LSTM (bidirectional, over chunk_size dimension)
            └─ Inter-chunk LSTM (unidirectional, over num_chunks dimension)
                 └─ Optional: FiLM conditioning from embedding
  └─ Overlap-add reconstruction [batch, channel, time]
```

### Constructor

```python
DPRNN(
    in_channel: int,
    hid_channel: int,
    out_channel: int,
    num_layers: int,
    segment_size: int = 100,
    segment_overlap: int = 50,
    causal: bool = False,
    embed_dim: Optional[int] = None,
    embed_on_all_layers: bool = False,
)
```

**Parameters:**
- `in_channel` – Input feature dimension
- `hid_channel` – LSTM hidden dimension
- `out_channel` – Output feature dimension
- `num_layers` – Number of DPRNN blocks (alternating intra + inter LSTM)
- `segment_size` – Chunk length in time steps
- `segment_overlap` – Overlap between consecutive chunks (must be < `segment_size`)
- `causal` – If `True`, inter-chunk LSTM is unidirectional (causal processing)
- `embed_dim` – Embedding dimension for FiLM conditioning (optional)
- `embed_on_all_layers` – If `True`, apply conditioning embedding to all layers; if `False`, only the first layer

### `forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor`

**Parameters:**
- `x` – Input feature tensor `[batch, in_channel, T]`
- `embed` – Optional conditioning embedding `[batch, embed_dim]`

**Returns:** Output feature tensor `[batch, out_channel, T]`.

## Chunking Behavior

| Parameter | Effect |
|-----------|--------|
| `segment_size` | Longer chunks capture more local context per intra-LSTM pass |
| `segment_overlap` | Higher overlap reduces boundary artifacts |
| `causal=True` | Enables streaming/online processing |

## Example

```python
from puresound.nnet.dprnn import DPRNN

dprnn = DPRNN(
    in_channel=256,
    hid_channel=64,
    out_channel=256,
    num_layers=6,
    segment_size=100,
    segment_overlap=50,
    causal=False,
)

# Standard (no conditioning)
out = dprnn(features)

# With speaker embedding conditioning
out = dprnn(features, embed=spk_emb)
```
