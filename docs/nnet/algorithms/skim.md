# puresound.nnet.skim

Speech-aware Memory LSTM (SkiM) — a memory-augmented recurrent architecture for efficient long-sequence speech processing.

**Reference:** Li et al., "SkiM: Skipping Memory LSTM for Low-Latency Real-Time Continuous Speech Separation," ICASSP 2022.

## Class: `MemLSTM`

Memory LSTM module that explicitly manages hidden and cell state propagation across segments, enabling long-range dependency modeling without full sequence processing.

### Constructor

```python
MemLSTM(
    hid_size: int,
    dropout: float = 0.0,
    bidirectional: bool = False,
    mem_size: int = 100,
)
```

**Parameters:**
- `hid_size` – LSTM hidden state dimension
- `dropout` – Dropout rate applied between layers (0 = disabled)
- `bidirectional` – If `True`, process both forward and backward directions (non-causal)
- `mem_size` – Memory bank size (number of memory slots)

### `forward(x: Tensor, hid: Optional[Tensor] = None, c: Optional[Tensor] = None) -> Tuple[Tensor, Tensor, Tensor]`

**Parameters:**
- `x` – Input feature sequence `[batch, channel, T]`
- `hid` – Initial hidden state (optional); if `None`, initialized to zeros
- `c` – Initial cell state (optional); if `None`, initialized to zeros

**Returns:** `(output, hidden_state, cell_state)`

- `output` – Processed sequence `[batch, hid_size, T]`
- `hidden_state` – Final LSTM hidden state for streaming continuity
- `cell_state` – Final LSTM cell state for streaming continuity

## Streaming Mode

`MemLSTM` supports causal/streaming inference by passing the returned `hidden_state` and `cell_state` from one chunk as the `hid` and `c` inputs to the next chunk:

```python
hid, c = None, None
for chunk in audio_chunks:
    out, hid, c = mem_lstm(chunk, hid=hid, c=c)
```

## Class: `SkiM`

> **Note:** The top-level `SkiM` model (exported from `puresound.nnet`) wraps `MemLSTM` within a full encoder-decoder pipeline. Refer to the `EncDecMaskBase` system class for the full architecture.

## Example

```python
from puresound.nnet.skim import MemLSTM

mem_lstm = MemLSTM(hid_size=256, bidirectional=False, mem_size=100)

out, hid, c = mem_lstm(features)
```
