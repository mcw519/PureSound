# puresound.nnet.lobe.group_op

Group operation modules for permutation-invariant and channel-grouped processing.

## Class: `TAC`

**Transform-Average-Concatenate (TAC)** — a global communication layer that enables channels within a group to exchange information through averaging.

**Reference:** Luo et al., "End-to-End Microphone Permutation and Number Invariant Multi-Channel Speech Separation," ICASSP 2020.

### Constructor

```python
TAC(
    in_channel: int,
    hid_channel: int,
)
```

**Parameters:**
- `in_channel` – Per-channel input dimension
- `hid_channel` – Hidden dimension for the transform step

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – Input tensor `[batch, num_channels, in_channel, T]`

**Returns:** Output tensor `[batch, num_channels, in_channel, T]`.

### TAC Operation

1. **Transform**: Per-channel linear transform → `[B, C, hid, T]`
2. **Average**: Global mean across channels → `[B, 1, hid, T]`
3. **Concatenate**: Append averaged context to each channel's transform output
4. **Project**: Linear projection back to `in_channel`

---

## Class: `GroupedGRULayer`

A GRU where channels are split into groups, each processed by a smaller GRU, reducing parameter count while maintaining sequence modeling capacity.

### Constructor

```python
GroupedGRULayer(
    in_size: int,
    hid_size: int,
    num_groups: int,
    batch_first: bool = True,
    bidirectional: bool = False,
)
```

**Parameters:**
- `in_size` – Total input feature dimension (split evenly across groups)
- `hid_size` – Total hidden dimension (split evenly across groups)
- `num_groups` – Number of independent GRU groups
- `batch_first` – If `True`, input format is `[batch, T, features]`
- `bidirectional` – If `True`, each group GRU is bidirectional

### `forward(x: Tensor, hid: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]`

**Parameters:**
- `x` – Input tensor `[batch, T, in_size]` (if `batch_first=True`)
- `hid` – Optional initial hidden state

**Returns:** `(output, hidden_state)`

## Example

```python
from puresound.nnet.lobe.group_op import TAC, GroupedGRULayer

tac = TAC(in_channel=64, hid_channel=128)
gru = GroupedGRULayer(in_size=128, hid_size=128, num_groups=4)

# TAC for multi-channel context sharing
out = tac(multi_channel_features)  # [B, num_ch, 64, T]

# Grouped GRU for efficient sequence modeling
out, hid = gru(features)
```
