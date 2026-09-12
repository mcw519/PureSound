# puresound.nnet.lobe.group_op

繁體中文版本：`group_op.zh-TW.md`

Group-structured operators: a cross-channel communication layer (`TAC`), and
a family of channel-grouped GRU/Linear layers that trade sequence-modeling
capacity for parameter count by splitting features into independent groups.

## Class: `TAC`

**Transform-Average-Concatenate.** A global communication layer letting
per-microphone (or otherwise grouped) channels exchange information through a
shared average, without needing a fixed channel count or ordering.

```python
TAC(input_dim: int, hidden_dim: int)
```

**Parameters:**
- `input_dim` – per-group input feature dimension
- `hidden_dim` – hidden dimension used for the transform/mean/output stages

**Reference:** Luo et al., "End-to-End Microphone Permutation and Number
Invariant Multi-Channel Speech Separation," ICASSP 2020
([code](https://github.com/yluo42/GC3/blob/main/utility/basics.py#L28)).

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, G, C, T]` where `G` is the number of groups (e.g. mics)

Steps: **transform** each group independently (`Linear(input_dim→hidden_dim) + PReLU`)
→ **average** across `G` → **project** the mean (`Linear(hidden_dim→hidden_dim) + PReLU`)
→ **concatenate** each group's transform output with the (broadcast) projected
mean → **output** projection back to `input_dim` (`Linear(2*hidden_dim→input_dim) + PReLU`)
→ `nn.BatchNorm1d(input_dim)` → residual-added onto the original input.

Note the normalization is a plain (non-causal, whole-batch) `BatchNorm1d`; the
source code comments that the original paper instead uses a non-causal
`nn.GroupNorm(1, input_size)` — a deliberate deviation, not an oversight.

**Returns:** `[N, G, C, T]`, same shape as input.

No caller in the repository's backbones today — a library building block.

---

## Class: `GroupedGRULayer`

A GRU where the channel dimension is split evenly into `groups` independent
`nn.GRU`s, each seeing `input_size / groups` channels — reduces parameter
count versus one full-width GRU while keeping per-group recurrence.

```python
GroupedGRULayer(
    input_size: int,
    hidden_size: int,
    groups: int,
    bidirectional: bool = False,
    bias: bool = True,
    dropout: float = 0.0,
)
```

**Parameters:**
- `input_size` – total input feature dimension, must be divisible by `groups`
- `hidden_size` – total hidden dimension, must be divisible by `groups`
- `groups` – number of independent GRUs; each handles a contiguous `input_size/groups`-wide channel slice
- `bidirectional` – forwarded to each inner `nn.GRU`
- `bias` – forwarded to each inner `nn.GRU`
- `dropout` – forwarded to each inner `nn.GRU`'s own `dropout` argument (inter-layer dropout, only relevant when that GRU has >1 internal layer — here it's always 1, so this has no effect through `GroupedGRULayer` itself)

There is **no `batch_first` parameter** — `batch_first=True` is hardcoded on
every inner `nn.GRU`.

> **Fixed in this pass:** this parameter was previously misspelled `droupout`
> and passed to `nn.GRU(..., droupout=droupout, ...)`, which doesn't accept
> that keyword — `nn.GRU.__init__` would raise `TypeError` immediately for
> any attempt to construct this class. No caller anywhere in the repository
> passed the misspelled keyword explicitly, so the fix is a pure rename with
> no external call sites to update. Verified live: `GroupedGRULayer(...)`
> now constructs and runs correctly (`forward`, `return_hidden=True`, and
> `flatten_parameters()` all checked).

### `forward(x, h0=None, return_hidden=False)`

**Parameters:**
- `x` – `[N, C, T]` (permuted to `[N, T, C]` internally, batch-first, then split into `groups` contiguous chunks along the last axis)
- `h0` – optional initial hidden state, `[groups * num_directions, N, hidden_size / groups]`; each group reads its own slice (`.detach()`-ed before use)
- `return_hidden` – if `True`, also return the concatenated final hidden state

**Returns:** `outputs` (`[N, C, T]`, `C = hidden_size` after concatenating all
groups' outputs), or `(outputs, h)` if `return_hidden=True`.

### `flatten_parameters()`

Calls `flatten_parameters()` on every inner `nn.GRU` (cuDNN weight-layout
housekeeping after loading a checkpoint or moving devices).

---

## Class: `GroupedGRU`

Stacks `num_layers` `GroupedGRULayer`s, optionally channel-shuffling between
groups after each layer (so groups aren't permanently isolated from each
other's features across depth — the same idea as ShuffleNet's channel
shuffle).

```python
GroupedGRU(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    groups: int = 4,
    bidirectional: bool = False,
    bias: bool = True,
    dropout: float = 0.0,
    shuffle: bool = True,
)
```

**Parameters:**
- `input_size` / `hidden_size` – as in `GroupedGRULayer`; the first layer maps `input_size → hidden_size`, later layers map `hidden_size → hidden_size`
- `num_layers` – number of stacked `GroupedGRULayer`s (must be `> 0`)
- `groups` – forwarded to every layer; forces `shuffle=False` if `groups == 1` regardless of the value passed
- `bidirectional`, `bias`, `dropout` – forwarded to every layer
- `shuffle` – interleave channels across groups between stacked layers (all but the last)

### `forward(x, h0=None, return_hidden=False)`

**Parameters:** as in `GroupedGRULayer`, with `h0` shaped
`[num_layers * groups * num_directions, N, hidden_size / groups]`.

Each stacked layer is driven as `x, s = gru(x, h0[...], return_hidden=True)`;
the per-layer hidden states are collected and concatenated, and the channel
shuffle (when `shuffle=True`) runs between layers, on all but the last.

**Returns:** `x` (`[N, hidden_size, T]`), or `(x, outstates)` if
`return_hidden=True`, where `outstates` is every layer's hidden state stacked
into `[num_layers * groups * num_directions, N, hidden_size / groups]` — the
same layout `h0` takes, so it can be fed straight back in on the next chunk.

> **Fixed in this pass:** `forward` used to call the inner layers without
> `return_hidden=True` while still unpacking the result as `x, s = gru(...)`.
> `GroupedGRULayer` returns a bare `Tensor` unless the hidden state is asked
> for, so the unpack tried to split that tensor along its batch axis — raising
> `ValueError` for every batch size except 2, and for batch size 2 silently
> dropping the batch axis before crashing in the channel-shuffle `.permute` a
> few lines later. Both call paths (`return_hidden` on and off) are verified
> working now.

---

## Class: `GroupedLinear`

Channel-grouped `nn.Linear`: splits the channel axis into `groups` slices,
applies an independent `Linear` per slice, optionally shuffles the output
channels across groups.

```python
GroupedLinear(
    input_size: int,
    hidden_size: int,
    groups: int = 1,
    shuffle: bool = True,
)
```

**Parameters:**
- `input_size` / `hidden_size` – total dims, each divisible by `groups`
- `groups` – number of independent `Linear` layers
- `shuffle` – interleave output channels across groups; forced `False` if `groups == 1`

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, C, T]` (permuted to `[N, T, C]` internally)

**Returns:** `[N, hidden_size, T]`.

---

## Class: `SqueezedGRU`

"Squeeze" pattern: project down to a shared width with a cheap
`GroupedLinear`, run one ordinary (non-grouped) `nn.GRU` at that width, then
optionally project back out with another `GroupedLinear`.

```python
SqueezedGRU(
    input_size: int,
    hidden_size: int,
    output_size: Optional[int] = None,
    num_layers: int = 1,
    linear_groups: int = 8,
)
```

**Parameters:**
- `input_size` – input feature dimension
- `hidden_size` – both the `GroupedLinear` projection target and the GRU's hidden size
- `output_size` – if given, a second `GroupedLinear(hidden_size→output_size) + ReLU` projects the GRU output; if `None`, `nn.Identity()` (output stays at `hidden_size`)
- `num_layers` – layers of the single shared `nn.GRU`
- `linear_groups` – groups used by both `GroupedLinear` stages

### `forward(x, h0=None, return_hidden=False)`

**Parameters:** `x` – `[N, C, T]`

**Returns:** `[N, output_size or hidden_size, T]`, or `(output, h)` if
`return_hidden=True`.

## Wiring

`GroupedLinear` and `SqueezedGRU` are used by
[`multiframe.DeepFilterDecoder`](multiframe.md) to predict per-frequency-bin
deep-filtering coefficients cheaply. `TAC`, `GroupedGRULayer`, and
`GroupedGRU` currently have no caller in the backbone library.

## Example

```python
from puresound.nnet.lobe.group_op import TAC, GroupedGRULayer

tac = TAC(input_dim=64, hidden_dim=128)
gru = GroupedGRULayer(input_size=128, hidden_size=128, groups=4)

out = tac(multi_channel_features)  # [N, G, 64, T]
out = gru(features)                # [N, 128, T]
```
