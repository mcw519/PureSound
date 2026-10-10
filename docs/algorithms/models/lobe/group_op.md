# puresound.nnet.lobe.group_op

繁體中文版本：[group_op.zh-TW.md](group_op.zh-TW.md)

Group-structured layers: `TAC`, which lets a set of channel groups (for
example microphones) exchange information through their mean, and grouped
GRU / Linear layers that split the feature axis into independent slices to cut
parameters and compute.

All sequence layers here take channel-first `[N, C, T]`.

## Class: `TAC`

Transform-Average-Concatenate.

```python
TAC(input_dim: int, hidden_dim: int)
```

`forward(x [N, G, C, T]) -> [N, G, C, T]` with `C = input_dim`:

```
h_g = PReLU(Linear(input_dim -> hidden_dim)(x_g))          # each group, each frame
m   = PReLU(Linear(hidden_dim -> hidden_dim)(mean_g h_g))
y_g = PReLU(Linear(2 * hidden_dim -> input_dim)([h_g, m]))
out = x + BatchNorm1d(y)
```

The paper normalises with `GroupNorm(1, C)`, which takes statistics over the
whole sequence; `BatchNorm1d` uses running statistics at inference, so the
layer stays causal.

Reference: Luo, Chen, Mesgarani, Yoshioka, "End-to-end microphone permutation
and number invariant multi-channel speech separation", ICASSP 2020
([reference code](https://github.com/yluo42/GC3/blob/main/utility/basics.py#L28)).

## Class: `GroupedGRULayer`

`groups` independent `nn.GRU`s (`batch_first=True`), each on a contiguous
`input_size / groups` slice of the channels.

```python
GroupedGRULayer(
    input_size: int,        # divisible by groups
    hidden_size: int,       # divisible by groups
    groups: int,
    bidirectional: bool = False,
    bias: bool = True,
    dropout: float = 0.0,   # nn.GRU's inter-layer dropout; each GRU has one layer
)
```

`forward(x, h0=None, return_hidden=False)`:

| | shape |
| --- | --- |
| `x` | `[N, input_size, T]` |
| `h0` | `[groups * D, N, hidden_size / groups]`, `D = 2` if bidirectional else 1; detached before use |
| output | `[N, hidden_size * D, T]` |
| hidden (if `return_hidden`) | same layout as `h0` |

`flatten_parameters()` calls it on every inner GRU.

## Class: `GroupedGRU`

`num_layers` stacked `GroupedGRULayer`s with an optional channel shuffle
between layers, so groups exchange features across depth (as in ShuffleNet).

```python
GroupedGRU(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    groups: int = 4,
    bidirectional: bool = False,
    bias: bool = True,
    dropout: float = 0.0,
    shuffle: bool = True,   # forced False when groups == 1; not applied after the last layer
)
```

`forward(x [N, input_size, T], h0=None, return_hidden=False)` returns
`[N, hidden_size * D, T]` (`D = 2` if bidirectional else 1) and, with
`return_hidden`, the hidden states stacked as
`[num_layers * groups * D, N, hidden_size / groups]`, the layout `h0` takes.

With `bidirectional=True` and `num_layers > 1`, each later layer is built for
the `hidden_size * 2` channels the layer below emits.

## Class: `GroupedLinear`

`groups` independent `nn.Linear`s over channel slices.

```python
GroupedLinear(
    input_size: int,    # divisible by groups
    hidden_size: int,   # output width, divisible by groups
    groups: int = 1,
    shuffle: bool = True,  # fixed interleave of output channels across groups; off when groups == 1
)
```

`forward(x [N, input_size, T]) -> [N, hidden_size, T]`.

## Class: `SqueezedGRU`

A grouped linear projection to `hidden_size`, one ordinary GRU at that width,
and an optional grouped projection out.

```python
SqueezedGRU(
    input_size: int,
    hidden_size: int,                  # projection width and GRU hidden size
    output_size: Optional[int] = None, # None -> no output projection
    num_layers: int = 1,               # layers of the GRU
    linear_groups: int = 8,            # groups of both projections
)
```

```
x = ReLU(GroupedLinear(input_size -> hidden_size)(x))
x, h = GRU(x, h0)
x = ReLU(GroupedLinear(hidden_size -> output_size)(x))    if output_size
```

`forward(x [N, input_size, T], h0=None, return_hidden=False)` returns
`[N, output_size or hidden_size, T]`, plus `h [num_layers, N, hidden_size]`
with `return_hidden`.

## Use

[`multiframe.DeepFilterDecoder`](multiframe.md) uses `SqueezedGRU` for its
temporal model and `GroupedLinear` (`shuffle=False`) for its coefficient
outputs. `TAC`, `GroupedGRULayer` and `GroupedGRU` have no caller in the
backbones.

## Design notes

- Splitting a width-`C` layer into `G` groups divides its weight count by
  about `G`; the shuffle between stacked layers keeps groups from staying
  isolated.
- `SqueezedGRU` runs the recurrence at a reduced width and does the wide
  input and output mappings with grouped linears, the pattern DeepFilterNet
  uses to keep its recurrent path cheap.
- `GroupedGRULayer` detaches `h0`, so carrying state from one chunk to the
  next does not backpropagate into the previous chunk.
