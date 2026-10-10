# puresound.nnet.skim

繁體中文版本：[skim.zh-TW.md](skim.zh-TW.md)

SkiM, the skipping-memory LSTM (Li et al., "SkiM: Skipping Memory LSTM for
Low-Latency Real-Time Continuous Speech Separation", ICASSP 2022;
[ESPnet reference implementation](https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/skim.py)).
Like [DPRNN](dprnn.md) it cuts a 1-D sequence into chunks, but it has no
inter-chunk LSTM over all chunks. Each block is a per-chunk `SegLSTM`; between
blocks a `MemLSTM` runs over the chunks' final `(h, c)` states and hands the
result to the next block as its initial state.

## Class: `SegLSTM`

An LSTM run inside each chunk, with the chunks as the batch.

```python
SegLSTM(input_size: int, hidden_size: int, causal: bool = True, dropout: float = 0.0)
# forward(x [NS, K, C], h [D, NS, H] | None, c [D, NS, H] | None)
#   -> (x_out [NS, K, C], h [D, NS, H], c [D, NS, H])
```

`causal=True` makes the LSTM unidirectional (`D = 1`), `False` bidirectional
(`D = 2`). `h` and `c` default to zeros. The output is
`x + LayerNorm(Linear(Dropout(LSTM(x))))`. `streaming_forward` takes the same
arguments and steps the LSTM one frame at a time along `K`.

## Class: `MemLSTM`

Refines the `SegLSTM` states across chunks between two blocks.

```python
MemLSTM(hidden_size: int, causal: bool = True, dropout: float = 0.0)
```

Its input width is `hidden_size` when `causal` and `2 * hidden_size`
otherwise, matching the `D * hidden_size` of a unidirectional or bidirectional
`SegLSTM` state.

```python
forward(
    h: Tensor,                      # [N, S, D, H] SegLSTM hidden states, one per chunk
    c: Tensor,                      # [N, S, D, H] SegLSTM cell states
    h_states: Optional[Tuple[Tensor, Tensor]] = None,  # initial state of h_net
    c_states: Optional[Tuple[Tensor, Tensor]] = None,  # initial state of c_net
    return_all: bool = False,       # also return h_net / c_net final states
    streaming: bool = False,        # skip the causal shift
)
# -> (h [D, N*S, H], c [D, N*S, H]), plus (h_net state, c_net state) with return_all
```

Two LSTMs (`h_net`, `c_net`) run along the chunk axis `S`, each followed by a
projection, LayerNorm and a residual add, and the result is folded back to the
`[D, N*S, H]` layout `nn.LSTM` takes as a state. When `causal` and not
`streaming`, the states are shifted by one chunk within each row (chunk 0 of
every row gets zeros), so the state that seeds chunk `i` in the next block
summarises chunks before `i` and not chunk `i` itself.

`streaming_forward(h, c, h_states=None, c_states=None, return_all=False)` runs
`h_net`/`c_net` one chunk at a time with batch size 1, carrying their state,
and applies neither dropout nor the shift.

## Class: `SkiM`

### Constructor

```python
SkiM(
    input_size: int,                   # input channels C
    hidden_size: int,                  # LSTM hidden width H
    output_size: int,                  # output channels
    n_blocks: int = 2,                 # SegLSTMs; n_blocks - 1 MemLSTMs between them
    seg_size: int = 20,                # chunk length K
    seg_overlap: bool = False,         # True: 50% overlap; False: contiguous chunks
    causal: bool = True,               # every SegLSTM and MemLSTM unidirectional
    embed_dim: int = 0,                # condition width; 0 disables conditioning
    embed_norm: bool = False,          # L2-normalise embed
    embed_fusion: Optional[str] = None,  # "film" | "gate" (case-insensitive)
    block_with_embed: Optional[List] = None,  # per block: condition or not
    dropout: float = 0.0,              # inside every SegLSTM / MemLSTM
)
```

- `seg_size` / `seg_overlap` behave as in DPRNN; SkiM carries its own
  `split`/`merge` methods (half-stride split, averaging merge).
- When `embed_dim != 0`, `embed_fusion` and `block_with_embed` are required.
  `"film"` builds a [`FiLM`](lobe/trivial.md); `"gate"` builds a
  [`Gate`](lobe/trivial.md) with a fixed internal width of 128, independent of
  `hidden_size`. Conditioning is applied to the chunk tensor before each
  flagged block's `SegLSTM`.

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x:     [N, input_size, T], or [N, 2, F, T] (see below)
# embed: [N, embed_dim]
# returns [N, output_size, T], or [N, 2, F, T]
```

A 4-D `[N, 2, F, T]` input, the real/imaginary tensor of a complex-mask front
end (`feats_type: complex`, see [features](features.md)), is folded to
`[N, 2F, T]`, run through the stack, and unfolded back to `[N, 2, F, T]`, so
`input_size` and `output_size` must both equal `2 * F`.

Per block: optional conditioning, then `seg_lstm[i]` initialised from the
previous `MemLSTM` output (zeros for block 0); for every block but the last,
the returned `(h, c)` are reshaped to `[N, S, D, H]` and passed through
`mem_lstm[i]`. After the last block the chunks are merged back to `T` frames
and `output_fc` (`PReLU` then a 1x1 `Conv1d`) maps to `output_size`.

### Example

```python
from puresound.nnet import SkiM

# causal, half-overlap chunks, no conditioning
model = SkiM(512, 256, 512, 4, 32, causal=True, seg_overlap=True)
y = model(torch.rand(1, 512, 1000))  # [1, 512, 1000]

# Gate conditioning on every block
model = SkiM(
    512, 256, 512, 4, 32,
    causal=True, seg_overlap=False,
    embed_dim=192, embed_norm=True,
    block_with_embed=[1, 1, 1, 1], embed_fusion="Gate",
)
y = model(torch.rand(1, 512, 1000), torch.rand(1, 192))
```

## Design notes

- Long-range context travels only through the per-chunk states that `MemLSTM`
  refines, so the cost across chunks is one small LSTM over `S` states rather
  than a full LSTM over every in-chunk position.
- The causal shift in `MemLSTM` keeps the model causal across chunks: the
  state a chunk starts from never includes that chunk's own content.
