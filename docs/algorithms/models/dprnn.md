# puresound.nnet.dprnn

繁體中文版本：[dprnn.zh-TW.md](dprnn.zh-TW.md)

Dual-path RNN (Luo et al., "Dual-Path RNN: Efficient Long Sequence Modeling
for Time-Domain Single-Channel Speech Separation", ICASSP 2020) on a 1-D
feature sequence. The sequence is cut into chunks; each block runs an LSTM
inside every chunk (intra) and an LSTM across chunks at each in-chunk position
(inter). It is a standalone sequence model, separate from DPCRN's
[`DPRNNblock2D`](dpcrn.md), which works on a `[N, CH, C, T]` frequency grid.

## Class: `DPRNN`

### Constructor

```python
DPRNN(
    input_size: int,                   # input channels C
    hidden_size: int,                  # LSTM hidden width
    output_size: int,                  # output channels
    n_blocks: int = 2,                 # stacked intra + inter pairs
    seg_size: int = 20,                # chunk length K in frames
    seg_overlap: bool = False,         # True: 50% overlap; False: contiguous chunks
    causal: bool = True,               # unidirectional intra and inter LSTMs
    embed_dim: int = 0,                # FiLM condition width; 0 disables FiLM
    embed_norm: bool = False,          # L2-normalise embed before FiLM
    block_with_embed: Optional[List] = None,  # per block: apply FiLM or not
    embedding_free_tse: bool = False,  # embed is an enrollment feature sequence
)
```

- `seg_overlap=True` splits with [`SplitMerge`](lobe/trivial.md) at stride
  `seg_size // 2` and averages the overlaps on merge. `False` pads the sequence
  up to whole chunks, reshapes, and crops back to `T`.
- `causal` sets the direction of every LSTM in every block together: both
  intra and inter are unidirectional when `True`, both bidirectional when
  `False`. There is no separate intra/inter control.
- `block_with_embed` is a list of length `n_blocks` and is required when
  `embed_dim != 0`; only flagged blocks build a [`FiLM`](lobe/trivial.md)
  layer.

### Conditioning on `embed`

**FiLM (`embed_dim != 0`, `embedding_free_tse=False`).** `embed` is a
`[N, embed_dim]` vector such as a speaker embedding. It is L2-normalised when
`embed_norm`, repeated for every chunk, and applied by FiLM to the chunk
tensor right before the intra LSTM of each flagged block.

**Embedding-free target-speaker extraction (`embedding_free_tse=True`).**
`embed` is an enrollment feature sequence `[N, input_size, T']`. It is run
through the same intra/inter LSTMs, projections and norms without FiLM
(`_get_hidden_states`), which returns one `(h, c)` pair per block from the
inter LSTM. Those states initialise each block's inter LSTM in the main pass,
so the enrollment's recurrent state takes the place of an embedding vector.
FiLM is not applied in this mode.

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x:     [N, input_size, T]
# embed: [N, embed_dim] (FiLM) or [N, input_size, T'] (embedding_free_tse)
# returns [N, output_size, T]
```

Split into chunks, then per block: optional FiLM, intra LSTM with a linear
projection, LayerNorm and residual, inter LSTM (initialised from the
enrollment state in embedding-free mode) with a linear projection, LayerNorm
and residual. Merge back to `T` frames and apply `output_fc` (`PReLU` then a
1x1 `Conv1d` to `output_size`).

### Example

```python
from puresound.nnet import DPRNN

# causal, half-overlap chunks, no conditioning
model = DPRNN(512, 256, 512, 4, 32, causal=True, seg_overlap=True)
y = model(torch.rand(1, 512, 1000))  # [1, 512, 1000]

# FiLM on blocks 1 and 2 (0-indexed)
model = DPRNN(
    512, 256, 512, 4, 32,
    causal=True, seg_overlap=True,
    embed_dim=192, embed_norm=True, block_with_embed=[0, 1, 1, 0],
)
y = model(torch.rand(1, 512, 1000), torch.rand(1, 192))
```

## Design notes

- Chunking turns one long recurrence into two short ones: the intra LSTM sees
  `K` frames and the inter LSTM sees `T / K` chunks, so both stay short for long
  inputs.
- With `causal=True` the intra LSTM reads only earlier positions of its own
  chunk and the inter LSTM only earlier chunks, so each output frame depends
  only on current and earlier input frames.
- With `seg_overlap=True` every frame lies in two chunks and the merge averages
  the two outputs.
