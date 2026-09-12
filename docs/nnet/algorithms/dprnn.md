# puresound.nnet.dprnn

繁體中文版本：[dprnn.zh-TW.md](dprnn.zh-TW.md)

Status: *library* (see [nnet index](../index.md)) — config-reachable via
`getattr(nnet, "DPRNN")`, covered by a forward smoke test
(`test/test_backbone.py::test_dprnn_backbone`), not used by a maintained
recipe today.

**Reference:** Luo et al., "Dual-Path RNN: Efficient Long Sequence Modeling
for Time-Domain Single-Channel Speech Separation," ICASSP 2020.

This is a standalone, fully-causal-configurable dual-path RNN — a different
implementation from [`DPRNNblock2D`](dpcrn.md) (the fixed-topology block
DPCRN's bottleneck hardcodes: always-bidirectional intra, always-unidirectional
inter, no `causal` knob at all). Here a single `causal` flag controls *both*
the intra-chunk and inter-chunk LSTM direction, uniformly across every block.

## Class: `DPRNN`

### Constructor

```python
DPRNN(
    input_size: int,
    hidden_size: int,
    output_size: int,
    n_blocks: int = 2,
    seg_size: int = 20,
    seg_overlap: bool = False,
    causal: bool = True,
    embed_dim: int = 0,
    embed_norm: bool = False,
    block_with_embed: Optional[List] = None,
    embedding_free_tse: bool = False,
)
```

**Parameters:**
- `input_size` / `hidden_size` / `output_size` – input feature dim, LSTM
  hidden width, output feature dim (`output_fc` is the only place the
  channel count actually changes)
- `n_blocks` – number of stacked intra+inter LSTM pairs
- `seg_size` – chunk length in time steps
- `seg_overlap` – **bool**, not a stride length: if `True`, chunks are built
  with [`SplitMerge`](../lobe/trivial.md)'s 50%-overlap split/merge
  (`seg_stride = seg_size // 2`); if `False` (the default), chunks are a
  plain contiguous reshape (`x.reshape(N, -1, seg_size, C)`, zero-padded to a
  multiple of `seg_size` and cropped back to `T` at the end) — there is no
  way to pick an arbitrary overlap amount, only "half" or "none"
- `causal` – **default `True`**. Sets `bi_direct = not causal` once and
  reuses it for *every* block's *both* LSTMs (`intra_rnn[i]` and
  `inter_rnn[i]` are both bidirectional together, or both unidirectional
  together) — there is no independent intra/inter causality control
- `embed_dim` – if not `0`, blocks flagged in `block_with_embed` FiLM-condition
  their chunked input on `embed` (see below); irrelevant when `embedding_free_tse=True`
- `embed_norm` – L2-normalize `embed` before conditioning (skipped when
  `embedding_free_tse=True`)
- `block_with_embed` – `List[int]` of length `n_blocks`; required (indexed
  unconditionally) whenever `embed_dim != 0`
- `embedding_free_tse` – switches what `embed` *means* (see next section)

### Two ways to condition on `embed`

**1. Fixed-embedding FiLM (`embed_dim != 0`, `embedding_free_tse=False`,
default):** `embed` is a `[N, embed_dim]` vector (e.g. a speaker embedding).
It is L2-normalized if `embed_norm`, then broadcast to every chunk
(`embed.repeat(1, total_seg, 1)`). Inside the loop over blocks, for each `i`
where `block_with_embed[i]` is truthy, the chunk tensor is FiLM-conditioned
(`self.input_film[i]`, a [`FiLM`](../lobe/trivial.md) layer built only for
those blocks) immediately before that block's intra-chunk LSTM.

**2. Embedding-free target-speaker extraction (`embedding_free_tse=True`):**
`embed` must instead be a 3-D enrollment **feature sequence**
`[N, input_size, T']` (asserted at the top of `forward`) — not a fixed-size
vector. It is run through `_get_hidden_states`, a second pass over the exact
same `intra_rnn`/`inter_rnn`/`proj`/`norm` weights (no FiLM, no embed
conditioning inside it) that returns each block's `inter_rnn` final
`(h, c)` state. Those per-block states then seed
`self.inter_rnn[i](inter_input, inter_hidd_init_states[i])` in the main
forward pass — i.e. the enrollment utterance's own LSTM dynamics become the
inter-chunk LSTM's initial state, standing in for a learned embedding
vector entirely. This is mutually exclusive in practice with mechanism 1:
`embedding_free_tse=True` skips the embed broadcast/reshape step that FiLM
conditioning depends on, so combining both on the same run is not a
supported configuration.

`_get_hidden_states`'s own docstring claims it returns a `[N, C, T]` tensor;
it actually returns `inter_hidd`, a `List[Tuple[Tensor, Tensor]]` (one
`(h, c)` pair per block) — the list is what `forward` consumes as
`inter_hidd_init_states`.

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x:     [N, input_size, T]
# embed: [N, embed_dim] (FiLM path) or [N, input_size, T'] (embedding_free_tse path)
# returns: [N, output_size, T]
```

Segment `x` into chunks → for each of `n_blocks`: optional FiLM → intra-chunk
LSTM (residual + `LayerNorm`) → inter-chunk LSTM, seeded by
`inter_hidd_init_states[i]` (residual + `LayerNorm`) → merge chunks back to
`T` frames → `output_fc` (`PReLU` then `Conv1d` to `output_size`).

### Example (mirrors `test/test_backbone.py::test_dprnn_backbone`)

```python
from puresound.nnet import DPRNN

# Causal, half-overlap chunking, no conditioning
model = DPRNN(512, 256, 512, 4, 32, causal=True, seg_overlap=True)
y = model(torch.rand(1, 512, 1000))  # [1, 512, 1000]

# Fixed-embedding FiLM conditioning on blocks 1 and 2 (0-indexed)
model = DPRNN(
    512, 256, 512, 4, 32,
    causal=True, seg_overlap=True,
    embed_dim=192, embed_norm=True, block_with_embed=[0, 1, 1, 0],
)
y = model(torch.rand(1, 512, 1000), torch.rand(1, 192))
```
