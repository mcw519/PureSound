# puresound.nnet.skim

繁體中文版本：[skim.zh-TW.md](skim.zh-TW.md)

Status: *library* (see [nnet index](../index.md)) — config-reachable via
`getattr(nnet, "SkiM")`, covered by a forward smoke test
(`test/test_backbone.py::test_skim_backbone`), not used by a maintained
recipe today.

**Reference:** Li et al., "SkiM: Skipping Memory LSTM for Low-Latency
Real-Time Continuous Speech Separation," ICASSP 2022.
([espnet reference implementation](https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/skim.py))

SkiM replaces [`DPRNN`](dprnn.md)'s inter-chunk LSTM (one recurrence over
*all* segments, every block) with a much cheaper design: each block is an
independent per-segment `SegLSTM`, and a small `MemLSTM` bridges each
block's hidden/cell state to the *next* block — "skipping" the
full-sequence inter-chunk pass entirely. This is the low-latency,
streaming-friendly trick the paper title refers to.

## Class: `SegLSTM`

Per-segment LSTM. Every segment in the batch is processed independently
(segments become the batch dimension), taking and returning an explicit
hidden/cell state so segments can be threaded together — or not — by the
caller.

```python
SegLSTM(input_size: int, hidden_size: int, causal: bool = True, dropout: float = 0.0)
# forward(x [NS, K, C], h [D, NS, C] or None, c [D, NS, C] or None)
#   -> (x_out [NS, K, C], h [D, NS, C], c [D, NS, C])
```

`causal=True` makes the internal `nn.LSTM` unidirectional (`D=1`); `False`
makes it bidirectional (`D=2`). `h`/`c` default to zeros when `None`.
`streaming_forward` is the same recurrence unrolled one frame at a time
(loops over `K`), for exact-parity online inference.

## Class: `MemLSTM`

Bridges `SegLSTM` states across segments *between* two `SkiM` blocks.

```python
MemLSTM(hidden_size: int, causal: bool = True, dropout: float = 0.0)
```

**Parameters:**
- `hidden_size` – must match the enclosing `SegLSTM`'s `hidden_size`
- `causal` – as in `SegLSTM`; also sets `input_size = hidden_size if causal
  else 2 * hidden_size`, because the incoming `(h, c)` states are
  `[N, S, D, hidden_size]` with `D = 1` (causal `SegLSTM`) or `D = 2`
  (bidirectional `SegLSTM`) — `MemLSTM`'s own LSTM runs *over the segment
  axis S*, so its input width has to match whatever `D * hidden_size`
  the folded `SegLSTM` state actually is
- `dropout` – shared by the internal `h_net`/`c_net` LSTMs

### `forward(h, c, h_states=None, c_states=None, return_all=False, streaming=False)`

```python
forward(
    h: Tensor,   # [N, S, D, C] -- SegLSTM hidden states, one per segment
    c: Tensor,   # [N, S, D, C] -- SegLSTM cell states
    h_states: Optional[Tuple[Tensor, Tensor]] = None,  # MemLSTM's own LSTM
    c_states: Optional[Tuple[Tensor, Tensor]] = None,  # recurrent state (streaming only)
    return_all: bool = False,
    streaming: bool = False,
) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, tuple, tuple]
# h, c returned as [D, N*S, C] -- ready to feed straight into the next SegLSTM
```

Runs two independent LSTMs (`h_net`, `c_net`) *along the segment axis*,
residual + `LayerNorm`, then folds back to the `[D, N*S, C]` shape
`nn.LSTM` expects for a state tuple. When `causal=True`, the result is
additionally shifted by one segment (`h_[:, 1:, :] = h[:, :-1, :]`,
segment 0 zeroed) — even though `h_net`/`c_net` are already unidirectional,
without this shift segment `i`'s refined state would still be a function of
segment `i`'s own `SegLSTM` output, which is circular (that state is meant
to *seed* segment `i`'s processing in the next block, not summarize it).
The shift is applied whenever `streaming=False`, so a single batched offline
forward gives bit-identical results to `streaming_forward`'s
one-segment-at-a-time loop. Pass `return_all=True` during streaming to also
get `h_net`/`c_net`'s own final recurrent state, for continuing across the
next chunk of segments.

## Class: `SkiM`

The exported backbone (`from puresound.nnet import SkiM`).

### Constructor

```python
SkiM(
    input_size: int,
    hidden_size: int,
    output_size: int,
    n_blocks: int = 2,
    seg_size: int = 20,
    seg_overlap: bool = False,
    causal: bool = True,
    embed_dim: int = 0,
    embed_norm: bool = False,
    embed_fusion: Optional[str] = None,
    block_with_embed: Optional[List] = None,
    dropout: float = 0.0,
)
```

**Parameters:**
- `input_size` / `hidden_size` / `output_size` – as in `DPRNN`
- `n_blocks` – number of `SegLSTM`s; `n_blocks - 1` `MemLSTM`s bridge them
  (the last block has nothing to bridge into)
- `seg_size` / `seg_overlap` – chunking, same semantics as
  [`DPRNN`](dprnn.md) (`seg_overlap` is a bool: half-overlap split/merge vs.
  contiguous reshape) — but `SkiM` does **not** reuse
  `lobe.trivial.SplitMerge`; it has its own near-identical `split`/`merge`
  instance methods
- `causal` – forwarded to every `SegLSTM` and `MemLSTM` uniformly (same
  single-flag-controls-everything pattern as `DPRNN`)
- `embed_dim` / `embed_norm` / `block_with_embed` – as in `DPRNN`: which
  blocks (by index) FiLM/Gate-condition their segment input on `embed`
- `embed_fusion` – `"film"` ([`FiLM`](../lobe/trivial.md)) or `"gate"`
  ([`Gate`](../lobe/trivial.md), whose bottleneck width is a hardcoded
  `hidden_size=128` **unrelated to** `SkiM`'s own `hidden_size` argument —
  a naming collision, not a shared value)
- `dropout` – shared by every `SegLSTM`/`MemLSTM`

There is no `mem_size` parameter and no memory-bank concept — `MemLSTM`
carries exactly one `(h, c)` pair per segment, sized by `hidden_size`.

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
```

`x` accepts two shapes:
- `[N, input_size, T]` — the general 1-D sequence case, identical to `DPRNN`.
- `[N, 2, F, T]` — a complex-mask front end's real/imag tensor
  (`feats_type="complex"`, see [nnet.features](../features.md)). SkiM is a
  1-D sequence model, so it folds `(2, F)` into the channel axis first
  (`input_size` must equal `2 * F`), runs the sequence stack, then unfolds
  the output back to `[N, 2, F, T']` so it can be consumed as a complex mask.
  `DPRNN` has no equivalent auto-fold — this reshape is SkiM-specific.

Per block: optional FiLM/Gate conditioning on the segment tensor → run
`seg_lstm[i]` (seeded by the previous block's `MemLSTM` output, or zeros for
block 0) → if not the last block, reshape the returned `(h, c)` into
`[N, S, D, hidden_size]` and run `mem_lstm[i]` to produce the next block's
seed state. After the last block: merge chunks back to `T` frames →
`output_fc` (`PReLU` then `Conv1d` to `output_size`).

### Example (mirrors `test/test_backbone.py::test_skim_backbone`)

```python
from puresound.nnet import SkiM

# Causal, half-overlap chunking, no conditioning
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
