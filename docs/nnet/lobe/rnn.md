# puresound.nnet.lobe.rnn

繁體中文版本：`rnn.zh-TW.md`

Recurrent building blocks: a unified RNN/LSTM/GRU wrapper with output
projection, and FSMN — a convolution-based recurrence alternative with
explicit, stateful memory chaining for streaming.

## Class: `SingleRNN`

Wraps `nn.RNN`/`nn.LSTM`/`nn.GRU` (single layer) and always projects the
result back to `input_size`, so the block is shape-preserving regardless of
`hidden_size` or `bidirectional` — there is **no `proj_size` parameter** to
project to some *other* width; the projection target is unconditionally
`input_size`.

```python
SingleRNN(
    rnn_type: str,
    input_size: int,
    hidden_size: int,
    bidirectional: bool = False,
    dropout: float = 0.0,
)
```

**Parameters:**
- `rnn_type` – `"RNN"`, `"LSTM"`, or `"GRU"` (case-insensitive; matched via `getattr(nn, rnn_type.upper())`)
- `input_size` – input feature dimension — also always the **output** dimension, via the trailing `nn.Linear(hidden_size * num_directions, input_size)`
- `hidden_size` – RNN hidden dimension
- `bidirectional` – if `True`, `num_direction = 2` and the projection input width doubles accordingly
- `dropout` – applied to the RNN output before the projection (not PyTorch's own inter-layer `dropout`, since there's always exactly 1 layer here)

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, C, T]` (`C = input_size`)

Permutes to `[N, T, C]`, runs the RNN, dropout, projects back to `input_size`,
permutes back. **Returns:** `[N, C, T]`, **always** `C = input_size` — same
shape as the input, never `hidden_size` or a custom projection width.

---

## Class: `FSMN`

**Feedforward Sequential Memory Network** — a dilated-depthwise-Conv1d
alternative to recurrence, with **explicit, stateful memory chaining**
between layers/calls rather than an internal hidden state.

```python
FSMN(
    input_dim: int,
    output_dim: int,
    project_dim: int,
    l_context: int,
    r_context: int,
    dilation: int = 1,
    dropout: float = 0.0,
    norm_type: str = "bN1d",
)
```

**Parameters:**
- `input_dim` – input feature dimension
- `output_dim` – output feature dimension
- `project_dim` – hidden "memory" width — the depthwise context conv and the memory tensor both live at this width
- `l_context` / `r_context` – past / future context frames; the depthwise conv kernel size is `l_context + r_context + 1`
- `dilation` – dilation of the depthwise context conv
- `dropout` – applied after `norm_type` normalization, at the very end
- `norm_type` – a [`norm.get_norm`](norm.md) code (default `"bN1d"`)

**Reference:** [aps FSMN component](https://github.com/funcwj/aps/blob/c814dc5a8b0bff5efa7e1ecc23c6180e76b8e26c/aps/asr/base/component.py#L310).

### `forward(x, memory=None) -> Tuple[Tensor, Tensor]`

This is **not** a plain `forward(x) -> Tensor`. It threads a memory tensor
through the call, matching the recurrent-network calling convention used
elsewhere in this codebase (`(out, new_state) = layer(x, state)`), which
matters for streaming inference — each chunk's call must pass in the
previous chunk's returned memory:

```python
in_proj = Conv1d(input_dim, project_dim, kernel_size=1)(x)
ctx     = depthwise_conv(pad(in_proj, l_context, r_context))
proj    = in_proj + ctx
if memory is not None:
    proj = proj + memory              # chain in the caller-supplied memory
out     = norm(Conv1d(project_dim, output_dim, kernel_size=1)(proj))
return out, proj                      # `proj` is the new memory for the *next* call
```

**Parameters:**
- `x` – `[N, C, T]` (`C = input_dim`)
- `memory` – previous memory block, `[N, P, T]` (`P = project_dim`), or `None` for the first call in a sequence/stream

**Returns:** `(out, new_memory)` — `out`: `[N, output_dim, T]`; `new_memory`:
`[N, project_dim, T]`, to be passed back in as `memory` on the next call
(e.g. the next chunk, or the next FSMN layer in a stack — see `Unet`'s usage
below, which chains memory *across stacked FSMN layers within one forward
pass*, not just across time chunks).

---

## Class: `ConditionFSMN`

`FSMN` plus FiLM-style or concatenative conditioning from an external
embedding (e.g. speaker vector). Subclasses `FSMN` and reuses all of its
constructor parameters, adding two:

```python
ConditionFSMN(
    input_dim: int,
    output_dim: int,
    project_dim: int,
    embed_dim: int,
    l_context: int,
    r_context: int,
    dilation: int = 1,
    dropout: float = 0,
    norm_type: str = "bN1d",
    use_film: bool = False,
)
```

**Parameters:**
- `embed_dim` – conditioning embedding dimension
- `use_film` – if `False` (default), the embedding is broadcast-concatenated onto the context branch and projected back down with a `Conv1d`; if `True`, the embedding instead predicts a FiLM `(scale, bias)` pair applied to both `proj` and `ctx` before they're summed
- all other parameters – as in `FSMN`

### `forward(x, embed, memory=None) -> Tuple[Tensor, Tensor]`

**Parameters:**
- `x` – `[N, C, T]`
- `embed` – `[N, embed_dim]`
- `memory` – as in `FSMN.forward`

**Returns:** `(out, new_memory)`, same shapes as `FSMN.forward`.

## Wiring

`DPARN`'s dual-path block (`puresound/nnet/dparn.py`) uses `SingleRNN("LSTM",
...)` for inter-chunk modeling (paired with
[`attention.MhaSelfAttenLayer`](attention.md) for intra-chunk modeling).
`DPCRN` also uses `SingleRNN` for its intra/inter RNN stages. `Unet`'s
FSMN-augmented variant (`puresound/nnet/unet.py`) stacks `FSMN`/`ConditionFSMN`
layers and threads `memory` from one layer's output into the next layer's
input — chaining memory *down the stack*, the same mechanism a streaming
caller would use to chain memory *across time chunks*.

## Example

```python
from puresound.nnet.lobe.rnn import FSMN

fsmn = FSMN(input_dim=256, output_dim=256, project_dim=192, l_context=3, r_context=3)

memory = None
out1, memory = fsmn(chunk1, memory)   # memory=None on the first chunk
out2, memory = fsmn(chunk2, memory)   # carries state forward
```
