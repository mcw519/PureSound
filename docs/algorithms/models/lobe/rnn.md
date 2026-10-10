# puresound.nnet.lobe.rnn

繁體中文版本：[rnn.zh-TW.md](rnn.zh-TW.md)

Sequence blocks: `SingleRNN`, a one-layer RNN/LSTM/GRU with a projection back
to the input width, and `FSMN` / `ConditionFSMN`, feedforward sequential memory
blocks that model context with a depthwise convolution instead of recurrence.

## Class: `SingleRNN`

```
y = Linear(Dropout(RNN(x)))        # x: [N, C, T] -> [N, T, C] -> ... -> [N, C, T]
```

```python
SingleRNN(
    rnn_type: str,              # "RNN" | "LSTM" | "GRU", case-insensitive
    input_size: int,            # C; also the output width
    hidden_size: int,
    bidirectional: bool = False,
    dropout: float = 0.0,       # on the RNN output, before the projection
)
```

`forward(x [N, C, T]) -> [N, C, T]`. The RNN has one layer; the projection
`Linear(hidden_size * num_directions, input_size)` always maps back to
`input_size`, so the block is shape-preserving whatever `hidden_size` or
`bidirectional` is. Hidden state is not returned; streaming callers that need
it (the DPCRN streaming runner) call the inner `nn.LSTM` directly.

Used by `DPCRN`'s `DPRNNblock2D` (intra path: bidirectional LSTM over
frequency; inter path: unidirectional LSTM over time) and by `DPARN`'s inter
path.

## Class: `FSMN`

Feedforward sequential memory block with a Deep-FSMN memory skip (Zhang et al.,
"Deep-FSMN for Large Vocabulary Continuous Speech Recognition", ICASSP 2018;
implementation after
[aps](https://github.com/funcwj/aps/blob/c814dc5a8b0bff5efa7e1ecc23c6180e76b8e26c/aps/asr/base/component.py#L310)):

```
p   = Conv1d_1x1(x)                                      # input_dim -> project_dim
ctx = DepthwiseConv1d(pad(p, l_context, r_context))      # kernel l_context + r_context + 1
p   = p + ctx (+ memory, if given)
out = Dropout(ReLU(Norm(Conv1d_1x1(p))))                 # project_dim -> output_dim
return out, p                                            # p is the memory for the next layer
```

```python
FSMN(
    input_dim: int,
    output_dim: int,
    project_dim: int,          # memory width
    l_context: int,            # past frames
    r_context: int,            # future frames (look-ahead)
    dilation: int = 1,         # spacing of the memory taps; padding scales with it
    dropout: float = 0.0,
    norm_type: str = "bN1d",   # a norm.get_norm name
)
```

`forward(x [N, input_dim, T], memory=None) -> (out [N, output_dim, T], memory [N, project_dim, T])`.

`memory` is a skip connection **between stacked layers**, not a state carried
across time chunks: each call zero-pads its own input, so chunked streaming
would need the caller to carry `l_context` frames of `p`, which this class does
not do. `UnetFsmn` stacks FSMN layers and passes each layer's returned memory
into the next.

## Class: `ConditionFSMN`

`FSMN` conditioned on an embedding `e` (for example a speaker vector). Same
constructor plus two arguments:

```python
ConditionFSMN(
    input_dim: int,
    output_dim: int,
    project_dim: int,
    embed_dim: int,            # width of e
    l_context: int,
    r_context: int,
    dilation: int = 1,
    dropout: float = 0,
    norm_type: str = "bN1d",
    use_film: bool = False,
)
```

- `use_film=False` – `e` is broadcast over time, concatenated to `ctx`, and
  projected back to `project_dim`: `p = p + ctx + Conv1d([ctx; e])`.
- `use_film=True` – `e` predicts a FiLM scale `γ` and bias `β` applied to both
  branches: `p = (γ p + β) + (γ ctx + β)`.

`forward(x [N, input_dim, T], embed [N, embed_dim], memory=None) -> (out, memory)`,
shapes as in `FSMN`.

## Design notes

`SingleRNN` projects back to the input width so the dual-path blocks can add
its output residually and swap it for another operator of the same interface
(attention, [`MambaInter`](ssm.md)). FSMN replaces recurrence with a
finite-context convolution: its look-ahead is exactly `r_context` frames and it
trains in parallel over time.

## Example

```python
from puresound.nnet.lobe.rnn import FSMN

layers = [FSMN(256, 256, 192, l_context=3, r_context=0) for _ in range(2)]
memory = None
for layer in layers:
    x, memory = layer(x, memory)   # memory skips from one layer to the next
```
