# puresound.nnet.dparn

繁體中文版本：[dparn.zh-TW.md](dparn.zh-TW.md)

Status: *active* (see [nnet index](../index.md)). Dual-Path Attention RNN
(DPARN) — the same `Unet` chassis as [DPCRN](dpcrn.md), but the bottleneck
replaces DPCRN's bidirectional intra-frequency LSTM with two stacked
self-attention layers, keeping DPCRN's unidirectional inter-time LSTM.

## Class: `DPARNblock2D`

The bottleneck block.

### Constructor

```python
DPARNblock2D(
    input_size: int,
    hidden_size: int,
    nhead: int,
    dropout: float = 0.0,
)
```

**Parameters:**
- `input_size` – bottleneck channel dimension (`channels[-1]` from the
  enclosing `Unet`/`DPARN`)
- `hidden_size` – hidden width for the inter-chunk LSTM
- `nhead` – attention heads for the two intra-chunk `MhaSelfAttenLayer`s
  (no default — always supplied by `DPARN`)
- `dropout` – shared by both attention layers and the LSTM

There is no `bidirectional` or `causal` constructor argument. Both
intra-chunk attention layers are hardcoded `bidirectional=False` (the
`MhaSelfAttenLayer`'s LSTM-replacement flag, unrelated to sequence
direction) and every call passes `causal=False` at the `forward` call site,
not as stored construction state — this block does not currently expose a
causal-streaming switch.

### Architecture

- **Intra-chunk (frequency axis, per time frame):** two stacked
  `MhaSelfAttenLayer`s (see [lobe/attention](../lobe/attention.md)) —
  the first with sinusoidal position encoding, the second without — followed
  by a `Linear` + `LayerNorm`, replacing DPRNN/DPCRN's intra-frequency BiLSTM
  with self-attention.
- **Inter-chunk (time axis, per frequency bin):** one unidirectional
  `SingleRNN("LSTM", ...)` + `LayerNorm`, identical in structure to
  [`DPRNNblock2D`](dprnn.md)'s inter path.

### `forward(x, intra_skip=True, inter_skip=True) -> Tensor`

```python
forward(
    x: Tensor,             # [N, CH, C, T]
    intra_skip: bool = True,
    inter_skip: bool = True,
) -> Tensor                # [N, CH, C, T]
```

Each path is a residual add around itself (`intra_skip`/`inter_skip` toggle
whether that residual is actually added); `DPARN` always calls this with
both left at their default `True`.

## Class: `DPARN`

Extends `Unet` (see [algorithms/unet](unet.md)) exactly like `DPCRN` does:
same CNN down/up stack, bottleneck swapped in.

### Constructor

```python
DPARN(
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 32, 32, 32, 64, 128),
    transpose_t_size: int = 2,
    transpose_delay: bool = False,
    skip_conv: bool = False,
    kernel_t: Tuple = (2, 2, 2, 2, 2),
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (5, 3, 3, 3, 3),
    stride_f: Tuple = (2, 2, 1, 1, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 0, 0, 0),
    n_dparn_block: int = 2,
    rnn_hidden: int = 128,
    nhead: int = 1,
    spectral_compress: bool = False,
)
```

The first 15 parameters (`input_dim` … `delay`) are forwarded verbatim to
`Unet.__init__` — see [algorithms/unet](unet.md) for what each one does to
the CNN down/up stack. DPARN-specific:
- `n_dparn_block` – number of stacked `DPARNblock2D`s at the bottleneck
  (DPCRN hardcodes exactly 2; DPARN makes this configurable)
- `rnn_hidden` – `hidden_size` passed to every `DPARNblock2D`
- `nhead` – `nhead` passed to every `DPARNblock2D`
- `spectral_compress` – if `True`, applies
  `spectral_compression(x, alpha=0.3, dim=1)` (magnitude raised to the power
  `alpha`, phase preserved — see [lobe/trivial](../lobe/trivial.md)) to the
  input before anything else

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, CH, C, T] or [N, C, T] (unsqueezed to [N, 1, C, T])
# returns: [N, CH, C, T]
```

`spectral_compress` (optional) → `Unet.input_norm` → CNN-down (collecting
skip connections) → `n_dparn_block` stacked `DPARNblock2D`s → CNN-up
(skip-concat or `skip_conv`, matching `Unet`'s up path exactly, including the
`transpose_delay` cropping convention — see [algorithms/unet](unet.md)).

### `get_args` property

Returns a `Dict` of constructor arguments for checkpoint reconstruction.
Complete — every constructor parameter, including `nhead` and
`spectral_compress`, is stored on `self` and returned, so
`DPARN(**model.get_args)` rebuilds an architecturally identical model.

> **Fixed in this pass:** `get_args` used to omit both `nhead` (which was
> never stored on `self` at all, so it could not have been recovered even if
> listed) and `spectral_compress`. Rebuilding from saved args silently reset
> `nhead` to its default `1` and dropped whatever `spectral_compress` was.

### Example (mirrors `test/test_backbone.py::test_dparn_backbone`)

```python
from puresound.nnet import DPARN

model = DPARN(
    input_dim=256,
    norm_type="cLN",
    channels=(2, 32, 32, 32, 64, 128),
)
input_x = torch.rand(1, 2, 256, 200)
y = model(input_x)
assert input_x.shape == y.shape
```

## Streaming ONNX Runtime

DPARN can be exported as a feature-frame ONNX model for low-latency
inference. The streaming path keeps the offline model unchanged and wraps
the DPARN backbone with explicit frame state:

- CNN temporal caches for the downsampling path
- transpose-convolution pending caches for the upsampling path
- LSTM hidden and cell states for each `DPARNblock2D`

The ONNX model consumes one complex STFT frame at a time:

```python
enhanced_frame, next_state = forward_frame(noisy_frame, state)
```

`noisy_frame` and `enhanced_frame` have shape `[batch, 257, 2]`. Audio
buffering, Hann STFT, iSTFT, and overlap-add are handled outside ONNX by
`puresound.streaming.StreamingDparnOrt`.

See [DPARN Streaming ONNX Runtime](../../streaming/dparn_onnx.md) for the
supported config, export command, runtime API, and Gradio demo workflow.
