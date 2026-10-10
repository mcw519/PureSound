# puresound.nnet.tfgridnet

繁體中文版本：[tfgridnet.zh-TW.md](tfgridnet.zh-TW.md)

Status: library — reachable from a config as `type: TFGridNet`, covered by a
forward test in `test/nnet/test_backbone.py`, not used by a maintained
recipe.

**References:**
[1] Wang et al., "TF-GridNet: Integrating Full- and Sub-Band Modeling for
Speech Separation," IEEE/ACM TASLP, 2023.
[2] Cornell et al., "Multi-Channel Target Speaker Extraction with Refinement:
The WavLab Submission to the Second Clarity Enhancement Challenge," Clarity
CEC2 workshop, 2022; arXiv:2302.07928.

A time-frequency backbone: a strided `Conv2d` reduces frequency, `n_block`
`GridBlock`s each run a frequency-axis BiLSTM, a time-axis LSTM and a full-band
self-attention over time, and a `ConvTranspose2d` restores the input shape.

## Class: `ContextFeature`

Stacks shifted copies of the last axis along the channel axis.

```python
ContextFeature(num_right: int, num_left: int, equal_length: bool = True)
# forward: [N, C, T] -> [N, C * (num_right + 1 + num_left), T]   (equal_length=True)
```

- `num_right` – past taps; the output channel blocks are ordered
  `[x delayed by num_right, ..., x delayed by 1, x, x advanced by 1, ...,
  x advanced by num_left]`.
- `num_left` – future taps; each one needs input frames beyond the current
  one, so these are what add look-ahead on a time axis.
- `equal_length=True` repeats the edge frame into the shifted positions and
  keeps length `T`. `False` shifts in zeros and crops `num_right` frames from
  the front and `num_left` from the back.

## Class: `IntraSpectralLayer`

```python
IntraSpectralLayer(channels: int, kernel_size: int, hidd_size: int)
# forward: [N, CH, F, T] -> [N, CH, F, T]
```

Per frame: `ContextFeature(kernel_size // 2, kernel_size // 2)` over the
frequency axis, `cLN`, a bidirectional LSTM across frequency, a
`ConvTranspose1d(2 * hidd_size, channels, kernel_size)` with the first
`kernel_size - 1` bins cropped, residual add. Frames are processed
independently, so it adds no latency.

## Class: `SubbandTemporalLayer`

```python
SubbandTemporalLayer(channels: int, kernel_size: int, hidd_size: int, n_delay: int = 1)
# forward: [N, CH, F, T] -> [N, CH, F, T]
```

Per frequency bin: `ContextFeature(num_right=n_delay,
num_left=kernel_size - n_delay - 1)` over time, `cLN`, a unidirectional LSTM,
a `ConvTranspose1d(hidd_size, channels, kernel_size)` cropped to
`[n_delay : n_delay + T]`, residual add.

The layer looks `kernel_size - 1` frames ahead for every `n_delay`: the
context contributes `kernel_size - n_delay - 1` future frames and the
transpose-conv crop contributes `n_delay`. `n_delay` chooses where the
look-ahead sits, not how much there is.

## Class: `FullbandSelfAttention`

Multi-head self-attention over time in which each query and key is a whole
spectral frame.

```python
FullbandSelfAttention(
    fdim: int,                     # frequency bins (LayerNorm2D size)
    channels: int,                 # in/out channels; divisible by n_head (asserted)
    channels_qk: int,              # Q/K channels per head
    n_head: int,
    attent_range: Optional[int] = None,
)
# forward(x [N, CH, F, T]) -> (x [N, CH, F, T], atten_mat [N, n_head, T, T])
```

Each head has its own `1x1 Conv2d -> PReLU -> LayerNorm2D` for Q and K
(`channels_qk` channels) and V (`channels // n_head` channels). Q, K, V are
flattened to one vector per frame (`channels_qk * F` for Q/K), then

```
A   = softmax(Q^T K / sqrt(channels_qk * F) + mask)       # [T, T] per head
out = x + proj(concat_heads(A V^T))
```

`proj` is `1x1 Conv2d -> PReLU -> LayerNorm2D`. The mask always blocks future
frames. `attent_range=None` allows all past frames; an integer `R` also blocks
frames `R` or more steps back, so each frame attends to itself and the `R - 1`
previous frames. `forward` always returns the attention matrix.

## Class: `GridBlock`

```python
GridBlock(
    ch_dim: int,
    f_dim: int,
    hid_dim: int,
    kernel_t: int = 3,
    kernel_f: int = 3,             # odd (asserted)
    n_head: int = 4,
    approx_qk_dim: int = 8,        # channels_qk of the attention
    n_delay: int = 0,              # < kernel_t (asserted)
    attent_range: Optional[int] = None,
)
# forward(x [N, ch_dim, f_dim, T], return_atten_mat=False)
#   -> x, or (x, atten_mat) when return_atten_mat
```

`IntraSpectralLayer(ch_dim, kernel_f, hid_dim)` ->
`SubbandTemporalLayer(ch_dim, kernel_t, hid_dim, n_delay)` ->
`FullbandSelfAttention(f_dim, ch_dim, approx_qk_dim, n_head, attent_range)`.

## Class: `TFGridNet`

```python
TFGridNet(
    inp_channel_dim: int = 2,      # 2 for stacked real/imag
    input_dim: int = 256,          # frequency bins
    input_norm: str = "gLN",       # "gLN" | "LayerNorm2D", after the input conv
    channel_dim: int = 32,         # GridBlock ch_dim
    lstm_dim: int = 128,           # GridBlock hid_dim
    n_block: int = 6,
    block_delay_frames: int = 0,   # GridBlock n_delay, shared by every block
    kernel_f_size: int = 5,        # GridBlock kernel_f
    kernel_t_size: int = 5,        # GridBlock kernel_t
    f_stride: int = 4,             # frequency downsampling; kernel_f_size >= f_stride (asserted)
    n_head: int = 4,
    channel_qk: int = 4,           # GridBlock approx_qk_dim
    attent_range: int = 100,
)
```

```python
forward(x: Tensor) -> Tensor
# x: [N, inp_channel_dim, input_dim, T], or [N, input_dim, T] (unsqueezed to
#    one channel, so inp_channel_dim must be 1)
# returns the same shape as x
```

- `in_conv`: `ZeroPad2d((2, 0, 1, 1))` (two past frames, one bin each side),
  `Conv2d(inp_channel_dim, channel_dim, 3x3, stride=(f_stride, 1))`, then
  `input_norm`. The blocks run on `input_dim // f_stride` bins, so
  `input_dim` should be a multiple of `f_stride`.
- `out_conv`: `ConvTranspose2d(channel_dim, inp_channel_dim, 3x3,
  stride=(f_stride, 1), padding=(1, 0), output_padding=(f_stride - 1, 0))`,
  which restores `input_dim` bins; the two extra trailing frames are cropped.

### Causality

The input and output convs use only current and past frames, the per-frame
norms (`cLN`, `LayerNorm2D`) and the attention mask are causal. Two parts are
not:

- every `SubbandTemporalLayer` looks `kernel_t_size - 1` frames ahead, so the
  stack looks `n_block * (kernel_t_size - 1)` frames ahead whatever
  `block_delay_frames` is;
- `input_norm="gLN"` normalizes with whole-utterance statistics;
  `LayerNorm2D` is per frame.

### Example

```python
from puresound.nnet import TFGridNet

model = TFGridNet(
    inp_channel_dim=2, input_dim=256, channel_dim=32, lstm_dim=128,
    n_block=6, block_delay_frames=0, kernel_f_size=5, kernel_t_size=5,
    f_stride=4, n_head=4, channel_qk=4, attent_range=100,
)
y = model(torch.rand(1, 2, 256, 1000))  # [1, 2, 256, 1000]
```

### Design notes

- Sub-band modeling (the LSTMs) and full-band modeling (attention over whole
  frames) are interleaved in every block, following [1].
- `attent_range` bounds how far back a frame can attend. The full `T x T`
  score matrix is still computed and then masked.
- Downsampling frequency by `f_stride` before the blocks cuts the number of
  per-bin LSTM sequences by the same factor.
