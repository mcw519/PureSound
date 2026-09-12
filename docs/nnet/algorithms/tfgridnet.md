# puresound.nnet.tfgridnet

繁體中文版本：[tfgridnet.zh-TW.md](tfgridnet.zh-TW.md)

Status: *library* (see [nnet index](../index.md)) — config-reachable via
`getattr(nnet, "TFGridNet")`, covered by a forward smoke test
(`test/test_backbone.py::test_tfgrid_backbone`), not used by a maintained
recipe today.

**References:**
[1] Wang et al., "TF-GridNet: Integrating Full- and Sub-Band Modeling for
Speech Separation," IEEE/ACM TASLP, 2023.
[2] "Multi-Channel Target Speaker Extraction with Refinement: The WavLab
Submission to the Second Clarity Enhancement Challenge."

## Class: `ContextFeature`

Unfolds a tensor's last axis into a local context window by concatenating
shifted copies along the channel axis. **Asymmetric by construction** — it
takes independent past/future counts, not one symmetric `context_size`.

### Constructor

```python
ContextFeature(num_right: int, num_left: int, equal_length: bool = True)
```

**Parameters:**
- `num_right` – number of **past** taps (each one step further back,
  produced by right-shifting and edge/zero-padding the front). These need no
  extra latency to compute — the data is already available.
- `num_left` – number of **future** taps (left-shifting, padding the back).
  Each one requires the input to already have `T+i` available when producing
  output frame `T` — i.e. this is where look-ahead latency comes from.
- `equal_length` – if `True`, boundary frames are edge-repeated so the
  output keeps the same length as the input (only the channel axis grows,
  `(num_right + 1 + num_left)` copies stacked). If `False`, boundaries are
  zero-padded and then the *time axis itself* is cropped by
  `num_right`/`num_left` at the two ends instead.

Verified directly (`ContextFeature.add_context` on a ramp input,
`num_right=2, num_left=1`): the channel-stacked output is
`[x delayed by 2, x delayed by 1, x, x advanced by 1]` — confirming `num_right`
taps look backward and `num_left` taps look forward, never the other way
around.

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, C, T] -> [N, C * (num_right + 1 + num_left), T] (equal_length=True)
```

Used twice inside `GridBlock`, on two different axes:
- `IntraSpectralLayer` calls it with `num_right == num_left == kernel_size // 2`
  on the **frequency** axis (symmetric — "past/future" doesn't apply; it's
  just neighboring frequency bins, no time-causality implication).
- `SubbandTemporalLayer` calls it with `num_right=n_delay,
  num_left=kernel_size - n_delay - 1` on the **time** axis, where
  `n_delay` is `GridBlock`'s causality knob (see below).

## Class: `IntraSpectralLayer`

Frequency-axis context + bidirectional LSTM, applied independently per time
frame. `IntraSpectralLayer(channels, kernel_size, hidd_size)`: builds
`kernel_size` frequency-neighbor context (`cLN` normalized), runs a
`bidirectional=True` `nn.LSTM`, projects back to `channels` with a
`ConvTranspose1d` (kernel size compensates by cropping the front), residual
add. `[N, CH, C, T] -> [N, CH, C, T]`.

## Class: `SubbandTemporalLayer`

Time-axis context + a **unidirectional** LSTM, applied independently per
frequency bin.

```python
SubbandTemporalLayer(channels: int, kernel_size: int, hidd_size: int, n_delay: int = 1)
```

`n_delay` splits the `kernel_size`-wide temporal window into `n_delay` past
taps and `kernel_size - n_delay - 1` future taps for the pre-LSTM context
(built via `ContextFeature`, see above), then a `ConvTranspose1d`
(`inter_linear`) after the LSTM is cropped with a matching offset
(`x[..., n_delay : n_delay + nframes]`) — the same "crop the front vs. crop
the back" convention `Unet`'s `transpose_delay` uses. Verified empirically
(autograd on an impulse input, `kernel_size=3`): the block's overall
look-ahead footprint into the future is `kernel_size - 1` frames **for every
`n_delay` from `0` to `kernel_size - 1` tested** — changing `n_delay` shifts
*where* that look-ahead is realized (directly in the pre-LSTM context vs.
via the post-LSTM crop), it does not by itself remove it. `n_delay` is
`GridBlock`'s `n_delay` — see the causality note under `TFGridNet` below.
`[N, CH, C, T] -> [N, CH, C, T]`.

## Class: `FullbandSelfAttention`

Multi-head self-attention across the **time** axis, with **each head
projecting the full `[CH, F]` grid** (not per-frequency-bin) — the "full-band"
half of TF-GridNet, complementing `SubbandTemporalLayer`'s per-bin LSTM.
Previously undocumented.

### Constructor

```python
FullbandSelfAttention(
    fdim: int,
    channels: int,
    channels_qk: int,
    n_head: int,
    attent_range: Optional[int] = None,
)
```

**Parameters:**
- `fdim` – frequency-bin count (for the `LayerNorm2D` inside each Q/K/V head)
- `channels` – input/output channel count; must be divisible by `n_head`
  (each head gets `channels // n_head` value channels)
- `channels_qk` – Q/K projection width **per head** (independent of
  `channels // n_head`, unlike a standard transformer's Q=K=V split)
- `n_head` – number of heads, each with its own `Conv2d -> PReLU ->
  LayerNorm2D` projections for Q, K, and V (a `ModuleList` per role, not one
  shared projection reshaped — genuinely separate weights per head)
- `attent_range` – `None` = standard causal mask (attend to the current
  frame and all past frames, `triu(diagonal=1)` set to `-inf`); an int =
  **bounded causal window**, additionally masking anything more than
  `attent_range` frames in the past (`tril(diagonal=-attent_range)` also set
  to `-inf`). Either way
  the upper triangle (future frames) is always masked — this layer never
  looks ahead, regardless of `attent_range`.

### `forward(x, return_atten_mat=False) -> Tensor | Tuple[Tensor, Tensor]`

```python
forward(x: Tensor, return_atten_mat: bool = False) -> Tensor
# x: [N, CH, C, T] -> [N, CH, C, T] (+ optional [N, n_head, T, T] attention matrix)
```

Each head flattens its own `[channels_qk or channels//n_head, F]` projection
into one vector per time step, so attention scores are computed over
**whole spectral frames** (`Q^T K / sqrt(embed_dim)`), not per frequency bin
— this is what makes it "full-band": every frame attends to other frames
using their entire spectral content at once. Softmax over the masked scores,
weighted sum of `V`, concatenated back across heads, `1x1 Conv2d` projection,
residual add.

## Class: `GridBlock`

Assembles the three layers above in a fixed order — this is one "grid" of
TF-GridNet.

```python
GridBlock(
    ch_dim: int, f_dim: int, hid_dim: int,
    kernel_t: int = 3, kernel_f: int = 3,     # kernel_f must be odd
    n_head: int = 4, approx_qk_dim: int = 8,
    n_delay: int = 0,                          # must be < kernel_t
    attent_range: Optional[int] = None,
)
# forward(x [N, ch_dim, f_dim, T], return_atten_mat=False):
#   x = intra_frame_spectral(x)   # frequency-axis BiLSTM
#   x = subband_temporal(x)       # time-axis causal LSTM, n_delay-configurable
#   x, atten_mat = fullband_self_attention(x)   # causal (windowed) full-band attention
#   returns x, or (x, atten_mat) if return_atten_mat
```

## Class: `TFGridNet`

The exported backbone (`from puresound.nnet import TFGridNet`): a 3x3
frequency-downsampling `Conv2d`, `n_block` stacked `GridBlock`s, a matching
`ConvTranspose2d` back to the input's channel/frequency resolution.

### Constructor

```python
TFGridNet(
    inp_channel_dim: int = 2,
    input_dim: int = 256,
    input_norm: str = "gLN",       # "gLN" or "LayerNorm2D"
    channel_dim: int = 32,
    lstm_dim: int = 128,
    n_block: int = 6,
    block_delay_frames: int = 0,   # GridBlock's n_delay, shared by every block
    kernel_f_size: int = 5,
    kernel_t_size: int = 5,
    f_stride: int = 4,             # frequency downsampling factor (kernel_f_size >= f_stride)
    n_head: int = 4,
    channel_qk: int = 4,
    attent_range: int = 100,       # bounded-causal attention window, see FullbandSelfAttention
)
```

**Parameters:**
- `inp_channel_dim` – input channel count (`2` for real/imag stacked)
- `input_dim` – input frequency-bin count; each `GridBlock` operates on
  `input_dim // f_stride` bins after the input conv downsamples
- `channel_dim` / `lstm_dim` – `GridBlock`'s `ch_dim` / `hid_dim`, shared by every block
- `n_block` – number of stacked `GridBlock`s
- `block_delay_frames` – `n_delay` forwarded to every block's
  `SubbandTemporalLayer` (see the causality note below)
- `kernel_f_size` / `kernel_t_size` – `GridBlock`'s `kernel_f` / `kernel_t`
- `channel_qk` – `GridBlock`'s `approx_qk_dim`

**Causality:** the input `Conv2d` and output `ConvTranspose2d` are both
built with causal (past-only) time padding/cropping (`ZeroPad2d((2, 0, 1,
1))` on the way in, `x[..., :-2]` on the way out — no added latency from
either), and `FullbandSelfAttention`'s mask never allows attending to future
frames. The **only** source of look-ahead is `SubbandTemporalLayer` inside
each `GridBlock`: at the library default `block_delay_frames=0`, every block
looks `kernel_t_size - 1` frames into the future (verified — see
`SubbandTemporalLayer` above), so `TFGridNet` is **not causal** out of the
box; there is no single flag that makes the whole stack causal (`n_delay`
only moves *where* that per-block look-ahead is realized, not whether it
exists).

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, input_dim, T] (RI-concat, unsqueezed to a 1-channel map -- only
#    valid if inp_channel_dim was set to 1 to match) or
#    [N, inp_channel_dim, input_dim, T] (RI-stack, the normal case)
# returns: same shape as x -- a mask/mapping output, not a fixed-size embedding
```

`in_conv` (downsample frequency by `f_stride`) → `n_block` `GridBlock`s →
`out_conv` (upsample frequency back to `input_dim`, then drop 2 trailing
time frames the transpose-conv over-produced).

### Example (mirrors `test/test_backbone.py::test_tfgrid_backbone`)

```python
from puresound.nnet import TFGridNet

model = TFGridNet(
    inp_channel_dim=2, input_dim=256, channel_dim=32, lstm_dim=128,
    n_block=6, block_delay_frames=0, kernel_f_size=5, kernel_t_size=5,
    f_stride=4, n_head=4, channel_qk=4, attent_range=100,
)
y = model(torch.rand(1, 2, 256, 1000))  # [1, 2, 256, 1000]
```
