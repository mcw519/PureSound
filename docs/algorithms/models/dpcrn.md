# puresound.nnet.dpcrn

繁體中文版本：[dpcrn.zh-TW.md](dpcrn.zh-TW.md)

DPCRN, the dual-path convolutional recurrent network (Le et al., "DPCRN:
Dual-Path Convolution Recurrent Network for Single Channel Speech
Enhancement", Interspeech 2021), built on the [`Unet`](unet.md) chassis: a
strided CNN encoder over frequency, two `DPRNNblock2D` blocks at the
bottleneck, and a transposed-CNN decoder with skip connections. It predicts a
mask in the feature domain; the voice-isolation and noise-suppression recipes
use it with a complex mask.

## Structure

```
x [N, CH, C, T]
  -> spectral_compression (optional) -> input_norm (iLN)
  -> cnn_down x (len(channels) - 1)          # each output kept as a skip
  -> band_bottleneck.to_bands (optional)
  -> DPRNNblock2D -> DPRNNblock2D            # FiLM on dvec when dvec_dim is set
  -> band_bottleneck.to_units (optional)
  -> auxiliary heads read the bottleneck     # side outputs, see below
  -> cnn_up x (len(channels) - 1)            # skip concat, or skip_conv add
       (df_head taps the second-to-last up layer)
  -> mask [N, CH, C, T]
```

`EncDecMaskBase` (`puresound/system/siso.py`) applies the mask; with
`mask_type: complex` it calls `Masker.apply_complex_mask_with_df`, which adds
the `df_head` residual when the backbone has one (see [masker](masker.md)).

## Class: `DPCRN`

### Constructor

```python
DPCRN(
    input_dim: int = 512,                  # frequency bins C of the input
    dvec_dim: Optional[int] = None,        # speaker-embedding width; enables FiLM
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 32, 32, 32, 64, 128),  # channels[0] = CH; channels[-1] = bottleneck width
    transpose_t_size: int = 2,             # time kernel of the decoder's ConvTranspose2d
    transpose_delay: bool = False,         # crop the transpose conv's extra frames from the front
    skip_conv: bool = False,               # add skips through a conv instead of concatenating
    kernel_t: Tuple = (2, 2, 2, 2, 2),     # per down layer, time axis
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (5, 3, 3, 3, 3),     # per down layer, frequency axis
    stride_f: Tuple = (2, 2, 1, 1, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 0, 0, 0),        # look-ahead frames per down layer
    rnn_hidden: int = 128,                 # hidden width of both DPRNN paths
    inter_type: str = "lstm",              # lstm | mamba | mamba_context | lstm+mamba
    mamba_args: Optional[dict] = None,     # kwargs for lobe.ssm.MambaInter
    band_bottleneck: Optional[Dict] = None,  # kwargs for lobe.banding.BandBottleneck
    intra_type: str = "lstm",              # lstm | attention
    intra_nhead: int = 4,                  # attention heads when intra_type="attention"
    mamba_context: Optional[Dict] = None,  # BandBottleneck kwargs for the time path only
    spectral_compress: bool = False,       # |X|^0.3 with phase kept, before input_norm
    vad_head: Optional[Dict] = None,       # lobe.heads.VADHead
    background_vad_head: Optional[Dict] = None,  # a second VADHead, for non-target speech
    dist_head: Optional[Dict] = None,      # lobe.heads.DistHead
    identity_head: Optional[Dict] = None,  # lobe.heads.IdentityHead
    proximity_head: Optional[Dict] = None, # lobe.heads.ProximityHead
    expose_bottleneck: bool = False,       # keep a graph-carrying pooled bottleneck
    df_head: Optional[Dict] = None,        # lobe.multiframe.DeepFilterResidualHead
)
```

The arguments from `input_dim` to `delay`, except `dvec_dim` and
`transpose_delay`, are passed to `Unet.__init__`; see [unet](unet.md) for how
they shape the encoder and decoder. The DPRNN blocks and the heads work on the
`shape_info()[0][-1]` frequency positions left after the strides; with
`input_dim: 256` and `stride_f: [2, 2, 1]` that is 64.

`delay` is the look-ahead of each down layer. With `stride_t` all 1 the model's
algorithmic look-ahead is `sum(delay)` frames: `[1, 1, 1]` at hop 160 and
16 kHz is 30 ms.

### `forward(x, dvec=None) -> Tensor`

```python
forward(x: Tensor, dvec: Optional[Tensor] = None) -> Tensor
# x:    [N, CH, C, T], or [N, C, T] (unsqueezed to [N, 1, C, T])
# dvec: [N, dvec_dim]; L2-normalised before FiLM
# returns the mask, [N, CH, C, T]
```

### Side outputs

Each forward overwrites these attributes. A disabled head leaves its attribute
`None` and adds no parameters, so a config without the key builds the same
model and loads the same checkpoint.

| attribute | enabled by | shape | content |
| --- | --- | --- | --- |
| `last_vad_logits` | `vad_head` | `[N, T]` | target speech activity logits |
| `last_background_vad_logits` | `background_vad_head` | `[N, T]` | non-target speech activity logits |
| `last_dist_preds` | `dist_head` | `[N, n_out]` | `[fg_drr_db / drr_scale, log10(fg_dist_m), log10(nearest_itf_dist_m)]` |
| `last_identity_emb` | `identity_head` | `[N, T, dim]` | L2-normalised per-frame embedding |
| `last_proximity` | `proximity_head` | `[N, T]` | relative proximity score, arbitrary units |
| `last_bottleneck_graph` | `expose_bottleneck: true` | `[N, channels[-1], T]` | bottleneck averaged over frequency, with its graph |
| `last_bottleneck` | `stash_bottleneck = True` (runtime attribute) | `[N, channels[-1], F, T]` | bottleneck, detached |
| `last_df_coefs` | `df_head` | `[N, order, 2, bins, T]` | deep-filter taps for the low band |

The head blocks are parsed by each head's own config model, which rejects
unknown keys. Keys and defaults:

| block | keys (default) |
| --- | --- |
| `vad_head`, `background_vad_head` | `enabled` (false), `hidden` (`channels[-1]`), `kernel_t` (5), `ema_taus_s` (null), `frame_rate` (100.0) |
| `dist_head` | `enabled` (false), `hidden` (128), `n_out` (3) |
| `identity_head` | `enabled` (false), `dim` (64), `kernel_t` (5) |
| `proximity_head` | `enabled` (false), `hidden` (64) |

See [lobe/heads](lobe/heads.md) for what each head computes. No head except
`df_head` changes the audio output; the losses and the evaluation read them,
and the streaming export can emit the two VAD heads' logits as side outputs.
`stash_bottleneck` is set per call by `EncDecMaskBase.forward` for the
inference presence gate; `expose_bottleneck` is a build-time flag for losses
that run their own module on the bottleneck. They are separate so a gradient
path is never handed to a consumer that expects a detached tensor.

### Bottleneck options

**`intra_type`** chooses the frequency path of each `DPRNNblock2D`. `"lstm"` is
a bidirectional LSTM across the frequency positions of each frame.
`"attention"` is DPARN's intra path: two `MhaSelfAttenLayer`s
([lobe/attention](lobe/attention.md)) with `intra_nhead` heads, the first with
sinusoidal position encoding, followed by a linear layer. Frequency has no
causal constraint, so attention sees the same context as the bidirectional
walk while replacing its sequential steps with one matrix product per layer.

**`inter_type`** chooses the time path, which is the only state the model
carries across frames:

| `inter_type` | time path |
| --- | --- |
| `lstm` | unidirectional LSTM |
| `mamba` | `MambaInter` selective state-space block in place of the LSTM |
| `lstm+mamba` | the LSTM plus a parallel `MambaInter` whose output projection starts at zero |
| `mamba_context` | `MambaInter` on perceptual bands; requires `mamba_context` |

`mamba_args` is passed to `MambaInter`: `d_state` (16), `d_conv` (4),
`expand` (2), `dt_rank`, `dt_min`, `dt_max`, `dt_init_floor`; see
[lobe/ssm](lobe/ssm.md). `lstm+mamba` equals the LSTM alone at
initialisation, so it warm-starts from an LSTM checkpoint without
re-initialising the trained context carrier.

**`band_bottleneck`** pools the bottleneck onto `n_bands` perceptual bands
before both DPRNN blocks and expands it back afterwards
([lobe/banding](lobe/banding.md)). Keys are `BandBottleneck`'s: `n_bands`
(required), `sample_rate` (16000), `f_min` (50.0), `f_max` (`sample_rate / 2`),
`scale` (`erb` or `mel`), `learnable` (false). The encoder, decoder and heads
keep the strided grid. Banding after a uniform stride cannot recover what the
stride discarded, so pair it with `stride_f: [1, 1, 1]` to band straight from
full resolution.

**`mamba_context`** takes the same keys but bands only the time path: the
intra path stays on the full grid, and the banded time-path output is
expanded back and added through the block's residual. Per-bin detail then
bypasses the lossy band round trip while the number of independent Mamba
streams drops to `n_bands`. It requires `inter_type: mamba_context` and
cannot be combined with `band_bottleneck`.

```yaml
backbone_args:
  inter_type: mamba_context
  mamba_args: {d_state: 16, d_conv: 4, expand: 1}
  mamba_context: {n_bands: 24, scale: erb, sample_rate: 16000}
```

**`df_head`** adds a deep-filtering residual on the lowest `bins` bins
(DeepFilterNet, Schröter et al., ICASSP 2022). A `DeepFilterResidualHead`
([lobe/multiframe](lobe/multiframe.md)) reads the decoder's second-to-last map
(`channels[1]` wide, one `stride_f[0]` coarser than the mask) and predicts
`order` causal complex taps per bin; the enhanced low band becomes
`M * X[t] + sum_k w_k X[t - k]`. Keys: `bins` (128; a multiple of
`stride_f[0]` and at most `input_dim`), `order` (5), `hidden` (32). The block
is read with plain `dict.get`, so any non-empty dict enables it. The head's
last layer starts at zero, so a checkpoint without it loads into a model that
computes the same output. It needs at least two decoder layers.

### Streaming

The per-frame export ([streaming/dpcrn_onnx](../../usage/streaming/dpcrn_onnx.md))
reproduces the offline forward delayed by `sum(delay)` frames, including
banding, `mamba_context`, attention intra, `mamba` inter and `df_head`. It
rejects `inter_type: lstm+mamba` and `spectral_compress: true`, and does not
feed a `dvec`. Keep `transpose_delay: false`; `true` makes the decoder read
future frames.

### `get_args`

A property holding every constructor argument as passed, so
`DPCRN(**model.get_args)` builds the same network and the original's
`state_dict()` loads into it strictly.

### Config usage

```yaml
model:
  backbone:
    type: DPCRN
    backbone_args:
      input_dim: 256
      norm_type: bN2d
      channels: [2, 32, 64, 128]
      kernel_t: [2, 2, 2]
      stride_t: [1, 1, 1]
      dilation_t: [1, 1, 1]
      kernel_f: [5, 3, 3]
      stride_f: [2, 2, 1]
      dilation_f: [1, 1, 1]
      delay: [1, 1, 1]
      rnn_hidden: 96
      vad_head: {enabled: true, hidden: 64, kernel_t: 5}
      dist_head: {enabled: true, hidden: 128}
```

## Class: `DPRNNblock2D`

One bottleneck block: an intra-frequency path and an inter-time path, each
followed by LayerNorm and a residual add, with optional FiLM conditioning
between them.

```python
DPRNNblock2D(
    input_size: int,                       # bottleneck channels CH
    hidden_size: int,                      # hidden width of both paths
    dropout: float = 0.0,
    inter_type: str = "lstm",              # lstm | mamba | mamba_context | lstm+mamba
    mamba_args: Optional[dict] = None,
    context_bottleneck: Optional[Dict] = None,  # BandBottleneck kwargs; mamba_context only
    context_freqs: Optional[int] = None,        # frequency positions it bands from
    embedding_size: Optional[int] = None,  # FiLM condition width; None disables FiLM
    fused_type: Optional[str] = None,      # "film" (case-insensitive), the only fusion
    intra_type: str = "lstm",              # lstm | attention
    intra_nhead: int = 4,
)
```

`fused_type` is compared case-insensitively. It may be `None` when
`embedding_size` is `None`; with an `embedding_size` anything but `"film"` is
refused. `DPCRN` always passes `"FiLM"`.

```python
forward(x: Tensor, intra_skip: bool = True, inter_skip: bool = True,
        embed: Optional[Tensor] = None) -> Tensor
# x: [N, CH, C, T] -> [N, CH, C, T]
```

1. Intra: reshape to `[N*T, C, CH]`, run the frequency path, LayerNorm, add the
   block input when `intra_skip`.
2. FiLM: when `embedding_size` is set and `embed` (`[N, embedding_size]`) is
   given, modulate each frequency position's time sequence
   ([lobe/trivial](lobe/trivial.md)).
3. Inter: with `mamba_context`, band to `n_bands` positions; reshape to
   `[N*C, T, CH]`, run the time path (plus the parallel SSM for
   `lstm+mamba`), LayerNorm, expand back, add the step-1 output when
   `inter_skip`.

## Design notes

- The time path is unidirectional for every `inter_type`, so the model is
  causal apart from the `delay` look-ahead. The frequency path may look both
  ways because all bins of a frame are available at once.
- The recurrent blocks run on the smallest grid, where they are cheapest; the
  convolutional stack handles local spectral structure at full resolution.
- Heads read the bottleneck after the bands are expanded back, so banding
  changes neither their shapes nor their checkpoint keys. Head attribute names
  are the checkpoint keys (`backbone.vad_head.*`).
