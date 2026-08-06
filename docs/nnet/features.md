# puresound.nnet.features

繁體中文版本：[features.zh-TW.md](features.zh-TW.md)

Feature transforms that sit between the waveform encoder and the mask-predicting
backbone: `Wav -> Encoder -> Features -> Backbone -> Apply Mask -> Decoder -> Wav`
(see `system.siso.EncDecMaskBase` / `system.miso.EncDecCondMaskBase`). Only
`FeatureEncoder` is exported from `puresound.nnet` (`puresound/nnet/__init__.py`);
`MelBank` and `WeightedSum` are internal helpers imported directly from
`puresound.nnet.features` when needed.

## Class: `FeatureEncoder`

The real entry point of this module — every recipe builds one via
`nnet.FeatureEncoder(**model_dict["features"])` (`puresound/recipes.py`). It
converts the encoder's raw output into whatever representation the backbone
was designed for, and hands back a *second*, untouched tensor that the mask
gets multiplied against afterwards.

### Constructor

```python
FeatureEncoder(
    feats_type: str = "complex",
    drop_stft_first_bin: bool = True,
    include_specaug: bool = False,
    specaug_args: Optional[Dict] = None,
    peq_module: Optional[FrequencyEQLayer] = None,
    normalized_mode: Optional[str] = None,
    trainable: bool = False,
)
```

**Parameters:**
- `feats_type` – one of `"free"`, `"complex"`, `"magnitude"`, `"log1p"`,
  `"fbank80_16k"`, `"logfbank80_16k"`, `"fbank128_16k"`, `"shrink_channel"`
  (asserted at construction time). See table below.
- `drop_stft_first_bin` – drop the DC bin (index 0). Only affects the
  `complex` / `magnitude` / `log1p` branches; the `fbank*` branches build a
  `MelBank` that always consumes the full `n_fft//2+1` spectrum regardless of
  this flag (a learned Mel filter already down-weights the DC region through
  its lowest triangular filter, so there is nothing to drop).
- `include_specaug` – wrap the backbone-facing features with
  [`SpecAugment`](lobe/trivial.md) (`specaug_args` is splatted straight into
  its constructor, e.g. `freq_mask_length`, `time_mask_length`, `fill_value`,
  `n_freq_mask`, `n_time_mask`, `prob` — see a real config below).
- `peq_module` – an already-constructed `FrequencyEQLayer` instance (built by
  `recipes.py` from a `freq_eq:` config block). `FeatureEncoder` does not use
  it as-is: it reads `peq_module.get_args`, forces `trainable` to this
  encoder's own `trainable` flag, and builds a *new* instance from those args.
  This lets one config flag (`FeatureEncoder.trainable`) decide whether both
  the PEQ and the Mel filterbank (see below) are learnable, independent of
  how the `freq_eq:` block itself was written.
- `normalized_mode` – `None` (default, no normalization) or one of three
  standardization modes applied to the backbone-facing tensor only. See
  [normalization](#normalization) below.
- `trainable` – forwarded to the `MelBank` filterbank and to the re-built
  `peq_module` (see above). Has no effect for `complex` / `magnitude` /
  `log1p` / `free` / `shrink_channel`, which have no learnable parameters of
  their own.

### `feats_type` dispatch

| `feats_type` | Transform | Output (pre channel-unsqueeze) |
|---|---|---|
| `complex` | drop DC bin (optional) then `permute(0, 3, 1, 2)` | `[N, 2, F(-1), T]` |
| `magnitude` | [`Magnitude`](lobe/trivial.md)`(drop_first=...)` | `[N, F(-1), T]` |
| `log1p` | `Magnitude(drop_first=..., log1p=True)` | `[N, F(-1), T]` |
| `fbank80_16k` | `MelBank(sr=16000, n_fft=512, n_banks=80)` | `[N, 80, T]` |
| `logfbank80_16k` | `MelBank(..., n_banks=80, apply_log=True)` | `[N, 80, T]` |
| `fbank128_16k` | `MelBank(sr=16000, n_fft=512, n_banks=128)` | `[N, 128, T]` |
| `free` | `nn.Identity()` | unchanged |
| `shrink_channel` | `x.squeeze(-1)` (deferred, see below) | drops the trailing size-1 axis |

`complex`/`magnitude`/`log1p` are the STFT-domain front ends used with
`ConvEncDec` (see [lobe/encoder](lobe/encoder.md)); `fbank*` variants are used
with speaker-embedding recipes (`EcapaTdnnExtractor` consumes the 80-bank
output — see [algorithms/ecapa_tdnn](algorithms/ecapa_tdnn.md)); `free` /
`shrink_channel` pair with a learned time-domain front end
(`FreeEncDec`) for 1-D sequence backbones. Neither `free` nor
`shrink_channel` is exercised by any current recipe.

### `forward(x: Tensor) -> Tuple[Tensor, Tensor]`

**Parameters:**
- `x` – encoder output, `[N, C, T, 2]` (complex STFT: real/imag last axis) or
  `[N, C, T]` (learned real-valued encoder; internally unsqueezed to
  `[N, C, T, 1]`).

**Returns:** `(feats, feats_for_enhanced)`, both `[N, CH, C, T]` for the
STFT/Mel branches (a leading singleton channel axis is inserted if the
transform produced a 3-D tensor):
- `feats` – what gets passed to the backbone (`self.backbone(features)` in
  `EncDecMaskBase`). SpecAugment, when enabled, is baked into *this* tensor
  only.
- `feats_for_enhanced` – the untouched transform output. This is the tensor
  `Masker.apply_*_mask_on_reim` actually multiplies the predicted mask
  against (see [nnet.masker](masker.md)), so augmentation never corrupts the
  signal being reconstructed — only the copy the mask-predictor sees.

If `peq_module` is set, the learnable EQ runs first, on the full `x` reshaped
to `[N, 2, C, T]` (real/imag as the channel axis) — it operates in the
STFT domain, not on the raw waveform.

### Normalization

When `normalized_mode` is not `None`, `_apply_normalization` standardizes the
backbone-facing tensor: `(x - mean) / (std + 1e-5)`, with mean and std taken
over the mode's reduce dims (all `keepdim=True`, so the result keeps the input
`[N, CH, C, T]` shape):

| `normalized_mode` | Reduce dims | Statistics are shared across | Kept independent per |
|---|---|---|---|
| `per_feature` | `(1, 2)` | channel + freq | batch item, time frame |
| `per_channel` | `1` | channel | batch item, freq bin, time frame |
| `all_feature` | `(1, 2, 3)` | channel + freq + time | batch item |

Any other non-`None` string raises `NameError`.

Only `feats` (the backbone input) is normalized. `feats_for_enhanced` stays at
its original scale on purpose, because the predicted mask is multiplied back
onto *it* — normalizing it too would rescale the reconstructed signal.
SpecAugment, when enabled, is applied after normalization, on `feats`.

When `normalized_mode` is `None` the tensor passes through untouched, so the
behavior is bit-identical to a pipeline with no normalization step at all —
which is the case for every active recipe.

> **Checkpoint note.** This path was silently inert until it was fixed
> (`_apply_normalization` computed the result but never returned it).
> `egs/speaker_embedding/conf/PS-spk-v1.yaml`, `PS-spk-v1-1.yaml` and
> `egs/target_speaker_extraction/config/default_config.yaml` used to read
> `normalized_mode: all_feature` while no normalization ever ran, so their
> shipped/derived checkpoints were in fact trained on un-normalized features.
> Those configs now carry an empty `normalized_mode:` plus an explanatory
> comment, keeping the released checkpoints consistent with their config.
> Switching any of them on requires a retrain.

### `back_forward(x: Tensor) -> Tensor`

Inverts `drop_stft_first_bin` after masking, before the decoder's inverse
STFT: for `complex` / `magnitude` / `log1p`, re-pads a zero DC bin at
`dim=2` so the tensor matches the encoder's original bin count; every other
`feats_type` passes `x` through unchanged. Called from
`EncDecMaskBase._spec_to_wav` / `EncDecCondMaskBase.forward` right after the
mask has been applied.

### Example (mirrors `egs/voice_isolate/config/train_dpcrn.yaml`)

```python
from puresound.nnet import FeatureEncoder

feats = FeatureEncoder(
    feats_type="complex",
    drop_stft_first_bin=True,
    trainable=False,
    include_specaug=False,
)

complex_spec = encoder(wav)                 # [N, 257, T, 2] (ConvEncDec, 512-pt FFT)
features, features_for_enhanced = feats(complex_spec)
# features            -> [N, 2, 256, T], fed to the backbone
# features_for_enhanced -> [N, 2, 256, T], multiplied by the predicted mask
```

```python
# mirrors egs/speaker_embedding/conf/PS-spk-v1.yaml
feats = FeatureEncoder(
    feats_type="fbank80_16k",
    drop_stft_first_bin=True,
    trainable=False,
    normalized_mode=None,            # see the checkpoint note above
    include_specaug=True,
    specaug_args=dict(
        freq_mask_length=4, time_mask_length=3, fill_value=0.0,
        n_freq_mask=3, n_time_mask=5, prob=0.5,
    ),
)
mel_feat, _ = feats(complex_spec)   # [N, 1, 80, T] -> squeeze(1) before EcapaTdnnExtractor
```

## Class: `MelBank`

Converts a complex STFT tensor to a Mel filterbank representation. Used
internally by `FeatureEncoder` for every `fbank*`/`logfbank*` `feats_type`;
constructing one directly is only needed for standalone experimentation.

### Constructor

```python
MelBank(
    sr: int = 16000,
    n_fft: int = 512,
    n_banks: int = 80,
    apply_log: bool = False,
    utt_norm: bool = False,
    trainable: bool = False,
)
```

**Parameters:**
- `sr` – sample rate in Hz, passed to `lobe.stft.mel_filterbank`
- `n_fft` – FFT size; the filterbank matrix is built for `n_fft // 2 + 1` linear bins
- `n_banks` – number of Mel filters (output channels)
- `apply_log` – apply `log(melspec + 1e-8)` after the filterbank matmul
- `utt_norm` – subtract the per-utterance mean over time from each Mel
  channel (mean-only; there is no variance division despite the name — see
  source below)
- `trainable` – if `True`, the filterbank matrix (`[n_fft//2+1, n_banks]`) is
  a learnable `nn.Parameter` instead of a fixed buffer

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – complex spectrum, `[N, F, T, 2]` where `F == n_fft // 2 + 1`
  (real/imag stacked on the last axis — matches the raw `ConvEncDec` output,
  *before* any DC-bin drop)

**Returns:** `[N, n_banks, T]`.

```python
spec_imag = x[..., 0]
spec_real = x[..., 1]
mag = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-8)  # unconditional epsilon,
# sqrt's gradient is Inf at 0, so any exact-zero STFT bin would poison backprop
# through this layer otherwise.
melspec = torch.matmul(mag.permute(0, 2, 1), self.filterbank)  # [N, T, n_banks]
```

## Class: `WeightedSum`

Learnable weighted combination over the **last axis** of a single stacked
tensor. Not currently constructed anywhere in the codebase (not by
`FeatureEncoder`, not by any recipe) — it exists as a building block for a
multi-representation combiner (e.g. SSL-layer-weighted-sum style features)
that nothing in this repo assembles yet.

### Constructor

```python
WeightedSum(n_samples: int, trainable: bool = True)
```

**Parameters:**
- `n_samples` – number of entries stacked along the input's last axis
- `trainable` – if `True`, the weight vector is a learnable parameter (`w`,
  initialized to `1/n_samples` each); if `False`, it is a fixed buffer at
  that same uniform-average initialization

Note the weights are plain learnable scalars — there is **no softmax
normalization**, unlike e.g. SUPERB-style layer-weighted-sum modules.

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – a single tensor with shape `[..., n_samples]` (already stacked by the
  caller — this is not a `List[Tensor]` API)

**Returns:** `(x * self.w).sum(dim=-1)`, i.e. `[...]` (last axis reduced away).
