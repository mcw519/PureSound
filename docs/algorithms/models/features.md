# puresound.nnet.features

繁體中文版本：[features.zh-TW.md](features.zh-TW.md)

Feature transforms between the waveform encoder and the mask-predicting
backbone:
`Wav -> Encoder -> Features -> Backbone -> Apply Mask -> Decoder -> Wav`
(`system.siso.EncDecMaskBase`, `system.miso.EncDecCondMaskBase`).
`FeatureEncoder` is exported from `puresound.nnet`; `MelBank` and `WeightedSum`
are helpers imported from `puresound.nnet.features`.

## Class: `FeatureEncoder`

Every recipe builds one from its `model.features` block
(`nnet.FeatureEncoder(**model_dict["features"])` in `puresound/recipes.py`). It
turns the encoder output into the representation the backbone expects and also
returns an unaugmented, unnormalized copy that the predicted mask is applied to.

### Constructor

```python
FeatureEncoder(
    feats_type: str = "complex",           # see the dispatch table
    drop_stft_first_bin: bool = True,      # drop the DC bin (complex / magnitude / log1p only)
    include_specaug: bool = False,         # SpecAugment on the backbone input
    specaug_args: Optional[Dict] = None,   # kwargs for lobe.trivial.SpecAugment
    peq_module: Optional[FrequencyEQLayer] = None,  # built by recipes.py from `freq_eq:`
    normalized_mode: Optional[str] = None, # None | per_feature | per_channel | all_feature
    trainable: bool = False,               # learnable Mel filterbank and PEQ
)
```

- `feats_type` is lower-cased and must be one of the eight values below
  (asserted).
- `drop_stft_first_bin` only acts on `complex`, `magnitude` and `log1p`. The
  `fbank*` branches always read the full `n_fft // 2 + 1` spectrum their Mel
  matrix is built for.
- `specaug_args` is passed as keyword arguments to
  [`SpecAugment`](lobe/trivial.md): `freq_mask_length`, `time_mask_length`,
  `fill_value`, and optionally `n_freq_mask`, `n_time_mask`, `prob`. It is
  required when `include_specaug=True`.
- `peq_module` is a constructed [`FrequencyEQLayer`](lobe/dsp.md).
  `FeatureEncoder` reads its `get_args`, overrides `trainable` with its own
  flag and builds a new instance, so the one `trainable` switch decides whether
  both the PEQ and the Mel filterbank learn.
- `trainable` has no effect on `complex`, `magnitude`, `log1p`, `free` and
  `shrink_channel`, which have no parameters.

### `feats_type` dispatch

| `feats_type` | Transform | Output before the channel unsqueeze |
|---|---|---|
| `complex` | optional DC drop, then `permute(0, 3, 1, 2)` | `[N, 2, F, T]` |
| `magnitude` | [`Magnitude`](lobe/trivial.md)`(drop_first=drop_stft_first_bin)` | `[N, F, T]` |
| `log1p` | `Magnitude(drop_first=..., log1p=True)` | `[N, F, T]` |
| `fbank80_16k` | `MelBank(sr=16000, n_fft=512, n_banks=80)` | `[N, 80, T]` |
| `logfbank80_16k` | `MelBank(sr=16000, n_fft=512, n_banks=80, apply_log=True)` | `[N, 80, T]` |
| `fbank128_16k` | `MelBank(sr=16000, n_fft=512, n_banks=128)` | `[N, 128, T]` |
| `free` | `nn.Identity()` | unchanged |
| `shrink_channel` | `squeeze(-1)`, applied after augmentation | trailing size-1 axis removed |

`F` is the encoder's bin count, minus one when the DC bin is dropped. The
`fbank*` types are fixed to a 16 kHz, 512-point STFT (257 bins).

`complex`, `magnitude` and `log1p` pair with the STFT front end `ConvEncDec`
([lobe/encoder](lobe/encoder.md)); the `fbank*` types feed speaker-embedding
models such as [`EcapaTdnnExtractor`](ecapa_tdnn.md); `free` and
`shrink_channel` pair with the learned time-domain front end `FreeEncDec` for
1-D backbones.

### `forward(x) -> (feats, feats_for_enhanced)`

```python
forward(x: Tensor) -> Tuple[Tensor, Tensor]
# x: [N, C, T, 2] (complex STFT, real/imag last) or [N, C, T] (unsqueezed to [N, C, T, 1])
# returns two [N, CH, C', T] tensors (a 3-D transform output gets CH = 1)
```

1. If `peq_module` is set, the PEQ runs on `x` viewed as `[N, 2, C, T]`, i.e.
   in the STFT domain.
2. The transform gives `feats_for_enhanced`.
3. `feats` is `feats_for_enhanced`, normalized if `normalized_mode` is set,
   then passed through SpecAugment if enabled (SpecAugment does nothing outside
   training).

The backbone receives `feats`. The mask is multiplied onto `feats_for_enhanced`
([nnet.masker](masker.md)), so augmentation and normalization never change the
signal being reconstructed.

### Normalization

`normalized_mode` standardizes `feats` as `(x - mean) / (std + 1e-5)` over the
mode's reduce dims (`keepdim=True`):

| `normalized_mode` | Reduce dims | Shared across | Independent per |
|---|---|---|---|
| `per_feature` | `(1, 2)` | channel + frequency | item, frame |
| `per_channel` | `1` | channel | item, bin, frame |
| `all_feature` | `(1, 2, 3)` | channel + frequency + time | item |

Any other non-`None` value raises `NameError`. `all_feature` takes statistics
over the whole utterance, so it cannot run frame by frame; `per_feature` and
`per_channel` are per frame. A checkpoint is only valid with the mode it was
trained with.

### `back_forward(x) -> Tensor`

Undoes the DC drop after masking and before the decoder: for `complex`,
`magnitude` and `log1p` with `drop_stft_first_bin=True`, it prepends a zero bin
on `dim=2`. Other types pass through. Called by
`EncDecMaskBase._spec_to_wav` and `EncDecCondMaskBase.forward`.

### Config usage

```yaml
# egs/voice_isolate/config/train_dpcrn.yaml
model:
  features:
    feats_type: complex
    drop_stft_first_bin: True
    trainable: False
    include_specaug: False
```

```yaml
# egs/speaker_embedding/conf/PS-spk-v1.yaml
model:
  features:
    feats_type: fbank80_16k
    drop_stft_first_bin: True
    trainable: False
    normalized_mode:
    include_specaug: True
    specaug_args:
      freq_mask_length: 4
      time_mask_length: 3
      fill_value: 0.0
      n_freq_mask: 3
      n_time_mask: 5
      prob: 0.5
```

With a 512-point `ConvEncDec`, the first config maps `[N, 257, T, 2]` to two
`[N, 2, 256, T]` tensors.

### Design notes

- The DC bin carries no speech and makes the frequency axis odd (257 bins).
  Dropping it leaves 256 bins, which the stride-2 CNN stacks divide cleanly;
  `back_forward` restores it for the inverse STFT.
- Returning two tensors keeps training-time perturbations (SpecAugment,
  normalization) on the predictor's input only.

## Class: `MelBank`

Magnitude spectrum times a Mel filterbank matrix. Built by `FeatureEncoder` for
the `fbank*` types.

```python
MelBank(
    sr: int = 16000,
    n_fft: int = 512,          # matrix is [n_fft // 2 + 1, n_banks]
    n_banks: int = 80,
    apply_log: bool = False,   # log(mel + 1e-8)
    utt_norm: bool = False,    # subtract the per-utterance mean over time (mean only)
    trainable: bool = False,   # filterbank as nn.Parameter instead of a buffer
)
# forward: [N, n_fft // 2 + 1, T, 2] -> [N, n_banks, T]
```

`mag = sqrt(re^2 + im^2 + 1e-8)`, `mel = mag^T @ filterbank`, then the optional
log and mean subtraction. The matrix comes from
[`lobe.stft.mel_filterbank`](lobe/stft.md). The `1e-8` inside the square root
keeps the gradient finite at exact-zero bins, which matters when a loss
backpropagates through this layer.

## Class: `WeightedSum`

Learnable weighted sum over the last axis of one stacked tensor, for combining
several representations (for example SSL layers). No recipe builds it.

```python
WeightedSum(n_samples: int, trainable: bool = True)
# forward: [..., n_samples] -> [...]   (x * w).sum(-1)
```

`w` starts at `1 / n_samples` for every entry and is a parameter when
`trainable`, otherwise a buffer. There is no softmax, so the weights are not
constrained to sum to one.
