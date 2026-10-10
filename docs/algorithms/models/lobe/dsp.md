# puresound.nnet.lobe.dsp

繁體中文版本：[dsp.zh-TW.md](dsp.zh-TW.md)

A parametric EQ expressed as a per-bin gain on an STFT: a cascade of biquad
filters is evaluated once at construction, and its magnitude response becomes
a (optionally trainable) gain curve.

## Class: `FrequencyEQLayer`

```python
FrequencyEQLayer(
    n_fft: int = 512,                 # gain curve has n_fft // 2 + 1 bins
    sample_rate: int = 16000,
    eq_band_gain: Tuple[float] = (0.5, 5.5, -3.25, -2.5, -4, -4, -4.5),        # dB, one peaking band each
    eq_band_cutoff: Tuple[float] = (500, 1000, 1500, 2500, 3500, 5500, 6000),  # Hz, centre frequencies
    eq_band_q_factor: Tuple[float] = (0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707),
    low_shelf_gain_dB: float = 0.0,
    low_shelf_cutoff_freq: float = 80,
    low_shelf_q_factor: float = 0.707,
    high_shelf_gain_dB: float = 0.0,
    high_shelf_cutoff_freq: float = 7800,
    high_shelf_q_factor: float = 0.707,
    trainable: bool = True,           # peq is an nn.Parameter; else a buffer
)
```

The number of peaking bands is `len(eq_band_gain)`; `eq_band_cutoff` and
`eq_band_q_factor` must be at least as long.

### What it computes

For the low shelf, every peaking band and the high shelf, `(b_k, a_k)` come
from `puresound.audio.dsp.get_biquad_params`. The stages are cascaded and
only the magnitude is kept:

```
H(f) = | prod_k  rfft(b_k, n_fft)(f) / rfft(a_k, n_fft)(f) |      # [n_fft // 2 + 1]
peq  = H.view(-1, 1)                                               # [n_fft // 2 + 1, 1]
```

`forward(x [..., n_fft // 2 + 1, T]) -> x * peq`, same shape. The input is a
spectral tensor, not a waveform.

`get_args` (property) returns the constructor arguments as a dict, so the
layer can be rebuilt with a different `trainable` flag.

## Use in a recipe

A `freq_eq` block next to `encoder` builds the layer; `puresound.recipes`
passes it to `FeatureEncoder` as `peq_module`. `FeatureEncoder` rebuilds it
from `get_args` with the feature block's own `trainable` flag and applies it
to the encoder's complex STFT (as `[N, 2, F, T]`, before the DC bin is
dropped). See [features](../features.md).

```yaml
model:
  freq_eq:
    type: FrequencyEQLayer
    eq_args:
      n_fft: 512
      sample_rate: 16000
      eq_band_gain:   [2.5, 5.5, 3.25, 2.5, -2, -4, -8]
      eq_band_cutoff: [500, 1000, 1500, 2500, 3500, 5500, 7500]
```

## Design notes

- Only the cascaded gain curve `peq` is trained. The band gains, cutoffs and
  Q factors set its initial value; after that any curve shape is reachable.
- The response is magnitude-only, so the same real gain scales the real and
  imaginary parts of each bin and the layer adds no phase change.
- Evaluating the biquads once and storing the curve makes the forward pass one
  broadcast multiply.
