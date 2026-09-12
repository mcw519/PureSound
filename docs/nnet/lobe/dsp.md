# puresound.nnet.lobe.dsp

繁體中文版本：`dsp.zh-TW.md`

A learnable DSP-inspired layer: a trainable parametric EQ built as a fixed
cascade of biquad filters, applied directly in the frequency domain.

## Class: `FrequencyEQLayer`

Not a generic "N-band" EQ — the constructor bakes in a fixed low-shelf +
peaking-bands + high-shelf biquad cascade, with real default frequencies,
gains, and Q factors. There is no separate band-count parameter; the band
count is implicitly `len(eq_band_gain)` (default 7), and `eq_band_cutoff` /
`eq_band_q_factor` must be at least that long or index lookups in
`init_eq_weight` fail.

```python
FrequencyEQLayer(
    n_fft: int = 512,
    sample_rate: int = 16000,
    eq_band_gain: Tuple[float] = (0.5, 5.5, -3.25, -2.5, -4, -4, -4.5),
    eq_band_cutoff: Tuple[float] = (500, 1000, 1500, 2500, 3500, 5500, 6000),
    eq_band_q_factor: Tuple[float] = (0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707),
    low_shelf_gain_dB: float = 0.0,
    low_shelf_cutoff_freq: float = 80,
    low_shelf_q_factor: float = 0.707,
    high_shelf_gain_dB: float = 0.0,
    high_shelf_cutoff_freq: float = 7800,
    high_shelf_q_factor: float = 0.707,
    trainable: bool = True,
)
```

**Parameters:**
- `n_fft` – FFT length used to evaluate each biquad's frequency response; the resulting curve has `n_fft // 2 + 1` bins
- `sample_rate` – Hz, used by `get_biquad_params` to convert cutoff frequencies to normalized angular frequency
- `eq_band_gain` / `eq_band_cutoff` / `eq_band_q_factor` – one **peaking**-filter band per tuple entry (gain in dB, center frequency in Hz, Q factor); defaults give 7 bands, but real configs use different lengths (e.g. `egs/default_config.yaml` configures 8)
- `low_shelf_gain_dB` / `low_shelf_cutoff_freq` / `low_shelf_q_factor` – one fixed low-shelf stage
- `high_shelf_gain_dB` / `high_shelf_cutoff_freq` / `high_shelf_q_factor` – one fixed high-shelf stage
- `trainable` – see "Learnable Parameters" below

### `init_eq_weight()`

Builds the filter cascade at construction time:
1. Computes `(b, a)` biquad coefficients for the low-shelf, every peaking band, and the high-shelf via `get_biquad_params` (`puresound/audio/dsp.py`) — `n_eq = 2 + len(eq_band_gain)` stages total.
2. Evaluates each stage's frequency response `H = rfft(b, n_fft) / rfft(a, n_fft)`.
3. **Cascades** all stages by taking the product across stages: `H = prod(H, dim=0)` — this is a series filter chain, not a parallel band-sum EQ.
4. Takes the magnitude only (`H.abs()`), discarding phase, and reshapes to `[n_fft // 2 + 1, 1]`.

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – **a frequency-domain tensor**, shape `[..., C, T]` where `C = n_fft // 2 + 1` frequency bins and `T` is the time/frame axis — **not a raw waveform**. `puresound/nnet/features.py` feeds it a `[N, 2, C, T]` complex-STFT tensor (real/imag stacked on axis 1); a plain magnitude spectrogram `[N, C, T]` works the same way since `peq` broadcasts over any leading axes.

**Returns:** `x * self.peq`, elementwise (broadcast) per frequency bin, same shape as input.

### Learnable Parameters

Only **one** parameter tensor is actually learned — not the individual band
gains/cutoffs/Q factors, which are plain Python floats consumed once at
construction time and never touched by autograd again:

| Attribute | Shape | Learnable when |
|-----------|-------|----------------|
| `peq` | `[n_fft // 2 + 1, 1]` | `trainable=True` → `nn.Parameter`; `trainable=False` → registered buffer |

`peq` is the **already-cascaded, magnitude-only** frequency response curve
computed by `init_eq_weight()`. Training updates this curve directly (any
shape, not necessarily one that still factors into shelf/peak stages);
`eq_band_gain` etc. only control its *initial* value.

### `get_args` (property)

Returns a `Dict` of every constructor keyword and its current value —
`puresound/nnet/features.py` uses this to re-instantiate an equivalent layer
with a different `trainable` flag (`peq_module.__class__(**peq_args)`).

## Wiring

`puresound/nnet/features.py`'s `FeatureEncoder` accepts an optional
`peq_module: FrequencyEQLayer`; if given, it clones it via `get_args` and
applies it to the encoder's complex-STFT output (permuted to `[N, 2, C, T]`)
before the rest of the feature pipeline. Configured in recipes through a
`freq_eq: {type: FrequencyEQLayer, eq_args: {...}}` block (see
`egs/default_config.yaml`, `egs/noise_suppression/config/dparn.yaml`); also
re-exported as `puresound.nnet.FrequencyEQLayer`.

## Example

```python
from puresound.nnet.lobe.dsp import FrequencyEQLayer

eq_layer = FrequencyEQLayer(n_fft=1024, sample_rate=32000, trainable=True)

spec = torch.rand(1, 2, 513, 100)   # [N, 2, C=n_fft//2+1, T], e.g. complex STFT
spec_eq = eq_layer(spec)
spec_eq.sum().backward()
```
