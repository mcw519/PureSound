# puresound.nnet.lobe.banding

繁體中文版本：[banding.zh-TW.md](banding.zh-TW.md)

Pools a bottleneck's uniform frequency grid onto perceptual (ERB or mel) bands
and expands it back, so a recurrent block runs on fewer frequency positions,
dense at low frequencies and sparse at high ones, while everything around it
keeps the full grid.

## Scales and band edges

```python
erb_rate(hz)             # 21.4 * log10(1 + 0.00437 * hz)   (Glasberg & Moore ERB number)
erb_rate_to_hz(rate)
mel_rate(hz)             # 2595 * log10(1 + hz / 700)
mel_rate_to_hz(rate)

band_edges_hz(n_bands: int, f_min: float, f_max: float, scale: str = "erb") -> Tensor
# n_bands + 1 edges, equally spaced on `scale`, returned in Hz.
# ValueError unless scale in {"erb", "mel"}, n_bands >= 1 and 0 <= f_min < f_max.
```

## Function: `triangular_band_matrix`

```python
triangular_band_matrix(
    n_bands: int, n_units: int, *, f_min: float, f_max: float, scale: str = "erb"
) -> Tensor   # [n_bands, n_units]
```

Unit `u` is taken to sit at `f_min + (u + 0.5) * (f_max - f_min) / n_units`.
Band `b` spans edges `[e_b, e_{b+1}]` with centre `c_b`; its weight rises
linearly from `e_{b-1}` to `c_b` and falls to `e_{b+2}`, so each band overlaps
its neighbours (the outermost bands extend by mirroring). Each row is
normalised to sum to 1. A band narrower than one unit puts weight 1 on its
nearest unit, so no row is empty.

## Class: `BandBottleneck`

```python
BandBottleneck(
    n_units: int,               # frequency positions of the incoming grid
    n_bands: int,               # must be <= n_units, else ValueError
    *,
    sample_rate: int = 16000,
    f_min: float = 50.0,
    f_max: float | None = None, # None -> sample_rate / 2
    scale: str = "erb",         # "erb" | "mel"
    learnable: bool = False,    # make both maps trainable, initialised from the fixed ones
)
```

| method | shape |
| --- | --- |
| `to_bands(x)` | `[N, CH, n_units, T] -> [N, CH, n_bands, T]`, with `pool` `[n_bands, n_units]` |
| `to_units(x)` | `[N, CH, n_bands, T] -> [N, CH, n_units, T]`, with `expand` `[n_units, n_bands]` |

`expand` is `pool` transposed and renormalised so every unit receives total
weight 1. With `learnable=False` both maps are buffers.

## Use in a recipe

`DPCRN` builds it in two places; the two options are mutually exclusive.

- `band_bottleneck` bands the input of both `DPRNNblock2D`s and expands the
  result before the heads and the decoder, so only the dual-path blocks see
  bands.
- `mamba_context` (with `inter_type: mamba_context`) keeps the intra-frequency
  path on the full grid and bands only the inter-time Mamba path; its output
  is expanded and added back as a residual.

`n_units` is the bottleneck's frequency count, filled in by `DPCRN`; the
config gives the rest.

```yaml
backbone:
  type: DPCRN
  backbone_args:
    stride_f: [2, 2, 1]          # the grid the bands are pooled from
    band_bottleneck:
      n_bands: 32
      scale: erb                 # or mel
      sample_rate: 16000
      f_min: 50.0
      learnable: false
```

`stride_f` sets the finest a low band can be. With 256 bins at 16 kHz and
`stride_f: [2, 2, 1]` the pooled grid is 125 Hz wide, so the lowest ERB bands
are 125 Hz however narrow the scale would make them. `stride_f: [1, 1, 1]`
bands straight off the full resolution at a higher convolution cost.

## Design notes

- A uniform frequency stride removes resolution at 200 Hz, where the voice
  harmonics are, at the same rate as at 7 kHz, where there is little left.
  Perceptual bands spend the same number of positions densely at low
  frequencies and sparsely at high ones. RNNoise, PercepNet and DeepFilterNet
  group frequencies the same way.
- Banding is not a replacement for the STFT. Analysis, synthesis and the mask
  stay on the full complex grid, because band gains cannot reconstruct
  structure inside a band.
- Rows sum to 1, so pooling is an average and values do not grow with band
  width. Normalising the expansion per unit keeps units at a band edge from
  coming back quieter than units at a centre, which would act as a comb
  filter.
- Triangular, overlapping weights avoid a hard edge where a harmonic crosses
  from one band to the next as pitch moves.
- `learnable` is off by default so the band layout stays an explicit, fixed
  design choice and its effect can be attributed.
- Both maps are fixed linear maps over frequency with no state in time. The
  streaming DPCRN (`puresound/streaming/dpcrn.py`) reimplements the backbone
  forward and applies the same banding, with recurrent state sized by the band
  count; `test/streaming/test_dpcrn_streaming.py` checks it against the
  offline model.
