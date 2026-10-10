# Multiband FDN — `puresound.audio.rir.render.multiband_fdn`

繁體中文版本：[multiband_fdn.zh-TW.md](multiband_fdn.zh-TW.md)

A deterministic, passive feedback delay network (FDN) that renders a late
reverberant tail with a prescribed RT60 per octave band. It is the late-field
engine of the `path-events-m4` backend (through the
[late coupling](rir_late_coupling.md)) and of the [spatial renderer](spatial_rir.md).
It only renders the tail; splicing it into an RIR is the coupling's job.
Policy string: `MULTIBAND_FDN_POLICY = "puresound.multiband_fdn.v2"`.

## Why an FDN

A coherent image-source renderer stops producing paths at its maximum order,
long before a long decay has finished, and its late paths are sparse. An FDN
produces a dense, exponentially decaying tail at fixed cost, and its decay per
band can be set exactly from a target RT60.

## Design

```python
design = design_multiband_fdn(
    sample_rate, target_rt60_s_by_hz,     # {octave center Hz: RT60 s}
    target_mixing_time_s=0.024, delay_line_count=16,
    delay_range_ms=None, seed=0,
)  # -> MultibandFDNDesign
```

- **Delays** (`select_prime_delay_lengths`): `delay_line_count` distinct primes
  closest to geometrically spaced targets between `max(1, 0.125·t_mix)` and
  `0.75·t_mix` ms (3–18 ms at the default 24 ms mixing time), each target
  jittered by ±2 % from the seed. Mutually prime lengths avoid common periods,
  so echoes do not line up into audible periodicity.
- **Feedback matrix** (`randomized_hadamard_matrix`): a normalized Hadamard
  matrix with seeded row/column permutation and signs. It is orthogonal, so the
  loop without gains preserves energy, and dense, so every line feeds every
  other and the echo density builds up quickly. `delay_line_count` must be a
  power of two.
- **Loop gains** (`delay_proportional_loop_gains`): for band `b` and delay
  `d_i` samples,

  ```text
  g_{b,i} = 10^(−3 · d_i / (fs · RT60_b))
  ```

  A signal circulating through line `i` loses `60 · d_i / (fs · RT60_b)` dB
  per pass, i.e. 60 dB per `RT60_b` seconds whatever the line length. Every
  gain lies in (0, 1), so each band is a contractive, passive loop.
- **Input and output** vectors are seeded ±1/√N; band weights are equal with
  unit squared norm.

Every octave center must lie below Nyquist (`valid_octave_centers`).

## Rendering

`render_multiband_fdn(design, excitation, output_gain=1.0, filter_order=4)`
runs one FDN recurrence per band (shared delays and matrix, per-band loop
gains) on the mono excitation, then band-limits each band's output and sums.
`render_multiband_fdn_impulse(design, duration_s)` does the same for a unit
impulse. The result keeps the full-band `rir`, the filtered `band_rirs` and the
raw per-band outputs.

The recurrence is computed in blocks no longer than the shortest delay, which is
sample-exact: nothing written can return to an output sooner than that delay.

**Filterbank.** Band edges are the geometric means of adjacent centers. Band `k`
is the cascade `H_0 ⋯ H_{k−1} L_k` of Butterworth high-passes and one low-pass
(the lowest band is `L_0`, the highest `H_0 ⋯ H_{n−1}`). This binary split is
power-complementary for Butterworth pairs and reaches from DC to Nyquist, so the
late energy has no spectral hole at the band edges or below `fs/2`.
`fdn_filterbank_power_response` returns the summed power response; rendering
records its ripple.

Everything is seeded: identical inputs render byte-identical tails.

## Diagnostics

`analyze_fdn_coloration(rir, sample_rate, centers_hz)` reports, per octave, the
spectral flatness (geometric over arithmetic mean power) and the 95th-percentile
to median power ratio in dB — a measure of modal coloration.
`puresound.audio.rir.metrics.analyze_multiband_late_field` checks a rendered tail
against its per-octave targets.

## Limits

The FDN realizes the targets it is given. In the renderers those targets are
Sabine predictions from the scene's materials, corrected for air absorption, so
any error in the Sabine estimate appears in the output unchanged.
