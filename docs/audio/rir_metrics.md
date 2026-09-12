# RIR metrics — `puresound.audio.rir.metrics`

繁體中文版本：[`rir_metrics.zh-TW.md`](rir_metrics.zh-TW.md)

Acoustic measurements over impulse responses. Single entry point
`analyze_rir(signal, sample_rate, ...)` returns the full report; every metric
is also callable on its own. Algorithm-to-code overview:
[`rir_realism_algorithm.md`](rir_realism_algorithm.md) §7.

| Metric | Function |
|---|---|
| Schroeder decay | `schroeder_decay_db`, `noise_compensated_schroeder_decay_db` |
| T20/T30 | `estimate_decay_time` → `DecayEstimate` (value + fit R²) |
| Clarity C50/C80 | `clarity_db` |
| Direct-to-reverberant ratio | `compute_drr_db`, `direct_sample` |
| Spectral tilt | `spectral_tilt_db_per_octave` |
| Noise floor (Lundeby) | `estimate_noise_floor_lundeby` → `NoiseFloorEstimate` |
| Echo density / mixing time | `analyze_echo_density`, `abel_normalized_echo_density_profile`, `estimate_abel_mixing_time` |
| Octave banding | `octave_band_rir`, `valid_octave_centers`, `DEFAULT_OCTAVE_CENTERS_HZ` |
| Multiband late field | `analyze_multiband_late_field` |
| Spatial (synchronized receivers only) | `analyze_binaural_iacc`, `analyze_array_spatial_coherence`, `diffuse_field_coherence`, `interaural_cross_correlation` |

Versioned policy strings shipped in reports: `ABEL_ECHO_DENSITY_POLICY`,
`MULTIBAND_LATE_FIELD_POLICY`, `IACC_POLICY`, `DIFFUSE_FIELD_COHERENCE_POLICY`
— each is stamped into its own function's return dict, not necessarily into
`analyze_rir`'s combined report. Only `ABEL_ECHO_DENSITY_POLICY` reaches that
combined report, and only when `echo_density=True`; the other three ship
solely in their own dedicated-function reports.

## Interpretation rules

- A decay estimate is only as good as its fit: check `DecayEstimate.r_squared`
  (M6 QC's physical policy uses ≥ 0.70, `bank/qc.py`) before trusting `rt60_s`.
- `estimate_noise_floor_lundeby` needs ≥ 15 dB of dynamic range
  (`min_dynamic_range_db=15.0`); below that it reports
  `reason="insufficient_dynamic_range"` — corpora with faded tails hit this
  for reasons unrelated to measurement noise.
- Spatial metrics assume channels are synchronized receivers of one source.
  Bank items whose channels are independent source-to-mic paths are not
  evaluable this way (M6 QC marks them `not_applicable`).

## Example

```python
from puresound.audio.rir.metrics import analyze_rir, DEFAULT_OCTAVE_CENTERS_HZ

report = analyze_rir(
    rir_channel, 16000,
    octave_centers_hz=DEFAULT_OCTAVE_CENTERS_HZ,
    echo_density=True,
)
report["t20"]["rt60_s"], report["drr_db"], report["octave_bands"]["1000"]["t20_s"]
```

`octave_bands` is only present when `octave_centers_hz` is passed — the key
is omitted (not empty) when it is left at its `None` default.

Bank-level distribution comparison lives in
`puresound.audio.rir.bank.evaluation.compare_release_distributions`; the
standalone CLI is `egs/rir_generation/compare_bank_acoustics.py`.
