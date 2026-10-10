# RIR metrics — `puresound.audio.rir.metrics`

繁體中文版本：[rir_metrics.zh-TW.md](rir_metrics.zh-TW.md)

Room-acoustic measurements on an impulse response. They are used to record what
a renderer actually produced (`realized_acoustics` in item metadata), to gate
bank items in QC, and to compare banks. Every function takes an array and
explicit analysis settings; none knows which renderer made the signal. The
decay, clarity and noise-floor definitions follow ISO 3382-1.

## Report facade

```python
from puresound.audio.rir.metrics import analyze_rir, DEFAULT_OCTAVE_CENTERS_HZ

report = analyze_rir(
    rir_channel, 16000,                 # mono RIR, sample rate
    direct_window_ms=2.5,               # DRR direct window
    direct_index=None,                  # default: absolute peak
    octave_centers_hz=DEFAULT_OCTAVE_CENTERS_HZ,
    noise_compensation=True,            # Lundeby-corrected Schroeder curve
    echo_density=True,
)
report["t20"]["rt60_s"], report["drr_db"], report["octave_bands"]["1000"]["t20_s"]
```

Broadband keys: `direct_sample`, `direct_delay_ms`, `peak_abs`, `energy`,
`drr_db`, `c50_db`, `c80_db`, `spectral_tilt_db_per_octave`, `noise_floor`,
`edt` / `t20` / `t30` (full fit records) and `edt_s` / `t20_s` / `t30_s`.
`octave_bands` appears only when `octave_centers_hz` is given; each band holds
DRR, C50, EDT/T20/T30, their fit R² and its own noise floor. `echo_density`
appears only when `echo_density=True`. Values that are infinite (for example a
response with no reverberant energy) are reported as `None`.

## Definitions

The direct sample `n_d` is the absolute peak unless `direct_index` is given.
Hybrid-renderer metadata passes the geometric arrival `round(d / c · fs)`.

| Metric | Function | Definition |
|---|---|---|
| DRR | `compute_drr_db` | `10 log10(Σ h² over [n_d, n_d + W) / Σ h² after)`, `W = direct_window_ms`; energy before `n_d` is ignored |
| C50, C80 | `clarity_db` | Same form with the boundary at 50 or 80 ms after `n_d` |
| Schroeder decay | `schroeder_decay_db` | `10 log10(∫_t^∞ h² / ∫_0^∞ h²)` from `n_d`, floored at −120 dB |
| Noise-compensated decay | `noise_compensated_schroeder_decay_db` | Schroeder integral truncated at the Lundeby intersection with the expected noise energy subtracted, kept monotonic; falls back to the raw curve when no reliable floor is found |
| EDT / T20 / T30 | `estimate_decay_time` → `DecayEstimate` | Least-squares line on the decay curve over 0…−10, −5…−25, −5…−35 dB; `RT60 = −60 / slope`; needs ≥ 8 points spanning ≥ 10 ms |
| Noise floor | `estimate_noise_floor_lundeby` → `NoiseFloorEstimate` | Lundeby iteration on 10 ms block energies: tail floor, decay fit above it, intersection, refine |
| Spectral tilt | `spectral_tilt_db_per_octave` | Least-squares slope of `20 log10 |H(f)|` against `log2 f`, 200 Hz – 4 kHz |
| Octave bands | `octave_band_rir`, `valid_octave_centers` | Causal fourth-order Butterworth band-pass over `[f_c/√2, f_c·√2]`; centers whose upper edge exceeds 0.99 × Nyquist are dropped |
| Echo density | `abel_normalized_echo_density_profile`, `analyze_echo_density` | Abel–Huang: fraction of samples in a 20 ms window with `|h| > σ`, divided by the Gaussian fraction `erfc(1/√2)` (≈ 1 for a diffuse field) |
| Mixing time | `estimate_abel_mixing_time` | First time after `n_d` where the profile stays ≥ 0.9 for 10 ms |
| Multiband late field | `analyze_multiband_late_field` | Per-octave noise-aware decay and echo density, anchored on the broadband direct sample |
| IACC | `interaural_cross_correlation`, `analyze_binaural_iacc` | Maximum absolute normalized cross-correlation within ±1 ms lag, early (0–80 ms) and late windows, broadband and 500 Hz – 4 kHz octaves |
| Array coherence | `analyze_array_spatial_coherence`, `diffuse_field_coherence` | Complex coherence `S₁₂ / √(S₁₁ S₂₂)` (Welch) of a synchronized pair after 80 ms, compared per octave with the diffuse-field target `sinc(2 f d / c)` |

`DEFAULT_OCTAVE_CENTERS_HZ` is 63 Hz – 8 kHz.

## Interpretation rules

- A decay estimate is only as good as its fit. Check
  `DecayEstimate.r_squared` before trusting `rt60_s`; bank QC requires
  `RIRBankQCPolicy.minimum_decay_r_squared` (0.70).
- `estimate_noise_floor_lundeby` corrects only when the decay has at least
  `min_dynamic_range_db` (15 dB) above the floor. Otherwise it reports
  `correction_applied=False` with a `reason` such as
  `insufficient_dynamic_range`, `insufficient_decay_above_noise` or
  `tail_is_not_stationary_noise`. Measured corpora whose tails were faded
  before publication hit these for reasons unrelated to measurement noise.
- IACC and coherence assume the channels are synchronized receivers of one
  source. Bank items whose channels are independent source-to-microphone paths
  are not evaluable this way; bank QC marks them `not_applicable`.

## Policy strings

`ABEL_ECHO_DENSITY_POLICY`, `MULTIBAND_LATE_FIELD_POLICY`, `IACC_POLICY` and
`DIFFUSE_FIELD_COHERENCE_POLICY` are stamped into their own functions'
results. Of these, only the echo-density policy reaches `analyze_rir`'s report,
inside `echo_density`. A change to what a metric computes gets a new policy
string, so stored reports stay interpretable.

## Where it is used

- `generate_hybrid_rir(..., record_realized_metrics=True)` stores
  `analyze_rir` per channel.
- Bank QC (`puresound.audio.rir.bank.qc`) runs it on every item.
- `puresound.audio.rir.bank.evaluation.compare_release_distributions`
  compares release variants; `egs/rir_generation/compare_bank_acoustics.py`
  compares DRR, clarity, decay and spectral statistics across bank folders.

Splitting a response into direct, early and later parts for auditing a
renderer is a separate module: [attribution](rir_attribution.md).
