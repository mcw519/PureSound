# puresound.audio.rir_metrics

Shared, per-channel room-impulse-response analysis for generated and measured
RIRs.

The module accepts a one-dimensional NumPy array, a torch tensor, or a
singleton-channel array. A genuine multi-channel array is rejected: call the
metrics once per channel so signals cannot accidentally be concatenated.

## Available metrics

| Function | Purpose |
|----------|---------|
| `direct_sample` | Absolute-peak sample used as the default direct arrival |
| `compute_drr_db` | Direct-to-reverberant energy ratio using a configurable direct window |
| `clarity_db` | Early-to-late ratio, including C50 and C80 |
| `schroeder_decay_db` | Backward-integrated energy-decay curve |
| `estimate_noise_floor_lundeby` | Stationary tail floor, dynamic range, and decay/noise intersection |
| `noise_compensated_schroeder_decay_db` | Truncated, expected-noise-subtracted decay curve |
| `estimate_decay_time` | EDT/T20/T30-style RT60 extrapolation with fit quality |
| `spectral_tilt_db_per_octave` | Log-magnitude coloration slope |
| `octave_band_rir` | Causal nominal-octave filtering |
| `abel_normalized_echo_density_profile` | Abel-Huang local normalized echo density versus time after the direct sample |
| `estimate_abel_mixing_time` | First or sustained normalized-density threshold crossing |
| `analyze_echo_density` | JSON-safe mixing-time, density-probe, and late-density summary |
| `analyze_multiband_late_field` | Noise-aware octave decay plus cycle-aware echo-density targets |
| `interaural_cross_correlation` | One early/late time-window IACC value and peak lag |
| `analyze_binaural_iacc` | Broadband and 500–4000 Hz IACC_E/IACC_L summaries |
| `diffuse_field_coherence` | Ideal isotropic 3D omni-pair sinc coherence target |
| `analyze_array_spatial_coherence` | Welch pair-coherence error against the diffuse target |
| `analyze_rir` | JSON-safe broadband and optional octave-band report |

## Example

```python
from puresound.audio.rir_metrics import (
    DEFAULT_OCTAVE_CENTERS_HZ,
    analyze_rir,
)

metrics = analyze_rir(
    rir,
    sample_rate=16000,
    direct_window_ms=2.5,
    octave_centers_hz=DEFAULT_OCTAVE_CENTERS_HZ,
    echo_density=True,
)
print(
    metrics["drr_db"],
    metrics["t30_s"],
    metrics["echo_density"]["mixing_time_s"],
)
```

`analyze_rir` represents an unbounded ratio or an invalid decay fit as `None`,
so the result can be serialized with strict JSON (`allow_nan=False`).
Echo-density analysis is opt-in so existing reports and generation cost remain
unchanged.

## Echo density and mixing time

For every full local window, the Abel-Huang estimator counts samples whose
absolute value exceeds the window standard deviation and divides that fraction
by

```text
erfc(1 / sqrt(2)) = 0.3173105...
```

the expected exceedance fraction of zero-mean Gaussian noise. A normalized
density near one is therefore consistent with a dense Gaussian late field.
The profile is not clipped and can exceed one. Windows begin at the direct
sample, reported times are direct-relative, and pre-arrival silence is not
included.

`estimate_abel_mixing_time(..., minimum_sustain_ms=0)` implements the original
first crossing of a configurable threshold (normally `0.9`). A positive
`minimum_sustain_ms` is a PureSound robustness extension for rejecting one-hop
fluctuations. Reports record the window, hop, threshold, sustain duration, and
policy string so these two definitions cannot be silently mixed.

This criterion is monophonic and statistical. It does not test octave-band
decay, coloration, array coherence, IACC, directionality, or perceptual
quality. Those remain independent M4 gates.

## M4.2 multiband late field

`analyze_multiband_late_field` applies the same causal nominal-octave filter to
generated and measured RIRs. Every band preserves the broadband direct sample
as its physical time anchor. Its echo-density window is

```text
max(base_window, minimum_cycles / lower_octave_edge)
```

with defaults of 20 ms and four cycles. At 125 Hz this is about 45.3 ms; from
500 Hz upward it is 20 ms. A fixed 20 ms low-band window would contain too
little waveform history to be a meaningful density statistic.

Each band also runs the existing Lundeby analysis. When a reliable stationary
noise intersection exists, echo density stops at that exclusive sample and
the report sets `echo_density_truncated_at_noise_intersection=true`. EDT, T20,
and T30 use the same corrected decay curve and retain their complete fit
quality. This prevents a measured recording's noise floor from being counted
as a perfectly dense physical reverberation tail.

The M4 late-field target currently covers 125 Hz through 4 kHz. The 63 Hz band
continues to be treated primarily as a modal/wave validation region rather than
an assumed diffuse late field.

## Spatial late-field contract

For binaural RIRs, `analyze_binaural_iacc` reports maximum absolute normalized
cross-correlation within ±1 ms for early 0–80 ms and late 80 ms–end windows. It
also reports the mean of the valid 500, 1000, 2000, and 4000 Hz octave values as
`iacc_e4` and `iacc_l4`.

For two omnidirectional array receivers separated by distance `d`, the ideal
isotropic three-dimensional diffuse-field coherence is

```text
sin(2*pi*f*d/c) / (2*pi*f*d/c) = np.sinc(2*f*d/c)
```

`analyze_array_spatial_coherence` estimates complex pair coherence with
overlapped Welch spectra over the late window and reports octave-band and
overall complex RMSE against that signed target. Imaginary residual is retained
instead of collapsing everything to magnitude-squared coherence.

Both spatial APIs require matched responses from one source to multiple
receivers. Different source channels observed by one receiver are not a spatial
array and must not be passed as if they were simultaneous microphone channels.

## Decay-fit interpretation

Decay estimates use Schroeder backward integration and a linear least-squares
fit:

- EDT: 0 to -10 dB, extrapolated to 60 dB;
- T20: -5 to -25 dB, extrapolated to 60 dB;
- T30: -5 to -35 dB, extrapolated to 60 dB.

Each successful fit includes its slope, R², time interval, and sample count.
Insufficient dynamic range or a non-negative slope returns `None`.

By default, `estimate_decay_time` and `analyze_rir` use a Lundeby-style
engineering correction:

1. average squared pressure in 10 ms blocks;
2. estimate a stationary floor from the final 20% of blocks;
3. fit decay blocks at least 10 dB above that floor;
4. iteratively solve the decay/noise intersection;
5. truncate the Schroeder integration there and subtract expected noise energy.

The correction is applied only with at least 15 dB dynamic range, an
intersection inside the recording, and a tail that is sufficiently stationary.
Otherwise the raw curve is retained and `noise_floor.reason` explains why.
Every broadband and octave report includes noise power, level relative to the
direct block, dynamic range, intersection time, truncation sample, decay-fit R²,
iteration count, and correction status. This is suitable for deterministic
generated-versus-real screening, but is not a claim of complete ISO 3382
instrument conformance.

## Compare RIR banks

`egs/rir_generation/compare_bank_acoustics.py` uses this module for both
generated and measured banks:

```bash
python egs/rir_generation/compare_bank_acoustics.py \
  synthetic=egs/rir_generation/exp/rir_realism/m1/hybrid_rir_16k_levels/wide \
  measured=/work/any_exp_link/puresound_exp/real_rir_16k_train_view/all \
  --per-bank 300 \
  --octave-bands \
  --json-output egs/rir_generation/exp/rir_realism/m0/rir_benchmark_v0/acoustics.json
```

The terminal table is a compact median summary. The JSON output contains each
sampled channel, fit diagnostics, optional octave bands, room origin, distance
bucket, and per-bucket medians.

Sampling is deterministic and without replacement for a fixed seed. The tool
lists metadata filenames but does not parse them or `stat()` every matching WAV;
only selected candidates are validated and loaded. A small comparison therefore
does not first index every item in a large or remotely mounted bank.

Bucket decay medians include only fits whose R² passes `--min-decay-r2`
(default `0.9`). The report also includes valid-fit counts/fractions, the
Lundeby correction fraction, median dynamic range, and median valid noise
intersection. A noise-floor-dominated T30 is therefore exposed as low coverage
instead of appearing as an extreme but apparently valid room decay.

For the M4 late-field baseline:

```bash
.venv/bin/python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_late_field_baseline.py \
  measured=/work/any_exp_link/puresound_exp/real_rir_16k_train_view/items \
  m1=egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100 \
  m3=egs/rir_generation/exp/rir_realism/m3/rir_m3_late_baseline \
  --reference-tag measured --per-bank 100 --seed 20260731 \
  --json-output egs/rir_generation/phases/m4_spatial_late_field/reports/m4_late_field_baseline.json
```

The validator samples at most one source channel per item and reports complete
distributions overall and by distance. Its measured central-80% envelope checks
are diagnostic only; they do not replace multiband, spatial, or listening
criteria.

The M4.2 multiband target is reproduced with:

```bash
.venv/bin/python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_multiband_late_field.py \
  measured=/work/any_exp_link/puresound_exp/real_rir_16k_train_view/items \
  m1=egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100 \
  m3=egs/rir_generation/exp/rir_realism/m3/rir_m3_late_baseline \
  --reference-tag measured --per-bank 50 --seed 20260731 \
  --json-output egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json
```

This report includes 125–4000 Hz distributions and an explicit
`spatial_contract.evaluable_from_these_banks=false` marker because these banks
contain multiple sources, not multiple receivers.
