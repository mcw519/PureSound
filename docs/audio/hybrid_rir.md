# Hybrid RIR rendering — `puresound.audio.rir.render`

One public entry point composes the whole render chain:

```python
from puresound.audio.rir.render.hybrid import generate_hybrid_rir
# also re-exported from puresound.audio.rir.api
```

The chain (details and code map in
[`rir_realism_algorithm_zh-TW.md`](rir_realism_algorithm_zh-TW.md) §3–§5):

1. **Low band (20–1000 Hz)** — ARD/DCT wave solve with per-mode material
   damping (`render/low_frequency/pytard.py`; GPU variant
   `GpuARDPytARDCuPyBackend`).
2. **High band (1000 Hz–Nyquist)** — one of three backends
   (`render/high_frequency/`): `pyroomacoustics` (ISM + ray tracing),
   `path-events-m3` (coherent PathEvents), `path-events-m4`
   (PathEvents early field + multiband FDN late field).
3. **Crossover** — causal 4th-order Linkwitz–Riley at
   `HybridRIRConfig.crossover_hz` with RMS energy matching over
   [700, 1300] Hz (`render/crossover.py`). Matching gain is clamped to
   `HybridRIRConfig.crossover_match_gain_range` and recorded per channel.
4. **Output calibration** — `output_mode`: `calibrated` (physical SPL
   semantics, peak may exceed 1.0) or `peak_normalized`.

## Causality contract

Every stage preserves leading zeros up to
`floor(distance / sound_speed * sample_rate)`: the low band clips before its
causal low-pass, the high band clips after direct alignment before its causal
high-pass, PathEvents use one-sided fractional-delay kernels, and the FDN
coupling is sample-exact before its transition. M6 QC enforces this as the
`prearrival_energy` gate.

## Backend selection

`render/backend.py` defines the backend protocol and
`BackendCapabilities` (including determinism declarations —
pyroomacoustics ray tracing is not byte-reproducible per seed).
The M6 wrapper `egs/rir_generation/generate_m6_bank.py` defaults to
`path-events-m4`; the low-level generator
`egs/rir_generation/generate_hybrid_rir.py` keeps `pyroomacoustics` as its
own default because the M4/M5 exit gates pin that layer.

## Applying an RIR

Convolve dry speech with a bank item's channel; the JSON sidecar is part of
the data contract (channel map, distances, level policy) — never separate WAV
from metadata. Training integration goes through the bank loaders
([`rir_bank.md`](rir_bank.md)) rather than ad-hoc convolution.
