# Hybrid RIR rendering — `puresound.audio.rir.render.hybrid`

繁體中文版本：[hybrid_rir.zh-TW.md](hybrid_rir.zh-TW.md)

`generate_hybrid_rir` renders one room impulse response per source by solving
the low band with a wave/modal backend and the high band with a geometric
backend, then joining the two at a causal crossover. Where each backend lives
in the package is in the [implementation guide](rir_realism_algorithm.md).

## Why two bands

At low frequencies a room behaves as a set of discrete modes; geometric
acoustics (image sources, rays) cannot represent them, while a wave solver can.
At high frequencies the mode density is large enough for geometric acoustics to
be accurate, and a wave solver would be too expensive: its grid must resolve the
wavelength, so its cost grows roughly with the cube of the top frequency. The
hybrid renderer uses each method only where it is valid and affordable.

## Entry point

```python
from puresound.audio.rir.render.hybrid import generate_hybrid_rir

rir, metadata = generate_hybrid_rir(
    config,               # HybridRIRConfig
    scene=None,           # HybridRIRScene | RoomSceneV2; sampled from config when None
    low_backend=None,     # default GpuARDPytARDBackend() (pytARD, CPU)
    high_backend=None,    # default PyroomacousticsHighFrequencyBackend()
    seed=None,            # seed for scene sampling when scene is None
)
# rir: torch.float32 [num_sources, num_samples]; metadata: JSON-safe dict
```

Any object with `simulate(scene, config) -> np.ndarray [num_sources, samples]`
satisfies the `RIRBackend` protocol (`render/backend.py`). A `RoomSceneV2` must
contain exactly `config.num_sources` sources and one receiver; receiver arrays
go through the [spatial renderer](spatial_rir.md).

## Configuration

`HybridRIRConfig` (`puresound.audio.rir.contracts`) holds scene-sampling ranges
and rendering settings. The rendering fields:

| Field | Default | Meaning |
|---|---|---|
| `sample_rate` | 48000 | Output rate, Hz |
| `duration` | 1.5 | Output length, s |
| `low_fmin_hz`, `low_fmax_hz` | 20, 1000 | Band the low backend is responsible for, Hz |
| `crossover_hz` | 1000 | Linkwitz–Riley crossover frequency, Hz |
| `match_crossover_energy` | `True` | Scale the low band to the high band around the crossover |
| `crossover_match_band_hz` | `None` | Energy-match band; `None` means `[0.7, 1.3] × crossover_hz` |
| `crossover_match_target_db` | 0.0 | Target low/high RMS ratio in the match band, dB |
| `crossover_match_gain_range` | (1e-4, 8.0) | Clamp on the low-band match gain |
| `preserve_source_convention_at_crossover` | `True` | Skip the match when the low band already shares the high band's source convention |
| `tail_fade_ms` | 20 | Fade-out at the end of the response, ms |
| `output_mode` | `peak_normalized` | `peak_normalized` or `calibrated` |
| `normalize_peak` | 0.98 | Peak target for `peak_normalized` |
| `calibrated_reference_source_spl_db` | 94 | Reference source level for transducer gains, dB SPL |
| `record_realized_metrics` | `False` | Store `analyze_rir` results per channel |
| `num_near_sources`, `num_far_sources` | 2, 3 | The channel count is their sum |

For a `RoomSceneV2` the sound speed comes from the scene environment, not from
`HybridRIRConfig.sound_speed`.

## Algorithm

1. **Low band.** The low backend renders every source channel. Samples before
   `floor(d / c · fs)` are set to zero (`clip_rir_before_physical_arrival`):
   a finite modal or voxel solve leaves a small numerical precursor that is not
   a physical path.
2. **High band.** The high backend renders every channel. Pyroomacoustics
   output is shifted to remove its constant fractional-delay offset so the
   direct path lands at `d / c` (`align_high_band_direct`), then clipped the
   same way. PathEvent backends are causal by construction.
3. **Transducer gain** (`RoomSceneV2` only). Both bands are multiplied per
   channel by `10^((L_src − L_ref + G_rx) / 20)`, where `L_src` is the source
   `power_db_spl_at_1m`, `L_ref` is `calibrated_reference_source_spl_db` and
   `G_rx` is the receiver `calibration_gain_db`.
4. **Energy match.** Both bands are band-passed over the match band
   (second-order Butterworth) and the low band is scaled by
   `g = clip(RMS_high · 10^(target_dB / 20) / RMS_low, gain_range)`. The pytARD
   low band is peak-normalized by its own calibration, so this match is what
   gives it a level relative to the high band; channels whose gain hit the
   clamp are listed in the metadata. The match is skipped when the low backend
   reports `direct_path_source_convention_matched` and
   `preserve_source_convention_at_crossover` is set. The metadata records both
   whether matching was requested and whether it was applied.
5. **Crossover.** A fourth-order Linkwitz–Riley pair — a second-order
   Butterworth applied twice — filters each band causally (`sosfilt`) and the
   bands are summed. The low- and high-pass share one phase response, so the
   independently rendered bands stay time-aligned and sum flat in magnitude;
   causal filtering keeps pre-ringing out of the direct arrival.
6. **Tail fade.** The last `tail_fade_ms` is multiplied by a
   raised-cosine-squared window that ends at exactly zero.
7. **Output level.** `peak_normalized` scales the whole item so
   `max |h| = normalize_peak`. `calibrated` applies no normalization: the
   amplitude carries source level, distance, receiver calibration and
   cross-room level, and the peak may exceed 1.0.

## Backends

| CLI name (`generate_hybrid_rir.py`) | Class | Notes |
|---|---|---|
| `pytard` | `GpuARDPytARDBackend` | pytARD DCT-grid wave solve on CPU; broadband RT60 envelope |
| `pytard-material` | same, `material_modal_damping=True` | Per-mode damping from surface materials ([modal damping](modal_damping.md)); needs a material-first scene |
| `pytard-cupy`, `pytard-cupy-material` | `GpuARDPytARDCuPyBackend` | The same solver on the GPU through CuPy |
| `analytic`, `analytic-material` | `AnalyticModalLowFrequencyBackend` | Rectangular-room modes; fast fallback for tests and smoke runs |
| `analytic-impedance` | `ImpedanceModalLowFrequencyBackend` | Complex-impedance modes; needs an explicit boundary JSON ([impedance measurements](impedance_measurements.md)) |
| `pyroomacoustics` | `PyroomacousticsHighFrequencyBackend` | Image sources up to `max_order` (12) plus ray tracing (20000 rays) |
| `path-events-m3` | `PathEventHighFrequencyBackend` | Coherent image-source PathEvents with boundary filters, directivity, ISO 9613-1 air absorption and object visibility |
| `path-events-m4` | `PathEventFDNHighFrequencyBackend` | PathEvent early field plus a multiband FDN late field ([late coupling](rir_late_coupling.md), [FDN](multiband_fdn.md)) |

pytARD is an optional AGPL dependency and is not distributed with PureSound;
point `PURESOUND_PYTARD_ROOT` at a checkout. The PathEvent backends require a
`RoomSceneV2`.

## Causality contract

Every stage keeps a channel exactly zero before `floor(d / c · fs)` and keeps
the arrival sample itself: the low band is clipped before its causal low-pass,
the high band after direct alignment, PathEvents use one-sided fractional-delay
kernels, and the FDN coupling leaves every sample before its transition
untouched. `RIRArray.violates_causality` checks the same boundary from the
consumer side, and bank QC fails an item on `prearrival_energy`.
`resolve_sound_speed(metadata)` returns the sound speed that defines the
boundary for a stored item (scene environment first, config as fallback).

## Determinism

PathEvent and FDN backends are deterministic for fixed inputs. Pyroomacoustics
ray tracing draws from libroom's process-global generator and a package-local
NumPy generator; its output is byte-reproducible only when
`PyroomacousticsHighFrequencyBackend.set_rng_seed` pins both before the render.
`generate_hybrid_rir.py` does this per task when it writes a bank manifest.
`BackendCapabilities` (`contracts.py`) is where a backend declares whether it
is reproducible for a fixed seed.

## Metadata

The returned dictionary carries `config`, `scene` (including `channel_map`: each
channel's label, position and distance), `obstacle_effects`, `bands.low` and
`bands.high` (backend, band, boundary model, excitation, late field, air
absorption), `output_calibration`, `crossover` and, when requested,
`realized_acoustics`. The dataset writer adds `sample_id`, `room_id`,
`room_index`, `rir_index` and the file names; `validate_rir_metadata` checks the
required keys (`RIR_METADATA_REQUIRED_KEYS`) and the channel map.

## Tools

| Tool | Default high band | Use |
|---|---|---|
| `egs/rir_generation/generate_hybrid_rir.py` | `pyroomacoustics` | Low-level generator: one WAV/JSON pair per item, optional bank manifest (`--emit-m6-manifest`) |
| `egs/rir_generation/generate_m6_bank.py` | `path-events-m4` | Generation, QC and release packaging in one command ([bank format](rir_bank_v2.md)) |

The low-level default stays `pyroomacoustics` so existing invocations keep
producing the same data; the bank wrapper always passes the backend explicitly.
Commands are in the [RIR generation guide](../../../egs/rir_generation/README.md).

A generated item is a 32-bit float WAV with a same-stem JSON sidecar. The
sidecar is part of the data (channel map, distances, level policy) and must not
be separated from the WAV. Training reads items through the
[bank loaders](rir_bank.md), not by convolving files directly.
