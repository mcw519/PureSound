# M4 spatial RIR and BRIR

繁體中文版本：`spatial_rir.zh-TW.md`

The M4 spatial renderer is an explicit opt-in API; the production dataset
generators (`generate_hybrid_rir.py` and its `generate_m6_bank.py` wrapper —
see [`rir_scene_v2.md`](rir_scene_v2.md)) still default to Pyroomacoustics and
render one mono, per-channel RIR at a time. The high-level entry point is

```python
puresound.audio.rir.render.spatial.render_room_scene_spatial_rir(
    scene: RoomSceneV2,
    *,
    sample_rate: int = 16000,
    duration_s: float = 1.2,
    source_index: int = 0,
    max_order: int = 4,
    mixing_time_s: float = 0.024,
    transition_duration_s: float = 0.016,
    delay_line_count: int = 16,
    plane_wave_count: int = 128,
    seed: int = 20260731,
    minimum_fdn_center_hz: float = 500.0,
    material_reference_frequency_hz: float = 1000.0,
    decoder: AmbisonicBinauralDecoder | None = None,
) -> SpatialRoomRIRRender
```

which takes one `RoomSceneV2` and returns, in a single call:

- the complete RIR from one source to every synchronized receiver;
- first-order Ambisonics in ACN channel order with SN3D normalization, fixed
  as `W/Y/Z/X`;
- coherent PathEvent early components and a shared spatial-FDN late
  component, for both the receiver array and the Ambisonic channels;
- a two-channel BRIR, if an `AmbisonicBinauralDecoder` is supplied.

## One shared late field, not one late field per channel

The late field is not sampled independently per channel. Every receiver and
the Ambisonic output are projections of the *same* set of seeded,
rotated-Fibonacci-sphere plane-wave directions
(`fibonacci_sphere_directions()`, at least 16 directions and 128 by default —
`plane_wave_count`). Each direction carries an independent octave-band noise
carrier whose envelope comes from one shared passive multiband FDN (see
[`multiband_fdn.md`](multiband_fdn.md)), so inter-channel delay, coherence,
and IACC all trace back to one physical late field rather than to
independently generated mono tails.

The direct/early portion is instead rendered per receiver from coherent
PathEvents. Coupling early and late happens with an equal-power crossfade
centered at `mixing_time_s` (24 ms by default) after each channel's own
direct arrival, over a `transition_duration_s` window (16 ms by default):
every sample before the transition start is preserved exactly, and past the
transition each channel's diffuse component is scaled by one shared,
positive-quadratic-root gain so the coupled RIR's post-transition energy
matches an explicit per-channel target derived from the scene's own
material-predicted octave RT60s. That gain is deliberately shared across the
whole array rather than solved independently per channel
(`"energy_policy": "one_shared_array_gain_preserves_spatial_ratios"` in the
render metadata), so inter-channel energy ratios — and therefore spatial
cues — survive the coupling step instead of being renormalized away.

## Directivity

Sources and receivers support `omnidirectional`, `cardioid`,
`hypercardioid`, and `figure_eight` real first-order pressure patterns
(`directivity_pressure_gain()` in `puresound.audio.rir.path_events`),
evaluated as

```text
gain = alpha + (1 - alpha) * dot(forward, direction)
```

with `alpha = 1.0 / 0.5 / 0.25 / 0.0` respectively for the four patterns
(`omnidirectional` is the degenerate case and returns `1.0` directly, with no
orientation dependence). Because `alpha` is below `0.5` for `hypercardioid`
and exactly `0` for `figure_eight`, both patterns go negative on their rear
lobe; that negative gain is a genuine pressure-phase reversal and is never
clamped to zero. An unsupported directivity id raises instead of silently
falling back to omnidirectional.

## Binaural (BRIR) decoding

The optional BRIR decoder (`AmbisonicBinauralDecoder` in
`puresound.audio.rir.render.binaural`) is a causal two-ear FIR with shape
`[2 ears, 4 ambisonic channels, taps]`. Constructing one enforces a positive
sample rate, the `[2, 4, *]` shape, finite taps, and a non-empty
`reference_id` and `decoder_kind`; every decoder retains its sample rate,
reference identity, decoder kind, and a `provenance` mapping, and
`render_ambisonic_brir()` requires the FOA input's sample rate to match the
decoder's exactly.

The bundled `analytic_first_order_binaural_decoder()` is explicitly a
one-tap, lateral, headless demonstration decoder
(`decoder_kind="analytic_demonstration_not_hrtf"`) whose only job is to prove
the optional BRIR pipeline end to end; its provenance records
`"measurement": false` and `"production_hrtf_replacement_required": true`. A
production binaural render should inject a measured, licensed HRTF-derived
FIR through the same `AmbisonicBinauralDecoder` contract instead — the
spatial renderer itself does not need to change to accept one.

## Array-shape plumbing

`puresound.audio.rir.render.arrays` is unrelated to the acoustic
"synchronized receiver array" concept described above — it is the shared,
low-level shape contract (`pad_or_trim()`, `coerce_rir_array()`) that every
render backend and the low/high-band crossover use to agree on one
`[channels, samples]` float64 array. It carries no receiver-array or spatial
semantics of its own; it is plumbing that was moved out of
`puresound.audio.rir.render.hybrid` during the R2 module split recorded in
`RIR_EXP_LOG.md`.

## Quick generation

```bash
PYTHONPATH=. python egs/rir_generation/render_spatial_rir.py \
  --sample-rate 16000 --duration 1.2 \
  --binaural-spacing-m 0.17 --binaural-decoder analytic \
  --output-dir egs/rir_generation/exp/rir_realism/m4/spatial_demo
```

## Formal validation

```bash
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_spatial_rir.py
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_binaural_brir.py
PYTHONPATH=. python egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_m4_exit.py
```

The M4 implementation exit currently passes; measured multi-receiver
validation, licensed HRTF calibration, and controlled listening remain
separate empirical and production exits. Detailed formulas and the current
numbers are in [`rir_realism_algorithm.md`](rir_realism_algorithm.md).
