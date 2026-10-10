# Spatial RIR and BRIR — `puresound.audio.rir.render.spatial`

繁體中文版本：[spatial_rir.zh-TW.md](spatial_rir.zh-TW.md)

Renders one source into a set of **synchronized** receivers — a microphone
array, first-order Ambisonics (FOA) and, optionally, a binaural pair — so that
inter-channel delay, coherence and IACC come from one physical sound field. It
is a separate, opt-in API: bank generation (`generate_hybrid_rir`) renders one
mono receiver per item and does not use it.

## Entry point

```python
from puresound.audio.rir.render.spatial import render_room_scene_spatial_rir

render = render_room_scene_spatial_rir(
    scene,                          # RoomSceneV2 with one or more receivers
    sample_rate=16000, duration_s=1.2, source_index=0,
    max_order=4,                    # PathEvent image-source order (0..20)
    mixing_time_s=0.024,            # transition centre after each direct arrival
    transition_duration_s=0.016,
    delay_line_count=16,            # FDN delay lines (power of two)
    plane_wave_count=128,           # late-field directions (>= 16)
    seed=...,                       # base seed of the late field
    minimum_fdn_center_hz=500.0,
    material_reference_frequency_hz=1000.0,
    decoder=None,                   # AmbisonicBinauralDecoder for a BRIR
)
```

`SpatialRoomRIRRender` holds `receiver_rirs` `[receivers, samples]`,
`ambisonic_acn_sn3d` `[4, samples]` (ACN order `W, Y, Z, X`, SN3D
normalization), their coherent-only and late-only components, the coupling
records, an optional `binaural` BRIR `[2, samples + taps − 1]`, and metadata
(policy `puresound.spatial_room_rir.v1`).

## Algorithm

1. **Early field, per receiver.** Coherent PathEvents are generated for each
   receiver (boundary filters from the material relaxation prior, directivity,
   object visibility) and filtered for ISO 9613-1 air absorption at that
   receiver's direct distance. The FOA early field uses an omnidirectional
   reference receiver at the array centroid.
2. **One shared late field.** A multiband FDN ([FDN](multiband_fdn.md)) is
   designed from the scene's predicted octave RT60s at 500 Hz – 4 kHz (those at
   or above `minimum_fdn_center_hz` and below Nyquist), corrected for air
   absorption. It is excited by a unit impulse at the earliest direct arrival.
   `plane_wave_count` directions come from a Fibonacci sphere rotated by the
   seed (`fibonacci_sphere_directions`). For each octave band, every direction
   carries an independent Gaussian octave-band noise carrier multiplied by the
   causal RMS envelope (10 ms) of that FDN band; the sum is scaled so the W
   channel carries the FDN's energy.
3. **Projection.** Each receiver sums the plane waves, each delayed by
   `(r − r̄) · k / c` relative to the array centroid `r̄` (plus a common causal
   margin equal to the aperture radius over `c`) and weighted by the receiver's
   directivity for that arrival direction. FOA is the projection at the
   centroid: `W = Σ p`, `Y = Σ n_y p`, `Z = Σ n_z p`, `X = Σ n_x p`, normalized by
   `√N`.
4. **Early/late coupling.** Each channel gets the equal-power crossfade of the
   [late coupling](rir_late_coupling.md) centred `mixing_time_s` after its own
   direct arrival, so samples before the transition are exact. The diffuse part
   of all channels is then scaled by **one shared gain**, the positive root that
   makes the array's total post-transition energy equal the sum of the
   per-channel targets (each extrapolated from the material RT60). A shared gain
   keeps the inter-channel energy ratios — the spatial cues — that per-channel
   renormalization would erase (`"energy_policy":
   "one_shared_array_gain_preserves_spatial_ratios"`).
5. **Binaural (optional).** `render_ambisonic_brir(foa, sample_rate, decoder)`
   convolves the four FOA channels with the decoder's causal FIRs and sums per
   ear.

Because every channel is a projection of the same set of plane waves driven by
one FDN, the late field has physically consistent inter-channel coherence
instead of independently generated mono tails.

## Directivity

Sources and receivers use real first-order pressure patterns
(`directivity_pressure_gain` in `puresound.audio.rir.path_events`):

```text
gain = α + (1 − α) · (forward · direction)
```

| Pattern | α |
|---|---|
| `omnidirectional` | 1 (gain is 1, no orientation dependence) |
| `cardioid`, `speech_cardioid` | 0.5 |
| `hypercardioid` | 0.25 |
| `figure_eight` | 0 |

`forward` follows from the pose's yaw and pitch. Hypercardioid and figure-eight
go negative on the rear lobe; that sign is a real pressure-phase reversal and is
not clamped. An unknown pattern raises `NotImplementedError` rather than
falling back to omnidirectional.

## Binaural decoder

`AmbisonicBinauralDecoder` (`puresound.audio.rir.render.binaural`) is a causal
two-ear FIR bank of shape `[2 ears, 4 FOA channels, taps]`. Construction
requires a positive sample rate, that shape, finite taps, a `reference_id` and
a `decoder_kind`, and keeps a `provenance` mapping; `render_ambisonic_brir`
requires the FOA sample rate to equal the decoder's.

`analytic_first_order_binaural_decoder(sample_rate)` is a one-tap, lateral,
headless decoder (`decoder_kind="analytic_demonstration_not_hrtf"`) that only
exercises the pipeline; its provenance records `"measurement": false` and
`"production_hrtf_replacement_required": true`. A real binaural render injects a
measured, licensed HRTF-derived FIR set through the same class; the spatial
renderer does not change.

## Command line

```bash
PYTHONPATH=. python egs/rir_generation/render_spatial_rir.py \
  --sample-rate 16000 --duration 1.2 \
  --binaural-spacing-m 0.17 --binaural-decoder analytic \
  --output-dir <out>
```

Without `--scene-json` the tool samples a deterministic office scene;
`--binaural-spacing-m` expands a one-receiver scene into an x-axis pair
(`<= 0` disables it). `--max-order`, `--delay-lines`, `--plane-waves`,
`--source-index` and `--seed` map to the arguments above.

The array plumbing in `puresound.audio.rir.render.arrays` (`pad_or_trim`,
`coerce_rir_array`) is unrelated to receiver arrays: it only enforces the
`[channels, samples]` float64 layout that backends and the crossover share.
