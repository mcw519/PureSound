# Moving-source rendering — `puresound.audio.rir.render.dynamic`

Traditional Chinese: [dynamic_scene.zh-TW.md](dynamic_scene.zh-TW.md)

Renders a `DynamicSceneSpec` (`puresound.audio.rir.scene.dynamic`): a static
`RoomSceneV2` room, one fixed mono microphone and up to eight sources — target
talkers, other talkers and noise — each standing still or moving along
keyframed trajectories. It produces the microphone mixture, per-source stems
and three evaluation references. The web workbench's
[Acoustic world](../../usage/world.md) screen is built on it.
Renderer version: `DYNAMIC_RENDER_VERSION = "puresound.dynamic_geometric_fdn.v2.1"`.

```python
from puresound.audio.rir.scene.world_presets import world_preset
from puresound.audio.rir.render.dynamic import render_dynamic_scene

scene = world_preset("approach", seed=4)
render = render_dynamic_scene(scene, assets)  # asset_id -> 16 kHz mono array
render.mixture, render.stems, render.references["target"], render.references["near"]
```

## Scenes and roles

A `DynamicSceneSpec` (schema `puresound.dynamic_scene.v2`) holds 1 to 8 sources
(`MAX_SOURCES`), each a `DynamicSource` with a role:

| Role | Meaning | Directivity of its room transducer |
| --- | --- | --- |
| `target` | a talker the model should keep | `speech_human` |
| `interferer` | another talker | `speech_human` |
| `noise` | a noise source | `omnidirectional` |

The renderer reads directivity from the room transducer; presets and the web
editor set it from the role. Every source follows its keyframes, and a single
keyframe means it stands still — noise included. The speed, wall, obstacle
and microphone-distance checks apply to every source. `reference_rir` chooses
what the references keep (below).

Speech plays complete clips only. With `repeat`, a talker plays as many whole
clips as fit and is then quiet, while propagation and reverberation still render
over the remaining time; a clip that cannot fit once is rejected. Noise may fill
a partial final cycle.

A scene in the older `puresound.dynamic_scene.v1` schema (`fixed_source_id` and
the roles `speech` / `noise`) converts on read: the fixed talker becomes
`target`, the other talkers `interferer`, and the reference type is `early`.

## Signal model

Every coherent path `p` — the direct sound and each image reflection up to
`max_order` (≤ 2) — is a short **path filter** `h_p` followed by a **travel
time** `τ_p`. A sample emitted at time `e` arrives at `e + τ_p(e)`:

```
y(t) = Σ_p  (h_p,e ∗ s)(e)   at   t = e + τ_p(e)
```

`h_p,e` is evaluated at the source's pose at the *emission* time `e`: spherical
spreading, source directivity, the receiver pattern, wall reflection filters
and obstacle insertion loss. Keeping the filter free of delay and moving the
delay into a continuous time map is what lets a path change length without
the comb filtering and double arrivals that cross-fading delayed impulse
responses produces; Doppler shift follows from the time map itself.

| Grid | Step | What is evaluated |
| --- | --- | --- |
| Geometry | 10 ms (plus every keyframe instant) | Pose, distance, near weight, travel time of every path |
| Path filter | 40 ms (plus keyframes and the last frame) | Path events, wall filters, directivity, occlusion |

Between path-filter updates the taps are interpolated linearly, and so is each
path's image-source position. The latter is exact: an image source is an
affine function of the source position, and the source moves linearly between
keyframes, all of which are update frames. Travel times computed from the
interpolated image positions therefore equal those of a 10 ms evaluation. The
filters themselves (incidence angles, directivity, Fresnel numbers) change by
a few degrees per 40 ms at walking speed (≤ 2 m/s).

A path is followed by its type and image order. The order in which a
second-order path meets its two walls can swap as the source moves, so it is
not part of the identity; and where a reflection point lands exactly on a
room edge at one update, that update's filter is interpolated from its
neighbours instead of dropping the path.

### Time-varying filtering

Segment `j` between two update frames cross-fades the outputs of the filters
at its ends. Filtering is linear in the taps, so this equals filtering with
linearly interpolated taps. All segments are processed with one batched
overlap-save FFT per input signal (`_SegmentFilter`); a source that never
moves is a plain FFT convolution.

### Fractional travel time

Output sample `r` reads the emission instant `e` solving `e + τ(e) = r`. That
read is a fractional delay. A Kaiser-windowed sinc (32 taps, β = 6;
`path_events/fractional_delay.py`) keeps the magnitude within 0.01 dB up to
7 kHz for every fraction (Laakso et al., "Splitting the unit delay", IEEE
Signal Processing Magazine, 1996). Low-order interpolators lose high
frequencies by an amount that depends on the fractional part:

| Interpolator | 4 kHz | 6 kHz | 7 kHz |
| --- | --- | --- | --- |
| Linear | 0 … −3.0 dB | 0 … −8.3 dB | 0 … −14.2 dB |
| Causal 3rd-order Lagrange (static renderer default) | −0.1 … +0.7 dB | −3.2 … +1.3 dB | −8.4 … +1.4 dB |
| Windowed sinc, 32 taps | ±0.01 dB | ±0.01 dB | ±0.01 dB |

For a moving source the fraction cycles at `v / c · fs` Hz (47 Hz at 1 m/s), so
the spread in the table becomes amplitude modulation of the high band; with
the sinc it is gone (measured 0.00 dB peak to peak at 6 kHz). The symmetric
kernel rings for 15 samples (< 1 ms) before an arrival. No Doppler amplitude
factor is applied; at Mach < 0.006 it is below 0.05 dB.

## Source directivity

Talkers in the presets use `speech_human`: horizontal-plane octave-band
levels of normal-level speech, 0–180° in 15° steps, from Monson, Hunter &
Story, "Horizontal directivity of low- and high-frequency energy in speech and
singing", JASA 132(1), 433–441 (2012), Table I. Relative to the mouth axis:

| Angle | 125 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz | 8 kHz |
| --- | --- | --- | --- | --- | --- | --- |
| 90° | −1.6 | −1.6 | −1.6 | −7.0 | −6.3 | −8.7 |
| 180° | −3.6 | −5.9 | −6.3 | −13.1 | −18.9 | −26.3 |

The pattern is treated as symmetric about the mouth axis (elevation reuses the
horizontal data). It travels with each path as a `PathBandGain` and is applied
as a minimum-phase FIR (`path_events/band_filter.py`).
`speech_cardioid` is an ideal frequency-independent cardioid with a null
directly behind; scenes that name it still render it.

## Obstacles

Objects attenuate paths continuously instead of switching them off
(`occlusion_model="fresnel_kirchhoff"`, `path_events/occlusion.py`). Each
object is an opaque rectangular screen seen from each path leg; by Babinet's
principle the field behind it is

```
U / U0 = 1 − (1 − t) · [F(a2) − F(a1)] · [F(b2) − F(b1)] / (2j),   F(v) = C(v) + j S(v)
```

with the screen's extent `[a1, a2] × [b1, b2]` in Fresnel units
`v = h · sqrt(2 (d1 + d2) / (λ d1 d2))` and `t = sqrt(SceneObject.transmission)`
(Born & Wolf, *Principles of Optics*, §8.7–8.9; Pierce, *Acoustics*, ch. 9). It
gives −6 dB on the shadow boundary, more loss at higher frequency and deeper in
the shadow, and is continuous across the boundary. An object standing on the
floor or reaching the ceiling cannot be passed below or above it. The gain at
each third-octave centre is the octave-averaged energy, so interference nulls
between the three edges do not become notches sliding through the spectrum.

In the "occlusion" preset (0.4 × 0.8 m screen, 2 m high) the direct sound
behind the screen loses 2.4–7.3 dB at 250 Hz (more near the shadow's edges
than behind its middle), 6–16 dB at 1 kHz, 6–19 dB at 4 kHz and 6–24 dB at
8 kHz (less near the edges). While the talker walks behind it the loss changes
by at most 0.1 dB per 50 ms at 250 Hz and 5 dB at 8 kHz, where the shadow's
edge is sharpest. Limits: the object is a thin screen (thick objects attenuate somewhat
more), reflections from the object's faces are not modelled, and Kirchhoff
theory overstates the effect of objects much smaller than the wavelength, so
the loss below about 250 Hz is an upper bound.

## Late field

The diffuse field is the project's energy-matched PathEvent–FDN coupling
([early/late coupling](rir_late_coupling.md)), so a moving scene and a static
render of the same room agree. For a pose the static PathEvent RIR
(second order, windowed-sinc delays) is coupled to a multiband FDN whose tail
energy follows the octave RT60 of the materials plus air absorption (the
target of the static FDN backends), and its diffuse component, re-timed to start
at the direct arrival, becomes that pose's late response `L_k`. Coherent paths
fade out with the coupling's equal-power weight between 16 and 32 ms after the
direct sound, evaluated per tap from each tap's arrival time.

Responses are computed at every keyframe and at evenly spaced poses between
two keyframes, so neighbours are at most 1 m and 45° apart. An emitted sample
drives the two responses around its emission time:

```
late(t) = Σ_k  (g(u) · w_k(u) · s(e)) ∗ L_k,   w = (1 − u, u)
g(u)² = ((1 − u) E_a + u E_b) / ((1 − u)² E_a + u² E_b + 2 u (1 − u) C_ab)
```

with `E` the responses' energies and `C` their inner product. Neighbouring
responses are only partly correlated (correlation 0.35–0.66), so plain linear
weights would lose up to 3 dB half-way; `g` keeps the blend's energy linear
between them. Half-way along a 3.6 m walk the late level is within 1 dB of a
source standing there.

The late energy comes from a second-order path field even when the scene
renders fewer reflections coherently (`LATE_FIELD_ORDER`), because the diffuse
level belongs to the room.
Each source has its own FDN seed (`SeedSequence([scene.seed, index])`), so two
talkers' tails are not one filter.

## References for evaluation

Each source gets a reference stem of the scene's `reference_rir` type, on the
microphone's time axis. The types and windows are training's `target_rir_type`
in `wav_apply_rir`, measured from each emission's direct sound
(`REFERENCE_WINDOWS_S`):

| `reference_rir` | What it keeps of each emission |
| --- | --- |
| `early` (default) | Coherent paths arriving within 50 ms of the direct sound, and the first 50 ms of the diffuse field |
| `direct` | Coherent paths within 6 ms of the direct sound; the diffuse field starts later, so none of it |
| `full` | Everything: the reference equals the source's stem |
| `anechoic` | The dry emission moved to the direct sound's arrival time at unit gain — no spreading loss, directivity, air or walls |

The reference stems sum into three references (`render.references`):

- **`target`**: the target talkers. Silent when the scene has none.
- **`speech`**: every talker, target or not.
- **`near`**: every talker weighted at emission time by a raised cosine across
  a 0.2 m band around `near_radius_m` (`near_weights`), then propagated like
  the source. Silent while nobody is inside.

All signals share one gain that keeps the mixture's peak at or below 0.95.
`render.audio()` names them `input`, `source-{id}`, `reference-source-{id}` and
`reference-{target,speech,near}`.

`puresound.evaluation.world.world_metrics` scores an output against each
reference; against a silent one it reports the output's residual level per
window instead of SI-SDR. The web workbench scores noise-suppression models
against `speech` and voice-isolation models against `near` unless the user
picks another (`puresound.web.world.POLICY_DEFAULTS`); without either, `target`
when the scene has a target, else `speech`.

## Verification

`test/rir/test_dynamic_scene.py` and `test/rir/test_rir_path_band_effects.py`:

| Property | Check |
| --- | --- |
| A stationary source renders as the static pipeline | Error below −120 dB without a late field (−149 dB measured), below −40 dB with it (−44 dB measured) against windowed-sinc PathEvent RIR + coupling |
| Late field follows the room | Energy after 50 ms within 0.6 dB of the static coupling in three materials and two room sizes |
| No high-frequency modulation | 6 kHz envelope of a moving source flat within 0.2 dB |
| Doppler | 1 kHz tone receding at 1 m/s measures `f · c / (c + 1)` |
| Occlusion is continuous | Direct-path loss at 250 Hz, 1 kHz and 4 kHz changes < 1 dB per 20 ms along the preset path; low frequencies bend around the screen |
| Late field follows motion | Half-way along a walk the late level is within 1 dB of a source standing there |
| Directivity | Band gains equal the measured table; min-phase FIR within 0.15 dB |
| Separate tails | Two talkers' late tails correlate below 0.3 |
| Reference types | An impulse's `early` and `direct` references end 50 ms and 6 ms after its direct sound; `full` equals the stem; `anechoic` is the impulse at the direct arrival with unit gain |
| References sum their roles | With two targets and an interferer, `target` is the targets' sum, `speech` adds the interferer and `near` weights every talker |
| Many sources | Two targets, two interferers and two noise sources, one noise moving, validate; a ninth source is refused |

## Limits

A static room with one omnidirectional microphone; no moving microphone, no
binaural rendering, no opening doors. At most eight sources; render time grows
about linearly with their number. Several target talkers are scored as one
sum, not separately. Coherent reflections stop at second
order; later energy is the statistical FDN field. Path filters update every
40 ms. Directivity is horizontal data applied axisymmetrically. Obstacles are
thin Kirchhoff screens.
