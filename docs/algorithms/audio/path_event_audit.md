# Known limitations of the PathEvent stack

Traditional Chinese: [path_event_audit.zh-TW.md](path_event_audit.zh-TW.md)

Limitations of the shared RIR code that both the static renderer and the
[moving-source renderer](dynamic_scene.md) build on. Each lists a measurement,
its reach and, where one exists, the opt-in correction. The shared defaults keep
the uncorrected behaviour: the static renderer and scene sampling feed the
training banks, and changing them changes the synthesis distribution. Adopting a
correction for training is a separate decision with its own retraining
comparison.

## 1. Third-order Lagrange delay colours high frequencies by fraction

`path_events/renderer.py`, default `render_path_events(fractional_delay="lagrange", fractional_delay_order=3)`.
The forward kernel keeps the strict causal support (nothing before
`floor(delay · fs)`), at the cost of a magnitude that depends on the
fractional part of each path's delay:

| | 4 kHz | 6 kHz | 7 kHz |
| --- | --- | --- | --- |
| Spread over fractions | −0.1 … +0.7 dB | −3.2 … +1.3 dB | −8.4 … +1.4 dB |

Every path of a static RIR gets its own pseudo-random high-band error, up to
several dB above 6 kHz.

- **Reach:** every PathEvent-based RIR (path-event and FDN high-frequency
  backends, spatial renderer).
- **Available:** `fractional_delay="windowed_sinc"` (32-tap Kaiser sinc,
  ±0.01 dB to 7 kHz; about 1 ms of pre-ringing instead of strict support, so a
  bank QC rule that relies on exact zeros before the arrival bin must tolerate
  the pre-ringing).

## 2. `speech_cardioid` is an ideal cardioid

`path_events/directivity.py` and `render/high_frequency/pyroomacoustics.py`
(`CardioidFamily(p=0.5)`). The pattern is frequency-independent with a null
directly behind the talker (−∞ dB at every frequency). Measured speech
(Monson et al. 2012) is −3.6 dB behind at 125 Hz, −6 dB at 1 kHz and −19 to
−26 dB at 4–8 kHz.

- **Reach:** training scene sampling uses it (`scene/sampling.py`) and draws
  talker yaw uniformly over ±180°, so talkers facing away from the microphone
  are sampled.
- **Available:** `directivity_id="speech_human"` (measured octave-band table,
  realized as a minimum-phase band filter). PathEvent backends only; the
  Pyroomacoustics backend rejects it.

## 3. Discrete diffraction switches at the shadow boundary

`path_events/interactions.py` (`include_scene_interactions=True`):

1. Diffraction paths are created only while the direct segment is blocked. On
   the shadow boundary the direct path (gain 1) disappears and edge paths
   appear at `0.5 · sqrt(1 − α − τ) ≤ 0.5`: a step of at least 6 dB, audible
   for a moving source.
2. Only vertical edges are candidates; the top edge of a low object (sofa,
   desk screen) is never a diffraction path.
3. The edge coefficient `0.5 / sqrt(1 + v²)` is evaluated at one reference
   frequency (1 kHz), so diffraction does not depend on frequency.
4. Reflections that cross an object are hidden, not attenuated.

- **Available:** `occlusion_model="fresnel_kirchhoff"` — a continuous,
  frequency-dependent Fresnel–Kirchhoff screen model on every leg of every
  path, including top and side edges (exclusive with
  `include_scene_interactions`).

## 4. The renderer realizes frequency-dependent gains only for walls

`render_path_events` accepts a frequency-dependent gain spectrum only through
boundary admittance filters (`ComplexPathGainSpectrum.constant_real_value`
raises otherwise). This is why items 2 and 3.3 are frequency-independent.

- **Available:** `PathEvent.band_gain` (`PathBandGain`), a real magnitude per
  frequency realized as a minimum-phase FIR (`path_events/band_filter.py`);
  absent by default, so existing events render unchanged.

## 5. The coupling target vanishes at low image order

`render/coupling.py`, `extrapolated_path_tail_energy_target`. The late energy
is extrapolated from the last active window of the PathEvent tail. With
`max_order=0` there is no tail after the transition and the FDN gain is 0 (no
late field); with order 1 the estimate rests on a handful of reflections. The
coupling does not warn when its target comes from too few paths.

- **Reach:** any caller of `couple_path_event_rir_with_fdn` with a low-order
  path field.
- **Handled:** the moving-source renderer always derives the late field from a
  second-order path field (`LATE_FIELD_ORDER`).
