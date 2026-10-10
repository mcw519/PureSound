# PathEvent–FDN late-field coupling — `puresound.audio.rir.render.coupling`

繁體中文版本：[rir_late_coupling.zh-TW.md](rir_late_coupling.zh-TW.md)

Joins a coherent PathEvent early field to a [multiband FDN](multiband_fdn.md)
late field for one channel. It is the core of the `path-events-m4` backend
(`render/high_frequency/fdn.py`); the [spatial renderer](spatial_rir.md) uses the
same pieces for synchronized arrays. Policy string:
`PATH_EVENT_FDN_COUPLING_POLICY = "puresound.path_event_fdn_coupling.v1"`.

## Why couple

PathEvents give exact, inspectable direct and early arrivals, but an image-source
enumeration becomes sparse late in the response and stops at its maximum order.
An FDN gives a dense tail with the right decay but no specific early structure.
The coupling keeps the early field exactly and replaces what follows with the
FDN, at the energy the path field would have carried.

## Contract

```python
couple_path_event_rir_with_fdn(
    path_event_rir, sample_rate, direct_sample, target_rt60_s_by_hz,
    *, mixing_time_s=0.024, transition_duration_s=0.016,
    delay_line_count=16, seed=0, filter_order=4,
) -> PathEventFDNCouplingResult
# .rir, .coherent_component, .diffuse_component, .early_weight, .late_weight,
# .design, .metadata
```

`direct_sample` is the geometric arrival sample; `target_rt60_s_by_hz` maps
octave centers (Hz) to RT60 (s).

## Algorithm

1. **Transition window** (`transition_samples`). Centre
   `n_c = n_d + round(t_mix · fs)`, width `W = max(2, round(t_trans · fs))`,
   start `max(n_d + 1, n_c − W/2)`. The window is moved left if it would run past
   the end; the call raises if the RIR is too short for a transition after the
   direct sample.
2. **Weights** (`equal_power_transition_weights`). Across the window
   `w_e = cos φ` and `w_l = sin φ`, φ from 0 to π/2, so `w_e² + w_l² = 1`;
   `w_e = 1` before the window and 0 after it.
3. **FDN excitation.** The FDN is designed from the targets (same mixing time,
   given seed) and driven by `h_pe · w_e`: the direct and early events seed the
   late state, while paths after the transition cannot keep injecting a sparse
   coherent pattern.
4. **Energy target** (`extrapolated_path_tail_energy_target`). The PathEvent
   energy from the transition start to the end of the RIR, plus the energy the
   path field would still have carried after its last event: the mean energy per
   sample of the final 50 ms of events, decaying at the median target RT60 to the
   end of the render. This anchors the tail to the decay law rather than to a
   truncated sum, on the premise that the material decay holds over the
   extrapolated span.
5. **Gain** (`energy_preserving_diffuse_gain`). With coherent part
   `c = h_pe · w_e` and diffuse part `f = FDN · w_l`, `g` is the positive root of
   `‖c + g f‖² = E_target` over the post-transition samples:

   ```text
   g = (−⟨c, f⟩ + √(⟨c, f⟩² + ‖f‖² (E_target − ‖c‖²))) / ‖f‖²
   ```

6. **Output** `h = c + g f`. Every sample up to the transition start equals the
   PathEvent input exactly; the metadata records the maximum deviation there, the
   transition samples, the energies, the extrapolation and the FDN design.

## Use in the backend

`PathEventFDNHighFrequencyBackend` (a subclass of the PathEvent backend) renders
the coherent field, then couples each source channel with:

- targets from `RoomSceneV2.predicted_octave_rt60_s()` corrected for air
  absorption (`air_adjusted_rt60_s`: `60 / (60 / RT60 + a(f) · c)`, with `a` the
  ISO 9613-1 attenuation in dB/m), keeping octaves below Nyquist and at or above
  `max(minimum_fdn_center_hz, 0.5 × crossover_hz)` (500 Hz by default);
- `direct_sample = round(d / c · fs)`;
- a per-channel seed from a BLAKE2b hash of the FDN seed, scene id and source
  index, so the channels of one room have independent tails and the same scene
  always renders the same tail.

`generate_hybrid_rir.py` exposes `--fdn-mixing-time-ms`, `--fdn-transition-ms`,
`--fdn-delay-lines` and `--fdn-seed`.

## Tests

`test/rir/test_rir_late_coupling.py` pins exact early preservation,
post-transition energy, seed determinism and the long-RT60 extrapolation;
`test/rir/test_hybrid_rir.py` checks that the backend keeps the coherent
early paths and serializes the coupling.
