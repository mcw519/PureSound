# PathEvent–FDN late-field coupling — `puresound.audio.rir.render.coupling`

繁體中文版本：[`rir_late_coupling.zh-TW.md`](rir_late_coupling.zh-TW.md)

Couples a coherent PathEvent early field to a deterministic multiband FDN
tail. Used by the M4 backend
(`render/high_frequency/fdn.py`); design map in
[`rir_realism_algorithm.md`](rir_realism_algorithm.md) §4.3.

## Coupling contract

```python
couple_path_event_rir_with_fdn(
    path_event_rir, sample_rate, direct_sample, target_rt60_s_by_hz,
    *, mixing_time_s=0.024, transition_duration_s=0.016,
    delay_line_count=16, seed=0, filter_order=4,
) -> PathEventFDNCouplingResult  # `.rir`, `.metadata`
```

- The early field is preserved **sample-exact** before the transition window.
- The crossfade is *centered* `mixing_time_s` after the direct sample and
  spans `transition_duration_s` (clamped to stay after the direct sample and
  inside the RIR); it uses `equal_power_transition_weights` over the window
  returned by `transition_samples`.
- Policy string: `PATH_EVENT_FDN_COUPLING_POLICY`.

## Energy matching

`extrapolated_path_tail_energy_target` integrates the material-RT60 decay law
from the truncated PathEvent tail out to the render boundary, so the FDN gain
is anchored to the energy the tail *would* have carried, not to the truncated
sum. `energy_preserving_diffuse_gain` converts that target into the diffuse
gain. Premise: the material decay law holds over the extrapolated span —
revisit if scene RT60s exceed ~1.5 s.

## Validation

Sample-exactness, transition energy preservation, and distinct per-channel
seeds are asserted by
`egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_path_event_fdn_coupling.py`
and `test/test_hybrid_rir.py`.
