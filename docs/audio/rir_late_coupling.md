# PathEvent–FDN late-field coupling — `puresound.audio.rir.render.coupling`

Couples a coherent PathEvent early field to a deterministic multiband FDN
tail. Used by the M4 backend
(`render/high_frequency/fdn.py`); design map in
[`rir_realism_algorithm_zh-TW.md`](rir_realism_algorithm_zh-TW.md) §4.3.

## Coupling contract

`couple_path_event_rir_with_fdn(rir, sample_rate=..., direct_sample=...,
target_rt60_s_by_hz=..., mixing_time_s=..., transition_duration_s=..., ...)`
→ `PathEventFDNCouplingResult` (`.rir`, `.metadata`).

- The early field is preserved **sample-exact** before the transition window.
- The crossfade starts `mixing_time_s` after the direct sample and uses
  `equal_power_transition_weights` over `transition_samples`.
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
