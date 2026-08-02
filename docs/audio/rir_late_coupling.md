# `audio.rir_late_coupling`

`puresound.audio.rir_late_coupling` implements the M4.4 causal transition from
coherent PathEvent direct/early response to the deterministic M4.3 multiband
FDN tail. Production use is explicit opt-in through
`--high-backend path-events-m4`; Pyroomacoustics remains the default and
`path-events-m3` remains unchanged.

## Coupling contract

For one PathEvent response `p[n]`, the transition is centered at a configured
mixing time after the physical direct sample. The default center is 24 ms and
the equal-power transition lasts 16 ms.

Within the transition, the weights are:

```text
theta[n] = (pi / 2) * (n - start) / (end - start)
early[n] = cos(theta[n])
late[n]  = sin(theta[n])
early[n] ** 2 + late[n] ** 2 = 1
```

Before the transition, `early=1` and `late=0`; after it, `early=0` and
`late=1`. The FDN excitation is the actual early PathEvent response:

```text
x_fdn[n] = early[n] * p[n]
h[n] = early[n] * p[n] + gain * late[n] * fdn(x_fdn)[n]
```

This gives three testable invariants:

- the coupled response is sample-exact with the coherent response through the
  transition start;
- causal PathEvents and causal FDN/filter stages cannot create pre-arrival
  output;
- later sparse paths cannot continue injecting the FDN after the transition.

## Energy matching

Independent RMS matching is insufficient because the weighted coherent and
FDN components can have a nonzero cross term. M4.4 solves the positive root of

```text
gain**2 * E_fdn + 2 * gain * cross + E_early = E_original
```

over the finite response from the transition start onward. The reported
post-transition relative energy error is required to be at most `1e-10`.
This preserves total finite-window energy, not instantaneous or per-octave
energy. The late spectral shape comes from the material-derived octave RT60
targets.

## Room and channel parameters

`PathEventFDNHighFrequencyBackend` obtains RT60 targets from
`RoomSceneV2.predicted_octave_rt60_s()`. It retains full octave bands below
Nyquist and at or above both 500 Hz and half the low/high crossover frequency.
Lower frequencies remain the responsibility of the modal branch and hybrid
crossover.

Each source channel gets a stable seed derived from the configured base seed,
serialized scene ID, and source index with BLAKE2. This avoids process-random
Python hashes and prevents every source channel from sharing identical FDN
signs and output taps.

## API

| Symbol | Purpose |
|--------|---------|
| `equal_power_transition_weights(...)` | Build exact cosine/sine transition weights. |
| `couple_path_event_rir_with_fdn(...)` | Couple one coherent RIR to an energy-matched multiband FDN tail. |
| `PathEventFDNCouplingResult` | Return the coupled response, separated components, weights, design, and metadata. |
| `PathEventFDNHighFrequencyBackend` | Generate all source channels from a material-first scene with M4 coupling. |

Generate an opt-in M4 item:

```bash
PYTHONPATH=. python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m4/path_event_fdn \
  --scene-version v1 --room-type office \
  --high-backend path-events-m4 \
  --fdn-mixing-time-ms 24 --fdn-transition-ms 16 \
  --fdn-delay-lines 16 --fdn-seed 20260731 \
  --low-backend analytic-material \
  --sample-rate 16000 --duration 1.2 \
  --n-rooms 1 --rir-per-room 1
```

## Validation

Run the deterministic formal gate:

```bash
PYTHONPATH=. python \
  egs/rir_generation/phases/m4_spatial_late_field/scripts/validate_path_event_fdn_coupling.py \
  --target-report \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json \
  --output-report \
    egs/rir_generation/phases/m4_spatial_late_field/reports/m4_path_event_fdn_coupling_report.json \
  --output-dir egs/rir_generation/exp/rir_realism/m4/rir_m4_coupling
```

The frozen fixture passes all 8 coupling structure gates, all qualified
500 Hz–4 kHz decay/mixing/density gates, and all 5 complete-hybrid
compatibility gates. M4 median mixing times are 22/26/30/34 ms at
500/1000/2000/4000 Hz, compared with no stable M3 crossing at 2/4 kHz.

This is one deterministic material-first fixture, not a measured-room or
perceptual coloration exit. The current channels represent different sources
at one receiver, so M4.5 must add synchronized multi-receiver/Ambisonic output
before coherence or IACC can be evaluated.
