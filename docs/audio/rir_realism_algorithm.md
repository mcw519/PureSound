# RIR implementation guide

繁體中文版本：[rir_realism_algorithm.zh-TW.md](rir_realism_algorithm.zh-TW.md)

This page maps the RIR pipeline to its implementation. For runnable examples,
see [RIR generation](../../egs/rir_generation/README.md).

Policy strings such as `puresound.iso9613_1.minimum_phase_direct.v2` identify
observable behavior. A behavior change requires a new policy version so
generated banks remain traceable.

## Pipeline

```text
RoomSceneV2
  ├─ low-frequency renderer
  ├─ high-frequency renderer
  └─ crossover and level calibration
       └─ optional FOA, array, or binaural output
            └─ bank manifest, QC, and release
```

| Area | Package |
|---|---|
| Contracts and array layout | `puresound.audio.rir.contracts` |
| Scene schema and sampling | `puresound.audio.rir.scene` |
| Propagation and impedance | `puresound.audio.rir.physics` |
| Coherent reflection paths | `puresound.audio.rir.path_events` |
| Render backends and assembly | `puresound.audio.rir.render` |
| Acoustic metrics | `puresound.audio.rir.metrics` |
| Measurement calibration | `puresound.audio.rir.calibration` |
| Training-bank packaging | `puresound.audio.rir.bank` |

Import the layer you need. `puresound.audio.rir.api` is a convenience module,
not a long-term compatibility boundary.

## Scene model

`RoomSceneV2` stores geometry, surface materials, environment, sources,
receivers, and optional objects. RT60 is derived from those causes; it is not
an independent sampled parameter.

Important behavior:

- Material absorption and scattering use seven octave bands from 125 Hz to
  8 kHz.
- Surface patches are area-weighted into an effective boundary material.
- Sound speed is derived from temperature and humidity unless supplied.
- `predicted_octave_rt60_s()` uses a Sabine estimate.
- Upgrading a v1 scene does not copy its RT60 into v2.

The Sabine estimate assumes a diffuse field and only accounts for boundary
surfaces. It is useful metadata, but it is not a reliable render target when
absorption is highly localized or furniture dominates the room.

## Low-frequency rendering

Available implementations live in `render/low_frequency/`:

| Backend | Use |
|---|---|
| `analytic_modal.py` | Lossless reference |
| `impedance_modal.py` | Experimental complex-impedance modes |
| `pytard.py` | DCT-grid wave solve with optional material modal damping |

Material modal damping assigns each mode a decay rate from its participation
at the six room boundaries. The recurrence uses damped modal frequency and a
pole radius of `exp(-gamma * dt)`. Samples before the geometric arrival are
zeroed.

The pytARD output is not on an absolute physical scale. Hybrid rendering uses
the crossover-band energy match to align it with the high-frequency path.
The impedance-modal path remains research infrastructure.

## High-frequency rendering

### Pyroomacoustics

`render/high_frequency/pyroomacoustics.py` uses image sources and ray
tracing. V2 scenes pass the full material spectrum to pyroomacoustics.

Ray tracing uses process-global randomness inside libroom. A seed alone does
not guarantee byte-identical output.

### PathEvents

`render/high_frequency/path_event.py` builds coherent image-source paths and
renders each path with:

- geometric delay and attenuation;
- passive boundary filters;
- source and receiver directivity;
- ISO 9613-1 air absorption;
- optional object occlusion.

Fractional delays are causal by construction. Object occlusion affects the
direct and early window; it does not suppress the whole late field.

### PathEvents with FDN

`render/high_frequency/fdn.py` preserves the PathEvents early field and adds
a deterministic multiband feedback delay network. The target decay comes
from the scene's octave-band RT60 estimate with air-loss correction.

The early and late fields use an equal-power transition after the direct
arrival. A per-channel seed is derived from the FDN seed, scene ID, and
channel.

This renderer follows its target accurately; errors in the Sabine-derived
target therefore remain visible in the output.

## Crossover and output

`render/crossover.py` combines the bands with a causal fourth-order
Linkwitz–Riley crossover, normally at 1 kHz. Low-band RMS is matched to the
high band over the configured match range, with a bounded gain.

`render/hybrid.py::generate_hybrid_rir()` then:

1. renders both bands;
2. aligns and clips samples before physical arrival;
3. applies the crossover and tail fade;
4. applies either calibrated or peak-normalized output policy.

Use calibrated output when physical level relationships matter. It may have a
peak above 1. Use peak-normalized output only when that normalization is part
of the data contract.

## Spatial output

| Output | Implementation |
|---|---|
| FOA and room arrays | `render/spatial.py` |
| Synchronized-array late field | `render/spatial_late_field.py` |
| FOA to binaural RIR | `render/binaural.py` |

FOA uses ACN channel order and SN3D normalization. Array channels share one
late-field gain so spatial ratios are preserved. Spatial coherence and IACC
are meaningful only for synchronized receivers.

## Metrics

`metrics.analyze_rir()` produces the combined report. Individual modules
provide:

- Schroeder decay, T20/T30, and fit quality;
- C50, C80, and DRR;
- spectral tilt and octave-band analysis;
- Lundeby noise-floor estimation;
- echo density and mixing time;
- binaural IACC and array coherence.

Do not use a decay estimate without checking its fit quality and available
dynamic range.

## Measured RIR ingest

`bank/measured_ingest.py` converts published measured RIRs into the bank
contract. Many corpora place direct sound at time zero, while PureSound uses
emission time as zero. Ingest restores the geometric propagation delay and
records the assumed environment when the corpus does not provide one.

An item is rejected when alignment would reveal an earlier distinct arrival,
discard too much energy, or require an implausible time shift.

CLI:

```bash
python egs/rir_generation/phases/m6_bank/scripts/ingest_measured_m6_variant.py --help
```

## Calibration and banks

Measurement and inverse-calibration contracts live in
`puresound.audio.rir.calibration`; see
[Measurement campaign](rir_measurement_campaign.md).

Bank manifests, split indexes, QC, release recipes, and production evidence
live in `puresound.audio.rir.bank`; see [RIR bank v2](rir_bank_v2.md).
