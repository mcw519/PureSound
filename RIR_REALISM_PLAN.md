# Physically Grounded RIR Realism Plan

Status: active  
Primary target: synthetic-to-real room acoustics for near/far speech training  
Secondary target: a reusable path toward microphone arrays, Ambisonics, and BRIR rendering

繁體中文的整體算法、物理解釋、驗證方法與目前結論見
[`docs/audio/rir_realism_algorithm_zh-TW.md`](docs/audio/rir_realism_algorithm_zh-TW.md)。

Experiment organization follows the milestone layout documented in
[`egs/rir_generation/phases/README.md`](egs/rir_generation/phases/README.md):
phase scripts and frozen evidence live under `phases/m0_baseline` through
`phases/m6_bank`, while `egs/rir_generation/` itself contains only stable
public commands.

## 1. Objective

PureSound already combines a low-frequency wave model with a high-frequency
geometric model. The next generation should make room-acoustic observables such
as RT60, DRR, early reflections, modal decay, and spatial coherence emerge from
the sampled scene rather than treating a broadband RT60 as the scene's primary
physical parameter.

The intended production model is:

```text
scene geometry + materials + environment + transducers
    -> low-frequency lossy wave response
    -> coherent direct and early path events
    -> spatial, multiband late field
    -> optional measured-room calibration / learned residual
    -> mono, array, Ambisonic, or binaural RIR
```

The project is not attempting a full-band brute-force wave solve for every
training sample. Wave, path, and statistical methods will be used where their
assumptions are valid and where their cost is justified.

## 2. Definition of success

No single waveform distance or RT60 number is sufficient. A release candidate
must pass all four gates below.

### 2.1 Physical consistency

- Causal direct arrival consistent with source-receiver distance and sound speed.
- Stable relative gain when peak normalization is disabled.
- Reciprocity within the limits of source and receiver directivity.
- No unexplained energy gain at reflections or across solver crossovers.
- Smooth change of delay, energy, and direction across nearby positions.

### 2.2 Acoustic distribution

Synthetic and measured banks are compared using at least:

- direct-to-reverberant ratio (DRR);
- EDT, T20, and T30 in octave bands;
- C50 and C80;
- spectral decay and coloration;
- low-frequency modal frequencies and Q;
- echo density;
- spatial/inter-channel coherence when applicable.

### 2.3 Downstream synthetic-to-real performance

Every material generator change is evaluated with a fixed training recipe on
room-disjoint measured RIR test sets. A change is accepted only when it improves
the primary speech task consistently across multiple real-room subsets or when
it fixes a demonstrated physical defect without causing a statistically
meaningful regression.

### 2.4 Operational quality

- Deterministic scene sampling for a fixed seed.
- Versioned metadata with complete provenance.
- Resume-safe dataset generation.
- Measured and generated RIRs share the same analysis and bank interfaces.
- Generation cost is reported together with acoustic quality.

## 3. Architecture principles

1. **Material first.** Sample geometry and frequency-dependent material
   properties; measure the realized decay afterwards. A requested RT range may
   constrain or reject scenes, but must not be imposed as a global envelope.
2. **Preserve causes.** Direct and early reflections should remain identifiable
   path events until the final RIR assembly.
3. **Separate acoustics from transducers.** Room propagation, source
   directivity, microphone response, HRTF, and device coloration are separate
   composable stages.
4. **Keep absolute and normalized modes distinct.** Dataset recipes may still
   request peak-normalized RIRs, but physically calibrated gain must remain
   available and recorded.
5. **Learn residuals, not basic physics.** Neural generation is introduced only
   after a measured baseline identifies error that the physical model cannot
   cover economically.
6. **Benchmark before replacing.** Every new backend runs beside the current
   hybrid generator until it passes the same benchmark.

## 4. Milestones

The estimates below are approximate single-engineer effort, not calendar
commitments.

### M0 — Measurement and benchmark foundation (1–2 weeks)

Goal: turn the current generator into a frozen, reproducible baseline and make
the synthetic-to-real gap measurable.

Deliverables:

- a reusable `puresound.audio.rir_metrics` module;
- one analysis path for generated and measured bank WAVs;
- per-channel and per-distance-bucket JSON summaries;
- a frozen v0 generation configuration and seeds;
- a baseline report comparing synthetic, measured, and mixed training;
- tests using analytic decay signals and small fixture banks.

Initial metrics:

- direct sample and peak;
- DRR using the training pipeline's configurable direct window;
- C50 and C80;
- broadband EDT, T20, and T30 with fit quality;
- magnitude-response tilt;
- octave-band EDT/T20/T30/C50.

Exit gate:

- the benchmark is deterministic;
- invalid or insufficient decay fits are explicit rather than silently replaced;
- the largest measured/synthetic gaps can be ranked by metric and frequency
  band;
- one fixed downstream v0 result is recorded.

### M1 — Material-first scene schema (completed 2026-07-30)

Goal: replace the single broadband room RT60 input with physical surface and
environment properties.

Introduce a versioned scene schema containing:

- named surfaces and meshes;
- octave or third-octave absorption, scattering, and transmission;
- optional complex surface impedance;
- correlated material families for walls, floor, ceiling, windows, doors, and
  furniture;
- temperature, humidity, pressure, and sound speed;
- source power, pose, and directivity identity;
- receiver pose, pattern, calibration, and array identity.

Implementation steps:

1. Add schema types and JSON serialization without changing the v0 generator.
2. Add a documented material catalog with provenance and uncertainty ranges.
3. Sample plausible material combinations by room type.
4. Add frequency-dependent materials to the high-frequency backend.
5. Record realized octave-band decay instead of copying a requested scalar RT60.
6. Add a calibrated output mode that does not normalize each item to a fixed
   peak.

Exit gate:

- material changes produce the expected band-dependent decay;
- metadata round-trips without loss;
- v1 and legacy banks remain readable;
- M0 shows an improvement over the broadband-material baseline.

Implementation:

- [x] Added `rir_scene.v2` types for named boundary meshes, material spectra,
  window/door patches, optional complex impedance, environment, poses, source
  power/directivity, receiver calibration, arrays, and interior objects.
- [x] Added lossless JSON serialization plus legacy `room_dim`, `mic_pos`,
  `source_pos`, `channel_map`, and material-derived `rt60` compatibility fields.
- [x] Added the documented `puresound-materials.v1` catalog and correlated
  office, meeting-room, classroom, and living-room sampling.
- [x] Passed per-boundary frequency-dependent absorption and scattering to
  Pyroomacoustics; temperature/humidity and scene sound speed now participate in
  rendering.
- [x] Separated material-predicted octave RT60 from realized broadband/octave
  metrics measured on the output RIR.
- [x] Added calibrated output that preserves source level, receiver gain,
  distance, and cross-room amplitude without per-item peak normalization.
- [x] Kept v0 as the CLI default and verified a generated v1 bank with the
  existing `PreGeneratedRoomBank` reader.

M1 acoustic probe:

- report: `egs/rir_generation/exp/rir_realism/m1/rir_benchmark_m1_probe100/acoustics.json`;
- 20 rooms × 5 positions = 100 v1 items/channels, seed 1337;
- analytic low-band probe backend, Pyroomacoustics high band, 16 kHz, 1.0 s;
- wall time: 156 s with four CPU workers (0.64 items/s on this host);
- measured comparison: 100 channels, seed 0.

| Distance | v0 tilt | M1 tilt | measured tilt | v0→M1 absolute-gap reduction |
|----------|---------|---------|---------------|------------------------------|
| 0–1 m | +1.12 | -0.25 | -2.51 | 3.63 → 2.26 dB/oct (38%) |
| 2–3.5 m | +2.50 | +0.44 | -3.32 | 5.82 → 3.76 dB/oct (35%) |
| 3.5–6 m | +3.31 | +0.62 | -2.52 | 5.83 → 3.14 dB/oct (46%) |

The comparison is directional rather than a downstream acceptance result
because v0 and M1 use different-sized samples and the probe uses the analytic
low backend. It nevertheless passes the M1 targeted acoustic gate: the
synthetic upward spectral tilt is reduced in every populated distance bucket.
The C50 gap did not improve (M1 medians 9.84/3.88/3.82 dB versus measured
17.16/9.31/6.02 dB), so early/late path energy remains an explicit next target
rather than being attributed to materials.

M1 audition acceptance:

- recipe: `egs/rir_generation/phases/m1_material/config/audition_m1.json`;
- artifacts: `egs/rir_generation/exp/rir_realism/m1/rir_m1_audition_v1_bank` and
  `egs/rir_generation/exp/rir_realism/m1/rir_m1_audition_v1`;
- 10 rooms × 5 positions = 50 items and 250 RIR channels;
- all WAV/metadata, finite-sample, calibrated-peak, tail-energy, and physical
  `distance/c` causality checks pass;
- the gate found and fixed Pyroomacoustics fractional-delay/ray-tail energy
  left at `t=0` after direct-path alignment; regenerated RIRs have exactly zero
  samples before each source's physical arrival;
- median near/far DRR gap is 10.63 dB and valid T20 coverage is 79.6% at
  `R² >= 0.9`;
- six RT60-quantile rooms have dry plus shared-gain `near_0`/`far_0` listening
  previews. This passes the local usability gate, not measured-room or M2
  acceptance.

### M2 — Lossy low-frequency wave model (active since 2026-07-30)

Goal: make low-frequency modal decay emerge from boundaries instead of a shared
post-hoc exponential envelope.

The first implementation extends the current modal recurrence to damped modes:

```text
q_n'' + 2 * zeta_n * omega_n * q_n' + omega_n^2 * q_n = f_n
```

`zeta_n` is derived from modal boundary participation and the material loss at
the modal frequency.

Implementation steps:

1. Derive and test a discrete damped modal recurrence.
2. Compute per-mode loss from the six room surfaces.
3. Remove the global low-band RT60 envelope in the v1 path.
4. Validate modal frequency and Q against analytic cases.
5. Compare small scenes with a reference FDTD/FEM solver.
6. Decide from measured error whether a full impedance-boundary ARD backend is
   justified.

Exit gate:

- different modes can have different decay rates;
- changing one surface affects the expected modes and frequency bands;
- crossover energy remains bounded and causal;
- M0 low-band metrics improve without a downstream regression.

Current implementation status:

- [x] Derived an exact sampled damped recurrence that reduces to the previous
  pytARD recurrence when damping is zero.
- [x] Derived per-mode amplitude loss from six frequency-dependent surfaces and
  rigid-wall cosine-mode boundary participation.
- [x] Added explicit `pytard-material`, `pytard-cupy-material`, and
  `analytic-material` experimental backends.
- [x] Disabled the shared low-band RT60 envelope in material-modal mode and
  serialized modal frequency, decay rate, RT60, and Q.
- [x] Verified surface selectivity and absorbing-versus-reflective late-energy
  behavior with the exact pytARD recurrence.
- [x] Added an independent staggered pressure/velocity 3D FDTD reference with
  locally reacting impedance boundaries.
- [x] Validated the first two axial frequencies within 1% and their Q within
  20% of the impedance-boundary prediction; the estimator recovers an analytic
  damped-sinusoid Q within 2%.
- [x] Added a response-level low-frequency peak, bandwidth, and Q estimator plus
  a generated/measured bank comparison CLI.
- [x] Replaced the analytic probe's artificial rank amplitude/position phase
  with reciprocal eigenfunction source/receiver coupling and causal onset.
- [x] Replaced its fixed first-64 mode truncation with a serialized 256-mode cap,
  covering all 215 default index triplets and restoring 200–300 Hz coverage.
- [x] Ran development and zero-item-overlap measured modal comparisons; the
  corrected M1 bridge improves Q/spacing/bandwidth distributions, while the
  current material-modal loss and a scalar calibration both fail the joint
  distribution gate.
- [x] Established the phase-aware complex-impedance foundation in Pa·s/m:
  passive `Z <-> Gamma`, absorption-plus-explicit-phase conversion, schema
  interpolation/round-trip validation, and area-patch admittance mixing. No
  impedance is inferred when only absorption is known.
- [x] Added a positive-real first-order relaxation admittance, bilinear
  time-domain reflection filter, and per-wall-cell FDTD boundary state.
  Single-wall magnitude/phase, digital-pole stability, the exact static-boundary
  limit, and a nonzero-relaxation 3D case are covered by tests.
- [x] Added two provenance-bearing 100 mm glass-wool references using Tarnow's
  measured flow resistivity and the Miki phase-aware porous-layer model. Both
  pass the one-pole complex-reflection fit gate without extrapolating below
  their empirical validity range.
- [x] Demonstrated in a controlled FDTD room that a phase-aware fit moves the
  dominant peak from 85.7 Hz/Q 10.5 to 64.0 Hz/Q 15.8 versus a boundary with
  matched 80 Hz reflection magnitude but zero phase.
- [x] Added a boundary-admittance time-step gate after the first fitted prior
  exposed an edge/corner instability not covered by the interior CFL limit.
- [x] Added a strict normal-incidence complex-impedance CSV/JSON ingestion
  contract with phase, uncertainty, SI units, sample configuration, source,
  license, and passive-reflection validation.
- [x] Added a positive-real multi-pole admittance and fixed-real-pole bounded
  fit in complex pressure-reflection space. Non-negative static, low-pass, and
  high-pass parallel branches enforce passivity by construction.
- [x] Generalized the validation FDTD to one auxiliary state per wall cell and
  pole, including multi-pole metadata and the boundary time-step gate.
- [x] Connected the same rational reflection function to a minimal 1D complex
  cavity eigenproblem. The static case matches closed-form frequency/decay/Q,
  and a phase-aware boundary shifts the mode relative to a magnitude-matched
  real boundary.
- [x] Added the first licensed direct complex-impedance benchmark: CC BY 4.0
  normalized resistance/reactance for nominally identical NASA/UFSC
  perforated liners at no flow and 130 dB. It is explicitly validation-only
  because the grazing-duct, high-SPL configuration is not a room finish.
- [x] Added a passive series-RLC resonant admittance after the measured
  Helmholtz reactance crossing demonstrated that relaxation-only real poles
  fail. Alternating-frequency holdout, dense passivity, FDTD biquad state, and
  1D modal diagnostics pass.
- [x] Added a strict two-microphone impedance-tube acquisition path for the
  missing room-finish data: repeated complex H12, microphone-switch
  calibration, circular-tube and spacing validity gates, coherence,
  repeatability uncertainty, passive reduction, and uncertainty-weighted
  complex-reflection fitting.
- [ ] Obtain compatible normal-incidence room-finish measurements and map only
  matching installed configurations into the scene catalog.
- [x] Extend the accepted 1D rational-boundary formulation to a separable 3D
  nonlinear eigenproblem, connect it to an explicit experimental low-band
  renderer, and cross-check its first mode against independent FDTD.
- [ ] Validate nonlinear modal residues, ingest a compatible measured
  room-finish boundary, and only then rerun the corrected room-disjoint modal
  comparison as a production-candidate backend.
- [ ] Run the fixed downstream synthetic-to-real experiment before making the
  material-modal backend a default.

The implementation is usable for experiments but M2 has not passed its exit
gate. The default material-modal loss law is still the rejected first-order
Sabine surface-participation model. Complex phase now drives FDTD, 1D and
separable 3D nonlinear eigenvalues, and an explicit experimental renderer. The
first direct dataset validates this pipeline but is not a compatible room
material; normal-incidence room-finish data, nonlinear modal-residue
validation, angle dependence, and non-shoebox mode coupling remain required.

M2.5 direct-measurement gate:

- source: Zenodo `10.5281/zenodo.15195587`, CC BY 4.0;
- converted data: NASA and UFSC no-flow, 130 dB, KT normalized impedance,
  500–2500 Hz, with HDF5 dataset paths and checksum in each sidecar;
- report:
  `egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/impedance_zenodo_15195587_validation.json`;
- fitted NASA resonance: 1646.66 Hz, branch Q 11.69;
- held-out complex Cayley error: RMS 0.0385, maximum 0.0590;
- 4096-point passivity sweep: maximum magnitude 0.9178;
- NASA–UFSC cross-rig RMS difference: 0.1182;
- 1D phase-aware versus magnitude-only Q ratio: 2.94;
- decision: pass the pipeline-validation gate, reject automatic room-material
  catalog mapping.

M2.6 room-finish acquisition gate:

- second public-source audit found no machine-readable room-finish dataset
  satisfying phase, mounting, atmosphere/SPL, and license requirements;
- implemented
  `puresound.impedance_tube_transfer_measurement.v1` for repeated complex H12;
- microphone-switch calibration removes complex channel mismatch with a
  continuous square-root branch;
- selected bins must remain below the circular-tube transverse-mode cutoff,
  away from microphone-spacing singularities, and above the coherence gate;
- repeated installations produce real/imaginary impedance standard deviations,
  which are propagated to inverse-uncertainty reflection-domain fit weights;
- exact synthetic H12/channel-mismatch/impedance round-trip and CLI output
  contract are tested;
- decision: pass the acquisition-software gate; physical room-finish specimens
  and material-disjoint validation are still required before scene mapping.

M2.7 separable 3D impedance-mode gate:

- each wall receives an explicit passive rational admittance; no phase is
  inferred from the scene absorption catalog;
- one temporal pole jointly satisfies three complex Robin boundary
  characteristics and the 3D wave-equation dispersion relation;
- continuation from weak to full boundary strength tracks each requested
  rigid-wall mode branch;
- static x-only loss matches the existing exact 1D frequency, decay, and Q;
- a uniform cube preserves the three axial permutation degeneracies;
- the phase-aware 2.0 x 1.2 x 1.0 m glass-wool reference predicts
  63.8846 Hz / Q 17.32 versus independent FDTD 64.0049 Hz / Q 15.78;
- `analytic-impedance` is wired through the dataset generator with a strict
  six-wall JSON and validity-band check; a one-item full CLI smoke passes;
- eigenvalue and separable eigenfunction calculations pass, but the current
  RIR modal residue uses a documented engineering scale and is not yet
  production-validated;
- decision: pass the 3D eigenvalue/integration gate, keep production mapping
  and room-disjoint bank comparison blocked on real material and residue data.

M2.8 fixed-pole modal-residue gate:

- keep every M2.7 complex pole and eigenfunction fixed; fit only one global
  complex scale and one smooth frequency exponent;
- convolve the positive-pole quadrature bases with the exact FDTD Ricker
  source, use snapped cell centers and the pressure-cell volume convention;
- compare target and basis in the same 60–240 Hz validity band and only after
  source excitation, rather than fitting out-of-band FDTD energy;
- use two shoebox geometries with four training positions and two untouched
  position holdouts; each axial fundamental remains inside boundary validity;
- fitted `C = 0.444175 + j0.129146`, exponent `0.35`;
- train mean correlation / NRMSE / energy ratio:
  `0.9916 / 0.1299 / 0.9831`;
- position-holdout mean correlation / NRMSE / energy ratio:
  `0.9734 / 0.2304 / 1.0039`;
- the fitted-gain legacy `1/f` sine baseline reaches only
  `0.2467 / 0.9703 / 0.0913` on the same holdout;
- the versioned calibration is wired into `analytic-impedance`; a complete
  one-item generator smoke records `modal_residue_fdtd_validated: true`;
- decision: pass the controlled single-boundary numerical residue gate; keep
  `modal_residue_production_validated: false` until multiple installed
  boundaries and measured-room transfer functions pass.

M2.9 multi-boundary / room / grid residue gate:

- add a 50 mm rigid-backed installation-thickness variant using the same
  measured Tarnow flow resistivity and Miki model as the 100 mm reference;
  its one-pole fit has RMS/max complex-reflection error `0.00313 / 0.00506`;
- retain four training positions in rooms A/B, separate two position
  holdouts, and add an entirely unseen room C with two room holdouts;
- rerun one room-C geometry at 0.08 m after all training uses 0.12 m, creating
  an explicit grid holdout rather than silently changing numerical resolution;
- every position/room/grid split must reach correlation >=0.90, NRMSE <=0.35,
  and improve over the fitted-gain legacy `1/f` sine baseline;
- 50 mm position/room/grid correlation:
  `0.9904 / 0.9929 / 0.9961`; NRMSE: `0.1366 / 0.1187 / 0.0919`;
- 100 mm position/room/grid correlation:
  `0.9734 / 0.9933 / 0.9929`; NRMSE: `0.2304 / 0.1177 / 0.1193`;
- separate fits give `C50=0.639407+j0.060622, eta50=0.25` and
  `C100=0.444175+j0.129146, eta100=0.35`;
- a diagnostic shared fit across both variants also passes all three holdouts:
  position/room/grid correlation `0.9766 / 0.9882 / 0.9924`, NRMSE
  `0.2561 / 0.2161 / 0.1838`;
- the shared result is scoped only to two model-derived thickness variants of
  one measured flow resistivity; general boundary invariance remains false;
- a complete 50 mm generator smoke renders 123 calibrated modes and preserves
  `modal_residue_production_validated: false`;
- decision: pass numerical thickness, unseen-room and grid-transfer gates;
  next require a different directly measured room-finish material and a
  measured-room transfer function.

M2.10 source-convention and sample-rate gate:

- identify the FDTD calibration input as a pressure increment added to one
  finite-volume cell, rather than a digital source whose free-field RIR is
  `delta[n-D]/r`;
- derive and implement the free-field mapping
  `x = cell_volume/(dt*4*pi*c^2) * dq/dt`;
- convert each fitted pressure-state modal residue with
  `4*pi*c^2/(fs*pole)` before rendering it beside the `1/r` direct tap;
- version the fitted source, rendered RIR, and residue transform conventions;
  bump calibration output to v2 while retaining v1 read compatibility;
- matched-boundary direct-path validation passes axis-aligned 50/40 mm grids
  and a diagonal 30 mm grid: minimum correlation `0.9685`, maximum theoretical
  amplitude error `10.8%`, and zero best-fit sample offset;
- modal convolution identity passes at 4/8/32 kHz with maximum NRMSE
  `0.00591`; 60–240 Hz complex response across 4/8/16 kHz differs from the
  32 kHz reference by at most `4.07%`;
- decision: pass the FDTD-to-digital-RIR source-convention gate. The `1/r`
  direct tap and calibrated modal tail now share one input definition; absolute
  transducer pressure and low/high crossover level remain uncalibrated.

M2.11 complex crossover and gain-policy gate:

- find and remove the fixed 700–1300 Hz assumption: automatic RMS audit bands
  now track `[0.7*crossover, 1.3*crossover]`, while explicit bands must contain
  the actual crossover;
- demonstrate that branch-RMS equality is not a physical calibration rule:
  identical low/high inputs still produce a low gain near `0.9505` and up to
  `0.44 dB` flat-sum error;
- when the impedance backend reports a validated `1/r` source convention,
  bypass per-channel low-band energy rescaling and serialize the requested,
  applied, policy, effective band, and channel gains in item metadata;
- retain bounded RMS matching only as an explicitly reported bridge for
  uncalibrated legacy backends;
- validate fourth-order causal Linkwitz–Riley complex complementarity at
  8/16/48 kHz: maximum magnitude error `3.21e-12 dB` and branch phase
  difference `1.42e-12 degrees`;
- combine a low `1/r` direct tap with an anechoic Pyroomacoustics direct path
  at five distances. Across 60–1000 Hz, maximum magnitude/phase errors are
  `1.049 dB / 5.39 degrees`, passing `1.1 dB / 6 degrees`;
- decision: pass the digital-filter and anechoic direct crossover gate.
  Full-room modal/geometric phase continuity, boundary-reflection phase, and
  measured-room crossover remain open.

M2.12 full-room complex-crossover failure baseline:

- compare one train position, one position holdout, and one unseen-room
  holdout against the same pressure-cell FDTD transfer, without fitting gain;
- keep the 100 mm M2.10 boundary/residue pair and snapped FDTD source/receiver
  cell centers; source magnitude remains well conditioned across 168–300 Hz;
- enumerate exact shoebox image geometry through order 12 (2625 images) and
  test magnitude-only normal, complex normal, and locally reacting
  angle-aware complex reflection hypotheses;
- modal-only 60–240 Hz still passes the planned complex gate:
  correlation `0.934`, NRMSE `0.363`, energy ratio `0.919`;
- the 240 Hz magnitude-only full-room proxy fails all three cases:
  correlation `0.420`, NRMSE `4.285`, energy ratio `3.952`;
- the modal transfer itself becomes over-energetic near the upper calibration
  edge: 168–300 Hz raw/low-pass energy ratios are `5.84 / 3.67`;
- angle-aware complex reflection materially changes high-branch energy but
  does not make the hybrid pass;
- a 120/150/180/210/240 Hz scan selects 120 Hz plus angle-aware reflection as
  the least-bad diagnostic (`0.866 / 1.628 / 0.710` correlation/NRMSE/energy),
  still outside the gate;
- an 8 ms raised-cosine modal onset is the best smoothing value that preserves
  the original low-band gate, but crossover NRMSE remains `3.967`;
- decision: record the failure baseline. Do not fit a scalar, silently lower
  the crossover, refit residues, or ship onset smoothing. Move to shared
  fractional-delay `PathEvent` and coherent angle-aware early reflections,
  then rerun this exact protocol.

M3.1 versioned PathEvent and exact shoebox reference (completed 2026-07-31):

- add lossless `puresound.path_event.v1` and `path_event_set.v1` JSON contracts;
- keep physical delay separate from a complex pressure-gain spectrum so
  propagation phase cannot be counted twice;
- generate one direct and six exact first-order shoebox image paths with
  distance, delay, departure/arrival direction, surface, reflection point,
  incidence cosine, visibility, interaction policy, and directivity identity;
- preserve the project-wide free-field `1/r` RIR convention;
- store angle-aware locally reacting
  `Gamma(theta,f)=(cos(theta)-y(f))/(cos(theta)+y(f))` without pretending that
  arbitrary complex frequency samples are already a causal time-domain filter;
- add a causal one-sided third-order Lagrange renderer for real,
  frequency-independent path gains. Samples before the discrete arrival bin
  `floor(delay*fs)` are exactly zero;
- validate three geometries, including a near-grazing case: maximum geometry
  and reciprocity error `8.88e-16`, stable paths under a 1 mm source movement,
  and lossless schema round-trip;
- validate 8/16/48 kHz over 60–1000 Hz: worst fractional-delay magnitude/phase
  error `0.1001 dB / 0.5554 degrees`, zero samples before the arrival bin, and
  DC gain preserved to floating-point precision;
- decision: accept the M3.1 geometry and scalar fractional-delay gate. Do not
  render sampled complex spectra by raw IFFT. Next realize each boundary as a
  passive causal digital filter, integrate it into event rendering, and rerun
  the frozen M2.12 protocol.

M3.2 passive causal angle-filter realization (completed 2026-07-31):

- represent each digital normalized admittance as `Y(z)=B(z)/A(z)` and form
  the exact angle-aware digital Cayley transform
  `Gamma_theta(z)=(cos(theta)A(z)-B(z))/(cos(theta)A(z)+B(z))`;
- realize first-order, positive-real multi-pole, and passive resonant RLC
  admittances without raw complex-spectrum IFFT;
- serialize the stable causal filter as
  `puresound.digital_boundary_reflection_filter.v1`;
- require the supplied boundary model to reproduce every stored PathEvent
  complex-gain sample before rendering; reject mismatched models;
- cascade each causal boundary filter after the shared fractional propagation
  delay, retaining the physical arrival bin and the `1/r` convention;
- M2 reference filter at 8/16/48 kHz and five incidence cosines passes:
  worst digital-versus-analog error is `0.00279`, `0.0218 dB`, and
  `0.2397 degrees`; all tested rational models are stable and bounded-real;
- in all three frozen rooms, the rendered direct-plus-six-first-order response
  matches the identical analytic complex-angle path set with NRMSE at most
  `0.00385`, correlation at least `0.999998`, and energy ratio
  `0.99883–0.99953`;
- the same seven-path branch does not pass the full-room FDTD crossover gate:
  hybrid correlation/NRMSE/energy are `0.589 / 4.578 / 4.086`. This is not a
  filter-realization error; the early path set does not yet reproduce the
  full multiple-reflection/wave response;
- decision: accept the M3.2 causal filter gate while keeping the full-room
  gate open. Do not promote the seven-path diagnostic to the production high
  backend. Extend the coherent path set or connect a mesh tracer, then audit
  direct/early/late energy separately.

M3.3 ordered higher-order shoebox PathEvents (completed 2026-07-31):

- enumerate every integer image-source lattice point through an arbitrary
  Manhattan reflection order (currently bounded to 20), giving 7, 25, 129,
  833, and 2625 paths at orders 1, 2, 4, 8, and 12;
- fold the straight unfolded-room ray back into the physical shoebox and store
  its chronologically ordered surfaces, reflection points, incidence cosines,
  and interaction groups;
- preserve reciprocity and reproduce the existing same-order analytic complex
  image-source transfer to floating-point precision through order 12;
- cascade one passive causal angle filter for every ordered surface hit.
  Across the three frozen rooms, the order-12 time renderer matches the
  identical analytic path set with maximum NRMSE `0.01752`, minimum
  correlation `0.99991`, and energy ratio `0.99245–0.99900`;
- separate simultaneous edge/corner hits from ordinary face hits. Production
  geometry excludes them pending a physical diffraction model; an explicitly
  named diagnostic policy groups the coincident hits and multiplies legacy
  face filters only when comparison with the old analytic image product is
  required;
- order convergence is not monotonic in a coherent field: relative to order
  12, mean complex NRMSE is `9.551`, `8.093`, `3.300`, and `1.718` at orders
  1, 2, 4, and 8. More paths change both reinforcement and cancellation;
- order 12 still fails the full-room FDTD crossover gate. Its hybrid aggregate
  correlation/NRMSE/energy is `0.760 / 4.332 / 3.787`, effectively reproducing
  the existing analytic complex-angle failure rather than fixing it;
- decision: accept the M3.3 higher-order representation and renderer, reject
  “increase shoebox image order” as a standalone full-room fix, and do not
  replace the production high backend. Next audit direct/early/late complex
  energy against FDTD and evaluate a mesh engine with explicit visibility.

M3.4 direct/early/later coherent-error attribution (completed 2026-07-31):

- use the already validated free-field `1/r` PathEvent direct response as an
  explicit anchor. Do not claim that a time window can isolate direct sound:
  the 180 Hz validation pulse is wider than the 0.7–1.1 ms separation between
  direct and first-reflection arrivals in the frozen rooms;
- subtract the direct anchor from each FDTD and PathEvent output, then split
  the residual with exactly complementary 8 ms raised-cosine masks centered
  at direct plus 50 ms. Direct + early + later reconstructs the original
  response to numerical precision;
- retain a second, geometry-defined PathEvent partition by arrival time so
  boundary-filter tails remain attributable to the path that generated them;
- in 168–300 Hz, the early-reflection component alone has mean correlation
  `0.948`, NRMSE `0.331`, and transfer-norm ratio `0.999`, but its remaining
  complex error is `1.045` times the full FDTD transfer norm. Strong
  direct/early cancellation makes small phase errors consequential;
- the later component has a much larger component-relative NRMSE (`2.537`)
  but only `0.086` times the FDTD full-transfer norm. Its mean error is `0.224`
  times the full norm, far below the early-reflection contribution;
- coherent error accounting confirms that early and later error energies are
  not additive. Their squared self terms plus the explicit cross term equal
  the total squared error in every case;
- decision: accept the attribution protocol and localize the next model work
  to early-reflection phase and crossover branch interaction. Before adding
  more late rays or selecting a mesh engine, validate oblique single-wall FDTD
  reflection magnitude/phase against the same locally reacting model and
  audit the modal low-pass branch.

M3.5 staggered-grid oblique-boundary phase audit (completed 2026-07-31):

- derive the harmonic reflection coefficient of the implemented FDTD update,
  including the 3D discrete dispersion relation, pressure/velocity half-time
  staggering, and the half-cell distance between the wall face and first
  pressure cell;
- compare that coefficient with both the continuous locally reacting
  `Gamma(theta,f)` and the M3.2 PathEvent digital Cayley filter over
  168–300 Hz, five canonical incidence cosines, three tangential azimuths,
  all three wall axes, and representative directions from the actual
  order-12 early paths;
- the PathEvent filter remains close to the continuous model: actual-path
  worst complex/magnitude/phase errors are `0.00288`, `0.0218 dB`, and
  `0.243 degrees`;
- the current approximately 6 cm FDTD grid does not meet the continuous
  boundary parity gate. Canonical worst errors are `0.1238`, `0.475 dB`, and
  `7.68 degrees`; representative actual early paths reach `0.1665`,
  `0.656 dB`, and `9.75 degrees`;
- the discrete FDTD reflection remains passive, and its error decreases
  monotonically under linear refinement. At grid scales 1, 1/2, 1/4, and 1/8,
  worst phase error falls from `7.68` to `2.44`, `0.815`, and `0.303`
  degrees;
- decision: reject the coarse FDTD boundary phase as continuous ground truth.
  Do not fit the physical PathEvent reflection phase to this artifact. Next
  validate a face-pressure or phase-compensated boundary update in a
  time-domain plane-wave case, then audit the modal/geometric crossover.

M3.6 face-pressure and half-time correction experiment (completed 2026-07-31):

- add two opt-in FDTD boundary-pressure schemes while retaining
  `cell_center` as the default:
  `face_extrapolated` uses `1.5*p0-0.5*p1`, and
  `face_time_extrapolated` additionally predicts the next half step with
  `1.5*p[n]-0.5*p[n-1]`;
- extend the harmonic equation with the exact spatial extrapolation and causal
  time-predictor transfer functions;
- validate that equation independently with a 20 m 1D time-domain
  normal-incidence cross-ratio probe. Across all three schemes, worst
  measured-versus-predicted error is about `1.1e-4` complex and
  `0.0051 degrees`;
- both experimental schemes remain passive and materially reduce average
  actual-early-path error. Face plus time extrapolation reduces mean complex,
  magnitude, and phase errors from `0.0574 / 0.146 dB / 4.26 degrees` to
  `0.0224 / 0.0610 dB / 1.54 degrees`;
- neither scheme passes the worst-case continuous-reference gate. Actual
  grazing early paths still reach about `0.139` complex, `0.582 dB`, and
  `7.99 degrees`; even against a dispersion-matched target, the balanced
  face-time scheme reaches `0.0237` complex and `2.10 degrees`;
- decision: reject simple local extrapolation as a complete correction and do
  not change the production default. The remaining dominant error is the
  interior grid's angle-dependent characteristic admittance, which a local
  wall pressure predictor cannot remove. Evaluate a higher-order
  characteristic/dispersion-aware reference before rerunning full-room
  crossover.

M3.7 fourth-order staggered-grid harmonic candidate (completed 2026-07-31):

- derive the fourth-order staggered spatial symbol
  `D4/2 = sin(k*dx/2)/dx * (1 + sin(k*dx/2)^2/6)` and use it in both the
  three-dimensional dispersion relation and the normal characteristic
  admittance;
- evaluate a quadratic wall-face extrapolation
  `15/8*p0 - 5/4*p1 + 3/8*p2` and the same three-tap causal half-time
  predictor. These remove the quadratic interpolation error left by M3.6's
  two-tap linear predictors;
- compare four equations on the frozen approximately 6 cm grids: second-order
  cell center, second-order linear face/time, fourth-order linear face/time,
  and fourth-order quadratic face/time;
- fourth-order dispersion alone is insufficient: with the M3.6 linear
  predictor, actual-path worst error is `0.02372 / 0.0349 dB / 2.10 degrees`;
- the combined fourth-order/quadratic equation passes both canonical and
  actual-early continuous gates. Actual-path worst complex/magnitude/phase
  errors are `0.00750`, `0.0458 dB`, and `0.595 degrees`; corresponding means
  are `0.00229`, `0.0104 dB`, and `0.165 degrees`;
- the candidate remains passive over every scanned canonical and actual
  direction. Its fourth-order CFL number is at most `0.3927` on the frozen
  grids, below the unit stability bound;
- under joint grid/time refinement by factors 1, 1/2, 1/4, and 1/8, canonical
  worst complex error decreases `0.00750 -> 0.00106 -> 0.000267 ->
  0.0000669`, and worst phase error decreases `0.595 -> 0.0640 -> 0.0158 ->
  0.00396 degrees`;
- decision: accept exactly one *harmonic candidate*,
  `fourth_order_quadratic_face_time`, for time-domain prototyping. It is not
  yet an accepted FDTD reference: no fourth-order near-wall closure or
  time-domain cross-ratio has been implemented. Keep production/default
  `cell_center` unchanged and next validate a 1D time-domain prototype before
  modifying the 3D solver or rerunning the full-room crossover.

M3.8 fourth-order 1D near-wall time-domain closure (completed 2026-07-31):

- implement the M3.7 candidate in an independent 1D staggered time-domain
  prototype while leaving the production 3D solver unchanged;
- use the fourth-order centered interior derivative
  `9/8*(f[i]-f[i-1])-1/24*(f[i+1]-f[i-2])`;
- close the first pressure cell and first interior velocity face with the
  cubic-exact one-sided derivative weights
  `[-23/24, 7/8, 1/8, -1/24]`; mirror the weights at the rigid far wall;
- implement two samples of wall-face history for the quadratic spatial and
  causal half-time predictor selected in M3.7;
- validate the update with the same 20 m target/rigid/reference cross-ratio
  protocol. On the frozen base grid, time-domain versus harmonic worst
  complex/magnitude/phase errors are `0.000276`, `0.00184 dB`, and
  `0.00471 degrees`;
- jointly refine space/time to 1/2 and 1/4 scale. Worst errors remain
  `0.0000156 / 0.000108 dB / 0.000345 degrees` and
  `0.0000184 / 0.0000912 dB / 0.000716 degrees`; mean complex error decreases
  monotonically from `0.000104` to `0.00000616` and `0.000000963`;
- run a 1 s passive stability smoke: all samples are finite, the maximum
  absolute pressure is `2.575`, the one-dimensional fourth-order CFL number
  is `0.2267`, and tail/early RMS is `0.2367`;
- decision: accept the one-dimensional time-domain prototype and promote the
  candidate to an opt-in 3D implementation task. This is not yet an accepted
  3D boundary reference or an energy-stability proof; keep the production
  default unchanged and do not rerun full-room crossover until oblique 3D
  time-domain validation passes.

M3.9 opt-in fourth-order 3D FDTD reference (completed 2026-07-31):

- extend `FDTDReferenceConfig` with opt-in
  `spatial_derivative_order=4`,
  `near_wall_closure="third_order_one_sided"`, and
  `boundary_pressure_scheme="face_quadratic_time_quadratic"` while preserving
  the second-order `cell_center` defaults;
- implement fourth-order centered pressure-gradient and velocity-divergence
  updates on all three axes, mirrored one-sided closures at all six faces,
  and additive x/y/z divergence at edges and corners;
- retain two wall-face pressure histories per boundary and apply the M3.7
  quadratic spatial/half-time predictor independently at every face cell;
- discover that the bulk fourth-order CFL bound alone is insufficient for the
  tensor one-sided/quadratic closure: a long 3D run at effective CFL `0.39`
  excites a transverse numerical mode. Enforce and serialize the conservative
  opt-in closure cap `0.25`; the production second-order CFL path is
  unchanged;
- add validation-only spatial source/receiver weights, controlled source
  signals, and optional DC-removal bypass. A uniform 3D plane mode reduces to
  the independently implemented M3.8 1D update to `1e-12` sample tolerance;
- validate normal incidence with the 20 m broadband target/rigid/reference
  cross-ratio. Worst 168–300 Hz complex/magnitude/phase errors are
  `0.000294`, `0.00195 dB`, and `0.00520 degrees`;
- validate oblique time-domain reflection without an arbitrary wave-packet
  window. Solve the rigid tangential closure eigenproblem, drive steady
  harmonics at 270/285/300 Hz, and decompose incident/reflected amplitudes
  from two source-free pressure probes. Incidence cosine spans `0.250–0.490`;
  worst complex/magnitude/phase errors are `0.000608`, `0.00592 dB`, and
  `0.0253 degrees`;
- decision: accept the fourth-order/quadratic implementation as an opt-in 3D
  reference for the validated plane-mode scope. Do not change the production
  default or rerun the full-room crossover yet. Next require other normal
  axes, tangential azimuths/mode pairs, and edge/corner holdouts before using
  the reference to reopen crossover attribution.

M3.10 expanded fourth-order 3D holdouts (completed 2026-07-31):

- validate x/y/z normal axes, single tangential modes, and one non-equal
  two-tangential azimuth holdout at 285 Hz;
- worst plane-mode complex/phase errors are `0.000442 / 0.0268 degrees`,
  passing the frozen `0.02 / 1 degree` gates;
- raw face/edge/corner point-source reciprocity NRMSE reaches `0.637`, so the
  reference API reports it and returns the explicit bidirectional Green
  average `(Gsr + Grs) / 2`; reciprocalized NRMSE is exactly zero;
- retain the production second-order default. A stable centered/mimetic
  experimental closure was not promoted because its physical reflection
  phase error remained about 12 degrees;
- decision: accept the expanded opt-in 3D reference.

M3.11 expanded-reference crossover audit (completed 2026-07-31):

- rerun the three frozen train/position-holdout/room-holdout cases with the
  accepted fourth-order reciprocal reference and causal order-12 PathEvents;
- the 240 Hz full-room complex gate still fails; the magnitude-only production
  proxy has mean NRMSE/correlation/energy `4.567 / 0.399 / 4.217`;
- the best diagnostic is 120 Hz complex-angle with
  `1.622 / 0.869 / 0.702`, still outside the frozen gate;
- direct/early/later attribution passes its reconstructive protocol. Direct
  is exact; early-reflection NRMSE/correlation/energy is
  `0.315 / 0.953 / 1.004`, while later error has a much smaller norm relative
  to the full FDTD transfer;
- decision: complete the qualified audit and retain the explicit full-room
  complex limitation; do not fit a scalar or promote onset smoothing.

M3.12 existing mesh-engine evaluation (completed 2026-07-31):

- Pyroomacoustics 0.10.1 passes a non-convex 3D visibility/RIR smoke and
  remains useful as an independent cross-check and production default;
- its public contract does not expose ordered complex causal PathEvents,
  interior-solid furniture transmission, edge diffraction, or lossless event
  serialization;
- decision: keep PureSound PathEvents authoritative and implement only the
  scoped vertical-prism visibility required by M3, rather than creating a new
  general mesh tracer.

M3.13 furniture visibility geometry (completed 2026-07-31):

- validate serialized SceneObjects as closed vertical prisms with nonzero
  footprints, valid height/material coefficients, in-room placement, and
  transducers outside solids;
- test every source/interactions/receiver polyline segment against the prism
  in full 3D, preserving height-aware visibility and source-receiver
  reciprocity;
- serialize blocked-event to occluder IDs and keep invisible events
  inspectable rather than silently deleting their geometry;
- decision: accept height-aware furniture visibility.

M3.14 transmission, diffraction, and controlled scattering
(completed 2026-07-31):

- straight-through object transmission uses pressure gain
  `sqrt(product energy transmission)`;
- blocked direct paths receive up to two shortest visible vertical-edge
  detours using a bounded reference-frequency Fresnel-like coefficient;
- visible first-order reflections partition energy exactly into `1-s`
  specular plus four deterministic `s/N` scattering branches;
- add per-event energy partition, interaction model provenance, serialization,
  rendering, and per-path source-cardioid gain;
- the formal controlled scene passes visibility, transmission, diffraction,
  scattering-energy, reciprocity, serialization, and rendering gates;
- decision: accept the interactions and an opt-in
  `PathEventHighFrequencyBackend`; leave Pyroomacoustics as the default.

### M3 — Coherent early-path representation and mesh backend
(completed 2026-07-31)

Goal: represent the direct path and early reflections as physically inspectable
events before rendering samples.

A `PathEvent` should carry:

- total distance and fractional delay;
- departure and arrival direction;
- ordered reflection/transmission surfaces;
- complex gain or filter per band;
- visibility, diffraction, and scattering information;
- source and receiver directivity gains.

Implementation steps:

1. [x] Define and serialize `PathEvent`.
2. [x] Generate exact shoebox direct/first-order paths as a reference.
3. [x] Render frequency-independent real gains with a shared causal
   fractional-delay filter.
4. [x] Realize angle-aware complex boundary gain as a passive causal digital
   filter and rerun the frozen M2.12 rooms as a first-order diagnostic.
5. [x] Extend coherent shoebox events through order 12, including ordered
   surfaces and identifiable direct/early/late path buckets.
6. [x] Attribute the frozen full-room complex error to direct-anchored early
   and later components with an exactly reconstructive decomposition.
7. [x] Derive and audit the implemented staggered-grid oblique boundary phase
   against continuous and PathEvent reflection models.
8. [x] Evaluate face-pressure and half-time corrections with a time-domain
   plane-wave probe; reject them as incomplete grazing-angle fixes.
9. [x] Select a fourth-order dispersion-aware harmonic boundary candidate on
   frozen canonical and actual-path directions.
10. [x] Implement and validate the fourth-order one-dimensional time-domain
   near-wall closure.
11. [x] Implement the opt-in 3D fourth-order reference and validate one
   normal-axis/oblique-tangential plane mode.
12. [x] Validate multi-axis, multi-azimuth, and edge/corner 3D holdouts.
13. [x] Audit low/high crossover branch interaction using a boundary reference
   that has passed the expanded oblique phase gate.
14. [x] Evaluate integration of an existing mesh acoustic engine before
   implementing a new tracer.
15. [x] Move furniture from post-processing into actual visibility geometry.
16. [x] Add edge diffraction, transmission, and controlled diffuse scattering.

Exit gate:

- path delays match geometry;
- nearby positions produce continuous paths;
- source-receiver swaps obey reciprocity where expected;
- early-reflection timing and C50 improve against measured rooms.

Exit result:

- maximum exact PathEvent distance/delay errors:
  `8.88e-16 m / 3.47e-18 s`;
- 1 mm position perturbations preserve event identities and change path
  distance by at most `0.9515` times the displacement;
- exact PathEvents, scoped scene interactions, and reciprocalized FDTD all
  pass their expected reciprocity gates;
- against 100 held-out measured channels, the paired complete-hybrid M3 probe
  reduces distance-bucket mean absolute median C50 gap by `38.8%`
  (`5.289 -> 3.239 dB`) and direct-excluded early-energy-centroid gap by
  `26.4%` (`3.046 -> 2.243 ms`);
- diagnostic dominant-peak timing regresses (`1.104 -> 2.740 ms`) and remains
  recorded as an M4 warning because coherent peak identity is discontinuous;
- decision: M3 exit accepted. Advance to M4 late-field density, multiband
  decay, and spatial output while retaining the failed 240 Hz full-room
  complex-crossover audit as an explicit limitation.

### M4 — Late field, directivity, and spatial output (2–4 weeks)

Goal: produce a dense, frequency-dependent, spatially coherent late response.

Implementation decision:

- preserve causal PathEvents for the direct and early coherent response;
- render the dense tail with a multiband feedback delay network (FDN);
- parameterize FDN decay and spatial injection/output from material-derived
  octave decay plus path-traced directional-energy histograms.

The FDN is the primary dense-tail renderer because enumerating enough coherent
high-order paths to reach a diffuse tail is too expensive and retains
comb-like path regularity. Path tracing remains the parameter estimator, not a
second unbounded waveform renderer.

Implementation sequence:

1. **M4.1 measurement foundation — complete.** Add Abel-Huang normalized
   echo-density profiles, direct-relative mixing-time estimation, deterministic
   bank sampling, distance buckets, and measured-reference envelopes.
2. **M4.2 multiband/spatial target contract — complete.** Add octave-filtered
   echo density/decay, separate physical decay from measured noise floors,
   define target distributions by room class and distance, and freeze IACC plus
   diffuse-field coherence APIs without misusing source channels as receivers.
3. **M4.3 deterministic multiband FDN — complete.** Add room-scaled mutually
   incommensurate delays, an energy-preserving feedback matrix, passive
   frequency-dependent loop filters, deterministic seeding, and unit-energy
   diagnostics.
4. **M4.4 early/late coupling — complete.** Inject causal PathEvent energy into the FDN,
   crossfade around the estimated mixing time, preserve direct arrival and
   early phase, and calibrate octave decay without changing legacy mono output
   unless explicitly enabled.
5. **M4.5 spatial late field — complete.** Project one shared deterministic
   isotropic plane-wave late field to synchronized receiver arrays and
   ACN/SN3D first-order Ambisonics; validate coherence and IACC.
6. **M4.6 transducers — implementation complete.** Add real first-order source
   and receiver directivity plus an optional provenance-retaining
   HRTF-derived FIR decoder contract. The bundled audition decoder is
   explicitly analytic and is not mislabeled as measured HRTF data.

M4.1 baseline (seed `20260731`):

- 100 measured channels, 100 M1 channels, and 10 opt-in M3 channels;
- 20 ms Abel-Huang window, 1 ms hop, threshold `0.9`, and a documented 10 ms
  sustained-crossing robustness rule;
- measured/M1/M3 median mixing times: `16.0 / 21.0 / 20.5 ms`;
- all six M1 and all six M3 broadband monophonic median checks lie inside the
  measured central-80% envelopes.

That last result is a negative diagnostic, not an M4 pass: broadband
monophonic echo density alone does not distinguish the renderers. M4.2
therefore freezes octave-band and spatial/coherence gates before FDN tuning.
The report is
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_late_field_baseline.json`.

M4.2 baseline (seed `20260731`):

- 50 measured channels, 50 M1 channels, and the 10-channel opt-in M3 probe;
- nominal octave bands `125, 250, 500, 1000, 2000, 4000 Hz`;
- echo-density windows use at least four cycles at each lower octave edge;
- measured echo density is truncated at a reliable Lundeby intersection;
- M1 passes `31/48` and M3 passes `32/48` diagnostic median-envelope checks;
- M3 median mixing time at 2/4 kHz is `152/149 ms`, versus measured medians
  `24/24 ms` and measured p90 limits `70.2/88.2 ms`;
- M1 high-band density falls below the measured envelope later in the tail;
- measured channels require noise truncation in `16–38%` of bands, versus 0%
  for these deterministic synthetic banks;
- IACC/coherence is intentionally not computed because current bank WAV
  channels represent different sources at one receiver, not simultaneous
  receivers.

This is the first discriminative M4 target and directly motivates high-band
density injection in M4.3. The report is
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_late_field.json`.

M4.3 isolated-core baseline (seed `20260731`):

- 16 distinct prime delay lengths from `53` to `283` samples
  (`3.31–17.69 ms` at 16 kHz), selected deterministically from the measured
  high-band mixing-time target;
- signed/permuted normalized Hadamard feedback with zero measured
  orthogonality error, unit-energy band weights, and strictly contractive
  per-band feedback operators;
- delay-proportional pressure loop gain
  `g[b,i] = 10^(-3 d[i] / (fs T60[b]))` for every measured octave target;
- all 8 structural checks pass: prime delays, orthogonality, contraction,
  unit band-weight energy, determinism, finite output, causal onset, and decay
  before the render end;
- qualified 500/1000/2000/4000 Hz T20 relative errors are
  `1.15 / 4.85 / 3.03 / 0.02%`; mixing times are
  `12 / 26 / 10 / 24 ms`; all qualified decay, mixing-time, and late-NED
  checks lie inside the frozen M4.2 requirements;
- 125/250 Hz remain diagnostic: their T20 errors are `21.72 / 15.57%` because
  the production hybrid low/modal branch and the parallel-octave crossover
  have not yet been coupled to this isolated core;
- spectral-flatness/ripple output is diagnostic only; metallic coloration
  still requires the M4 listening exit.

The M4.3 report is
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_multiband_fdn_report.json`; its audition impulse
is `egs/rir_generation/exp/rir_realism/m4/rir_m4_fdn_core/rir_m4_fdn_core.wav`. This does not change the
production hybrid generator or the default Pyroomacoustics backend. M4.4 adds
that coupling behind a separate explicit opt-in backend.

M4.4 coupled baseline (scene/FDN seed `20260731`):

- add the explicit `path-events-m4` high backend; `pyroomacoustics` remains the
  default and `path-events-m3` remains the unchanged coherent-path diagnostic;
- preserve PathEvent samples exactly through the transition start, use a
  complementary cosine/sine equal-power transition centered 24 ms after each
  physical direct arrival, and stop coherent-path injection after the 16 ms
  transition;
- drive each source channel's FDN with its own early PathEvent response and a
  stable room/source-derived seed; octave RT60 comes from the serialized
  material-first scene rather than the fixed M4.2 median;
- solve the positive quadratic gain root so finite post-transition energy is
  equal to the original PathEvent response, including the coherent/FDN cross
  term rather than matching RMS independently;
- pass all 8 structural gates: shape, same-seed determinism, finite output,
  causality, exact early preservation, post-transition energy preservation,
  distinct channel seeds, and unchanged production default;
- material-target T20 relative errors at 500/1000/2000/4000 Hz are
  `2.83 / 5.32 / 0.69 / 3.47%`; M4 median mixing times are
  `22 / 26 / 30 / 34 ms`, all inside the frozen M4.2 measured envelopes;
- late normalized echo density changes from M3
  `0.560 / 0.000 / 0.000 / 0.000` to M4
  `1.041 / 1.109 / 0.992 / 1.001`, improving the distance to each measured
  median;
- the complete modal/high crossover is sample-exact before the transition;
  median/max absolute C50 changes are at most `1/3 dB`, and early-energy
  centroid changes are at most `1/2.5 ms` under the formal gates.

The report is
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_path_event_fdn_coupling_report.json`; paired M3
and M4 high/full-hybrid WAVs are under `egs/rir_generation/exp/rir_realism/m4/rir_m4_coupling/`. This is one
deterministic material-first fixture, not a measured-room or coloration exit.
M4.5 owns multi-receiver output matrices, coherence, and IACC.

M4.5 spatial baseline (scene/spatial seed `20260731`):

- one source is rendered to a synchronized 17 cm two-receiver array and to
  ACN/SN3D first-order Ambisonics (`W/Y/Z/X`) from the same plane-wave field;
- 256 seeded, rotated Fibonacci-sphere directions carry independent
  octave-band noise shaped by causal RMS envelopes from one passive multiband
  FDN; receiver delays are physical fractional propagation delays;
- all 12 structural gates pass, including exact determinism, causal onset,
  exact early preservation, per-channel post-transition energy, and unchanged
  production default;
- all 5 spatial gates pass. The bin-weighted qualified 500 Hz–4 kHz
  complex-coherence RMSE is `0.187197`, late IACC_L4 is `0.323144`, and every
  reported octave-coherence error remains inside the frozen model-derived
  tolerance. Full-spectrum RMSE `0.216982`, which includes frequencies outside
  the qualified M4 FDN bands, remains a diagnostic rather than an exit gate.

The report is `egs/rir_generation/phases/m4_spatial_late_field/reports/m4_spatial_rir_report.json`; array and
Ambisonic artifacts are under `egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/`.

M4.6 decoder baseline:

- source and receiver PathEvents support omni, cardioid, hypercardioid, and
  figure-eight real first-order pressure patterns;
- `AmbisonicBinauralDecoder` accepts causal FIRs with shape `[2, 4, taps]` and
  retains sample rate, reference identity, decoder kind, and provenance;
- all 11 decoder contract, causality, determinism, distinct-ear, and IACC
  analysis checks pass;
- the audition BRIR uses an analytic headless decoder and is explicitly not a
  measured HRTF. A multi-tap fixture proves external FIR injection but is also
  not labeled as measurement evidence.

The report is `egs/rir_generation/phases/m4_spatial_late_field/reports/m4_binaural_brir_report.json`; the
analytic audition artifact is `egs/rir_generation/exp/rir_realism/m4/rir_m4_spatial/m4_analytic_brir_2ch.wav`.

M4 completion is deliberately reported at two evidence levels:

- **implementation exit: passed.** M4.1 through M4.6 and their computational
  artifacts are complete;
- **empirical/production exit: open.** Measured synchronized multi-receiver
  validation, a licensed calibrated HRTF decoder, and controlled listening for
  metallic coloration/spatial plausibility are still required.

This split is serialized by
`egs/rir_generation/phases/m4_spatial_late_field/reports/m4_exit_report.json`. Pyroomacoustics remains the
default; the M4 spatial renderer is an explicit opt-in API/CLI.

Add:

- room-dependent mixing time;
- band-dependent echo density and decay;
- talker/loudspeaker spherical-harmonic directivity;
- omni, cardioid, and device-mounted microphone models;
- microphone arrays and Ambisonic output;
- optional HRTF rendering to BRIR.

Exit gate:

- late tails do not exhibit obvious metallic coloration;
- octave decay and echo density match the target distributions;
- array coherence and IACC are plausible;
- mono output remains backward compatible.

### M5 — Measured-room inverse calibration and learned residual (implementation complete; empirical exit open)

Goal: close the remaining measured/synthetic gap using sparse, controlled real
measurements.

Measurement protocol:

- repeated exponential sine sweeps;
- calibrated loudspeaker and microphone response;
- source/receiver position and orientation;
- room geometry or coarse mesh;
- temperature and humidity;
- retained raw sweeps, deconvolution configuration, and noise estimates.

For each representative room, begin with roughly 12–30 source-receiver
measurements. Fit material, scattering, directivity, and late-field parameters
with a differentiable approximate renderer. Only then train a residual model.

Residual losses should include:

- multiresolution STFT distance;
- energy-decay curve distance;
- direct/early arrival timing;
- octave-band acoustic metrics;
- spatial coherence;
- causality and decay regularization.

Exit gate:

- held-out positions in calibrated rooms improve;
- held-out rooms improve rather than merely being memorized;
- the physical and learned contributions can be ablated independently;
- generated output remains physically valid under interpolation.

Implementation sequence:

1. **M5.1 — measurement and loss contract.** Freeze the controlled-room
   campaign schema, room-disjoint split semantics, retained acquisition assets,
   and an independently auditable reference loss.
2. **M5.2 — synthetic recovery.** Hide parameters from synthetic scenes, fit
   them back from their RIRs, and reject parameters that are not identifiable
   even before real measurement uncertainty is introduced.
3. **M5.3 — measured-room fit.** Fit train positions in controlled rooms and
   report held-out positions separately from held-out physical rooms.
4. **M5.4 — joint spatial calibration.** Add scattering, directivity, and
   multiband late-field parameters only where synchronized receivers provide
   enough evidence.
5. **M5.5 — constrained learned residual.** Learn only the remaining error,
   with explicit causality/decay constraints and physical-only,
   residual-only, and combined ablations.
6. **M5.6 — exit decision.** Run held-out-room acoustic, listening, and
   downstream gates before enabling the calibrated renderer for production.

M5.1 implementation result (2026-08-01):

- [x] Added strict `puresound.rir_measurement_campaign.v1` types for room
  geometry, calibrated transducers, pose uncertainty, environment, repeated
  ESS captures, raw/noise/inverse/deconvolved assets, SHA-256 provenance, and
  room-disjoint train/validation/test assignment.
- [x] Added a campaign audit that verifies retained assets and hashes, 12 or
  more measurements per room, repeated sweeps, and at least one synchronized
  spatial capture.
- [x] Added `puresound.rir_calibration_loss.v1`, combining multiresolution
  STFT, direct-relative energy decay, arrival timing, octave energy/T20,
  complex spatial coherence, causality, and late-decay regularization. The
  NumPy/SciPy implementation is a reference/evaluation contract, not an
  autograd optimizer.
- [x] Added a structurally valid template and deterministic contract validator.
  All M5.1 implementation probes pass.
- [x] Audited 64 metadata items from each existing measured train/held-out
  bank. They remain useful acoustic references, but contain none of the nine
  controlled acquisition evidence groups required by M5.1; measured inverse
  fitting therefore remains open rather than being falsely claimed.

Evidence:

- report: `egs/rir_generation/phases/m5_calibration/reports/m5_measurement_contract_report.json`;
- non-evidence template:
  `egs/rir_generation/phases/m5_calibration/config/m5_measurement_campaign_template.json`;
- protocol and field semantics:
  `docs/audio/rir_measurement_campaign_zh-TW.md`.

M5.1 implementation exit is **PASS**. Controlled-measurement readiness is
**OPEN**. M5.2 may proceed with synthetic recovery without claiming a
measured-room fit; M5.3 cannot close until a real controlled campaign satisfies
the contract.

M5.2 synthetic-recovery baseline (2026-08-01):

- [x] Added a causal approximate renderer that keeps geometry/direct response
  fixed and exposes shared mixing time, coherent early-reflection gain, four
  octave RT60 values, and four octave late gains under explicit box bounds.
- [x] Fit the ten hidden parameters jointly from three synthetic source/receiver
  positions with bounded nonlinear least squares and three deliberately distant
  initializations.
- [x] All starts converged to the same noise-free ground truth; maximum parameter
  spread was `2.35e-12`. Scaled-Jacobian condition number was `17.22`, with full
  local column rank.
- [x] The independent M5.1 oracle mean loss on two unseen positions fell from
  `2.26933` to `1.69e-14` (greater than 99% reduction), while every output
  remained causal.
- [x] Added target/initial/recovered holdout WAVs and a strict report at
  `egs/rir_generation/phases/m5_calibration/reports/m5_synthetic_recovery_report.json`.

This is intentionally an **inverse-crime baseline**: the same noise-free
approximate renderer family creates and fits the targets. It proves parameter
ordering, bounds, multi-position fitting, multi-start convergence, local
sensitivity, holdout evaluation, and independent loss plumbing. It does not
prove robustness to measurement noise/model mismatch, invert the complete M4
renderer, or fit a real room. Those limitations are serialized in the report.

M5.2b robust synthetic recovery (2026-08-01):

- [x] Added deterministic measurement perturbations spanning 32–38 dB SNR,
  per-position calibrated gain error up to 1.2 dB, latency offsets from -6 to
  +11 samples, unmodelled early paths, and an independent 1.25x-decay late
  component.
- [x] Retained raw perturbed RIRs and applied only the known gain/latency
  correction before fitting. Noise and renderer mismatch remain in the target.
- [x] Added a noise-aware smooth M4-proxy objective with waveform, coherent
  early, broadband decay, and octave decay residuals. It is optimized with
  SciPy finite differences and is explicitly not an autograd or complete-M4
  inverse renderer.
- [x] Recovered mixing time within `0.0235 ms`, early gain within `0.0214 dB`,
  all octave RT60 values within `1.43%`, and all late gains within `0.106 dB`.
- [x] Two distant starts converged within `1.03e-6`; the scaled Jacobian
  condition number is `8.12` and remains full rank locally.
- [x] On two separately perturbed held-out positions, the independent M5.1
  total fell `61.4%` from the initial model. Octave error was `0.00846` versus
  `0.03341` for the waveform-only ablation; maximum RT60 and late-gain errors
  also improved.
- [x] Recorded that the global absolute peak was not the direct arrival in two
  of five noisy/model-mismatched cases. The controlled protocol must use the
  geometry-bounded arrival window or a validated onset detector.

The 15/15 M5.2b gates pass. Evidence and six holdout RIR artifacts are in
`egs/rir_generation/phases/m5_calibration/reports/m5_robust_recovery_report.json` and
`egs/rir_generation/exp/rir_realism/m5/rir_m5_robust_recovery/`. M5.3 remains blocked on a real controlled
campaign.

M5.2c actual M4 parameter-profile mapping (2026-08-01):

- [x] Connected the inverse fit to the real M4
  `PathEvent -> equal-power transition -> multiband FDN` renderer rather than
  the smooth proxy. Pyroomacoustics remains the unchanged production default.
- [x] Treated mixing time as a discrete outer profile because it changes prime
  FDN delay lengths discontinuously; fitted coherent-reflection gain and three
  octave RT60 targets in the bounded continuous inner solve.
- [x] Used two order-4 PathEvent training positions and two unseen positions.
  Targets retain 8% alternate-FDN-topology mismatch and 42 dB SNR noise.
- [x] Selected the hidden `24 ms` mixing profile over `20/28 ms`; the best to
  second-best cost ratio is `0.0820`. Every inner solve converged with locally
  full-rank Jacobian; best scaled condition number is `3.14`.
- [x] Recovered coherent-reflection gain within `0.661 dB`; octave RT60 errors
  are `0.392%`, `1.86%`, and `4.23%`. The independent held-out M5.1 total fell
  `71.7%`, while physical-arrival causality and exact pre-transition
  preservation remained intact.
- [x] Added a strict 14-gate report and target/initial/recovered holdout WAVs at
  `egs/rir_generation/phases/m5_calibration/reports/m5_m4_parameter_mapping_report.json` and
  `egs/rir_generation/exp/rir_realism/m5/rir_m5_m4_parameter_mapping/`.

M5.2c is the first mapping to the actual M4 renderer, but its
`coherent_reflection_gain_db` is still an aggregate proxy. It does not identify
individual wall absorption/scattering, prove global identifiability outside the
supplied mixing grid, or complete measured-room fitting. The next parallel
model task is a grouped material/path identifiability ablation; M5.3 remains
blocked on a qualifying controlled campaign.

M5.2d grouped material/path identifiability (2026-08-01):

- [x] Reparameterized ordered PathEvents by six effective boundary-reflection
  adjustments, with one dB pressure adjustment applied per boundary hit before
  the actual M4 early/FDN coupling.
- [x] All six boundary groups are locally full rank on three order-4 training
  positions: normalized condition number `2.74`, maximum column correlation
  `0.540`, maximum recovery error `0.00123 dB`, and two-start spread
  `4.21e-10 dB` at 48 dB SNR.
- [x] The held-out M5.1 total fell `64.8%`.
- [x] Explicitly duplicated each sensitivity into absorption-loss and
  specular-scattering-loss columns. Rank stayed `6/12`; all six scattering
  duplicates were rejected. Mono coherent RIRs identify only effective
  reflection loss, so scattering is deferred to synchronized M5.4 evidence.

M5.3 measured runner implementation (2026-08-01):

- [x] Added retained-asset/hash readiness, deterministic SHA-256 position
  fit/holdout assignment, per-train-room M4/profile fitting, held-out-position
  reports, and separate validation/test physical-room reports.
- [x] Ran the complete runner on an explicitly non-evidence synthetic campaign
  with train/validation/test rooms and synchronized receivers; all eight runner
  implementation gates pass.
- [x] Ran the same CLI against the current campaign template. It exits blocked
  before fitting because assets/hashes and acquisition evidence are absent.
- [x] Defined what `all_train_room_m4_profiles_converged` asserts
  (2026-08-03, `puresound.m4_profile_convergence.stable_minimum.v1`). The M6
  excitation fix moved the optimization landscape: the fit now reaches a
  *lower* cost (0.0333 against the frozen 0.0404) while no longer tripping
  `ftol`. Measured on the fixture, the point is stationary in every direction
  that matters — no coordinate step of 1e-3, 1e-2 or 1e-1 of the bound span
  lowers the cost — while the reported first-order optimality reads 6.2e-2
  because the objective is locally rough. `success` and `optimality` are
  therefore both wrong criteria. Convergence is now decided by restarting the
  solve from its own answer, which resets the collapsed trust region: the
  point counts as a minimum only if the restart cannot lower the cost by more
  than 1e-3 relative. A deliberately truncated fit is still rejected, and both
  solves are recorded in the report rather than a single boolean.
- [ ] Execute the runner on a real controlled repeated-ESS campaign. Current
  legacy banks remain ineligible (`0/64` for all nine required evidence groups).

M5.4 synchronized spatial calibration implementation (2026-08-01):

- [x] Added a fail-closed candidate profiler that requires at least two
  synchronized receivers and evaluates scattering/directivity/late-field
  candidates with the M5.1 spatial-aware loss.
- [x] The actual M4 four-candidate fixture selects the hidden
  scattering-plus-cardioid candidate; 11/11 gates pass. Mono input is rejected.
- [x] Recorded that current scattering changes coherent early PathEvents, while
  the post-80-ms FDN field is scattering-independent. Scattering is therefore
  selected by synchronized early/spectral/octave evidence, not falsely by an
  identical late-coherence term.
- [ ] Repeat on measured synchronized arrays.

M5.5 constrained residual implementation (2026-08-01):

- [x] Added a direct-relative shared residual fitted only from physical-model
  error. It has zero pre-direct support, a 50-ms-onward block decay ceiling, and
  an explicit residual/physical energy budget.
- [x] Added physical-only, residual-only, and combined ablations plus an
  interpolated-output causality check.
- [x] On a third, room-disjoint synthetic room, combined M5.1 total fell
  `67.9%` from physical-only; all 12 structural/ablation gates pass.
- [ ] Train/evaluate the residual on real controlled train/held-out rooms and
  run the fixed downstream speech task.

M5.6 aggregate exit (2026-08-01):

- [x] Added `puresound.m5_exit.v1`, nine implementation-stage gates, seven
  required artifact checks, and invariants preventing false measured claims.
- [x] **M5 implementation exit: PASS.** M5.1–M5.6 code, validators, synthetic
  fixtures, fail-closed measured runner, and ablations are complete.
- [ ] **M5 empirical/production exit: OPEN.** A qualifying controlled campaign,
  measured held-out positions/rooms, synchronized spatial calibration,
  measured residual training, controlled listening, and room-disjoint
  downstream evidence are still missing. Production remains disabled.

Evidence is aggregated in
`egs/rir_generation/phases/m5_calibration/reports/m5_exit_report.json`. This is the terminal M5
implementation result; the remaining work is external empirical evidence, not
an unimplemented optimizer path.

### M6 — Production RIR bank v2 (M6.1–M6.6 implementation complete; promotion blocked)

Goal: package the accepted simulator into reproducible training banks.

Required outputs:

- room-disjoint train, validation, and test manifests;
- scene and renderer version hashes;
- material and room-type distributions;
- calibrated and normalized RIR variants where required;
- quality-control metrics per item;
- generation throughput and failure statistics;
- a final synthetic-only, mixed, and real-RIR downstream comparison.

Implementation sequence:

1. **M6.1 — bank contract.** Freeze deterministic acoustic-space splits,
   content hashes, generator/renderer provenance, signal variants, QC state,
   and fail-closed release semantics.
2. **M6.2 — reproducible generator integration.** Materialize the task plan,
   generate/resume by split, and emit the M6 manifest from actual outputs.
3. **M6.3 — per-item QC and quarantine.** Measure causality, clipping, DRR,
   clarity, decay, spectral, density, and spatial contracts before admission.
4. **M6.4 — distribution and variant release.** Publish available synthetic,
   mixed, and real variants with frozen distributions; unavailable measured or
   mixed recipes remain explicitly blocked rather than being fabricated.
5. **M6.5 — bank-level and downstream evaluation.** Compare measured acoustic
   distributions, throughput/failures, listening, and room-disjoint tasks.
6. **M6.6 — production decision.** Promote only content-addressed evidence-
   backed profiles/items; otherwise remain candidate or draft.

M6.1 bank-contract result (2026-08-01):

- [x] Added `puresound.rir_bank.v2` with canonical manifest SHA-256, per-asset
  WAV/metadata/scene hashes, audio shapes, signal/level variants, and QC state.
- [x] Added SHA-256 acoustic-space assignment and independent room/acoustic-
  space leakage checks across train/validation/test.
- [x] Added generator/config/code and renderer/backend/scene/evidence
  provenance. A development profile cannot claim a production release.
- [x] The deterministic non-evidence fixture passes 12/12 contract gates,
  including asset/manifest tamper, unsafe-path, leakage, false-production, and
  legacy-reader controls.
- [x] Integrate the contract into the actual parallel/resumable generator
  (M6.2); current fixture is not a production or acoustic-quality bank.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_bank_contract_report.json`;
- fixture manifest: `egs/rir_generation/exp/rir_realism/m6/rir_m6_bank_contract/rir_bank_manifest.json`;
- Chinese contract: `docs/audio/rir_bank_v2_zh-TW.md`.

M6.2 reproducible-generator result (hardened 2026-08-02):

- [x] Added opt-in `--emit-m6-manifest` integration to the real
  `generate_hybrid_rir.py`; the legacy path and Pyroomacoustics default remain
  unchanged.
- [x] Materialize and hash the complete task plan before rendering. Each item
  has a stable task seed, acoustic-space split, scene hash, renderer profile,
  expected audio shape, and generation-config hash.
- [x] Emit content-addressed train/validation/test JSONL indexes, the v2
  manifest, and a strict post-generation audit from actual WAV/JSON outputs.
- [x] Require an explicit split when `PreGeneratedRoomBank` reads an M6 root;
  unsplit access fails closed while legacy banks retain their old behavior.
- [x] Resume now verifies task/config/code-revision identity, scene hash, WAV
  content hash, audio header, and records critical runtime package versions. A
  corrupted item is regenerated without touching valid items; changing the
  generation config or code revision invalidates incompatible outputs.
- [x] Pyroomacoustics generation seeds both NumPy and libroom RNGs. The 17-gate
  fixture uses the actual default Pyroomacoustics high backend and produces
  identical serial, two-worker, and independent fresh-run manifest/item/task-
  plan hashes. After one WAV is tampered, resume regenerates exactly `1/6` and
  restores the original manifest; revision/config changes are fail-closed.
- [x] Add M6.3 per-item acoustic QC and quarantine before any candidate bank is
  admitted or promoted.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_reproducible_generation_report.json`;
- hardened validation outputs: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/reproducible/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_reproducible_generation.py`.

M6.3 per-item QC/quarantine result (hardened 2026-08-02):

- [x] Added a content-addressed `puresound.rir_bank_qc.physical.v1` policy and
  deterministic reports for structural integrity, causality/direct timing,
  tail energy, DRR, C50/C80, noise-aware decay, spectral tilt, octave bands,
  and Abel echo density.
- [x] Separate `fail`, `not_evaluable`, and `not_applicable`. Source-indexed
  5-channel RIRs cannot falsely claim synchronized-array IACC/coherence.
- [x] Publish pass-only train/validation/test candidate indexes and a quarantine
  index with per-item report path/hash and explicit reasons. Original WAV and
  metadata assets remain unchanged.
- [x] `PreGeneratedRoomBank` excludes failed items by default; candidate and
  production manifests admit only QC PASS items. Debug inclusion is explicit.
- [x] If quarantine empties any split, release remains `draft`. A complete
  three-split QC bank becomes `candidate`, never `production` from QC alone.
- [x] The actual M6.2 generator fixture passes `6/6`; silent, pre-arrival,
  sparse-late-field, and deliberately late-direct-arrival negative controls
  are quarantined. The formal validator passes 13/13 gates, including active
  octave-decay coverage, deterministic hashes, and QC-report tamper.
- [x] Start M6.4 distribution and variant release; publish the two valid
  synthetic recipes and fail closed for missing measured/mixed variants.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_item_qc_report.json`;
- hardened outputs and negative controls: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/item_qc/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_item_qc.py`.

M6.4 distribution/variant-release result (2026-08-01):

- [x] Added `puresound.rir_bank_release.v1`, content-addressed acoustic
  distribution snapshots, variant lineage, and train/validation/test recipe
  indexes.
- [x] Published `synthetic_calibrated` and a deterministic
  `synthetic_peak_normalized` variant. The latter applies one common gain per
  item to peak `0.98`; DRR, C50/C80, T20, acoustic-space identity, and split
  remain invariant within numerical tolerance.
- [x] Added `PreGeneratedReleaseBank(recipe_id=..., split=...)`. It audits the
  release, enforces recipe origin weights, and propagates release/recipe/
  variant identity into sample metadata.
- [x] Canonicalized the non-acoustic libsndfile float-WAV `PEAK` timestamp so
  repeated builds are byte-identical instead of changing once per wall-clock
  second.
- [x] The formal release validator passes 12/12 gates, including two-build
  determinism, recipe consumption, sample-exact parent/child transform and
  parent-hash lineage, and index-tamper rejection.
- [ ] `real_native` and `mixed_calibrated_real` remain blocked because no
  QC-passed measured M6 variant was supplied. This is a truthful incomplete
  empirical release, not an M6.4 implementation failure.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_variant_release_report.json`;
- hardened candidate release fixture: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/variant_release/release_a/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_variant_release.py`.

M6.5 bank-evaluation result (hardened 2026-08-02):

- [x] Added content-addressed bank-level distribution comparison, actual
  generation-throughput/failure validation, controlled-listening contract,
  and room-disjoint multi-seed downstream contract.
- [x] Distribution evaluation confirms calibrated/normalized scale-invariant
  metrics and records synthetic-to-measured as `not_evaluable` when a measured
  reference is absent; it never substitutes synthetic fixtures for real data.
- [x] Listening evidence requires randomized double blind assignment, common
  loudness gain, hidden reference/degraded anchor, room-disjoint stimuli,
  content-addressed responses/analysis, and at least 20 people for empirical
  status.
- [x] Downstream evidence recomputes recipe train/test acoustic-space hashes,
  requires at least three unique seeds and frozen model/training recipes, and
  requires every primary confidence lower bound to improve.
- [x] Downstream confidence intervals are recomputed from paired per-seed
  improvements. Empirical listening evidence must include content-addressed
  assignment, response, and analysis records whose participant result and
  confidence interval can be recomputed.
- [x] The formal validator passes 14/14 implementation gates and rejects
  unblinded listening, split-identity tamper, a single-seed claim, a forged
  positive confidence interval, and non-human responses relabelled empirical.
- [ ] **M6.5 empirical exit remains OPEN.** Current listening/downstream inputs
  are clearly labelled contract fixtures, not human responses or trained-model
  results; measured-distribution evidence is also absent. Production remains
  disabled.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_bank_evaluation_report.json`;
- hardened evaluation artifacts: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/evaluation/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_evaluation.py`.

M6.6 production-decision result (hardened 2026-08-02):

- [x] Added `puresound.m6_production_decision.v1`: an immutable candidate
  release is promoted by a content-addressed certificate instead of rewriting
  every bank asset, variant manifest, distribution, and recipe hash.
- [x] The decision binds the M6.4 release SHA, M6.5 evaluation SHA, ready
  synthetic/real/mixed recipes, QC state, pinned generator revisions,
  production-approved renderer profiles, evidence files, and three role
  sign-offs (acoustics, ML, release owner).
- [x] Evidence files must exist under safe relative paths and match their
  declared SHA-256. Listening/downstream artifact hashes must equal the hashes
  evaluated by M6.5; renderer approval files must match every profile.
- [x] Certificate validation re-runs the bound release/evidence audits and
  recomputes the canonical decision checks; it does not trust self-declared
  booleans merely because an attacker also recomputed the outer hash.
- [x] `PreGeneratedReleaseBank(require_production=True)` accepts only a valid
  approved certificate. Candidate usage remains available without that flag.
- [x] The formal validator passes 15/15 implementation gates, rejecting unsafe
  evidence paths, evaluation flag/hash tamper, forged approval flags, a
  competent all-true/rehashed certificate forgery, direct editing of candidate
  status, and production-reader access to a blocked bank.
- [ ] **Production promotion remains BLOCKED.** The current certificate lists
  missing real/mixed recipes, production renderer approval, measured/listening/
  downstream empirical evidence, evidence files, and three sign-offs.

Evidence:

- report: `egs/rir_generation/phases/m6_bank/reports/m6_production_decision_report.json`;
- hardened decision artifacts: `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/production/`;
- validator: `egs/rir_generation/phases/m6_bank/scripts/validate_m6_production_decision.py`.

Post-M6 training-pilot readiness (updated 2026-08-02):

- [x] Wire `PreGeneratedReleaseBank` into the existing
  `AudioEffectAugmentor`/dynamic-dataset YAML path with `bank_type: release`.
- [x] Require an explicit recipe and split; reject release-only options in
  legacy room-bank mode. Dataset role must equal the release split, and
  release/variant/split provenance reaches the collated training batch.
- [x] Bound the augmentor's simulated-RIR cache with LRU eviction; reject
  manifestless M6 layouts, invalid manifest items, out-of-range channels, and
  pending items in candidate/production readers.
- [x] Add a ready-to-copy training YAML and a non-overwriting pilot script for
  matched 1,000-room × 4-RIR Pyroomacoustics and PathEvents-M4 candidates.

Hardened matched preflight result (2026-08-02):

- [x] Completed a new 30-room × 2-item preflight for both high backends with
  fixed `v1/mixed`, seed `1337`, calibrated `16 kHz / 1.6 s`, and shared
  `pytard-cupy-material` low band. Each backend produced 60 items / 300
  channels; all 60 matched scene/room/acoustic-space/split/seed/shape identities
  agree, both banks are 60/60 QC PASS with zero quarantine, and both releases
  pass audit. The paired report is
  `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardened_preflight_20260802/preflight_validation_summary.json`.
- [x] Confirmed the intended acoustic difference: M4 has median C50/C80 of
  10.28/15.01 dB versus Pyroomacoustics 6.10/8.03 dB, median T20 0.472 s versus
  0.990 s, and absolute T20-to-scene-RT60 error 0.080 s versus 0.327 s. M4 is
  therefore drier and more early-energy dominant in this sample, and tracks the
  material RT60 more closely; this is not a measured-room realism verdict.
- [ ] The earlier A/B run was stopped and predates the hardening, so it remains
  diagnostic only. Re-run both full 4,000-item pilots in clean, pinned output
  directories, compare throughput/QC yield/distributions, obtain measured,
  listening, and downstream evidence, then train the first room-disjoint model
  A/B before scaling to a 50k–200k item bank. Pyroomacoustics remains the default
  while M4 is opt-in.

The preflight is explicitly `pass_with_provenance_warnings`: it used a dirty code
revision, has only two validation and two test items per backend, and does not
contain measured RIRs, human listening, or downstream-model results. Pyroom's
configured air absorption is applied internally but is not serialized in the
high-band metadata as completely as M4's policy, so provenance work remains.

## 5. Experiment discipline

Each experiment changes one major factor at a time:

```text
v0  current hybrid baseline
v1a + multiband materials, legacy peak normalization
v1b + calibrated gain using the same material scenes
v2  + source/receiver directivity rendering
v3  + lossy low-frequency modes
v4  + coherent mesh early paths
v5  + spatial multiband late field
v6  + inverse calibration / residual
```

Every row records:

- code and configuration revision;
- bank seed and manifest;
- generation time;
- M0 acoustic summary;
- downstream result with confidence interval;
- known limitations.

## 6. Risks and controls

| Risk | Control |
|------|---------|
| Full wave simulation becomes computationally prohibitive | Start with per-mode boundary loss and maintain a small reference solver only for validation |
| Metrics are optimized without audible or downstream benefit | Require real-room downstream and, for rendering, listening-test gates |
| Real measurements entangle room and transducer response | Preserve calibration data and model transducers separately |
| A learned model memorizes rooms | Split by room and report held-out-position and held-out-room results separately |
| Multiple RIR metric implementations disagree | Centralize them in one tested library module |
| Backend integration creates licensing or reproducibility problems | Complete a make-or-integrate review before adopting it |
| New metadata breaks existing recipes | Version the schema and retain legacy readers |

## 7. Immediate development queue

The active queue starts with M0:

- [x] Centralize DRR, clarity, decay, and spectral metrics.
- [x] Add analytic unit tests for decay estimates and failure cases.
- [x] Refactor `compare_bank_acoustics.py` to use the shared metrics.
- [x] Add deterministic JSON output suitable for checked experiment reports.
- [x] Add octave-band analysis.
- [x] Freeze and document the v0 generator configuration.
- [x] Run the first generated-versus-measured bank comparison.
- [x] Select the first M1 change from measured evidence.
- [x] Complete the M1 material-first schema, renderer, calibrated mode, and
  100-item acoustic probe.
- [x] Build a causal 50-item M1 audition bank with full WAV/metadata quality
  gates and deterministic dry/wet near/far previews.
- [x] Add measured-RIR noise-floor detection/truncation to decay fits.
- [x] Begin M2 per-mode low-frequency boundary damping.
- [x] Add a small independent 3D FDTD case for modal frequency/Q validation.
- [x] Benchmark the first M2 material-modal low band against measured modal
  distributions; it improves modal spacing but fails the Q gate due to excess
  damping, so it remains disabled for training banks.
- [x] Define and test the phase-aware complex-impedance/reflection contract
  without inventing phase from diffuse absorption.
- [x] Integrate a passive causal frequency-dependent admittance into the
  validation FDTD and verify its single-wall complex reflection response.
- [x] Add provenance-bearing glass-wool/Miki references, a complex-reflection
  fitting gate, and a phase-versus-magnitude FDTD modal diagnostic.
- [x] Add the direct complex-measurement contract, passivity-by-construction
  multi-pole fit, multi-state FDTD boundary, and 1D complex modal eigenvalue
  reference.
- [x] Audit public sources, ingest CC BY 4.0 NASA/UFSC normalized complex liner
  measurements, fit a passive series-RLC branch with alternating-frequency
  holdout, and validate the same branch in FDTD and the 1D eigenproblem.
- [x] Implement the calibrated repeated-H12 impedance-tube reduction and
  Traditional Chinese physical measurement protocol needed to acquire a
  compatible room-finish prior without inventing reflection phase.
- [x] Record the M2.12 full-room complex-crossover failure baseline and reject
  scalar gain, crossover relocation, and onset smoothing as standalone fixes.
- [x] Define and serialize shared direct/first-order `PathEvent` geometry,
  angle-aware complex gain spectra, and a causal scalar fractional-delay
  renderer; pass the M3.1 geometry/reciprocity/continuity gate.
- [x] Realize stored angle-aware complex boundary spectra as passive causal
  digital filters, verify the renderer against the identical analytic paths,
  and rerun the frozen M2.12 rooms as a seven-path diagnostic.
- [x] Extend the coherent PathEvent set through order 12, preserve ordered
  surface interactions, and verify spectral plus causal time rendering against
  the same-order analytic image set.
- [x] Audit direct-anchored early/later complex response against FDTD with
  complementary windows and coherent cross-term accounting; localize the
  dominant full-transfer error to early-reflection phase/interference.
- [x] Derive the exact staggered-grid oblique reflection and show that the
  approximately 6 cm FDTD boundary phase is not a continuous-reference match,
  while remaining passive and converging under grid refinement.
- [x] Validate face-pressure and half-time boundary predictors against the
  harmonic equation in a time-domain probe; reject both as incomplete
  continuous-reference fixes and retain the legacy default.
- [x] Evaluate a higher-order characteristic/dispersion-aware boundary, pass
  expanded multi-axis/azimuth holdouts, audit modal/geometric crossover
  branches, and evaluate an existing mesh engine; the order-12 full-room
  crossover remains an explicit failed diagnostic.
- [ ] Replace diffuse absorption-as-modal-loss with a low-frequency impedance
  prior from a compatible room-finish measurement in the production 3D
  backend, then rerun the room-disjoint M2 probe.
- [x] Audit direct and early reflection energy, integrate opt-in furniture
  interactions, and pass the paired measured C50/continuous early-timing M3
  exit before changing the late-field model.
- [x] Begin M4 with Abel-Huang normalized echo density, a room-response mixing
  time estimator, and a deterministic measured/M1/M3 bank baseline.
- [x] Add noise-aware octave-band late-field echo-density/decay targets and
  explicit binaural-IACC/diffuse-field-coherence contracts before FDN tuning;
  record that existing source-channel banks cannot evaluate spatial output.
- [x] Implement the isolated deterministic internally contractive multiband
  FDN core against the M4.2 targets; pass structural and 500 Hz–4 kHz measured
  decay/density/mixing gates while retaining 125/250 Hz as diagnostics.
- [x] Couple causal PathEvent direct/early energy to the FDN at the estimated
  mixing region with exact early preservation and post-transition energy
  matching; expose it only through `path-events-m4` and keep all earlier/default
  backends unchanged.
- [x] Add synchronized multi-receiver/ACN-SN3D Ambisonic output from one shared
  plane-wave field; pass deterministic model-derived diffuse-coherence/IACC
  gates, first-order receiver directivity, and the optional HRTF FIR decoder
  contract without changing the default backend.
- [ ] Close the separate M4 empirical/production exit with measured
  synchronized multi-receiver RIRs, a licensed calibrated HRTF decoder, and
  controlled listening for metallic coloration and spatial plausibility.
- [x] Complete M5.1 controlled-room campaign schema, room-disjoint split,
  multi-objective calibration-loss contract, reusable template, and legacy-bank
  readiness audit.
- [x] Complete the M5.2 noise-free synthetic-recovery inverse calibration and
  local-identifiability baseline with multi-start and held-out-position gates.
- [x] Extend M5.2 evidence with controlled noise/model-mismatch perturbations,
  calibrated nuisance correction, a smooth M4-consistent multi-term proxy, and
  waveform-only ablation.
- [x] Map mixing time, aggregate coherent-path gain, and octave RT60 through the
  actual M4 PathEvent/multiband-FDN coupling with discrete topology profiling.
- [x] Split the coherent proxy into six effective boundary/path groups, accept
  the full-rank groups, and reject mono absorption/scattering separation.
- [x] Complete the readiness-gated M5.3 runner, synchronized M5.4 candidate
  profile, constrained M5.5 residual/ablations, and M5.6 aggregate exit.
- [ ] Acquire or ingest a controlled M5 campaign containing retained repeated
  raw ESS recordings, transducer calibrations, geometry/poses/environment,
  deconvolution provenance, and synchronized receiver channels.
- [ ] Close M5 empirical/production exit with measured position/room holdouts,
  controlled listening, and room-disjoint downstream evidence.
- [x] Complete M6.1 versioned bank contract with deterministic acoustic-space
  splits, content hashes, provenance, negative controls, and fail-closed
  production claims.
- [x] Complete M6.2 actual generator integration with exact
  serial/parallel/resume reproducibility and config isolation.
- [x] Complete M6.3 per-item QC/quarantine before building M6 release
  candidates.
- [x] Complete M6.4 content-addressed distributions, signal variants, release
  recipes, and fail-closed measured/mixed recipe handling.
- [x] Complete M6.5 bank/listening/downstream evaluation contracts and
  implementation gates without manufacturing empirical evidence.
- [x] Complete M6.6 immutable promotion-certificate implementation and
  fail-closed production reader.
- [x] Integrate release recipes into training augmentation and freeze a matched
  Pyroomacoustics/PathEvents-M4 pilot workflow.
- [ ] Supply measured, human-listening, trained downstream, renderer-approval,
  and sign-off evidence so the M6.6 decision can change from blocked to
  approved.

## 8. First M0 baseline observation

Date: 2026-07-30  
Report: `egs/rir_generation/exp/rir_realism/m0/rir_benchmark_v0/acoustics.json`  
Frozen configuration: `egs/rir_generation/phases/m0_baseline/config/baseline_v0.json`  
Sampling: 300 channels per bank, seed 0, broadband plus valid octave bands

Compared banks:

- synthetic: `hybrid_rir_16k_levels/wide`;
- measured: `real_rir_16k_train_view`.

Selected distance-bucket medians:

| Bank | Distance | DRR (dB) | C50 (dB) | qualified T30 (s) | spectral tilt (dB/oct) |
|------|----------|----------|----------|-------------------|------------------------|
| synthetic | 0–1 m | 1.61 | 10.48 | 0.55 | 1.12 |
| measured | 0–1 m | -2.11 | 15.02 | 0.32 | -2.29 |
| synthetic | 2–3.5 m | -8.51 | 4.98 | 0.56 | 2.50 |
| measured | 2–3.5 m | -4.25 | 12.91 | 0.33 | -2.60 |
| synthetic | 3.5–6 m | -9.62 | 4.10 | 0.67 | 3.31 |
| measured | 3.5–6 m | -6.50 | 6.02 | 0.34 | -2.63 |

Decay medians above require fit R² >= 0.9. Synthetic T30 coverage was 100%;
measured coverage was only 62–79% below 6 m and 0% above 6 m. Before decay
parameters are used for material fitting, M0 therefore needs measured-RIR
noise-floor detection or truncation. The raw low-quality 6 m+ fits produced
spurious values near 100 s and are now excluded from summaries rather than
reported as room decay.

The strongest stable gap is spectral: synthetic responses tilt upward while
measured responses tilt downward by roughly 2–3 dB/octave. The 2–3.5 m C50 gap
also shows that synthetic energy is distributed too heavily into the late field
relative to measured early reflections. The first M1 implementation target is
therefore **frequency-dependent surface materials**, followed by a focused
early/late energy audit. Broadband RT60 tuning alone cannot fix either gap.

## 9. Noise-corrected M0 decay baseline

Date: 2026-07-30  
Report: `egs/rir_generation/exp/rir_realism/m0/rir_benchmark_v0/acoustics_noise_corrected.json`  
Sampling: the same 300 channels per bank and seed 0

The shared analyzer now estimates 10 ms block energy, a stationary tail floor,
the Lundeby-style decay/noise intersection, available dynamic range, and
decay-line R². Reliable responses are truncated at the intersection and have
expected integrated noise energy removed. Clean synthetic decays retain their
full uncorrected curve.

| Bank | Distance | T30 (s) | qualified T30 | noise-corrected | median dynamic range |
|------|----------|---------|---------------|-----------------|----------------------|
| synthetic | 0–1 m | 0.55 | 100.0% | 0.0% | n/a (no stationary floor) |
| measured | 0–1 m | 0.33 | 65.9% | 39.0% | 61.8 dB |
| measured | 1–2 m | 0.29 | 67.9% | 34.0% | 63.9 dB |
| measured | 2–3.5 m | 0.32 | 82.4% | 42.6% | 60.8 dB |
| measured | 3.5–6 m | 0.35 | 65.7% | 22.9% | 56.5 dB |
| measured | 6 m+ | 2.82 | 9.5% | 9.5% | 47.5 dB |

The correction removes the earlier approximately 100 s noise-tail artifacts.
The 6 m+ result remains low-confidence because it represents only two qualified
channels; it must not be treated as the typical measured decay. Below 6 m, the
stable conclusion remains a measured T30 near 0.3–0.35 s, substantially shorter
than the v0 synthetic 0.55–0.67 s.

## 10. M2 modal validation and corrected probe

Date: 2026-07-30  
Development report:
`egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/modal_acoustics_dense_dev.json`  
Item-heldout report:
`egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/modal_acoustics_dense_item_heldout.json`

All probe variants contain the same 20 rooms × 5 positions: their 100
serialized scenes and generator configurations compare equal. The analyzer
uses 35–300 Hz, a fixed 0.8 s gate beginning 20 ms after the direct peak, and
the same peak/Q settings for synthetic and measured responses.

The first report at `modal_acoustics.json` exposed two analytic-probe defects
and is now superseded for model selection:

1. mode amplitude and phase were rank/position heuristics instead of
   `phi(source) * phi(receiver)` coupling;
2. sorting and retaining only the first 64 modes removed almost all synthetic
   200–300 Hz peaks.

The corrected probe uses reciprocal rectangular eigenfunction coupling,
`1/omega` impulse-response weighting, causal modal onset, and a 256-mode cap
that covers all 215 possible non-DC triplets at the default index limit.
Node, reciprocity, and causality tests pass.

Development medians (measured sampling seed 0):

| Variant | peaks/channel | median Q | bandwidth | peak spacing | prominence |
|---------|---------------|----------|-----------|--------------|------------|
| corrected M1 bridge | 11 | 39.17 | 4.05 Hz | 14.04 Hz | 13.90 dB |
| M2 material loss, scale 1.0 | 9 | 22.97 | 5.86 Hz | 22.58 Hz | 13.73 dB |
| M2 material loss, scale 0.58 | 14 | 39.27 | 3.39 Hz | 15.01 Hz | 14.42 dB |
| measured | 6 | 35.76 | 3.97 Hz | 14.89 Hz | 13.56 dB |

Item-heldout medians (measured sampling seed 1, zero measured-item overlap):

| Variant | peaks/channel | median Q | bandwidth | peak spacing | prominence |
|---------|---------------|----------|-----------|--------------|------------|
| corrected M1 bridge | 11 | 38.58 | 4.21 Hz | 14.77 Hz | 14.01 dB |
| M2 material loss, scale 1.0 | 8 | 22.24 | 6.06 Hz | 23.56 Hz | 12.89 dB |
| M2 material loss, scale 0.58 | 13 | 38.96 | 3.50 Hz | 15.50 Hz | 13.64 dB |
| measured | 6 | 42.58 | 3.48 Hz | 12.57 Hz | 14.07 dB |

For the item-heldout set, Wasserstein distances normalized by measured IQR are:

| Variant | Q | spacing | bandwidth | peak count |
|---------|---|---------|-----------|------------|
| corrected M1 bridge | 0.087 | 0.080 | 0.183 | 0.280 |
| M2 material loss, scale 1.0 | 0.505 | 0.614 | 0.364 | 0.325 |
| M2 material loss, scale 0.58 | 0.304 | 0.183 | 0.270 | 0.387 |

The scalar 0.58 was estimated from the development aggregate. It restores the
Q median but creates too many narrow visible modes and does not beat the
corrected M1 bridge jointly on development or heldout items. It is retained
only as a serialized diagnostic knob, not an accepted constant.

Decision:

- accept eigenfunction endpoint coupling, causal onset, and complete modal
  coverage as analytic-probe corrections;
- reject the current diffuse-absorption-to-modal-loss mapping and its scalar
  calibration as an M2 production model;
- retain the exact damped recurrence and boundary-participation code as tested
  infrastructure;
- next model low-frequency complex impedance, validate frequency-dependent
  phase/loss against FDTD, and require a true room-disjoint measured split.

The heldout sample is item-disjoint, not proven room-disjoint, so even the
corrected M1 bridge result is diagnostic rather than a training-bank acceptance.
