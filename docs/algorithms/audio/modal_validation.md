# Low-frequency modal validation

繁體中文版本：[modal_validation.zh-TW.md](modal_validation.zh-TW.md)

This page describes how the low-frequency impedance and modal implementations
are checked against each other and against an independent wave solver. It
documents methods; passing them does not make a model ready for production
rooms.

## Validation layers

1. algebraic impedance/reflection round trips;
2. passive, causal digital boundary responses;
3. one-dimensional complex cavity modes;
4. separable three-dimensional impedance modes;
5. an independent three-dimensional FDTD reference;
6. held-out source/receiver positions.

The layers are independent: scene metadata is never reused as the expected
acoustic answer, so agreement between two layers is evidence about both.

## Impedance primitives

With $Z_0 = \rho c$:

$$\Gamma = \frac{Z - Z_0}{Z + Z_0},\qquad \alpha = 1 - |\Gamma|^2,\qquad Z = Z_0\,\frac{1+\Gamma}{1-\Gamma}$$

Tests (`test/rir/test_acoustic_impedance.py`) cover complex round trips,
phase and magnitude, the matched and rigid limits, and rejection of a
negative real impedance. Absorption alone cannot supply phase, so a
`SurfaceMaterial` carries impedance only when both `impedance_real` and
`impedance_imag` spectra are given explicitly; nothing fills them from
absorption.

## Passive time-domain boundary

Normalized admittance models are positive-real and discretized with the
bilinear transform ([complex impedance](impedance_measurements.md)); the
digital poles must stay inside the unit circle. Checks: finite output and
state, strict causality, passivity across the declared band, the
zero-relaxation limit against the real-impedance boundary, and the digital
reflection against the analog target.

## FDTD reference

`puresound.audio.rir.physics.wave.fdtd.simulate_fdtd_reference(config, ...)`
is a small staggered pressure/particle-velocity solver written only for
validation; it shares no code with the modal recurrences it checks.

- `FDTDReferenceConfig` sets room size, grid spacing, duration, $c$, $\rho$,
  interior CFL (default 0.92), source/receiver positions, and a Ricker source
  (default 180 Hz centre). `spatial_derivative_order` is 2 or 4, with a
  matching `near_wall_closure` and `boundary_pressure_scheme`; incompatible
  combinations are rejected.
- Pass exactly one boundary: `boundary_absorption` (frequency-independent
  real impedance) or `boundary_admittance` (relaxation, multi-pole or
  resonant models, with independent state per wall cell and per branch).
- The interior CFL condition does not bound an explicit admittance term
  accumulated at edges and corners, so the time step is also limited by a
  boundary Courant number,
  $c\,\Delta t \sum_\text{axes} \max(y_-, y_+)/\Delta x \le 0.9$, using each
  model's all-frequency admittance bound (0.25 for the fourth-order
  `face_quadratic_time_quadratic` scheme; the centered quadratic scheme has
  no such cap). Both limits are serialized in the result
  (`interior_cfl_time_step_s`, `boundary_time_step_limit_s`).
- `simulate_reciprocal_fdtd_reference` solves the exchanged problem for
  reciprocity checks.

`physics.wave.low_frequency.estimate_low_frequency_modes(rir, fs, ...)` reads
modal peaks from a gated response: zero-padded FFT (factor 8), peaks with
≥ 6 dB prominence and ≥ 3 Hz separation, and
$Q = f_\text{peak}/\Delta f_{-3\,\text{dB}}$. It records the native resolution
of the gate, which zero padding does not improve.
`egs/rir_generation/compare_modal_acoustics.py TAG=PATH ...` applies the same
estimator after each channel's direct arrival to compare modal peak and Q
distributions across RIR banks.

## One-dimensional modes

For a cavity of length $L$ with the same locally reacting boundary at both
ends, `solve_1d_impedance_cavity_modes` solves

$$1 - \Gamma(s)^2 e^{-2sL/c} = 0,\qquad s = -\gamma + j\omega,\qquad Q = \frac{\omega}{2\gamma},$$

starting from the rigid mode $f = nc/(2L)$ with the decay implied by
$|\Gamma|$ there. A frequency-independent real boundary is checked against
its closed form; a phase-aware boundary must shift the mode relative to a
magnitude-matched real boundary.

## Three-dimensional modes

At a wall, normalized admittance $y(s)$ gives the Robin condition

$$\partial_n p + \frac{s}{c}\,y(s)\,p = 0.$$

For one axis of length $L$ with $h_\pm = (s/c)\,y_\pm(s)$, the spatial
wavenumber satisfies

$$(h_-h_+ - k^2)\sin(kL) + k(h_- + h_+)\cos(kL) = 0,$$

and the three axes share one temporal pole:

$$k_x^2 + k_y^2 + k_z^2 + (s/c)^2 = 0.$$

`solve_rectangular_impedance_modes(room_dim_m, boundary_config, *,
mode_indices, sound_speed_m_s=343.0, continuation_steps=8)` ramps the
admittance from weak to full over `continuation_steps`, tracking the requested
rigid-wall branch instead of jumping to an unrelated root. Eigenfunctions are
normalized with the complex bilinear volume norm. Checks: reduction to the
exact 1D case, the expected degeneracy of a uniform cube, frequency and Q
against the FDTD reference, and source/receiver eigenfunction evaluation.

## Residue calibration

`ImpedanceModalLowFrequencyBackend` needs complex residues as well as poles.
Without calibration it uses an engineering scale times the eigenfunction
coupling, recorded as such in metadata. With a calibration
(`--impedance-residue-calibration`), residues come from
`fit_fixed_pole_modal_residues`, which keeps the solved poles and
eigenfunctions fixed and fits one complex scale and a frequency power law
$(f_\text{ref}/f)^p$ (default $f_\text{ref} = 100$ Hz) against FDTD responses:

- training cases are RMS-normalized so a loud position does not dominate;
  holdout positions never enter the fit;
- acceptance requires mean holdout correlation ≥ 0.9 and NRMSE ≤ 0.35;
- the fitted residue is converted from the FDTD pressure-cell source
  convention to the free-field $1/r$ RIR convention
  (`convert_pressure_state_modal_residue`, factor $4\pi c^2/(f_s s)$);
- the result is stored as schema
  `puresound.impedance_modal_residue_calibration.v2` (v1 files still load);
- the backend refuses a band outside the boundary config's or the
  calibration's `valid_frequency_range_hz`.

Metadata distinguishes an FDTD-calibrated residue
(`modal_residue_fdtd_validated: true`) from the engineering fallback, and
always reports `modal_residue_production_validated: false`: production
validation would need measured-room transfer functions, and a good synthetic
FDTD fit does not grant it.

## Evidence tiers

| Tier | Meaning |
|---|---|
| direct complex measurement | real and imaginary impedance were measured |
| measured property plus model | a measured property feeds a named physical model |
| engineering prior | parameters chosen for simulation or diagnosis |
| synthetic reference | one implementation checked against an independent solver |

A porous model built from measured flow resistivity still has modelled phase
([impedance priors](impedance_priors.md)). Public grazing-duct liner data keeps
its original geometry and is not relabelled as normal-incidence wall data
([public source criteria](complex_impedance_source_audit.md)).

## Running the checks

Implementation: `puresound/audio/rir/physics/impedance/`,
`puresound/audio/rir/physics/wave/`, `puresound/audio/rir/render/low_frequency/`.

Tests: `test/rir/test_fdtd_reference.py`, `test_impedance_modes.py`,
`test_impedance_residues.py`, `test_acoustic_impedance.py`.

Validators and calibration tools live in
`egs/rir_generation/phases/m2_impedance/scripts/` (for example
`calibrate_impedance_modal_residues.py --boundary-config <json>
--output-calibration <json> --output-report <json>`) and
`egs/rir_generation/phases/m3_wave_path/scripts/` (FDTD boundary and
higher-order checks); each takes `--help`. Generated reports are local
experiment output unless a release explicitly keeps them.

A production boundary model additionally needs traceable real-room evidence,
held-out rooms and positions, declared applicability limits, and a reviewed
mapping into the material catalog.
