# Material-derived low-frequency modal damping

繁體中文版本：[`modal_damping.zh-TW.md`](modal_damping.zh-TW.md)。完整背景、算法與
目前實驗判斷見 [`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md)。

M2 introduces an explicit experimental low-frequency path in
`puresound.audio.rir.render.low_frequency`. Select it with one of:

```text
analytic-material
pytard-material
pytard-cupy-material
```

These variants require `--scene-version v1`. The corresponding legacy backend
names retain the shared RT60 behavior so M1 and M2 remain independently
benchmarkable.

## Boundary participation

For a rectangular room, the rigid-wall pressure eigenfunctions are:

```text
phi_n(x,y,z) =
  cos(nx pi x / Lx) cos(ny pi y / Ly) cos(nz pi z / Lz)
```

The volume norm along one dimension is `L` for mode index zero and `L/2`
otherwise. For each mode frequency, PureSound interpolates the six serialized
surface absorption spectra and computes:

```text
P_n =
  (alpha_west + alpha_east) / Ix
  + (alpha_south + alpha_north) / Iy
  + (alpha_floor + alpha_ceiling) / Iz

gamma_n = s_loss c P_n / 8
zeta_n = gamma_n / omega_n
RT60_n = ln(1000) / gamma_n
Q_n = omega_n / (2 gamma_n)
```

`c P_n / 4` is the first-order Sabine energy-loss rate; modal amplitude decays
at half that rate. This weak-loss model deliberately uses absorption rather
than a fitted room-wide RT60. A mode whose index is nonzero along one dimension
has twice the participation of that opposing wall pair, so changing one surface
affects modes differently.

`s_loss` defaults to 1.0. The CLI exposes
`--material-modal-loss-scale` only for controlled diagnostics and serializes it
with every result. A development-derived value of 0.58 restored aggregate
median Q but worsened the joint heldout peak-count/bandwidth distribution, so
it is not an accepted physical constant.

## Source/receiver modal coupling

The analytic probe backend excites and observes each rectangular mode with its
eigenfunction values at the actual endpoints:

```text
A_n is proportional to
  (2 - delta_nx0) (2 - delta_ny0) (2 - delta_nz0)
  phi_n(source) phi_n(receiver) / omega_n
```

The factors in parentheses are the inverse cosine-mode volume norms. A source
or receiver on a modal node therefore suppresses that mode; swapping source and
receiver leaves the response unchanged. Modal tails begin at the geometric
direct-arrival time and at zero phase, so the finite analytic expansion does
not emit a pre-arrival tail.

The probe enumerates every index triplet from 0 through 5 that lies in the low
band. Its explicit safety cap is 256 modes, above the 215 possible non-DC
triplets at that index limit. The previous fixed first-64 truncation omitted
most 200–300 Hz resonances and is not used by new probes. Metadata records the
coupling model, index limit, and mode cap.

## Exact damped recurrence

The vendored pytARD adapter already diagonalizes the room into independent
modes. M2 replaces the undamped homogeneous recurrence with the exact sampled
poles of:

```text
q_n'' + 2 gamma_n q_n' + omega_n^2 q_n = f_n
```

For sample interval `dt`, damped frequency `wd`, and pole radius `r`:

```text
r   = exp(-gamma_n dt)
wd  = sqrt(omega_n^2 - gamma_n^2)
a1  = 2 r cos(wd dt)
a2  = -r^2
b   = (1 - a1 - a2) / omega_n^2

q[k+1] = a1 q[k] + a2 q[k-1] + b f[k]
```

At zero damping this reduces exactly to the previous pytARD recurrence. The
material path disables `apply_rt60_decay_envelope`; metadata records
`global_rt60_envelope_applied: false`.

## Metadata and validation

Each material-damped item records:

- modal indices and frequency;
- amplitude decay rate;
- predicted modal RT60 and Q;
- min/median/max RT60 and Q across the sampled modes;
- the boundary model and absence of a global envelope.

Current tests verify:

- the zero-damping-compatible recurrence remains stable;
- a change to the west wall affects the x-axial mode twice as strongly as a
  mode with zero x index;
- modes in one scene have different decay rates;
- strongly absorbing boundaries reduce the exact pytARD late modal energy by
  more than 40 dB relative to reflective boundaries;
- an independent staggered-grid 3D FDTD reference recovers the first two room
  modes within 1% in frequency and 20% in Q;
- a damped-sinusoid fixture recovers spectral Q within 2%;
- the analytic probe obeys modal nodes, causality, and source/receiver
  reciprocity;
- existing causal crossover and bounded-gain regression tests still pass.

The FDTD formulation, numeric reference values, response-level peak/Q
estimator, and bank comparison command are documented in
[`modal_validation.md`](modal_validation.md).

## Current M2 boundary

This is the first M2 implementation, not its final acceptance:

- the boundary loss is a first-order absorption/participation model;
- complex impedance phase is exercised in validation FDTD, and the same
  rational boundary now drives a validated 1D complex cavity eigenproblem, but
  neither is yet applied to this production-oriented 3D modal recurrence;
  angle dependence is also absent;
- mode coupling from non-shoebox geometry is absent;
- the corrected dense paired comparison shows excess, incorrectly shaped
  damping; a scalar loss calibration does not fix the joint distribution;
- direct installed-material impedance evidence, 3D modal-eigenproblem
  integration, and downstream tests are required before making the material
  backend a production default.

The complex measurement contract, passive multi-pole fitting, FDTD state, and
1D eigenvalue reference are documented in
[`impedance_measurements.zh-TW.md`](impedance_measurements.zh-TW.md).
