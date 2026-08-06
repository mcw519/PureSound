# Low-frequency modal validation

繁體中文版本：[`modal_validation.zh-TW.md`](modal_validation.zh-TW.md)。完整背景、
算法與目前實驗判斷見
[`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md)。

M2 has two validation components that do not reuse scene metadata as their
answer: an independent 3D finite-difference reference and a pressure-response
modal estimator.

## Phase-aware impedance primitives

`puresound.audio.rir.physics.impedance.admittance` establishes the first
complex-boundary validation layer in SI units of Pa·s/m:

```text
Z0 = rho c
Gamma = (Z - Z0) / (Z + Z0)
alpha = 1 - |Gamma|^2
Z = Z0 (1 + Gamma) / (1 - Gamma)
```

It enforces passive reflection magnitude and non-negative real impedance.
The inverse path from absorption requires an explicit reflection phase:
`|Gamma| = sqrt(1 - alpha)`. Tests verify complex reflection/impedance
round-trips, magnitude, phase, matched/rigid/pressure-release limits, and the
fact that identical absorption with different phase produces different
impedance.

The scene schema now defines `impedance_real` and `impedance_imag` as Pa·s/m,
validates passive spectra, preserves them through JSON, and combines fully
specified surface patches by area-weighted admittance. The catalog intentionally
does not invent impedance for materials that have absorption-only evidence.

## Passive causal relaxation boundary

The first time-domain complex boundary uses a normalized positive-real
admittance:

```text
y(s) = rho*c*Y(s) = g_infinite + g_relaxation / (1 + s*tau)
tau = 1 / (2*pi*f_relaxation)
Z(s) = rho*c / y(s)
Gamma(s) = (1 - y(s)) / (1 + y(s))
```

The infinite-frequency and zero-frequency admittances are constrained to be
non-negative; their difference may have either sign. The model is discretized
with a bilinear transform, which maps its stable analog pole inside the digital
unit circle. A single-wall impulse-response test compares the digital complex
reflection with the analog target. At 16 kHz the absolute complex error remains
below 0.003 through the 0–300 Hz validation band, including the relaxation
frequency.

`simulate_fdtd_reference(..., boundary_admittance=model)` maintains the
relaxation pressure state independently at every wall cell and writes the
complete model/discretization metadata. Setting the relaxation conductance to
zero reduces to the old real boundary; an automated regression verifies that
the two FDTD RIRs then agree sample for sample. A nonzero relaxation case is
also checked for finite output, stable state, and strict-JSON metadata.

This model proves the time-domain boundary mechanism. The reference priors
below remain validation-only and are not connected to production scene
materials or the pytARD modal backend.

## Direct-measurement, multi-pole, and resonant path

M2.4 adds a strict normal-incidence complex-impedance ingestion path rather
than treating absorption as a phase-bearing measurement. A measurement consists
of frequency, real impedance, and imaginary impedance columns in Pa·s/m plus a
JSON sidecar containing the method, environment, sample configuration, source,
and license. Optional real/imaginary standard deviations must be supplied
together. Non-increasing frequency, negative resistance, passive-reflection
violations, diffuse incidence, and missing provenance are rejected.

The imported response is fit with the positive-real multi-pole normalized
admittance

```text
y(s) = g_static
     + sum(g_low[k]  / (1 + s*tau[k]))
     + sum(g_high[k] * s*tau[k] / (1 + s*tau[k]))
```

All branch strengths are non-negative and all poles are stable real relaxation
poles. Passivity is therefore enforced by the model structure. The current
fitter holds the poles fixed and minimizes real and imaginary pressure-
reflection error with bounded least squares. It deliberately does not claim to
be full vector fitting with pole relocation and passivity repair.

The FDTD reference now holds one low-pass state per wall cell and pole. A
high-pass branch reuses that state as `p - lowpass(p)`. A synthetic two-pole
measurement is recovered below `1e-7` maximum complex-reflection error when the
known poles are supplied, and a nonzero multi-pole boundary remains passive and
finite in the 3D reference case.

The complete schema, equations, caveats, and Traditional Chinese explanation
are in [`impedance_measurements.zh-TW.md`](impedance_measurements.zh-TW.md).

M2.5 adds the first licensed direct complex dataset: CC BY 4.0 normalized
resistance/reactance for nominally identical NASA and UFSC perforated liners
under no-flow, 130 dB grazing-duct conditions. The source values remain
dimensionless `Z/(rho*c)` because the HDF5 file does not report the exact
normalization atmosphere. Metadata preserves the measurement geometry and
marks both samples as validation-only.

The measured Helmholtz reactance crossing cannot be represented by the
real-relaxation-pole model, so a passive series-RLC admittance branch was
added. Its fitted conjugate poles are stable by construction and its Tustin
biquad is prewarped at resonance. The NASA fit has 0.0385 held-out RMS and
0.0590 held-out maximum complex Cayley error over alternating frequency bins.
A 4096-point sweep remains passive, the resonant FDTD state is finite, and the
1D resonant/magnitude-only modal-Q ratio is 2.94. Source decisions and
limitations are recorded in
[`complex_impedance_source_audit.zh-TW.md`](complex_impedance_source_audit.zh-TW.md).

## First provenance-bearing porous references

`puresound.audio.rir.physics.impedance.priors` implements Miki's 1990
positive-real porous model for a rigid-backed layer. The first two references use Tarnow's measured
normal-direction flow resistivities for 100 mm, 14 kg/m³ and 30 kg/m³ glass
wool. Their evidence tier is explicitly
`measured_flow_resistivity_plus_miki_model`: impedance phase is modeled, not
directly measured.

The one-pole FDTD boundary is fit in complex reflection space. Maximum complex
errors are 0.0571 over 60–300 Hz for the 14 kg/m³ reference and 0.0160 over
155–300 Hz for the 30 kg/m³ reference. Both pass the diagnostic 0.08 gate.
Extrapolation below the empirical validity range is rejected.

A controlled 2.0 × 1.2 × 1.0 m FDTD case compares the 14 kg/m³ phase-aware fit
with a real boundary having the same reflection magnitude at 80 Hz. The
dominant peak changes from 85.7 Hz and Q 10.5 to 64.0 Hz and Q 15.8. This
demonstrates why absorption magnitude cannot substitute for phase; it is not a
measured-room validation.

The fitted high-frequency admittance also exposed an edge/corner time-step
limit beyond the interior CFL criterion. FDTD now enforces a summed normalized
boundary Courant limit of 0.9 and serializes both limits. Full derivation,
sources, and caveats are in [`impedance_priors.md`](impedance_priors.md).

## Complex modal eigenvalue reference

`puresound.audio.rir.physics.impedance.modes` is the minimal connection
between the same rational boundary used by FDTD and a complex modal
eigenproblem. For a 1D
cavity of length `L` with the same locally reacting boundary at both ends it
solves:

```text
1 - Gamma(s)^2 exp(-2 s L / c) = 0
s = -decay_rate + j*angular_frequency
Q = angular_frequency / (2*decay_rate)
```

A frequency-independent real boundary matches the closed-form
`f_n = n*c/(2*L)` and `decay_rate = -(c/L) log(|Gamma|)` to numerical precision.
A phase-aware two-pole boundary also shifts the mode relative to a real
boundary matched to its reflection magnitude. This validates the nonlinear
root formulation.

## Separable 3D rational-impedance eigenproblem

M2.7 extends the same rational \(y(s)\) to a rectangular 3D room. At every
wall:

```text
normal derivative(p) + (s/c) y(s) p = 0
```

For an axis of length \(L\), define \(h_\pm=(s/c)y_\pm(s)\). Its complex
spatial wavenumber must satisfy:

```text
(h_minus*h_plus - k^2) sin(kL)
    + k(h_minus + h_plus) cos(kL) = 0
```

The three axes share the same temporal pole and additionally satisfy:

```text
kx^2 + ky^2 + kz^2 + (s/c)^2 = 0
```

The implementation solves these four complex equations together and uses
boundary-strength continuation to remain on the requested rigid-mode branch.
It also evaluates the resulting complex separable eigenfunction at the source
and receiver with a bilinear volume normalization.

Validation results:

| Gate | Result |
|------|--------|
| x-only static boundary | 3D frequency, decay and Q match exact 1D |
| uniform cubic boundary | (1,0,0), (0,1,0), (0,0,1) remain degenerate |
| controlled phase-aware room | eigenproblem 63.8846 Hz / Q 17.32 |
| same room, independent FDTD | 64.0049 Hz / Q 15.78 |

The frequency difference is 0.19%; Q differs by 9.8%, inside the existing 15%
controlled-reference gate.

`ImpedanceModalLowFrequencyBackend` connects these poles and eigenfunctions to
the hybrid generator. The six-wall boundary comes from an explicit versioned
JSON and is never inferred from diffuse absorption. The eigenvalues are the
physical nonlinear solution. M2.8 now optionally loads a versioned fixed-pole
FDTD residue calibration. It uses the exact Ricker excitation, FDTD cell
volume/center convention, four training positions in two rooms, and two
untouched position holdouts. The holdout mean correlation is 0.9734 and NRMSE
is 0.2304, versus 0.2467 and 0.9703 for the fitted-gain legacy `1/f` sine
residue.

With that JSON, metadata sets `modal_residue_fdtd_validated: true`; without it,
the engineering fallback remains explicitly unvalidated. Both paths keep
`modal_residue_production_validated: false` and
`production_material_mapping_enabled: false` until measured-room transfer
functions pass.

M2.9 separates three evaluation axes instead of pooling all holdouts. For the
model-derived 50 mm and 100 mm rigid-backed thickness variants, respectively:

| Split | 50 mm correlation / NRMSE | 100 mm correlation / NRMSE |
|-------|----------------------------|-----------------------------|
| unseen positions | 0.9904 / 0.1366 | 0.9734 / 0.2304 |
| entirely unseen room C | 0.9929 / 0.1187 | 0.9933 / 0.1177 |
| 0.08 m grid after 0.12 m training | 0.9961 / 0.0919 | 0.9929 / 0.1193 |

A shared diagnostic fit across both thicknesses also passes all three gates.
This supports transfer across those two variants of one measured flow
resistivity, not general boundary invariance. The machine report therefore
keeps `general_boundary_invariance_established: false`.

## Independent FDTD reference

`puresound.audio.rir.physics.wave.fdtd` implements a small validation-only
solver. It
uses pressure at cell centers and particle velocity on staggered cell faces:

```text
rho dv/dt = -grad(p)
dp/dt = -rho c^2 div(v)
```

The time step follows the three-dimensional CFL limit. The original reference
case gives each wall a locally reacting, frequency-independent real impedance:

```text
R = sqrt(1 - alpha)
Z = rho c (1 + R) / (1 - R)
v_normal = p / Z
```

This is deliberately separate from the cosine-mode recurrence in
`hybrid_rir.py`. It is a reference for small rectangular scenes, not a dataset
renderer: its cost grows with grid volume, duration, and the inverse fourth
power of grid spacing when spatial and temporal refinement are combined.
The `alpha -> Z` compatibility path above selects a zero-phase, high-impedance
real branch. It must not be interpreted as a unique impedance recovered from
absorption.

The automated reference case uses a 3.0 × 2.5 × 2.0 m room, 0.15 m nominal
spacing, absorption 0.08, and a band-limited Ricker source. Results are:

| Mode | Rigid-room frequency | FDTD estimate | Frequency error |
|------|----------------------|---------------|-----------------|
| (1, 0, 0) | 57.17 Hz | 57.33 Hz | 0.3% |
| (0, 1, 0) | 68.60 Hz | 68.35 Hz | 0.4% |

The measured half-power Q values also agree within 20% with the decay expected
from the specified wall reflection coefficient. This tolerance includes grid
dispersion, the discrete boundary approximation, finite observation time, and
spectral overlap.

## RIR modal peak and Q estimator

`puresound.audio.rir.physics.wave.low_frequency` gates one measured or
synthetic RIR,
finds prominent low-frequency response peaks, and reports:

- peak frequency and prominence;
- lower and upper half-power crossings;
- `bandwidth = upper - lower`;
- `Q = peak frequency / bandwidth`;
- native time-gate resolution separately from the zero-padded FFT bin spacing.

Zero padding only improves crossing interpolation; it does not claim additional
physical resolution. Peaks whose half-power crossings cannot be resolved stay
in the result with `Q: null`.

An analytic 80 Hz sinusoid with an amplitude decay rate of 10/s has
`Q = 25.13`; the estimator returns approximately 25.10.

## Bank comparison

Use the same gate and estimator for generated and measured banks:

```bash
python egs/rir_generation/compare_modal_acoustics.py \
  v0=/path/to/v0 \
  m1=egs/rir_generation/exp/rir_realism/m1/rir_m1_probe100 \
  m2=egs/rir_generation/exp/rir_realism/m2/rir_m2_probe100 \
  measured=/path/to/measured \
  --per-bank 100 \
  --analysis-duration-s 0.8 \
  --reference-tag measured \
  --json-output egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/modal_acoustics.json
```

The report includes per-channel peaks plus bank distributions for peak count,
frequency, prominence, spacing, half-power bandwidth, and Q. A fixed delay
after the direct arrival suppresses the excitation pulse, and a fixed maximum
gate duration keeps spectral resolution comparable across banks.

These are response-level engineering metrics, not automatic room-mode labels.
Closely spaced modes, source/receiver nodes, measurement noise, non-rectangular
geometry, and finite recording length can merge, hide, or broaden peaks. Bank
acceptance therefore compares distributions rather than demanding a one-to-one
analytic mode match for measured rooms.

The initial 64-mode probe is superseded because it omitted almost all synthetic
200–300 Hz modes. With reciprocal eigenfunction endpoint coupling and complete
default modal coverage, the corrected M1 bridge is the strongest diagnostic
baseline. On zero-overlap heldout measured items its normalized Wasserstein
distances are 0.087 for Q, 0.080 for spacing, and 0.183 for bandwidth. Current
material-modal loss gives 0.505, 0.614, and 0.364 respectively.

A scalar material-loss calibration restores median Q but produces too many
narrow visible modes and still loses to the corrected M1 bridge jointly. The
new 3D complex-impedance eigenvalues supersede that damping law for controlled
experiments. The renderer has passed controlled numerical position-holdout
residue validation, but remains experimental pending multiple installed
boundaries, a compatible measured room finish, measured transfer functions,
and a true room-disjoint comparison.
