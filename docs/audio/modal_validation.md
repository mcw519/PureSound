# Low-frequency modal validation

繁體中文版本：[modal_validation.zh-TW.md](modal_validation.zh-TW.md)

This page describes how low-frequency impedance and modal implementations are
checked. It documents validation methods, not a claim that a model is ready
for production rooms.

## Validation layers

The implementation uses independent checks:

1. algebraic impedance/reflection round trips;
2. passive causal digital boundary responses;
3. one-dimensional complex modes;
4. separable three-dimensional modes;
5. an independent three-dimensional FDTD reference;
6. held-out source/receiver positions.

Agreement between layers is useful because scene metadata is never reused as
the expected acoustic answer.

## Impedance primitives

For (Z_0=ho c):

```text
Gamma = (Z - Z0) / (Z + Z0)
alpha = 1 - |Gamma|^2
Z = Z0 (1 + Gamma) / (1 - Gamma)
```

Tests cover complex round trips, phase, magnitude, matched and rigid limits,
and the requirement that passive surfaces have non-negative real impedance.

Absorption alone cannot supply reflection phase. Scene materials therefore
leave impedance unset unless phase-aware evidence or an explicitly versioned
prior exists.

## Passive time-domain boundary

Normalized admittance models are positive-real and discretized with a bilinear
transform. The resulting poles must remain inside the digital unit circle.

The FDTD solver keeps independent boundary state at each wall cell. Validation
checks:

- finite output and state;
- strict causality;
- passivity across the declared band;
- the zero-relaxation limit against the real-boundary implementation;
- digital reflection against the analog target.

Multi-pole and resonant boundaries use one state per branch. See
[Complex impedance](impedance_measurements.md) for the fitting equations and
measurement contracts.

## One-dimensional modes

For a cavity of length (L) with the same locally reacting boundary at both
ends, solve:

[
1-Gamma(s)^2 e^{-2sL/c}=0,
qquad
s=-	ext{decay rate}+j,	ext{angular frequency}.
]

[
Q=rac{	ext{angular frequency}}{2,	ext{decay rate}}.
]

A frequency-independent real boundary is checked against its closed-form mode
frequency and decay. A phase-aware boundary must produce the expected shift
relative to a magnitude-matched real boundary.

## Three-dimensional modes

At a wall, normalized admittance (y(s)) gives:

```text
normal_derivative(p) + (s/c) y(s) p = 0
```

For one axis of length (L), with
(h_pm=(s/c)y_pm(s)), the spatial wavenumber satisfies:

[
(h_-h_+-k^2)sin(kL)+k(h_-+h_+)cos(kL)=0.
]

The axes share one temporal pole:

[
k_x^2+k_y^2+k_z^2+(s/c)^2=0.
]

The solver uses continuation from the rigid-room mode to avoid jumping to an
unrelated root. Validation includes:

- reduction to the exact 1D case;
- expected degeneracy in a uniform cube;
- frequency and Q comparison with independent FDTD output;
- source/receiver eigenfunction evaluation.

## Residue calibration

`ImpedanceModalLowFrequencyBackend` needs both complex poles and residues.
Optional residue calibration uses fixed-pole FDTD responses at several
source/receiver positions. Evaluation positions must be excluded from fitting.

Metadata distinguishes:

- a calibrated residue used within its validated geometry and frequency range;
- an engineering fallback without FDTD validation;
- production validation, which requires measured-room transfer functions.

Do not infer production approval from a good synthetic FDTD fit.

## Evidence tiers

Keep the evidence source explicit:

| Tier | Meaning |
|---|---|
| Direct complex measurement | Real and imaginary impedance were measured |
| Measured property plus model | A measured property feeds a named physical model |
| Engineering prior | Parameters are selected for simulation or diagnosis |
| Synthetic reference | One implementation is checked against an independent solver |

For example, a porous model built from measured flow resistivity still has
modeled impedance phase. Public grazing-duct liner data preserves its original
geometry and is not relabeled as normal-incidence wall data.

## Reproducing checks

Relevant implementation:

```text
puresound/audio/rir/physics/impedance/
puresound/audio/rir/physics/wave/
puresound/audio/rir/render/low_frequency/
```

Relevant validators and calibration tools:

```text
egs/rir_generation/phases/m2_impedance/scripts/
```

Use each script's `--help` for its inputs. Generated reports belong in local
experiment output unless a release process explicitly retains them.

A production boundary model additionally needs traceable real-room evidence,
held-out rooms and positions, declared applicability limits, and a reviewed
mapping into the material catalog.
