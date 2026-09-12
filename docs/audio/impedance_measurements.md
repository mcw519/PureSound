# Complex acoustic impedance

繁體中文版本：[impedance_measurements.zh-TW.md](impedance_measurements.zh-TW.md)

This path turns phase-preserving impedance measurements into passive boundary
models for modal and time-domain solvers. It is research and validation
infrastructure; measurements are not automatically copied into the production
material catalog.

The same material label does not imply the same thickness, backing, air gap,
mounting, or incidence condition.

## Why complex impedance matters

For characteristic impedance (Z_0=ho c),

[
Gamma(f)=rac{Z(f)-Z_0}{Z(f)+Z_0},
qquad
alpha(f)=1-|Gamma(f)|^2.
]

Absorption retains only the magnitude of reflection. It loses phase, which
changes reflection delay, modal frequency, and Q. This pipeline therefore
requires real and imaginary impedance instead of reconstructing phase from
diffuse-field absorption.

## SI measurement contract

`ComplexImpedanceMeasurement.from_csv_and_metadata()` reads one CSV and one
JSON sidecar.

Required CSV columns:

```csv
frequency_hz,impedance_real_pa_s_m,impedance_imag_pa_s_m
60.0,421.2,-880.4
80.0,398.1,-631.7
120.0,376.5,-402.8
```

Optional real and imaginary uncertainty columns must appear together.

The loader rejects non-increasing or non-positive frequency, fewer than three
points, non-finite values, negative real impedance, incomplete uncertainty,
and data implying a reflection magnitude above one.

Minimal metadata:

```json
{
  "schema_version": "puresound.complex_impedance_measurement.v1",
  "measurement_id": "lab_sample_001",
  "method": "ISO 10534-2 two-microphone transfer function",
  "incidence": "normal",
  "environment": {
    "air_density_kg_m3": 1.204,
    "sound_speed_m_s": 343.0
  },
  "sample": {
    "material_label": "100 mm glass wool",
    "thickness_m": 0.1,
    "backing": "rigid",
    "air_gap_m": 0.0
  },
  "provenance": {
    "source_url": "https://example.org/measurement/001",
    "license": "CC-BY-4.0"
  }
}
```

Production measurements should also record tube geometry, microphone spacing,
sample batch and tolerance, mounting, temperature and humidity, calibration,
and repeatability.

ISO 10534-2 normal-incidence tube data is not interchangeable with
reverberation-room diffuse-incidence absorption.

## Normalized impedance contract

Some publications provide (z=Z/(ho c)) without the exact normalization
values. Use
`puresound.normalized_complex_impedance_measurement.v1` and CSV columns:

```csv
frequency_hz,normalized_impedance_real,normalized_impedance_imag
500,0.785,-3.596
600,0.474,-3.213
```

Metadata must state field geometry, phasor convention, flow Mach number,
source SPL, sample, provenance, license, and applicability limits.

The bounded fitting coordinate is

[
mathcal C(z)=rac{z-1}{z+1}.
]

It equals normal-incidence pressure reflection only for the appropriate
locally reacting interpretation. For grazing-duct data it is primarily a
stable Cayley transform; do not relabel that data as an impedance-tube
measurement.

## Passive boundary fitting

Discrete measurements need a causal model before use in a time-domain solver.
The normalized admittance model uses non-negative parallel branches:

[
y(s)=g_s+
sum_krac{g_{L,k}}{1+s	au_k}+
sum_k g_{H,k}rac{s	au_k}{1+s	au_k},
qquad
	au_k=(2pi f_{p,k})^{-1}.
]

[
Z(s)=rac{ho c}{y(s)},
qquad
Gamma(s)=rac{1-y(s)}{1+y(s)}.
]

All gains are constrained non-negative, making the parallel sum passive by
construction. `fit_passive_multi_pole_admittance()` fits real and imaginary
reflection jointly with fixed poles and reports complex, magnitude, and phase
errors.

Fixed poles are deliberately conservative. If the acceptance gate fails,
adjust model order, pole placement, or the measurement band; do not remove
the passivity constraint.

Resonant materials can require passive conjugate-pole or series-RLC branches.
With (u=s/omega_0),

[
y_k(s)=rac{g_{mathrm{peak},k}}{Q_k}
       rac{u}{u^2+u/Q_k+1}.
]

Use such branches only when the measured reactance supports the resonance and
the fit remains within its declared applicability range.

## Solver use

The fitted model supplies boundary reflection or admittance consistently to:

- path-event boundary filters;
- the impedance-modal rectangular-room solver;
- the FDTD reference solver.

For rectangular dimensions (L_x,L_y,L_z), the rigid reference frequencies
are

[
f_{n_xn_yn_z}=rac{c}{2}
sqrt{left(rac{n_x}{L_x}ight)^2+
      left(rac{n_y}{L_y}ight)^2+
      left(rac{n_z}{L_z}ight)^2}.
]

Complex impedance shifts those frequencies and introduces decay. Compare
modal and FDTD results only with matching geometry, environment, boundary
orientation, and phasor convention.

These solvers are validation tools. A good fit to one normal-incidence sample
does not prove a diffuse, angle-dependent room boundary model.

## Measurement workflow

A template for two-microphone impedance-tube reduction is available at:

```text
egs/rir_generation/phases/m2_impedance/measurements/
  impedance_tube_template/
```

Reduce raw transfer data:

```bash
python egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py --help
```

Validate a measurement and fit:

```bash
python egs/rir_generation/phases/m2_impedance/scripts/validate_impedance_measurement.py --help
```

The retained public liner examples under `measurements/zenodo_15195587/`
exercise normalized grazing-duct ingestion and resonant fitting. They are
validation data, not production wall-material measurements.

## Release rules

Before promoting a fitted boundary:

1. Verify source files, provenance, and redistribution license.
2. Preserve measurement geometry and phasor convention.
3. Check passivity over and beyond the fit band.
4. Report held-out complex, magnitude, and phase error.
5. Cross-check modal and FDTD behavior.
6. Keep thickness, backing, air gap, mounting, and environmental limits.
7. Add the model to a material catalog only through an explicit reviewed
   mapping.
