# Phase-aware low-frequency impedance priors

繁體中文版本：[impedance_priors.zh-TW.md](impedance_priors.zh-TW.md)

`puresound.audio.rir.physics.impedance.priors` holds a small catalog of
complex surface-impedance references for exercising the complex-boundary path
(passive fitting, FDTD boundaries, impedance modal solvers). It never attaches
an impedance to a scene material: every prior's metadata reports
`production_material_mapping_enabled: false`.

## Evidence tier

The module separates three kinds of evidence:

| Tier | Meaning |
|---|---|
| measured complex impedance | real and imaginary $Z(f)$ measured directly (see [complex impedance](impedance_measurements.md)) |
| measured parameter + published model | a measured non-acoustic property fed through a named physical model |
| engineering prior | parameters chosen for simulation; not evidence |

The catalog contains only the second tier, identified by
`EVIDENCE_TIER_MEASURED_PARAMETER_MODEL = "measured_flow_resistivity_plus_miki_model"`:
the airflow resistivity is measured, the complex impedance (in particular its
phase) is a model prediction, and it must not be described as a complex
impedance measurement. `MikiPorousLayerPrior` rejects any other tier and
requires at least one provenance URL.

A stronger prior needs a phase-aware normal-incidence measurement such as
ISO 10534-2 ([overview](https://www.iso.org/standard/81294.html)), which the
standard itself distinguishes from diffuse-incidence reverberation-room
absorption. The ingestion path for such data is the
[impedance-tube protocol](impedance_tube_protocol.md).

## Miki rigid-backed porous layer

`MikiPorousLayerPrior` evaluates Miki's positive-real modification of the
Delany–Bazley model ([Miki 1990](https://doi.org/10.1250/ast.11.19)) for a
porous layer of thickness $d$ (m) on a rigid backing. With airflow
resistivity $\sigma$ (Pa·s/m²), $X = 1000 f/\sigma$ and the
$e^{+j\omega t}$ convention:

$$Z_c = \rho c\,\bigl[1 + 5.50X^{-0.632} - j\,8.43X^{-0.632}\bigr],\qquad
k = \frac{\omega}{c}\bigl[1 + 7.81X^{-0.618} - j\,11.41X^{-0.618}\bigr]$$

$$Z_s = -j\,Z_c \cot(kd),\qquad
\Gamma = \frac{Z_s - \rho c}{Z_s + \rho c},\qquad
\alpha = 1 - |\Gamma|^2$$

| Method | Returns |
|---|---|
| `characteristic_impedance_pa_s_m(f, rho, c)` | $Z_c$, Pa·s/m |
| `complex_wavenumber_rad_m(f, c)` | $k$, rad/m |
| `surface_impedance_pa_s_m(f, rho, c)` | $Z_s$, Pa·s/m |
| `reflection_coefficient(f, rho, c)` | complex $\Gamma$ at normal incidence |
| `absorption_coefficient(f, rho, c)` | $\alpha$, clipped to [0, 1] |
| `metadata()` | catalog version, provenance, validity band, model and convention strings |

Every evaluation raises outside $0.01 \le f/\sigma \le 1$ (the
`validity_frequency_range_hz` property): the empirical model is never
extrapolated silently.

## Catalog

`reference_impedance_priors()` returns three immutable priors keyed by
`prior_id` (catalog version `puresound-impedance-priors.v0`). The flow
resistivities are Tarnow's measurements on glass-wool slabs
([Tarnow 2002](https://doi.org/10.1121/1.1476686)):

| `prior_id` | $\sigma$ | $d$ | valid band |
|---|---|---|---|
| `glass_wool_14kgm3_50mm_normal` | 5 880 Pa·s/m² | 0.05 m | 58.8–5 880 Hz |
| `glass_wool_14kgm3_100mm_normal` | 5 880 Pa·s/m² | 0.10 m | 58.8–5 880 Hz |
| `glass_wool_30kgm3_100mm_normal` | 15 500 Pa·s/m² | 0.10 m | 155–15 500 Hz |

The 50 mm entry is a model-derived thickness variant, not a measured installed
50 mm layer. None of these values stands in for carpet, ceiling tiles or other
fibrous finishes.

## Fitting a time-domain boundary

A time-domain solver cannot use $Z_s(f)$ directly; it needs a causal, passive
boundary. `fit_first_order_relaxation(prior, f_min, f_max, *,
air_density_kg_m3=1.204, sound_speed_m_s=343.0, num_frequencies=128,
acceptance_threshold=0.08)` fits `FirstOrderRelaxationAdmittance`, the
one-pole normalized admittance ($y = \rho c\,Y_\text{surface}$)

$$y(s) = g_\infty + \frac{g_0 - g_\infty}{1 + s\tau},\qquad \tau = \frac{1}{2\pi f_r},\qquad \Gamma = \frac{1-y}{1+y}$$

stored as `normalized_admittance_infinite` ($g_\infty$),
`normalized_admittance_relaxation` ($g_0 - g_\infty$) and
`relaxation_frequency_hz` ($f_r$).

- The fit runs in the complex pressure-reflection domain on
  `num_frequencies` log-spaced points; real and imaginary errors are stacked,
  so magnitude and phase are matched together.
- $g_0$, $g_\infty$ and $f_r$ are optimized in log space, which keeps both
  endpoint admittances non-negative and therefore the model positive-real.
  Three starting points are tried and the lowest cost wins.
- The fit band must lie inside the prior's validity band, and at least eight
  frequencies are required.
- `ImpedancePriorFit` reports RMS and maximum complex error, maximum
  magnitude and phase error, and `accepted` (maximum complex error
  $\le$ `acceptance_threshold`).

A fit with $g_0 < g_\infty$ (negative relaxation strength) is valid: the
endpoints stay non-negative, and a rising admittance is how the compliant
phase of a rigid-backed porous layer appears in this model.

## How it is used

- The six-wall configs
  `impedance_reference_glass_wool_14kgm3_{50,100}mm.json` (schema
  `puresound.rectangular_impedance_boundary.v1`) in
  `egs/rir_generation/phases/m2_impedance/config/` hold such fits. They are
  the input to `generate_hybrid_rir.py --low-backend analytic-impedance
  --impedance-boundary-config <json>` and to the FDTD validators described in
  [modal validation](modal_validation.md).
- Tests: `test/rir/test_impedance_priors.py` (provenance, validity band,
  passivity, fitting).

## Limits

These priors validate solvers; they do not describe rooms. Before a material
could carry an impedance in production it needs a direct normal-incidence
measurement of the installed configuration (thickness, backing, air gap,
mounting, uncertainty), a passive fit through the
[measurement contract](impedance_measurements.md), an explicit reviewed
mapping to compatible scene materials, and a treatment of oblique incidence,
which a normal-incidence locally reacting value does not provide.
