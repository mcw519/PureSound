# Complex acoustic impedance

繁體中文版本：[impedance_measurements.zh-TW.md](impedance_measurements.zh-TW.md)

This path turns phase-preserving impedance measurements into passive, causal
boundary models for the path-event renderer, the impedance modal solver and
the FDTD reference. It is research and validation infrastructure: nothing here
writes into the scene material catalog.

| Module | Contents |
|---|---|
| `physics/impedance/admittance.py` | reflection/impedance conversions; the passive admittance models; bilinear discretization |
| `physics/impedance/measurements.py` | the SI and normalized measurement contracts |
| `physics/impedance/fitting.py` | passive multi-pole and resonant fitting |
| `physics/impedance/modes.py` | 1D and separable 3D impedance modes; six-wall boundary config |

(all under `puresound/audio/rir/`)

## Why complex impedance

For a locally reacting surface with impedance $Z(f)$ in air of characteristic
impedance $Z_0 = \rho c$:

$$\Gamma(f) = \frac{Z(f) - Z_0}{Z(f) + Z_0},\qquad \alpha(f) = 1 - |\Gamma(f)|^2,\qquad Z = Z_0\,\frac{1+\Gamma}{1-\Gamma}$$

Absorption keeps only $|\Gamma|$. The phase it discards sets reflection delay,
modal frequency shift and modal Q, so this path requires real and imaginary
impedance and never reconstructs phase from diffuse-field absorption.
`impedance_from_absorption_and_phase()` makes the phase an explicit argument
for exactly that reason. Passivity means $\operatorname{Re} Z \ge 0$, i.e.
$|\Gamma| \le 1$; `normal_incidence_reflection_coefficient()` rejects a
negative real part and maps infinite impedance to a rigid $\Gamma = 1$.

The same material label does not imply the same thickness, backing, air gap,
mounting or incidence condition; the contracts below record all of them.

## SI measurement contract

`ComplexImpedanceMeasurement.from_csv_and_metadata(csv, json)` reads schema
`puresound.complex_impedance_measurement.v1`, normal incidence only.

CSV columns (Pa·s/m), with optional uncertainty columns
`impedance_real_std_pa_s_m` and `impedance_imag_std_pa_s_m`, which must appear
together:

```csv
frequency_hz,impedance_real_pa_s_m,impedance_imag_pa_s_m
60.0,421.2,-880.4
80.0,398.1,-631.7
120.0,376.5,-402.8
```

Minimal sidecar:

```json
{
  "schema_version": "puresound.complex_impedance_measurement.v1",
  "measurement_id": "lab_sample_001",
  "method": "ISO 10534-2 two-microphone transfer function",
  "incidence": "normal",
  "environment": {"air_density_kg_m3": 1.204, "sound_speed_m_s": 343.0},
  "sample": {"material_label": "100 mm glass wool", "thickness_m": 0.1,
             "backing": "rigid", "air_gap_m": 0.0},
  "provenance": {"source_url": "https://example.org/measurement/001",
                 "license": "CC-BY-4.0"}
}
```

The loader rejects: another schema or incidence; fewer than three points;
frequencies that are non-positive or not strictly increasing; non-finite
values; negative real impedance; unpaired or negative uncertainties; missing
`sample.material_label`, `provenance.source_url` or `provenance.license`; and
any point whose reflection magnitude exceeds one. A real measurement should
also record tube geometry, microphone spacing, sample batch and tolerance,
temperature, humidity, calibration and repeatability. The
[impedance-tube protocol](impedance_tube_protocol.md) produces this format.

ISO 10534-2 normal-incidence tube data is not interchangeable with
reverberation-room diffuse-incidence absorption.

## Normalized measurement contract

Some publications give $z = Z/(\rho c)$ without the exact $\rho$ and $c$ used
to normalize. Multiplying back with an assumed atmosphere would present that
assumption as the measurement environment, so
`NormalizedComplexImpedanceMeasurement` keeps the dimensionless values under
schema `puresound.normalized_complex_impedance_measurement.v1`:

```csv
frequency_hz,normalized_impedance_real,normalized_impedance_imag
500,0.785,-3.596
600,0.474,-3.213
```

(optional `normalized_impedance_real_std` / `normalized_impedance_imag_std`).
The sidecar must give `acoustic_field_geometry` (`normal_incidence_tube` or
`grazing_duct`), `phasor_convention` (`exp(+i*omega*t)`),
`conditions.mean_flow_mach` (≥ 0), `conditions.source_spl_db` (> 0),
`sample.material_label`, `provenance.source_url`, `provenance.license` and
`applicability.scope`.

Its bounded fitting coordinate is the Cayley transform

$$\mathcal{C}(z) = \frac{z - 1}{z + 1},$$

which equals the normal-incidence pressure reflection only under a locally
reacting normal-incidence interpretation. For grazing-duct data it is a
bounded numerical coordinate; such data is never relabelled as an
impedance-tube measurement.

## Passive boundary models

A time-domain solver needs a causal model, not samples. All models are
written as a dimensionless admittance $y(s) = \rho c / Z(s)$ with
$\Gamma = (1 - y)/(1 + y)$ at normal incidence.

**Multi-pole relaxation** (`PassiveMultiPoleAdmittance`):

$$y(s) = g_s + \sum_k \frac{g_{L,k}}{1 + s\tau_k} + \sum_k g_{H,k}\,\frac{s\tau_k}{1 + s\tau_k},\qquad \tau_k = \frac{1}{2\pi f_{p,k}}$$

Every gain is non-negative and each branch is positive-real, so the parallel
sum is passive by construction; no post-fit passivity repair exists.
`FirstOrderRelaxationAdmittance` is the one-pole case used by the
[impedance priors](impedance_priors.md).

**Resonant** (`PassiveResonantAdmittance`), a parallel sum of series-RLC
branches with $u = s/\omega_0$:

$$y_k(s) = \frac{g_{\mathrm{peak},k}}{Q_k}\,\frac{u}{u^2 + u/Q_k + 1}$$

The conjugate pole pair represents a Helmholtz-like resonance that real
relaxation poles cannot. Use it only where the measured reactance shows the
resonance.

**Discretization.** `digital_normalized_admittance_filter()` maps relaxation
sections with the bilinear transform and resonant branches with individually
prewarped bilinear biquads; the parallel sum is combined exactly as
polynomials in $z^{-1}$. The bilinear transform preserves positive-realness,
so `digital_locally_reacting_reflection_filter(model, cos_theta, fs)`, which
realizes the angle-aware reflection

$$\Gamma_\theta = \frac{\cos\theta - y}{\cos\theta + y},$$

is bounded-real ($|\Gamma_\theta| \le 1$ on the unit circle). The result is a
`DigitalBoundaryReflectionFilter` (schema
`puresound.digital_boundary_reflection_filter.v1`).

## Fitting

| Function | Model | Defaults |
|---|---|---|
| `fit_passive_multi_pole_admittance(f, Z, *, air_density_kg_m3, sound_speed_m_s, pole_frequencies_hz, reflection_weights=None, acceptance_threshold=0.05)` | multi-pole, fixed real poles | gains bounded to [0, 100]; three starts |
| `fit_complex_impedance_measurement(measurement, *, num_poles=3, pole_frequencies_hz=None, acceptance_threshold=0.05)` | multi-pole | poles log-spaced over $[0.5 f_\min, 2 f_\max]$ (one pole: $\sqrt{f_\min f_\max}$) |
| `fit_passive_single_resonance_admittance(f, z, *, training_frequency_indices=None, acceptance_threshold=0.1)` | one RLC branch + static conductance, free pole pair | held-out metrics on indices not trained on |
| `fit_normalized_complex_impedance_measurement(measurement, *, alternating_frequency_holdout=True, acceptance_threshold=0.1)` | resonant | trains on even bins, holds out odd bins |

All fits minimise stacked real and imaginary reflection error and report RMS
and maximum complex error plus maximum magnitude and phase error; `accepted`
compares the maximum complex error with the threshold. When a measurement
carries uncertainties, `fit_complex_impedance_measurement` propagates them to
the reflection domain,

$$\sigma_\Gamma = \left|\frac{\partial\Gamma}{\partial Z}\right|\sqrt{\tfrac12(\sigma_R^2 + \sigma_I^2)},\qquad \frac{\partial\Gamma}{\partial Z} = \frac{2Z_0}{(Z + Z_0)^2},$$

and weights each point by $1/\max(\sigma_\Gamma, \text{floor})$ with
floor $= \max(10^{-6}, 0.1\cdot\operatorname{median}\sigma_\Gamma)$.

Fixed poles are deliberately conservative: they avoid an unstable pole
relocation stage. If a fit fails its threshold, change the model order, the
pole placement or the band; never drop the passivity constraint.

## Where the models are used

- **Path events** — given `surface_admittance_models`, `render_path_events`
  realizes every boundary hit of a reflection or scattering path with a
  `digital_locally_reacting_reflection_filter` (cached per surface model and
  incidence angle). Every surface on a filtered path needs a model, and the
  event's stored gain spectrum must agree with the model within tolerance.
- **Impedance modal backend** — `RectangularImpedanceBoundaryConfig`
  (schema `puresound.rectangular_impedance_boundary.v1`) assigns a passive
  model to each of the six shoebox walls, with `source.evidence_tier` and
  `applicability.scope` required. `ImpedanceModalLowFrequencyBackend` solves
  the resulting nonlinear eigenproblem
  ([modal validation](modal_validation.md)); select it with
  `generate_hybrid_rir.py --low-backend analytic-impedance
  --impedance-boundary-config <json>`.
- **FDTD reference** — `simulate_fdtd_reference(..., boundary_admittance=...)`
  keeps per-branch boundary state at every wall cell.

For a rectangular room $L_x \times L_y \times L_z$ the rigid reference
frequencies are

$$f_{n_x n_y n_z} = \frac{c}{2}\sqrt{\left(\frac{n_x}{L_x}\right)^2 + \left(\frac{n_y}{L_y}\right)^2 + \left(\frac{n_z}{L_z}\right)^2};$$

a complex boundary shifts them and adds decay. Compare modal and FDTD results
only at matching geometry, environment, wall orientation and phasor
convention. A good fit to one normal-incidence sample does not validate a
diffuse, angle-dependent room boundary.

## Tools

```bash
# raw two-microphone H12 sweeps -> complex_impedance_measurement.v1
python egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py --help
# fit a normalized measurement, check dense-grid passivity, 1D modes, optional repeat comparison
python egs/rir_generation/phases/m2_impedance/scripts/validate_impedance_measurement.py --help
```

The data template is
`egs/rir_generation/phases/m2_impedance/measurements/impedance_tube_template/`.
The public liner series under `.../measurements/zenodo_15195587/` exercise
normalized grazing-duct ingestion and resonant fitting; they are validation
data, not wall materials (see
[public source criteria](complex_impedance_source_audit.md)).

Tests: `test/rir/test_acoustic_impedance.py`,
`test_impedance_measurements.py`, `test_impedance_modes.py`,
`test_impedance_tube.py`.

## Promoting a fitted boundary

1. Verify source files, provenance and redistribution license.
2. Preserve measurement geometry and phasor convention.
3. Check passivity over and beyond the fit band.
4. Report held-out complex, magnitude and phase error.
5. Cross-check modal and FDTD behaviour.
6. Keep the thickness, backing, air gap, mounting and environmental limits.
7. Add the model to a material catalog only through an explicit, reviewed
   mapping.
