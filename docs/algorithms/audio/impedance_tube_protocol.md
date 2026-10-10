# Normal-incidence complex impedance measurement and ingestion

繁體中文版本：[impedance_tube_protocol.zh-TW.md](impedance_tube_protocol.zh-TW.md)

`puresound.audio.rir.physics.impedance.tube` reduces repeated two-microphone
impedance-tube sweeps to the complex surface impedance consumed by the
[measurement contract](impedance_measurements.md). This page defines the
convention, the reduction, its quality gates, the raw file formats and the
measurement protocol.

## Why measure $H_{12}$ rather than absorption

A low-frequency room boundary changes reflection phase as well as removing
energy. $\alpha(f) = 1 - |\Gamma(f)|^2$ gives the magnitude only, so it
determines neither the surface impedance, nor the modal frequency shift, nor
a causal time-domain boundary. Two fixed microphones in a tube measure the
complex transfer function

$$H_{12}(f) = \frac{P(x_2, f)}{P(x_1, f)},$$

which keeps amplitude and phase. ISO 10534-2 covers exactly normal-incidence
absorption and surface impedance; its result is not the diffuse-incidence
absorption of a reverberation room.

## Convention and reflection coefficient

- The sample surface is at $x = 0$; $+x$ points from the sample toward the
  source; $x_1, x_2 > 0$ are the microphone distances from the sample.
- $H_{12} = P(x_2)/P(x_1)$; phasor convention $e^{+i\omega t}$.

With plane waves only, $p(x) = A e^{ikx} + B e^{-ikx}$ with $k = 2\pi f/c$;
$A$ travels toward the sample, $B$ is reflected, and $\Gamma = B/A$.
Eliminating $A$ and $B$ (`reflection_from_two_microphone_transfer`):

$$\Gamma = \frac{e^{ikx_2} - H_{12}e^{ikx_1}}{H_{12}e^{-ikx_1} - e^{-ikx_2}},\qquad Z_s = Z_0\,\frac{1+\Gamma}{1-\Gamma},\quad Z_0 = \rho c.$$

$Z_s(f)$ is the common input to passive rational fitting, the FDTD boundary
and the complex modal eigenproblem. `transfer_from_surface_reflection` is the
forward model used for calibration and round-trip tests.

## Microphone-swap calibration

The two channels never have identical complex sensitivity. With mismatch
$C(f)$, measuring one calibration field in the normal and swapped positions
gives $H_\mathrm{I} = CH$ and $H_\mathrm{II} = C/H$, hence

$$C(f) = \sqrt{H_\mathrm{I}(f)\,H_\mathrm{II}(f)},\qquad H_{12,\text{corrected}} = H_{12,\text{raw}}/C.$$

`microphone_switch_calibration_factor` unwraps the phase of
$H_\mathrm{I}H_\mathrm{II}$ before the square root so the branch stays
continuous across frequency, and picks the sign with positive median real
part. The reduction script requires this calibration unless the metadata sets
`acquisition.microphone_switch_calibration_required: false`; two microphones
of the same model are not assumed matched.

## Valid-band gates

`TwoMicrophoneTubeGeometry.validate_frequencies` rejects a band that violates
either condition:

- **Plane-wave limit.** The first transverse mode of a rigid circular tube of
  diameter $D$ cuts on at $f = 1.841c/(\pi D)$; every selected frequency must
  lie below it.
- **Spacing conditioning.** With spacing $s = |x_1 - x_2|$, small
  $|\sin(ks)|$ means the two microphones barely separate incident from
  reflected waves and the inversion amplifies error. The whole band must meet
  $|\sin(ks)| \ge$ `minimum_spacing_sine` (default 0.05). This is a numerical
  guard, not a substitute for a laboratory's standard band-selection
  procedure.

## Reduction, uncertainty and passivity

`reduce_two_microphone_repeats(frequencies, h12_repeats, coherence_repeats,
geometry, *, air_density_kg_m3, microphone_correction=None,
minimum_coherence=0.95, passivity_tolerance=1e-6)`:

1. requires every repeat on the same frequency grid;
2. requires the mean magnitude-squared coherence at every frequency to reach
   `minimum_coherence`;
3. divides by the microphone correction, then computes $\Gamma$ and $Z_s$ per
   repeat;
4. averages $Z_s$ across repeats and reports the sample standard deviation of
   its real and imaginary parts (zero with a single repeat);
5. rejects a mean $\operatorname{Re} Z_s$ below $-$`passivity_tolerance` or a
   resulting $|\Gamma| > 1 +$ `passivity_tolerance`; a negative real part
   within the tolerance is clamped to zero.

Downstream, `fit_complex_impedance_measurement` propagates the standard
deviation to the reflection domain with
$\partial\Gamma/\partial Z = 2Z_0/(Z + Z_0)^2$ and uses it as an inverse
weight. Repeats must be genuine remove–reinstall–remeasure cycles: replaying
the same installation cannot capture sealing, compression or edge-leakage
variation. A single sweep still yields an impedance but no repeatability
uncertainty (the output then has no uncertainty columns), and it should not
pass a production material gate.

## Raw data contract

Long-form transfer CSV (`load_transfer_repeats_csv`):

```csv
repeat_id,frequency_hz,h12_real,h12_imag,coherence
install_01,200,0.91,-0.13,0.997
install_01,250,0.88,-0.17,0.998
install_02,200,0.90,-0.14,0.996
install_02,250,0.87,-0.18,0.997
```

Microphone-swap CSV (`load_microphone_switch_csv`, at least three rows, on the
same grid as the transfer CSV):

```csv
frequency_hz,h12_original_real,h12_original_imag,h12_swapped_real,h12_swapped_imag
200,1.02,0.03,1.01,0.02
250,1.02,0.03,1.01,0.02
```

JSON sidecar, schema `puresound.impedance_tube_transfer_measurement.v1`,
checked by the reduction script:

| Section | Required fields |
|---|---|
| top level | `schema_version`, `measurement_id`, `method`, `phasor_convention: "exp(+i*omega*t)"` |
| `environment` | `air_density_kg_m3`, `sound_speed_m_s`, `temperature_c`, `relative_humidity_percent` (0–100), `pressure_pa` |
| `tube` | `shape: "circular"`, `diameter_m`, `microphone_1_distance_from_sample_m`, `microphone_2_distance_from_sample_m` |
| `acquisition` | `analysis_frequency_range_hz` `[min, max]`, `minimum_coherence`, `source_spl_db`; optional `minimum_spacing_sine` (0.05), `microphone_switch_calibration_required` (true), `passivity_tolerance` (1e-6) |
| `sample` | `sample_id`, `material_label`, `mounting`, `backing`, `thickness_m`, `air_gap_m` |
| `provenance` | `source_url` (or lab-notebook location), `license` |
| `applicability` | optional; defaults to `scope: "measurement_validation_pending"` with `automatic_scene_catalog_mapping: false`, which stays false until the measurement is accepted |

A reusable template (`metadata.template.json`, `raw_transfer.template.csv`,
`microphone_switch.template.csv`) is in
`egs/rir_generation/phases/m2_impedance/measurements/impedance_tube_template/`.

## Running the reduction

```bash
python egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py \
  --transfer-csv path/to/raw_h12.csv \
  --metadata path/to/raw_h12.json \
  --microphone-switch-csv path/to/microphone_switch.csv \
  --output-csv path/to/complex_impedance.csv \
  --output-metadata path/to/complex_impedance.json
```

`--minimum-frequency` / `--maximum-frequency` override the configured
analysis band. The output is a `puresound.complex_impedance_measurement.v1`
pair that `ComplexImpedanceMeasurement.from_csv_and_metadata` loads and
`fit_complex_impedance_measurement()` fits directly. The sidecar keeps the
SHA-256 of the raw and calibration files, the tube geometry and band gates,
repeat ids, coherence, coordinate and transfer definitions, and the
transformation list.

## Measurement protocol

Start with one common porous absorber that has a clear mounting and can be
cut into repeatable specimens (for example 50 or 100 mm glass or rock wool,
rigid backing, no air gap):

1. at least three independent specimens;
2. at least three reinstallations per specimen;
3. two source levels (for example 75 and 85 dB) to check level dependence;
4. record thickness tolerance, density or areal density, batch, cut diameter
   and perimeter sealing;
5. choose the band from the intersection of the spacing and tube-cutoff
   constraints;
6. after fitting, validate on a held-out specimen, not only held-out
   frequencies;
7. map an installed configuration to the scene catalog only after the
   reflection magnitude/phase, passivity, FDTD and 1D-mode checks pass for
   exactly that configuration.

Always retain the raw $H_{12}$ spectra: when tube-loss or geometric
corrections are added, the reduction must be rerun from raw data, and a final
absorption curve is not enough.

## Limits of the reduction

- Propagation in the tube uses the real wavenumber $k = 2\pi f/c$; the
  thermoviscous loss correction for narrow tubes is not applied.
- Uncertainty comes only from specimen and reinstallation repeats; errors in
  microphone position, air parameters and the calibration spectrum are not
  propagated.
- High coherence shows a stable linear spectral estimate, not the absence of
  edge leakage, specimen compression, lateral constraint or sample-to-sample
  variation.
- Circular tubes and normal plane-wave incidence only.
- A normal-incidence, locally reacting impedance says nothing about oblique
  incidence, finite patches or non-local reaction in a room model.

Tests: `test/rir/test_impedance_tube.py`.
