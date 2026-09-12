# Normal-incidence complex impedance measurement and ingestion pipeline

繁體中文版本：`impedance_tube_protocol.zh-TW.md`

Status: the M2.6 measurement reduction is complete; the first batch of
physical room-finish specimen data is still pending.

## 1. Why we measure H12 instead of just copying absorption

A room's low-frequency boundary does not only remove energy — it also
changes reflection phase. Knowing only

\[
\alpha(f)=1-|\Gamma(f)|^2
\]

tells us the reflection coefficient's magnitude, not its phase, so we
cannot uniquely determine surface impedance, the modal frequency shift, or
a causal time-domain boundary.

Two fixed microphones in an impedance tube measure the complex transfer
function

\[
H_{12}(f)=\frac{P(x_2,f)}{P(x_1,f)}
\]

which preserves both amplitude and phase. ISO 10534-2's formal scope is
exactly normal-incidence absorption and normal surface impedance; it is not
the same quantity as a reverberation room's diffuse-incidence absorption.

## 2. Coordinates, wave decomposition, and the reflection coefficient

The code adopts the following explicit convention:

- the sample surface is at \(x=0\);
- the positive \(x\) direction points from the sample towards the source;
- \(x_1,x_2>0\) are the two microphones' distances from the sample;
- the raw transfer function is fixed as \(H_{12}=P(x_2)/P(x_1)\);
- the phasor convention is fixed as \(\exp(+i\omega t)\).

With only plane waves in the tube,

\[
p(x)=A e^{ikx}+B e^{-ikx},
\qquad
k=\frac{2\pi f}{c}.
\]

\(A\) is the wave travelling towards the sample and \(B\) is the reflected
wave; the sample surface's complex pressure reflection coefficient is
\(\Gamma=B/A\). Eliminating \(A,B\) from the two microphone pressures gives:

\[
\Gamma=
\frac{e^{ikx_2}-H_{12}e^{ikx_1}}
{H_{12}e^{-ikx_1}-e^{-ikx_2}}.
\]

Then, from the air's characteristic impedance \(Z_0=\rho c\):

\[
Z_s=Z_0\frac{1+\Gamma}{1-\Gamma}.
\]

This \(Z_s(f)\) is the common input to the downstream passive rational fit,
the FDTD boundary, and the complex modal eigenproblem.

## 3. Microphone-swap calibration

The two microphones/measurement channels will not have exactly identical
complex sensitivity. Let their mismatch be \(C(f)\); the same calibration
sound field, measured with the microphones in their normal positions and
then swapped, gives:

\[
H_\mathrm{I}=C H,\qquad H_\mathrm{II}=C/H.
\]

so:

\[
C(f)=\sqrt{H_\mathrm{I}(f)H_\mathrm{II}(f)},
\qquad
H_{12,\mathrm{corrected}}=H_{12,\mathrm{raw}}/C.
\]

The implementation first unwraps the phase of \(H_\mathrm{I}H_\mathrm{II}\),
then takes a continuous complex square root, avoiding a sudden branch flip
between frequencies. Production metadata requires this swap calibration by
default; two microphones of the same model must not be treated as already
calibrated.

## 4. Valid-band gates

Not every FFT bin is usable. The code checks two conditions together.

### 4.1 The circular tube's first transverse mode

For a rigid circular tube of diameter \(D\), the first non-planar mode's
cutoff is approximately:

\[
f_\mathrm{plane,max}
=\frac{1.841c}{\pi D}.
\]

The selected band must lie entirely below this frequency.

### 4.2 Microphone-spacing ill-conditioning

With microphone spacing \(s=|x_1-x_2|\), when

\[
|\sin(ks)|
\]

is too small, the two points can barely distinguish the incident wave from
the reflected wave, and inversion amplifies measurement error. The default
requirement is \(|\sin(ks)|\ge0.05\) across the entire selected band. This is
a numerical-conditioning gate, not a substitute for a laboratory's
standard-driven band-selection procedure.

## 5. Quality, uncertainty, and passivity

Every specimen should be measured multiple times with a genuine "remove,
reinstall, remeasure" cycle; replaying the same signal without reinstalling
cannot capture the uncertainty from sealing, compression, or edge leakage.

The current reduction:

1. requires every repeat to use the same frequency grid;
2. requires, by default, that the mean magnitude-squared coherence at every
   frequency be no lower than 0.95;
3. computes \(\Gamma\) and \(Z_s\) separately for each sweep;
4. outputs the cross-repeat sample standard deviation of \(Z_s\)'s real and
   imaginary parts;
5. requires the mean \(\operatorname{Re}Z_s\ge0\) and \(|\Gamma|\le1\); data
   outside tolerance is rejected outright, not silently clipped;
6. propagates the impedance standard deviation into the reflection domain
   via
   \[
   \frac{\partial\Gamma}{\partial Z}
   =\frac{2Z_0}{(Z+Z_0)^2}
   \]
   for the rational fit's inverse-uncertainty weighting.

With only a single sweep, an impedance can still be produced, but it must
not claim repeatability uncertainty, and it should not pass the production
material gate.

## 6. Raw data contract

Raw long-form CSV:

```csv
repeat_id,frequency_hz,h12_real,h12_imag,coherence
install_01,200,0.91,-0.13,0.997
install_01,250,0.88,-0.17,0.998
install_02,200,0.90,-0.14,0.996
install_02,250,0.87,-0.18,0.997
```

Microphone-swap calibration CSV:

```csv
frequency_hz,h12_original_real,h12_original_imag,h12_swapped_real,h12_swapped_imag
200,1.02,0.03,1.01,0.02
250,1.02,0.03,1.01,0.02
```

The JSON sidecar's schema is:

```text
puresound.impedance_tube_transfer_measurement.v1
```

Required contents include:

- air density, sound speed, temperature, relative humidity, and barometric
  pressure;
- tube diameter and the two microphones' distances from the sample
  surface;
- the analysis band, minimum coherence, source SPL, and calibration
  requirements;
- sample id, material label, thickness, mounting, backing, and air gap;
- source URL/lab-notebook location and license;
- `automatic_scene_catalog_mapping`, which must be `false` until acceptance
  is complete.

A reusable template lives at
`egs/rir_generation/phases/m2_impedance/measurements/impedance_tube_template/`.

## 7. Running the reduction

```bash
.venv/bin/python egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py \
  --transfer-csv path/to/raw_h12.csv \
  --metadata path/to/raw_h12.json \
  --microphone-switch-csv path/to/microphone_switch.csv \
  --output-csv path/to/complex_impedance.csv \
  --output-metadata path/to/complex_impedance.json
```

The output conforms to the existing
`puresound.complex_impedance_measurement.v1` and can be passed directly to
`fit_complex_impedance_measurement()`. The raw file's SHA-256, the
calibration file's SHA-256, the band gate, repeat ids, coherence,
coordinates, and the transformation are all preserved in the sidecar.

## 8. Recommendations for the first batch of experiments

Do not measure a dozen unknown materials at once in the first batch.
Start with a common porous absorber that has a clear mounting and can be
cut into repeatable samples — for example 50 mm or 100 mm glass wool or
rock wool, rigid backing, no air gap:

1. at least three independent specimens;
2. at least three reinstallations per specimen;
3. measure at two linear SPLs, 75 and 85 dB, to check level dependence;
4. record thickness tolerance, density/areal density, batch, cut diameter,
   and perimeter sealing;
5. first select the band from the intersection of microphone spacing and
   tube-cutoff constraints;
6. after fitting, hold out a specimen — not just a frequency — for
   validation;
7. only after passing the reflection magnitude/phase, passivity, FDTD, and
   1D-mode gates may the exact same installed configuration be mapped to
   the scene catalog.

This batch of physical measurements does not exist yet, so what M2.6
completes is "a trustworthy algorithm and interface for acquiring the next
dataset," not a claim that ground truth for common room materials has
already been obtained.

## 9. Known limits of the current reduction

- in-tube propagation currently uses the real wavenumber \(k=2\pi f/c\); it
  does not yet include the thermoviscous propagation-loss correction that
  applies inside narrow tubes;
- uncertainty currently comes only from specimen/reinstallation repeats; it
  does not yet fully Monte Carlo-propagate errors in microphone position,
  air parameters, and the calibration spectrum;
- high coherence only means the linear spectral estimate is stable — it
  does not mean there is no edge leakage, specimen compression, lateral
  constraint, or sample-to-sample variability;
- only circular tubes and normal plane incidence are currently supported;
- once a normal-incidence, locally reacting impedance enters a room model,
  oblique incidence, finite patches, and non-local reaction still need
  separate treatment.

The first real measurement batch must therefore also retain the raw
spectra; when tube-loss or geometric corrections are added later, the
reduction must be rerun from the raw H12 — the final absorption curve alone
is not sufficient to keep.
