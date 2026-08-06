# Complex acoustic impedance: real measurements, passive resonant fitting, FDTD, and modes

繁體中文版本：`impedance_measurements.zh-TW.md`

This document describes the M2.4–M2.5 vertical slice: how a phase-preserving
complex impedance measurement becomes a traceable, passive, causal boundary
model usable by both the time-domain FDTD solver and the complex modal
equations.

This path is currently research and validation infrastructure; it does not
automatically apply measurement results to the production material catalog.
An identical material name does not imply identical thickness, backing, air
gap, mounting, or incidence condition.

## 1. Why absorption alone is not enough

The normal-incidence pressure reflection coefficient and surface impedance
are related by:

\[
\Gamma(f)=\frac{Z(f)-Z_0}{Z(f)+Z_0},
\qquad
Z_0=\rho c
\]

\[
\alpha(f)=1-|\Gamma(f)|^2
\]

Absorption keeps only \(|\Gamma|\), not \(\angle\Gamma\). The same \(\alpha\)
can therefore correspond to many different complex impedances, and those
boundaries produce different reflection delays, modal frequencies, and Qs.
The ingestion contract accordingly requires \(\operatorname{Re}Z\) and
\(\operatorname{Im}Z\) directly; it does not accept guessing a phase from
diffuse-field absorption.

The SI contract supports normal-incidence, two-microphone
transfer-function-type complex impedance data. ISO 10534-2's scope is
exactly this: obtaining normal-incidence absorption and complex surface
impedance from such measurements. It is not directly interchangeable with a
reverberation room's diffuse-incidence absorption. M2.5 additionally adds a
normalized \(Z/(\rho c)\) contract that preserves the original dimensionless
representation and measurement geometry of public liner-eduction data,
rather than mislabeling a grazing duct as a normal-incidence tube.

- [ISO 10534-2:2023 overview](https://www.iso.org/standard/81294.html)

## 2. File contract

`ComplexImpedanceMeasurement.from_csv_and_metadata()` reads two files
together:

1. one per-frequency CSV;
2. one JSON sidecar recording environment, sample, and provenance.

### 2.1 CSV

Required columns and units:

```csv
frequency_hz,impedance_real_pa_s_m,impedance_imag_pa_s_m
60.0,421.2,-880.4
80.0,398.1,-631.7
120.0,376.5,-402.8
```

The optional uncertainty columns must appear as a pair:

```text
impedance_real_std_pa_s_m
impedance_imag_std_pa_s_m
```

The loader rejects:

- fewer than three frequency points;
- non-finite, non-positive, or non-strictly-increasing frequencies;
- NaN/Inf impedance;
- \(\operatorname{Re}Z<0\);
- only one of the two uncertainty columns;
- data whose implied \(|\Gamma|>1\).

### 2.2 JSON metadata

Minimal sidecar:

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

The schema currently enforces:

- an exact schema version;
- `measurement_id`, `method`, and `incidence: "normal"`;
- positive air density and sound speed;
- `sample.material_label`;
- `provenance.source_url` and `provenance.license`.

Production data should additionally record tube diameter, microphone
spacing, sample batch, thickness tolerance, backing, air gap, temperature and
humidity, calibration method, and repeatability statistics. These fields are
not yet all mandatory in the schema, but they must not be omitted when
mapping to materials.

### 2.3 Normalized complex impedance contract

Some literature publishes directly:

\[
z(f)=\frac{Z(f)}{\rho c}
\]

without preserving, per record, the exact \(\rho,c\) used for
normalization. This kind of data uses a separate schema:

```text
puresound.normalized_complex_impedance_measurement.v1
```

The CSV must contain:

```csv
frequency_hz,normalized_impedance_real,normalized_impedance_imag
500,0.785,-3.596
600,0.474,-3.213
```

The JSON must state explicitly:

- `acoustic_field_geometry`: `normal_incidence_tube` or `grazing_duct`;
- `phasor_convention`: currently fixed to `exp(+i*omega*t)`;
- mean-flow Mach number and source SPL;
- sample, provenance, license, and applicability limits.

The loader uses

\[
\mathcal C(z)=\frac{z-1}{z+1}
\]

as a bounded fitting coordinate and checks \(\operatorname{Re}z\ge0\) and
\(|\mathcal C(z)|\le1\). \(\mathcal C(z)\) equals the actual normal-incidence
pressure reflection only when a locally reacting, normal-incidence
interpretation holds; for grazing-duct data it is, first and foremost, a
numerically stable Cayley transform.

## 3. Passive rational admittance

Imported discrete-frequency data cannot be dropped directly into a
time-domain solver. M2.4 uses a rational normalized admittance with fixed
real poles:

\[
y(s)=g_s
+\sum_{k=1}^{K}\frac{g_{L,k}}{1+s\tau_k}
+\sum_{k=1}^{K}g_{H,k}\frac{s\tau_k}{1+s\tau_k}
\]

\[
\tau_k=\frac{1}{2\pi f_{p,k}},
\qquad
Z(s)=\frac{\rho c}{y(s)},
\qquad
\Gamma(s)=\frac{1-y(s)}{1+y(s)}
\]

where every coefficient is constrained to:

\[
g_s\ge0,\qquad g_{L,k}\ge0,\qquad g_{H,k}\ge0
\]

Every low-pass and high-pass branch is positive-real over the right half
\(s\)-plane; a non-negative parallel sum remains positive-real. The model is
therefore passive by construction, not verified after the fact by sampling
\(|\Gamma|\) at a few frequencies.

The low- and high-frequency endpoints are:

\[
y(0)=g_s+\sum_k g_{L,k},
\qquad
y(\infty)=g_s+\sum_k g_{H,k}
\]

`fit_passive_multi_pole_admittance()` fixes \(f_{p,k}\) and minimizes real
and imaginary error jointly in the complex pressure-reflection domain with
bounded least squares. The output records:

- RMS and maximum complex reflection error;
- maximum magnitude error;
- maximum phase error;
- fit band, pole strategy, passivity strategy;
- source measurement id and acceptance gate.

This is not full vector fitting. Fixed poles avoid pole relocation, unstable
pole flips, and post-fit passivity repair, at the cost of possibly needing
more poles, or being unable to reach the error threshold when pole placement
is unsuitable. If data fails the gate, the correct response is to adjust
model order, pole placement, or measurement range — not to disable the
passivity constraint.

### 3.1 Why a real liner needs a conjugate pole

The Zenodo liner's reactance crosses zero from negative to positive around
1.6 kHz — a Helmholtz/series-RLC resonance. A model with only real
relaxation poles cannot represent this sign flip; the first attempt's
held-out complex error was near 0.9, clearly unacceptable.

M2.5 therefore adds a passive series-RLC admittance branch. With
\(u=s/\omega_0\):

\[
y_k(s)=
\frac{g_{\mathrm{peak},k}}{Q_k}
\frac{u}{u^2+u/Q_k+1}
\]

The implementation equivalently multiplies the two factors above:

\[
y_k(s)=
\frac{g_{\mathrm{peak},k}}{Q_k}
\cdot
\frac{s/\omega_0}
{(s/\omega_0)^2+s/(Q_k\omega_0)+1}
\]

where \(f_0>0,Q>0,g_{\mathrm{peak}}\ge0\). This is exactly the admittance of
a passive series RLC circuit; its conductance at \(f_0\) is
\(g_{\mathrm{peak}}\), and its pole is necessarily in the left half-plane.
Added in parallel to a non-negative static conductance, it remains
positive-real.

`fit_passive_single_resonance_admittance()` fits static admittance,
\(f_0\), \(Q\), and peak admittance simultaneously. By default it trains on
even-indexed frequency bins and holds out odd-indexed bins, to avoid
reporting only training interpolation.

## 4. How multiple poles enter the FDTD

Every pole maintains its own bilinear-transform low-pass state
\(u_k[n]\):

\[
u_k[n]
=b_kp[n]+b_kp[n-1]-a_{1,k}u_k[n-1]
\]

\[
b_k=\frac{1}{1+2\tau_k/\Delta t},
\qquad
a_{1,k}=\frac{1-2\tau_k/\Delta t}
              {1+2\tau_k/\Delta t}
\]

Because

\[
\frac{s\tau_k}{1+s\tau_k}
=1-\frac{1}{1+s\tau_k},
\]

the high-pass branch needs no second filter state; it uses
\(p[n]-u_k[n]\) directly. The wall-normal velocity is:

\[
v_n[n]=\frac{
g_sp[n]
+\sum_k g_{L,k}u_k[n]
+\sum_k g_{H,k}\left(p[n]-u_k[n]\right)
}{\rho c}
\]

`simulate_fdtd_reference()` keeps independent state per wall, per wall
cell, and per pole. The old first-order relaxation is a special case of
this representation; a static real admittance still degenerates to the
original boundary update.

The RLC branch becomes a second-order biquad via the bilinear transform:

\[
w_k[n]=b_0p[n]+b_1p[n-1]+b_2p[n-2]
-a_1w_k[n-1]-a_2w_k[n-2]
\]

Each branch is prewarped at its own \(f_0\) so the analog and digital
resonance align. The solver's boundary time-step gate no longer looks only
at the DC/infinite-frequency endpoints; it uses a conservative
all-frequency admittance bound of `static + sum(g_peak)`, so a narrow-band
high-admittance resonance cannot slip past the stability check.

Beyond the 3D interior CFL condition, the explicit solver also uses
\(\max(y(0),y(\infty))\) to estimate the boundary time-step limit
accumulated at edges/corners. The full metadata serializes the interior
limit, the boundary limit, the actual sample rate, and each pole's digital
coefficients.

## 5. Minimal complex-mode connection point

Before wiring the boundary into the full 3D pytARD eigenproblem, M2.4 first
builds a 1D cavity whose two ends share the same locally reacting boundary.
The round-trip self-consistency condition is:

\[
F(s)=1-\Gamma(s)^2e^{-2sL/c}=0
\]

The resulting pole is defined as:

\[
s_n=-\gamma_n+j\omega_n
\]

so that:

\[
f_n=\frac{\omega_n}{2\pi},
\qquad
Q_n=\frac{\omega_n}{2\gamma_n}
\]

`solve_1d_impedance_cavity_modes()` solves
\(\operatorname{Re}F=\operatorname{Im}F=0\) directly in the complex
\(s\)-plane. The regression test for a static real reflection checks
against the closed-form solution:

\[
f_n=\frac{nc}{2L},
\qquad
\gamma_n=-\frac{c}{L}\ln|\Gamma|
\]

A second test fixes \(|\Gamma|\) at a reference frequency and compares a
phase-aware multi-pole boundary against a zero-phase real boundary,
confirming the two produce different complex modal frequencies.

This 1D solver has demonstrated that the same rational boundary can be
shared by the FDTD solver and the nonlinear eigenvalue equation, but it is
not a full room solver. 3D six-surface materials, oblique incidence,
non-rectangular geometry, mode coupling, and swapping into the production
renderer are still pending.

## 6. M2.5 real-data results

The first accepted dataset is
[Zenodo 15195587](https://zenodo.org/records/15195587)'s NASA GFIT
no-flow, 130 dB, KT eduction, with the UFSC nominally identical sample used
as an independent cross-rig comparison. The source HDF5 is CC BY 4.0; the
conversion preserves the original normalized resistance/reactance over
500–2500 Hz with no interpolation.

NASA's alternating-frequency fit obtains:

| Parameter/gate | Result |
|-------------|------|
| resonance \(f_0\) | 1646.66 Hz |
| branch \(Q\) | 11.69 |
| peak normalized admittance | 7.69 |
| training RMS / max complex error | 0.0437 / 0.0815 |
| held-out RMS / max complex error | 0.0385 / 0.0590 |
| dense 4096-point max \(|\mathcal C|\) | 0.9178 |
| NASA–UFSC cross-rig RMS difference | 0.1182 |

The held-out error is smaller than the difference between the two
NASA–UFSC nominally identical samples/rigs, and the 4096-point dense sweep
stays entirely positive-real with \(|\mathcal C|\le1\). In the 1D
diagnostic that sets the first axial mode at resonance, the phase-aware RLC
boundary gives \(Q=17.73\), while the real boundary matched to \(|\mathcal
C|\) at the reference frequency gives \(Q=6.04\) — a ratio of 2.94. This
again shows that matching absorption magnitude alone does not predict modal
decay.

Reproducible report:

```bash
python egs/rir_generation/phases/m2_impedance/scripts/validate_impedance_measurement.py \
  --csv egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/nasa_gfit_noflow_130db_kt.csv \
  --metadata egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/nasa_gfit_noflow_130db_kt.json \
  --comparison-csv egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/ufsc_noflow_130db_kt.csv \
  --comparison-metadata egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/ufsc_noflow_130db_kt.json \
  --output egs/rir_generation/exp/rir_realism/m2/rir_benchmark_m2/impedance_zenodo_15195587_validation.json
```

The full source acceptance/rejection rationale is in
[`complex_impedance_source_audit.md`](complex_impedance_source_audit.md).

This dataset is a high-SPL perforated aircraft liner's grazing-duct
eduction, used only to validate the phase-aware pipeline, the resonance
model, the FDTD solver, and the eigenproblem. Its sidecar explicitly sets
`automatic_scene_catalog_mapping: false`; it is not treated as a generic
wall, carpet, or ceiling.

## 7. M2.6 normal-incidence impedance tube acquisition

A second round of public-source review did not find a directly acceptable
room-finish complex dataset. This phase therefore adds:

- the `puresound.impedance_tube_transfer_measurement.v1` raw H12 contract;
- a repeated \(H_{12}=P(x_2)/P(x_1)\) long-form CSV;
- microphone-switch complex channel calibration;
- the circular tube's first transverse-mode cutoff and a
  \(|\sin(ks)|\) microphone-spacing conditioning gate;
- mean coherence, passivity, and cross-installation repeatability
  uncertainty;
- inverse-uncertainty fit weighting after propagating impedance
  uncertainty through
  \(\partial\Gamma/\partial Z=2Z_0/(Z+Z_0)^2\);
- a reduction CLI that reads directly back into
  `ComplexImpedanceMeasurement`.

The full derivation, CSV/JSON formats, and experiment checklist are in
[`impedance_tube_protocol.md`](impedance_tube_protocol.md). The synthetic
H12/complex-channel-mismatch/microphone-swap-calibration round trip
currently passes, but physical room-finish specimen measurements are still
pending.

## 8. M2.7 3D modes for six-surface rational impedance

Every face of a rectangular room can specify a
`FirstOrderRelaxationAdmittance`, `PassiveMultiPoleAdmittance`, or
`PassiveResonantAdmittance` in an explicit boundary JSON. The code does not
auto-generate this JSON from scene absorption.

With \(s=-\gamma+j\omega\), let each wall's normalized admittance be
\(y(s)\). The momentum equation gives a Robin boundary:

\[
\partial_n p+\frac{s}{c}y(s)p=0.
\]

Along one axis of length \(L\), let \(h_\pm=(s/c)y_\pm(s)\); the complex
wavenumber \(k\) must satisfy:

\[
(h_-h_+-k^2)\sin(kL)+k(h_-+h_+)\cos(kL)=0.
\]

All three axes must additionally share the same temporal pole:

\[
k_x^2+k_y^2+k_z^2+\left(\frac{s}{c}\right)^2=0.
\]

So it is not "compute the rigid frequency, then attach a Q" — the four
complex equations are solved simultaneously. This jointly changes the modal
frequency, decay, and complex spatial eigenfunction.

Validation results:

- when only the two x-axis faces have a static admittance, the 3D solution
  matches the exact 1D frequency/decay/Q;
- a uniform cube's three axial modes retain permutation degeneracy;
- for a 2.0 × 1.2 × 1.0 m phase-aware controlled case: the 3D solution
  gives 63.8846 Hz, \(Q\) 17.32; an independent FDTD gives 64.0049 Hz,
  \(Q\) 15.78;
- the generator's `analytic-impedance` backend passes a full CLI smoke
  test;
- the fixed-pole modal residue has been calibrated against an independent
  FDTD using two controlled rooms, four sets of training positions, and two
  position-holdout sets;
- position-holdout mean correlation 0.9734, NRMSE 0.2304, energy ratio
  1.0039, passing the pre-registered 0.90/0.35 gate;
- the calibration JSON is wired into the `analytic-impedance` backend, and
  the full generator CLI smoke test passes;
- M2.9 adds 50/100 mm rigid-backed thickness variants sharing the same
  measured flow resistivity, and separates position, unseen-room, and
  fine-grid holdouts;
- both thickness variants pass all three holdout types at correlation
  ≥0.90, NRMSE ≤0.35; the shared residue across both variants also passes;
- this only supports the tested thickness scope — the report still
  explicitly sets `general_boundary_invariance_established=false`.

Example boundary:

`egs/rir_generation/phases/m2_impedance/config/impedance_reference_glass_wool_14kgm3_100mm.json`

It is valid only over 60–300 Hz and is a measured-flow-resistivity +
Miki-model + one-pole fit — not a direct complex measurement, and not a
production recipe for a room with all six faces carpeted in glass wool.

## 9. Validation and scope of use

Automated tests currently cover:

- CSV/JSON round-trip, uncertainty, and provenance;
- rejection of active, diffuse-incidence, or provenance-less data;
- coefficient and complex-reflection recovery for a known two-pole
  synthetic measurement;
- provenance and passivity for two real normalized-impedance datasets;
- alternating-frequency train/held-out RLC fitting;
- passivity over a dense frequency sweep;
- FDTD relaxation, multi-pole, and RLC-biquad per-wall-cell state;
- 1D static analytic modes and phase-induced frequency shift;
- two-microphone H12 forward/inverse round-trip;
- microphone-switch complex mismatch correction;
- transverse-mode, spacing-conditioning, and coherence rejection;
- the repeated-H12 CLI down to the strict complex-impedance contract;
- the 3D static-to-1D limit, cube degeneracy, and the dynamic phase-aware
  root;
- the 3D complex mode against independent FDTD frequency/Q;
- explicit six-wall config down to hybrid-generator metadata;
- fixed-pole complex residue recovery and the position/room/grid split
  gate;
- the 50/100 mm controlled-thickness protocol and shared-parameter
  diagnostic.

Not yet done:

- a train/development/held-out material split;
- direct normal-incidence, normal-SPL measurements of common indoor
  materials;
- models for different mounting, air gap, and oblique incidence;
- residue FDTD calibration and real-room transfer-measurement calibration
  for different measured materials and air gap/mounting configurations;
- production mapping for an accepted room-finish boundary;
- room-disjoint measured-bank and fixed downstream-task validation.

## 10. Code entry points

The old flat modules under `puresound/audio/` were removed during the
package migration (see [`rir_package_migration.md`](rir_package_migration.md)).
The table below is the current post-migration path for every entry; each
row has been individually confirmed importable with
`.venv/bin/python -c "import ..."` and its corresponding test.

| File | Responsibility |
|------|------|
| `puresound/audio/rir/physics/impedance/measurements.py` | SI and normalized CSV/JSON measurement contracts (`ComplexImpedanceMeasurement`, `NormalizedComplexImpedanceMeasurement`) |
| `puresound/audio/rir/physics/impedance/tube.py` | H12, microphone-swap calibration, tube-diameter/spacing gates, coherence, and uncertainty reduction (`TwoMicrophoneTubeGeometry`, `reduce_two_microphone_repeats`) |
| `puresound/audio/rir/physics/impedance/admittance.py` | passive relaxation, multi-pole, and series-RLC admittance, plus the digital boundary reflection filter (`FirstOrderRelaxationAdmittance`, `PassiveMultiPoleAdmittance`, `PassiveResonantAdmittance`) |
| `puresound/audio/rir/physics/impedance/fitting.py` | fixed-real-pole multi-pole fitting and conjugate-RLC-pole fitting (`fit_passive_multi_pole_admittance()`, `fit_passive_single_resonance_admittance()`) |
| `puresound/audio/rir/physics/wave/fdtd.py` | the per-wall-cell first-order/biquad boundary-state 3D FDTD reference solver (`FDTDReferenceConfig`, `simulate_fdtd_reference()`) |
| `puresound/audio/rir/physics/impedance/modes.py` | 1D cavity and separable 3D six-surface rational-impedance eigenproblem (`solve_1d_impedance_cavity_modes()`, `solve_rectangular_impedance_modes()`) |
| `puresound/audio/rir/physics/impedance/residues.py` | fixed-pole complex residue fitting, position holdout, and versioned calibration (`fit_fixed_pole_modal_residues()`, `ImpedanceModalResidueCalibration`) |
| `puresound/audio/rir/render/low_frequency/impedance_modal.py` | the experimental `ImpedanceModalLowFrequencyBackend`: the separable 3D rational-impedance modal renderer, split out of `hybrid_rir` into its own file in R2 (`RIR_EXP_LOG.md`) |
| `puresound/audio/rir/render/hybrid.py` | `generate_hybrid_rir()`: the orchestration entry point that assembles the low-/high-frequency backends, the causality clip, and the crossover into one hybrid RIR; the direct successor to the old `hybrid_rir.py` |
| `egs/rir_generation/phases/m2_impedance/scripts/validate_impedance_measurement.py` | held-out, passivity, cross-rig, and modal report |
| `egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py` | raw repeated H12 to strict complex impedance |
| `egs/rir_generation/phases/m2_impedance/scripts/calibrate_impedance_modal_residues.py` | multi-room FDTD residue calibration and holdout report |
| `egs/rir_generation/phases/m2_impedance/scripts/calibrate_impedance_residue_protocol.py` | the M2.9 multi-boundary, unseen-room, grid-holdout protocol |
| `test/test_acoustic_impedance.py` | admittance-model passivity and digital-boundary-filter tests |
| `test/test_impedance_measurements.py` | ingestion, passivity, and fitting tests |
| `test/test_impedance_tube.py` | H12 round-trip, calibration, gate, and CLI tests |
| `test/test_impedance_modes.py` | analytic-mode and phase-shift tests |
| `test/test_impedance_priors.py` | Miki prior fitting and FDTD modal-diagnostic tests |
| `test/test_fdtd_reference.py` | FDTD boundary-state and 1D plane-wave-reflection tests |
| `test/test_impedance_residues.py` | fixed-pole residue fitting and split-holdout gate tests |
| `test/test_impedance_residue_protocol.py` | M2.9 residue-protocol CLI tests |
| `test/test_impedance_validation_cli.py` | impedance-validation CLI smoke test |
