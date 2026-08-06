# Public complex acoustic impedance source audit

繁體中文版本：`complex_impedance_source_audit.zh-TW.md`

This document records M2.5's item-by-item audit of public data sources. The
goal is not to collect as many absorption curves as possible — it is to admit
only direct complex-impedance evidence that preserves reflection phase,
installation conditions, and licensing.

## 1. Acceptance criteria

A dataset must satisfy at least:

1. it directly provides complex surface impedance \(Z(f)\), complex
   reflection coefficient \(\Gamma(f)\), or a calibrated complex
   two-microphone transfer function from which they can be reconstructed;
2. frequency, real part, imaginary part, and phasor convention are
   explicit;
3. sample thickness, backing/air gap, measurement geometry, and material
   operating conditions are traceable;
4. source URL, version, and license are traceable;
5. it does not treat diffuse-field absorption, or data that only gives
   \(\alpha(f)=1-|\Gamma(f)|^2\), as a substitute for a unique phase;
6. before any automatic mapping to the room material catalog, the
   measurement configuration must be compatible with the actual
   installation.

## 2. Review results

| Source | Verdict | Rationale |
|------|------|------|
| [Zenodo 15195587](https://zenodo.org/records/15195587) | Accepted, but for pipeline validation only | CC BY 4.0 HDF5 directly provides per-frequency normalized resistance/reactance; the paper describes sample geometry, 130/145 dB, no-flow and grazing-flow conditions, the NASA/UFSC test rigs, and the eduction method. |
| [FOAM 02](https://zenodo.org/records/15407780) | Rejected as a complex source | Apache-2.0 licensed with clear sample labels, but the public files contain only absorption coefficient, diameter, and classification — no reflection phase or complex impedance. |
| [FOAM 01](https://zenodo.org/records/10551344) | Rejected as a complex source | The public files contain only absorption coefficient and sample labels; phase cannot be uniquely recovered from \(\alpha\). |
| [vyhyb/imptube](https://github.com/vyhyb/imptube) | Method reference, not a measurement data source | The MIT-licensed implementation can compute reflection and surface impedance from a calibrated transfer function, but the repository has no already-measured sample data suitable for direct inclusion. |
| [MDPI Mathematics 10(18), 3264](https://www.mdpi.com/2227-7390/10/18/3264) | Rejected as a production room-finish source | The article includes normalised surface-impedance plots and measurement configurations for six sintered metal-fiber samples, but does not provide traceable per-frequency raw values; the material is also not a common indoor finish prioritized at this stage. |
| [Frontiers in Physics 14:1785611](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2026.1785611/full) | Method reference, not a physical measurement source | The study tests complex surface-impedance deduction, but its validation is a numerical experiment, and the paper itself lists physical validation as future work. |

"Rejected" only means the source does not satisfy this specific
complex-impedance path — it is not a statement about data quality. FOAM
01/02 remain usable for normal-incidence absorption distributions,
classification, or other work that does not require phase.

## 3. The first accepted dataset

M2.5 admits the two no-flow, 130 dB KT eduction series from Zenodo
15195587 Figure 6:

- NASA GFIT: `Resistance-NASA-KT`, `Reactance-NASA-KT`;
- UFSC: `Resistance-UFSC-KT`, `Reactance-UFSC-KT`;
- using the paper's shared comparison range of 500–2500 Hz;
- values are kept as-published, \(z=Z/(\rho c)\), with no interpolation;
- source `paper_data.hdf5`:
  SHA-256
  `ba4cf7cf293d2b20ed590eb78ed8c133484771acbd011c680466d289abdc1a72`.

The two converted files are at:

- `egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/nasa_gfit_noflow_130db_kt.csv`
- `egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/ufsc_noflow_130db_kt.csv`

Each has a JSON sidecar recording the HDF5 dataset path, checksum, paper
citation, license, actual sweep geometry, measurement conditions, and
transformation.

## 4. Why we added the normalized contract

The source publishes a dimensionless quantity:

\[
z(f)=\frac{Z(f)}{\rho c}=r(f)+j x(f)
\]

but the HDF5 does not preserve, per record, the exact \(\rho\) and \(c\)
used for that normalization at the time. Arbitrarily picking a standard
atmosphere and multiplying back to Pa·s/m would mislabel a "reference
conversion environment" as the "measurement environment." The repository
therefore keeps the original dimensionless values and defines a separate
schema,
`puresound.normalized_complex_impedance_measurement.v1`. Converting to SI
under a specific air environment is left to that scene's own \(\rho c\)
when needed.

## 5. Scope of applicability

This is a high-SPL, perforated, back-cavity aircraft liner, with surface
impedance inferred from a grazing duct's acoustic field. It is well suited
to validating:

- phase-aware ingestion;
- Helmholtz resonance fitting;
- passivity and causal digitization;
- wiring the same rational boundary into the FDTD solver/eigenproblem.

It is not suitable as a direct stand-in for:

- painted walls, carpet, curtains, ceilings;
- low-SPL indoor speech conditions;
- unmatched incidence, flow velocity, perforation size, cavity depth, or
  backing.

Both sidecars therefore set
`automatic_scene_catalog_mapping: false`. The first dataset that can be
directly mapped to general room materials should still be a normal-SPL,
normal-incidence measurement with a complete installation configuration,
covering porous absorbers or room finishes.

## 6. M2.6 second-round conclusions and disposition

As of 2026-07-30, a second round of public-source review still had not
found data that was directly includable and simultaneously satisfied:

- a common room-finish or porous absorber;
- per-frequency complex \(Z\), complex \(\Gamma\), or a calibrated complex
  \(H_{12}\) at normal incidence;
- thickness, backing, air gap, environment, and normal indoor SPL;
- a reusable license and machine-readable raw values.

Pixel-digitizing a paper's figure cannot stand in for an original
measurement, and FOAM 01/02's \(\alpha\) cannot be used to guess a phase.
This round therefore adds no newly "accepted room material"; instead it
fills in a reproducible two-microphone acquisition/reduction pipeline:

- the raw repeated \(H_{12}=P(x_2)/P(x_1)\) contract;
- microphone-swap complex calibration;
- circular-tube plane-wave cutoff and microphone-spacing conditioning;
- a coherence gate, cross-installation repeatability, and a passivity gate;
- weight propagation from \(Z\) uncertainty into the complex reflection
  fit;
- direct output of `puresound.complex_impedance_measurement.v1`.

The full algorithm, raw file formats, and first-batch experiment design are
in [`impedance_tube_protocol.md`](impedance_tube_protocol.md). ISO
10534-2:2023's public description confirms that the two-microphone complex
transfer technique obtains normal surface impedance, and explicitly states
that impedance-tube normal-incidence results and reverberation-room
random/diffuse-incidence results are not directly comparable:
[ISO 10534-2:2023](https://www.iso.org/standard/81294.html).
