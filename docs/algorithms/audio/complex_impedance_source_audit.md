# Public complex-impedance sources: acceptance criteria

繁體中文版本：[complex_impedance_source_audit.zh-TW.md](complex_impedance_source_audit.zh-TW.md)

The complex-boundary path ([complex impedance](impedance_measurements.md))
admits only evidence that preserves reflection phase, installation conditions
and licensing. This page states the rule a public dataset must meet, lists
the public sources checked against it, and describes the one dataset
retained as validation data.

## Acceptance criteria

A dataset is admitted only if it:

1. directly provides complex surface impedance $Z(f)$, complex reflection
   $\Gamma(f)$, or a calibrated complex two-microphone transfer function from
   which they can be reconstructed;
2. states frequency, real part, imaginary part and phasor convention;
3. makes sample thickness, backing and air gap, measurement geometry and
   operating conditions traceable;
4. makes source URL, version and license traceable;
5. does not substitute diffuse-field absorption, or any data giving only
   $\alpha(f) = 1 - |\Gamma(f)|^2$, for a unique phase;
6. has a measurement configuration compatible with the actual installation
   before it is mapped to a room material.

Pixel-digitizing a published figure does not replace the original values,
and $\alpha$ cannot be used to guess a phase.

## Public sources checked

| Source | Status | Reason |
|---|---|---|
| [Zenodo 15195587](https://zenodo.org/records/15195587) | admitted, pipeline validation only | CC BY 4.0 HDF5 with per-frequency normalized resistance/reactance; the paper documents sample geometry, SPL, flow conditions, rigs and eduction method |
| [FOAM 02](https://zenodo.org/records/15407780) | not a complex source | absorption coefficient, diameter and class only; no phase or impedance |
| [FOAM 01](https://zenodo.org/records/10551344) | not a complex source | absorption coefficient and labels only; phase is not recoverable from $\alpha$ |
| [vyhyb/imptube](https://github.com/vyhyb/imptube) | method reference | MIT implementation of transfer-function reduction; no measured sample data |
| [MDPI Mathematics 10(18), 3264](https://www.mdpi.com/2227-7390/10/18/3264) | not a room-finish source | impedance plots for sintered metal-fibre samples without traceable per-frequency values |
| [Frontiers in Physics 14:1785611](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2026.1785611/full) | method reference | impedance-deduction method validated numerically only |

"Not a complex source" says nothing about data quality: FOAM 01/02 remain
usable for normal-incidence absorption distributions, classification or other
work that does not need phase.

No public dataset found so far meets the criteria for a common room finish or
porous absorber at normal incidence and indoor SPL with machine-readable raw
values. Such data is to be produced with the
[impedance-tube protocol](impedance_tube_protocol.md).

## The retained validation dataset

From Zenodo 15195587, Figure 6, the two no-flow, 130 dB
Kumaresan–Tufts eduction series are converted to the
[normalized contract](impedance_measurements.md#normalized-measurement-contract):

| File (under `egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/`) | HDF5 datasets |
|---|---|
| `nasa_gfit_noflow_130db_kt.csv` / `.json` | `/Figure6/Resistance-NASA-KT`, `/Figure6/Reactance-NASA-KT` |
| `ufsc_noflow_130db_kt.csv` / `.json` | `/Figure6/Resistance-UFSC-KT`, `/Figure6/Reactance-UFSC-KT` |

- Band: the paper's common comparison range, 500–2500 Hz.
- Values are copied as published, $z = Z/(\rho c)$, without interpolation.
- Source `paper_data.hdf5`, SHA-256
  `ba4cf7cf293d2b20ed590eb78ed8c133484771acbd011c680466d289abdc1a72`.
- Each sidecar records the HDF5 dataset paths, checksums, citation,
  license, sweep geometry, conditions and transformations, with
  `acoustic_field_geometry: grazing_duct`.

The values stay dimensionless because the HDF5 does not preserve the $\rho$
and $c$ used to normalize each record; choosing a standard atmosphere would
present a conversion assumption as the measurement environment.

**Scope.** This is a high-SPL, perforated, back-cavity aircraft liner whose
impedance was educed from a grazing-duct field. It validates phase-aware
ingestion, Helmholtz-resonance fitting, passivity and causal digitization, and
wiring one rational boundary into the FDTD solver and the eigenproblem. It
does not stand in for painted walls, carpet, curtains or ceilings, low-SPL
indoor speech, or any mismatched incidence, flow, perforation, cavity depth
or backing. Both sidecars therefore set `applicability.scope:
pipeline_validation_only` and `automatic_scene_catalog_mapping: false`.
