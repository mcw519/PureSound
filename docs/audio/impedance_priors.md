# Phase-aware low-frequency impedance priors

`puresound.audio.impedance_priors` contains experimental references for
validating the M2 complex-boundary path. It does not automatically attach an
impedance to production scene materials.

## Evidence contract

Every prior records an evidence tier, source URLs, validity band, layer
configuration, and whether production material mapping is enabled. The initial
evidence tier is:

```text
measured_flow_resistivity_plus_miki_model
```

This means the airflow resistivity was measured, while complex surface
impedance is predicted by a published model. It must not be described as a
direct complex-impedance measurement.

The measurement method required for a stronger prior is a phase-aware normal
incidence method such as ISO 10534-2. The standard explicitly distinguishes
normal-incidence impedance-tube results from diffuse-incidence reverberation
room absorption:

- [ISO 10534-2:2023 overview](https://www.iso.org/standard/81294.html)

## Miki rigid-backed porous layer

The phase-aware reference follows Miki's positive-real modification of the
Delany-Bazley porous-material model:

- [Miki 1990, DOI 10.1250/ast.11.19](https://doi.org/10.1250/ast.11.19)

For `X = 1000 f / sigma` and the `exp(+j omega t)` convention:

```text
Zc = rho*c [1 + 5.50 X^-0.632 - j 8.43 X^-0.632]
k  = omega/c [1 + 7.81 X^-0.618 - j 11.41 X^-0.618]
Zs = -j Zc / tan(k d)
```

`sigma` is airflow resistivity in Pa·s/m² and `d` is layer thickness in metres.
The code conservatively enforces `0.01 <= f/sigma <= 1.0`; it does not silently
extrapolate outside the empirical model's original validity range.

## Initial references

Flow-resistivity parameters come from Tarnow's measurements of 100 mm glass
wool slabs:

- [Tarnow 2002, DOI 10.1121/1.1476686](https://doi.org/10.1121/1.1476686)

| Prior | Flow resistivity | Thickness | Model-valid range | Production mapping |
|-------|------------------|-----------|-------------------|--------------------|
| 14 kg/m³ glass wool, normal direction | 5.88 kPa·s/m² | 50 mm model-derived variant | 58.8–5880 Hz | disabled |
| 14 kg/m³ glass wool, normal direction | 5.88 kPa·s/m² | 100 mm source configuration | 58.8–5880 Hz | disabled |
| 30 kg/m³ glass wool, normal direction | 15.5 kPa·s/m² | 100 mm source configuration | 155–15500 Hz | disabled |

The flow resistivity was measured on the source specimens. The 50 mm layer is
an explicit Miki rigid-backed thickness experiment, not a direct installed
50 mm impedance measurement. None of these values is generic for every
carpet, ceiling tile, or fibrous object.

## Time-domain fitting

`fit_first_order_relaxation()` fits the passive causal FDTD boundary directly
in the complex pressure-reflection domain. It optimizes the zero-frequency
admittance, infinite-frequency admittance, and relaxation frequency while
keeping both endpoint admittances non-negative.

| Prior | Fit band | RMS complex error | Maximum complex error | Maximum phase error |
|-------|----------|-------------------|-----------------------|---------------------|
| 14 kg/m³, 50 mm | 60–300 Hz | 0.00313 | 0.00506 | 0.00413 rad |
| 14 kg/m³, 100 mm | 60–300 Hz | 0.0189 | 0.0571 | 0.0710 rad |
| 30 kg/m³ | 155–300 Hz | 0.00782 | 0.0160 | 0.00741 rad |

Both pass the diagnostic maximum-error threshold of 0.08. The negative
relaxation difference in these fits is valid: the low- and high-frequency
admittances remain non-negative, while the increasing admittance represents
the compliant phase of a rigid-backed porous layer.

## FDTD modal diagnostic

A 2.0 × 1.2 × 1.0 m reference room compares:

1. the 14 kg/m³ phase-aware fit;
2. a frequency-independent real boundary with the same reflection magnitude
   at 80 Hz but zero phase.

The dominant response peak moves from 85.7 Hz in the magnitude-only case to
64.0 Hz with the phase-aware boundary. Estimated Q changes from approximately
10.5 to 15.8. This is not a measured-room accuracy claim; it demonstrates that
reflection phase materially changes modal frequency and Q even when one
frequency's reflection magnitude is held fixed.

The high instantaneous admittance also exposed a corner/edge stability limit
not covered by the interior CFL condition. The FDTD solver now reduces its time
step until the summed normalized boundary Courant number is below 0.9 and
serializes both the interior and boundary time-step limits.

## Remaining gate

Before production use:

- obtain direct normal-incidence measurements for room-finish configurations;
- represent mounting, thickness, backing, air gaps, and uncertainty;
- pass those measurements through the strict CSV/JSON contract and passive
  relaxation/resonant fitting described in
  [`impedance_measurements.md`](impedance_measurements.md);
- map only compatible scene materials;
- extend the validated 1D complex eigenvalue connection to the production 3D
  modal problem;
- rerun room-disjoint measured-bank and downstream speech tests.
