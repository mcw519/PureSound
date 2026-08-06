# Zenodo 15195587 normalized liner impedance

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

This directory contains a 500–2500 Hz subset of the normalized complex
impedance data published with:

> Nicolas T. Quintino, Lucas A. Bonomo, Julio A. Cordioli, Michael G. Jones,
> Brian M. Howerton, Douglas M. Nark, and Francesco Avallone,
> “Comparison of Impedance Eduction Test Rigs with Different Boundary-Layer
> Profiles,” *AIAA Journal* 63(11), 2025.

- Dataset DOI: <https://doi.org/10.5281/zenodo.15195587>
- Article DOI: <https://doi.org/10.2514/1.J065173>
- Dataset license: CC BY 4.0
- Source file: `paper_data.hdf5`
- Source MD5: `ba222c796afe37e4afe911f33df6a7c4`
- Source SHA-256:
  `ba4cf7cf293d2b20ed590eb78ed8c133484771acbd011c680466d289abdc1a72`

Changes from the source:

- copied Figure 6 frequency, KT resistance, and KT reactance arrays;
- retained the paper's common 500–2500 Hz comparison band;
- split NASA and UFSC into strict CSV/JSON measurement pairs;
- preserved values as dimensionless \(Z/(\rho c)\);
- added machine-readable sample geometry, provenance, and applicability;
- performed no interpolation, smoothing, or SI atmosphere conversion.

These high-SPL grazing-duct liner data are included for pipeline validation.
They are not automatically mapped to the room-material catalog.
