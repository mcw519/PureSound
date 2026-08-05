# RIR generation phases

This folder keeps milestone-specific experiments separate from the seven stable
public commands in `egs/rir_generation/`. Each phase follows the same
convention:

The public commands are `generate_hybrid_rir.py`, `generate_m6_bank.py`,
`render_spatial_rir.py`, `plot_rir.py`, `inspect_bank.py`,
`compare_bank_acoustics.py`, and `compare_modal_acoustics.py`. Everything else
belongs here or under the supporting `tools/` and `examples/` directories.

- `scripts/`: validators, calibration runners, and exploratory commands for
  that milestone;
- `config/`: input configurations and templates;
- `reports/`: frozen JSON evidence emitted by a run;
- `fixtures/`: small non-production fixtures used by tests;
- `measurements/`: raw/reduced measurement assets (M2 only).

| Phase | Scope |
|---|---|
| `m0_baseline` | frozen baseline configuration |
| `m1_material` | material-first scene and priors |
| `m2_impedance` | complex impedance, modal residues, and tube measurements |
| `m3_wave_path` | coherent PathEvents and low-frequency wave-path validation |
| `m4_spatial_late_field` | spatial rendering and multiband late field |
| `m5_calibration` | measured-room inverse calibration and constrained residuals |
| `m6_bank` | deterministic bank, QC, release, evaluation, production evidence, and measured-reference target validity |

Reports are intentionally kept beside the phase that produced them. Paths in
reports and documentation should use this layout rather than the removed flat
configuration directory.
