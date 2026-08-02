# Two-microphone impedance-tube input template

Copy the three template files before recording a measurement:

- `raw_transfer.template.csv`: one row per repeat and frequency;
- `microphone_switch.template.csv`: normal and microphone-swapped calibration;
- `metadata.template.json`: environment, tube, acquisition, sample, provenance,
  and applicability.

The CSV files intentionally contain headers only. Do not treat this directory
as measured material data. The reduction command and Traditional Chinese
protocol are documented in
`docs/audio/impedance_tube_protocol_zh-TW.md`.

Keep `automatic_scene_catalog_mapping` false until independent-specimen,
passivity, fit, FDTD, and modal gates have passed.
