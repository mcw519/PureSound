# v19c diagnostics — as-run scripts and tables (2026-09-04)

Promoted verbatim from the planning session's scratchpad (R0.4 item 3 of `../v19c_round_design.md`). Scripts
carry the scratchpad paths they were run with (`/tmp/claude-1001/.../scratchpad/agcache`, `.../v19_plan`); the
per-row `.jsonl` outputs (tens of MB) were NOT copied — the tables/READMEs in each folder are the record.
Re-running requires rebuilding the per-frame cache with `../anchor_gate_cache.py`.

| folder | question | headline |
|---|---|---|
| `onset_profile/` | is onset deletion present on synthetic data? | real-only: Dawn −3.85/−3.90 dB @0.5 s vs synthetic −0.25..+0.30 (11–23×) |
| `anchor_synthetic/` | does the anchor behaviour reproduce on synthetic data? | yes, all three (1 s knee, 2 s saturation, 4–5 s half-life; other-talker anchor deletes next talker Δon1 −1.25/−1.71 dB) |
| `training_data_audit/` | what does the v16 training data contain? | target onset at t≈0 in 90.7 % of rows; interferer-first ≥1 s 6.0 %; re-entry after ≥5 s 0.3 %; no loss weights time |
| `chain_readability/` | does v11b fix the QVF readout inversion? | no (0.258 vs 0.265); features carry the cue (device-fit probe reads QVF 0.79–0.83); `eq_probe.py` compressor bug found |
| `objective_landscape/` | which mechanisms have evidence? | ranked landscape; persistence mechanisms come after "which talker" is fixed |
