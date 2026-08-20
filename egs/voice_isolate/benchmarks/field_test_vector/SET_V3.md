# Field benchmark v3 (2026-08-20) -- what changed and what it invalidates

Rebuilt by `build_cases.py` from the hand labelling in `spans/` (exported from
`tools/audio_annotator.html`). Replaces v2 and retires the separate
`qvf22_real_cases` set; both are archived under
`data_report/field_cases/_superseded/`.

## Inventory

| group | session | near | far | dt | flags |
|---|---|---|---|---|---|
| 0d | -- | 1 | -- | -- | **sentinel** |
| 90d | 1 | 6 | 6 | 2 | |
| 180d | 1 | 6 | 4 | 2 | **held_out** |
| 270d | 1 | 5 | 4 | 2 | |
| qvf_gym | 1 | 2 | 1 | 2 | |
| qvf_price | 1 | 2 | 2 | 2 | |
| qvf_scenario3 | 1 | 2 | 1 | -- | reference |
| qvf_scenario1 | -- | 1 | -- | -- | reference |
| qvf_scenario2 | -- | -- | 1 | -- | reference |
| qvf_keep_in_touch | -- | 1 | -- | 1 | |
| qvf_plumbing | -- | 1 | 1 | -- | |

64 entries (6 session, 58 cold-start) against v2's 27. Three carry QVF2.2's
published output as `reference`, so the cross-chain comparison that used to need
a second set now runs in the same pass -- `run_full_benchmark.sh` stage 1 is one
invocation instead of two.

Two conventions are new and are enforced, not just documented:

* **`held_out: true` (180d)** -- a third device orientation no readout has seen.
  `b_traj.py:training_rows` **raises** if asked to fit on it. This is the only
  defence against the failure the whole probe series exists to catch.
* **`sentinel: true` (0d)** -- a single-span keep clip that exists to catch a
  regression, not to grade one. It gets no session row: a one-span recording IS
  its span, and emitting both would score the same audio twice.

## v2 numbers do not carry over

The clip inventory and the labels both changed, so every field number on record
was measured on different material. Affected, and **not** to be compared against
v3 runs:

* `benchmarks/full_gate/v8_baseline.txt` / `v8_gate050.txt` -- stage 1 rows
* `benchmarks/probes/b_traj_README.md` -- all of it, including the operating
  curve; the readout was fitted on v2's 90D cut
* `benchmarks/probes/presence_readout_v8.npz` -- fitted material changed
* `benchmarks/probes/dist_head_field_*` -- 27-clip tables
* `benchmarks/field_test_vector/RESULTS.md`

Stages 2-9 of the full benchmark are untouched by this change.

## The sentinel already found something v2 could not

v8 at `dry_blend 0.9` on `0d_near1`: **keep -4.05 dB = KEEP-VIOLATION**, against
-0.05 to -0.81 dB on every 90D/270D near clip. The reason is in the reference
block:

| group | floor_dbfs | near_ref_dbfs | headroom |
|---|---|---|---|
| **0d** | **-53.43** | -39.20 | **14.23 dB** |
| 90d | -78.71 | -40.95 | 37.76 dB |
| 180d | -78.85 | -42.45 | 36.40 dB |
| 270d | -78.53 | -43.36 | 35.17 dB |

0D is a **25 dB noisier capture** with only 14 dB of SNR headroom, and v8 deletes
4 dB of the user there. v2 had no recording in that condition, so no v2 scorecard
could have shown it. This is a deployment-relevant keep failure on the shipped
default, found by widening the set rather than by any model change.
