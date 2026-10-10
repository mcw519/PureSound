# RIR bank format — `puresound.audio.rir.bank`

繁體中文版本：[rir_bank_v2.zh-TW.md](rir_bank_v2.zh-TW.md)

This page is the authoritative description of the RIR bank format (the "M6"
bank; manifest schema `puresound.rir_bank.v2`): how a bank is laid out, split,
checked, released and promoted. How a training job reads a released bank is in
[bank loaders](rir_bank.md).

A directory of WAV files cannot guarantee that train and test rooms never mix,
that an item is the one that was generated, or that it passed any check. The
bank format makes each of those a hash-verified, fail-closed property of the
bank itself.

## Layout

```text
<bank>/
├── rir_bank_manifest.json         # the source of truth
├── rir_bank_generation_audit.json # generation run record
├── rir_bank_qc_summary.json       # after QC
├── indexes/{train,validation,test}.jsonl
├── qc/items/<item_id>.json        # one QC report per item
├── qc/candidate_indexes/{split}.jsonl
├── qc/quarantine/index.jsonl
└── room_000000/room_000000_000000.{wav,json}   # 32-bit float WAV + sidecar
```

## Manifest

`RIRBankManifest` holds:

| Field | Content |
|---|---|
| `bank_id`, `release_status` | Identity; status `draft`, `candidate` or `production` |
| `split_policy` | `BankSplitPolicy`: policy id, seed, split fractions |
| `generator` | `BankGeneratorProvenance`: generator id and version, `code_revision`, `config_sha256`, `task_plan_sha256`, seed |
| `renderer_profiles` | `BankRendererProfile`: low and high backend, scene schema, renderer config hash, `evidence_tier` (`development`, `empirical_candidate`, `production_approved`), optional calibration, residual and approval report hashes |
| `items` | `RIRBankItem` per item (below) |
| `split_indexes` | Path, SHA-256 and count of each split's JSONL index |
| `manifest_sha256` | SHA-256 of the canonical JSON of everything else |

Each `RIRBankItem` records `item_id`, `room_id`, `acoustic_space_id`,
`scene_id`, `split`, `generation_seed`, `origin` (`synthetic`, `real`, `mixed`),
`renderer_profile_id`, `signal_variant` (`physical`, `physical_residual`,
`measured`), `level_policy` (`calibrated`, `peak_normalized`,
`native_measured`), the WAV and JSON paths with their SHA-256, the SHA-256 of
the canonical scene, the audio header (rate, channels, frames) and its QC status
(`pending`, `pass`, `fail`) with the report path and hash.

Hashes use canonical JSON (sorted keys, no NaN) and
`canonicalize_float_wav_header` so that the same content always produces the
same bytes.

## Splits

Splits are assigned per **acoustic space**, not per item
(`puresound.m6_split.sha256_acoustic_space.v1`):

```text
u = int(SHA-256(policy_id ‖ seed ‖ acoustic_space_id)[:8]) / 2^64
split = train if u < f_train; validation if u < f_train + f_val; else test
```

Default fractions are 0.8 / 0.1 / 0.1. For a synthetic room the acoustic space
id hashes the room-level scene — dimensions, surfaces, materials, environment,
objects — and excludes source and receiver placement, so every item rendered in
one room lands in one split. The assignment depends only on the id, not on item
order or bank size. Generation refuses a task plan that leaves any split empty.

`audit_rir_bank_manifest` verifies the manifest hash, that every split is
non-empty, that each assignment matches the policy, that acoustic spaces and
room ids are split-disjoint, that every asset exists and matches its hash and
header, that sidecar identity and scene hash match, that the split indexes
match the items, that the task-plan hash matches, and that a `production`
status is backed by approved renderer profiles, passing QC and a pinned
(non-dirty) code revision.

## Generation

`generate_hybrid_rir.py --emit-m6-manifest` writes the manifest, split indexes
and generation audit after all items; `generate_m6_bank.py` wraps generation,
QC and release packaging in one command. Every content-producing option
(backends, scene version, rate, duration, seeds, ranges) enters the config hash,
so changing one produces a different bank. With the same arguments two fresh
runs produce the same `manifest_sha256`.

Resume re-uses an item only when its recorded task identity, WAV hash, scene
hash and audio header all match the current plan; a missing or modified item is
regenerated. A changed renderer, revision, duration or other generating input
cannot be mixed into an existing bank.

## Quality control

`run_rir_bank_qc` evaluates every item under `RIRBankQCPolicy`
(`puresound.rir_bank_qc.physical.v1`) and writes one report per item. Each check
has one of four states:

| State | Meaning |
|---|---|
| `pass` | Measurable and within its bound |
| `fail` | A hard invariant is violated; with severity `quarantine`, the item is quarantined |
| `not_evaluable` | The signal or metadata cannot support a reliable estimate |
| `not_applicable` | The check does not apply to this item |

Missing evidence is never turned into a pass or a fail.

**Structural checks:** assets exist and match their hashes; metadata readable;
identity and scene hash match; audio readable, of the declared shape, finite and
non-silent; the peak obeys the level policy (`peak_normalized` and
`native_measured` require `|peak| ≤ 1`, `calibrated` only a finite peak); the
channel map covers every channel exactly once; the sound speed is physical.

**Per-channel physical checks** (a failure quarantines the item):
`prearrival_energy` (energy before `floor(d / c · fs)` above
`maximum_prearrival_relative_peak` of the peak), `direct_arrival_timing`
(detected direct path within 1 ms of `d / c`), `insufficient_tail_energy`
(after 50 ms), `implausible_spectral_tilt` (beyond ±18 dB/octave),
`implausible_t20` (RT60 outside 0.02–20 s), decay fit R² ≥ 0.70 and decay
coverage, octave decay coverage, and late echo density ≥ 0.4.

**Informational:** the near/far DRR relationship, and spatial pair metrics,
which are `not_applicable` because the channels of a bank item are separate
source-to-microphone paths, not synchronized receivers.

QC writes `qc/candidate_indexes/` (passing items per split) and
`qc/quarantine/index.jsonl`, updates each item's status and report hash in the
manifest, and stores the policy with its content hash, so a threshold change
produces distinguishable evidence. Failed items never enter a release index.

## Release

`build_m6_variant_release(source_bank, output, measured_bank_root=None, ...)`
builds an immutable release (`rir_bank_release.json`, schema
`puresound.rir_bank_release.v1`) from a QC'd bank. It refuses to write into an
existing directory.

| Variant | Content |
|---|---|
| `synthetic_calibrated` | Copy of the QC'd bank (`variants/calibrated/`) |
| `synthetic_peak_normalized` | Same items, one common gain per item to peak 0.98 (`variants/peak_normalized/`) |
| `measured_native` | A QC-passed measured bank, when supplied (`variants/measured_native/`) |

| Recipe | Variants | Origin weights |
|---|---|---|
| `synthetic_calibrated` | calibrated | synthetic 1.0 |
| `synthetic_peak_normalized` | peak-normalized | synthetic 1.0 |
| `real_native` | measured | real 1.0 |
| `mixed_calibrated_real` | calibrated + measured | synthetic 0.5, real 0.5 by default |

Without a measured bank, `real_native` and `mixed_calibrated_real` are written
as `blocked` with the reason. Each ready recipe has per-split indexes under
`recipes/<recipe_id>/`, and each variant a `distribution.json` of its acoustic
statistics. Origin stays visible per item rather than being merged into an
unlabelled directory. `audit_m6_variant_release` re-derives every child variant
from its parent and caches its verdict beside the release
(`.m6_release_audit_cache.json`).

## Measured RIRs

`bank/measured_ingest.py` (`build_measured_m6_bank`) turns published measured
RIRs into a `measured` bank. Published corpora reference time to the direct
arrival, while the bank defines `t = 0` as the emission instant, so raw corpora
fail `prearrival_energy`. The ingest re-inserts the propagation delay without
modifying the response
(`puresound.measured_time_origin.iso3382_onset_to_geometric_arrival.v1`):

1. Find the ISO 3382-1 onset: the first sample above 20 dB below the peak that
   also clears the noise floor by 20 dB.
2. Shift the channel so the onset lands on `floor(d / c · fs)` for the
   published source–receiver distance, and mute the region before it.
3. Reject the channel instead of repairing it when a distinct earlier arrival
   precedes the onset (`earlier_arrival`), when muting would remove more than
   1 % of the energy (`removed_energy`), or when the shift is implausible
   (`implausible_shift`). One rejected channel rejects the item.

Corpora that publish no environment get an assumed 20 °C, 50 % RH, recorded in
each item's scene so QC recomputes the arrival from the same sound speed. The
aligned items then pass through the same QC as synthetic items. Use measured
data only where its license and provenance allow derived training use.

## Production promotion

A technically valid bank is a `candidate`. Promotion needs evidence that unit
tests cannot provide, collected by `evaluate_m6_release` and certified by
`build_m6_production_decision` (`rir_bank_production_decision.json`):

- controlled listening at the `empirical` tier with at least
  `MINIMUM_EMPIRICAL_PARTICIPANTS` (20) participants, hidden references and
  degraded anchors; a run without listeners is a `contract_fixture` and does
  not count;
- room-disjoint downstream training and evaluation;
- renderer profiles approved (`production_approved`) before QC, since the QC
  summary is bound to the manifest hash;
- all four recipes ready, all variant items passing QC, pinned revisions;
- acoustics, ML and release sign-offs and an auditable evidence bundle.

The certificate is validated against the immutable release; a loader with
`require_production=True` refuses a release without a valid, approved
certificate. Do not bypass missing evidence or relabel a development profile to
satisfy a production flag.

## Release checklist

1. Generate or resume with a pinned recipe and a clean code revision.
2. Run the manifest audit and QC.
3. Build the release from the QC'd bank (and the pruned measured bank, if any).
4. Audit licenses and provenance of measured sources.
5. Record empirical evidence separately from implementation tests.
6. Load every published split from the final release before training on it.

Commands for each step are in the
[RIR generation guide](../../../egs/rir_generation/README.md).
