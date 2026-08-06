# M6 Production RIR Bank v2 Contract

繁體中文版本：`rir_bank_v2.zh-TW.md`

Status: M6.1–M6.6 implementation is complete. As of the 2026-08-04 evidence
pilot (`RIR_EXP_LOG.md` §6, §6.4), 10 of the 13 production-decision checks
pass — including `all_renderer_profiles_are_production_approved` and the
`real_native`/`mixed_calibrated_real` recipe checks, now that generation can
be given a measured bank (§5, §8 below). M6.6 production promotion remains
**BLOCKED** on the three checks that only genuine external evidence can
satisfy: ≥20-participant controlled listening, room-disjoint downstream
training, and the evidence-bundle audit that depends on both — see §12 and
`RIR_EXP_LOG.md` §6 for the exact check list.
Last updated: 2026-08-05.

This document explains how PureSound packages a validated RIR renderer into a
training-data bank that is reproducible, auditable, and free of room leakage.
What M6.1 solves is "what must a bank remember to be eligible for comparison
or release" — it is not a claim that the current renderer has already cleared
production evidence.

## 1. Why WAVs alone are not enough

The old `PreGeneratedRoomBank` used same-stem WAV/JSON pairs, which is fine
for fast loading, but a directory alone cannot answer:

- Does this item belong to train, validation, or test?
- Has the same physical room, or the same synthetic parent room, leaked into
  another split?
- Which scene schema, renderer backend, config, and code revision produced
  it?
- Have the WAV, metadata, or scene been modified after generation?
- Has a bank built from a development renderer been mislabeled as
  production?

M6.1 keeps the existing WAV/JSON layout and adds a `rir_bank_manifest.json` at
the bank root. Old readers can still use the files directly; release and
evaluation tooling treats the manifest as the single provenance contract.

## 2. Three layers of identity

Every item carries all three:

- `item_id`: the unique name of one WAV/JSON pair;
- `room_id`: one room geometry/material realization;
- `acoustic_space_id`: the atomic unit for splitting.

`acoustic_space_id` matters more than `room_id`. For real measurements it
should denote the corpus namespace plus the physical room; for a synthetic
bank it denotes the parent room realization. Every position, source/receiver
pose, normalized variant, or residual variant sharing an `acoustic_space_id`
must stay in the same split.

The audit therefore checks both:

\[
|\{s_i:a_i=a\}|=1,\qquad |\{s_i:r_i=r\}|=1,
\]

where \(a_i\) is the acoustic-space identity, \(r_i\) is the room identity,
and \(s_i\) is the split.

## 3. Deterministic room-disjoint split

M6.1 freezes the policy:

```text
puresound.m6_split.sha256_acoustic_space.v1
```

SHA-256 over the seed, policy id, and `acoustic_space_id`; the top 64 bits map
to \([0,1)\) and the manifest's train/validation/test fractions decide the
split. The same identity yields the same result regardless of machine, worker
count, or file-enumeration order.

This is not a per-item random split. If one room has 1,000 positions, all
1,000 items still land in the same split together.

## 4. Content-addressed provenance

Every manifest records two layers of hash:

1. the SHA-256 of every RIR WAV, metadata JSON, and canonical scene JSON;
2. the canonical-JSON SHA-256 of the full manifest, sorted-key and NaN-free,
   excluding `manifest_sha256` itself.

So the following are all detected:

- a replaced or truncated WAV;
- modified metadata or scene;
- a changed split, renderer profile, item id, or asset hash without
  re-signing the manifest.

The manifest also stores each WAV's sample rate, channel count, and frame
count. The audit cross-checks these against the actual audio header — it
never trusts the file extension or a metadata claim alone.

## 5. Generator and renderer provenance

Bank-level generator provenance carries:

- generator id and version;
- code revision;
- the canonical generation-config SHA-256;
- the generation seed.

A renderer profile carries:

- renderer id/version;
- low/high backend;
- scene schema version;
- the renderer-config SHA-256;
- calibration/residual/approval evidence hashes;
- an evidence tier.

The evidence tier is one of only:

```text
development
empirical_candidate
production_approved
```

The M4/M5 implementation validators already PASS, but the M5 empirical exit
is still OPEN, so the M6.1 fixture must be tagged `development`.

## 6. Release status must fail closed

Release status is `draft`, `candidate`, or `production`. Claiming production
requires:

- every renderer profile is `production_approved`;
- every profile carries an approval-report hash;
- every item's M6.3 QC is PASS with a QC-report hash;
- the generator code revision is neither `dirty` nor `unknown`;
- the split, assets, audio, metadata, and manifest audits all pass.

The M6.1 fixture's `release_status=draft` and `qc.status=pending` let
`ready_for_m6_bank_generation=true` and `ready_for_production=false` hold at
the same time. The former means only that the contract and assets are
consistent; the latter is the release decision.

## 7. M6.1 formal result

(This section originally held a result record; it has moved to the
[`RIR_EXP_LOG.md`](../../RIR_EXP_LOG.md) appendix.)

## 8. M6.2: wiring up the real generator

`generate_hybrid_rir.py --emit-m6-manifest` now materializes the full task
plan before serial or parallel rendering begins. Every item derives its own
seed from the base seed and `sample_id`, so worker scheduling never changes
the acoustic realization.

M6 resume no longer just checks "does a WAV exist and parse as JSON"; it also
checks:

- generation-config, renderer profile, split, and task identity;
- the code revision and key runtime package versions;
- the canonical scene hash;
- the WAV SHA-256;
- sample rate, channel count, and frame count.

Only when everything matches is an item skipped. If a WAV was modified, only
that item is rebuilt; if duration, renderer, code revision, or any other
content-generating config changed, old items are never mixed into the new
bank.

Generation automatically emits:

- `rir_bank_manifest.json`;
- `indexes/train.jsonl`, `validation.jsonl`, `test.jsonl`;
- `rir_bank_generation_audit.json`.

`PreGeneratedRoomBank` requires an explicit `split="train"`, `"validation"`,
or `"test"` when it encounters an M6 multi-split root; an unspecified split
fails closed, so a training script that reuses old directory-scanning code
cannot accidentally mix validation/test RIRs into train. If the manifest is
missing but the directory still carries M6 metadata or all three split
indexes, the reader refuses to fall back to legacy behavior; only a genuinely
legacy layout keeps the old behavior.

The formal non-evidence fixture used the real Pyroomacoustics high backend
with NumPy and libroom RNGs pinned, comparing serial, two-worker parallel, an
independent fresh rerun, tampered resume, and changed-revision/config resume
— 17/17 gates PASS. Tampering one WAV rebuilds only `1/6` and restores the
original hash; a revision/config change never generates a hybrid bank.

This feature is opt-in; the existing generation flow and the Pyroomacoustics
default are unchanged when `--emit-m6-manifest` is not passed.

## 9. M6.3: per-item physical QC and quarantine

M6.3 never treats "a metric can't be computed" as automatically zero, and
never mistakes five independent source-to-one-receiver transfer paths for
five synchronized microphones. Every check has one of four states:

- `pass`: measurable, and inside the versioned policy's physical bounds;
- `fail`: violates a hard invariant — the item must go to quarantine;
- `not_evaluable`: the data length, noise floor, or metadata is insufficient
  for a reliable estimate;
- `not_applicable`: the metric's physical premise does not apply.

The default policy id is `puresound.rir_bank_qc.physical.v1`. The policy
itself has a canonical-JSON SHA-256, so adjusting a threshold produces
distinct QC evidence instead of silently overwriting old results.

### 9.1 Structural and provenance checks

Every item is first checked for:

- WAV/metadata existing, with SHA-256 matching the manifest;
- metadata identity and canonical scene hash matching the manifest;
- the WAV decoding, with sample rate, frame count, and channel count matching
  the manifest;
- every sample finite, total energy nonzero, and peak conforming to the
  level policy — calibrated float has no artificial unit-peak ceiling, only
  the peak-normalized variant requires peak ≤ 1;
- `channel_map` having exactly one unique index per audio channel.

Failing this layer means the asset or contract itself cannot be trusted, and
downstream acoustic numbers should not be treated as evidence.

### 9.2 Causality and direct arrival

For distance \(d\), sound speed \(c\), and sample rate \(f_s\), the geometric
direct arrival is

\[
n_d = \frac{d}{c} f_s.
\]

A signal disproportionate to the channel peak before `floor(n_d)` is flagged
`prearrival_energy`; the first reliable onset must also land within the
policy's tolerance of \(n_d\). The onset search only runs inside a local
window after the geometric arrival, so it does not mistake the tiny numerical
residue of a finite modal/voxel expansion for the direct arrival; the
generator itself already clears low-frequency samples with `n <
floor(n_d)` before crossover. DRR, C50/C80, and decay analysis use the local
direct peak near the geometric arrival, not the whole RIR's global peak, so a
late reflection is never mistaken for the direct path.

### 9.3 Early field, decay, spectrum, and diffusion

Every channel records:

- peak, total energy, tail-energy fraction after 50 ms;
- DRR, C50, C80;
- noise-aware EDT, T20, T30, and fit \(R^2\);
- broadband spectral tilt and valid octave-band metrics; when the length is
  sufficient, octave T20/fit coverage feeds into the hard admission gate;
- Abel–Huang normalized echo density, mixing time, and late median density.

An unreliable single decay fit is first marked `not_evaluable`. Only when the
RIR is long enough overall, yet the item still fails to reach the minimum
reliable T20 channel coverage, does `decay_fit_coverage` become a hard
failure. This keeps "genuinely no reasonable decay evidence" separate from
"the file is too short to estimate."

The current generator's 5 channels are five independent source-to-one-receiver
transfer paths, not one source captured by five synchronized receivers. So
IACC/array coherence is explicitly `not_applicable`; future spatial QC can
only run these metrics when the metadata declares a synchronized
receiver pair/array.

### 9.4 Candidate indexes and quarantine

Run:

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/phases/m6_bank/scripts/run_m6_item_qc.py \
  --bank exp/my_m6_bank
```

which produces:

- `qc/items/<item_id>.json`: full per-item metrics, checks, and failure
  reasons;
- `qc/candidate_indexes/{train,validation,test}.jsonl`: QC-PASS items only;
- `qc/quarantine/index.jsonl`: QC-FAIL items, report references, and reasons;
- `rir_bank_qc_summary.json`: policy, indexes, counts, release decision, and
  hashes;
- an updated `rir_bank_manifest.json`: each item's QC status and
  report path/hash.

QC never deletes or modifies the original WAV/metadata. `PreGeneratedRoomBank`
always excludes `qc.status=fail` by default; candidate/production manifests
accept only `pass`. Only a diagnostic tool that explicitly passes
`include_failed_qc=True` can read quarantined items. If any candidate split
would become empty, release stays `draft` rather than producing a fake
candidate missing test or validation data.

The formal validator used real M6.2 generator output. The normal set is
`6/6` PASS; four negative controls — injecting a silent RIR, a pre-arrival
impulse, a direct-only sparse late field, and a direct arrival 3 ms late —
all land in quarantine and nowhere else. 13/13 gates PASS overall, and
tampering the QC report makes the content audit FAIL. This is still
synthetic implementation evidence, not measured-acoustic or production
evidence.

## 10. M6.4: freezing distribution, variant, and recipe

M6.4's release root uses `puresound.rir_bank_release.v1`, keeping three
things separate:

1. **variant**: one bank with a consistent signal semantic, e.g. calibrated
   or peak-normalized;
2. **distribution**: that variant's scene and acoustic-metric snapshot;
3. **recipe**: the variant, origin weights, and three split indexes a
   training run can select.

The two synthetic variants that can currently be released are:

- `synthetic_calibrated`: an identity copy of the M6.3 QC candidate;
- `synthetic_peak_normalized`: every item's channels are multiplied by the
  same single gain so the global peak reaches `0.98`.

"All channels share one gain" matters. Normalizing per channel would break
the energy relationship between channels; a shared gain only changes the
level-dependent peak, and in theory preserves

\[
\mathrm{DRR},\ C_{50},\ C_{80},\ T_{20}
\]

unchanged. The formal validator actually recomputes these metrics, with a
maximum error within `1e-4`, and also requires the parent/child pair to share
identical acoustic-space, room, scene, and split identity.

The distribution stores full rows, quantiles, and SHA-256 for room volume,
RT60, distance, peak, tail energy, DRR, C50/C80, T20, spectral tilt, mixing
time, and late echo density. This makes bank-level comparisons recomputable
rather than resting on a single chart or a hand-written summary.

The ready recipe's train/validation/test JSONL all carry an item count and a
file hash. `PreGeneratedReleaseBank` can consume it directly:

```python
from puresound.audio.rir.bank.loader import PreGeneratedReleaseBank

bank = PreGeneratedReleaseBank(
    "exp/my_m6_release",
    recipe_id="synthetic_calibrated",
    split="train",
)
scene = bank.sample_scene()
rir, metadata, sample_rate = bank.select_channel(scene, "foreground")
```

At construction the reader first audits the whole release; it samples
synthetic/real according to the recipe's `origin_weights`, then the variant
within that origin — never an implicit weighting formed from file counts on
disk. Metadata carries `release_sha256`, the recipe, variant, and origin.

Without a measured bank, the `real_native` and `mixed_calibrated_real`
recipes stay `blocked`; a blocked recipe has no readable index and must carry
a reason, so tooling never passes off synthetic data as real. M6.4's formal
result is **12/12 implementation PASS**. As of `RIR_EXP_LOG.md` §5, supplying
`--measured-bank` at generation time now makes both recipes `ready`: the
measured-ingest path (§12 below) resolves 313/313 channels from five public
corpora through the causality gate, and the production-decision checks
`all_required_recipes_are_ready` / `real_and_mixed_recipe_semantics_are_valid`
flip from `False` to `True` (§12). Without `--measured-bank`, behavior — and
the blocker text — is exactly as before.

### 10.1 Why float WAV needs header canonicalization

libsndfile's IEEE-float WAV writer normally adds a `PEAK` chunk carrying the
Unix timestamp of the write. It does not affect any audio sample, yet it
makes the same RIR hash to a different SHA-256 depending on the second it was
written. M6.4 zeroes only this non-acoustic timestamp after writing, without
touching chunk size, peak value, or the waveform. This is a byte-
reproducibility fix, not audio processing.

## 11. M6.5: evaluating the evidence, not just whether it runs

`puresound.m6_bank_evaluation.v1` aggregates four kinds of evidence at once:

- bank-level acoustic distributions;
- generator throughput, skip, and failure counts;
- controlled listening;
- room-disjoint downstream tasks.

It splits into two exits:

- **implementation exit**: schema, recomputation, hashing, negative controls,
  and fail-closed logic all work;
- **empirical exit**: there is a genuine measured reference, real human
  responses, and multi-seed trained-model results.

The two are not interchangeable. The scale-invariant metric comparison
between calibrated/normalized variants already PASSes; because the release
has no real variant, the synthetic-to-measured distance is honestly reported
as `not_evaluable` rather than fabricating a number.

A listening report requires randomized, double-blind trials with a shared
loudness master gain, a hidden reference, a degraded anchor, room-disjoint
stimuli, and content-addressed assignment/response/analysis records; the
`empirical` tier additionally requires at least 20 participants and
recomputes the participant mean and confidence interval from the response
records. A downstream report recomputes the train/test acoustic-space hashes
from the recipe index, needs at least three unique seeds, a frozen
training/model recipe, and recomputes the 95% t confidence interval from the
paired per-seed improvement; every primary metric's lower bound must show
improvement.

The formal M6.5 validator's 14/14 implementation gates PASS, and confirms
five negative controls are all rejected: unblinded listening, a tampered test
split identity, single-seed downstream, a forged positive CI, and relabeling
non-human responses as empirical. The current listening/downstream numbers
are tagged `contract_fixture`, explicitly not the result of human responses
or a trained model. So **M6.5 implementation PASSes; empirical/production
remain OPEN.**

## 12. M6.6: promotion is a certificate, not a string edit

Flipping `release_status` from `candidate` straight to `production` is not a
release: it would invalidate the release content hash, and it proves neither
that M6.5 evidence exists nor that a reviewer approved it. M6.6 therefore
keeps the whole M6.4 candidate untouched and instead produces a
`puresound.m6_production_decision.v1` certificate, binding:

- the release SHA and the M6.5 evaluation SHA;
- the four ready recipes — synthetic calibrated/normalized, real native,
  mixed;
- every item's QC PASS and the pinned generator revision;
- each renderer profile's `production_approved` tier and approval-report
  hash;
- the actual files used for human listening and trained downstream training;
- approve records from the acoustics, ML, and release-owner roles.

The evidence bundle is not just a pile of hashes. Every artifact must use a
safe bundle-relative path, actually exist, and have a file SHA-256 matching
the bundle. The listening assignment/responses/analysis and the downstream
training recipe/checkpoints/analysis hashes must also exactly match the
hashes M6.5 evaluated at the time. Renderer-approval files must cover every
profile too.

The certificate has only two decisions: `approved` or `blocked`. Any check
being false forces `production_ready` to false, with a deterministic blocker
list preserved. The candidate remains usable for research; a production
consumer must ask for it explicitly:

```python
bank = PreGeneratedReleaseBank(
    "exp/my_m6_release",
    recipe_id="mixed_calibrated_real",
    split="train",
    require_production=True,
)
```

At this point the reader verifies the certificate's schema, decision hash,
release hash, the fixed canonical gate set, and blocker consistency, and
re-runs the release/evidence audits it is bound to plus decision
recomputation. A blocked, tampered, or missing certificate is always
rejected.

The formal validator's 15/15 implementation gates PASS, including an unsafe
evidence path, evaluation pass-flag/hash tampering, a competent forged
certificate that sets everything true and recomputes the outer hash,
directly editing the candidate status, and negative controls on the
production reader. As of the 2026-08-04 evidence pilot
(`RIR_EXP_LOG.md` §6, §6.4), 10 of the 13 checks now PASS: real/mixed recipe
availability, renderer-profile approval, and the evaluation/hash checks are
all resolved. The three remaining blockers are exclusively empirical —
≥20-participant controlled listening, multi-seed room-disjoint downstream
training, and the external evidence-bundle audit that depends on both — and
all three sit outside this codebase. This still means the M6.6 decision
engine is complete, not that production evidence has appeared out of thin
air.

Content addressing can prove "the reviewed bytes did not change"; it cannot
by itself prove a reviewer's identity. A real deployment should still let
sign-off records be produced and held by controlled CI/permission systems;
if the threat model includes an attacker who can rewrite the workspace at
will, an organizational digital signature belongs at the deployment layer
too.

### 12.1 Wiring into the existing training augmentation

The existing `AudioEffectAugmentor` can already build a release reader
directly from YAML:

```yaml
pregenerated:
  used: true
  bank_type: release
  folder: egs/rir_generation/exp/rir_realism/m6/training_pilot/pyroomacoustics_release
  recipe_id: synthetic_calibrated
  split: train
  usage_role: train
  require_production: false
```

Release mode requires a matching `recipe_id`, `split`, and `usage_role`; the
dynamic dataset also cross-checks against its own train/validation/test role,
so validation/test RIRs never end up mixed into train. Old configs that omit
`recipe_id` still go through `PreGeneratedRoomBank`, keeping backward
compatibility. Only a real production deployment should set
`require_production` to true; a blocked certificate is rejected exactly as
designed.

Generate two matched 4,000-item pilots first, rather than jumping straight to
a hundred thousand items:

```bash
PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot \
  --backend pyroomacoustics --num-workers 8

PYTHONPATH=. .venv/bin/python egs/rir_generation/generate_m6_bank.py \
  --output-dir egs/rir_generation/exp/rir_realism/m6/training_pilot_m4 \
  --backend path-events-m4 --num-workers 8
```

Both pairs fix the same scene v1, low backend, seed, room/item count, and
calibrated level, changing only the high backend. The script runs resumable
M6 generation, M6.3 QC, and M6.4 release in order, and never overwrites an
existing release. For a quick smoke test, add `--n-rooms 6 --rir-per-room 1
--num-workers 1` (M6 still needs to cover all three of train/validation/test).
A training example lives at
`egs/rir_generation/phases/m6_bank/config/m6_release_training_example.yaml`.

### 12.2 Hardened matched-backend preflight (2026-08-02)

(This section originally held a result record; it has moved to the
[`RIR_EXP_LOG.md`](../../RIR_EXP_LOG.md) appendix.)

## 13. How to reproduce

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_contract.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_reproducible_generation.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_item_qc.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_variant_release.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank_evaluation.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m6_bank/scripts/validate_m6_production_decision.py

.venv/bin/pytest -q \
  test/test_rir_bank_manifest.py \
  test/test_m6_bank_contract_validator.py \
  test/test_generate_hybrid_rir_m6.py \
  test/test_m6_reproducible_generation_validator.py \
  test/test_rir_bank_qc.py \
  test/test_m6_item_qc_validator.py \
  test/test_rir_bank_release.py \
  test/test_m6_variant_release_validator.py \
  test/test_rir_bank_evaluation.py \
  test/test_m6_bank_evaluation_validator.py \
  test/test_rir_bank_production.py \
  test/test_m6_production_decision_validator.py \
  test/test_m6_release_training_integration.py
```

Main outputs:

- `egs/rir_generation/exp/rir_realism/m6/rir_m6_bank_contract/rir_bank_manifest.json`;
- `egs/rir_generation/phases/m6_bank/reports/m6_bank_contract_report.json`;
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/reproducible/`;
- `egs/rir_generation/phases/m6_bank/reports/m6_reproducible_generation_report.json`;
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/item_qc/`;
- `egs/rir_generation/phases/m6_bank/reports/m6_item_qc_report.json`;
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/variant_release/`;
- `egs/rir_generation/phases/m6_bank/reports/m6_variant_release_report.json`;
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/evaluation/`;
- `egs/rir_generation/phases/m6_bank/reports/m6_bank_evaluation_report.json`;
- `egs/rir_generation/exp/rir_realism/m6/rir_m6_hardening_validation/production/`;
- `egs/rir_generation/phases/m6_bank/reports/m6_production_decision_report.json`.

The next step is not another schema layer but the missing evidence that
lives outside this codebase. Measured distributions and renderer-approval
records are now in place (`RIR_EXP_LOG.md` §5, §6); what remains is real
controlled listening (≥20 participants), room-disjoint downstream training,
and the three-role sign-off that certifies them. Only once all of that is
auditable will rerunning the same M6.6 CLI emit an `approved` certificate.
