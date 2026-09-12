# RIR bank v2

繁體中文版本：[rir_bank_v2.zh-TW.md](rir_bank_v2.zh-TW.md)

The v2 bank format packages RIR audio with split, provenance, integrity, QC,
and release information. A directory of WAV files is not sufficient for a
reproducible training release.

## Bank contents

A bank root contains:

```text
bank/
├── rir_bank_manifest.json
├── rir_bank_generation_audit.json
├── indexes/
│   ├── train.jsonl
│   ├── validation.jsonl
│   └── test.jsonl
├── scenes/
├── metadata/
└── audio/
```

The manifest is the source of truth. Each item records:

- stable item, room, and parent-room identity;
- train, validation, or test split;
- scene schema and canonical scene hash;
- renderer profile, configuration, and code revision;
- sample rate, channels, frames, and level policy;
- paths and SHA-256 hashes for retained assets;
- QC status and release eligibility.

Rooms, including descendants of the same synthetic parent room, must not
cross splits.

## Generate a bank

See all options:

```bash
python egs/rir_generation/phases/m6_bank/scripts/generate_m6_bank.py --help
```

A typical run selects the output directory, renderer backend, room count,
seed, sample rate, and duration explicitly. Keep these values in the release
recipe; changing a content-producing option creates a different bank.

Generation writes the manifest, split indexes, and generation audit. It is
safe to resume:

- an item is reused only when its identity, configuration, and hashes match;
- a modified or missing asset is rebuilt;
- a changed renderer, revision, duration, or other generating input cannot be
  silently mixed into the existing bank.

When a v2 manifest or complete split indexes are present, readers fail closed
instead of falling back to legacy directory scanning.

## Load a split

`PreGeneratedRoomBank` requires an explicit split for a multi-split v2 bank:

```python
from puresound.audio.rir.bank import PreGeneratedRoomBank

bank = PreGeneratedRoomBank(
    "/data/rir-bank",
    split="train",
)
```

Never point training at the bank root without choosing a split.

## Quality control

QC uses four states:

| State | Meaning |
|---|---|
| `pass` | The check is measurable and within its limits |
| `fail` | A hard invariant is violated; quarantine the item |
| `not_evaluable` | The signal or metadata cannot support a reliable estimate |
| `not_applicable` | The check does not apply to this item |

The QC policy and its canonical hash are stored with the results. Threshold
changes therefore produce distinct evidence.

Structural checks cover:

- file existence and SHA-256 integrity;
- item identity and canonical scene hash;
- decoded sample rate, frame count, and channel count;
- finite, nonzero audio and the declared level policy;
- a complete, unique channel map;
- renderer and provenance fields.

Physical checks cover the metrics that apply to the item, including causality,
arrival timing, decay, clarity, spectrum, echo density, and spatial behavior.
Unsynchronized transfer paths must not be treated as a microphone array.

Run QC:

```bash
python egs/rir_generation/phases/m6_bank/scripts/validate_m6_bank.py --help
```

Failed items belong in quarantine and must not appear in release indexes.
Generated audit and QC reports are local artifacts unless a release process
explicitly retains them.

## Release variants and recipes

A release recipe defines the exact mixture used by training or evaluation.
Keep measured and synthetic origin visible instead of merging files into an
unlabeled directory.

Common variants are:

- calibrated synthetic RIRs;
- peak-normalized synthetic RIRs;
- native measured RIRs;
- an explicit measured/synthetic mixture.

A recipe should pin:

- bank and manifest identity;
- allowed splits and variants;
- renderer profiles;
- sampling weights;
- level policy;
- QC policy and required state.

Use measured data only when its license and provenance permit redistribution
or derived training use.

## Measured RIRs

Measured corpora enter through the measured-ingest path described in
[the implementation guide](rir_realism_algorithm.md#measured-rir-ingest).
The measurement chain acts as the renderer profile. Alignment policy,
environment assumptions, original provenance, and rejection reasons remain
attached to every item.

## Production evidence

A renderer can be technically valid without being production-approved.
Production promotion requires evidence external to unit tests, such as:

- controlled listening with the required participant count;
- room-disjoint downstream training and evaluation;
- an auditable bundle linking the bank, recipe, results, and code revision.

The production certificate distinguishes implemented checks from empirical
evidence. Do not set a production-required flag by bypassing missing evidence
or relabeling a development profile.

## Training configuration

Reference a released bank and split explicitly in training configuration.
The exact field names depend on the task configuration; the important
contract is that training resolves to a manifest-backed release recipe, not
an arbitrary directory of WAV files.

Keep validation and test recipes separate from training, even when all files
are stored under one bank root.

## Release checklist

Before publishing a bank:

1. Regenerate or resume using a pinned recipe and clean code revision.
2. Verify the manifest and every retained asset hash.
3. Run structural and applicable physical QC.
4. Confirm room and parent-room split disjointness.
5. Build release indexes from passing items only.
6. Audit licenses and provenance for measured sources.
7. Record empirical evidence separately from implementation tests.
8. Test loading every published split from the final bundle.
