# RIR bank loaders — `puresound.audio.rir.bank`

Training-side access to pre-generated RIR banks. The generation/QC/release
pipeline is documented in
[`rir_realism_algorithm_zh-TW.md`](rir_realism_algorithm_zh-TW.md) §10 and the
M6 contract in [`rir_bank_v2_zh-TW.md`](rir_bank_v2_zh-TW.md).

## `PreGeneratedReleaseBank` (`bank/loader.py`)

Loads an M6.4 release (`rir_bank_release.json`) and serves items from one
recipe and one split.

- `recipe_id` and `split` are required; the loader re-audits the release on
  construction and fails closed on hash mismatches.
- `split == usage_role` is cross-checked so a train reader cannot silently
  serve test rooms.
- A release missing its manifest (or with M6 metadata present but indexes
  removed) raises instead of degrading to directory scanning.

## `PreGeneratedRoomBank`

Legacy directory-of-WAV loader for pre-M6 banks. Detects manifestless M6
layouts and refuses them rather than mixing splits.

## Training configuration

Wired through `AudioEffectAugmentor.init_room_bank` (`puresound/audio/augmentation.py`):

```yaml
room_bank:
  used: true
  bank_type: release          # or "room" for the legacy loader
  folder: <release root>
  recipe_id: synthetic_calibrated   # real_native / mixed_calibrated_real when ready
  split: train
  usage_role: train
```

Provenance travels with every sample: `rir_release_sha256`, `rir_recipe_id`,
`rir_variant_id`, `rir_split`, `rir_origin`, `rir_renderer_profile_id`, and the
production certificate hash when present (`puresound/task/ns.py`).
