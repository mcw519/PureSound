# RIR bank loaders — `puresound.audio.rir.bank`

繁體中文版本：[`rir_bank.zh-TW.md`](rir_bank.zh-TW.md)

Training-side access to pre-generated RIR banks. The generation/QC/release
pipeline is documented in
[`rir_realism_algorithm.md`](rir_realism_algorithm.md) §10 and the
M6 contract in [`rir_bank_v2.md`](rir_bank_v2.md).

## `PreGeneratedReleaseBank` (`bank/loader.py`)

Loads an M6.4 release (`rir_bank_release.json`) and serves items from one
recipe and one split.

- `recipe_id` and `split` are required; the loader re-audits the release on
  construction and fails closed on hash mismatches.
- `split` must agree with the dataset's `usage_role`, but that check is
  enforced one layer up (see below) — `PreGeneratedReleaseBank` itself takes
  no `usage_role` argument.
- A release missing its manifest (or with M6 metadata present but indexes
  removed) raises instead of degrading to directory scanning.

## `PreGeneratedRoomBank`

Legacy directory-of-WAV loader for pre-M6 banks. Detects manifestless M6
layouts and refuses them rather than mixing splits.

## Training configuration

Wired through `AudioEffectAugmentor.init_room_bank`
(`puresound/audio/augmentation.py`), called in turn from
`puresound/dataset/dynamic_base.py`'s `init_augmentor`. The YAML lives under
`augmentation_reverb.simulator.pregenerated`, a sibling of the on-the-fly
`room_simulator` path (not a standalone top-level key):

```yaml
augmentation_reverb:
  used: true
  simulator:
    used: true
    pregenerated:
      used: true
      bank_type: release          # or "room" for the legacy loader
      folder: <release root>
      recipe_id: synthetic_calibrated   # real_native / mixed_calibrated_real when ready
      split: train
      usage_role: train
      require_production: false
```

Two checks keep splits from leaking: `init_augmentor` rejects a `usage_role`
that disagrees with the dataset's own train/validation/test role (and fills
it in from that role when omitted); `init_room_bank` then rejects a `split`
that disagrees with the resulting `usage_role`. Either mismatch raises before
any RIR is read — a train dataset object can no more be pointed at `split:
test` than at a mismatched `usage_role`.

Provenance travels with every sample: `rir_release_sha256`, `rir_recipe_id`,
`rir_variant_id`, `rir_split`, `rir_origin`, `rir_renderer_profile_id`, and the
production certificate hash when present (`puresound/task/ns.py`).
