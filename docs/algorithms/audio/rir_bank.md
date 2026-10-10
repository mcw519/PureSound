# RIR bank loaders — `puresound.audio.rir.bank.loader`

繁體中文版本：[rir_bank.zh-TW.md](rir_bank.zh-TW.md)

The training-side view of pre-generated RIR banks: pick a room, then pick one
channel per source role. This module is the only part of `puresound.audio.rir`
a training run imports. The bank format itself (manifest, splits, QC, release)
is defined in [RIR bank format](rir_bank_v2.md).

All loaders expose the same two calls as the on-the-fly
[room simulator](room_simulator.md), so the augmentation path is the same for
both:

```python
scene = bank.sample_scene()                  # one room for the whole mixture
impulse, meta, sr = bank.select_channel(
    scene, source_role="foreground",         # or interferer / media / echo / other
    distance_range_override=None,            # optional [lo, hi] in metres
)                                            # impulse: [1, L]
```

## `PreGeneratedRoomBank`

```python
PreGeneratedRoomBank(
    folder, near_labels=("near_0", "near_1"), far_labels=("far_0", "far_1", "far_2"),
    drr_window_ms=2.5, wav_name="rir_5ch.wav", meta_name="metadata.json",
    cache_size=64, split=None, manifest_name="rir_bank_manifest.json",
    include_failed_qc=False, allow_legacy_layout=True,
)
```

It reads two layouts:

- **Manifest bank** (a `rir_bank_manifest.json` is present). `split` is
  required — a multi-split bank cannot be read without choosing one. The loader
  verifies the manifest content hash, serves only that split's items, skips
  items that failed QC (unless `include_failed_qc`) and, in a `candidate` or
  `production` bank, items whose QC is still pending. A served item whose RIR
  file is missing, or whose metadata is unreadable or lacks the scene layout the
  loader reads, raises at construction rather than failing mid-training.
- **Directory bank** (no manifest). Each room subdirectory holds a
  `rir_5ch.wav` + `metadata.json` pair or same-stem `<item>.wav` + `<item>.json`
  pairs. This is how banks without a manifest are read, such as measured-corpus
  banks written by `egs/rir_generation/tools/measured/real_rir_to_bank.py` and
  training views assembled from several sources. `split=` is rejected here
  because nothing records a split. If the folder shows signs of being a manifest
  bank with its manifest missing — complete `indexes/` or item metadata carrying
  bank fields — the loader refuses rather than mixing splits.
  `allow_legacy_layout=False` disables this layout entirely.

**Scene and channel choice.** `sample_scene()` picks a room uniformly and splits
its channels into a near pool and a far pool by label; when no label matches,
it splits at the median distance. `select_channel` draws from the near pool for
`foreground`, from the far pool for `interferer`, `media` and `echo`, and from
all channels otherwise, and avoids re-using a channel within one scene.
`distance_range_override` keeps only channels inside the range, falling back to
the channel nearest the range centre. The returned metadata carries the
source–receiver distance, the DRR over `drr_window_ms`, RT60, label, room and
split identity, origin, renderer profile and QC report. WAVs are held in an LRU
cache of `cache_size` rooms. The augmentor resamples the impulse to the dataset
rate with the transparent resampler.

## `PreGeneratedReleaseBank`

```python
PreGeneratedReleaseBank(
    folder, *, recipe_id, split,
    release_manifest_name="rir_bank_release.json",
    require_production=False, production_decision_name="rir_bank_production_decision.json",
    audit=True, audit_cache=True, **bank_kwargs,
)
```

Serves one recipe of a release, one split at a time. On construction it audits
the release and fails closed on any mismatch (`audit_cache` reuses a verdict
stored beside the release; `audit=False` skips the audit for callers that
already ran it, such as a second DDP rank). With `require_production=True` it
also requires a valid, approved production certificate. The recipe must be
`ready`; each of its variants becomes a `PreGeneratedRoomBank` on the requested
split, and their item count must equal the recipe's index count.

`sample_scene()` samples in three steps — an origin by the recipe's frozen
origin weights, then a variant of that origin, then an item — so a mixed
synthetic/measured recipe follows its weights rather than the number of files
on disk. Scenes and channel metadata carry `release_id`, `release_sha256`,
`release_recipe_id`, `release_variant_id`, `release_origin` and the production
certificate hash.

## `UnionRoomBank`

Serves several banks as one pool at fixed per-bank sampling probabilities
(weights are probabilities, not item counts). Each member keeps its own labels,
DRR window and cache, and every scene is routed back to the bank that produced
it. `set_weights({name: weight})` re-weights some members and renormalizes; a
curriculum uses it to move the room pool between epochs
(`puresound.config.curriculum`). A weight may be zero, but not all of them.

## Configuration

Banks are configured under `augmentation_reverb.simulator.pregenerated`, beside
the on-the-fly simulator settings (`puresound.config.augmentation`:
`PreGeneratedBankConfig`, `RoomBankMemberConfig`). Exactly one of `folder` or
`banks` must be given.

```yaml
augmentation_reverb:
  used: true
  simulator:
    used: true
    pregenerated:
      used: true
      bank_type: release              # default: release if recipe_id is set, else room
      folder: <release root>
      recipe_id: synthetic_calibrated # real_native / mixed_calibrated_real when ready
      split: train
      usage_role: train
      require_production: false
```

A union lists members instead; only `usage_role` may be shared at the top level:

```yaml
    pregenerated:
      used: true
      banks:
        - {name: simulated, weight: 0.6, bank_type: room, folder: <bank>}
        - {name: measured,  weight: 0.4, bank_type: room, folder: <bank>}
```

**Split-leak guards.** The dataset fills `usage_role` from its own pipeline role
and rejects a configured `usage_role` that disagrees
(`puresound/dataset/dynamic_base.py`). A `release` bank then requires `split`
to equal `usage_role`, so a training dataset cannot be pointed at validation or
test rooms; both checks raise before any RIR is read. A `room` bank does not
cross-check its `split` against the role, and rejects release-only options
(`recipe_id`, `require_production`, `audit`, ...).

## Provenance in samples

The dataset copies RIR identity into every sample (`RIR_PROVENANCE_KEYS` in
`puresound/task/ns.py`): `rir_release_id`, `rir_release_sha256`,
`rir_recipe_id`, `rir_variant_id`, `rir_split`, `rir_origin`,
`rir_renderer_profile_id`, `rir_production_certificate_sha256` and
`rir_interferer_variant_ids`. Keys are empty strings when a bank does not
provide them.
