# puresound.audio.rir_bank

Serve RIR channels from a folder of pre-rendered multi-source rooms — the
production alternative to simulating a room per item (see
[audio.room_simulator](room_simulator.md)). Banks are built by
`egs/rir_generation` (`generate_hybrid_rir.py`,
`tools/measured/real_rir_to_bank.py`, `tools/bank/build_bank_view.py`); a
bank/view is a folder of same-stem `.wav`/`.json` pairs, either directly per
room or under an `items/` child (symlink views).

M6 release candidates additionally carry a content-addressed
`rir_bank_manifest.json` with deterministic acoustic-space-level splits,
generator/renderer provenance, asset and scene hashes, audio shapes, QC state,
and fail-closed production evidence. The WAV/JSON item layout remains backward
compatible. M6.2 generation also writes content-addressed split JSONL indexes
and verifies task/config/scene/WAV identity before resume skips an item. See the
[M6 Bank v2 contract](rir_bank_v2_zh-TW.md).

M6.3 writes content-addressed reports under `qc/items/`, pass-only candidate
indexes under `qc/candidate_indexes/`, and a failure index under
`qc/quarantine/`. Structural or causal violations are failures; unavailable
decay evidence is recorded as `not_evaluable`, and source-indexed channels do
not falsely claim synchronized-array spatial metrics.

M6.4 adds `rir_bank_release.json`, calibrated/peak-normalized variant lineage,
frozen acoustic distributions, and content-addressed recipe indexes. Ready
recipes can be consumed with `PreGeneratedReleaseBank`; missing measured data
leaves real/mixed recipes blocked. M6.5 evaluates distributions, throughput,
controlled listening, and room-disjoint downstream reports while separating
implementation validity from empirical evidence.

M6.6 keeps that candidate immutable and publishes a separate content-addressed
production decision certificate. The certificate binds the release/evaluation,
evidence files, production renderer approvals, and acoustics/ML/release-owner
sign-offs. The current certificate is valid but blocked; no status string is
treated as sufficient production evidence.

## Class: `PreGeneratedRoomBank`

```python
PreGeneratedRoomBank(
    folder: str,
    near_labels=("near_0", "near_1"),
    far_labels=("far_0", "far_1", "far_2"),
    drr_window_ms: float = 2.5,     # direct-path window for DRR metadata
    wav_name: str = "rir_5ch.wav",
    meta_name: str = "metadata.json",
    cache_size: int = 64,           # LRU of decoded room wavs
    split: str | None = None,       # required for an M6 multi-split root
    manifest_name: str = "rir_bank_manifest.json",
    include_failed_qc: bool = False,# explicit diagnostics only
)
```

- `sample_scene() -> dict` – pick a room; returns the near/far channel pools,
  `room_id`, and `origin` (`"real"` for measured-RIR banks, else None) so the
  dataset can gate behaviour per origin.
- `select_channel(scene, role, ...) -> (rir, metadata)` – draw one channel for
  a role (`near`/`far`/`media`); metadata carries `source_receiver_distance`,
  `drr_db`, `rt60`, `origin`.
- `__len__` – number of indexed rooms.

When the folder contains an M6 manifest, `split` must be explicitly set to
`train`, `validation`, or `test`; the reader refuses an unsplit M6 root. Legacy
banks without an M6 manifest keep the original directory-scanning behaviour.
Failed items are excluded by default, and candidate/production manifests admit
only QC PASS items. `include_failed_qc=True` exists for quarantine diagnostics,
not normal training or evaluation.

## Class: `PreGeneratedReleaseBank`

```python
PreGeneratedReleaseBank(
    folder: str,                 # M6.4 release root
    recipe_id: str,              # a ready release recipe
    split: str,                  # train, validation, or test
    release_manifest_name: str = "rir_bank_release.json",
    require_production: bool = False,
    production_decision_name: str = "rir_bank_production_decision.json",
    **room_bank_kwargs,
)
```

The reader audits the complete release before opening a recipe. Sampling first
uses the frozen recipe origin weights, then a variant within that origin, then
an item. This prevents a mixed synthetic/real recipe from accidentally using
file-count proportions. `sample_scene()` and `select_channel()` propagate the
release id/hash, recipe id, variant id, origin, and split into metadata. Blocked
recipes and unknown scenes fail closed.

With `require_production=True`, initialization additionally requires an M6.6
certificate whose content hash and release identity are valid and whose
decision is `approved`. A missing, tampered, or correctly blocked certificate
is rejected. Approved samples propagate the production-certificate hash in
their metadata.

## Training configuration

`AudioEffectAugmentor.init_room_bank()` keeps the legacy `bank_type: room`
behavior and dispatches `bank_type: release` to the M6 recipe reader. If
`bank_type` is omitted, the presence of `recipe_id` selects release mode.

```yaml
simulator:
  used: true
  source_level: true
  pregenerated:
    used: true
    bank_type: release
    folder: /path/to/m6_release
    recipe_id: synthetic_calibrated
    split: train
    usage_role: train
    require_production: false
```

Release mode requires `recipe_id`, an explicit split, and a matching
`usage_role`; the dynamic dataset also cross-checks that role against its own
train/validation/test role. Release-only options are rejected in room mode. Use
`require_production: false` for current candidate experiments; changing it to
true correctly rejects the present blocked M6.6 certificate.

## Recipe wiring

```yaml
augmentation_reverb:
  simulator:
    pregenerated:
      used: True
      folder: /path/to/bank_or_view
      near_labels: [near_0, near_1]
      far_labels: [far_0, far_1, far_2]
      drr_window_ms: 2.5
      cache_size: 128
```
