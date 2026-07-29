# puresound.audio.rir_bank

Serve RIR channels from a folder of pre-rendered multi-source rooms — the
production alternative to simulating a room per item (see
[audio.room_simulator](room_simulator.md)). Banks are built by
`egs/rir_generation` (`generate_hybrid_rir.py`, `real_rir_to_bank.py`,
`build_bank_view.py`); a bank/view is a folder of same-stem `.wav`/`.json`
pairs, either directly per room or under an `items/` child (symlink views).

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
)
```

- `sample_scene() -> dict` – pick a room; returns the near/far channel pools,
  `room_id`, and `origin` (`"real"` for measured-RIR banks, else None) so the
  dataset can gate behaviour per origin.
- `select_channel(scene, role, ...) -> (rir, metadata)` – draw one channel for
  a role (`near`/`far`/`media`); metadata carries `source_receiver_distance`,
  `drr_db`, `rt60`, `origin`.
- `__len__` – number of indexed rooms.

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
