# Test Utilities

繁體中文版本：`README.zh-TW.md`

Bootstrap the repository-managed environment first:

```bash
uv sync --locked --group dev
```

Run focused checks while iterating:

```bash
uv run python test/run_repo_checks.py
```

87 pytest files make up the suite. The majority — 64 files, about 74% —
sit directly at `test/` root rather than in a named subdirectory; nearly
all of them cover the RIR/impedance/room-acoustics subsystem (`test_rir_*`,
`test_impedance_*`, `test_m5_*`, `test_m6_*`, and related FDN/path-event/
calibration coverage for the `egs/rir_generation` subsystem). That's simply
the current shape of the suite, not a gap to close: the RIR subsystem's
test surface is larger than the rest of the repository combined, so it
outweighs every named subdirectory.

The remaining 23 files are grouped by domain into five subdirectories:

- `test/test_audio`: audio I/O, DSP, augmentation, and simulation
- `test/test_metrics`: evaluation metrics such as DNSMOS
- `test/test_losses`: loss functions
- `test/test_utils`: recipe smoke tests, CLI helpers, and data adapters
- `test/test_system`: system-level integration tests — channel-consistency
  regularization, DPCRN gate/VAD training, optimizer param-group plumbing,
  and SISO `compute_loss` routing

Run the full pytest suite:

```bash
uv run python test/run_repo_checks.py --suite full --skip-ruff
```

To audition what a recipe actually trains on, use
`egs/voice_isolate/scripts/check_training_data.py --dump` (real recipe pipeline)
or `egs/rir_generation/tools/audition/simulate_room_scene.py` (on-the-fly room simulator).
