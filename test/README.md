# Test Utilities

Bootstrap the repository-managed environment first:

```bash
uv sync --locked --group dev
```

Run focused checks while iterating:

```bash
uv run python test/run_repo_checks.py
```

The pytest suite is grouped by domain:

- `test/test_audio`: audio I/O, DSP, augmentation, and simulation
- `test/test_metrics`: evaluation metrics such as DNSMOS
- `test/test_losses`: loss functions
- `test/test_utils`: recipe smoke tests, CLI helpers, and data adapters

Run the full pytest suite:

```bash
uv run python test/run_repo_checks.py --suite full --skip-ruff
```

To audition what a recipe actually trains on, use
`egs/voice_isolate/scripts/check_training_data.py --dump` (real recipe pipeline)
or `egs/rir_generation/tools/audition/simulate_room_scene.py` (on-the-fly room simulator).
