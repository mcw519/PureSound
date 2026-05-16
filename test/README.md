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

Generate a small set of simulated `voice_isolate` training samples for manual
listening:

```bash
uv run python test/generate_simulated_training_data.py \
  --output-dir test/test_case/outputs/manual_voice_isolate_data \
  --num-samples 3 \
  --foreground-distance 0.4 1.0 \
  --interferer-distance 2.0 4.0
```

Each sample folder contains `noisy_speech.wav`, `clean_speech.wav`,
`consistency_noise.wav`, source-level dry/reverberant files, and
`comparison_channels.wav` for quick auditioning.
