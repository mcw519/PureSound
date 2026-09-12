# Test Utilities

繁體中文版本：`README.zh-TW.md`

Bootstrap the repository-managed environment first:

```bash
uv sync --locked --group dev
```

The suite runs in three tiers. All of them run under pytest-xdist with one
compute thread per worker — serially the same suite takes over an hour,
dominated by the RIR physics and M5/M6 evidence tests.

| Tier | Scope | Wall clock |
|---|---|---|
| `--suite quick` | unit tests only (the `test/*/` subpackages) | ~20 s |
| `--suite standard` | everything except tests marked `slow` | ~35 s |
| `--suite full` *(default)* | everything | ~2 min |

```bash
uv run python test/run_repo_checks.py                    # full, before committing
uv run python test/run_repo_checks.py --suite quick      # while iterating
```

Timings are on 24 cores. `full` is the default because at ~2 minutes there is
rarely a reason to verify less; `standard` earns its keep on a smaller machine,
where that gap widens.

Add `--jobs 0` to disable xdist when you need a debugger or readable live
output, or `--jobs N` to pin the worker count. Leave the thread pinning alone
unless you know why it is there: torch otherwise starts one intra-op thread
per core in *every* worker, and the resulting oversubscription cost the 63
`slow` tests 339 CPU-minutes to do 3 minutes of work.

63 of the 668 tests carry the `slow` marker: the M5/M6 evidence-chain
validators (each builds, QCs and releases a real bank), the two
offline-vs-ONNX streaming equivalence checks (~2.5 min each), and the forward
smoke tests for the library-status backbones (SkiM, DPRNN, TF-GridNet). The
active deployment path — DPCRN and DPARN — deliberately stays in `standard`.

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

To audition what a recipe actually trains on, use
`egs/voice_isolate/scripts/check_training_data.py --dump` (real recipe pipeline)
or `egs/rir_generation/tools/audition/simulate_room_scene.py` (on-the-fly room simulator).
