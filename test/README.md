# Tests

繁體中文版本：[README.zh-TW.md](README.zh-TW.md)

Bootstrap the repository-managed environment first:

```bash
uv sync --locked --group dev
```

## Layout

`test/` mirrors the library: `test/<package>/test_<module>.py`. Test file
basenames are unique across the tree (the directories are not Python packages,
so pytest imports each file by its basename).

| directory | covers |
| --- | --- |
| `audio/` | `puresound.audio` outside the RIR stack: I/O, DSP, augmentation knobs |
| `config/` | recipe schema and loading (`puresound.config`) |
| `dataset/` | the dataset framework and every corpus-preparation tool (`puresound.dataset`, `puresound.dataset.corpus`) |
| `evaluation/` | the benchmark gate: tools, statistics, records, metrics |
| `inference/` | model zoo, runtimes, recipes registry, CLI |
| `nnet/` | backbones, building blocks, heads and losses (`puresound.nnet`) |
| `rir/` | the RIR generation stack (`puresound.audio.rir`) and its `egs/rir_generation` tools |
| `streaming/` | ONNX export, streaming parity and the standalone SDK runtime |
| `system/` | Lightning modules, training driver, postprocessing, presence gate, onset guard |
| `task/` | synthesis datasets, device chain, noise stage, sampler, session rows |
| `web/` | the web playground server |

`fixtures/` holds small recipes and data used by several tests; `test_case/`
holds the audio samples they read.

## Running

The suite runs in three tiers under pytest-xdist, one compute thread per worker:

| tier | scope |
| --- | --- |
| `--suite quick` | the unit-test directories (everything except `rir/` and `web/`), without `slow` tests |
| `--suite standard` | everything except tests marked `slow` |
| `--suite full` *(default)* | everything |

```bash
uv run python test/run_repo_checks.py                    # full, before committing
uv run python test/run_repo_checks.py --suite quick      # while iterating
```

Add `--jobs 0` to disable xdist when you need a debugger or readable live
output, or `--jobs N` to pin the worker count. Keep the thread pinning: torch
otherwise starts one intra-op thread per core in every worker and the workers
oversubscribe the machine.

A test that takes more than a few seconds is marked `slow`: bank build/QC/release
chains, offline-versus-ONNX streaming equivalence, long session synthesis. The
deployment path (DPCRN, DPARN, the streaming runtime) keeps fast tests in
`standard`.

## Writing tests

A test pins a behaviour a caller relies on: an output contract, a shipped path,
a failure that must be loud. Cases of one behaviour are one parametrised test.
See the "Comments, documentation and tests" section of
[docs/repository_layout.md](../docs/repository_layout.md).

To audition what a recipe actually trains on, use
`egs/voice_isolate/scripts/check_training_data.py --dump` (real recipe pipeline)
or `egs/rir_generation/tools/audition/simulate_room_scene.py` (on-the-fly room
simulator).
