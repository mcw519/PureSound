# Repository layout

Traditional Chinese: [repository_layout.zh-TW.md](repository_layout.zh-TW.md)

## Logic lives in the library; recipes stay thin

`egs/<task>/` is a **driver**. It names a corpus, a recipe and a list of benchmark
stages; it does not implement them. A recipe directory holds:

| file | what it is |
| --- | --- |
| `main.py` | builds the task dataset, hands off to `puresound.system.runner` |
| `config/*.yaml`, `config/eval/*.yaml` | the released training, inference and evaluation recipes |
| `run_full_gate.sh` | the benchmark stage list — invocations and a summary, no logic |
| `README.md`, `README.zh-TW.md` | how to run this recipe |
| `pretrained_ckpt/` | released checkpoints, ONNX artifacts, and the per-version judgement table |
| `benchmarks/` | benchmark definitions and result records — numbers only, never audio |

Everything else with logic in it belongs to a library package:

| concern | module |
| --- | --- |
| corpus preparation and manifests | `puresound.dataset.corpus` |
| synthesis and augmentation | `puresound.audio`, `puresound.task` |
| training driver, DDP, stages | `puresound.system.runner` |
| evaluation protocols, statistics, records | `puresound.evaluation` |
| streaming export and runtime | `puresound.streaming` |
| released-model registry and inference | `puresound.inference` |

**When a second recipe needs a tool the first one has, the tool moves into
`puresound/` and both import it. It is never copied.** A task-local `scripts/`
directory of Python is how a repository ends up maintaining the same evaluation
twice and discovering, a version later, that the two disagree.
`egs/voice_isolate/scripts/` predates this and is the one exception on disk; it
migrates on demand, one tool at a time, as another recipe needs it.

`puresound.cli` is the model-zoo and inference surface (`puresound models`,
`puresound infer`, `puresound web`). Importing it does not load Torch; Torch comes
in only when a command reads or writes an audio file. Training and evaluation tools
are library modules run as `python -m puresound.evaluation.tools.<name>`, not
subcommands there.

## Naming a run, and naming a release

A run directory and a catalog id are read by people, months later, next to
dozens of siblings. Both name the **architecture that was actually trained**, not
the family it started from.

**Run directories** — `exp/<task>_<arch>_<variable>_<stage>`, e.g.
`ns_dpcrn-mamba_activebin_s2`.

| segment | why |
| --- | --- |
| `<task>` | run directories from different recipes end up side by side — in a shared `exp/`, an archive, a results table — and without this a noise-suppression run and a voice-isolation run are indistinguishable |
| `<arch>` | the skeleton **and what differs from it**. The noise-suppression models are DPCRN with the temporal path replaced by Mamba over ERB bands; calling them `dpcrn` misreports what was trained |
| `<variable>` | what this run is testing — the thing a bare version number never carried. `v3_s2` says nothing; `activebin_s2` says what changed |
| `<stage>` | `s0`..`sN` for a curriculum ladder, omitted when there is no ladder; `ft` for a fine-tune probe |

**Catalog ids** — `<task>-<arch>-<version>`, e.g.
`noise-suppression-dpcrn-mamba-v0`. A zoo id is typed by users and pinned in
their code, so unlike a run directory it keeps a plain ordered version and puts
the meaning in `display_name` and `description`. The architecture segment obeys
the same accuracy rule. **Renaming one is a breaking change**: add the old
spelling to the alias table in `puresound/inference/zoo.py` and pin it with a
test, rather than expecting callers to migrate.

## Release artifacts versus experiment artifacts

The dividing question is whether someone cloning the repository would need the file
to reproduce a released model.

| kept | not kept |
| --- | --- |
| `puresound/`, `test/`, `docs/`, `sdk/` | `exp/` — runs, logs, intermediate checkpoints |
| `main.py`, `config/*.yaml`, `config/eval/*.yaml` | `config/exp/` — experiment recipes |
| `run_full_gate.sh`, recipe READMEs | `data/` — corpora, metafiles, list files (local absolute paths) |
| `benchmarks/` — records, zero audio | `data_report/` — evaluation audio |
| released ONNX artifacts + `model_zoo/catalog.yaml` | `proc/`, `dummy_samples/`, `backup/` |
| `pretrained_ckpt/README.md` — judgement table | `pretrained_ckpt/*.ckpt` — until one is promoted |

A released recipe is promoted out of `config/exp/`; a released checkpoint is promoted
out of the run directory alongside the catalog entry and the judgement-table row that
justify it. What each version was judged on is that judgement table; the
experiment log is not part of the repository (below).

Records carry numbers, never the audio they were computed on, and never material
derived from private recordings. That is why `egs/noise_suppression/benchmarks/` is
kept while `egs/voice_isolate/benchmarks/` is not: the first scores public corpora,
the second uses private recordings that are not distributed.

## Comments, documentation and tests

**Comments and docstrings** say what the code does, why it is designed that way,
and what it assumes or does not handle. They do not carry dates, version or run
names (`v8`, `ns_dpcrn-mamba_*_s1`), experiment results (metric values,
before/after numbers, "measured on ..."), history ("used to", "previously"), or
references to experiment logs. Where a design decision rests on an experiment,
the comment states the decision and its reason in words; if the reader needs
evidence, it points to the test that pins the behaviour or to a paper. Numbers
that are properties of the design itself -- a ceiling that follows from
arithmetic, a buffer size, a sample rate -- stay.

**Experiment logs** -- run-by-run results, dates, comparisons between versions,
what each experiment tried and ruled out -- are kept outside the public
repository. Code, documentation and recipe READMEs do not point to them.

**Documentation** in `docs/` is kept in English and Traditional Chinese, changed
together in the same commit, and organised in three parts:

| part | covers |
| --- | --- |
| `docs/architecture/` | how the packages fit together; how data flows through corpus preparation, synthesis, training, evaluation, export and deployment; the config system and the model zoo |
| `docs/algorithms/` | what each model, loss, synthesis and augmentation stage, RIR generator, metric and statistic computes, and why |
| `docs/usage/` | preparing data, training and curriculum ladders, running the gate, exporting and deploying, the streaming SDK, the web playground |

**Tests** mirror the package layout (`test/<package>/test_<module>.py`). A test
pins a behaviour a caller relies on: an output contract, a shipped path, a failure
that must be loud. Cases of one behaviour are one parametrised test; a test that
only reproduces one experiment's configuration is not kept. A test that takes
more than a few seconds is marked `slow`.

**Frozen legacy** -- `puresound/task/tse.py`, `puresound/task/sv.py`,
`puresound/system/miso.py`, `puresound/dataset/kaldi_base.py` and their recipes
-- is not rewritten to these rules; its documentation states only its status.

The Web SDK lives in `sdk/web/`: TypeScript source, pinned npm lockfile and build/test
tools. `node_modules` and the compiled SDK are generated locally. `npm run assets`
writes the compiled runtime, ONNX Runtime Web and the device builds of the released
models, with their hash catalog, into `puresound/web/static/device/`, beside the
version-controlled `device.js`, `worker.js` and third-party notices that Playground's
*This device* runs use; Python package data includes these payloads when built.
