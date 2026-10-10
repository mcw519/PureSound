# Architecture

繁體中文版本：[index.zh-TW.md](index.zh-TW.md)

How the packages fit together, and how data moves from a corpus on disk to a
model running in someone else's process. What each stage computes is in
[Algorithms](../index.md#algorithms); how to run it is in
[Usage](../index.md#usage).

The repository has three kinds of code:

- **`puresound/`**, the library. Every piece of logic lives here: corpus
  preparation, synthesis, models, the training driver, evaluation, export and
  inference.
- **`egs/<task>/`**, the recipes. Each is a thin driver: a `main.py` that names
  its dataset class and hands off to `puresound.system.runner`, the recipe YAML
  files, a benchmark stage list, and the released checkpoints. See
  [Repository layout](../repository_layout.md) for the rule and its reasons.
- **`sdk/python/`**, a standalone streaming runtime that imports nothing from
  `puresound` and needs only NumPy and ONNX Runtime.

## Data flow

```text
 corpora on disk
     │  puresound.dataset.corpus         scan · resample · split · write
     ▼
 metafiles (7-column CSV) + JSONL inventories
     │  SpeakerSampler                   batch of item keys
     │  puresound.task.* datasets        one row synthesised per key:
     │    (on DynamicBaseDataset)          speech + rooms + noise + device chain
     ▼
 batches: noisy, clean, VAD / auxiliary targets
     │  puresound.system.EncDecMaskBase  encoder → features → backbone → mask → decoder
     │  puresound.system.runner          Lightning Trainer, DDP, checkpoints
     ▼                                   (recipe YAML loaded by puresound.config)
 checkpoints
     │  puresound.evaluation             paired scoring, statistics, records
     │  egs/<task>/run_full_*.sh         the stage list
     ▼
 benchmark records ──► release decision
     │  puresound.streaming              per-frame ONNX graph + JSON manifest
     ▼
 ONNX artifacts + model_zoo/catalog.yaml
     ├─► puresound.inference             ModelZoo, load_model, `puresound infer`
     │     └─► puresound.web             local HTTP playground
     └─► sdk/python/puresound_streaming  standalone ORT runtime
```

### 1. Corpus preparation

`puresound.dataset.corpus` turns a corpus on disk into **metafiles**: one row per
utterance, `uttid, spkid, gender, path, length, sample rate, channels`. Anything
else a corpus knows (a noise category, a capture device) goes into a JSONL
inventory beside it, so the metafile never grows a column. Splits are made by
speaker so no talker appears on both sides, and audio is resampled once into a
mirrored tree instead of on every read. Corpus-specific layouts (DNS Challenge,
VCTK-DEMAND, LibriLight, Kaldi lists) are small modules with a
`python -m puresound.dataset.corpus.<name>` entry point. See
[Data preparation](../usage/data_preparation.md).

### 2. Dynamic synthesis

Nothing is pre-mixed. A training row is built when it is asked for:

- `puresound.task.sampler.SpeakerSampler` is the `batch_sampler`. It draws
  speakers from the dataset's metadata and yields **item keys**
  `(speaker, sample_rate[, seed[, seconds[, epoch]]])`: a per-item seed for a
  reproducible validation set, a row length for mixed-length batches, and the
  epoch for a curriculum.
- `puresound.dataset.dynamic_base.DynamicBaseDataset` parses the metafile,
  validates the augmentation blocks, owns the `AudioEffectAugmentor` (noise
  folders, RIR banks or a room simulator) and turns a key into its row length,
  its epoch's knobs and its seed.
- A task dataset (`puresound.task.ns.NoiseSuppressionDataset`,
  `puresound.task.voice_isolation.VoiceIsolationDataset`) implements
  `__getitem__`: choose the speech, place each source in a room, gate who talks
  when (`overlap_gating`), add non-speech sound (`noise_stage`), run the mixture
  through a capture and transmission chain (`device_chain`), and emit the noisy
  input with its clean reference and labels. Session rows (`session_rows`) are a
  further row type; `paired_views` adds a second capture view of the same
  mixture for a consistency loss.
- A collate function stacks the rows into a batch.

Synthesising per item keeps the training distribution a set of recipe knobs
rather than a dataset on disk, which is also what lets a curriculum move those
knobs while a run is going. See [Datasets](dataset/index.md),
[Tasks](task/index.md) and [Data augmentation](../algorithms/augmentation/index.md).

### 3. Models

`puresound.system.siso.EncDecMaskBase` is the Lightning module every
enhancement task trains: waveform → encoder (STFT or learned) → feature
transform (`puresound.nnet.FeatureEncoder`) → backbone → mask
(`puresound.nnet.masker`) → decoder → waveform. The backbone is any model in
`puresound.nnet` (`DPCRN`, `DPARN`, `DPRNN`, `TFGridNet`, `SkiM`, `ConvTasNet`,
the U-Net family), built from reusable blocks in `puresound.nnet.lobe`; losses
live in `puresound.nnet.loss`. `puresound.system.base.BaseLightningModule` owns
what every module shares: the weighted loss list, optimizer and scheduler
plumbing, warm-up, and batched GPU VAD labelling. Inference-time post-processing
(`postprocess`, `onset_guard`, and the offline-only `presence_gate`) lives beside
the modules, so offline evaluation and the streaming export apply the same code.
See [Training systems](system/index.md) and [Models](../algorithms/models/index.md).

### 4. Training driver and recipes

A recipe is one YAML file, loaded by `puresound.config.load_recipe` into a typed
model (see [Configuration system](#configuration-system)). `egs/<task>/main.py`
passes it, with its dataset and collate classes, to
`puresound.system.runner.build_dataloaders`, then calls `runner.run_stages`,
which runs whichever of `--dump_training_samples`, `--training`, `--scoring`
and `--inference` were asked for. The runner builds the sampler and both
dataloaders, the model (`puresound.recipes.init_model_for_task`), the losses,
the optimizer and scheduler, the DDP strategy and the checkpoint callbacks, and
handles warm starts (`--pretrained_ckpt_path`). Keeping this in the library is
what lets two recipes share one training loop instead of two copies that drift.
See [Recipes](../usage/recipes.md).

### 5. Evaluation gate

A checkpoint is released on benchmark records, not on its validation loss.
`puresound.evaluation` holds the protocol: every stage scores the candidate
**and** the unprocessed input (`systems.Passthrough`) and reports the paired
difference with its confidence interval (`statistics`), because an absolute
score on one test set says little on its own. The stages are library tools run
as `python -m puresound.evaluation.tools.<name>`: `preflight` (the checkpoint
loads whole into every recipe the gate uses), `build_eval_set` (freeze a
synthetic set), `reference` (PESQ, STOI, SI-SDR), `noreference` (DNSMOS),
`wer` (word deletions under a recogniser), `rtf` (CPU real-time factor) and
`collect`, which merges the stages into one record (`records`) and states the
gate's verdict. A recipe's `run_full_gate.sh` (`run_full_benchmark.sh` for
voice isolation) is only the list of stages. See
[Evaluation](../usage/evaluation.md) and [Metrics](../algorithms/metrics.md).

### 6. Export

`puresound.streaming` turns a trained offline model into a **per-frame** model
that consumes one STFT frame and carries its recurrent and convolution state as
explicit inputs and outputs, built to reproduce the offline forward frame by
frame. It is traced to ONNX (`export_streaming_dpcrn_onnx`,
`export_streaming_dparn_onnx`); the export runs the graph once under ONNX
Runtime, checks it against the PyTorch frame model, and writes a JSON manifest
beside the graph: STFT geometry, state port names and shapes, the streaming
delay, and the recommended inference settings (`dry_blend`, an onset guard). Post-processing is recorded in the
manifest, not baked into the graph, so a runtime reproduces the evaluated
configuration without a flag having to be remembered. See
[Streaming](../usage/streaming/index.md).

### 7. Deployment

- **Model zoo** — `model_zoo/catalog.yaml` lists the released models (see
  [Model zoo](#model-zoo)).
- **`puresound.inference`** — `ModelZoo` reads the catalog; `load_model(id)`
  resolves an artifact and hands it to the processor the catalog names
  (`stft_frame_ort` for enhancement, `waveform_embedding_ort` for speaker
  embedding). The CLI (`puresound models`, `puresound infer`,
  `puresound providers`, `puresound web`) is a thin layer over it that does not
  load the training stack (Lightning, datasets, models).
- **`puresound.web`** — a standard-library HTTP server and static browser client
  over `puresound.inference`, started with `puresound web`: run models, compare
  them, measure audio, export a comparison. It has no authentication and is meant
  for a local or trusted network. See [Web playground](../usage/web.md).
- **`sdk/python/puresound_streaming`** — `PureSoundStreamingRuntime` runs an
  exported ONNX graph from its manifest, frame by frame, with the manifest's
  post-processing. It re-implements the few pieces it needs rather than
  importing `puresound`, so a product can embed it with only NumPy and ONNX
  Runtime installed; tests pin the duplicated pieces (the onset-guard knobs,
  post-processing) against the library's.

## Package map

| package | responsibility | details |
| --- | --- | --- |
| `puresound.audio` | audio I/O (`AudioIO`), DSP, noise, volume, spectrum, VAD labelers, the `AudioEffectAugmentor`, room simulation; `audio.rir` builds and audits RIR banks offline, and only its bank loader is on the training path | [Audio](../algorithms/audio/index.md) |
| `puresound.dataset` | corpus preparation, metafile parsing, the dynamic-synthesis base dataset | [Datasets](dataset/index.md) |
| `puresound.task` | task datasets, collate functions, the speaker sampler, and the synthesis stages they compose | [Tasks](task/index.md) |
| `puresound.nnet` | models, reusable layers (`lobe`), maskers, feature encoders, losses | [Models](../algorithms/models/index.md), [Losses](../algorithms/losses/index.md) |
| `puresound.system` | Lightning modules, the training driver (`runner`), curriculum callback, MetricGAN critic, inference post-processing | [Training systems](system/index.md) |
| `puresound.config` | typed recipe schemas and `load_recipe` | [Configuration](../usage/configuration.md) |
| `puresound.recipes` | builds the model and loss objects a validated recipe names | [below](#configuration-system) |
| `puresound.evaluation` | benchmark protocols, paired statistics, records, transcribers; stage tools under `evaluation.tools` | [Evaluation](../usage/evaluation.md) |
| `puresound.streaming` | per-frame models, ONNX export and manifest, the in-package ORT runtime (`StreamingOrt`) | [Streaming](../usage/streaming/index.md) |
| `puresound.inference` | catalog schema, `ModelZoo`, providers, processors, `load_model` | [Model zoo](#model-zoo) |
| `puresound.web` | local HTTP playground | [Web](../usage/web.md) |
| `puresound.cli` | the `puresound` command over `puresound.inference`; does not load the training stack | [Web](../usage/web.md) |
| `puresound.metrics`, `puresound.utils`, `puresound.logging_setup` | objective metrics, small shared helpers, the library's logging handler | [Metrics](../algorithms/metrics.md), [Utilities](utils.md) |
| `puresound.third_party` | notes on optional third-party code PureSound can use but does not ship | — |
| `sdk/python` | standalone streaming runtime package `puresound_streaming` | [Streaming](../usage/streaming/index.md) |
| `egs/` | recipe drivers: `noise_suppression`, `voice_isolate`, `speaker_embedding`, `target_speaker_extraction`, and `rir_generation` for building RIR banks | [Recipes](../usage/recipes.md) |

## Configuration system

A recipe is a YAML mapping with three discriminators: `schema_version` (2),
`purpose` (`train` or `inference`) and `task` (`noise_suppression`,
`voice_isolation`, `speaker_embedding`, `target_speaker_extraction`).
`load_recipe(path, expected_task=..., expected_purpose=...)` reads it with the
safe YAML loader and validates it into the task's schema (`TASK_SCHEMAS` in
`puresound.config.recipe`), or into `InferenceRecipe` for a model-only recipe;
any mismatch raises `RecipeConfigError`. A recipe driver passes the task it
expects, so a voice-isolation file handed to the noise-suppression driver fails
at load, not halfway through a run.

- Every config model derives from `StrictConfig`: unknown fields are errors, and
  models are frozen. `with_overrides` makes a validated copy with some fields
  changed.
- Pipeline control flow — the dataset, trainer, optimizer, scheduler,
  augmentation, VAD and curriculum blocks — is fully typed. The `model` block
  and each loss's `args` stay open mappings, validated by the constructors they
  are passed to.
- `delegated_kwargs` forwards only the keys a recipe actually wrote to
  components that own their defaults (the room simulator, RIR bank loaders, VAD
  labelers), so a default is defined in one place.
- A `curriculum` block is checked at load time: every track must name a block, a
  bank member or a loss the recipe has.
- `puresound.recipes` builds the objects: `getattr(puresound.nnet, type)` for the
  encoder and backbone, `getattr(puresound.system, type)` for the Lightning
  module, `getattr(puresound.nnet.loss, type)` for each loss;
  `MODEL_FACTORY_FOR_TASK` picks the single-input or the conditioned (legacy
  target-speaker-extraction) model shape from the task.

Field reference: [Recipe configuration](../usage/configuration.md).

## Model zoo

`model_zoo/catalog.yaml` is **metadata only**: ids, task, lifecycle and roles,
audio contract, parameters and their bounds, and for each artifact its relative
path, manifest, processor name and SHA-256. Weights and ONNX files stay in their
recipe's `pretrained_ckpt/`. The catalog is validated by
`puresound.inference.schema` on load.

- **Ids** are `<task>-<arch>-<version>` and are pinned by users. A renamed id
  keeps its old spelling in the alias table of `ModelZoo.get`.
- **Roles** mark the model to use per task: exactly one runnable `default` per
  task (`ModelZoo.default_model`), with `candidate`, `alternative` and other
  roles beside it.
- **Location** — `ModelZoo.default()` reads the checked-in catalog, or the file
  named by `PURESOUND_MODEL_ZOO`.
- **Validation** — `puresound models validate` checks every path, hash, manifest
  and ONNX input/output name; the test suite validates the checked-in
  catalog the same way.

## Legacy

Target-speaker extraction and the speaker-embedding task dataset are frozen:
`puresound.task.tse`, `puresound.task.sv`, `puresound.system.miso` and
`puresound.dataset.kaldi_base` are kept for their existing recipes and not
developed further. The runner reads a recipe's `test_folder` through
`KaldiFormBaseDataset` for `--scoring` and `--inference`, so that module stays on
the scoring path.

## Detailed pages

| page | covers |
| --- | --- |
| [Datasets](dataset/index.md) | metafile parser, `DynamicBaseDataset`, Kaldi-form dataset |
| [Tasks](task/index.md) | noise-suppression and voice-isolation datasets, sampler, synthesis stages |
| [Training systems](system/index.md) | Lightning modules, optimizer factory, logger |
| [Utilities](utils.md) | `puresound.utils` helpers |
