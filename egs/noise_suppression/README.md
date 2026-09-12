# Noise Suppression

Generic single-channel speech-enhancement recipe: on-the-fly synthesis of (noisy, clean) pairs
from a clean-speech corpus, trained against a mask-based encoder/backbone/decoder model. This is
the most active shared entry point in the repo — `main.py` also *is* the training entry point for
the separate `egs/voice_isolate` product line (near-field foreground-voice isolation), selected
purely through a `dataset.task` config switch — see "The `dataset.task` switch" further down.

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

## Files in this recipe

| file | purpose |
|---|---|
| `prepare_metafile.py` | Converts Kaldi-style `wav.scp` / `utt2spk` (+ optional `utt2gender`) into the CSV metafile the dataset classes read. |
| `main.py` | Training / scoring / inference entry point. Also the real entry point behind `egs/voice_isolate/main.py` (see below). |
| `config/dpcrn.yaml` | Example config: DPCRN backbone, 32 kHz, complex ratio mask. |
| `config/dparn.yaml` | Example config: DPARN backbone, 16 kHz (VCTK+DEMAND paths), adds a `FrequencyEQLayer` front-end. |

There is no `demo.py` in this recipe.

## 1. Prepare the metafiles

`prepare_metafile.py` reads three Kaldi-style text files (`<key> <value...>`, space-separated by
default) and writes one CSV metafile with a fixed header:

```
uttid, spkid, gender, path, length, sample rate, channels
```

```bash
uv run python egs/noise_suppression/prepare_metafile.py \
    data/train_metafile.csv \
    data/train_wav2scp.txt \
    data/train_utt2spk.txt \
    --utt2gender_path data/train_utt2gender.txt \
    --insert_root_path /corpus/root \
    --separator " "
```

| argument | required | default | meaning |
|---|---|---|---|
| `output_path` | yes (positional) | — | path to write the CSV metafile |
| `wav2scp_path` | yes (positional) | — | `<uttid> <wav_path>` per line |
| `utt2spk_path` | yes (positional) | — | `<uttid> <spk_id>` per line |
| `--utt2gender_path` | no | `None` | `<uttid> <gender>` per line |
| `--separator` | no | `" "` | column separator for the **input** files only — the output CSV is always comma-separated |
| `--insert_root_path` | no | `None` | prefix prepended to every path read from `wav2scp`; omit it if `wav2scp` already has absolute/usable paths |

Notes:
- Run this **twice** — once for your train split, once for your valid split — and point
  `dataset.train_metafile` / `dataset.valid_metafile` at the two output CSVs.
- If `--utt2gender_path` is omitted, every row gets a literal `None` in the `gender` column.
  Downstream (`DynamicBaseDataset`) that is bucketed as an "unknown gender" speaker rather than
  rejected — gender only drives gender-balanced sampling, it is not a hard requirement.
- `dataset.test_folder` (used by `--scoring` / `--inference`, see below) is a **different format**
  entirely: a directory read by `KaldiFormBaseDataset`, not a metafile produced by this script. It
  needs `wav2scp.txt` inside it (plus `wav2ref.txt` for `--scoring`), in the same `<uttid> <path>`
  form `prepare_metafile.py` reads as *input* — do not point it at this script's CSV output.

## 2. Train

```bash
# from scratch
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --training

# resume an interrupted run (restores model + optimizer + scheduler + epoch)
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --training \
    --ckpt_path exp/work/lightning_logs/version_0/checkpoints/epoch=12-step=13000.ckpt

# warm-start a new run from another checkpoint's weights (fresh optimizer/scheduler/loss)
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --training \
    --pretrained_ckpt_path some_other_run/epoch=49.ckpt

# sanity-check the augmentation pipeline before a long run (writes ./dummy_samples/*.wav)
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --dump_training_samples

# fix the random seed
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml --training --set_seed 1234
```

`config_path` is a positional argument (first, before any flags). Unlike
`egs/speaker_embedding` and `egs/target_speaker_extraction` (which take `--training True` /
`type=str2bool`), **this recipe's `--training`, `--scoring`, `--inference` and
`--dump_training_samples` are plain `argparse` `store_true` flags** — just write `--training`,
not `--training True` (the latter fails: `True` has nothing left to bind to and argparse raises
`unrecognized arguments`).

`--dump_training_samples` writes 3 batches under `./dummy_samples/`; each
`batch_XX-YY.wav` is a **3-channel** file stacking `[noisy_speech, clean_speech,
consistency_noise]` (`noisy - clean`) for that one item — load it in an editor that shows
per-channel waveforms (e.g. Audacity) to eyeball the augmentation before committing to a full run.

Two example configs ship in `config/`; either is a starting point — copy it, point `dataset.*`
at your own metafiles, and adjust the `augmentation_*` blocks:

| | `config/dpcrn.yaml` | `config/dparn.yaml` |
|---|---|---|
| backbone | DPCRN | DPARN (+ `FrequencyEQLayer` front-end) |
| sample rate | 32000 | 16000 |
| mask | complex | complex |
| `vad_label` | `silero`, frame 800 / hop 320 | `silero`, frame 400 / hop 160 |
| example corpus | (paths left generic) | VCTK (speech) + DEMAND (noise) |

Not every schema block is exercised by these two examples: `augmentation_codec`,
`augmentation_packet_loss`, `augmentation_target_absent`, `augmentation_realfar` and
`augmentation_realnear` are all real, supported blocks that neither config turns on. Full block
list and semantics: [`docs/task/ns.md`](../../docs/task/ns.md) and
[`docs/task/voice_isolation.md`](../../docs/task/voice_isolation.md).

## 3. Inference and scoring

Both stages read `dataset.test_folder` as a Kaldi-style folder (`KaldiFormBaseDataset`):
`wav2scp.txt` is required; `--scoring` additionally needs `wav2ref.txt` (clean reference) to
compute metrics against; `--inference` does not read a reference at all.

```bash
# enhance audio only -> written under dataset.proc_output_folder
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml \
    --inference --ckpt_path exp/work/.../epoch=49.ckpt

# compute metrics (pesq_wb/nb, stoi, estoi, sisnr, bss_sdr, dnsmos_p835)
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml \
    --scoring --ckpt_path exp/work/.../epoch=49.ckpt

# override the sample rate everything runs at (e.g. ckpt trained at 32k, test set is 16k)
uv run python egs/noise_suppression/main.py egs/noise_suppression/config/dpcrn.yaml \
    --inference --ckpt_path exp/work/.../epoch=49.ckpt --inference_sr 16000
```

`--ckpt_path` is functionally required for both stages (it is passed straight to `torch.load`).
Both stages build a bare `L.Trainer(inference_mode=True)` — the DDP strategy, `precision`, and
every other `trainer.lightning_trainer_args` key from the "DDP, precision and performance flags"
section below apply to `--training` **only**; `--scoring`/`--inference` never read
`trainer.num_gpus` and always run single-process.

## `dataset.task`: this recipe or `voice_isolate`

The two recipes are separate entry points. This one trains `NoiseSuppressionDataset`;
`egs/voice_isolate/main.py` trains `VoiceIsolationDataset` and accepts two row types this recipe
has no use for. Neither imports the other — everything they share (sampler, dataloaders, CLI,
Lightning wiring, DDP strategy, scoring and inference stages) lives in
[`puresound/system/runner.py`](../../puresound/system/runner.py), so the two can diverge without
either carrying a copy of the driver.

`dataset.task` names which recipe a config belongs to, and each entry point refuses configs that
are not its own rather than silently training the wrong distribution:

| `dataset.task` | entry point | dataset class | collate class | extra kwargs |
|---|---|---|---|---|
| `noise_suppression` (default when the key is absent — both example configs here omit it) | `egs/noise_suppression/main.py` | `NoiseSuppressionDataset` | `NoiseSuppressionCollateFunc` | — |
| `voice_isolation` | `egs/voice_isolate/main.py` | `VoiceIsolationDataset` | `VoiceIsolationCollateFunc` | `augmentation_realfar_args`, `augmentation_realnear_args` |

Both classes live in `puresound.task.ns` / `puresound.task.voice_isolation`; `VoiceIsolationDataset`
subclasses the generic one and only overrides its row-type hooks (`_plan_row`,
`_prepare_foreground`, `_sample_interferers`, ...) — there is exactly one synthesis pipeline
underneath, not two. Full detail: [`docs/task/ns.md`](../../docs/task/ns.md) (the shared
skeleton) and [`docs/task/voice_isolation.md`](../../docs/task/voice_isolation.md) (the
specialization).

Two guard rails enforce the pairing, both raised before any data loads:

- a config whose `dataset.task` is not `noise_suppression` → exits pointing at the recipe that
  owns it.
- `augmentation_realfar.used: True` or `augmentation_realnear.used: True` → exits, because those
  two blocks only mean something for the voice-isolation row types.

To turn a `noise_suppression` config into a `voice_isolation` one:

```yaml
dataset:
  task: voice_isolation

augmentation_realfar:
  used: True
  prob: 0.20
  lone_far_prob: 0.15          # share of rows with no near speaker at all (target = silence)
  pool_manifest: data/realfar_pool/voices.train.jsonl
  turn_taking_prob: 0.3

augmentation_realnear:
  used: True
  prob: 0.15
  turn_taking_prob: 0.5
  pool_manifest: data/realfar_pool/voices.near.train.jsonl
```

`augmentation_realfar` inserts **finished loudspeaker→air→mic recordings** as far-field
interferers (no RIR applied on top — a convolved channel would only add back the LTI part of a
capture chain that's already in the recording); `augmentation_realnear` swaps the foreground for a
**genuine close-mic recording** so the near/keep side of the mixture is real too. Pool manifests
are one JSON object per line, built by `egs/voice_isolate/scripts/build_real_recording_pool.py`.
The maintained reference recipe is `egs/voice_isolate/config/train_dpcrn.yaml`; any change made
here to `init_dataloader`, the DDP/precision setup, the VAD labeler wiring, or the
checkpoint-loading logic applies to both product lines at once.

## DDP, precision and performance flags

Set unconditionally at import time, before anything else runs:

```python
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")
```

Training uses a fixed sample length (`dataset.training_length_seconds`), so every step sees
identical tensor shapes — `cudnn.benchmark` can safely autotune the fastest conv algorithm once
and reuse it (the default heuristic picks a pathologically slow `dgrad` algorithm for the dilated
encoder convs; ~3.6x slower per step end to end without this). TF32 gives the attention/linear
matmuls tensor-core throughput at negligible precision cost on Ampere+.

### Strategy: DDP vs auto

```python
strategy = (
    DDPStrategy(
        gradient_as_bucket_view=True,
        find_unused_parameters=trainer_dict.get("find_unused_parameters", False),
    )
    if trainer_dict["num_gpus"] > 1
    else "auto"
)
```

- `trainer.num_gpus > 1` → `DDPStrategy`; `<= 1` → Lightning's `"auto"` (single device or CPU).
- `gradient_as_bucket_view=True` is always on with DDP: it makes all-reduce read gradients in
  place from the bucket, which removes a "grad strides do not match bucket view strides" warning
  (triggered by cuDNN's 1x1-conv weight-grad layout) and lowers memory.
- `find_unused_parameters` is config-driven via `trainer.find_unused_parameters` (default
  `False`). Neither `config/dpcrn.yaml` nor `config/dparn.yaml` sets it — plain noise-suppression
  models touch every parameter on every step. Turn it on when the model has **data-dependent**
  parameter groups that some batches never exercise — e.g. a cold-started VAD gate head, or a
  distance auxiliary head only computed on rows that carry that label. Every `egs/voice_isolate`
  training config sets `trainer.find_unused_parameters: true` for exactly this reason. Leaving it
  `False` when it's needed raises a DDP error; leaving it `True` when it isn't costs a per-step
  autograd-graph walk for nothing.

### Precision

`trainer.lightning_trainer_args` is splatted directly into `lightning.Trainer(**trainer_dict["lightning_trainer_args"], ...)`,
so it accepts **any** Lightning `Trainer` kwarg, not just the ones the example configs use
(`max_epochs`, `gradient_clip_val`, `accumulate_grad_batches`). Precision defaults to Lightning's
full-precision `32-true`. To trade a little accuracy for speed/memory:

```yaml
trainer:
  lightning_trainer_args:
    precision: bf16-mixed
```

(`egs/voice_isolate/config/train_dpcrn.yaml` does this.) bf16 rather than fp16 is preferred here
because the complex-spectral magnitude/division ops in the mask/loss path need fp32 range and no
`GradScaler`; measured ~1.57x faster steps and ~40% less activation memory on Ampere when enabled.

Two more `Trainer` args are hardcoded, not config-driven: `sync_batchnorm=True` and
`use_distributed_sampler=False` (batching is already handled by the custom `SpeakerSampler`, so
Lightning must not additionally wrap it in a distributed sampler).

## GPU-side Silero VAD labeling

```yaml
vad_label:
  used: True
  backend: silero
  args:
    frame_length: 800     # samples per label frame -- match the encoder's hop_length
    hop_length: 320
    threshold: 0.5
    min_overlap: 0.5
    model_sample_rate: 16000
```

Both example configs turn this on (`dpcrn.yaml` at 800/320, `dparn.yaml` at 400/160 — each sized
to its own encoder's `hop_length`). Two backends exist:

- `backend: energy` (the default when `vad_label.backend` is omitted) — labeled directly inside
  the dataset, on CPU, in the DataLoader worker process.
- `backend: silero` — a real neural VAD, too heavy to run per-sample inside a CPU worker, so it is
  lifted out of the dataset entirely and run **batched, on GPU**, once the batch has already been
  moved to device:

```python
if vad_label_dict.get("backend", "energy").lower() == "silero":
    from puresound.audio.vad import BatchedSileroVADLabeler
    lightning_model.register_gpu_vad_labeler(
        BatchedSileroVADLabeler(**vad_label_dict.get("args", {}))
    )
```

`register_gpu_vad_labeler` (`puresound/system/base.py`) stores the labeler wrapped in a plain
Python list (`self._gpu_vad_labeler = [labeler]`) specifically so `nn.Module` does **not**
register the Silero TorchScript model as a submodule — it must never land in `state_dict()` /
checkpoints, it's a labeling tool, not trained weights. Lightning's `on_after_batch_transfer` hook
then calls `ensure_vad_targets(batch)` every step: if the batch still carries a raw
`vad_reference` (or `background_vad_reference`) waveform — the dataset defers labeling instead of
computing it itself whenever this backend is active — the labeler runs once over the whole batch
and materializes `vad_target` (/ `background_vad_target`). This needs the optional `silero-vad`
package (`uv pip install silero-vad`); without it you get a clear `RuntimeError` telling you to
install it or fall back to `backend: energy`. The resulting labels feed `VADActivityLoss` (present
in both example configs' `loss_func` list) and, in `voice_isolation` configs, the background-VAD
head.

## Checkpoint loading: three mechanisms behind two flags

`--ckpt_path` and `--pretrained_ckpt_path` both take a path to a `.ckpt` file, but they are not
interchangeable — and `--ckpt_path` itself means something different depending on which stage flag
is set alongside it:

| flag | stage | mechanism | restores |
|---|---|---|---|
| `--ckpt_path` | `--training` | `trainer.fit(..., ckpt_path=...)` (Lightning-native resume) | model + optimizer + scheduler + epoch/global step + callback states |
| `--pretrained_ckpt_path` | `--training` | `lightning_model.load_state_dict(state_dict, strict=False)` | only name-matching weights; fresh optimizer/scheduler/loss from the current config |
| `--ckpt_path` | `--scoring` / `--inference` | `lightning_model.reload_checkpoint(state_dict)` | only name-matching weights, copied key by key; nothing else (there's no optimizer at inference time) |

### Warm start (`--pretrained_ckpt_path`, training only)

```python
state_dict = torch.load(args.pretrained_ckpt_path, map_location="cpu")["state_dict"]
missing, unexpected = lightning_model.load_state_dict(state_dict, strict=False)
```

`strict=False` means shape/name mismatches don't crash the load: parameters the **current** model
has but the checkpoint doesn't are silently kept at their random init (`missing`); parameters the
checkpoint has but the current model doesn't are silently dropped (`unexpected`). `main.py` prints
each count (and a short sample of the names) whenever it is non-empty, so a config problem isn't
silent — e.g. warm-starting into a config that only *added* a head prints just:

```
[pretrained] 2 new param(s) kept at init: ['backbone.vad_head.weight', 'backbone.vad_head.bias']
```

(the "ckpt param(s) ignored" line is skipped entirely when `unexpected` is empty; it only appears
if the checkpoint also contains params the current model no longer has, e.g. a renamed/removed
head).

Use this when the current config's model has **grown** relative to the checkpoint you're starting
from — e.g. warm-starting into a config that adds a new auxiliary head (VAD gate head, distance
head, ...) which the source checkpoint never had — or, more generally, whenever you want the old
weights purely as an initialization prior for a differently-configured run (new loss weights, new
optimizer/scheduler, a new curriculum stage). This is exactly how `egs/voice_isolate`'s staged
training pipeline works: each stage's config is `--pretrained_ckpt_path`-warm-started from the
previous stage's checkpoint while the loss/augmentation config changes underneath it.

### Resume (`--ckpt_path` during `--training`)

Passed straight to `trainer.fit(..., ckpt_path=...)` — Lightning's own mechanism for continuing
the **same** run under the **same** config: same optimizer state, same scheduler position, same
epoch count. Use it to continue an interrupted run, not to change the recipe underneath it.

### Reload for scoring/inference (`--ckpt_path` during `--scoring`/`--inference`)

```python
state_dict = torch.load(args.ckpt_path, map_location="cpu")["state_dict"]
lightning_model.reload_checkpoint(state_dict)
```

A third, distinct mechanism (`BaseLightningModule.reload_checkpoint`, not `nn.Module.load_state_dict`):
it copies each loaded parameter into the freshly-constructed model by name, printing
`"{name} is not in the model."` for any it can't place, then printing
`"Needed param name but missing: [...]"` for any current-model parameter the checkpoint never
supplied. There is no optimizer/scheduler involved — `--scoring`/`--inference` build a bare
`L.Trainer(inference_mode=True)`.

## Library reference

| doc | covers |
|---|---|
| [`docs/task/ns.md`](../../docs/task/ns.md) | `NoiseSuppressionDataset` / `NoiseSuppressionCollateFunc` / `RowPlan` — the synthesis skeleton this recipe drives, and every `augmentation_*` block's meaning. |
| [`docs/task/voice_isolation.md`](../../docs/task/voice_isolation.md) | `VoiceIsolationDataset` / `VoiceIsolationCollateFunc` — the `voice_isolation` task specialization, `augmentation_realfar`/`augmentation_realnear` schema, emitted per-sample labels. |
| [`docs/system/siso.md`](../../docs/system/siso.md) | `EncDecMaskBase` / `EncPredClassBase` — the Lightning module types both example configs use (`model.lightning_module.type`), and the encoder → features → backbone → mask → decoder architecture. |
