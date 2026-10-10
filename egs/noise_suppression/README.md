# Noise suppression

Traditional Chinese: [`README.zh-TW.md`](README.zh-TW.md)

Single-channel speech enhancement: (noisy, clean) pairs are synthesised on the fly
from a clean-speech corpus, a noise corpus and a room-impulse-response bank, and a
mask-based encoder / backbone / decoder model learns to remove the noise and
reverberation while keeping every voice. The released models are 16 kHz
streaming DPCRN-Mamba checkpoints with no enrollment.

`main.py` is this recipe's training entry point. Everything downstream of the
dataset -- sampler, dataloaders, CLI, Lightning wiring, DDP, scoring and
inference -- lives in [`puresound/system/runner.py`](../../puresound/system/runner.py)
and is shared with [`egs/voice_isolate`](../voice_isolate/README.md), which has
its own entry point and dataset.

## Files

| File | Purpose |
|---|---|
| `main.py` | training, scoring and inference entry point (`NoiseSuppressionDataset`) |
| `config/train_dpcrn_mamba_s0.yaml`, `_s1.yaml`, `_activebin_ft.yaml`, `_metricgan_ft.yaml` | the released lineage, run in order (below) |
| `config/infer_dpcrn_mamba_wide.yaml` | model-only inference config for v3 |
| `config/infer_dpcrn.yaml` | model-only config that loads v1/v2 checkpoints in `pretrained_ckpt/` (gate, streaming export, one-off enhancement) |
| `config/eval/ns_testset.yaml` | the recipe the gate's frozen synthetic set is built from |
| `config/dpcrn.yaml`, `config/dparn.yaml` | starting templates: DPCRN at 32 kHz, and an offline DPARN at 16 kHz with a `FrequencyEQLayer` front-end. Not released recipes |
| `run_full_gate.sh` | the benchmark stage list, see [`benchmarks/stages.md`](benchmarks/stages.md) |
| `benchmarks/` | stage definitions and scored records, see [`benchmarks/README.md`](benchmarks/README.md) |
| `pretrained_ckpt/` | released checkpoints and their streaming exports, see [`pretrained_ckpt/README.md`](pretrained_ckpt/README.md) |

## 1. Prepare data

Corpus preparation is library code shared by every recipe:
[`docs/usage/data_preparation.md`](../../docs/usage/data_preparation.md). The
released recipes read DNS Challenge speech and noise at 16 kHz:

```bash
# clean speech -> speaker-disjoint train/valid metafiles, converted to 16 kHz once
uv run python -m puresound.dataset.corpus.dns_challenge speech /path/to/audio/dns-5 \
    --output-dir egs/noise_suppression/data/dns5 \
    --subset read_speech --valid-ratio 0.05 \
    --resample-to 16000 \
    --resample-root /path/to/audio/dns-5/datasets_fullband_16k/clean_fullband

# noise -> the folder augmentation_noise.noise_folder points at
uv run python -m puresound.dataset.corpus.dns_challenge noise /path/to/audio/dns-5 \
    --output-dir egs/noise_suppression/data/dns5 \
    --resample-to 16000 \
    --resample-root /path/to/audio/dns-5/datasets_fullband_16k/noise_fullband
```

The recipes also read two views of the pre-generated RIR bank
(`augmentation_reverb.simulator.pregenerated.banks`: a synthetic `wide` view and
a measured-room view), built as in
[`egs/voice_isolate/DATA_SETUP.md`](../voice_isolate/DATA_SETUP.md) §3–4. Point
the absolute paths in the recipe at your copies.

Other speech and noise sources -- VCTK, LibriLight, FSD50K, CochlScene, MUSAN,
speech-shaped noise, targets cleaned through a checkpoint -- and the knobs that
mix them (`trainer.speaker_source_weights`, `augmentation_noise.noise_sources`,
`augmentation_noise.snr_bands`) are covered on the same page.

`dataset.test_folder`, read by `--scoring` and `--inference`, is a different
format: a directory with `wav2scp.txt` (`<uttid> <path>` per line) and, for
`--scoring`, `wav2ref.txt` with the clean references.

## 2. Train

### The released lineage

The released checkpoints come from four recipes run in order, each warm-started
from the previous one's last checkpoint with `--pretrained_ckpt_path`. Model,
optimiser, data and the other losses are identical between stages; what changes
is below. Inside a stage the `curriculum:` block ramps the augmentation
probabilities up from the previous stage's values, so a hand-off is a slope
rather than a cliff.

| Recipe | Warm start | What the stage sets | Produces |
|---|---|---|---|
| `train_dpcrn_mamba_s0.yaml` | from scratch | SNR [0, 20], reverb probability 0.45, 20 epochs | stage 0 |
| `train_dpcrn_mamba_s1.yaml` | s0, epoch 19 | SNR [-5, 15], reverb probability 0.65, 20 epochs | stage 1 |
| `train_dpcrn_mamba_activebin_ft.yaml` | s1, epoch 19 | `+ ActiveBinLogMagLoss`, 2 epochs at LR 1e-4 | `dpcrn_mamba_v1` |
| `train_dpcrn_mamba_metricgan_ft.yaml` | activebin_ft, epoch 1 | `+` a MetricGAN PESQ critic, 4 epochs at LR 1e-4 | `dpcrn_mamba_v2` |

Run from the repository root (the recipes' paths are relative to it):

```bash
uv run python egs/noise_suppression/main.py \
    egs/noise_suppression/config/train_dpcrn_mamba_s0.yaml --training --set_seed 1234

uv run python egs/noise_suppression/main.py \
    egs/noise_suppression/config/train_dpcrn_mamba_s1.yaml --training --set_seed 1234 \
    --pretrained_ckpt_path egs/noise_suppression/exp/ns_dpcrn-mamba_curriculum_s0/lightning_logs/version_0/checkpoints/epoch=19-<step>.ckpt
```

and likewise for the two short fine-tuning stages. Each 20-epoch stage runs one
cosine cycle, so its last epoch (19) is the trough to warm-start from and to
judge. The calibration and MetricGAN stages are short, low-learning-rate passes
by design; folding either into a full stage does not reproduce it.

Training the Mamba temporal path on GPU uses the fused selective-scan kernel of
the `mamba_ssm` package when it is importable and built against the installed
torch; otherwise the model falls back to an exact pure-PyTorch scan, which is
correct but much slower.

### Wider released candidate (v3)

`noise-suppression-dpcrn-mamba-v3` uses a wider model and a separate training
chain: `train_dpcrn_mamba_capW_s0.yaml` → `train_dpcrn_mamba_capWL3_s1.yaml` →
`train_dpcrn_mamba_capWL3_ft.yaml` → `train_dpcrn_mamba_capWL3_mg.yaml`.
Training snapshots, warm-start checkpoints and release measurements are in
[`pretrained_ckpt/README.md`](pretrained_ckpt/README.md#reproduce-v3).
v3 needs `config/infer_dpcrn_mamba_wide.yaml` for export; `infer_dpcrn.yaml`
continues to serve v1/v2. The default remains v2.

### Command-line flags

`config_path` is positional and comes first. The stage flags are plain switches
(`--training`, not `--training True`).

| Flag | Effect |
|---|---|
| `--training` | train |
| `--scoring` | score `dataset.test_folder` (PESQ-WB/NB, STOI, ESTOI, SI-SNR, BSS-SDR, DNSMOS) |
| `--inference` | enhance `dataset.test_folder` into `dataset.proc_output_folder` |
| `--dump_training_samples` | write 3 synthesised batches to `./dummy_samples/` |
| `--ckpt_path PATH` | with `--training`, resume; with `--scoring` / `--inference`, the checkpoint to run |
| `--pretrained_ckpt_path PATH` | with `--training`, warm-start from another checkpoint's weights |
| `--pretrained_allow_reshaped` | with `--pretrained_ckpt_path`, rebuild at init the parameters whose shape changed instead of refusing |
| `--set_seed N` | seed everything |
| `--inference_sr HZ` | run scoring / inference at this sample rate |

`--dump_training_samples` writes one `batch_XX-YY.wav` per row, a three-channel
file stacking `[noisy, clean, noisy - clean]`; open it in an editor that shows
channels separately to hear the augmentation before committing to a run.

### Resume, warm start, reload

`--ckpt_path` and `--pretrained_ckpt_path` both take a `.ckpt`, and they are
three different mechanisms:

| Flag | Stage | Restores |
|---|---|---|
| `--ckpt_path` | `--training` | Lightning's own resume: model, optimiser, scheduler, epoch and callback state. Continue the **same** run under the **same** config |
| `--pretrained_ckpt_path` | `--training` | weights only; optimiser, scheduler and losses start fresh from the current config. The warm start every lineage stage uses |
| `--ckpt_path` | `--scoring` / `--inference` | weights only, copied by name into the freshly built model |

A warm start is **non-strict on names and strict on shapes**. Parameters the
current model has and the checkpoint does not -- a new head, a critic -- stay at
their initialisation; parameters the checkpoint has and the model does not are
ignored; both are logged (`[pretrained] N new param(s) kept at init: ...`,
`[pretrained] N ckpt param(s) ignored: ...`). A parameter whose **shape** changed
means the checkpoint is not the model it is being loaded into, and the load
refuses. The one deliberate case is an STFT window change, where the fixed STFT
kernels, band matrices and input normalisation follow the frequency length while
every conv and RNN keeps its shape: pass `--pretrained_allow_reshaped` and those
parameters are rebuilt at init (and logged as `[pretrained] N param(s) rebuilt at
init`).

For scoring and inference the reload reports a checkpoint parameter the model
does not have (`... is not in the model.`) and a model parameter the checkpoint
did not supply (`Needed param name but missing: [...]`); both run on a single
process with `L.Trainer(inference_mode=True)`.

### Training setup

- **Backends.** `runner.configure_torch_backends()` turns on cuDNN autotuning and
  TF32 at import. Training rows have a fixed length per batch, so autotuning pays
  once per shape.
- **DDP.** `trainer.num_gpus > 1` selects `DDPStrategy` with
  `gradient_as_bucket_view=True`; otherwise Lightning's `auto`. `n_spk_per_batch`
  is per device. `trainer.find_unused_parameters` (default false) must be true for
  a model with parameter groups some batches never touch -- an auxiliary head read
  only on some rows; every voice-isolation recipe sets it. Batching is the
  recipe's own `SpeakerSampler`, so Lightning's distributed sampler is off, and
  `sync_batchnorm` is on.
- **Precision.** `trainer.lightning_trainer_args` is passed to `lightning.Trainer`
  as is, so it takes any Trainer argument. The default is full precision; add
  `precision: bf16-mixed` to trade a little accuracy for speed and memory. bf16
  rather than fp16, because the complex-spectral magnitude and division ops need
  fp32 range and no gradient scaler.
- **VAD labels.** `vad_label.backend: energy` (the default) labels inside the
  dataloader workers. `backend: silero` is lifted out of the workers and run
  batched on GPU after the batch reaches the device; it needs the `silero-vad`
  package, and the labeler is kept out of `state_dict()` so it never lands in a
  checkpoint.

## 3. The gate

Benchmarking is one script, and it runs before there is a model:

```bash
bash egs/noise_suppression/run_full_gate.sh baseline
bash egs/noise_suppression/run_full_gate.sh <tag> <ckpt> <recipe>
```

How to build its sets, its variables and how collection decides:
[`docs/usage/evaluation.md`](../../docs/usage/evaluation.md). What each stage can
decide: [`benchmarks/stages.md`](benchmarks/stages.md). The result is a record
under `benchmarks/records/` and a non-zero exit when a gate stage failed or could
not resolve.

## 4. Use a released model

```bash
# through the model zoo (ONNX, streaming runtime)
uv run puresound infer noise-suppression-dpcrn-mamba-v2 \
    --input audio=in.wav --output audio=out.wav --provider auto

# or the streaming export directly
uv run python egs/voice_isolate/scripts/streaming_onnx.py infer \
    egs/noise_suppression/pretrained_ckpt/streaming/dpcrn_mamba_v2.onnx in.wav out.wav
```

To re-export a checkpoint for streaming, pass `--dry-blend 1.0` -- the operating
point every noise-suppression record was scored at; the exporter's default 0.9 is
the voice-isolation setting:

```bash
uv run python egs/voice_isolate/scripts/streaming_onnx.py export \
    egs/noise_suppression/config/infer_dpcrn.yaml \
    egs/noise_suppression/pretrained_ckpt/dpcrn_mamba_v2.ckpt \
    /path/to/dpcrn_mamba_v2.onnx --dry-blend 1.0
```

See [`docs/usage/streaming/dpcrn_onnx.md`](../../docs/usage/streaming/dpcrn_onnx.md).
Which checkpoint to choose: [`pretrained_ckpt/README.md`](pretrained_ckpt/README.md).

## This recipe or `voice_isolate`

The two recipes are separate entry points over one synthesis pipeline.
`VoiceIsolationDataset` subclasses `NoiseSuppressionDataset` and overrides its
row-type hooks; it keeps the near speaker and suppresses far ones, where this
recipe keeps every voice. The top-level `task:` names which recipe a config
belongs to, and each entry point loads only its own task, so a
`voice_isolation` config fails here before any data loads. The voice-isolation
blocks (`augmentation_realfar`, `augmentation_realnear`,
`augmentation_session_rows`, `augmentation_speech.mix_mode`) are unknown fields
in a noise-suppression recipe.

## Library reference

| Document | Covers |
|---|---|
| [`docs/architecture/task/ns.md`](../../docs/architecture/task/ns.md) | `NoiseSuppressionDataset` and the synthesis skeleton this recipe drives, with every `augmentation_*` block |
| [`docs/architecture/task/voice_isolation.md`](../../docs/architecture/task/voice_isolation.md) | the voice-isolation specialisation |
| [`docs/architecture/system/siso.md`](../../docs/architecture/system/siso.md) | `EncDecMaskBase`, the Lightning module the configs use, and the encoder → features → backbone → mask → decoder path |
| [`docs/usage/configuration.md`](../../docs/usage/configuration.md) | recipe validation, sampler knobs and the `curriculum` block |
