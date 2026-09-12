# Target Speaker Extraction (TSE)

> **Status: frozen legacy.** No new features, no rewrites — kept working for reference only.

繁體中文版本：[`README.zh-TW.md`](README.zh-TW.md)

Target speaker extraction is **enrollment-based**: given a noisy mixture and a separate clean
enrollment utterance from the target speaker, the model conditions on a speaker embedding derived
from the enrollment audio (MISO trainer `EncDecCondMaskBase`) to extract that speaker from the
mixture. This is a different cue from the near-field `egs/voice_isolate` recipe used elsewhere in
this repo, which is enrollment-free and relies only on the near/far DRR (direct-to-reverberant
ratio) contrast.

## Train / inference

```bash
# Prepare metadata
python prepare_metafile.py --help

# Train
python main.py --training config/default_config.yaml

# Inference
python main.py --inference --ckpt_path ckpt_path config/default_config.yaml
```

`--ckpt_path` on `--training` resumes a run (restores optimizer/scheduler/epoch); use
`--pretrained_ckpt_path` instead to warm-start from a checkpoint's weights only.

## Full reference

This README only covers the minimal commands for this recipe. For the dataset (mixture
construction, enrollment sampling, augmentation) and the training system (MISO architecture,
per-component learning-rate scaling), see:

- [`docs/task/tse.md`](../../docs/task/tse.md) — `TargetSpeakerExtractDataset`
- [`docs/system/miso.md`](../../docs/system/miso.md) — `EncDecCondMaskBase`
