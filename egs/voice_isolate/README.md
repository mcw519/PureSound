# Voice Isolate

Train a near-field voice isolation model. The foreground speaker is simulated
as a near-field source, while other sampled speakers are placed farther away in
the same room and treated as interference to suppress.

This recipe reuses the noise suppression training loop with:

- source-level shoebox room simulation
- near-field foreground distance range
- far-field interferer distance range
- dynamic Silero VAD labels
- SkiM baseline backbone

## Prepare metadata

The metadata format is the same as the noise suppression recipe.

```bash
uv run python egs/voice_isolate/prepare_metafile.py \
  data/voice_isolate_train \
  /path/to/wav.scp \
  /path/to/utt2spk
```

For DNS-Challenge-4, use the adapter to scan `clean_fullband` and create the
PureSound train/valid metafiles:

```bash
uv run python egs/voice_isolate/prepare_dns_challenge.py \
  /path/to/DNS-Challenge-4 \
  --output-dir ./data/voice_isolate_dns4 \
  --speaker-id-strategy parent \
  --valid-ratio 0.05
```

The adapter writes:

- `voice_isolate_dns4_train.csv`
- `voice_isolate_dns4_valid.csv`
- `voice_isolate_dns4_config_hint.yaml`

Use the generated config hint to update `train_metafile`, `valid_metafile`,
`augmentation_noise.noise_folder`, and `augmentation_reverb.rir_folder` in
`config/skim.yaml`. If your DNS clean speech directory groups speakers one
level above the waveform folders, use `--speaker-id-strategy grandparent`.

## Inspect training samples

```bash
uv run python egs/voice_isolate/main.py \
  egs/voice_isolate/config/skim.yaml \
  --dump_training_samples True
```

The generated dummy samples contain three channels:

1. `noisy_speech`: near-field target plus far-field interferers
2. `clean_speech`: near-field target reference
3. `consistency_noise`: residual noise/interference

## Train

```bash
uv run python egs/voice_isolate/main.py \
  egs/voice_isolate/config/skim.yaml \
  --training True
```

Before training, update `train_metafile`, `valid_metafile`, `test_folder`, and
worker/GPU counts in [config/skim.yaml](config/skim.yaml).
