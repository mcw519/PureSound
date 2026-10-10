# puresound.nnet.lobe.metric_discriminator

繁體中文版本：[metric_discriminator.zh-TW.md](metric_discriminator.zh-TW.md)

A metric discriminator: a small network that learns to predict PESQ from a
(clean, enhanced) pair, used as a differentiable stand-in for PESQ in
MetricGAN-style training (Fu et al., "MetricGAN: Generative Adversarial
Networks based Black-box Metric Scores Optimization for Speech Enhancement",
ICML 2019; Fu et al., "MetricGAN+: An Improved Version of MetricGAN for Speech
Enhancement", Interspeech 2021). The architecture follows CMGAN's discriminator
(Cao et al., "CMGAN: Conformer-based Metric GAN for Speech Enhancement",
Interspeech 2022), also used by MP-SENet.

Training-time only: its weights are saved in training checkpoints, but
inference and the streaming export never read it.

## Class: `MetricDiscriminator`

```
M(w)  = |STFT(w)|^0.3                                    # Hann window, fp32, autocast off
pair  = stack(M(reference), M(estimate))                 # [B, 2, F, frames]
h     = 4 x [SN-Conv2d 4x4 stride 2 -> InstanceNorm2d -> PReLU]   # ndf, 2ndf, 4ndf, 8ndf channels
v     = AdaptiveMaxPool2d(1)(h)                          # [B, 8ndf]
score = LearnableSigmoid(SN-Linear(PReLU(Dropout0.3(SN-Linear(v)))))   # 8ndf -> 4ndf -> n_outputs
```

`SN` is spectral normalisation. The target for an (clean, enhanced) pair is the
normalised PESQ-WB `(pesq - 1) / 3.5`, clipped to `[0, 1]`; the target for
(clean, clean) is 1.

```python
MetricDiscriminator(
    n_fft: int = 400,      # STFT of the waveforms it reads
    hop: int = 100,
    ndf: int = 16,         # width of the first conv; doubles per layer
    beta: float = 1.2,     # ceiling of the output sigmoid
    n_outputs: int = 1,    # metrics predicted per row
)
```

- `forward(reference [B, T], estimate [B, T]) -> [B]`, or `[B, n_outputs]` when
  `n_outputs > 1`. Both waveforms are trimmed to the shorter length.
- `magnitude(wav [B, T]) -> [B, n_fft // 2 + 1, frames]`, `|STFT|^0.3` in fp32.

With `n_outputs > 1` the last layer predicts several metrics from one shared
trunk, each with its own output unit and sigmoid slope.

## Class: `LearnableSigmoid`

```python
LearnableSigmoid(features: int = 1, beta: float = 1.2)   # beta * sigmoid(slope * x), slope learned
```

## Use in training

`EncDecMaskBase` builds the discriminator when its `metric_gan` block is
enabled, passing `n_fft`, `hop` and `ndf` (`beta` and `n_outputs` keep their
defaults):

```yaml
model:
  lightning_module:
    type: EncDecMaskBase
    module_args:
      metric_gan:
        enabled: true
        weight: 5.0
        warmup_steps: 1000
        lr_factor: 5.0
        pesq_workers: 4
        rows_per_step: 12
        buffer_rows: 256
        d_rows: 12
```

| key | default | meaning |
|---|---|---|
| `enabled` | `false` | build the discriminator and add both terms |
| `weight` | `0.0` | generator-term weight; `0` trains the discriminator alone |
| `warmup_steps` | `1000` | optimizer steps before the generator term switches on |
| `n_fft`, `hop`, `ndf` | `400`, `100`, `16` | discriminator STFT and width |
| `lr_factor` | `1.0` | learning-rate factor of the discriminator's parameter group |
| `pesq_workers` | `4` | PESQ worker processes per rank; `0` scores inline |
| `rows_per_step` | `12` | rows of each batch sent for PESQ scoring |
| `buffer_rows` | `256` | replay-buffer capacity in rows |
| `d_rows` | `12` | replayed rows per discriminator update |
| `sample_rate` | `16000` | sample rate given to PESQ |

Each training step adds two terms to the ordinary loss
(`puresound/system/metric_gan.py`):

```
L_D = (D(c, c) - 1)^2  +  (D(c', ê') - q')^2        # c', ê', q': replayed rows and their PESQ labels
L_G = weight * (D(c, ê) - 1)^2                      # D's parameters frozen; after warmup_steps
```

Target-absent rows (silent clean reference) have no PESQ and are left out of
both terms. See [EncDecMaskBase](../../../architecture/system/siso.md) for the
system side.

## Design notes

- **Compressed magnitude input.** `|X|^0.3` approximates the loudness domain
  PESQ works in.
- **`beta` above 1** lets the output reach the clean-vs-clean target of 1.0
  without driving the logit to infinity (MetricGAN+ uses 1.2).
- **Asynchronous labels.** PESQ runs on the CPU and a batch's worth would
  dominate the step time, so labels come from a background process pool into a
  replay buffer, and the discriminator learns from past enhancements (the
  replay MetricGAN+ keeps on purpose).
- **Each term reaches one side.** The generator term runs with the
  discriminator's parameters frozen, and the discriminator term reads detached
  audio, so one optimizer can step both. The clean-vs-clean term runs every
  step, so every discriminator parameter has a gradient on every rank, as DDP
  requires.
