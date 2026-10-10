# puresound.audio.noise

繁體中文版本：[noise.zh-TW.md](noise.zh-TW.md)

Two additive-mixing primitives. `add_bg_noise` scales a noise bed so the mixture
reaches a requested SNR; the same operator mixes an interfering talker at an SIR.
`add_bg_white_noise` adds Gaussian noise at an SNR. Both take a **list** of
ratios and return **lists**, one result per ratio.

[`AudioEffectAugmentor`](augmentation.md) wraps both with noise-pool handling
and resampling; the task datasets (`puresound/task/ns.py`,
`voice_isolation.py`) and `puresound.evaluation.tools.mix_paired_set` call
`add_bg_noise` directly when the second signal is already in hand.

## `add_bg_noise(wav, noise, snr_list) -> (noisy_list, noise_list)`

For `wav: [1, L]` (or `[L]`) and a list of noise tensors:

1. Each noise tensor is reduced to its first channel, RMS-normalised, and the
   list is concatenated along time into one bed, which is RMS-normalised again.
   Passing two clips gives one bed made of both, end to end.
2. The bed is fitted to `L`: a random crop (offset from `torch.randint`) if
   longer, tiled and cut if shorter or equal.
3. For each `snr_db`:

   ```
   g        = rms(wav) / 10^(snr_db / 20)
   noisy    = wav + g · bed
   noise    = g · bed
   ```

Returns `(noisy_list, noise_list)`, each of length `len(snr_list)`.

The ratio is a whole-clip energy ratio: `rms(wav)` includes the pauses in
`wav`, and the bed has unit RMS over its full length. It is not an active-speech
level (ITU-T P.56); a clip with long silences reaches a lower speech-active SNR
than the nominal value. When `wav` is a foreground talker and `noise` an
interfering talker, `snr_db` is the SIR.

## `add_bg_white_noise(wav, snr_list) -> (noisy_list, noise_list)`

For each `snr_db`, draws `n ~ N(0, σ²)` of shape `[1, L]` with
`σ = rms(wav) / 10^(snr_db/20)` from the torch generator, and returns
`wav + n` and `n`. Mono input only (`σ` is converted to a Python float). There
is no pool and no file I/O. Coloured noise is not implemented.

## Example

```python
from puresound.audio.noise import add_bg_noise, add_bg_white_noise

(noisy_0db, noisy_10db), _ = add_bg_noise(wav=speech, noise=[noise_wav], snr_list=[0.0, 10.0])
(mix,), (itf_scaled,) = add_bg_noise(wav=near, noise=[far_talker], snr_list=[5.0])   # SIR 5 dB
(noisy_5db,), _ = add_bg_white_noise(wav=speech, snr_list=[5.0])
```
