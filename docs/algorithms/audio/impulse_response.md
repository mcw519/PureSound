# puresound.audio.impulse_response

繁體中文版本：[impulse_response.zh-TW.md](impulse_response.zh-TW.md)

Applying a room impulse response (RIR) to a waveform, plus two channel
manipulations that act on the RIR or the signal: a random second-order
coloration and a direct-arrival smear. `AudioEffectAugmentor.apply_rir` is the
normal entry point (see [augmentation.md](augmentation.md)); the physics of the
RIR axis is in [room acoustics](../augmentation/room_acoustics.md).

## `wav_apply_rir(wav, impaulse, sample_rate, rir_mode="full") -> Tensor`

Convolves `wav: [C, T]` with `impaulse: [C_rir, T_rir]` (both 2-D; the RIR
argument is spelled `impaulse`).

1. **Window.** `rir_mode` selects how much of the RIR is kept, measured from
   the peak index `p` (the index of the largest sample of the flattened RIR):

   | `rir_mode` | RIR kept | use |
   | --- | --- | --- |
   | `"full"` | all of it | the mixture |
   | `"early"` | `h[:, : p + 50 ms]` | a target with early reflections |
   | `"direct"` | `h[:, : p + 6 ms]` | a near-dry target |

   The windowed modes assume a mono RIR whose direct arrival is its largest
   sample.
2. **Peak normalisation.** `h ← h / max|h|` over all channels (skipped below
   `1e-12`). The window always contains the peak, so the three modes apply the
   same scale to a given RIR: an `"early"` target and a `"full"` mixture built
   from one RIR carry the same direct-path level and differ only in reflected
   energy.
3. **Convolve and align.** `fftconvolve(..., mode="full")`, then the output is
   sliced to `[d, d + T)` with `d = argmax|h[0]|`, channel 0's peak. The result
   has the input length and starts at the direct arrival, so the propagation
   delay is removed and mixture and target stay sample-aligned with the dry
   source.

Channel layout: a mono RIR convolves every `wav` channel (`[C, T] → [C, T]`); a
multichannel RIR requires mono `wav` and gives one output per RIR channel
(`[1, T] → [C_rir, T]`), all aligned to channel 0's peak so inter-channel delays
survive.

**Level design.** Normalising by the peak removes the absolute `1/r` level a
bank RIR carries on disk (inter-channel ratios are kept, since one scale is
used for all channels). A reverberant source therefore does not get quieter
with distance; the recipe sets levels explicitly through SNR/SIR, and near and
far differ only by DRR, decay shape and spectral tilt. The voice-isolation
`mix_mode` block builds on this: its `physical` mode sums the post-RIR sources
without rescaling (so the ratio lands near 0 dB), and its `distance_level` mode
puts the level cue back explicitly as `SIR = 20·log10(d_itf / d_fg)` plus
jitter.

## `compute_drr_db(rir, sample_rate, direct_window_ms=2.5) -> float`

Re-export of `puresound.audio.rir.metrics.compute_drr_db` for one channel:

```
DRR = 10·log10( Σ_{n=p}^{p+W−1} h[n]²  /  Σ_{n≥p+W} h[n]² ),   W = round(direct_window_ms · fs / 1000)
```

with `p = argmax|h|`. Energy before `p` is ignored; an empty tail returns
`+inf`. The 2.5 ms window is the one the room simulator, the bank loader and
the DRR-contrast augmentation use, so every DRR in the pipeline is the same
measure (see [RIR metrics](rir_metrics.md)).

## `rand_add_2nd_filter_response(wav, a=None, b=None) -> (wav, a, b)`

A random pole/zero biquad as a cheap model of a transducer's frequency response
(J.-M. Valin, *A Hybrid DSP/Deep Learning Approach to Real-Time Full-Band
Speech Enhancement*, MMSP 2018):

```
H(z) = (1 + b1 z⁻¹ + b2 z⁻²) / (1 + a1 z⁻¹ + a2 z⁻²),   a1, a2, b1, b2 ~ U(−3/8, 3/8)
```

drawn from the torch generator when `a` or `b` is not given. With
`|a1| + |a2| ≤ 3/4 < 1` both poles lie inside the unit circle, so every draw is
stable. Filtering uses `torchaudio.functional.lfilter(..., clamp=False)`: a
frequency response is linear, and the default clamp would clip a hot mixture
but not its quieter target. The coefficients are returned so the same response
can be applied to the paired target.

## `smear_direct_arrival(impaulse, sample_rate, *, smear_ms, generator=None) -> Tensor`

Scrambles the fine timing of the first `smear_ms` after the direct arrival,
per channel, and leaves the rest of the RIR unchanged:

1. Window `h[p : p + n]`, `n = round(smear_ms · fs / 1000)`; no-op when
   `smear_ms ≤ 0` or `n < 2`.
2. Convolve the window causally with a random unit-energy Gaussian kernel
   (`torch.randn`, optional `generator`), so nothing arrives before `p`.
3. Keep the first sample the largest (`1.001 × max`), then rescale the window to
   its original energy.

The peak index is preserved because `wav_apply_rir` reads it for the windows
and the alignment; the window energy is preserved so the manipulation is not a
level change; the late tail is untouched. The augmentor applies it before the
RIR is cached, so the `"full"` mixture and the `"early"` target see the same
smeared impulse. Configured as `augmentation_reverb.direct_smear`
(`used`, `prob`, `smear_ms_range`); off unless a recipe sets it. See
[distance cues](../augmentation/distance_cues.md) for what it removes.

## Example

```python
from puresound.audio.io import AudioIO
from puresound.audio.impulse_response import rand_add_2nd_filter_response, wav_apply_rir

rir, _ = AudioIO.open("rir.wav", resample_to=16000)
clean, sr = AudioIO.open("clean.wav", resample_to=16000)

mixture = wav_apply_rir(clean, rir, sr, rir_mode="full")
target = wav_apply_rir(clean, rir, sr, rir_mode="early")
colored, a, b = rand_add_2nd_filter_response(mixture)
target_colored, _, _ = rand_add_2nd_filter_response(target, a=a, b=b)
```
