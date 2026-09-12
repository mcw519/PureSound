# puresound.audio.noise

繁體中文版本：[`noise.zh-TW.md`](noise.zh-TW.md)

Two low-level noise-mixing primitives. [`augmentation.AudioEffectAugmentor`](augmentation.md)
wraps both with pool management, resampling, and id bookkeeping
(`AudioEffectAugmentor.add_bg_noise`/`add_bg_white_noise`); `puresound/task/*.py`
also calls these module-level functions directly when the noise waveform is
already in hand. Both process **a list of target SNRs in one call** and
return **lists** of results, one per SNR — there is no single-SNR,
single-tensor-return overload.

## Functions

### `add_bg_noise(wav: Tensor, noise: List[Tensor], snr_list: List[float]) -> Tuple[List[Tensor], List[Tensor]]`

Mixes one or more noise waveforms into `wav` at every SNR in `snr_list`:

```
noisy = wav + scale * noise      # scale solved per snr_db so that SNR holds
```

Algorithm:
1. Every tensor in `noise` is forced to 1 channel (`n[0].view(1, -1)` if it
   isn't already) and RMS-normalized individually, then all of them are
   **concatenated along time** into a single bed and RMS-normalized again as
   a whole. Passing `len(noise) > 1` is exactly what
   `AudioEffectAugmentor.add_bg_noise(..., dynamic_type=True)` does under
   the hood — 2 clips end-to-end as one noise bed instead of 1.
2. The bed is cropped at a random offset if longer than `wav`, or tiled
   (`repeat` then cropped) if shorter — the output always matches `wav`'s
   length exactly.
3. For each `snr_db` in `snr_list`: `bg_rms = rms(wav) / 10**(snr_db / 20)`
   (`rms(wav)` reduces only the last axis — see
   [`volume.calculate_rms`](volume.md), so a multi-channel `wav` gets a
   per-channel `bg_rms`, broadcast against the single-channel noise bed);
   appends `wav + bg_rms * bed` to the noisy-speech list and `bg_rms * bed`
   (the actual noise added, at that SNR) to the noise list.

Returns `(noisy_speech_list, noise_list)`, **both length `len(snr_list)`** —
not tensors shaped like `wav`.

---

### `add_bg_white_noise(wav: Tensor, snr_list: List[float]) -> Tuple[List[Tensor], List[Tensor]]`

Same per-SNR-list / list-output contract as above, but the "noise" is
zero-mean Gaussian drawn fresh every call
(`torch.FloatTensor(wav.shape[-1]).normal_(0, std)`, with `std` solved from
`snr_db` the same way as above) — no noise pool, no file I/O. Useful as a
cheap noise source when a real recorded bed isn't needed.

Colored (non-white) noise generation is listed as a `# TODO` in the source
and is not implemented.

## Example

```python
from puresound.audio.noise import add_bg_noise, add_bg_white_noise

noisy_list, added_noise_list = add_bg_noise(wav=speech, noise=[noise_wav], snr_list=[0.0, 10.0])
noisy_0db, noisy_10db = noisy_list

white_noisy_list, _ = add_bg_white_noise(wav=speech, snr_list=[5.0])
noisy_5db = white_noisy_list[0]
```
