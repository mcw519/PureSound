# puresound.audio.spectrum

繁體中文版本：[spectrum.zh-TW.md](spectrum.zh-TW.md)

Waveform-level STFT/iSTFT helpers and complex ↔ (magnitude, phase)
conversions. They are thin wrappers over `torch.stft`/`torch.istft` for scripts
and analysis; models use the STFT layers in `puresound.nnet.lobe` instead (see
[lobe/stft](../models/lobe/stft.md)).

## `wav_to_stft(wav, nfft=512, win_size=512, hop_size=128, window_type="hann_window", stft_normalized=False) -> (cpx_stft, stft_info)`

`torch.stft(..., return_complex=True)` with torch's defaults otherwise
(`center=True`, reflect padding, one-sided).

- `wav: [..., L]` → `cpx_stft: [..., nfft // 2 + 1, 1 + L // hop_size]`, complex.
- `window_type` is the *name* of a torch window factory, resolved with
  `getattr(torch, window_type)` (`"hann_window"`, `"hamming_window"`, ...),
  built with length `win_size` on the input's device.
- `stft_normalized=True` passes `normalized=True` to torch, which scales the
  frames by `1/√nfft`.
- `stft_info` is the dict `{nfft, win_size, hop_size, window_type,
  stft_normalized}`: exactly the keyword set of `stft_to_wav`, so
  `stft_to_wav(cpx, **stft_info)` inverts with the same settings. It does not
  record the input length.

## `stft_to_wav(x, nfft=512, win_size=512, hop_size=128, window_type="hann_window", stft_normalized=False) -> wav`

`torch.istft` with the same settings. `x` may be complex or real with a
trailing size-2 axis. The output length is `hop_size · (frames − 1)`, i.e. the
input length rounded down to a multiple of `hop_size` with the defaults above;
it equals the original length only when that is already a multiple. Keep the
original length and trim or pad yourself:

```python
from puresound.audio.spectrum import wav_to_stft, stft_to_wav

cpx, info = wav_to_stft(wav, nfft=512, win_size=512, hop_size=128)
rec = stft_to_wav(cpx, **info)            # len = 128 · (L // 128)
wav_aligned = wav[..., : rec.shape[-1]]
```

## `tensor_as_complex(x) -> Tensor`

Returns `x` if it is already complex; otherwise views a real `[..., 2]` tensor
(real, imaginary on the last axis) as complex with `torch.view_as_complex`,
making it contiguous first if needed. Any other last-axis size raises
`ValueError`.

## `cpx_stft_as_mag_and_phase(x, eps=1e-8) -> (mag, phase)`

```
mag   = sqrt(re² + im² + eps)        # eps=None gives the exact magnitude
phase = atan2(im, re)
```

`eps` keeps the square-root gradient finite at zero magnitude.

## `mag_and_phase_as_cpx_stft(mag, phase) -> Tensor`

`mag · exp(j·phase)`; the two shapes must match.
