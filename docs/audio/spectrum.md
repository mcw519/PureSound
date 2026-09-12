# puresound.audio.spectrum

繁體中文版本：[`spectrum.zh-TW.md`](spectrum.zh-TW.md)

STFT/iSTFT helpers and complex ↔ (magnitude, phase) conversions. `wav_to_stft`
and `stft_to_wav` share one FFT/window parameter naming convention —
`nfft`/`win_size`/`hop_size`/`window_type`/`stft_normalized` — and
`wav_to_stft`'s second return value is designed to be splatted straight into
`stft_to_wav` (`stft_to_wav(x=cpx, **stft_info)`; see the example).

## Functions

### `tensor_as_complex(x: Tensor) -> Tensor`

Converts a **real tensor whose last dimension has length 2**
(`[..., 2]`, real and imaginary packed as the final axis) into a complex
tensor `[...]`, via `torch.view_as_complex` (which needs the last dim
contiguous — `.contiguous()` is applied first if it isn't). If `x` is
already complex, it's returned unchanged. Raises `ValueError` if the last
dimension isn't 2.

This is **not** a 2-argument `(real, imag)` combiner despite the name —
pack real/imag into one tensor's trailing axis yourself first if you have
them as separate tensors, or use `mag_and_phase_as_cpx_stft` if you have
magnitude/phase instead.

---

### `cpx_stft_as_mag_and_phase(x: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]`

`x` may already be complex, or real with a trailing size-2 axis (routed
through `tensor_as_complex` first). Returns `(magnitude, phase)`, both real,
same shape as the complex tensor:

```python
mag = sqrt(real**2 + imag**2 + eps)   # eps=None skips the epsilon (exact magnitude, can be exactly 0)
phase = atan2(imag, real)
```

---

### `mag_and_phase_as_cpx_stft(mag: Tensor, phase: Tensor) -> Tensor`

Inverse of the above: `mag * exp(1j * phase)`. Requires `mag.shape == phase.shape`.

---

### `wav_to_stft(wav: Tensor, nfft: int = 512, win_size: int = 512, hop_size: int = 128, window_type: str = "hann_window", stft_normalized: bool = False) -> Tuple[Tensor, Dict]`

Thin wrapper over `torch.stft(..., return_complex=True)`. `window_type` is a
**string** resolved via `getattr(torch, window_type)` (e.g. `"hann_window"`
→ `torch.hann_window`, `"hamming_window"` → `torch.hamming_window`) — not a
window tensor you construct yourself; it must name a real `torch.*_window`
factory.

Input `wav: [..., L]` (typically `[N, L]` or `[L]`); output
`cpx_stft: [..., nfft // 2 + 1, T]` — a one-sided real-input STFT, so the
frequency-bin count is `nfft // 2 + 1`, **not** `nfft`.

Also returns `stft_info`, a dict that is **exactly** the kwarg set
`stft_to_wav` needs:

```python
{"nfft": nfft, "win_size": win_size, "hop_size": hop_size,
 "window_type": window_type, "stft_normalized": stft_normalized}
```

There is **no `original_length` key** — this function does not remember
the input length for you (see `stft_to_wav` below for why that matters).

---

### `stft_to_wav(x: Tensor, nfft: int = 512, win_size: int = 512, hop_size: int = 128, window_type: str = "hann_window", stft_normalized: bool = False) -> Tensor`

`torch.istft` inverse of `wav_to_stft`. `x` is routed through
`tensor_as_complex` first, so a real `[..., F, T, 2]` tensor works too, not
just an already-complex one.

**No length restoration, and no `original_length` parameter.**
`torch.istft`'s output length is a function of `nfft`/`hop_size`/`win_size`/
frame count and does **not**, in general, equal the original waveform's
length — verified empirically: reconstructing a length-16001 input at
`nfft=1024, win_size=512, hop_size=160` comes back at length 16000; only
lengths that land exactly on the hop/frame grid round-trip unchanged. If you
need the exact original length back, save it yourself and trim/pad the
result — this is also the tested contract:
`test/test_audio/test_audio_func.py::test_audio_to_spectrum_func` trims the
*original* waveform down to the reconstruction's length before comparing
(`wav[..., :wav_gen.shape[-1]]`), not the other way around.

## Example

```python
from puresound.audio.spectrum import wav_to_stft, stft_to_wav

cpx_stft, stft_info = wav_to_stft(wav, nfft=512, win_size=512, hop_size=128)
# ... process cpx_stft (masking, etc.) ...
reconstructed = stft_to_wav(x=cpx_stft, **stft_info)
reconstructed = reconstructed[..., : wav.shape[-1]]  # caller's job, not stft_to_wav's
```
