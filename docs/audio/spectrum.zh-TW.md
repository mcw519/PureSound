# puresound.audio.spectrum

English version: [`spectrum.md`](spectrum.md)

STFT/iSTFT 輔助函式,以及 complex ↔ (magnitude, phase) 的互轉。
`wav_to_stft` 跟 `stft_to_wav` 共用同一套 FFT/window 參數命名慣例 ——
`nfft`/`win_size`/`hop_size`/`window_type`/`stft_normalized` —— 而且
`wav_to_stft` 的第二個回傳值,設計上就是要直接展開餵給 `stft_to_wav`
(`stft_to_wav(x=cpx, **stft_info)`;見下方範例)。

## Functions

### `tensor_as_complex(x: Tensor) -> Tensor`

把一個**最後一維長度為 2 的實數 tensor**(`[..., 2]`,實部與虛部打包在
最後一個軸)透過 `torch.view_as_complex` 轉成複數 tensor `[...]`
(這個函式要求最後一維是 contiguous 的 —— 如果不是,會先呼叫
`.contiguous()`)。如果 `x` 本來就已經是複數,會原封不動回傳。如果最後
一維不是 2,會丟出 `ValueError`。

雖然名字看起來像,但這**不是**一個吃 `(real, imag)` 兩個參數的合併函式
—— 如果你手上是分開的 real/imag tensor,要自己先打包成一個 tensor 的
最後一軸;如果手上是 magnitude/phase,則改用 `mag_and_phase_as_cpx_stft`。

---

### `cpx_stft_as_mag_and_phase(x: Tensor, eps: float = 1e-8) -> Tuple[Tensor, Tensor]`

`x` 可以已經是複數,也可以是最後一軸長度為 2 的實數(會先經過
`tensor_as_complex` 轉換)。回傳 `(magnitude, phase)`,兩者皆為實數,
shape 跟複數 tensor 相同:

```python
mag = sqrt(real**2 + imag**2 + eps)   # eps=None 會跳過這個 epsilon(精確的 magnitude,可以恰好是 0)
phase = atan2(imag, real)
```

---

### `mag_and_phase_as_cpx_stft(mag: Tensor, phase: Tensor) -> Tensor`

上面函式的反運算:`mag * exp(1j * phase)`。要求 `mag.shape == phase.shape`。

---

### `wav_to_stft(wav: Tensor, nfft: int = 512, win_size: int = 512, hop_size: int = 128, window_type: str = "hann_window", stft_normalized: bool = False) -> Tuple[Tensor, Dict]`

`torch.stft(..., return_complex=True)` 的薄包裝。`window_type` 是一個
**字串**,透過 `getattr(torch, window_type)` 解析(例如 `"hann_window"`
→ `torch.hann_window`、`"hamming_window"` → `torch.hamming_window`)——
不是你自己建構的 window tensor;它必須是某個真實存在的
`torch.*_window` factory 函式名稱。

輸入 `wav: [..., L]`(通常是 `[N, L]` 或 `[L]`);輸出
`cpx_stft: [..., nfft // 2 + 1, T]` —— 這是單邊(one-sided)實數輸入
STFT,所以頻率 bin 數是 `nfft // 2 + 1`,**不是** `nfft`。

同時會回傳 `stft_info`,這個 dict **正好**就是 `stft_to_wav` 需要的
關鍵字參數組合:

```python
{"nfft": nfft, "win_size": win_size, "hop_size": hop_size,
 "window_type": window_type, "stft_normalized": stft_normalized}
```

**裡面沒有 `original_length` 這個 key** —— 這個函式不會幫你記住輸入的
長度(下面 `stft_to_wav` 會說明這為什麼重要)。

---

### `stft_to_wav(x: Tensor, nfft: int = 512, win_size: int = 512, hop_size: int = 128, window_type: str = "hann_window", stft_normalized: bool = False) -> Tensor`

`wav_to_stft` 的反運算,底層是 `torch.istft`。`x` 會先經過
`tensor_as_complex` 處理,所以最後一軸長度為 2 的實數 `[..., F, T, 2]`
tensor 也能用,不是只能接受已經是複數的 tensor。

**不會還原長度,也沒有 `original_length` 參數。** `torch.istft` 輸出的
長度是 `nfft`/`hop_size`/`win_size`/frame 數的函式,**通常不會**剛好等於
原始波形的長度 —— 已經實際驗證過:把長度 16001 的輸入,用
`nfft=1024, win_size=512, hop_size=160` 重建回來,長度會變成 16000;
只有長度剛好落在 hop/frame 網格上的輸入,才會原封不動地來回還原。如果
你需要精確拿回原始長度,要自己存好長度,再自行裁切/補零結果 —— 測試裡
驗證的合約也是這樣做的:
`test/test_audio/test_audio_func.py::test_audio_to_spectrum_func` 是把
*原始*波形裁到跟重建結果一樣長再比較(`wav[..., :wav_gen.shape[-1]]`),
而不是反過來。

## 範例

```python
from puresound.audio.spectrum import wav_to_stft, stft_to_wav

cpx_stft, stft_info = wav_to_stft(wav, nfft=512, win_size=512, hop_size=128)
# ... 處理 cpx_stft(套用 mask 等等)...
reconstructed = stft_to_wav(x=cpx_stft, **stft_info)
reconstructed = reconstructed[..., : wav.shape[-1]]  # 這是呼叫端自己的工作,不是 stft_to_wav 的
```
