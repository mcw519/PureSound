# puresound.audio.spectrum

English version: [spectrum.md](spectrum.md)

波形層級的 STFT/iSTFT 輔助函式，以及複數 ↔（幅度、相位）轉換。它們是
`torch.stft`/`torch.istft` 的薄包裝，給腳本與分析使用；模型改用
`puresound.nnet.lobe` 裡的 STFT 層（見 [lobe/stft](../models/lobe/stft.zh-TW.md)）。

## `wav_to_stft(wav, nfft=512, win_size=512, hop_size=128, window_type="hann_window", stft_normalized=False) -> (cpx_stft, stft_info)`

`torch.stft(..., return_complex=True)`，其他沿用 torch 預設（`center=True`、
reflect padding、單邊頻譜）。

- `wav: [..., L]` → `cpx_stft: [..., nfft // 2 + 1, 1 + L // hop_size]`，複數。
- `window_type` 是 torch 窗函數工廠的*名稱*，以 `getattr(torch, window_type)`
  解析（`"hann_window"`、`"hamming_window"`……），長度 `win_size`，建在輸入所在
  的 device 上。
- `stft_normalized=True` 會傳 `normalized=True` 給 torch，把每個 frame 乘上
  `1/√nfft`。
- `stft_info` 是 `{nfft, win_size, hop_size, window_type, stft_normalized}`：
  正好是 `stft_to_wav` 的關鍵字參數集合，所以 `stft_to_wav(cpx, **stft_info)`
  會用相同設定反轉。它不記錄輸入長度。

## `stft_to_wav(x, nfft=512, win_size=512, hop_size=128, window_type="hann_window", stft_normalized=False) -> wav`

以相同設定呼叫 `torch.istft`。`x` 可以是複數，也可以是最後一軸大小為 2 的實數。
輸出長度是 `hop_size · (frames − 1)`，在上述預設下即輸入長度無條件捨去到
`hop_size` 的倍數；只有原長度本來就是倍數時才會相等。請自行保存原長度並裁切或
補齊：

```python
from puresound.audio.spectrum import wav_to_stft, stft_to_wav

cpx, info = wav_to_stft(wav, nfft=512, win_size=512, hop_size=128)
rec = stft_to_wav(cpx, **info)            # 長度 = 128 · (L // 128)
wav_aligned = wav[..., : rec.shape[-1]]
```

## `tensor_as_complex(x) -> Tensor`

`x` 已是複數就原樣回傳；否則用 `torch.view_as_complex` 把實數 `[..., 2]`
（最後一軸為實部、虛部）視為複數，必要時先轉成 contiguous。最後一軸不是 2 則
raise `ValueError`。

## `cpx_stft_as_mag_and_phase(x, eps=1e-8) -> (mag, phase)`

```
mag   = sqrt(re² + im² + eps)        # eps=None 得到精確幅度
phase = atan2(im, re)
```

`eps` 讓平方根在幅度為零處的梯度保持有限。

## `mag_and_phase_as_cpx_stft(mag, phase) -> Tensor`

`mag · exp(j·phase)`；兩者形狀必須相同。
