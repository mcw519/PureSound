# puresound.nnet.lobe.stft

English version: [stft.md](stft.md)

[`encoder.ConvSTFT`](encoder.zh-TW.md) 與 [`FeatureEncoder`](../features.zh-TW.md) Mel 前端背後的
自由函式：Fourier kernel 生成、overlap-add 重建、Mel 尺度轉換。改寫自
[nnAudio](https://github.com/KinWaiCheuk/nnAudio) 與 librosa 的 Mel 工具。

## `create_fourier_kernels`

建立 `Conv1d` 計算 DFT 用的 sine 與 cosine kernel：
`wcos[k, 0, n] = cos(2π f_k n / n_fft)`、`wsin[k, 0, n] = sin(2π f_k n / n_fft)`，
其中 `f_k` 是第 `k` 個 bin 以 DFT-bin 為單位的頻率。

```python
create_fourier_kernels(
    n_fft,                 # window 大小
    win_length=None,       # 預設 n_fft；不改變 kernel 寬度
    freq_bins=None,        # 預設 n_fft // 2 + 1
    fmin=50,               # "linear" / "log" 的範圍；"no" 時忽略
    fmax=6000,
    sr=44100,              # 把 fmin / fmax 換算成 bin 單位
    freq_scale="linear",   # "linear" | "log" | "no"
) -> (wsin, wcos, bins2freq, binslist)
```

- `"linear"`——`freq_bins` 個 bin 從 `fmin` 往 `fmax` 等距排列。
- `"log"`——從 `fmin` 往 `fmax` 以對數間距排列。
- `"no"`——標準 DFT bin，0 Hz 到 Nyquist；忽略 `fmin`/`fmax`。其他值會記一筆 warning
  並回傳未初始化的 kernel。

回傳 float32 NumPy 陣列 `wsin`/`wcos`，shape `(freq_bins, 1, n_fft)`，以及
`bins2freq`（每個 bin 的頻率，Hz）與 `binslist`（同一對應，DFT-bin 單位）。
`ConvSTFT` 會傳入它自己的 `sr`/`fmin`/`fmax`/`freq_scale`（預設 `"no"`），所以這些
預設值只影響直接呼叫。

## `mel_filterbank`

```python
mel_filterbank(
    sr: int,
    n_fft: int,
    n_banks: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,   # 預設 sr / 2
    norm: int = 1,                  # 1：Slaney 面積正規化
) -> torch.Tensor                   # [n_banks, n_fft // 2 + 1]
```

三角形 Mel filter，中心頻率在 `fmin` 與 `fmax` 之間的 Mel 尺度上等距。`norm=1` 時每個
filter 乘上 `2 / (f_{i+2} - f_i)`，讓每個頻帶的能量大致相同。有 filter 為空
（`n_banks` 對 `sr`/`n_fft`/`fmax` 來說太多）時丟 `ValueError`。

參數順序是 `(sr, n_fft, n_banks)`。前兩個參數都是 int，位置參數寫反會建出錯的
filterbank 而不報錯；請用 keyword 傳。`FeatureEncoder` 呼叫
`mel_filterbank(sr, n_fft, n_banks)`，再把結果轉置成 `[n_fft // 2 + 1, n_banks]`。

## `overlap_add`

```python
overlap_add(X: Tensor, stride: int) -> Tensor
```

用 `torch.nn.functional.fold` 把分幀訊號 `X [batch, n_fft, n_frames]` 以 hop
`stride` 相加；回傳 `[batch, n_fft + stride * (n_frames - 1)]`。不套 window，也不做
正規化。

## `torch_window_sumsquare`

```python
torch_window_sumsquare(w, n_frames: int, stride: int, n_fft: int, power=2) -> Tensor
```

`w ** power` 重複 `n_frames` 個 frame 後的 overlap-add：
`[1, 1, 1, n_fft + stride * (n_frames - 1)]`。把 `overlap_add` 的輸出除以它（在它不接近
零的位置）就得到補償過 window 的 inverse STFT。`ConvSTFT.inverse` 的做法：

```python
real = overlap_add(real, stride)
w_sum = torch_window_sumsquare(window_mask, n_frames, stride, n_fft)
nonzero = w_sum > 1e-10
real[:, nonzero] = real[:, nonzero].div(w_sum[nonzero])
```

## `extend_fbins`

```python
extend_fbins(X: Tensor) -> Tensor   # [batch, n_fft//2 + 1, T, 2] -> [batch, n_fft, T, 2]
```

從單邊頻譜重建雙邊頻譜：把 bin `1 .. -2`（不含 DC 與 Nyquist）鏡射到上半部，並把
虛部取負號（實數訊號頻譜的 Hermitian 對稱）。`ConvSTFT.inverse` 使用。

## Mel 尺度 helper

- `hz2mel(frequencies)` / `mel2hz(mels)`——Slaney 式 Mel 尺度（librosa 的預設，
  `htk=False`）：1000 Hz 以下為線性（每 Mel `f_sp = 200/3` Hz），以上為對數。
- `fft_frequencies(sr=16000, n_fft=512)`——`n_fft // 2 + 1` 個 FFT bin 各自的中心頻率。
- `mel_frequencies(n_mels=128, fmin=0.0, fmax=8000)`——在 `fmin` 與 `fmax` 之間的
  Mel 尺度上等距的 `n_mels` 個頻率。

`mel_filterbank` 建立在這三個之上。

## 範例

```python
from puresound.nnet.lobe.stft import create_fourier_kernels, mel_filterbank

wsin, wcos, bins2freq, _ = create_fourier_kernels(n_fft=512, sr=16000, freq_scale="no")
mel_fb = mel_filterbank(sr=16000, n_fft=512, n_banks=80, fmin=20.0, fmax=8000.0)
```
