# puresound.nnet.lobe.stft

English version: `stft.md`

支撐 [`encoder.ConvSTFT`](encoder.zh-TW.md) 的自由函式:Fourier kernel
產生、overlap-add 重建,以及 Mel 尺度轉換。改寫自
[nnAudio](https://github.com/KinWaiCheuk/nnAudio) 與
[librosa](https://librosa.org/) 風格的 Mel 工具函式。

## `create_fourier_kernels`

```python
create_fourier_kernels(
    n_fft,
    win_length=None,
    freq_bins=None,
    fmin=50,
    fmax=6000,
    sr=44100,
    freq_scale="linear",
) -> Tuple[np.ndarray, np.ndarray, list, list]
```

**Parameters:**
- `n_fft` – window 大小
- `win_length` – 若為 `None` 則預設為 `n_fft`(為了與呼叫端的 API 對稱而保留;本身不會拿來改變 kernel 寬度)
- `freq_bins` – 頻率 bin 數;若為 `None` 則預設為 `n_fft // 2 + 1`
- `fmin` / `fmax` – `"linear"`/`"log"` 分bin時的頻率範圍;`freq_scale="no"` 時會被忽略
- `sr` – 取樣率,用來把 `fmin`/`fmax` 換算成 bin index
- `freq_scale` – `"linear"`(在 `fmin`/`fmax` 之間均勻分布)、`"log"`(在 `fmin`/`fmax` 之間對數分布),或 `"no"`(從 0 Hz 到 Nyquist 均勻分布,忽略 `fmin`/`fmax`)

**Returns:** `(wsin, wcos, bins2freq, binslist)`——`wsin`/`wcos` 是形狀
`(freq_bins, 1, n_fft)` 的實數 NumPy array;`bins2freq` 把每個 bin index
對應到 Hz 的中心頻率;`binslist` 是同一個對應關係,但用 normalized
DFT-bin 單位表示。

`ConvSTFT` 每次呼叫都會明確帶入 `sr`/`fmax`(它自己的預設值是
`sr=22050, fmax=6000`——見 [`encoder.md`](encoder.zh-TW.md)),所以上面
裸函式的預設值(`sr=44100`、`fmax=6000`)只有在直接呼叫
`create_fourier_kernels` 時才會有影響。

## `mel_filterbank`

```python
mel_filterbank(
    sr: int,
    n_fft: int,
    n_banks: int = 128,
    fmin: float = 0.0,
    fmax: Optional[float] = None,
    norm: int = 1,
) -> torch.Tensor
```

> **參數順序是 `(sr, n_fft, n_banks, ...)`,不是 `(n_fft, n_mels,
> sr, ...)`。** 若用後者的順序以 positional 方式呼叫,取樣率會被
> 悄悄地當成 `n_fft` 傳進去,反之亦然——不會有任何例外拋出(兩個都是
> 普通的 int),filterbank 就是算錯而已。除非確定順序無誤,否則請一律
> 用 keyword 傳入 `sr` 與 `n_fft`。band 數量的參數名稱也是 `n_banks`,
> 不是 `n_mels`。

**Parameters:**
- `sr` – 取樣率
- `n_fft` – FFT 大小(線性 bin 數 = `n_fft // 2 + 1`)
- `n_banks` – Mel filter 數量(輸出的 feature 維度)
- `fmin` / `fmax` – Mel 頻率範圍;`fmax` 若為 `None` 則預設為 `sr / 2`
- `norm` – 若為 `1`(預設),做 Slaney 風格的面積正規化(每個 band 能量大致相等)

**Returns:** `[n_banks, n_fft // 2 + 1]`。若任何一個 filter 全部落空
(`fmax` 太低,或 `n_banks` 相對 `sr`/`n_fft` 太多),會拋出 `ValueError`。

## `overlap_add`

```python
overlap_add(X: Tensor, stride: int) -> Tensor
```

**Parameters:**
- `X` – 已分 frame 的訊號 `[batch, n_fft, n_frames]`
- `stride` – frame 之間的 hop size

用 `kernel_size=(1, n_fft)` 的 `torch.nn.functional.fold` 實作。
**沒有 `window` 參數**——`overlap_add` 只做重疊 frame 的加總,並不知道
也不會套用任何 window。

**Returns:** `[batch, n_fft + stride * (n_frames - 1)]`。

> **這個原始輸出還沒有正規化。** 疊加起來的 windowed frame 經過 `fold`
> 加總後,每個樣本點被縮放的程度取決於該點被多少個 window 重疊覆蓋——
> 正確的重建需要事後除以
> [`torch_window_sumsquare`](#torch_window_sumsquare),且只在該
> sum-square 不可忽略的位置才除。`ConvSTFT.inverse` 就是這麼做的
> (見 [`encoder.md`](encoder.zh-TW.md)):
> ```python
> real = overlap_add(real, stride)
> w_sum = torch_window_sumsquare(window_mask, n_frames, stride, n_fft)
> nonzero = w_sum > 1e-10
> real[:, nonzero] = real[:, nonzero].div(w_sum[nonzero])
> ```

## `torch_window_sumsquare`

```python
torch_window_sumsquare(w, n_frames: int, stride: int, n_fft: int, power=2) -> Tensor
```

**Parameters:**
- `w` – window 函式 tensor,長度 `n_fft`
- `n_frames` – overlap-add 的 frame 數
- `stride` – hop size
- `n_fft` – FFT/window 大小
- `power` – 加總之前套用在 window 上的指數(`2` 對應一般的平方和正規化)

計算方式與 `overlap_add` 相同,只是對象是重複 `n_frames` 次的
`w**power`,而不是真正的 frame 資料——也就是「單靠這個 window,在每個
樣本點上,overlap-add 加起來會是多少」。**Returns:**
`[batch=1, n_fft + stride * (n_frames - 1)]`。

## `extend_fbins`

```python
extend_fbins(X: Tensor) -> Tensor
```

**Parameters:** `X` – 單邊頻譜 `[batch, n_fft//2+1, T, 2]`

把 bin `1..-2`(排除 DC 與 Nyquist)鏡射到上半部,並把鏡射部分的虛部
取負(奇對稱),重建出反轉 convolution kernel 需要的完整
`n_fft` 寬雙邊頻譜。**Returns:** `[batch, n_fft, T, 2]`。

## Mel-scale helpers

- **`hz2mel(frequencies)`** / **`mel2hz(mels)`** – HTK 風格的 Hz↔Mel 轉換,1000 Hz 以下是線性區間,以上是對數區間(公式與 librosa 相容)。
- **`fft_frequencies(sr=16000, n_fft=512) -> np.ndarray`** – `n_fft//2+1` 個線性 FFT bin 各自的中心頻率。
- **`mel_frequencies(n_mels=128, fmin=0.0, fmax=8000)`** – `n_mels` 個 Mel band 的中心頻率,在 Mel 尺度上於 `fmin`/`fmax` 之間均勻分布。

`mel_filterbank` 內部會用到以上三個函式。

## Wiring

[`encoder.md`](encoder.zh-TW.md) 裡的 `ConvSTFT`/`ConvEncDec` 靠
`create_fourier_kernels`、`extend_fbins`、`overlap_add`、
`torch_window_sumsquare` 完成 forward/inverse STFT。`mel_filterbank`
則直接被 `puresound/nnet/features.py` 的 `FeatureEncoder` 使用
(`mel_filterbank(sr, n_fft, n_banks)`——以安全的 positional 順序呼叫)
來建構 Mel 投影矩陣。

## Example

```python
from puresound.nnet.lobe.stft import create_fourier_kernels, mel_filterbank

wsin, wcos, bins2freq, _ = create_fourier_kernels(n_fft=512, sr=16000, fmax=8000)
mel_fb = mel_filterbank(sr=16000, n_fft=512, n_banks=80, fmin=20.0, fmax=8000.0)
```
