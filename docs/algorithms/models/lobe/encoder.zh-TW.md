# puresound.nnet.lobe.encoder

English version: [encoder.md](encoder.md)

波形 encoder 及其反變換：一個可學的 filterbank（`FreeEncDec`），以及用明確的
Fourier kernel 以 convolution 實作的 STFT（`ConvSTFT`，由 `ConvEncDec` 與
`UnifiedConvEncDec` 包裝）。每個 class 都以 `forward()` 做分析、`inverse()` 做合成。

它們都不對訊號做 padding：`L` 個輸入樣本得到 `T = (L - win) // hop + 1` 幀，
`inverse` 回傳 `(T - 1) * hop + win` 個樣本，所以不足一個 hop 的尾端會被丟掉。

## Class: `FreeEncDec`

可學的分析 `Conv1d` 與合成 `ConvTranspose1d`，兩者都沒有 bias，彼此也不綁定。

```python
FreeEncDec(
    win_length: int = 512,               # 兩個 conv 的 kernel size
    laten_length: int = 512,             # 可學 filter 的數量
    hop_length: int = 128,               # 兩個 conv 的 stride
    output_active: Optional[str] = None, # 接在 encoder 後的 nn class 名稱，例如 "ReLU"
)
```

| method | shape |
| --- | --- |
| `forward(x)` | `[N, L]` 或 `[N, 1, L]` -> `[N, laten_length, T]` |
| `inverse(x)` | `[N, laten_length, T]` -> `[N, L']` |

## Class: `ConvEncDec`

依名稱建立 window 的 `ConvSTFT`，另可加 pre-emphasis。

```python
ConvEncDec(
    fft_length: int = 512,
    win_type: str = "hann",          # "hann" | "hamming" | "blackman"
    win_length: int = 512,           # 必須等於 fft_length，否則 TypeError
    freq_bins: int = None,           # None -> fft_length // 2 + 1
    hop_length: int = 128,
    freq_scale: str = "no",          # "no" | "linear" | "log"，見 stft.create_fourier_kernels
    iSTFT: bool = True,              # 建立反變換 kernel
    fmin: int = 0,                   # 只有 "linear" / "log" 會用到
    fmax: int = 8000,
    sr: int = 16000,
    preemphasis: Optional[float] = None,  # STFT 前做 x[t] - p * x[t-1]
    trainable: bool = True,          # Fourier kernel 為參數
)
```

| method | shape |
| --- | --- |
| `forward(x)` | `[N, L]` -> `[N, F, T, 2]`（實部、虛部） |
| `inverse(x)` | `[N, F, T, 2]` -> `[N, L']` |

`inverse` 不會還原 pre-emphasis。`trainable` 預設為 `True`；要固定 STFT 的 recipe
請設為 `False`。

## Class: `ConvSTFT`

以 convolution 實作的 STFT 與反 STFT，改寫自
[nnAudio](https://github.com/KinWaiCheuk/nnAudio)。

```python
ConvSTFT(
    window_mask: torch.Tensor,        # 長度為 n_fft 的 window tensor，否則 TypeError
    n_fft: int = 2048,
    win_length: Optional[int] = None, # None -> n_fft
    freq_bins: Optional[int] = None,  # None -> n_fft // 2 + 1
    hop_length: Optional[int] = None, # None -> win_length // 4
    freq_scale: str = "no",
    iSTFT: bool = False,
    fmin: int = 50,
    fmax: int = 6000,
    sr: int = 22050,
    trainable: bool = False,          # 加窗後的 kernel wsin / wcos 為參數
)
```

`forward(x [N, 1, L]) -> [N, F, T, 2]`，以 window `w`、hop `H` 計算

```
X[k, t] = sum_n w[n] x[t*H + n] exp(-j 2 pi k n / n_fft)
```

實作為兩個 kernel 分別是 `w * cos` 與 `w * sin` 的 strided `Conv1d`；輸出時虛部取負號。

`inverse(X [N, F, T, 2], refresh_win: bool = True) -> [N, L']` 把 bin 以 Hermitian
對稱補回完整 `n_fft`，套用反變換 kernel、乘上 window、除以 `n_fft`、overlap-add，
再在 window 平方和大於 `1e-10` 的位置除以它。需要 `iSTFT=True`（否則
`NameError`）。平方和只取決於幀數，會被快取；`refresh_win=False` 直接重用。

## Class: `UnifiedConvEncDec`

每個取樣率一個 `ConvSTFT`，都是 25 ms window、10 ms hop
（`n_fft = 0.025 * sr`、`hop = 0.01 * sr`）、`freq_scale="no"`、`iSTFT=True`。

```python
UnifiedConvEncDec(win_type: str = "hann", trainable: bool = False)
```

支援的取樣率：8000、16000、22050、24000、32000、44100、48000 Hz。
`forward(x [N, L], sr)` 回傳 `[N, F, T, 2]`，`inverse(x, sr)` 回傳 `[N, L']`。
`sr` 是整批共用的 `int`，或逐列分派再串接的 `Tensor[N]`，因此所有列必須產生相同 shape。

各取樣率的 encoder 是 `nn.ModuleDict` 裡的 submodule，以字串形式的取樣率為鍵
（`encoder["16000"]`）：`.to(device)` 會搬動它們，kernel 在 `state_dict()` 中，
`trainable=True` 時加窗後的 kernel 也在 `parameters()` 中。沒有 recipe 使用這個 class。

## 在 recipe 中的用法

`ConvEncDec` 與 `FreeEncDec` 由 `puresound.nnet` 匯出，以 `model.encoder.type` 選擇：

```yaml
model:
  encoder:
    type: ConvEncDec
    encoder_args:
      fft_length: 512
      win_type: "hann"
      win_length: 512
      hop_length: 160
      fmin: 0
      fmax: 8000
      sr: 16000
      trainable: False
```

## 設計說明

- 以明確 kernel 的 convolution 表示 STFT，分析與合成就是一般的 layer：kernel 可以
  訓練，計算圖裡也沒有 `torch.stft`。
- 除以 window 平方和，讓反變換對任何重疊足夠的 window 與 hop 都是精確的，而不只限於
  滿足 constant-overlap-add 條件的 window。
