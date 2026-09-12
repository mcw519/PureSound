# puresound.nnet.lobe.encoder

English version: `encoder.md`

把波形轉換成 latent feature、再轉換回來的 encoder/decoder module。這裡的每個
class 都只公開 `forward()`(分析)與 `inverse()`(合成)——整個 module 裡
沒有任何 `encode()`/`decode()` 這種方法名稱。

## Class: `FreeEncDec`

完全可學習的分析/合成 filterbank:`Conv1d` encoder、`ConvTranspose1d`
decoder,兩者之間沒有任何強制對應關係。之所以叫「free」encoder,是因為
filter 是端到端學出來的,而不是固定在某組已知的基底上(對照下方的
`ConvSTFT`)。

> Exported from `puresound.nnet` as `FreeEncDec`.

```python
FreeEncDec(
    win_length: int = 512,
    laten_length: int = 512,
    hop_length: int = 128,
    output_active: Optional[str] = None,
)
```

**Parameters:**
- `win_length` – 分析/合成的 window 長度(即 `Conv1d`/`ConvTranspose1d` 的 kernel size)
- `laten_length` – latent feature 維度(encoder 輸出 channel 數 / decoder 輸入 channel 數)
- `hop_length` – encoder 與 decoder 共用的 stride
- `output_active` – 若有給值,會在 encoder 之後加上一個 `nn.{output_active}()` activation(透過 `getattr(nn, output_active)` 查找,例如 `"ReLU"`)——**沒有 `bias` 參數**;兩個 conv 一律 `bias=False`

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, L]` 或 `[N, 1, L]`

**Returns:** `[N, laten_length, T]`,`T = (L - win_length) // hop_length + 1`。

### `inverse(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, laten_length, T]`

**Returns:** `[N, L]`(`ConvTranspose1d` 之後把 channel 維 squeeze 掉)。

---

## Class: `ConvEncDec`

STFT-based 的 encoder/decoder:把可訓練的 convolutional STFT
(`ConvSTFT`,見下方)包上 window 建構與選用的 pre-emphasis。

> Exported from `puresound.nnet` as `ConvEncDec`.

```python
ConvEncDec(
    fft_length: int = 512,
    win_type: str = "hann",
    win_length: int = 512,
    freq_bins: int = None,
    hop_length: int = 128,
    freq_scale: str = "no",
    iSTFT: bool = True,
    fmin: int = 0,
    fmax: int = 8000,
    sr: int = 16000,
    preemphasis: Optional[float] = None,
    trainable: bool = True,
)
```

**Parameters:**
- `fft_length` – FFT 大小(傳給 `ConvSTFT` 的 `n_fft`)
- `win_type` – `"hann"`、`"hamming"` 或 `"blackman"`——其他值會拋出 `NotImplementedError`
- `win_length` – 分析 window 長度
- `freq_bins` – 要保留的頻率 bin 數;`None` 表示 `n_fft // 2 + 1`
- `hop_length` – STFT hop size
- `freq_scale` – 頻率 bin 間距,是 **`"linear"`、`"log"`、`"no"`** 三選一(不只 2 種)——見 [`stft.create_fourier_kernels`](stft.zh-TW.md);這裡預設是 `"no"`(從 0 Hz 到 Nyquist 均勻分布,忽略 `fmin`/`fmax`),與 `create_fourier_kernels` 自己裸函式的預設值 `"linear"` 不同
- `iSTFT` – 若為 `True`,也會建構 `inverse()` 需要的反轉 kernel
- `fmin` / `fmax` – 只有在 `freq_scale` 為 `"linear"` 或 `"log"` 時才會用到
- `sr` – 取樣率,用於 `"linear"`/`"log"` 的 bin-頻率對應
- `preemphasis` – 若有設定,會在 STFT 前套用 `x[t] - preemphasis * x[t-1]`(第一個樣本靠 zero-padding 保持不變)
- `trainable` – 若為 `True`,底層的 STFT kernel(`wsin`/`wcos`)會是可學習參數,而不是固定的 buffer

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, L]`

**Returns:** `[N, C, T, 2]`(實部/虛部疊在最後一軸;`C = freq_bins`)。

### `inverse(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, C, T, 2]`

**Returns:** `[N, L]`。

---

## Class: `ConvSTFT`

用 `Conv1d`/`Conv2d` 搭配正弦 kernel 實作的可訓練 STFT/iSTFT(不透過
`torch.stft` 走 autograd),改寫自
[nnAudio](https://github.com/KinWaiCheuk/nnAudio)。這就是 `ConvEncDec`
內部建構的東西;也可以單獨使用。

```python
ConvSTFT(
    window_mask: torch.Tensor,
    n_fft: int = 2048,
    win_length: Optional[int] = None,
    freq_bins: Optional[int] = None,
    hop_length: Optional[int] = None,
    freq_scale: str = "no",
    iSTFT: bool = False,
    fmin: int = 50,
    fmax: int = 6000,
    sr: int = 22050,
    trainable: bool = False,
)
```

**Parameters:**
- `window_mask` – **必填**,一個長度為 `n_fft` 的預先算好的 1D window tensor(例如 `torch.hann_window(win_length)`)——這是一個 tensor,不是 `window: str` 這種名稱;若 `len(window_mask) != n_fft` 會拋出 `TypeError`
- `n_fft` – FFT 大小
- `win_length` – 若為 `None` 則預設為 `n_fft`
- `freq_bins` – 若為 `None` 則預設為 `n_fft // 2 + 1`
- `hop_length` – 若為 `None` 則預設為 `win_length // 4`
- `freq_scale`、`fmin`、`fmax`、`sr` – 傳給 [`stft.create_fourier_kernels`](stft.zh-TW.md)
- `iSTFT` – 若為 `True`,也會註冊 `inverse()` 需要的鏡射反轉 kernel(`kernel_sin_inv`、`kernel_cos_inv`)
- `trainable` – 若為 `True`,`wsin`/`wcos`(乘上 window 後的 kernel)會是 `nn.Parameter`;否則註冊為 buffer

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, channel, L]`

以 stride `hop_length` 跑兩個 `Conv1d`(`wsin`、`wcos`),再截到
`freq_bins`。**Returns:** `[N, C, T, 2]`——注意輸出時虛部會被取負號
(`torch.stack((spec_real, -spec_imag), -1)`)。

### `inverse(X: Tensor, refresh_win: bool = True) -> Tensor`

建構時需要 `iSTFT=True`(否則拋出 `NameError`),且 `X.dim() == 4`。
先把 bin 數鏡射回完整的 `n_fft` 寬度
([`stft.extend_fbins`](stft.zh-TW.md)),跑反轉 `Conv2d`,用
[`stft.overlap_add`](stft.zh-TW.md) 重建,再**正規化**——把 overlap-add
的原始輸出除以 [`stft.torch_window_sumsquare`](stft.zh-TW.md),但只在
該值不可忽略(`> 1e-10`)的位置才除——這個正規化步驟不是可有可無的;
省略它會讓重建結果殘留 window 重疊造成的縮放。sum-square 會被快取
(`self.w_sum`),只有在 `refresh_win=True` 或第一次呼叫時才重新計算,
因為它只跟 frame 數量有關,與 `X` 的數值無關。

---

## Class: `UnifiedConvEncDec`

透過為**每個**支援的取樣率各保留一份 `ConvSTFT`,處理任意輸入取樣率,
所有取樣率都建構成相同的 25 ms window / 10 ms hop:

```python
UnifiedConvEncDec(win_type: str = "hann", trainable: bool = False)
```

支援的取樣率(寫死在 `get_stft_parms` 裡的表):`8000, 16000, 22050,
24000, 32000, 44100, 48000` Hz,各自搭配該取樣率下對應 25 ms/10 ms 的
`n_fft`/`hop_length`,`freq_scale="no"`,`iSTFT=True`。

### `forward(x: Tensor, sr: Union[Tensor, int]) -> Tensor`

**Parameters:**
- `x` – `[N, L]`
- `sr` – 可以是單一 `int`(整個 batch 共用同一個取樣率 → 分派給單一 `ConvSTFT`),也可以是逐樣本的 `Tensor[N]`(逐一樣本迴圈,各自分派到對應取樣率的 encoder 再串接——支援混合取樣率的 batch,只是沒有向量化)

**Returns:** `[N, C, T, 2]`。

### `inverse(x: Tensor, sr: Union[Tensor, int]) -> Tensor`

與 `forward` 對稱。**Returns:** `[N, L]`。

目前的 recipe 裡沒有任何呼叫端使用——直接由
`test/test_lobe.py::test_unified_stft_encoder` 針對全部 7 種取樣率做測試;
是給多取樣率部署情境用的 library 積木。

## Wiring

`ConvEncDec` 與 `FreeEncDec` 會以 `puresound.nnet.ConvEncDec` /
`puresound.nnet.FreeEncDec` 的名義重新匯出,是 recipe 設定檔裡
`encoder.type` 可以選的兩種(例如 `egs/default_config.yaml` 的
`encoder: {type: ConvEncDec, encoder_args: {...}}`)。

## Example

```python
from puresound.nnet.lobe.encoder import FreeEncDec, ConvEncDec

# 可學習的 filterbank encoder
enc = FreeEncDec(win_length=16, hop_length=8, laten_length=512)
feat = enc(wav)
wav_out = enc.inverse(feat)

# STFT encoder
enc_stft = ConvEncDec(fft_length=512, hop_length=128, win_length=512, freq_scale="no")
spec = enc_stft(wav)          # [N, C, T, 2]
wav_out = enc_stft.inverse(spec)
```
