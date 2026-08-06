# puresound.nnet.lobe.cnn

English version: `cnn.md`

1D depthwise-separable convolution(Conv-TasNet 風格的 TCN block),以及給 2D
time-frequency feature 用的 Fast Fourier Convolution(FFC-SE)積木。

## Class: `DepthwiseSeparableConv1d`

Depthwise-separable 1D convolution,可選擇性地加上改變維度的輸入投影、
causal padding,以及 skip connection。

```python
DepthwiseSeparableConv1d(
    in_channels: int,
    out_channels: int,
    hid_channels: Optional[int] = None,
    norm_cls: str = "gGN",
    kernel: int = 3,
    stride: int = 1,
    dilation: int = 1,
    skip: bool = False,
    causal: bool = False,
)
```

**Parameters:**
- `in_channels` – 輸入 channel 數
- `out_channels` – 輸出 channel 數——**可以與 `in_channels` 不同**,不是單純保留原本維度
- `hid_channels` – 若有給值,會先用 `Conv1d(1×1) → norm → PReLU` 把 `in_channels → hid_channels`(一個「dense transform」)再進入 depthwise 階段;若為 `None`,depthwise 階段就直接在 `in_channels` 上運作
- `norm_cls` – 透過 [`norm.get_norm`](norm.zh-TW.md) 解析的短代碼字串——例如 `"gLN"`、`"gGN"`、`"cLN"`、`"iLN"`、`"bN1d"`——**不是** `"global_layer_norm"` 這種長名稱
- `kernel` – depthwise kernel 大小
- `stride` – depthwise stride
- `dilation` – depthwise dilation
- `skip` – 若為 `True`,會另外加一個 `Conv1d(in_channels, out_channels, 1)`,把原始輸入直接投影到輸出,與主要的 depthwise/pointwise 路徑彼此獨立
- `causal` – 若為 `True`,depthwise conv 會在左側 pad `(kernel - 1) * dilation`,結束後再把等量的尾端 frame 裁掉,確保不會有未來的 frame 洩漏進來。並會 assert `norm_cls not in ["gLN", "gGN"]`,因為這兩者是對整個序列做 normalize,不是 causal-safe 的

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – `[N, in_channels, T]`

**Returns:** `[N, out_channels, T']`(`stride > 1` 時 `T' != T`)。

流程:視 `hid_channels` 是否有值決定要不要跑 `in_conv` → depthwise conv +
norm + PReLU → pointwise `1×1` conv + norm + PReLU →(若 `causal`,裁掉尾端
padding)→(若 `skip`,加上 `skip_conv(x)`)。

---

## Class: `SpectralTransform`

`FFC`(見下方)的組成部分。對 `[N, CH, C, T]` feature map 的**頻率軸**做
real FFT——原始碼註解形容為「與 cepstrum-space transformation 相同的概念」
——在轉換後的空間裡用 `1×1` conv 處理實部/虛部,再反轉 FFT,並殘差相加回
空間域分支。

```python
SpectralTransform(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    causal: bool = True,
)
```

**Parameters:**
- `in_channels` / `out_channels` – channel 數
- `kernel_size` – `(kernel_f, kernel_t)`
- `stride` – `(stride_f, stride_t)`
- `causal` – 若為 `True`,時間軸只做左側 padding;若為 `False` 則對稱 padding。頻率軸的 padding 一律對稱(兩側各 `kernel_f // 2`)

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, CH, C, T]`(這裡的 `C` 是**頻率軸**,不是 channel 數;`CH` 才是 channel 數)

流程:`Conv2d + BN + ReLU` → 沿頻率軸做 `rfft`(`dim=2`) → 把實部/虛部疊到
channel 軸上 → 在 FFT 空間內做 `1×1 Conv2d + BN + ReLU` → 拆開、`irfft`
轉回來 → 殘差相加回 FFT 前的分支 → `1×1 Conv2d` 輸出投影。

**Returns:** `[N, CH, out_channels, T]`。

目前 repository 內沒有任何呼叫端使用——是個尚未接入任何 backbone 的
library 積木。

---

## Class: `FFC`

**Fast Fourier Convolution。** 把 channel 拆成「local」(一般的空間 conv)
分支與「global」(FFT、涵蓋整個頻率軸感受野)分支,兩者互相餵給對方,
最後串接起來。

```python
FFC(
    in_channels: int,
    out_channels: int,
    alpha: float = 0.3,
    kernel_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    causal: bool = True,
)
```

**Parameters:**
- `in_channels` / `out_channels` – 總 channel 數,會拆給 local/global 兩個分支
- `alpha` – global 分支的 channel 比例:`fft_in_ch = int(in_channels * alpha)`、`fft_out_ch = int(out_channels * alpha)`;兩側剩下的部分就是 local 分支(`local_in_ch`、`local_out_ch`)
- `kernel_size`、`stride` – 與 `SpectralTransform` 相同
- `causal` – 與 `SpectralTransform` 相同,並且同樣套用在全部四條內部 conv 路徑上

**Reference:** "FFC-SE: Fast Fourier Convolution for Speech Enhancement"。

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, CH, C, T]`,沿 `CH` 拆成
`global_in = x[:, :fft_in_ch]` 與 `local_in = x[:, fft_in_ch:]`。

四條 conv 路徑組合成兩個分支輸出(變數名稱與原始碼一致):

| 輸出 | = | FFT/global 路徑 | + | 交叉/local 路徑 |
|---|---|---|---|---|
| `global_out` | = | `global_spec_trans(global_in)`(一個 `SpectralTransform`) | + | `local_global_conv(local_in)`(單純 `Conv2d`,local→global) |
| `local_out` | = | `global_conv(global_in)`(單純 `Conv2d`,global→local) | + | `local_local_conv(local_in)`(單純 `Conv2d`,local→local) |

每個分支輸出接著各自的 `BatchNorm2d + ReLU`。

**Returns:** `torch.cat([local_out, global_out], dim=1)` → `[N, out_channels, C, T]`(local channel 在前,global channel 在後)。

目前 repository 內沒有任何呼叫端使用——與 `SpectralTransform` 一樣是
library 積木。

## Wiring

`ConvTasNet` 的 TCN block(`puresound/nnet/conv_tasnet.py`)用
`hid_channels=None, skip=False` 以及依設定決定的 `dconv_norm`(預設
`"gGN"`)包住 `DepthwiseSeparableConv1d`;維度轉換與 channel 的
`in_conv`/`out_conv` 在那個 backbone 裡是由 `conv_tasnet.py` 自己周圍的
layer 處理,不是靠這裡的 `hid_channels`。

## Example

```python
from puresound.nnet.lobe.cnn import DepthwiseSeparableConv1d

conv = DepthwiseSeparableConv1d(
    in_channels=256,
    out_channels=256,
    kernel=3,
    dilation=4,
    causal=False,
    norm_cls="gLN",
)
out = conv(features)  # [N, 256, T]
```
