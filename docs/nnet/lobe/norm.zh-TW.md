# puresound.nnet.lobe.norm

English version: `norm.md`

給 1D(`[N, C, T]`)與 2D(`[N, C, F, T]`)語音 feature 用的 normalization
layer,以及一個名稱→class 的 `get_norm` 工廠函式,整個 backbone library
都用它從設定字串選擇 normalization 策略。

## Base Class: `_LayerNorm`

不是公開 API 的一部分(沒有對外重新匯出,前面加了底線)。持有可學習的
`gamma`/`beta` affine 參數,各自形狀為 `[channel_size]`,以及
`apply_gain_and_bias`——把 channel 軸換到最後、縮放、再換回來,所以子類別
只需要在 `forward` 裡實作實際的 mean/var 計算即可。

```python
_LayerNorm(channel_size: int)
```

---

## Class: `GlobLN`(別名 `gLN`)

**Global Layer Normalization** —— 一次對*所有*非 batch 軸做 normalize
(所有 channel 與所有時間步一起算,每個 batch item 一組 mean/var)。

```python
GlobLN(channel_size: int)
```

### `forward(x: Tensor) -> Tensor`

**Input:** `[N, C, *]`(適用任何 2D 以上的 rank,不只是 `[N, C, T]`)。
不是 causal-safe 的——統計量會看到整段序列,包含未來的 frame。

---

## Class: `ChanLN`(別名 `cLN`)

**Channel-wise Layer Normalization** —— 每個 `(batch, time)` 組合各自
一組 mean/var,只沿著 channel 軸計算。

```python
ChanLN(channel_size: int)
```

### `forward(x: Tensor) -> Tensor`

**Input:** `[N, C, *]`。是 causal-safe 的(每個時間步各自獨立
normalize,不受相鄰時間步影響)——例如 `TFGridNet` 的 intra/inter norm
就是用這個。

---

## Class: `InstantLN`(別名 `iLN`)

**Instant Layer Normalization** —— 在每個時間步上,對展平後的
`(channel, frequency)` 軸做 normalize;適合 causal/streaming 的 2D model。

```python
InstantLN(channel_size: int)
```

### `forward(x: Tensor) -> Tensor`

**Input:** `[N, CH, C, T]`——reshape 成 `[N, CH*C, T]`,在每個時間步上
對展平的 channel/frequency 軸做 normalize,再 reshape 回來。

---

## Class: `LayerNorm2D`(別名 `LN2D`)

**Channel 與 frequency 雙軸的 Layer Normalization。** 與上面三個 class
不同,這個需要事先知道頻率軸的寬度,因為它的 affine 參數是依
`(channel, frequency)` 組合各自一份,不是單純依 channel:

```python
LayerNorm2D(ch: int, f: int)
```

**Parameters:**
- `ch` – channel 數
- `f` – 頻率軸寬度——`self.w`/`self.b` 的形狀是 `[1, ch, f, 1]`,**不是** `[ch]`

### `forward(x: Tensor) -> Tensor`

**Input:** `[N, ch, C, T]`(`C` 必須等於 `f`)。在每個時間步上,對
`(channel, freq)` 兩軸一起做 normalize(`dims=[1,2]`),再套用依
`(ch, f)` 的 affine 轉換。

> **已知的缺口:** `LN2D` 是這個 module 裡真實存在的別名
> (`LN2D = LayerNorm2D`),但**無法透過下面的 `get_norm` 工廠取得**——
> 工廠的白名單裡從來沒有包含 `"LN2D"`。想要 `LayerNorm2D` 的呼叫端必須
> 自己直接 import 並建構它(`TFGridNet` 就是這樣做的——見下方 Wiring)。
> 這是 `get_norm` 裡一個已知、目前未修正的缺口,本次不處理。

---

## Aliases

```python
gLN  = GlobLN
cLN  = ChanLN
iLN  = InstantLN
bN1d = nn.BatchNorm1d
bN2d = nn.BatchNorm2d
gGN  = lambda x: nn.GroupNorm(1, x, 1e-8)
LN2D = LayerNorm2D
```

## Function: `get_norm`

```python
get_norm(name: str) -> nn.Module
```

把一個**短代碼字串**解析成 normalization **class**(不是 instance——
拿到回傳值後,自己再用 channel 數去實例化它)的工廠函式。只有一個參數;
這裡**沒有**另外的 `channel_size` 參數——實例化的工作交給呼叫端自己做。

**Supported values**(剛好就這 6 個字串——`"global_layer_norm"` 這種
長名稱不被接受):

| Code | Class |
|------|-------|
| `"gLN"` | `GlobLN` |
| `"cLN"` | `ChanLN` |
| `"iLN"` | `InstantLN` |
| `"bN1d"` | `nn.BatchNorm1d` |
| `"bN2d"` | `nn.BatchNorm2d` |
| `"gGN"` | `nn.GroupNorm(1, ·, 1e-8)` |

**Raises:** 其他任何字串都會拋出 `NameError`(包含 `"LN2D"`——見上方的
已知缺口)。

## Choosing a Normalization Strategy

| Class | Causal-safe | 需要頻率寬度 | 常見用途 |
|-------|:-----------:|:----------------------:|-------------|
| `GlobLN` | 否 | 否 | 非 causal 的 1D(Conv-TasNet TCN) |
| `ChanLN` | 是 | 否 | causal 1D,或 `TFGridNet` 的 intra/inter norm |
| `InstantLN` | 是 | 否 | causal/streaming 2D(`Unet` 的 input norm) |
| `LayerNorm2D` | 否 | 是 | 2D time-frequency attention(`TFGridNet`) |

## Wiring

[`cnn.DepthwiseSeparableConv1d`](cnn.zh-TW.md)、
[`rnn.FSMN`](rnn.zh-TW.md)/`ConditionFSMN` 都會用到 `get_norm`,
`ConvTasNet`/`Unet` 也直接用它從設定字串(`tcn_norm`、`norm_type` 等)
選擇 normalization layer。`TFGridNet` 則是直接 import
`LayerNorm2D`、`cLN`、`gLN`——繞過 `get_norm`,自行處理
`input_norm in ["LayerNorm2D", "gLN"]` 這個設定切換,原因正是
`get_norm` 沒辦法產生 `LayerNorm2D`(見上方已知缺口)。

## Example

```python
from puresound.nnet.lobe.norm import get_norm

NormCls = get_norm("gLN")
norm = NormCls(256)
out = norm(features)  # [N, 256, T]
```
