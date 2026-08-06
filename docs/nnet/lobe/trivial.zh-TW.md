# puresound.nnet.lobe.trivial

English version: `trivial.md`

一些小工具 layer:把函式包成 module 的 wrapper、complex→magnitude 轉換、
兩種不同的條件化機制(`Gate`、`FiLM`)、給 dual-path model 用的分段/合併、
純量的 moving average、頻譜的 power-law 壓縮,以及 SpecAugment 風格的
masking。

## Class: `LambdaLayer`

把任意函式包成 `nn.Module`,方便把不含參數的轉換塞進 `nn.Sequential` 裡。

```python
LambdaLayer(lambda_func: LambdaType)
```

注意建構子參數名稱是 `lambda_func`,不是 `fn`。

### `forward(x: Tensor, **kwargs) -> Any`

回傳 `lambda_func(x, **kwargs)`——額外的 keyword 參數會直接傳遞下去,
所以 `lambda_func` 本身可以吃不只一個參數。

---

## Class: `Magnitude`

把 complex spectrum 轉成 magnitude spectrum。可以接受**兩種**輸入排列
方式:

```python
Magnitude(drop_first: bool = True, log1p: bool = False)
```

**Parameters:**
- `drop_first` – 若為 `True`,輸出時捨棄第一個頻率 bin(DC)
- `log1p` – 若為 `True`,對 magnitude 套用 `log1p`(log-magnitude 壓縮)

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x`,可以是以下任一種:
- `[N, C, T, 2]`(4D——實部/虛部疊在最後一軸,例如直接來自 [`encoder.ConvSTFT`](encoder.zh-TW.md) 的輸出),或
- `[N, 2*C, T]`(3D——實部/虛部串接在 channel 軸上,用 `torch.chunk(x, 2, dim=1)` 拆開)

其他 rank 一律拋出 `TypeError`。計算方式為
`sqrt(re**2 + im**2 + 1e-8)`(這裡的 `1e-8` 是為了在 magnitude 為零時
梯度仍然有限),然後視需要捨棄第一個 bin 與/或套用 `log1p`。

**Returns:** `[N, C(-1), T]`(若 `drop_first`,頻率 bin 數會減 1)。

---

## Class: `Gate`

GLU 風格的 gate:一個只看 `x` 的內容分支,逐元素乘上一個同時看得到
`x` 與條件向量的 sigmoid gate 分支。

```python
Gate(input_size: int, hidden_size: int, embed_size: int, dropout: float = 0.0)
```

**Parameters:**
- `input_size` – `x` 的 feature 維度(同時也是輸出維度——這個 block 是殘差式、保持形狀的)
- `hidden_size` – 內部投影寬度
- `embed_size` – 條件化 embedding 維度
- `dropout` – 同時套用在內容分支與 gate 分支內部

### `forward(x, condition) -> Tensor`

```
res  = x
h    = in_conv(x)                                # Conv1d 1x1, input_size -> hidden_size
cond = broadcast(condition) over time
h_c  = concat([h, cond], channel axis)            # hidden_size + embed_size
out  = left_conv(h) * right_conv(h_c)             # left 完全看不到 `condition`
out  = out_conv(out)                              # hidden_size -> input_size
return out + res
```

`left_conv` 是 `Conv1d → ChanLN → PReLU → Dropout`(只有內容);
`right_conv` 是 `Conv1d → ChanLN → PReLU → Dropout → Sigmoid`
(真正的 gate,同時依據 `x` 與 `condition`)。

**Parameters:**
- `x` – `[N, input_size, T]`
- `condition` – `[N, embed_size]`

**Returns:** `[N, input_size, T]`。

---

## Class: `FiLM`

**Feature-wise Linear Modulation。** 與「教科書版」FiLM 不同(教科書版
scale/bias 只由條件向量決定),這裡的 `cond_scale` / `cond_bias` 是從
`x` 與 broadcast 過的條件向量**串接**後算出來的——所以 affine 參數
也會感知內容,不是單純只取決於 `condition`。

```python
FiLM(feats_size: int, embed_size: int, input_norm: bool = True)
```

**Parameters:**
- `feats_size` – `x` 的 feature 維度
- `embed_size` – 條件化 embedding 維度
- `input_norm` – 若為 `True`(預設),在計算 modulation 之前先對 `x` 套用 `nn.LayerNorm(feats_size)`(沿 channel 軸)

**Reference:** Perez et al., "FiLM: Visual Reasoning with a General
Conditioning Layer," AAAI 2018。

### `forward(x, condition) -> Tensor`

```
x        = LayerNorm(x)  if input_norm else x
cond_cat = concat([x, broadcast(condition)], channel axis)
scale    = cond_scale(cond_cat)     # Conv1d 1x1, feats_size+embed_size -> feats_size
bias     = cond_bias(cond_cat)      # Conv1d 1x1, feats_size+embed_size -> feats_size
return scale * x + bias
```

**Parameters:**
- `x` – `[N, feats_size, T]`
- `condition` – `[N, embed_size]`

**Returns:** `[N, feats_size, T]`。

---

## Class: `SplitMerge`

**不是 channel split。** 這是 DPRNN/SkiM 風格的**時間軸**、50% 重疊
分段機制:把長序列切成固定大小、彼此重疊一半的 segment,給
dual-path(intra-chunk / inter-chunk)處理用,之後再用 overlap-average
的方式把它們拼回去。

```python
SplitMerge(seg_size: int, seg_overlap: bool = True)
```

**Parameters:**
- `seg_size` – 每個 chunk 的長度(frame 數)
- `seg_overlap` – 會被接受並存在 instance 上,但目前**完全沒有作用**:`split`/`merge` 都是 `@staticmethod`,完全不會讀取 `self`(或 `self.seg_overlap`),永遠使用寫死的 50% stride(`seg_stride = seg_size // 2`)。目前所有呼叫端(`test/test_lobe.py`、`puresound/nnet/dprnn.py`)都是直接在 class 上呼叫 `SplitMerge.split(...)` / `SplitMerge.merge(...)`,從來沒有真的建構過 instance——所以建構子與 `seg_overlap` 目前完全沒有被用到。

> **本次已修正:** `__init__` 先前從未呼叫 `super().__init__()`,導致
> instance 缺少 `nn.Module` 內部的 `_parameters`/`_buffers`/`_modules`
> 記錄機制——對單純儲存 int 屬性來說沒有影響,但只要對 instance 呼叫
> `.to(device)`、`.state_dict()`、`repr()`,或是做 module tree 的
> traversal(如 `.parameters()`)都會拋出 `AttributeError`。因為
> `__init__` 只會儲存純粹的 `int`/`bool` 屬性(沒有
> `nn.Parameter`/子 module),加上 `super().__init__()` 是純粹的衛生性
> 修正,對目前實際使用的 static-method 呼叫模式沒有任何行為上的改變。

### `split(x: Tensor, seg_size: int) -> Tuple[Tensor, int]`(staticmethod)

**Parameters:** `x` – `[N, C, T]`

先把 `T` 補零到能被 `seg_size` 寬、50% 重疊的 segment 整除地切分,
再從兩個各自位移半個 segment 的視角組出這些 segment(標準的 DPRNN
分段技巧)。

**Returns:** `(segments, rest)`——`segments`:`[N, S, K, C]`(`S` 為
segment 數,`K = seg_size`);`rest` 為新增的 padding 長度,`merge` 需要
用到。

### `merge(x: Tensor, rest: int) -> Tensor`(staticmethod)

**Parameters:** `x` – `[N, S, K, C]`,`rest` —— 來自對應的 `split` 呼叫

把分段時的兩個半段做 overlap-average 加回去,再裁掉 `rest`,還原分段
之前的樣子。**Returns:** `[N, C, T]`。

---

## Class: `MovingAverage1D`

對**每個時間步一個純量**的訊號(不是多 channel 的 feature map)做簡單
moving average——例如平滑一條增益曲線或 VAD 機率曲線。

```python
MovingAverage1D(
    kernel_size: int,
    stride: int,
    add_padding: bool = False,
    causal: bool = True,
)
```

**Parameters:**
- `kernel_size` / `stride` – 傳給內部的 `nn.AvgPool1d`
- `add_padding` – 若為 `True`,pooling 之前先補零,讓輸出長度(大致)維持不變
- `causal` – 當 `add_padding=True` 時,若為 `True` 只在左側補零(`kernel_size - 1` 個);否則兩側對稱補零(各 `kernel_size // 2` 個)

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, T]`

**Returns:** `[N, T']`(`add_padding=True` 時 `T' = T`,否則會變短)。

---

## Function: `spectral_compression`

```python
spectral_compression(x: Tensor, alpha: float = 0.3, dim: int = 1) -> Tensor
```

原意是做保留相位的 power-law magnitude 壓縮:沿 `dim` 把 `x` 拆成
實部/虛部,算出 `mag = sqrt(re**2+im**2+1e-8)`,取 `alpha` 次方,
再與原本的相位組回一個 complex tensor。

> **已驗證的 bug:** 相位重組那行寫的是
> `mag.pow(alpha) * torch.exp(1j * torch.angle(torch.atan2(_im, _re)))`。
> `torch.atan2(_im, _re)` 本身**已經是**相位角(一個實數 tensor);外面
> 多套的 `torch.angle(...)` 接著又對*這個實數*取角度,而任何實數的
> `torch.angle` 只會是 `0`(正數)或 `π`(負數)——把真正的相位壓縮成一個
> 二元的正負號,其餘資訊全部遺失。已實測驗證:照原樣重建出來的結果
> **不會**符合正確、保留相位的重建(`torch.allclose` 為 `False`;
> 輸出的虛部幾乎全部是 0)。推測原本想寫的應該是
> `torch.exp(1j * phase)`,不需要多餘的 `torch.angle(...)`。
>
> **目前是無效狀態,本次未修正:** `DPARNblock2D` / `DPRNNblock2D` 都把
> 這個呼叫包在建構子的 `spectral_compress: bool = False` 旗標之後,
> 且 `egs/` 底下**每一個**設定到這個旗標的 recipe 都明確寫
> `spectral_compress: False`——所以這個 bug 目前不會影響任何已訓練或
> 部署的 model。也沒有任何測試涵蓋 `spectral_compress=True`。已另外
> 標記出來待修,本次文件工作不處理。

**Parameters:**
- `x` – 實部/虛部沿 `dim` 疊在一起(例如 `dim=1` 時為 `[N, 2*C, T, ...]`)
- `alpha` – 壓縮指數
- `dim` – 實部/虛部疊放的軸

**Returns:** 一個 `torch.complex64` tensor,沿 `dim` 的大小是 `x` 的一半。

---

## Class: `SpecAugment`

訓練時對 spectrogram 做隨機時間/頻率 masking 的擴增。

```python
SpecAugment(
    freq_mask_length: int,
    time_mask_length: int,
    fill_value: float,
    n_freq_mask: int = 1,
    n_time_mask: int = 1,
    prob: float = 0.5,
)
```

**Parameters:**
- `freq_mask_length` / `time_mask_length` – **必填**;各軸的最大 mask 寬度(實際寬度會從 `[0, length]` 均勻取樣)
- `fill_value` – **必填**;寫進被 mask 的 bin/frame 的值——沒有預設值,呼叫端必須自行決定(例如該 feature 的平均值,或 `0.0`)
- `n_freq_mask` / `n_time_mask` – 每個軸各要套用幾個獨立的 mask
- `prob` – **每次呼叫、每個軸各自**的機率閘門——某軸是否做 mask,取決於 `torch.rand(1) < prob`;頻率與時間各自獨立擲一次骰子,所以同一次呼叫可能兩者都沒做、只做其中一個,或兩者都做

**Reference:** Park et al., "SpecAugment: A Simple Data Augmentation Method
for ASR," Interspeech 2019。

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, C, F, T]`(masking 使用
`torchaudio.functional.mask_along_axis`,頻率是 `axis=2`、時間是
`axis=3`——是**4D**,不是單看名稱可能以為的 3D `[N, F, T]`)

只有在 `self.training` 為 `True` 時才會做 masking;`.eval()` 模式下會
原樣回傳 `x`。

**Returns:** mask 過(或原樣)的 tensor,形狀與輸入相同。

## Wiring

`puresound/nnet/features.py` 的 `FeatureEncoder` 用到 `LambdaLayer`
(做一些臨時的 permute)、`Magnitude`,以及 `SpecAugment`(從
`specaug_args` 設定字典建構)。`FiLM`/`SplitMerge` 驅動 `DPRNN` 的
dual-path 分段(`puresound/nnet/dprnn.py`);`FiLM`/`Gate` 驅動 `SkiM`
的條件化(`puresound/nnet/skim.py`);`spectral_compression` 有接到
`DPARN`/`DPCRN`,但目前在所有地方都無法被觸發(見上方的 bug 說明)。

## Example

```python
from puresound.nnet.lobe.trivial import FiLM, SpecAugment, LambdaLayer

film = FiLM(feats_size=256, embed_size=192)
x_cond = film(features, speaker_embedding)

spec_aug = SpecAugment(freq_mask_length=27, time_mask_length=100, fill_value=0.0, n_freq_mask=2)
x_aug = spec_aug(mel_features)  # [N, C, F, T],只有在訓練時才會被 mask
```
