# puresound.nnet.lobe.trivial

English version: [trivial.md](trivial.md)

小型工具 layer：函式包裝、complex 轉 magnitude、兩種條件化 layer（`Gate`、
`FiLM`）、dual-path 分段（`SplitMerge`）、移動平均、power-law 頻譜壓縮，以及
SpecAugment 遮罩。

## Class: `LambdaLayer`

把函式包成 `nn.Module`，讓不帶參數的轉換能放進 `nn.Sequential`。

```python
LambdaLayer(lambda_func: LambdaType)
```

`forward(x, **kwargs)` 回傳 `lambda_func(x, **kwargs)`。`FeatureEncoder` 用它做軸的
permute 與 squeeze。

## Class: `Magnitude`

```
mag = sqrt(re² + im² + 1e-8)        （log1p 時再取 log1p(mag)）
```

```python
Magnitude(
    drop_first: bool = True,   # 丟掉頻率 bin 0（DC）
    log1p: bool = False,       # 回傳 log1p(mag)
)
```

`forward(x)` 接受 `[N, C, T, 2]`（實/虛部在最後一軸）或 `[N, 2C, T]`（實/虛部兩半在
channel 軸）；其他 rank 丟 `TypeError`。回傳 `[N, C, T]`，`drop_first` 時為
`[N, C - 1, T]`。`1e-8` 讓 magnitude 為零時梯度仍有限。

## Class: `Gate`

殘差式的閘控條件化：由 `x` 算出的內容分支，乘上同時看見 `x` 與條件向量的 sigmoid
閘。

```
h   = Conv1d_1x1(x)                                   # input_size -> hidden_size
g   = right_conv([h; broadcast(condition)])           # Conv1d -> ChanLN -> PReLU -> Dropout -> Sigmoid
out = x + Conv1d_1x1(left_conv(h) * g)                # left_conv: Conv1d -> ChanLN -> PReLU -> Dropout
```

```python
Gate(input_size: int, hidden_size: int, embed_size: int, dropout: float = 0.0)
```

`forward(x [N, input_size, T], condition [N, embed_size]) -> [N, input_size, T]`。
`SkiM` 用它做 speaker 條件化。

## Class: `FiLM`

Feature-wise linear modulation（Perez et al., "FiLM: Visual Reasoning with a
General Conditioning Layer", AAAI 2018），scale 與 bias 由特徵**與**條件向量**一起**
算出：

```
x     = LayerNorm(x)                     若 input_norm
c     = [x; broadcast(condition)]        # feats_size + embed_size 個 channel
out   = Conv1d_1x1(c) * x + Conv1d_1x1'(c)
```

```python
FiLM(feats_size: int, embed_size: int, input_norm: bool = True)
```

`forward(x [N, feats_size, T], condition [N, embed_size]) -> [N, feats_size, T]`。
`DPRNN`、`SkiM` 與 `DPCRN` 的 `DPRNNblock2D`（設定 `dvec_dim` 時）使用。affine 參數
同時由特徵與 embedding 算出，讓調變能隨每個 frame 的內容改變，而不只取決於目標是誰。

## Class: `SplitMerge`

dual-path 模型（DPRNN、SkiM）用的時間軸分段：把 `[N, C, T]` 切成 `seg_size` 個 frame、
彼此重疊 50% 的 chunk，之後以重疊平均接回去。

```python
SplitMerge(seg_size: int, seg_overlap: bool = True)
```

`split` 與 `merge` 是 static method，直接在 class 上呼叫；`seg_overlap` 有存下來但
不會被讀——重疊一律是 50%（`seg_stride = seg_size // 2`）。

- `SplitMerge.split(x [N, C, T], seg_size) -> (segments [N, S, K, C], rest)`——在尾端
  補 `rest` 個零 frame、兩端各補 `seg_stride` 個，再交錯兩個錯開半段的 view。
  `K = seg_size`，`S` 為段數。
- `SplitMerge.merge(x [N, S, K, C], rest) -> [N, C, T]`——平均兩個重疊的半段並裁掉
  `rest`。

## Class: `MovingAverage1D`

對每個 frame 一個值的軌跡 `[N, T]`（例如增益或 VAD 機率曲線）做移動平均，內部用
`nn.AvgPool1d`。

```python
MovingAverage1D(
    kernel_size: int,
    stride: int,
    add_padding: bool = False,   # 補零讓輸出維持（約）T 個 frame
    causal: bool = True,         # 有補零時：左邊補 kernel_size - 1 個零；
                                 # 否則兩邊各補 kernel_size // 2 個
)
```

`forward(x [N, T]) -> [N, T']`。

## Function: `spectral_compression`

```python
spectral_compression(x: Tensor, alpha: float = 0.3, dim: int = 1, eps: float = 1e-8) -> Tensor
```

保留相位的 power-law magnitude 壓縮，`|X|^alpha · exp(j·angle(X))`，作用在沿 `dim`
堆疊的實/虛部兩半上：

```python
_re, _im = torch.chunk(x, 2, dim=dim)
mag = (_re.pow(2) + _im.pow(2) + eps).sqrt()
scale = mag.pow(alpha - 1.0)
return torch.cat([_re * scale, _im * scale], dim=dim)
```

回傳與 `x` 同 shape、同 dtype 的實數 tensor。`alpha = 1` 是恆等；`eps` 讓
`mag ** (alpha - 1)` 在原點仍有限。

把兩個部分都乘上 `|X|^(alpha-1)`，與用 `atan2`/`cos`/`sin` 重建是同一個恆等式，但不必
繞經三角函數，而且在靜音 bin 上恰好回傳零（`atan2(0, 0) = 0` 會產生多餘的實部）。
維持堆疊的實數排列，讓實數 Conv2d backbone 能直接把它當前處理步驟使用。

`DPCRN` 與 `DPARN` 在以 `spectral_compress: True`（預設 `False`）建構時，以
`alpha = 0.3` 對輸入套用它。DPCRN 串流 runner 不接受 `spectral_compress=True`。

## Class: `SpecAugment`

訓練用的隨機時間與頻率遮罩（Park et al., "SpecAugment: A Simple Data Augmentation
Method for Automatic Speech Recognition", Interspeech 2019）。

```python
SpecAugment(
    freq_mask_length: int,   # 遮罩最大寬度（bin）；寬度從 [0, length] 抽
    time_mask_length: int,   # 遮罩最大寬度（frame）
    fill_value: float,       # 寫進被遮罩位置的值；沒有預設
    n_freq_mask: int = 1,
    n_time_mask: int = 1,
    prob: float = 0.5,       # 每次呼叫、每個軸各自判定
)
```

`forward(x [N, C, F, T])` 只在 training 模式下遮罩（透過
`torchaudio.functional.mask_along_axis`，`axis=2` 為頻率、`axis=3` 為時間）；頻率與
時間各自抽一次 `torch.rand(1) < prob`。eval 模式下原樣回傳 `x`。`FeatureEncoder` 由
`specaug_args` 建構它。

## 範例

```python
from puresound.nnet.lobe.trivial import FiLM, SpecAugment

film = FiLM(feats_size=256, embed_size=192)
x_cond = film(features, speaker_embedding)             # [N, 256, T]

spec_aug = SpecAugment(freq_mask_length=27, time_mask_length=100,
                       fill_value=0.0, n_freq_mask=2)
x_aug = spec_aug(mel_features)                         # [N, C, F, T]
```
