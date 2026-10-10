# puresound.nnet.lobe.cnn

English version: [cnn.md](cnn.md)

1-D depthwise-separable convolution（Conv-TasNet TCN 的核心），以及兩個給
`[N, CH, F, T]` 時頻圖用的 Fast Fourier Convolution 區塊。

## Class: `DepthwiseSeparableConv1d`

```python
DepthwiseSeparableConv1d(
    in_channels: int,
    out_channels: int,
    hid_channels: Optional[int] = None,  # 有設定時：先以 1x1 conv + norm + PReLU 轉到此寬度
    norm_cls: str = "gGN",               # 給 norm.get_norm 的名稱
    kernel: int = 3,
    stride: int = 1,
    dilation: int = 1,
    skip: bool = False,                  # 加上 Conv1d(in_channels, out_channels, 1)(x)
    causal: bool = False,
)
```

`forward(x [N, in_channels, T]) -> [N, out_channels, T']`；stride 1 且 kernel 為
奇數時 `T' = T`。

```
h = in_conv(x)                          if hid_channels, else x
h = PReLU(norm(depthwise_conv(h)))      # groups = width, kernel, stride, dilation
h = PReLU(norm(pointwise_conv(h)))      # 1x1, width -> out_channels
h = h[..., :-(kernel-1)*dilation]       if causal
h = h + skip_conv(x)                    if skip
```

- causal 時每側 padding 為 `(kernel - 1) * dilation`，再切掉尾端的幀，所以沒有任何
  輸出幀讀到未來。non-causal 時每側 padding 為 `((kernel - 1) // 2) * dilation`。
- `causal=True` 會 assert `norm_cls` 不是 `gLN` 或 `gGN`（兩者對整段序列做正規化）。
  `kernel == 1` 時 padding 為 0，不做切尾。
- `skip` 假設 stride 為 1，因為 skip 路徑沒有 stride。

## Class: `SpectralTransform`

FFC 的 global 分支：沿頻率軸做 real FFT，讓每個輸出位置都看得到整段頻譜。

```python
SpectralTransform(
    in_channels: int,
    out_channels: int,
    kernel_size: Tuple[int, int] = (3, 3),  # (kernel_f, kernel_t)
    stride: Tuple[int, int] = (1, 1),       # (stride_f, stride_t)
    causal: bool = True,                    # 時間只左側 padding；頻率 padding 對稱
)
```

`forward(x [N, in_channels, F, T]) -> [N, out_channels, F', T']`。

```
h = ReLU(BN(conv2d(pad(x))))
H = rfft(h, dim=F);  H = [Re H, Im H] stacked on channels
H = ReLU(BN(conv1x1(H)));  g = irfft(H, dim=F)
y = conv1x1(h + g)
```

`irfft` 以第一個 conv 之後 `h` 的頻率長度為輸出長度，所以奇數長度與偶數長度一樣可以還原。

## Class: `FFC`

Fast Fourier Convolution：channel 分成 local 分支（一般 convolution）與 global 分支
（`SpectralTransform`），兩個方向都有交叉路徑。

```python
FFC(
    in_channels: int,
    out_channels: int,
    alpha: float = 0.3,                     # global 佔比：int(channels * alpha)
    kernel_size: Tuple[int, int] = (3, 3),
    stride: Tuple[int, int] = (1, 1),
    causal: bool = True,
)
```

`forward(x [N, in_channels, F, T]) -> [N, out_channels, F', T']`；輸入切成
`global_in = x[:, :int(in_channels * alpha)]`、`local_in = 其餘`，且

```
global_out = ReLU(BN(SpectralTransform(global_in) + conv(local_in)))
local_out  = ReLU(BN(conv(global_in) + conv(local_in)))
return cat([local_out, global_out], dim=1)
```

輸出把 local channel 放在前面，輸入切分卻把前段讀成 global，所以疊兩層 `FFC` 時，
第一層的 local channel 會送進第二層的 global 分支。

Reference: Shchekotov et al., "FFC-SE: Fast Fourier Convolution for Speech
Enhancement", Interspeech 2022; Chi, Jiang, Mu, "Fast Fourier Convolution",
NeurIPS 2020.

## 用法

`ConvTasNet` 的 `TCN` layer 以 `hid_channels=None`、`skip=False`、
`norm_cls=dconv_norm`（預設 `"gGN"`）包住 `DepthwiseSeparableConv1d`，
前後是它自己的 1x1 輸入與輸出 conv。見 [Conv-TasNet](../conv_tasnet.zh-TW.md)。
`SpectralTransform` 與 `FFC` 在 backbone 中沒有呼叫者。

## 設計說明

- 把 convolution 拆成 depthwise 與 pointwise 兩段，權重數從 `C * C_out * k` 降為
  `C * k + C * C_out`，TCN 才能疊很多層 dilated layer。
- `SpectralTransform` 沿頻率的 FFT 讓單一層就有涵蓋整段頻譜的感受野；小頻率 kernel
  要疊很多層才到得了。
