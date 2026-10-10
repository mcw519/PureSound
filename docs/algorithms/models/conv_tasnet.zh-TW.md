# puresound.nnet.conv_tasnet

English version: [conv_tasnet.md](conv_tasnet.md)

Status: library——config 可用 `type: ConvTasNet` 取用，`test/nnet/test_backbone.py`
有 forward 測試，沒有維護中的 recipe 使用。

**Reference:** Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal Time–Frequency
Magnitude Masking for Speech Separation," IEEE/ACM TASLP, 2019.

Conv-TasNet 的 temporal convolutional network（TCN）mask 估計器，不含論文中學習式的
waveform encoder/decoder。它是 `Encoder -> Features -> Backbone -> Masker -> Decoder`
中的 backbone 段（[nnet.features](features.zh-TW.md)、[nnet.masker](masker.zh-TW.md)），
可搭配任何前端（`ConvEncDec`、`FreeEncDec`）。`TCN` 與 `GatedTCN` 也是
[`UnetTcn`](unet.zh-TW.md) 的 bottleneck。

## Class: `TCN`

一個 residual block：`1x1 conv -> depthwise dilated conv -> 1x1 conv`，再加回輸入。

```python
TCN(
    in_channels: int,        # 輸入/輸出寬度（residual 相加）
    hid_channels: int,       # block 內部寬度
    kernel: int,
    dilation: int,
    dropout: float = 0.0,    # 接在 depthwise conv 之後
    emb_dim: int = 0,        # > 0：in_conv 之前把 embed 串接到 x 上
    causal: bool = False,    # depthwise conv 只在左側 padding
    tcn_norm: str = "gLN",   # in_conv 之後的 norm（lobe.norm.get_norm）
    dconv_norm: str = "gGN", # DepthwiseSeparableConv1d 內部的 norm
)
# forward(x [N, in_channels, T], embed [N, emb_dim] | None) -> [N, in_channels, T]
```

```
x' = cat(x, embed broadcast over T)          # 只在有 embed 時
y  = out_conv(dropout(DSConv(PReLU(norm(in_conv(x'))))))
return y + x
```

depthwise 那一段是 [`DepthwiseSeparableConv1d`](lobe/cnn.zh-TW.md)。`causal=True` 時該層
拒絕全域 norm `gLN` 與 `gGN`（它們會讀整段序列），所以 `dconv_norm` 必須是逐 frame 的
norm，例如 `cLN`。

## Class: `GatedTCN`

兩個平行的 dilated conv 相乘，是以 PReLU 取代 tanh 的 WaveNet 式 gated activation：
`left_conv`（conv、norm、PReLU、dropout）乘上 `right_conv`（同樣結構，最後接 sigmoid）。

```python
GatedTCN(
    in_channels: int,
    hid_channels: int,
    kernel: int,
    dilation: int,
    dropout: float = 0.0,
    emb_dim: int = 0,
    causal: bool = False,
    tcn_norm: str = "gLN",   # 兩個分支內的 norm
    use_film: bool = False,  # embed 如何條件化 right_conv（見下）
)
# forward(x [N, in_channels, T], embed [N, emb_dim] | None) -> [N, in_channels, T]
```

```
h   = in_conv(x)
h_r = cat(h, embed)                                   # use_film=False
h_r = cond_scale(embed) * h + cond_bias(embed)        # use_film=True
y   = out_conv(left_conv(h) * right_conv(h_r))
return y + x
```

- `use_film=False`：`right_conv` 輸入 `hid_channels + emb_dim` 個 channel。
- `use_film=True`：兩個 `1x1` conv 把 `embed` 映成 scale 與 bias（FiLM）；
  `right_conv` 輸入 `hid_channels`。
- 沒有 `dconv_norm`。
- `causal=True`：兩個 conv 兩側各 pad `(kernel - 1) * dilation`，residual 相加前裁掉尾端
  `(kernel - 1) * dilation` 個 frame，因此每個輸出 frame 只依賴當下與過去的輸入。

## Class: `ConvTasNet`

`repeat_tcn` 個 stack，每個 stack 有 `per_tcn_stack` 個 block。每個 stack 的第 `i` 個
block dilation 為 `tcn_dilated_basic ** i`，所以預設 `per_tcn_stack=5`、底數 `2` 時
每個 stack 的 dilation 是 `1, 2, 4, 8, 16`。

```python
ConvTasNet(
    input_dim: int = 512,          # 輸入與輸出的 feature 寬度（shape 不變）
    embed_dim: int = 256,          # speaker embedding 寬度
    embed_norm: bool = False,      # 先對 dvec 做 L2 normalize
    tcn_layer: str = "normal",     # "normal" -> TCN，"gated" -> GatedTCN
    tcn_kernel: int = 3,
    tcn_dim: int = 256,            # 每個 block 的 hid_channels
    tcn_dilated_basic: int = 2,
    per_tcn_stack: int = 5,
    repeat_tcn: int = 4,
    tcn_with_embed: List = [1, 0, 0, 0, 0],  # 每個 block 的 flag；len == per_tcn_stack（assert）
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",       # 只給 TCN；不會傳給 GatedTCN
    causal: bool = False,
)
```

`tcn_with_embed[i] == 1` 讓每個 stack 的第 `i` 個 block 接收 `dvec`
（`emb_dim=embed_dim`）；其他 block 不條件化。`ConvTasNet` 以預設的
`use_film=False` 建 `GatedTCN`。未知的 `tcn_layer` 會 raise `NameError`。

```python
forward(x: Tensor, dvec: Optional[Tensor] = None) -> Tensor
# x:    [N, input_dim, T]
# dvec: [N, embed_dim]，只在某個 tcn_with_embed[i] == 1 時需要
# 回傳 [N, input_dim, T]，為 feature domain 的 mask
```

`get_args` 以 dict 回傳所有 constructor 參數，用於從 checkpoint 重建模型。

### Example

```python
from puresound.nnet import ConvTasNet

# 不條件化
model = ConvTasNet(
    input_dim=512, embed_dim=0, embed_norm=True,
    tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2,
    per_tcn_stack=8, tcn_with_embed=[0] * 8,
    tcn_norm="gLN", dconv_norm="gGN", causal=False, tcn_layer="normal",
)
mask = model(torch.rand(1, 512, 100))                      # [1, 512, 100]

# speaker 條件化：每個 stack 8 個 block 中前 3 個看得到 dvec
model = ConvTasNet(
    input_dim=512, embed_dim=192, embed_norm=True,
    tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2,
    per_tcn_stack=8, tcn_with_embed=[1, 1, 1, 0, 0, 0, 0, 0],
    tcn_norm="gLN", dconv_norm="gGN", causal=False, tcn_layer="normal",
)
mask = model(torch.rand(1, 512, 100), torch.rand(1, 192))  # [1, 512, 100]
```

### 設計說明

- dilation 指數成長，receptive field 為
  `repeat_tcn * (kernel - 1) * (base^per_tcn_stack - 1) / (base - 1) + 1` 個 frame
  （base > 1），參數量只隨 block 數線性增加。
- 不含 encoder/decoder，同一個估計器可以跑在 STFT 或學習式前端上。
- 條件化以 block 為單位，由 recipe 決定 speaker embedding 多早、多頻繁地注入。
