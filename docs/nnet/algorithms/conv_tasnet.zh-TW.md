# puresound.nnet.conv_tasnet

English version: [conv_tasnet.md](conv_tasnet.md)

Status: *library*（見 [nnet index](../index.zh-TW.md)）—— 可透過
`getattr(nnet, "ConvTasNet")` 以 config 方式取用，有 forward smoke test 覆蓋
（`test/test_backbone.py::test_conv_tasnet_backbone` /
`test_conv_tasnet_dvec_backbone`），但目前沒有任何維護中的 recipe 在用。

**Reference:** Luo & Mesgarani, "Conv-TasNet: Surpassing Ideal
Time–Frequency Magnitude Masking for Speech Separation," IEEE/ACM TASLP,
2019.

這個 module 只包含 temporal-convolution 的 mask 估計器 —— **不**包含原始論文裡
那個學到的 waveform encoder/decoder。跟這個 library 裡其他 backbone 一樣，它是
`Encoder -> Features -> Backbone -> Masker -> Decoder` 這條流程裡可替換的中間段
（見 [nnet.features](../features.zh-TW.md) / [nnet.masker](../masker.zh-TW.md)）；
前端可以任選（`ConvEncDec`、`FreeEncDec`……），這個 module 就負責從那個前端產出的
features 去預測 mask。

## Class: `TCN`

單一個非 gated 的 temporal convolution block：先做 bottleneck 投影，接
depthwise dilated conv，再投影回去，最後 residual 相加。

### Constructor

```python
TCN(
    in_channels: int,
    hid_channels: int,
    kernel: int,
    dilation: int,
    dropout: float = 0.0,
    emb_dim: int = 0,
    causal: bool = False,
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",
)
```

**Parameters:**
- `in_channels` – 輸入/輸出的 feature 維度（residual 相加要求兩者一致）
- `hid_channels` – block 內部的 bottleneck 寬度
- `kernel` / `dilation` – depthwise conv 的 kernel size 與 dilation rate
- `dropout` – depthwise conv 之後的 dropout（`0.0` 表示不啟用）
- `emb_dim` – 若不為零，`forward` 的 `embed` 參數會在進入輸入端的 `1x1 Conv`
  （`in_conv`）**之前**先 concat 到 `x` 上；`TCN` 並沒有 `right_conv`
  （那個 submodule 只存在於 `GatedTCN`，見下方）
- `causal` – 會傳給內部的 `DepthwiseSeparableConv1d`
  （[lobe/cnn](../lobe/cnn.md)）：`True` 時只在左側 padding，`False` 時用置中
  padding 保持輸入輸出等長
- `tcn_norm` – `in_conv` 外圍套的 norm layer（透過 [`get_norm`](../lobe/norm.md)）
- `dconv_norm` – depthwise separable conv **內部**用的 norm layer

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x:     [N, in_channels, T]
# embed: [N, emb_dim]，會沿 T 廣播後 concat 到 x 上
# returns: [N, in_channels, T] -- 永遠是單一個 Tensor（沒有 skip-path 的 tuple 回傳）
```

`in_conv -> dconv -> out_conv`，然後 `+ x`（residual）。沒有
`skip_connection` 這個選項，也沒有 tuple 回傳 —— 不管有沒有條件輸入，每次呼叫
都回傳一個 residual 相加後的 tensor。

## Class: `GatedTCN`

Gated 版本：用兩條並行分支（`left_conv`、`right_conv`）相乘，取代單一條 conv
路徑；條件輸入也可以選擇用 FiLM 而不是用 concatenation 注入。

### Constructor

```python
GatedTCN(
    in_channels: int,
    hid_channels: int,
    kernel: int,
    dilation: int,
    dropout: float = 0.0,
    emb_dim: int = 0,
    causal: bool = False,
    tcn_norm: str = "gLN",
    use_film: bool = False,
)
```

**Parameters:** 跟 `TCN` 一樣，外加 `use_film` —— **預設關閉**。FiLM
conditioning 並不是這個 block 的招牌功能，只是 `embed` 兩種 wiring 方式的其中一種：
- `use_film=False`（預設）：`embed` 沿時間軸廣播後 concat 到 `right_conv` 的輸入上
  （`right_conv` 的 in-channels 是 `hid_channels + emb_dim`）—— 這是唯一一個
  「concate in right_conv's input」這句被複製貼上的 docstring 描述真正準確的情境。
- `use_film=True`：改成用 `embed` 去驅動兩個 `1x1 Conv`（`cond_scale`、
  `cond_bias`），對 `left_conv` 的輸出做 FiLM 調變（`right_conv` 的 in-channels
  維持 `hid_channels`，不做 concatenation）。

跟 `TCN` 不同的是，這裡 `causal=True` 還會把輸出裁掉一段
（`x[..., :-self.padd]`），而不是靠對稱 padding，因為 left/right conv 是在
causal 那一側 pad 了 `(kernel-1)*dilation`。

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x: [N, in_channels, T] -> 回傳 [N, in_channels, T]
```

`x = left_conv(in_conv(x)) * right_conv(x_r)`（`x_r` 是 concat 過或 FiLM
調變過的版本），然後 `out_conv`，最後 residual 相加（causal 時會裁切）。

## Class: `ConvTasNet`

被 export 出來的 backbone（`from puresound.nnet import ConvTasNet`）——
把 `per_tcn_stack` 個 `TCN`/`GatedTCN` block 疊成一組 stack，dilation
按指數成長，重複 `repeat_tcn` 組 stack，並可選擇在指定層上以 speaker embedding
做條件化（target-speaker extraction）。

### Constructor

```python
ConvTasNet(
    input_dim: int = 512,
    embed_dim: int = 256,
    embed_norm: bool = False,
    tcn_layer: str = "normal",       # "normal" -> TCN，"gated" -> GatedTCN
    tcn_kernel: int = 3,
    tcn_dim: int = 256,
    tcn_dilated_basic: int = 2,
    per_tcn_stack: int = 5,          # 每組 stack 的 block 數；dilation = tcn_dilated_basic ** i
    repeat_tcn: int = 4,             # stack 組數
    tcn_with_embed: List = [1, 0, 0, 0, 0],  # 長度必須 == per_tcn_stack
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",         # tcn_layer == "gated" 時會被忽略
    causal: bool = False,
)
```

**Parameters:**
- `input_dim` – 輸入輸出的 feature（channel）維度；整個 stack 全程不變
  （每個 `TCN`/`GatedTCN` 都是 shape-preserving 的）
- `embed_dim` – 用於 target-speaker conditioning 的 speaker embedding 維度；
  如果 `tcn_with_embed` 裡沒有任何一項是 `1`，這個參數就無關緊要
- `embed_norm` – 條件化之前先對 `dvec` 做 L2 normalize
- `tcn_dilated_basic` – dilation 的底數；一組 stack 裡第 `i` 個 block 的
  dilation 是 `tcn_dilated_basic ** i`，所以預設底數 `2`、`per_tcn_stack=5`
  時，一組 stack 涵蓋 dilation `1, 2, 4, 8, 16`，下一組 stack 又從 `1` 重新開始
- `tcn_with_embed` – 每個 block 的旗標（長度為 `per_tcn_stack`，會 assert），
  標記*每一組* stack 裡哪些 block 會接收 `dvec`
- `dconv_norm` – 只有 `tcn_layer="normal"` 時才有意義；`GatedTCN` 沒有這個開關
  （共用同一個 block 的 `UnetTcn`，如果你在 `tcn_layer="gated"` 時還設了這個值，
  會印出警告）

### `forward(x, dvec=None) -> Tensor`

```python
forward(x: Tensor, dvec: Optional[Tensor] = None) -> Tensor
# x:    [N, input_dim, T]
# dvec: [N, embed_dim]，只有在 tcn_with_embed[i] == 1 時才需要
# returns: [N, input_dim, T] -- feature domain 上的 mask（由 Masker 套用）
```

### `get_args` property

回傳每個 constructor 參數組成的 `Dict` —— 這個 library 裡通用的
checkpoint 重建 pattern（可對照 [`DPCRN.get_args`](dpcrn.zh-TW.md)、
[`Unet.get_args`](unet.zh-TW.md)）。

### Example（對照 `test/test_backbone.py`）

```python
from puresound.nnet import ConvTasNet

# 單純分離/增強，不做條件化
model = ConvTasNet(
    input_dim=512, embed_dim=0, embed_norm=True,
    tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2,
    per_tcn_stack=8, tcn_with_embed=[0] * 8,
    tcn_norm="gLN", dconv_norm="gGN", causal=False, tcn_layer="normal",
)
mask = model(torch.rand(1, 512, 100))  # [1, 512, 100]

# Target-speaker extraction：每組 stack 8 個 block 裡，前 3 個看得到 dvec
model = ConvTasNet(
    input_dim=512, embed_dim=192, embed_norm=True,
    tcn_kernel=3, tcn_dim=256, repeat_tcn=3, tcn_dilated_basic=2,
    per_tcn_stack=8, tcn_with_embed=[1, 1, 1, 0, 0, 0, 0, 0],
    tcn_norm="gLN", dconv_norm="gGN", causal=False, tcn_layer="normal",
)
mask = model(torch.rand(1, 512, 100), torch.rand(1, 192))
```
