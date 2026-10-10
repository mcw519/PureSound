# puresound.nnet.unet

English version: [unet.md](unet.md)

Status: library——`Unet`、`UnetTcn`、`UnetFsmn` 都可由 config 取用（`type: Unet` 等），
`UnetTcn` 在 `test/nnet/test_backbone.py` 有 forward 測試；沒有維護中的 recipe
直接使用它們。`Unet` 同時是現役 backbone [DPCRN](dpcrn.zh-TW.md) 與
[DPARN](dparn.zh-TW.md) 的 base class。

在 `[frequency, time]` feature map 上的 2-D convolutional encoder/decoder，帶 skip
connection。down path 只在頻率上降採樣（各 recipe 的時間 stride 都是 1）；子類別在
bottleneck 插入時間模型。

## Class: `Unet`

```python
Unet(
    input_dim: int = 512,                    # 頻率 bin 數
    activation_type: str = "PReLU",          # lobe.activation.get_activation
    norm_type: str = "bN2d",                 # lobe.norm.get_norm
    dropout: float = 0.05,                   # 每個 down stage 之後
    channels: Tuple = (1, 1, 8, 8, 16, 16),  # len == n_cnn + 1；第 i 層 channels[i] -> channels[i+1]
    transpose_t_size: int = 2,               # 每個 up stage ConvTranspose2d 的時間 kernel
    skip_conv: bool = False,                 # False：串接 skip；True：1x1 conv 後相加
    kernel_t: Tuple = (5, 1, 9, 1, 1),       # 每個 down stage，時間軸
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (1, 5, 1, 5, 1),       # 每個 down stage，頻率軸
    stride_f: Tuple = (1, 4, 1, 4, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 1, 0, 0),          # 每個 down stage 的 look-ahead frame 數
    multi_output: int = 1,                   # 最後一個 up stage 的 channel 倍數
)
```

六個 kernel/stride/dilation tuple 長度必須相同，皆為 `n_cnn`（assert）。

```python
forward(x: Tensor) -> Tensor
# x: [N, CH, F, T] 或 [N, F, T]（unsqueeze 成 [N, 1, F, T]）
# 回傳 [N, channels[0] * multi_output, F, T]
```

結構：

1. `input_norm`：`iLN(channels[0] * input_dim)`，對 channel 與頻率做逐 frame 的
   layer norm（[lobe/norm](lobe/norm.zh-TW.md)）。
2. 第 `i` 個 down stage：`ZeroPad2d` -> `Conv2d(channels[i], channels[i+1],
   (kernel_f[i], kernel_t[i]), stride, dilation)` -> norm -> activation -> dropout。
   時間 padding 為 `((kernel_t[i] - 1) * dilation_t[i] - delay[i], delay[i])`，也就是
   `delay[i]` 個未來 frame、其餘為過去 frame；頻率 padding 兩側各
   `kernel_f[i] // 2 * dilation_f[i]`。
3. up stage 反向進行：對應 down stage 的 skip 被串接（up conv 輸入變成
   `2 * channels[i+1]` 個 channel），或在 `skip_conv=True` 時經過
   `1x1 Conv2d + activation` 後相加。
   `ConvTranspose2d(kernel=(kernel_f[i], transpose_t_size), stride=(stride_f[i],
   stride_t[i]), dilation=1)` 還原頻率大小；除最後一個 stage 外都接 norm 與
   activation，最後一個是線性輸出。
4. 時間 kernel 為 `K` 的 transpose conv 會多出 `K - 1` 個 frame；`Unet` 從尾端裁掉，
   讓 up path 保持 causal。

在沒有 `transpose_delay` 的情況下（`Unet` 只有這個模式），模型往前看 `sum(delay)` 個 frame。

`multi_output` 只加寬最後一個 up stage 的輸出 channel；`forward` 仍只回傳一個 tensor。
本 module 的子類別與 `DPCRN`/`DPARN` 都沒有把它往下傳，所以對它們而言永遠是 1。

`shape_info() -> (down_shape, up_shape)` 列出某組設定下每個 stage 的頻率大小；
`forward` 不會用到。`get_args` 回傳全部 15 個 constructor 參數。

## Class: `UnetTcn`

在 bottleneck 放一疊 [`TCN`/`GatedTCN`](conv_tasnet.zh-TW.md) block 的 `Unet`。
bottleneck `[N, channels[-1], F', T]` 先攤平成 `[N, channels[-1] * F', T]` 給 TCN stack，
再 reshape 回來；`F'` 是 `input_dim` 經過所有 `stride_f` 相除（無條件進位）後的大小。

```python
UnetTcn(
    embed_dim: int = 0,                # speaker embedding 寬度
    embed_norm: bool = False,          # 先對 dvec 做 L2 normalize
    input_type: str = "RI",            # 會接受但不使用
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 1, 8, 8, 16, 16),
    transpose_t_size: int = 2,
    transpose_delay: bool = False,     # 裁掉前端而非尾端的 frame
    skip_conv: bool = False,
    kernel_t: Tuple = (5, 1, 9, 1, 1),
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (1, 5, 1, 5, 1),
    stride_f: Tuple = (1, 4, 1, 4, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 1, 0, 0),
    tcn_layer: str = "normal",         # "normal" -> TCN，"gated" -> GatedTCN
    tcn_kernel: int = 3,
    tcn_dim: int = 256,
    tcn_dilated_basic: int = 2,
    per_tcn_stack: int = 5,
    repeat_tcn: int = 4,
    tcn_with_embed: List = [1, 0, 0, 0, 0],  # len == per_tcn_stack（assert）
    tcn_use_film: bool = False,        # 條件化的 GatedTCN block 使用 FiLM
    tcn_norm: str = "gLN",
    dconv_norm: str = "gGN",           # 只給 TCN
    causal: bool = False,              # causal 的 TCN block
)
# forward(x [N, CH, F, T] | [N, F, T], dvec [N, embed_dim] | None) -> [N, CH, F, T]
```

- `input_dim` ... `delay` 傳給 `Unet`；`tcn_*` 參數建 stack 的方式與
  [`ConvTasNet`](conv_tasnet.zh-TW.md) 完全相同（dilation `tcn_dilated_basic ** i`，
  `tcn_with_embed` 逐 block 指定）。
- `tcn_layer="gated"` 會記一條 warning，說明 `dconv_norm` 被忽略；`normal`/`gated`
  以外的值 raise `ValueError`。
- `tcn_use_film` 只影響接收 `dvec` 的 `GatedTCN` block。
- `transpose_delay=True` 保留每個 transpose conv 較晚的 frame，在 `sum(delay)` 之外
  每個 up stage 再多往前看 `transpose_t_size - 1` 個 frame。
- `get_args` 回傳除 `input_type` 外的所有參數。

### Example

```python
from puresound.nnet import UnetTcn

model = UnetTcn(
    embed_dim=192, embed_norm=True, input_dim=256,
    activation_type="PReLU", norm_type="gLN",
    channels=(2, 32, 64, 128, 128, 128, 128),
    transpose_t_size=2, transpose_delay=True, skip_conv=False,
    kernel_t=(2, 2, 2, 2, 2, 2), kernel_f=(5, 5, 5, 5, 5, 5),
    stride_t=(1, 1, 1, 1, 1, 1), stride_f=(2, 2, 2, 2, 2, 2),
    dilation_t=(1, 1, 1, 1, 1, 1), dilation_f=(1, 1, 1, 1, 1, 1),
    delay=(0, 0, 0, 0, 0, 0),
    tcn_layer="gated", tcn_kernel=3, tcn_dim=256, tcn_dilated_basic=2,
    per_tcn_stack=5, repeat_tcn=3, tcn_with_embed=[1, 0, 0, 0, 0],
    tcn_norm="gLN", dconv_norm=None, causal=False,
)
y = model(torch.rand(1, 2, 256, 100), torch.rand(1, 192))  # [1, 2, 256, 100]
```

## Class: `UnetFsmn`

在 bottleneck 放一疊 `FSMN` / `ConditionFSMN` block（[lobe/rnn](lobe/rnn.zh-TW.md)）的
`Unet`，攤平方式與 `UnetTcn` 相同。

```python
UnetFsmn(
    embed_dim: int = 0,
    embed_norm: bool = False,
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 1, 8, 8, 16, 16),
    transpose_t_size: int = 2,
    transpose_delay: bool = False,
    skip_conv: bool = False,
    kernel_t: Tuple = (5, 1, 9, 1, 1),
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (1, 5, 1, 5, 1),
    stride_f: Tuple = (1, 4, 1, 4, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 1, 0, 0),
    fsmn_l_context: int = 3,           # memory filter 的過去 tap
    fsmn_r_context: int = 0,           # 未來 tap；0 讓 stack 保持 causal
    fsmn_dim: int = 256,               # projection（memory）寬度
    num_fsmn: int = 8,                 # == len(fsmn_with_embed)（assert）
    fsmn_with_embed: List = [1, 1, 1, 1, 1, 1, 1, 1],  # 1 -> ConditionFSMN，0 -> FSMN
    fsmn_norm: str = "gLN",
    use_film: bool = True,             # ConditionFSMN：用 FiLM 而非串接
)
# forward(x [N, CH, F, T] | [N, F, T], dvec [N, embed_dim] | None) -> [N, CH, F, T]
```

每個 FSMN block 把輸入投影到 `fsmn_dim`，加上涵蓋 `fsmn_l_context` 個過去與
`fsmn_r_context` 個未來 frame 的 depthwise filter，再加上前一個 block 的 projection
（在 block 間傳遞的 memory），最後經 norm 與 ReLU 投影回去。預設的 `fsmn_with_embed`
讓每個 block 都條件化，所以除非改掉這組 flag，`forward` 一定要給 `dvec`；沒給時第一個
`ConditionFSMN` 就會失敗。`transpose_delay` 行為與 `UnetTcn` 相同。`get_args` 回傳所有參數。

## 設計說明

- 頻率與時間各有獨立的 kernel、stride、dilation tuple，因為只有時間有 causality 限制：
  頻率 padding 對稱，時間 padding 由 `delay` 決定位置。
- look-ahead 逐 down stage 設定，所以總延遲 `sum(delay)` 個 frame 在 config 裡一目了然。
- 逐 frame 的 `iLN` 輸入 norm 讓模型可以逐 frame 執行，utterance-level 的 norm 則不行。
