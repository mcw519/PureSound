# puresound.nnet.dparn

English version: [dparn.md](dparn.md)

Status: *active*（見 [nnet index](../index.zh-TW.md)）。Dual-Path Attention RNN
(DPARN) —— 跟 [DPCRN](dpcrn.zh-TW.md) 共用同一個 `Unet` chassis，但 bottleneck
把 DPCRN 那個雙向的 intra-frequency LSTM 換成了兩層疊起來的 self-attention，
inter-time 方向仍維持跟 DPCRN 一樣的單向 LSTM。

## Class: `DPARNblock2D`

Bottleneck block 本體。

### Constructor

```python
DPARNblock2D(
    input_size: int,
    hidden_size: int,
    nhead: int,
    dropout: float = 0.0,
)
```

**Parameters:**
- `input_size` – bottleneck 的 channel 維度（來自外層 `Unet`/`DPARN` 的
  `channels[-1]`）
- `hidden_size` – inter-chunk LSTM 的 hidden 寬度
- `nhead` – 兩層 intra-chunk `MhaSelfAttenLayer` 用的 attention head 數
  （沒有預設值 —— 一律由 `DPARN` 提供）
- `dropout` – 兩層 attention 跟 LSTM 共用同一個值

這裡沒有 `bidirectional` 或 `causal` 這種 constructor 參數。兩層 intra-chunk
attention 都寫死 `bidirectional=False`（這是 `MhaSelfAttenLayer` 裡「LSTM 替代」
的旗標，跟序列方向無關），而且每次呼叫都是在 `forward` 呼叫端傳入
`causal=False`，並不是存在建構時的狀態 —— 這個 block 目前沒有暴露一個真正的
causal-streaming 開關。

### Architecture

- **Intra-chunk（頻率軸，逐個時間 frame 處理）：** 兩層疊起來的
  `MhaSelfAttenLayer`（見 [lobe/attention](../lobe/attention.md)）——
  第一層有 sinusoidal position encoding，第二層沒有 —— 後面接一個 `Linear` +
  `LayerNorm`，取代了 DPRNN/DPCRN 那種 intra-frequency 的 BiLSTM，改用
  self-attention。
- **Inter-chunk（時間軸，逐個頻率 bin 處理）：** 一個單向的
  `SingleRNN("LSTM", ...)` + `LayerNorm`，結構上跟
  [`DPRNNblock2D`](dprnn.zh-TW.md) 的 inter 路徑完全一樣。

### `forward(x, intra_skip=True, inter_skip=True) -> Tensor`

```python
forward(
    x: Tensor,             # [N, CH, C, T]
    intra_skip: bool = True,
    inter_skip: bool = True,
) -> Tensor                # [N, CH, C, T]
```

每條路徑本身外圍都是一個 residual add（`intra_skip`/`inter_skip` 控制的就是
這個 residual 到底要不要真的加上去）；`DPARN` 呼叫時一律用預設值 `True`。

## Class: `DPARN`

跟 `DPCRN` 一樣繼承自 `Unet`（見 [algorithms/unet](unet.zh-TW.md)）：CNN
的 down/up stack完全相同，只是把 bottleneck 換掉。

### Constructor

```python
DPARN(
    input_dim: int = 512,
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 32, 32, 32, 64, 128),
    transpose_t_size: int = 2,
    transpose_delay: bool = False,
    skip_conv: bool = False,
    kernel_t: Tuple = (2, 2, 2, 2, 2),
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (5, 3, 3, 3, 3),
    stride_f: Tuple = (2, 2, 1, 1, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 0, 0, 0),
    n_dparn_block: int = 2,
    rnn_hidden: int = 128,
    nhead: int = 1,
    spectral_compress: bool = False,
)
```

前 15 個參數（`input_dim` … `delay`）會原封不動傳給 `Unet.__init__` ——
每一個對 CNN down/up stack 的作用請見 [algorithms/unet](unet.zh-TW.md)。
DPARN 專屬的參數：
- `n_dparn_block` – bottleneck 疊幾層 `DPARNblock2D`（DPCRN 寫死是 2 層；
  DPARN 把這個變成可設定的）
- `rnn_hidden` – 傳給每個 `DPARNblock2D` 的 `hidden_size`
- `nhead` – 傳給每個 `DPARNblock2D` 的 `nhead`
- `spectral_compress` – 若為 `True`，會在其他步驟之前先對輸入套用
  `spectral_compression(x, alpha=0.3, dim=1)`（magnitude 取 `alpha` 次方、
  phase 保持不變 —— 見 [lobe/trivial](../lobe/trivial.md)）

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, CH, C, T] 或 [N, C, T]（會被 unsqueeze 成 [N, 1, C, T]）
# returns: [N, CH, C, T]
```

`spectral_compress`（可選）→ `Unet.input_norm` → CNN-down（收集 skip
connection）→ 疊 `n_dparn_block` 層 `DPARNblock2D` → CNN-up（skip-concat 或
`skip_conv`，跟 `Unet` 的 up path 完全一致，包括 `transpose_delay` 的裁切慣例 ——
見 [algorithms/unet](unet.zh-TW.md)）。

### `get_args` property

回傳一個 constructor 參數組成的 `Dict`，供 checkpoint 重建用。內容是完整的 ——
每一個 constructor 參數（包含 `nhead` 與 `spectral_compress`）都有存在 `self`
上並回傳，所以 `DPARN(**model.get_args)` 能重建出架構完全相同的模型。

> **本次已修正：** `get_args` 先前同時漏掉了 `nhead`（它根本沒存在 `self` 上，
> 就算列進去也拿不回來）與 `spectral_compress`。從存下來的 args 重建時，
> `nhead` 會被悄悄重設回預設值 `1`，`spectral_compress` 原本設的值也會不見。

### Example（對照 `test/test_backbone.py::test_dparn_backbone`）

```python
from puresound.nnet import DPARN

model = DPARN(
    input_dim=256,
    norm_type="cLN",
    channels=(2, 32, 32, 32, 64, 128),
)
input_x = torch.rand(1, 2, 256, 200)
y = model(input_x)
assert input_x.shape == y.shape
```

## Streaming ONNX Runtime

DPARN 可以匯出成逐 frame 處理的 ONNX 模型，用於低延遲推論。Streaming 這條路徑
不會改動 offline 模型本身，而是在外面包一層明確的 frame state：

- downsampling 路徑用的 CNN 時間軸 cache
- upsampling 路徑用的 transpose-convolution 待處理 cache
- 每個 `DPARNblock2D` 的 LSTM hidden/cell state

這個 ONNX 模型一次吃一個 complex STFT frame：

```python
enhanced_frame, next_state = forward_frame(noisy_frame, state)
```

`noisy_frame` 跟 `enhanced_frame` 的 shape 都是 `[batch, 257, 2]`。音訊緩衝、
Hann STFT、iSTFT、overlap-add 這些都是在 ONNX 之外由
`puresound.streaming.StreamingDparnOrt` 處理。

支援的 config、匯出指令、runtime API 與 Gradio demo 的完整流程，見
[DPARN Streaming ONNX Runtime](../../streaming/dparn_onnx.md)。
