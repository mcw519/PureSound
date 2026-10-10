# puresound.nnet.dparn

English version: [dparn.md](dparn.md)

DPARN 是 dual-path attention recurrent network：沿用 [DPCRN](dpcrn.zh-TW.md) 的
[`Unet`](unet.zh-TW.md) chassis，但 bottleneck 的 intra-frequency 路徑是兩層
self-attention，而不是雙向 LSTM；inter-time 路徑仍是單向 LSTM。同樣的 intra 路徑
在 DPCRN 內可用 `intra_type: attention` 取得；DPARN 則是 block 數可調的獨立模型，
由 `egs/noise_suppression/config/dparn.yaml` 與 DPARN streaming export 使用。

## Class: `DPARNblock2D`

```python
DPARNblock2D(
    input_size: int,       # bottleneck channel 數 CH（channels[-1]）
    hidden_size: int,      # attention feed-forward 寬度，也是 inter-LSTM 的 hidden 寬度
    nhead: int,            # attention head 數；須整除 input_size
    dropout: float = 0.0,  # attention 層與 LSTM 共用
)
```

```python
forward(x: Tensor, intra_skip: bool = True, inter_skip: bool = True) -> Tensor
# x: [N, CH, C, T] -> [N, CH, C, T]
```

- **Intra（頻率，逐 frame）：** reshape 成 `[N*T, C, CH]`；兩層
  `MhaSelfAttenLayer`（[lobe/attention](lobe/attention.zh-TW.md)），第一層帶
  sinusoidal position encoding、第二層不帶，每層都是 Transformer encoder layer
  （self-attention 與寬 `hidden_size` 的 ReLU feed-forward，各自有 residual 與
  LayerNorm）；再接 `Linear` 與 `LayerNorm`；`intra_skip` 時加回 block 輸入。
- **Inter（時間，逐頻率位置）：** reshape 成 `[N*C, T, CH]`；一層單向
  `SingleRNN("LSTM")`（[lobe/rnn](lobe/rnn.zh-TW.md)）與 `LayerNorm`；
  `inter_skip` 時加回 intra 的輸出。

attention 一律以 `causal=False` 呼叫；它是在單一 frame 內沿頻率運算，所以沒有
問題。這兩層以 `improved=False` 建立，因此其 `bidirectional=False` 不起作用。

## Class: `DPARN`

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
    n_dparn_block: int = 2,          # bottleneck 疊幾個 DPARNblock2D
    rnn_hidden: int = 128,           # 每個 block 的 hidden_size
    nhead: int = 1,                  # 每個 block 的 nhead
    spectral_compress: bool = False, # 在 input_norm 前做 |X|^0.3、保留相位
)
```

`input_dim` 到 `delay` 之間除 `transpose_delay` 外的參數都傳給 `Unet.__init__`；
見 [unet](unet.zh-TW.md)。`transpose_delay` 決定 transpose convolution 多出的
`transpose_t_size - 1` 個 frame 從哪一端裁掉；`False` 裁尾端，decoder 保持因果。
`spectral_compress` 套用 `spectral_compression(x, alpha=0.3, dim=1)`
（[lobe/trivial](lobe/trivial.zh-TW.md)）。

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, CH, C, T] 或 [N, C, T]（會 unsqueeze 成 [N, 1, C, T]）
# 回傳 mask，[N, CH, C, T]
```

依序是 `spectral_compress`（可選）、`input_norm`、CNN down path（每層輸出留作
skip）、`n_dparn_block` 個 block，再接 skip concat 或 `skip_conv` 的 CNN up path，
與 [unet](unet.zh-TW.md) 相同。

### `get_args`

回傳所有 constructor 參數，所以 `DPARN(**model.get_args)` 能重建相同架構。

### Config 用法

```yaml
model:
  backbone:
    type: DPARN
    backbone_args:
      input_dim: 256
      norm_type: bN2d
      dropout: 0.1
      channels: [2, 32, 32, 32, 64, 128]
      kernel_t: [2, 2, 2, 2, 2]
      stride_t: [1, 1, 1, 1, 1]
      dilation_t: [1, 1, 1, 1, 1]
      kernel_f: [5, 3, 3, 3, 3]
      stride_f: [2, 2, 1, 1, 1]
      dilation_f: [1, 1, 1, 1, 1]
      delay: [1, 1, 1, 1, 1]
      n_dparn_block: 2
      rnn_hidden: 128
      nhead: 8
```

## Streaming

DPARN 可匯出為逐 frame 的 ONNX 模型。包裝層不改動 offline 模型，明確攜帶 frame
state：CNN down path 的時間 cache、transpose convolution 尚未輸出的部分，以及每個
block 的 LSTM `(h, c)`。attention 路徑不需要跨 frame 的 state。

```python
enhanced_frame, next_state = forward_frame(noisy_frame, state)
# noisy_frame, enhanced_frame: [batch, fft_length // 2 + 1, 2]
```

音訊 buffering、Hann STFT、iSTFT 與 overlap-add 在 ONNX 之外由
`puresound.streaming.StreamingDparnOrt` 處理。支援的 config、export 指令與
runtime API 見 [DPARN streaming ONNX runtime](../../usage/streaming/dparn_onnx.zh-TW.md)。

## 設計說明

- 在單一 frame 內頻率沒有因果限制，所以 intra 路徑可以一次 attend 全部位置；
  attention 把雙向 LSTM 的逐步運算換成每層一次矩陣乘法。
- 只有時間軸攜帶 state，而它仍是單向 LSTM，所以 streaming state 的配置與 `inter_type: lstm` 的 DPCRN
  相同。
