# puresound.nnet.tfgridnet

English version: [tfgridnet.md](tfgridnet.md)

Status: library——config 可用 `type: TFGridNet` 取用，`test/nnet/test_backbone.py`
有 forward 測試，沒有維護中的 recipe 使用。

**References:**
[1] Wang et al., "TF-GridNet: Integrating Full- and Sub-Band Modeling for
Speech Separation," IEEE/ACM TASLP, 2023.
[2] Cornell et al., "Multi-Channel Target Speaker Extraction with Refinement:
The WavLab Submission to the Second Clarity Enhancement Challenge," Clarity
CEC2 workshop, 2022; arXiv:2302.07928.

時頻域 backbone：strided `Conv2d` 先縮減頻率，`n_block` 個 `GridBlock` 各自依序跑
沿頻率軸的 BiLSTM、沿時間軸的 LSTM、以及跨時間的 full-band self-attention，最後
`ConvTranspose2d` 還原成輸入的 shape。

## Class: `ContextFeature`

把最後一軸的位移拷貝沿 channel 軸疊起來。

```python
ContextFeature(num_right: int, num_left: int, equal_length: bool = True)
# forward: [N, C, T] -> [N, C * (num_right + 1 + num_left), T]   (equal_length=True)
```

- `num_right` – 過去的 tap；輸出 channel block 的順序是
  `[x 延遲 num_right, ..., x 延遲 1, x, x 提前 1, ..., x 提前 num_left]`。
- `num_left` – 未來的 tap；每一個都需要當下之後的輸入 frame，所以在時間軸上
  look-ahead 就是從這裡來的。
- `equal_length=True` 以邊界 frame 補位，長度維持 `T`。`False` 則補零，並從前端裁掉
  `num_right`、從後端裁掉 `num_left` 個 frame。

## Class: `IntraSpectralLayer`

```python
IntraSpectralLayer(channels: int, kernel_size: int, hidd_size: int)
# forward: [N, CH, F, T] -> [N, CH, F, T]
```

逐 frame：沿頻率軸做 `ContextFeature(kernel_size // 2, kernel_size // 2)`、`cLN`、
跨頻率的雙向 LSTM、`ConvTranspose1d(2 * hidd_size, channels, kernel_size)` 並裁掉前
`kernel_size - 1` 個 bin，最後 residual 相加。各 frame 獨立處理，不增加延遲。

## Class: `SubbandTemporalLayer`

```python
SubbandTemporalLayer(channels: int, kernel_size: int, hidd_size: int, n_delay: int = 1)
# forward: [N, CH, F, T] -> [N, CH, F, T]
```

逐頻率 bin：沿時間做 `ContextFeature(num_right=n_delay,
num_left=kernel_size - n_delay - 1)`、`cLN`、單向 LSTM、
`ConvTranspose1d(hidd_size, channels, kernel_size)` 並裁成 `[n_delay : n_delay + T]`，
最後 residual 相加。

不論 `n_delay` 為何，這層都往前看 `kernel_size - 1` 個 frame：context 貢獻
`kernel_size - n_delay - 1` 個未來 frame，transpose-conv 的裁切貢獻 `n_delay` 個。
`n_delay` 決定 look-ahead 落在哪裡，不決定有多少。

## Class: `FullbandSelfAttention`

跨時間的 multi-head self-attention，每個 query 與 key 都是一整個頻譜 frame。

```python
FullbandSelfAttention(
    fdim: int,                     # 頻率 bin 數（LayerNorm2D 的大小）
    channels: int,                 # 輸入/輸出 channel；須可被 n_head 整除（assert）
    channels_qk: int,              # 每個 head 的 Q/K channel 數
    n_head: int,
    attent_range: Optional[int] = None,
)
# forward(x [N, CH, F, T]) -> (x [N, CH, F, T], atten_mat [N, n_head, T, T])
```

每個 head 各有自己的 `1x1 Conv2d -> PReLU -> LayerNorm2D`，分別產生 Q、K
（`channels_qk` 個 channel）與 V（`channels // n_head` 個 channel）。Q、K、V 攤平成每個
frame 一個向量（Q/K 為 `channels_qk * F` 維），接著

```
A   = softmax(Q^T K / sqrt(channels_qk * F) + mask)       # 每個 head 一個 [T, T]
out = x + proj(concat_heads(A V^T))
```

`proj` 是 `1x1 Conv2d -> PReLU -> LayerNorm2D`。mask 一律擋住未來 frame。
`attent_range=None` 允許所有過去 frame；整數 `R` 另外擋住往回 `R` 步或更早的 frame，
所以每個 frame 只 attend 到自己與前 `R - 1` 個 frame。`forward` 一律回傳 attention matrix。

## Class: `GridBlock`

```python
GridBlock(
    ch_dim: int,
    f_dim: int,
    hid_dim: int,
    kernel_t: int = 3,
    kernel_f: int = 3,             # 須為奇數（assert）
    n_head: int = 4,
    approx_qk_dim: int = 8,        # attention 的 channels_qk
    n_delay: int = 0,              # < kernel_t（assert）
    attent_range: Optional[int] = None,
)
# forward(x [N, ch_dim, f_dim, T], return_atten_mat=False)
#   -> x；return_atten_mat 時為 (x, atten_mat)
```

`IntraSpectralLayer(ch_dim, kernel_f, hid_dim)` ->
`SubbandTemporalLayer(ch_dim, kernel_t, hid_dim, n_delay)` ->
`FullbandSelfAttention(f_dim, ch_dim, approx_qk_dim, n_head, attent_range)`。

## Class: `TFGridNet`

```python
TFGridNet(
    inp_channel_dim: int = 2,      # real/imag 疊起來為 2
    input_dim: int = 256,          # 頻率 bin 數
    input_norm: str = "gLN",       # "gLN" | "LayerNorm2D"，接在輸入 conv 之後
    channel_dim: int = 32,         # GridBlock 的 ch_dim
    lstm_dim: int = 128,           # GridBlock 的 hid_dim
    n_block: int = 6,
    block_delay_frames: int = 0,   # GridBlock 的 n_delay，所有 block 共用
    kernel_f_size: int = 5,        # GridBlock 的 kernel_f
    kernel_t_size: int = 5,        # GridBlock 的 kernel_t
    f_stride: int = 4,             # 頻率降採樣倍率；kernel_f_size >= f_stride（assert）
    n_head: int = 4,
    channel_qk: int = 4,           # GridBlock 的 approx_qk_dim
    attent_range: int = 100,
)
```

```python
forward(x: Tensor) -> Tensor
# x: [N, inp_channel_dim, input_dim, T]，或 [N, input_dim, T]（unsqueeze 成單一
#    channel，因此 inp_channel_dim 必須為 1）
# 回傳與 x 相同的 shape
```

- `in_conv`：`ZeroPad2d((2, 0, 1, 1))`（時間往過去補兩個 frame、頻率兩側各補一個 bin）、
  `Conv2d(inp_channel_dim, channel_dim, 3x3, stride=(f_stride, 1))`，再接 `input_norm`。
  block 在 `input_dim // f_stride` 個 bin 上運作，所以 `input_dim` 應為 `f_stride` 的倍數。
- `out_conv`：`ConvTranspose2d(channel_dim, inp_channel_dim, 3x3,
  stride=(f_stride, 1), padding=(1, 0), output_padding=(f_stride - 1, 0))`，
  還原 `input_dim` 個 bin；多出的兩個尾端 frame 會被裁掉。

### Causality

輸入與輸出 conv 只用當下與過去的 frame，逐 frame 的 norm（`cLN`、`LayerNorm2D`）與
attention mask 都是 causal。不 causal 的有兩處：

- 每個 `SubbandTemporalLayer` 往前看 `kernel_t_size - 1` 個 frame，因此整個 stack 往前看
  `n_block * (kernel_t_size - 1)` 個 frame，與 `block_delay_frames` 無關；
- `input_norm="gLN"` 用整段 utterance 的統計量做 normalize；`LayerNorm2D` 是逐 frame 的。

### Example

```python
from puresound.nnet import TFGridNet

model = TFGridNet(
    inp_channel_dim=2, input_dim=256, channel_dim=32, lstm_dim=128,
    n_block=6, block_delay_frames=0, kernel_f_size=5, kernel_t_size=5,
    f_stride=4, n_head=4, channel_qk=4, attent_range=100,
)
y = model(torch.rand(1, 2, 256, 1000))  # [1, 2, 256, 1000]
```

### 設計說明

- 每個 block 都交錯進行 sub-band 建模（LSTM）與 full-band 建模（對整個 frame 做
  attention），做法依 [1]。
- `attent_range` 限制一個 frame 能往回 attend 多遠。完整的 `T x T` score matrix 仍會算出
  再套 mask。
- 進 block 前先把頻率降採樣 `f_stride` 倍，逐 bin 的 LSTM 序列數也跟著減少同樣倍數。
