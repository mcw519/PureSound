# puresound.nnet.tfgridnet

English version: [tfgridnet.md](tfgridnet.md)

Status: *library*（見 [nnet index](../index.zh-TW.md)）—— 可透過
`getattr(nnet, "TFGridNet")` 以 config 方式取用，有 forward smoke test 覆蓋
（`test/test_backbone.py::test_tfgrid_backbone`），但目前沒有任何維護中的
recipe 在用。

**References:**
[1] Wang et al., "TF-GridNet: Integrating Full- and Sub-Band Modeling for
Speech Separation," IEEE/ACM TASLP, 2023.
[2] "Multi-Channel Target Speaker Extraction with Refinement: The WavLab
Submission to the Second Clarity Enhancement Challenge."

## Class: `ContextFeature`

把一個 tensor 最後一軸展開成局部的 context window，做法是把位移過的拷貝
沿 channel 軸串接起來。**建構上就是不對稱的** —— 它吃的是各自獨立的
past/future 數量，不是單一個對稱的 `context_size`。

### Constructor

```python
ContextFeature(num_right: int, num_left: int, equal_length: bool = True)
```

**Parameters:**
- `num_right` – **過去**的 tap 數量（每多一個就再往回一步，作法是把訊號
  右移並在前端補值）。這些完全不需要額外等待就能算出來 —— 資料本來就已經有了。
- `num_left` – **未來**的 tap 數量（左移，在尾端補值）。每多一個，就代表要
  產生輸出 frame `T` 時，輸入必須已經拿到 `T+i` 的資料才行 —— look-ahead
  latency 就是從這裡來的。
- `equal_length` – 若為 `True`，邊界 frame 會用重複邊緣值來補，這樣輸出長度
  才會跟輸入一樣（只有 channel 軸會變大，疊上 `(num_right + 1 + num_left)`
  份拷貝）。若為 `False`，邊界改用補零，然後*時間軸本身*會在頭尾各裁掉
  `num_right`/`num_left`。

已經直接驗證過（對一個遞增序列跑 `ContextFeature.add_context`，
`num_right=2, num_left=1`）：疊在 channel 上的輸出是
`[x 延遲 2、x 延遲 1、x 本身、x 提前 1]`——確認 `num_right` 的 tap 是往回看，
`num_left` 的 tap 是往前看，順序不會反過來。

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, C, T] -> [N, C * (num_right + 1 + num_left), T]（equal_length=True 時）
```

在 `GridBlock` 裡用了兩次，分別作用在兩個不同的軸上：
- `IntraSpectralLayer` 呼叫時用 `num_right == num_left == kernel_size // 2`，
  作用在**頻率**軸上（對稱 —— 這裡沒有 past/future 的問題，純粹是鄰近的頻率
  bin，跟時間上的 causality 完全無關）。
- `SubbandTemporalLayer` 呼叫時用 `num_right=n_delay,
  num_left=kernel_size - n_delay - 1`，作用在**時間**軸上，其中 `n_delay`
  就是 `GridBlock` 的 causality 開關（見下方說明）。

## Class: `IntraSpectralLayer`

頻率軸的 context 加上雙向 LSTM，對每個時間 frame 各自獨立處理。
`IntraSpectralLayer(channels, kernel_size, hidd_size)`：先建出 `kernel_size`
個頻率鄰居的 context（用 `cLN` 做 normalize），跑一個 `bidirectional=True` 的
`nn.LSTM`，再用 `ConvTranspose1d` 投影回 `channels`（kernel size 造成的多餘
長度用裁掉前段來補償），最後 residual 相加。`[N, CH, C, T] -> [N, CH, C, T]`。

## Class: `SubbandTemporalLayer`

時間軸的 context 加上**單向** LSTM，對每個頻率 bin 各自獨立處理。

```python
SubbandTemporalLayer(channels: int, kernel_size: int, hidd_size: int, n_delay: int = 1)
```

`n_delay` 把 `kernel_size` 寬的時間窗切成 `n_delay` 個過去 tap 跟
`kernel_size - n_delay - 1` 個未來 tap，用在 LSTM 之前的 context 建構上
（透過上面的 `ContextFeature`），接著 LSTM 之後的 `ConvTranspose1d`
（`inter_linear`）會用對應的偏移量做裁切
（`x[..., n_delay : n_delay + nframes]`）—— 跟 `Unet` 的 `transpose_delay`
用的是同一套「裁前段 vs. 裁後段」慣例。已經用實驗驗證過（對一個脈衝輸入做
autograd 分析，`kernel_size=3`）：**測試過從 `0` 到 `kernel_size - 1` 的每個
`n_delay`**，這個 block 對未來的整體 look-ahead 涵蓋範圍都是固定的
`kernel_size - 1` frame —— 改變 `n_delay` 只是改變這個 look-ahead 是在哪裡
被實現的（直接在 LSTM 之前的 context 裡、還是透過 LSTM 之後的裁切），本身並
不會把它消除掉。這個 `n_delay` 就是 `GridBlock` 的 `n_delay` —— 見下方
`TFGridNet` 的 causality 說明。`[N, CH, C, T] -> [N, CH, C, T]`。

## Class: `FullbandSelfAttention`

沿**時間**軸做 multi-head self-attention，而且**每個 head 投影的是整個
`[CH, F]` 的網格**（不是逐頻率 bin）—— 這就是 TF-GridNet「full-band」的那一半，
跟 `SubbandTemporalLayer` 逐 bin 的 LSTM互補。先前完全沒有文件記錄過。

### Constructor

```python
FullbandSelfAttention(
    fdim: int,
    channels: int,
    channels_qk: int,
    n_head: int,
    attent_range: Optional[int] = None,
)
```

**Parameters:**
- `fdim` – 頻率 bin 數（給每個 Q/K/V head 內部的 `LayerNorm2D` 用）
- `channels` – 輸入/輸出 channel 數；必須能被 `n_head` 整除（每個 head 分到
  `channels // n_head` 個 value channel）
- `channels_qk` – **每個 head** 的 Q/K 投影寬度（跟 `channels // n_head`
  無關，跟一般 transformer Q=K=V 均分的做法不一樣）
- `n_head` – head 數，每個 head 各自有一組 `Conv2d -> PReLU ->
  LayerNorm2D` 投影分別給 Q、K、V（每個角色是一個 `ModuleList`，不是共用一組
  投影再 reshape —— 每個 head 真的是各自獨立的權重）
- `attent_range` – `None` = 標準的 causal mask（可以看到目前這個 frame
  跟所有過去的 frame，`triu(diagonal=1)` 設成 `-inf`）；設成整數則是**有限範圍的
  causal window**，額外把超過 `attent_range` 個 frame 之前的過去也遮掉
  （`tril(diagonal=-attent_range)` 也設成 `-inf`）。不管哪種情況，上三角
  （未來的 frame）永遠被遮住 —— 這一層不管 `attent_range` 怎麼設，都絕對不會
  看到未來。

### `forward(x, return_atten_mat=False) -> Tensor | Tuple[Tensor, Tensor]`

```python
forward(x: Tensor, return_atten_mat: bool = False) -> Tensor
# x: [N, CH, C, T] -> [N, CH, C, T]（可選回傳 [N, n_head, T, T] 的 attention matrix）
```

每個 head 會把自己的 `[channels_qk 或 channels//n_head, F]` 投影攤平成
每個時間點一個向量，所以 attention score 是用**整個頻譜 frame**去算的
（`Q^T K / sqrt(embed_dim)`），不是逐頻率 bin —— 這就是「full-band」的意思：
每個 frame 是拿其他 frame 的整段頻譜內容一次拿來對照。對遮罩過的 score 做
softmax、加權合併 `V`、把各個 head 的結果串接回去、`1x1 Conv2d` 投影、
residual 相加。

## Class: `GridBlock`

把上面三層按固定順序組裝起來 —— 這就是 TF-GridNet 的一個「grid」。

```python
GridBlock(
    ch_dim: int, f_dim: int, hid_dim: int,
    kernel_t: int = 3, kernel_f: int = 3,     # kernel_f 必須是奇數
    n_head: int = 4, approx_qk_dim: int = 8,
    n_delay: int = 0,                          # 必須小於 kernel_t
    attent_range: Optional[int] = None,
)
# forward(x [N, ch_dim, f_dim, T], return_atten_mat=False):
#   x = intra_frame_spectral(x)   # 頻率軸的 BiLSTM
#   x = subband_temporal(x)       # 時間軸的 causal LSTM，n_delay 可調
#   x, atten_mat = fullband_self_attention(x)   # causal（限定範圍）的 full-band attention
#   回傳 x，若 return_atten_mat 為真則回傳 (x, atten_mat)
```

## Class: `TFGridNet`

被 export 出來的 backbone（`from puresound.nnet import TFGridNet`）：一個
3x3、對頻率做 downsample 的 `Conv2d`、疊 `n_block` 個 `GridBlock`、再用對應的
`ConvTranspose2d` 把 channel/頻率解析度轉回輸入的大小。

### Constructor

```python
TFGridNet(
    inp_channel_dim: int = 2,
    input_dim: int = 256,
    input_norm: str = "gLN",       # "gLN" 或 "LayerNorm2D"
    channel_dim: int = 32,
    lstm_dim: int = 128,
    n_block: int = 6,
    block_delay_frames: int = 0,   # GridBlock 的 n_delay，每個 block 共用
    kernel_f_size: int = 5,
    kernel_t_size: int = 5,
    f_stride: int = 4,             # 頻率 downsample 倍率(kernel_f_size >= f_stride)
    n_head: int = 4,
    channel_qk: int = 4,
    attent_range: int = 100,       # 限定範圍的 causal attention window，見 FullbandSelfAttention
)
```

**Parameters:**
- `inp_channel_dim` – 輸入 channel 數（real/imag 疊起來的話是 `2`）
- `input_dim` – 輸入的頻率 bin 數；輸入 conv 做完 downsample 之後，每個
  `GridBlock` 實際處理的是 `input_dim // f_stride` 個 bin
- `channel_dim` / `lstm_dim` – `GridBlock` 的 `ch_dim` / `hid_dim`，每個
  block 共用
- `n_block` – 疊幾個 `GridBlock`
- `block_delay_frames` – 傳給每個 block 的 `SubbandTemporalLayer` 當
  `n_delay`（見下方 causality 說明）
- `kernel_f_size` / `kernel_t_size` – `GridBlock` 的 `kernel_f` / `kernel_t`
- `channel_qk` – `GridBlock` 的 `approx_qk_dim`

**Causality：** 輸入端的 `Conv2d` 跟輸出端的 `ConvTranspose2d` 在時間軸上都是
用 causal（只看過去）的 padding/裁切建的（進去的時候是
`ZeroPad2d((2, 0, 1, 1))`，出來的時候是 `x[..., :-2]` —— 兩邊都不會增加
latency），而且 `FullbandSelfAttention` 的 mask 永遠不允許看到未來的
frame。**唯一**的 look-ahead 來源是每個 `GridBlock` 裡的
`SubbandTemporalLayer`：在 library 預設的 `block_delay_frames=0` 下，
每個 block 都會往未來看 `kernel_t_size - 1` 個 frame（已驗證 —— 見上面
`SubbandTemporalLayer` 的說明），所以 `TFGridNet` 開箱**並不是 causal
的**；沒有單一個開關可以讓整個 stack 變成 causal（`n_delay` 只會移動
每個 block 的 look-ahead要在哪裡實現，不會決定它存不存在）。

### `forward(x) -> Tensor`

```python
forward(x: Tensor) -> Tensor
# x: [N, input_dim, T]（RI-concat，會 unsqueeze 成單一 channel 的 map ——
#    只有在 inp_channel_dim 也對應設成 1 時才合理）或
#    [N, inp_channel_dim, input_dim, T]（RI-stack，一般情況）
# returns: 跟 x 一樣的 shape -- 是個 mask/mapping 輸出，不是固定長度的 embedding
```

`in_conv`（把頻率用 `f_stride` 做 downsample）→ `n_block` 個 `GridBlock` →
`out_conv`（把頻率 upsample 回 `input_dim`，再丟掉 transpose-conv 多產生出來
的 2 個尾端時間 frame）。

### Example（對照 `test/test_backbone.py::test_tfgrid_backbone`）

```python
from puresound.nnet import TFGridNet

model = TFGridNet(
    inp_channel_dim=2, input_dim=256, channel_dim=32, lstm_dim=128,
    n_block=6, block_delay_frames=0, kernel_f_size=5, kernel_t_size=5,
    f_stride=4, n_head=4, channel_qk=4, attent_range=100,
)
y = model(torch.rand(1, 2, 256, 1000))  # [1, 2, 256, 1000]
```
