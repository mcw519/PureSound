# puresound.nnet.dpcrn

English version: [dpcrn.md](dpcrn.md)

DPCRN 是 dual-path convolutional recurrent network（Le et al., "DPCRN:
Dual-Path Convolution Recurrent Network for Single Channel Speech
Enhancement", Interspeech 2021），建在 [`Unet`](unet.zh-TW.md) chassis 上：沿頻率軸
做 strided CNN encoder，bottleneck 是兩個 `DPRNNblock2D` block，再接帶 skip
connection 的 transposed-CNN decoder。它在 feature domain 預測 mask；
voice-isolation 與 noise-suppression recipe 都把它當 complex mask 使用。

## 結構

```
x [N, CH, C, T]
  -> spectral_compression（可選）-> input_norm (iLN)
  -> cnn_down x (len(channels) - 1)          # 每層輸出留作 skip
  -> band_bottleneck.to_bands（可選）
  -> DPRNNblock2D -> DPRNNblock2D            # 設了 dvec_dim 時以 FiLM 接 dvec
  -> band_bottleneck.to_units（可選）
  -> auxiliary heads 讀 bottleneck           # 側向輸出，見下文
  -> cnn_up x (len(channels) - 1)            # skip concat，或 skip_conv 相加
       （df_head 接在倒數第二層 up layer）
  -> mask [N, CH, C, T]
```

套用 mask 的是 `EncDecMaskBase`（`puresound/system/siso.py`）；`mask_type: complex`
時它呼叫 `Masker.apply_complex_mask_with_df`，backbone 有 `df_head` 時會加上其
residual（見 [masker](masker.zh-TW.md)）。

## Class: `DPCRN`

### Constructor

```python
DPCRN(
    input_dim: int = 512,                  # 輸入的頻率 bin 數 C
    dvec_dim: Optional[int] = None,        # speaker-embedding 寬度；設了才啟用 FiLM
    activation_type: str = "PReLU",
    norm_type: str = "bN2d",
    dropout: float = 0.05,
    channels: Tuple = (1, 32, 32, 32, 64, 128),  # channels[0] = CH；channels[-1] = bottleneck 寬度
    transpose_t_size: int = 2,             # decoder ConvTranspose2d 的時間 kernel
    transpose_delay: bool = False,         # 從前端裁掉 transpose conv 多出的 frame
    skip_conv: bool = False,               # skip 經 conv 相加，而非 concat
    kernel_t: Tuple = (2, 2, 2, 2, 2),     # 每個 down layer，時間軸
    stride_t: Tuple = (1, 1, 1, 1, 1),
    dilation_t: Tuple = (1, 1, 1, 1, 1),
    kernel_f: Tuple = (5, 3, 3, 3, 3),     # 每個 down layer，頻率軸
    stride_f: Tuple = (2, 2, 1, 1, 1),
    dilation_f: Tuple = (1, 1, 1, 1, 1),
    delay: Tuple = (0, 0, 0, 0, 0),        # 每個 down layer 的 look-ahead frame 數
    rnn_hidden: int = 128,                 # 兩條 DPRNN 路徑的 hidden 寬度
    inter_type: str = "lstm",              # lstm | mamba | mamba_context | lstm+mamba
    mamba_args: Optional[dict] = None,     # 傳給 lobe.ssm.MambaInter 的參數
    band_bottleneck: Optional[Dict] = None,  # 傳給 lobe.banding.BandBottleneck 的參數
    intra_type: str = "lstm",              # lstm | attention
    intra_nhead: int = 4,                  # intra_type="attention" 時的 attention head 數
    mamba_context: Optional[Dict] = None,  # 只給時間路徑用的 BandBottleneck 參數
    spectral_compress: bool = False,       # 在 input_norm 前做 |X|^0.3、保留相位
    vad_head: Optional[Dict] = None,       # lobe.heads.VADHead
    background_vad_head: Optional[Dict] = None,  # 第二個 VADHead，對非目標語音
    dist_head: Optional[Dict] = None,      # lobe.heads.DistHead
    identity_head: Optional[Dict] = None,  # lobe.heads.IdentityHead
    proximity_head: Optional[Dict] = None, # lobe.heads.ProximityHead
    expose_bottleneck: bool = False,       # 保留帶 graph 的 pooled bottleneck
    df_head: Optional[Dict] = None,        # lobe.multiframe.DeepFilterResidualHead
)
```

`input_dim` 到 `delay` 之間的參數（`dvec_dim` 與 `transpose_delay` 除外）會傳給
`Unet.__init__`；它們如何決定 encoder 與 decoder 見 [unet](unet.zh-TW.md)。DPRNN
block 與各 head 處理的是 stride 之後剩下的 `shape_info()[0][-1]` 個頻率位置；
`input_dim: 256` 配 `stride_f: [2, 2, 1]` 時是 64。

`delay` 是每個 down layer 的 look-ahead。`stride_t` 全為 1 時，模型的
algorithmic look-ahead 是 `sum(delay)` 個 frame：`[1, 1, 1]` 在 hop 160、16 kHz
下是 30 ms。

### `forward(x, dvec=None) -> Tensor`

```python
forward(x: Tensor, dvec: Optional[Tensor] = None) -> Tensor
# x:    [N, CH, C, T]，或 [N, C, T]（會 unsqueeze 成 [N, 1, C, T]）
# dvec: [N, dvec_dim]；進 FiLM 前先做 L2 正規化
# 回傳 mask，[N, CH, C, T]
```

### 側向輸出

每次 forward 都會覆寫這些屬性。關閉的 head 其屬性為 `None`、也不建立任何參數，
所以沒有該 key 的 config 建出同一個模型、載入同一份 checkpoint。

| 屬性 | 啟用方式 | shape | 內容 |
| --- | --- | --- | --- |
| `last_vad_logits` | `vad_head` | `[N, T]` | 目標語音活動 logits |
| `last_background_vad_logits` | `background_vad_head` | `[N, T]` | 非目標語音活動 logits |
| `last_dist_preds` | `dist_head` | `[N, n_out]` | `[fg_drr_db / drr_scale, log10(fg_dist_m), log10(nearest_itf_dist_m)]` |
| `last_identity_emb` | `identity_head` | `[N, T, dim]` | L2 正規化的逐 frame embedding |
| `last_proximity` | `proximity_head` | `[N, T]` | 相對距離分數，單位任意 |
| `last_bottleneck_graph` | `expose_bottleneck: true` | `[N, channels[-1], T]` | 沿頻率平均的 bottleneck，帶 graph |
| `last_bottleneck` | `stash_bottleneck = True`（執行期屬性） | `[N, channels[-1], F, T]` | bottleneck，已 detach |
| `last_df_coefs` | `df_head` | `[N, order, 2, bins, T]` | 低頻段的 deep-filter taps |

各 head block 由該 head 自己的 config model 解析，未知 key 會被拒絕。key 與預設值：

| block | key（預設） |
| --- | --- |
| `vad_head`、`background_vad_head` | `enabled`（false）、`hidden`（`channels[-1]`）、`kernel_t`（5）、`ema_taus_s`（null）、`frame_rate`（100.0） |
| `dist_head` | `enabled`（false）、`hidden`（128）、`n_out`（3） |
| `identity_head` | `enabled`（false）、`dim`（64）、`kernel_t`（5） |
| `proximity_head` | `enabled`（false）、`hidden`（64） |

各 head 計算什麼見 [lobe/heads](lobe/heads.zh-TW.md)。除了 `df_head`，沒有 head 會
改變音訊輸出；讀它們的是 loss 與 evaluation，streaming export 可以把兩個 VAD head
的 logits 當側向輸出匯出。`stash_bottleneck` 由 `EncDecMaskBase.forward` 在每次
呼叫時為推論的 presence gate 設定；`expose_bottleneck` 是建構期旗標，給需要在
bottleneck 上跑自己模組的 loss 用。兩者分開，才不會把 gradient 路徑交給預期拿到
detached tensor 的使用端。

### Bottleneck 選項

**`intra_type`** 決定每個 `DPRNNblock2D` 的頻率路徑。`"lstm"` 是沿每個 frame 的
頻率位置跑的雙向 LSTM。`"attention"` 是 DPARN 的 intra 路徑：兩層
`MhaSelfAttenLayer`（[lobe/attention](lobe/attention.zh-TW.md)），各有 `intra_nhead`
個 head，第一層帶 sinusoidal position encoding，後接一層 linear。頻率軸沒有因果
限制，所以 attention 看到的上下文與雙向走訪相同，但把逐步的序列運算換成每層一次
矩陣乘法。

**`inter_type`** 決定時間路徑，這是模型跨 frame 唯一攜帶的狀態：

| `inter_type` | 時間路徑 |
| --- | --- |
| `lstm` | 單向 LSTM |
| `mamba` | 以 `MambaInter` selective state-space block 取代 LSTM |
| `lstm+mamba` | LSTM 加上並聯的 `MambaInter`，其輸出投影初始化為零 |
| `mamba_context` | 在 perceptual band 上跑 `MambaInter`；需搭配 `mamba_context` |

`mamba_args` 傳給 `MambaInter`：`d_state`（16）、`d_conv`（4）、`expand`（2）、
`dt_rank`、`dt_min`、`dt_max`、`dt_init_floor`；見 [lobe/ssm](lobe/ssm.zh-TW.md)。
`lstm+mamba` 在初始化時等於單獨的 LSTM，因此能從 LSTM checkpoint warm-start，
不必重新初始化已訓練好的 context carrier。

**`band_bottleneck`** 在兩個 DPRNN block 之前把 bottleneck pool 到 `n_bands` 個
perceptual band，之後再展開回來（[lobe/banding](lobe/banding.zh-TW.md)）。key 即
`BandBottleneck` 的參數：`n_bands`（必填）、`sample_rate`（16000）、`f_min`
（50.0）、`f_max`（`sample_rate / 2`）、`scale`（`erb` 或 `mel`）、`learnable`
（false）。encoder、decoder 與各 head 仍在 strided grid 上。均勻 stride 丟掉的
資訊 banding 補不回來，所以要搭配 `stride_f: [1, 1, 1]`，直接從完整解析度分帶。

**`mamba_context`** 接受同樣的 key，但只對時間路徑分帶：intra 路徑留在完整 grid，
分帶後的時間路徑輸出展開回來、經 block 的 residual 加回。如此逐 bin 的細節不必
經過有損的 band 來回，而獨立 Mamba stream 的數量降為 `n_bands`。它需要
`inter_type: mamba_context`，且不能與 `band_bottleneck` 併用。

```yaml
backbone_args:
  inter_type: mamba_context
  mamba_args: {d_state: 16, d_conv: 4, expand: 1}
  mamba_context: {n_bands: 24, scale: erb, sample_rate: 16000}
```

**`df_head`** 在最低的 `bins` 個 bin 上加 deep-filtering residual（DeepFilterNet，
Schröter et al., ICASSP 2022）。`DeepFilterResidualHead`
（[lobe/multiframe](lobe/multiframe.zh-TW.md)）讀 decoder 倒數第二層的 feature map
（寬 `channels[1]`，比 mask 粗一個 `stride_f[0]`），為每個 bin 預測 `order` 個
因果 complex taps；增強後的低頻段變成 `M * X[t] + sum_k w_k X[t - k]`。key：
`bins`（128；須是 `stride_f[0]` 的倍數且不超過 `input_dim`）、`order`（5）、
`hidden`（32）。這個 block 以一般 `dict.get` 讀取，所以任何非空 dict 都會啟用它。
head 的最後一層初始化為零，因此沒有它的 checkpoint 載入後算出相同輸出。它至少
需要兩層 decoder。

### Streaming

逐 frame export（[streaming/dpcrn_onnx](../../usage/streaming/dpcrn_onnx.zh-TW.md)）
重現 offline forward，延遲 `sum(delay)` 個 frame，涵蓋 banding、`mamba_context`、
attention intra、`mamba` inter 與 `df_head`。它拒絕 `inter_type: lstm+mamba` 與
`spectral_compress: true`，也不餵 `dvec`。`transpose_delay` 保持 `false`；
`true` 會讓 decoder 讀到未來的 frame。

### `get_args`

property，內容是 constructor 收到的全部參數，所以 `DPCRN(**model.get_args)`
會建出同一個網路，原模型的 `state_dict()` 能 strict 載入。

### Config 用法

```yaml
model:
  backbone:
    type: DPCRN
    backbone_args:
      input_dim: 256
      norm_type: bN2d
      channels: [2, 32, 64, 128]
      kernel_t: [2, 2, 2]
      stride_t: [1, 1, 1]
      dilation_t: [1, 1, 1]
      kernel_f: [5, 3, 3]
      stride_f: [2, 2, 1]
      dilation_f: [1, 1, 1]
      delay: [1, 1, 1]
      rnn_hidden: 96
      vad_head: {enabled: true, hidden: 64, kernel_t: 5}
      dist_head: {enabled: true, hidden: 128}
```

## Class: `DPRNNblock2D`

一個 bottleneck block：intra-frequency 路徑與 inter-time 路徑，各自接 LayerNorm
與 residual 相加，兩者之間可選 FiLM conditioning。

```python
DPRNNblock2D(
    input_size: int,                       # bottleneck channel 數 CH
    hidden_size: int,                      # 兩條路徑的 hidden 寬度
    dropout: float = 0.0,
    inter_type: str = "lstm",              # lstm | mamba | mamba_context | lstm+mamba
    mamba_args: Optional[dict] = None,
    context_bottleneck: Optional[Dict] = None,  # BandBottleneck 參數；僅 mamba_context
    context_freqs: Optional[int] = None,        # 分帶前的頻率位置數
    embedding_size: Optional[int] = None,  # FiLM condition 寬度；None 則不做 FiLM
    fused_type: Optional[str] = None,      # "film"（不分大小寫），唯一的 fusion
    intra_type: str = "lstm",              # lstm | attention
    intra_nhead: int = 4,
)
```

`fused_type` 不分大小寫比對。`embedding_size` 是 `None` 時可以是 `None`；有
`embedding_size` 時，`"film"` 以外的值都會被拒絕。`DPCRN` 固定傳 `"FiLM"`。

```python
forward(x: Tensor, intra_skip: bool = True, inter_skip: bool = True,
        embed: Optional[Tensor] = None) -> Tensor
# x: [N, CH, C, T] -> [N, CH, C, T]
```

1. Intra：reshape 成 `[N*T, C, CH]`，跑頻率路徑、LayerNorm，`intra_skip` 時加回
   block 輸入。
2. FiLM：設了 `embedding_size` 且有給 `embed`（`[N, embedding_size]`）時，對每個
   頻率位置的時間序列做調變（[lobe/trivial](lobe/trivial.zh-TW.md)）。
3. Inter：有 `mamba_context` 時先分帶到 `n_bands` 個位置；reshape 成
   `[N*C, T, CH]`，跑時間路徑（`lstm+mamba` 再加上並聯的 SSM）、LayerNorm、
   展開回來，`inter_skip` 時加回第 1 步的輸出。

## 設計說明

- 每種 `inter_type` 的時間路徑都是單向的，所以除了 `delay` 的 look-ahead 外模型
  是因果的。頻率路徑可以雙向，因為一個 frame 的所有 bin 同時可得。
- 遞迴 block 跑在最小的 grid 上，成本最低；局部頻譜結構交給完整解析度的
  convolution stack。
- 各 head 讀的是 band 展開回來之後的 bottleneck，所以 banding 不會改變它們的
  shape 或 checkpoint key。head 的屬性名稱就是 checkpoint key
  （`backbone.vad_head.*`）。
