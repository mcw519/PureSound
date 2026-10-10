# puresound.nnet.dprnn

English version: [dprnn.md](dprnn.md)

作用在 1-D feature 序列上的 dual-path RNN（Luo et al., "Dual-Path RNN:
Efficient Long Sequence Modeling for Time-Domain Single-Channel Speech
Separation", ICASSP 2020）。序列先切成 chunk；每個 block 在每個 chunk 內跑一次
LSTM（intra），再在每個 chunk 內位置上跨 chunk 跑一次 LSTM（inter）。它是獨立的
序列模型，與 DPCRN 的 [`DPRNNblock2D`](dpcrn.zh-TW.md) 不同；後者作用在
`[N, CH, C, T]` 的頻率 grid 上。

## Class: `DPRNN`

### Constructor

```python
DPRNN(
    input_size: int,                   # 輸入 channel 數 C
    hidden_size: int,                  # LSTM hidden 寬度
    output_size: int,                  # 輸出 channel 數
    n_blocks: int = 2,                 # 疊幾組 intra + inter
    seg_size: int = 20,                # chunk 長度 K（frame）
    seg_overlap: bool = False,         # True：50% 重疊；False：連續切塊
    causal: bool = True,               # intra 與 inter LSTM 都單向
    embed_dim: int = 0,                # FiLM condition 寬度；0 表示不做 FiLM
    embed_norm: bool = False,          # FiLM 前先對 embed 做 L2 正規化
    block_with_embed: Optional[List] = None,  # 每個 block 是否套 FiLM
    embedding_free_tse: bool = False,  # embed 是 enrollment feature 序列
)
```

- `seg_overlap=True` 用 [`SplitMerge`](lobe/trivial.zh-TW.md) 以 stride
  `seg_size // 2` 切塊，merge 時把重疊部分平均。`False` 則把序列補齊到整數個
  chunk、reshape，最後裁回 `T`。
- `causal` 一次決定所有 block 所有 LSTM 的方向：`True` 時 intra 與 inter 都單向，
  `False` 時都雙向。沒有分開控制 intra/inter 的選項。
- `block_with_embed` 是長度 `n_blocks` 的 list，`embed_dim != 0` 時必填；只有被
  標記的 block 會建立 [`FiLM`](lobe/trivial.zh-TW.md) 層。

### 以 `embed` 做 conditioning

**FiLM（`embed_dim != 0`、`embedding_free_tse=False`）。** `embed` 是
`[N, embed_dim]` 向量，例如 speaker embedding。`embed_norm` 時先做 L2 正規化，
再複製到每個 chunk，在每個被標記 block 的 intra LSTM 之前以 FiLM 作用在 chunk
tensor 上。

**Embedding-free target-speaker extraction（`embedding_free_tse=True`）。**
`embed` 是 enrollment feature 序列 `[N, input_size, T']`。它不經 FiLM，直接通過
同一組 intra/inter LSTM、projection 與 norm（`_get_hidden_states`），回傳每個
block 的 inter LSTM 一組 `(h, c)`。這些 state 在主要 forward 中作為各 block inter
LSTM 的初始 state，讓 enrollment 的遞迴 state 取代 embedding 向量。此模式下不套
FiLM。

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x:     [N, input_size, T]
# embed: [N, embed_dim]（FiLM）或 [N, input_size, T']（embedding_free_tse）
# 回傳 [N, output_size, T]
```

先切 chunk，然後每個 block 依序：可選的 FiLM、intra LSTM 接 linear projection、
LayerNorm 與 residual，inter LSTM（embedding-free 模式下以 enrollment state
初始化）接 linear projection、LayerNorm 與 residual。最後 merge 回 `T` 個 frame，
套 `output_fc`（`PReLU` 再接 1x1 `Conv1d` 到 `output_size`）。

### 範例

```python
from puresound.nnet import DPRNN

# 因果、半重疊 chunk、無 conditioning
model = DPRNN(512, 256, 512, 4, 32, causal=True, seg_overlap=True)
y = model(torch.rand(1, 512, 1000))  # [1, 512, 1000]

# 在 block 1 與 2（從 0 起算）套 FiLM
model = DPRNN(
    512, 256, 512, 4, 32,
    causal=True, seg_overlap=True,
    embed_dim=192, embed_norm=True, block_with_embed=[0, 1, 1, 0],
)
y = model(torch.rand(1, 512, 1000), torch.rand(1, 192))
```

## 設計說明

- 切 chunk 把一段長遞迴變成兩段短遞迴：intra LSTM 看 `K` 個 frame，inter LSTM
  看 `T / K` 個 chunk，輸入很長時兩者都仍然短。
- `causal=True` 時 intra LSTM 只讀自己 chunk 內較早的位置，inter LSTM 只讀較早的
  chunk，所以每個輸出 frame 只依賴當前與更早的輸入 frame。
- `seg_overlap=True` 時每個 frame 落在兩個 chunk 中，merge 時平均兩者的輸出。
