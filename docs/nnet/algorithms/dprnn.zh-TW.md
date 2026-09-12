# puresound.nnet.dprnn

English version: [dprnn.md](dprnn.md)

Status: *library*（見 [nnet index](../index.zh-TW.md)）—— 可透過
`getattr(nnet, "DPRNN")` 以 config 方式取用，有 forward smoke test 覆蓋
（`test/test_backbone.py::test_dprnn_backbone`），但目前沒有任何維護中的
recipe 在用。

**Reference:** Luo et al., "Dual-Path RNN: Efficient Long Sequence Modeling
for Time-Domain Single-Channel Speech Separation," ICASSP 2020.

這是一個獨立、causal 可完全自訂的 dual-path RNN —— 跟
[`DPRNNblock2D`](dpcrn.zh-TW.md)（DPCRN bottleneck 寫死的那個 block：intra
永遠雙向、inter 永遠單向，完全沒有 `causal` 開關）是不同的實作。這裡是用單一個
`causal` flag 同時控制 intra-chunk 跟 inter-chunk LSTM 的方向，且每個 block
都一致套用同一個設定。

## Class: `DPRNN`

### Constructor

```python
DPRNN(
    input_size: int,
    hidden_size: int,
    output_size: int,
    n_blocks: int = 2,
    seg_size: int = 20,
    seg_overlap: bool = False,
    causal: bool = True,
    embed_dim: int = 0,
    embed_norm: bool = False,
    block_with_embed: Optional[List] = None,
    embedding_free_tse: bool = False,
)
```

**Parameters:**
- `input_size` / `hidden_size` / `output_size` – 輸入 feature 維度、LSTM
  hidden 寬度、輸出 feature 維度（`output_fc` 是唯一真正改變 channel 數的地方）
- `n_blocks` – 疊幾組 intra+inter LSTM pair
- `seg_size` – chunk 在時間軸上的長度
- `seg_overlap` – **是個 bool，不是 stride 長度**：若為 `True`，會用
  [`SplitMerge`](../lobe/trivial.md) 的 50%-overlap split/merge 來切 chunk
  （`seg_stride = seg_size // 2`）；若為 `False`（預設），就是單純的連續
  reshape（`x.reshape(N, -1, seg_size, C)`，補零到 `seg_size` 的倍數，結尾再
  裁回 `T`）——沒辦法指定任意大小的 overlap，只有「一半」或「完全不重疊」兩種選擇
- `causal` – **預設是 `True`**。只算一次 `bi_direct = not causal`，然後
  每個 block 的兩個 LSTM 都共用這個值（`intra_rnn[i]` 跟 `inter_rnn[i]` 要嘛
  一起雙向、要嘛一起單向）—— 沒有辦法讓 intra/inter 各自獨立設定 causal
- `embed_dim` – 若不為 `0`，`block_with_embed` 裡標記的 block 就會用
  FiLM 把 `embed` 條件化進自己的 chunk 輸入（見下方）；`embedding_free_tse=True`
  時這個參數無關緊要
- `embed_norm` – 條件化之前先對 `embed` 做 L2-normalize（`embedding_free_tse=True`
  時會被跳過）
- `block_with_embed` – 長度為 `n_blocks` 的 `List[int]`；只要 `embed_dim != 0`
  就一定會被索引到（沒設的話會出錯）
- `embedding_free_tse` – 決定 `embed` 這個參數*代表什麼意思*（見下一節）

### 兩種對 `embed` 做條件化的方式

**1. 固定 embedding 的 FiLM（`embed_dim != 0`、`embedding_free_tse=False`，
預設）：** `embed` 是一個 `[N, embed_dim]` 的向量（例如 speaker embedding）。
若 `embed_norm` 為真會先做 L2-normalize，然後廣播到每個 chunk
（`embed.repeat(1, total_seg, 1)`）。在逐個 block 的迴圈裡，只要
`block_with_embed[i]` 為真，就會在該 block 的 intra-chunk LSTM 之前，
用 FiLM 對這個 chunk tensor 做條件化（`self.input_film[i]`，只有這些 block
才會建的一個 [`FiLM`](../lobe/trivial.md) layer）。

**2. Embedding-free 的 target-speaker extraction（`embedding_free_tse=True`）：**
這時 `embed` 必須改成一個 3-D 的 enrollment **feature 序列**
`[N, input_size, T']`（在 `forward` 一開始就會 assert）——而不是固定長度的
向量。它會被丟進 `_get_hidden_states`，用完全同一組
`intra_rnn`/`inter_rnn`/`proj`/`norm` 權重再跑一次（裡面沒有 FiLM、也沒有
embed 條件化），回傳每個 block 的 `inter_rnn` 最終 `(h, c)` state。這些
逐 block 的 state 接著會餵進主要 forward pass 裡的
`self.inter_rnn[i](inter_input, inter_hidd_init_states[i])` 當成初始狀態 ——
也就是說，enrollment 語句自身的 LSTM 動態，直接取代了原本該學一個 embedding
向量的角色。這跟第 1 種機制在實務上是互斥的：`embedding_free_tse=True`
會跳過 FiLM conditioning 依賴的那段 embed 廣播/reshape 步驟，所以同一次跑
兩種都用並不是一個支援的組合。

`_get_hidden_states` 自己的 docstring 宣稱它回傳一個 `[N, C, T]` 的 tensor；
但實際上它回傳的是 `inter_hidd`，一個 `List[Tuple[Tensor, Tensor]]`
（每個 block 一組 `(h, c)`）—— `forward` 就是把這個 list 當成
`inter_hidd_init_states` 來用的。

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x:     [N, input_size, T]
# embed: [N, embed_dim]（FiLM 路徑）或 [N, input_size, T']（embedding_free_tse 路徑）
# returns: [N, output_size, T]
```

把 `x` 切成 chunk → 對 `n_blocks` 裡的每一個：先做（可選的）FiLM →
intra-chunk LSTM（residual + `LayerNorm`）→ inter-chunk LSTM，用
`inter_hidd_init_states[i]` 當初始狀態（residual + `LayerNorm`）→ 把 chunk
合併回 `T` 個 frame → `output_fc`（`PReLU` 接 `Conv1d` 投影到 `output_size`）。

### Example（對照 `test/test_backbone.py::test_dprnn_backbone`）

```python
from puresound.nnet import DPRNN

# Causal、一半 overlap 的 chunking，不做條件化
model = DPRNN(512, 256, 512, 4, 32, causal=True, seg_overlap=True)
y = model(torch.rand(1, 512, 1000))  # [1, 512, 1000]

# 對第 1、2 個 block（0-indexed）做固定 embedding 的 FiLM 條件化
model = DPRNN(
    512, 256, 512, 4, 32,
    causal=True, seg_overlap=True,
    embed_dim=192, embed_norm=True, block_with_embed=[0, 1, 1, 0],
)
y = model(torch.rand(1, 512, 1000), torch.rand(1, 192))
```
