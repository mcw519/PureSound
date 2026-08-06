# puresound.nnet.skim

English version: [skim.md](skim.md)

Status: *library*（見 [nnet index](../index.zh-TW.md)）—— 可透過
`getattr(nnet, "SkiM")` 以 config 方式取用，有 forward smoke test 覆蓋
（`test/test_backbone.py::test_skim_backbone`），但目前沒有任何維護中的
recipe 在用。

**Reference:** Li et al., "SkiM: Skipping Memory LSTM for Low-Latency
Real-Time Continuous Speech Separation," ICASSP 2022.
（[espnet 參考實作](https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/skim.py)）

SkiM 把 [`DPRNN`](dprnn.zh-TW.md) 那種「每個 block 都對*所有* segment 跑一次
inter-chunk LSTM」的做法，換成便宜很多的設計：每個 block 都是獨立的
per-segment `SegLSTM`，再用一個小小的 `MemLSTM` 把這個 block 的 hidden/cell
state 橋接到*下一個* block —— 直接「跳過」整段 inter-chunk 的全序列運算。
這正是論文標題裡「低延遲」這個賣點的核心技巧。

## Class: `SegLSTM`

Per-segment LSTM。batch 裡的每個 segment 都是獨立處理的（segment 會變成
batch 維度），會吃進、也會回傳明確的 hidden/cell state，讓呼叫端可以選擇要不要
把 segment 串起來。

```python
SegLSTM(input_size: int, hidden_size: int, causal: bool = True, dropout: float = 0.0)
# forward(x [NS, K, C], h [D, NS, C] 或 None, c [D, NS, C] 或 None)
#   -> (x_out [NS, K, C], h [D, NS, C], c [D, NS, C])
```

`causal=True` 會讓內部的 `nn.LSTM` 是單向的（`D=1`）；`False` 則是雙向
（`D=2`）。`h`/`c` 若為 `None` 就預設補零。`streaming_forward` 是同一套遞迴
拆成逐 frame（沿 `K` 迴圈）跑，用來確保跟 online 推論的結果完全一致。

## Class: `MemLSTM`

在兩個 `SkiM` block *之間*，把 `SegLSTM` 的 state 跨 segment 橋接起來。

```python
MemLSTM(hidden_size: int, causal: bool = True, dropout: float = 0.0)
```

**Parameters:**
- `hidden_size` – 必須跟外層 `SegLSTM` 的 `hidden_size` 一致
- `causal` – 跟 `SegLSTM` 一樣;同時也決定了
  `input_size = hidden_size if causal else 2 * hidden_size`，因為傳進來的
  `(h, c)` state 是 `[N, S, D, hidden_size]`，`D = 1`（causal `SegLSTM`）或
  `D = 2`（雙向 `SegLSTM`）—— `MemLSTM` 自己的 LSTM 是*沿著 segment 軸 S*
  在跑的，所以它的輸入寬度得配合 `SegLSTM` state 疊起來後實際的
  `D * hidden_size`
- `dropout` – 內部 `h_net`/`c_net` 這兩個 LSTM 共用

### `forward(h, c, h_states=None, c_states=None, return_all=False, streaming=False)`

```python
forward(
    h: Tensor,   # [N, S, D, C] -- SegLSTM 的 hidden state，每個 segment 一組
    c: Tensor,   # [N, S, D, C] -- SegLSTM 的 cell state
    h_states: Optional[Tuple[Tensor, Tensor]] = None,  # MemLSTM 自己的 LSTM
    c_states: Optional[Tuple[Tensor, Tensor]] = None,  # 遞迴狀態（只有 streaming 時用）
    return_all: bool = False,
    streaming: bool = False,
) -> Tuple[Tensor, Tensor] | Tuple[Tensor, Tensor, tuple, tuple]
# 回傳的 h, c 都是 [D, N*S, C] -- 可以直接餵給下一個 SegLSTM
```

沿著 segment 軸跑兩個獨立的 LSTM（`h_net`、`c_net`），做 residual +
`LayerNorm`，再折回 `nn.LSTM` state tuple 要求的 `[D, N*S, C]` shape。當
`causal=True` 時，結果還會再往後位移一個 segment（`h_[:, 1:, :] =
h[:, :-1, :]`，segment 0 補零）—— 就算 `h_net`/`c_net` 本身已經是單向的，
沒有這個位移，segment `i` 精修過的 state 仍然會是 segment `i` 自己
`SegLSTM` 輸出的函數，形成循環（這個 state 應該是用來*餵給*下一個 block 裡
segment `i` 的處理，而不是拿來總結它自己）。只要 `streaming=False`，這個位移
就會套用，所以單次批次跑的 offline forward，結果會跟 `streaming_forward`
逐 segment 跑出來的結果位元級一致。Streaming 時傳 `return_all=True`，
還可以拿到 `h_net`/`c_net` 自己的最終遞迴狀態，接續到下一批 segment。

## Class: `SkiM`

被 export 出來的 backbone（`from puresound.nnet import SkiM`）。

### Constructor

```python
SkiM(
    input_size: int,
    hidden_size: int,
    output_size: int,
    n_blocks: int = 2,
    seg_size: int = 20,
    seg_overlap: bool = False,
    causal: bool = True,
    embed_dim: int = 0,
    embed_norm: bool = False,
    embed_fusion: Optional[str] = None,
    block_with_embed: Optional[List] = None,
    dropout: float = 0.0,
)
```

**Parameters:**
- `input_size` / `hidden_size` / `output_size` – 跟 `DPRNN` 一樣
- `n_blocks` – `SegLSTM` 的數量；`n_blocks - 1` 個 `MemLSTM` 負責橋接它們
  （最後一個 block 沒有下一個可以橋）
- `seg_size` / `seg_overlap` – chunking 方式，語意跟
  [`DPRNN`](dprnn.zh-TW.md) 一樣（`seg_overlap` 是個 bool：一半 overlap 的
  split/merge，還是連續 reshape）—— 但 `SkiM` **沒有**沿用
  `lobe.trivial.SplitMerge`；它自己另外寫了一組幾乎一樣的 `split`/`merge`
  instance method
- `causal` – 統一傳給每個 `SegLSTM` 跟 `MemLSTM`（跟 `DPRNN` 一樣是「一個
  flag 控制全部」的模式）
- `embed_dim` / `embed_norm` / `block_with_embed` – 跟 `DPRNN` 一樣：
  哪些（依 index）block 要用 FiLM/Gate 把 `embed` 條件化進自己的 segment 輸入
- `embed_fusion` – `"film"`（[`FiLM`](../lobe/trivial.md)）或 `"gate"`
  （[`Gate`](../lobe/trivial.md)，它的 bottleneck 寬度是寫死的
  `hidden_size=128`，**跟** `SkiM` 自己的 `hidden_size` 參數**無關**——
  只是名字剛好撞了，數值上並沒有共用）
- `dropout` – 每個 `SegLSTM`/`MemLSTM` 共用

沒有 `mem_size` 這個參數，也沒有「memory bank」的概念 —— `MemLSTM` 每個
segment 就只帶一組 `(h, c)`，大小就是 `hidden_size`。

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
```

`x` 可以吃兩種 shape：
- `[N, input_size, T]` —— 一般的 1-D 序列情境，跟 `DPRNN` 一樣。
- `[N, 2, F, T]` —— complex-mask 前端的 real/imag tensor
  （`feats_type="complex"`，見 [nnet.features](../features.zh-TW.md)）。
  SkiM 是個 1-D 序列模型，所以會先把 `(2, F)` 折進 channel 軸
  （`input_size` 必須等於 `2 * F`），跑完整個序列 stack 之後，再把輸出
  展開回 `[N, 2, F, T']`，這樣才能被當成 complex mask 使用。`DPRNN` 沒有
  對應的自動折疊機制 —— 這是 SkiM 特有的 reshape。

每個 block：先做（可選的）FiLM/Gate 條件化 → 跑 `seg_lstm[i]`（用前一個
block 的 `MemLSTM` 輸出當種子，block 0 則從零開始）→ 如果不是最後一個
block，把回傳的 `(h, c)` reshape 成 `[N, S, D, hidden_size]`，跑
`mem_lstm[i]` 產生下一個 block 要用的種子 state。跑完最後一個 block 後：
把 chunk 合併回 `T` 個 frame → `output_fc`（`PReLU` 接 `Conv1d` 投影到
`output_size`）。

### Example（對照 `test/test_backbone.py::test_skim_backbone`）

```python
from puresound.nnet import SkiM

# Causal、一半 overlap 的 chunking，不做條件化
model = SkiM(512, 256, 512, 4, 32, causal=True, seg_overlap=True)
y = model(torch.rand(1, 512, 1000))  # [1, 512, 1000]

# 每個 block 都做 Gate 條件化
model = SkiM(
    512, 256, 512, 4, 32,
    causal=True, seg_overlap=False,
    embed_dim=192, embed_norm=True,
    block_with_embed=[1, 1, 1, 1], embed_fusion="Gate",
)
y = model(torch.rand(1, 512, 1000), torch.rand(1, 192))
```
