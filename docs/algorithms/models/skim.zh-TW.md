# puresound.nnet.skim

English version: [skim.md](skim.md)

SkiM，skipping-memory LSTM（Li et al., "SkiM: Skipping Memory LSTM for
Low-Latency Real-Time Continuous Speech Separation", ICASSP 2022；
[ESPnet 參考實作](https://github.com/espnet/espnet/blob/master/espnet2/enh/layers/skim.py)）。
它和 [DPRNN](dprnn.zh-TW.md) 一樣把 1-D 序列切成 chunk，但沒有跨所有 chunk 的
inter-chunk LSTM。每個 block 是逐 chunk 的 `SegLSTM`；block 之間由 `MemLSTM`
在各 chunk 最後的 `(h, c)` state 上運算，結果交給下一個 block 當初始 state。

## Class: `SegLSTM`

在每個 chunk 內跑的 LSTM，chunk 當作 batch。

```python
SegLSTM(input_size: int, hidden_size: int, causal: bool = True, dropout: float = 0.0)
# forward(x [NS, K, C], h [D, NS, H] | None, c [D, NS, H] | None)
#   -> (x_out [NS, K, C], h [D, NS, H], c [D, NS, H])
```

`causal=True` 時 LSTM 單向（`D = 1`），`False` 時雙向（`D = 2`）。`h` 與 `c`
預設為零。輸出是 `x + LayerNorm(Linear(Dropout(LSTM(x))))`。`streaming_forward`
接受相同參數，沿 `K` 一次走一個 frame。

## Class: `MemLSTM`

在兩個 block 之間，跨 chunk 修正 `SegLSTM` 的 state。

```python
MemLSTM(hidden_size: int, causal: bool = True, dropout: float = 0.0)
```

輸入寬度在 `causal` 時為 `hidden_size`，否則為 `2 * hidden_size`，對應單向或
雙向 `SegLSTM` state 的 `D * hidden_size`。

```python
forward(
    h: Tensor,                      # [N, S, D, H] SegLSTM hidden state，每個 chunk 一組
    c: Tensor,                      # [N, S, D, H] SegLSTM cell state
    h_states: Optional[Tuple[Tensor, Tensor]] = None,  # h_net 的初始 state
    c_states: Optional[Tuple[Tensor, Tensor]] = None,  # c_net 的初始 state
    return_all: bool = False,       # 一併回傳 h_net / c_net 的最終 state
    streaming: bool = False,        # 不做因果位移
)
# -> (h [D, N*S, H], c [D, N*S, H])，return_all 時再加 (h_net state, c_net state)
```

兩個 LSTM（`h_net`、`c_net`）沿 chunk 軸 `S` 運算，各自接 projection、LayerNorm
與 residual 相加，結果再摺回 `nn.LSTM` 作為 state 所需的 `[D, N*S, H]`。`causal`
且非 `streaming` 時，state 會在每個 row 內位移一個 chunk（每個 row 的 chunk 0 補零），
使下一個 block 中 chunk `i` 的初始 state 只總結 `i` 之前的 chunk，而不含 chunk `i`
本身。

`streaming_forward(h, c, h_states=None, c_states=None, return_all=False)` 以
batch size 1 一次處理一個 chunk、攜帶 `h_net`/`c_net` 的 state，不做 dropout
也不做位移。

## Class: `SkiM`

### Constructor

```python
SkiM(
    input_size: int,                   # 輸入 channel 數 C
    hidden_size: int,                  # LSTM hidden 寬度 H
    output_size: int,                  # 輸出 channel 數
    n_blocks: int = 2,                 # SegLSTM 個數；之間有 n_blocks - 1 個 MemLSTM
    seg_size: int = 20,                # chunk 長度 K
    seg_overlap: bool = False,         # True：50% 重疊；False：連續切塊
    causal: bool = True,               # 所有 SegLSTM 與 MemLSTM 都單向
    embed_dim: int = 0,                # condition 寬度；0 表示不做 conditioning
    embed_norm: bool = False,          # 對 embed 做 L2 正規化
    embed_fusion: Optional[str] = None,  # "film" | "gate"（不分大小寫）
    block_with_embed: Optional[List] = None,  # 每個 block 是否做 conditioning
    dropout: float = 0.0,              # 每個 SegLSTM / MemLSTM 內部
)
```

- `seg_size` / `seg_overlap` 的行為同 DPRNN；SkiM 有自己的 `split`/`merge`
  方法（半 stride 切塊、merge 時平均）。
- `embed_dim != 0` 時 `embed_fusion` 與 `block_with_embed` 必填。`"film"` 建立
  [`FiLM`](lobe/trivial.zh-TW.md)；`"gate"` 建立內部寬度固定為 128 的
  [`Gate`](lobe/trivial.zh-TW.md)，與 `hidden_size` 無關。conditioning 在每個被
  標記 block 的 `SegLSTM` 之前作用在 chunk tensor 上。

### `forward(x, embed=None) -> Tensor`

```python
forward(x: Tensor, embed: Optional[Tensor] = None) -> Tensor
# x:     [N, input_size, T]，或 [N, 2, F, T]（見下）
# embed: [N, embed_dim]
# 回傳 [N, output_size, T]，或 [N, 2, F, T]
```

4-D 的 `[N, 2, F, T]` 輸入（complex-mask front end 的實部/虛部 tensor，
`feats_type: complex`，見 [features](features.zh-TW.md)）會先摺成 `[N, 2F, T]`
跑完整個 stack，再展開回 `[N, 2, F, T]`，所以 `input_size` 與 `output_size` 都
必須等於 `2 * F`。

每個 block：可選的 conditioning，接著以前一個 `MemLSTM` 的輸出（block 0 為零）
初始化 `seg_lstm[i]`；除最後一個 block 外，回傳的 `(h, c)` reshape 成
`[N, S, D, H]` 後送進 `mem_lstm[i]`。最後一個 block 之後把 chunk merge 回 `T` 個
frame，由 `output_fc`（`PReLU` 再接 1x1 `Conv1d`）映射到 `output_size`。

### 範例

```python
from puresound.nnet import SkiM

# 因果、半重疊 chunk、無 conditioning
model = SkiM(512, 256, 512, 4, 32, causal=True, seg_overlap=True)
y = model(torch.rand(1, 512, 1000))  # [1, 512, 1000]

# 每個 block 都做 Gate conditioning
model = SkiM(
    512, 256, 512, 4, 32,
    causal=True, seg_overlap=False,
    embed_dim=192, embed_norm=True,
    block_with_embed=[1, 1, 1, 1], embed_fusion="Gate",
)
y = model(torch.rand(1, 512, 1000), torch.rand(1, 192))
```

## 設計說明

- 長距離上下文只透過 `MemLSTM` 修正過的逐 chunk state 傳遞，所以跨 chunk 的成本
  是一個在 `S` 個 state 上跑的小 LSTM，而不是在每個 chunk 內位置上都跑一次完整
  LSTM。
- `MemLSTM` 的因果位移讓模型跨 chunk 仍是因果的：chunk 起始的 state 永遠不含該
  chunk 自己的內容。
