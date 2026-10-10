# puresound.nnet.lobe.rnn

English version: [rnn.md](rnn.md)

序列積木：`SingleRNN` 是一層 RNN/LSTM/GRU 再投影回輸入寬度；`FSMN` /
`ConditionFSMN` 是 feedforward sequential memory block，用 depthwise convolution
而非遞迴來建模上下文。

## Class: `SingleRNN`

```
y = Linear(Dropout(RNN(x)))        # x: [N, C, T] -> [N, T, C] -> ... -> [N, C, T]
```

```python
SingleRNN(
    rnn_type: str,              # "RNN" | "LSTM" | "GRU"，不分大小寫
    input_size: int,            # C；也是輸出寬度
    hidden_size: int,
    bidirectional: bool = False,
    dropout: float = 0.0,       # 作用在 RNN 輸出上、projection 之前
)
```

`forward(x [N, C, T]) -> [N, C, T]`。RNN 只有一層；projection
`Linear(hidden_size * num_directions, input_size)` 一律投影回 `input_size`，所以不論
`hidden_size` 或 `bidirectional` 為何，這個 block 都保持 shape。不回傳 hidden state；
需要它的串流呼叫端（DPCRN 串流 runner）直接呼叫內部的 `nn.LSTM`。

用在 `DPCRN` 的 `DPRNNblock2D`（intra 路徑：沿頻率的雙向 LSTM；inter 路徑：沿時間的
單向 LSTM）以及 `DPARN` 的 inter 路徑。

## Class: `FSMN`

帶 Deep-FSMN memory skip 的 feedforward sequential memory block（Zhang et al.,
"Deep-FSMN for Large Vocabulary Continuous Speech Recognition", ICASSP 2018；實作
參考 [aps](https://github.com/funcwj/aps/blob/c814dc5a8b0bff5efa7e1ecc23c6180e76b8e26c/aps/asr/base/component.py#L310)）：

```
p   = Conv1d_1x1(x)                                      # input_dim -> project_dim
ctx = DepthwiseConv1d(pad(p, l_context, r_context))      # kernel l_context + r_context + 1
p   = p + ctx (+ memory，若有給)
out = Dropout(ReLU(Norm(Conv1d_1x1(p))))                 # project_dim -> output_dim
return out, p                                            # p 是下一層的 memory
```

```python
FSMN(
    input_dim: int,
    output_dim: int,
    project_dim: int,          # memory 寬度
    l_context: int,            # 過去 frame 數
    r_context: int,            # 未來 frame 數（look-ahead）
    dilation: int = 1,         # memory tap 的間隔；padding 隨之縮放
    dropout: float = 0.0,
    norm_type: str = "bN1d",   # norm.get_norm 的名稱
)
```

`forward(x [N, input_dim, T], memory=None) -> (out [N, output_dim, T], memory [N, project_dim, T])`。

`memory` 是**堆疊層之間**的 skip connection，不是跨時間 chunk 傳遞的狀態：每次呼叫都
對自己的輸入補零，所以要做 chunk 串流時，呼叫端得自己保存 `l_context` 個 frame 的
`p`，這個 class 不會做。`UnetFsmn` 堆疊多層 FSMN，把每層回傳的 memory 傳給下一層。

## Class: `ConditionFSMN`

以 embedding `e`（例如 speaker 向量）為條件的 `FSMN`。constructor 相同，多兩個參數：

```python
ConditionFSMN(
    input_dim: int,
    output_dim: int,
    project_dim: int,
    embed_dim: int,            # e 的寬度
    l_context: int,
    r_context: int,
    dilation: int = 1,
    dropout: float = 0,
    norm_type: str = "bN1d",
    use_film: bool = False,
)
```

- `use_film=False`——`e` 沿時間廣播、串接到 `ctx` 後投影回 `project_dim`：
  `p = p + ctx + Conv1d([ctx; e])`。
- `use_film=True`——`e` 預測 FiLM 的 scale `γ` 與 bias `β`，同時套到兩個分支：
  `p = (γ p + β) + (γ ctx + β)`。

`forward(x [N, input_dim, T], embed [N, embed_dim], memory=None) -> (out, memory)`，
shape 同 `FSMN`。

## 設計說明

`SingleRNN` 投影回輸入寬度，讓 dual-path block 能以殘差方式加上它的輸出，也能換成
介面相同的其他運算子（attention、[`MambaInter`](ssm.zh-TW.md)）。FSMN 用有限上下文的
convolution 取代遞迴：look-ahead 恰好是 `r_context` 個 frame，而且訓練時可以沿時間
平行計算。

## 範例

```python
from puresound.nnet.lobe.rnn import FSMN

layers = [FSMN(256, 256, 192, l_context=3, r_context=0) for _ in range(2)]
memory = None
for layer in layers:
    x, memory = layer(x, memory)   # memory 從一層 skip 到下一層
```
