# puresound.nnet.lobe.group_op

English version: [group_op.md](group_op.md)

分組結構的 layer：`TAC` 讓一組 channel group（例如多支麥克風）透過平均值交換資訊；
分組的 GRU / Linear 則把 feature 軸切成互相獨立的片段，以減少參數與計算量。

這裡所有序列 layer 的輸入都是 channel-first 的 `[N, C, T]`。

## Class: `TAC`

Transform-Average-Concatenate。

```python
TAC(input_dim: int, hidden_dim: int)
```

`forward(x [N, G, C, T]) -> [N, G, C, T]`，`C = input_dim`：

```
h_g = PReLU(Linear(input_dim -> hidden_dim)(x_g))          # 每個 group、每一幀
m   = PReLU(Linear(hidden_dim -> hidden_dim)(mean_g h_g))
y_g = PReLU(Linear(2 * hidden_dim -> input_dim)([h_g, m]))
out = x + BatchNorm1d(y)
```

論文以 `GroupNorm(1, C)` 正規化，統計量取自整段序列；`BatchNorm1d` 在 inference
時使用 running statistics，因此這一層維持 causal。

Reference: Luo, Chen, Mesgarani, Yoshioka, "End-to-end microphone permutation
and number invariant multi-channel speech separation", ICASSP 2020
（[參考程式碼](https://github.com/yluo42/GC3/blob/main/utility/basics.py#L28)）。

## Class: `GroupedGRULayer`

`groups` 個互相獨立的 `nn.GRU`（`batch_first=True`），各自處理 channel 中連續的
`input_size / groups` 一段。

```python
GroupedGRULayer(
    input_size: int,        # 可被 groups 整除
    hidden_size: int,       # 可被 groups 整除
    groups: int,
    bidirectional: bool = False,
    bias: bool = True,
    dropout: float = 0.0,   # nn.GRU 的層間 dropout；每個 GRU 只有一層
)
```

`forward(x, h0=None, return_hidden=False)`：

| | shape |
| --- | --- |
| `x` | `[N, input_size, T]` |
| `h0` | `[groups * D, N, hidden_size / groups]`，雙向時 `D = 2` 否則為 1；使用前會 detach |
| 輸出 | `[N, hidden_size * D, T]` |
| hidden（`return_hidden` 時） | 與 `h0` 相同的排列 |

`flatten_parameters()` 會對每個內部 GRU 呼叫同名方法。

## Class: `GroupedGRU`

疊 `num_layers` 層 `GroupedGRULayer`，層與層之間可做 channel shuffle，讓各 group
在深度方向交換 feature（與 ShuffleNet 相同的想法）。

```python
GroupedGRU(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    groups: int = 4,
    bidirectional: bool = False,
    bias: bool = True,
    dropout: float = 0.0,
    shuffle: bool = True,   # groups == 1 時強制為 False；最後一層之後不做
)
```

`forward(x [N, input_size, T], h0=None, return_hidden=False)` 回傳
`[N, hidden_size * D, T]`（雙向時 `D = 2`，否則 `D = 1`）；`return_hidden` 時另回傳堆疊成
`[num_layers * groups * D, N, hidden_size / groups]` 的 hidden state，也就是 `h0`
的排列。

`bidirectional=True` 且 `num_layers > 1` 時，後面每一層是以下一層輸出的
`hidden_size * 2` 個 channel 為輸入寬度建立的。

## Class: `GroupedLinear`

`groups` 個互相獨立的 `nn.Linear`，各自作用在一段 channel 上。

```python
GroupedLinear(
    input_size: int,    # 可被 groups 整除
    hidden_size: int,   # 輸出寬度，可被 groups 整除
    groups: int = 1,
    shuffle: bool = True,  # 輸出 channel 跨 group 的固定交錯；groups == 1 時關閉
)
```

`forward(x [N, input_size, T]) -> [N, hidden_size, T]`。

## Class: `SqueezedGRU`

先以分組的線性投影降到 `hidden_size`，在此寬度跑一個一般的 GRU，最後可選擇再以分組
投影輸出。

```python
SqueezedGRU(
    input_size: int,
    hidden_size: int,                  # 投影寬度，也是 GRU hidden size
    output_size: Optional[int] = None, # None -> 不做輸出投影
    num_layers: int = 1,               # GRU 的層數
    linear_groups: int = 8,            # 兩個投影的 group 數
)
```

```
x = ReLU(GroupedLinear(input_size -> hidden_size)(x))
x, h = GRU(x, h0)
x = ReLU(GroupedLinear(hidden_size -> output_size)(x))    if output_size
```

`forward(x [N, input_size, T], h0=None, return_hidden=False)` 回傳
`[N, output_size or hidden_size, T]`；`return_hidden` 時另回傳
`h [num_layers, N, hidden_size]`。

## 用法

[`multiframe.DeepFilterDecoder`](multiframe.zh-TW.md) 以 `SqueezedGRU` 做時間建模，
以 `GroupedLinear`（`shuffle=False`）輸出係數。`TAC`、`GroupedGRULayer`、
`GroupedGRU` 在 backbone 中沒有呼叫者。

## 設計說明

- 把寬度 `C` 的 layer 分成 `G` 組，權重數大約除以 `G`；疊層之間的 shuffle 避免各組
  一直互相隔離。
- `SqueezedGRU` 以縮小的寬度跑循環，寬的輸入與輸出映射交給分組 linear——這是
  DeepFilterNet 用來壓低循環路徑成本的做法。
- `GroupedGRULayer` 會 detach `h0`，所以把狀態從一個 chunk 帶到下一個 chunk 時，
  梯度不會回傳到前一個 chunk。
