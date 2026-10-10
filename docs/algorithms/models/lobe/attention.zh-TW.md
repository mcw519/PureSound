# puresound.nnet.lobe.attention

English version: [attention.md](attention.md)

正弦位置編碼、一個在每次呼叫時才決定 causal 或局部窗 mask 的 multi-head attention
wrapper，以及由兩者組成的 post-LN Transformer encoder layer。

## Class: `PositionalEncoding`

加上 Transformer 的正弦位置編碼
`PE[t, 2i] = sin(t / 10000^(2i/d))`、`PE[t, 2i+1] = cos(t / 10000^(2i/d))`，
再套用 dropout。

```python
PositionalEncoding(
    d_model: int,          # feature 寬度；必須是偶數，否則 ValueError
    dropout: float = 0.1,  # 相加之後套用
    max_len: int = 5000,   # 預先算好的表所涵蓋的最長序列
)
```

`forward(x [N, T, C]) -> [N, T, C]`。

## Class: `MHA`

`nn.MultiheadAttention`（`batch_first=True`、投影無 bias、無 attention dropout）
加上 mask 的建構。

```python
MHA(embed_dim: int, heads: int = 1)   # embed_dim = head_dim * heads
```

`forward(query, key, value, causal: bool = True, context_range: int = None)`
接受 `[N, T, C]` tensor，回傳 `nn.MultiheadAttention` 的
`(output [N, T, C], weights)`。mask 依 query 長度建立，因此假設是 self-attention
（key 長度等於 query 長度）。第 `i` 幀可以看第 `j` 幀的條件：

| `causal` | `context_range` | 允許 |
| --- | --- | --- |
| `True` | `None` | `j <= i` |
| `True` | `k` | `i - k < j <= i`（目前幀加上 `k - 1` 個過去幀） |
| `False` | `k` | `abs(i - j) <= k - 2`；請用 `k >= 2` |
| `False` | `None` | 全部幀 |

## Class: `MhaSelfAttenLayer`

一層 encoder：self-attention、residual、LayerNorm；再接 feed-forward sublayer、
residual、LayerNorm（post-LN）。

```python
MhaSelfAttenLayer(
    feats_dim: int,                  # 模型寬度（attention 的 embed_dim）
    hidden_dim: int,                 # feed-forward 寬度；improved 時為 LSTM hidden size
    nhead: int,
    dropout: float = 0.0,            # attention 輸出、feed-forward、位置編碼
    improved: bool = False,          # 在 feed-forward 輸出層前加 LSTM [1]
    bidirectional: bool = False,     # LSTM 方向；只在 improved 時使用
    position_encoding: bool = True,  # attention 前加 PositionalEncoding
)
```

`forward(x, causal=False, context_range=None, return_atten_weight=False)`
接受 channel-first 的 `x [N, C, T]`，回傳 `[N, C, T]`；`return_atten_weight=True`
時回傳 `([N, C, T], weights)`。`causal` 與 `context_range` 原封不動傳給
`MHA.forward`。

```
src = x
x   = pos(x)                    if position_encoding
x   = norm1(src + dropout(attention(x, x, x)))
src = x
x   = lstm(x)                   if improved
x   = feedforward(x)            # Linear-ReLU-Dropout-Linear-Dropout,
                                # improved 時為 ReLU-Dropout-Linear-Dropout
x   = norm2(src + x)
```

`improved=True` 時由 LSTM 承載位置資訊，所以 `position_encoding=True` 會被忽略並
記錄警告，不建也不套用位置編碼。沒有 `improved` 時的 `bidirectional` 會被忽略並
記錄警告。

[1] Chen, Mao, Liu, "Dual-Path Transformer Network: Direct Context-Aware
Modeling for End-to-End Monaural Speech Separation", Interspeech 2020.

## 用法

`DPARN` 以及 `intra_type: attention` 的 `DPCRN`，會在每一幀的頻率位置上跑兩層
`MhaSelfAttenLayer`（bottleneck 的 channel 是 feature 軸），non-causal、
`improved=False`。第一層加位置編碼、第二層不加，位置資訊只注入一次。見
[DPARN](../dparn.zh-TW.md) 與 [DPCRN](../dpcrn.zh-TW.md)。

```yaml
backbone:
  type: DPCRN
  backbone_args:
    intra_type: attention
    intra_nhead: 4
```

## 設計說明

- mask 是呼叫參數而不是 constructor 參數，同一層可以在時間軸上跑 causal、在頻率軸上
  不加 mask。
- residual 取在位置編碼之前，skip 路徑只帶 feature，位置資訊只透過 attention
  從中推導出的東西進到輸出。
- 頻率軸沒有 causal 限制，所以沿頻率的 attention 一次矩陣乘法就看到所有位置；
  intra-frequency LSTM 則要一步一步走過。
