# puresound.nnet.lobe.norm

English version: [norm.md](norm.md)

1D（`[N, C, T]`）與 2D（`[N, C, F, T]`）特徵用的 normalization layer，以及把 config
字串對應到 normalization class 的 `get_norm`。

這裡所有 layer norm 都計算 `(x - mean) / sqrt(var + eps)`，`eps = 1e-8`
（`LayerNorm2D` 是 `/ (std + 1e-8)`），再套一個學習的 affine 轉換。差別在統計量取在
哪些軸上，而這決定了該 layer 是否 causal：只要不沿時間軸取統計量就是 causal。

## 基礎 class：`_LayerNorm`

私有。持有 `gamma`（全 1）與 `beta`（全 0），各為 `[channel_size]`，以及
`apply_gain_and_bias`：把 channel 軸移到最後、套用參數、再移回來。子類別只需實作
統計量的計算。

```python
_LayerNorm(channel_size: int)
```

## Class: `GlobLN`（別名 `gLN`）

Global layer norm：每個 batch 項目一組 mean 與 variance，取在所有非 batch 軸上
（channel 與時間一起）。輸入 `[N, C, *]`。**不是 causal**——統計量包含未來 frame。

```python
GlobLN(channel_size: int)
```

## Class: `ChanLN`（別名 `cLN`）

Channel-wise layer norm：每個 `(batch, time)` 位置一組 mean 與 variance，只取在
channel 軸上。輸入 `[N, C, *]`。Causal。

```python
ChanLN(channel_size: int)
```

## Class: `InstantLN`（別名 `iLN`）

2D map 用的 instant layer norm：把 `[N, CH, C, T]` reshape 成 `[N, CH*C, T]`，在每個
frame 上對攤平的 channel-頻率軸做 normalization。Causal。`channel_size` 必須是
`CH * C`，因為 affine 參數是每個 channel-頻率位置各一組。

```python
InstantLN(channel_size: int)
```

## Class: `LayerNorm2D`（別名 `LN2D`）

在每個 frame 上對 channel 與頻率一起（`dims=[1, 2]`）做 layer norm，affine 參數是每個
`(channel, frequency)` 配對一組——`w` 與 `b` 是 `[1, ch, f, 1]`，所以頻率寬度在建構時
就固定。輸入 `[N, ch, f, T]`。Causal。

```python
LayerNorm2D(ch: int, f: int)
```

`get_norm` 不接受 `"LN2D"`；要直接建構 `LayerNorm2D`（`TFGridNet` 就是這樣做）。

## 別名

```python
gLN  = GlobLN
cLN  = ChanLN
iLN  = InstantLN
bN1d = nn.BatchNorm1d
bN2d = nn.BatchNorm2d
gGN  = lambda x: nn.GroupNorm(1, x, 1e-8)
LN2D = LayerNorm2D
```

## Function: `get_norm`

```python
get_norm(name: str)   # 回傳 class（或 gGN 的 factory），不是 instance
```

用 channel 數呼叫回傳值來建 layer。接受的名稱只有下面六個，其他一律丟 `NameError`：

| 名稱 | layer | causal |
|---|---|:-:|
| `"gLN"` | `GlobLN` | 否 |
| `"cLN"` | `ChanLN` | 是 |
| `"iLN"` | `InstantLN` | 是 |
| `"bN1d"` | `nn.BatchNorm1d` | 推論時是（running statistics） |
| `"bN2d"` | `nn.BatchNorm2d` | 推論時是（running statistics） |
| `"gGN"` | `nn.GroupNorm(1, C, eps=1e-8)` | 否 |

## 各自用在哪裡

| layer | 使用者 |
|---|---|
| `get_norm(...)` | [`cnn.DepthwiseSeparableConv1d`](cnn.zh-TW.md)、[`rnn.FSMN`](rnn.zh-TW.md) / `ConditionFSMN`、`ConvTasNet`（`tcn_norm`）、`Unet` 及其子類別如 `DPCRN`（`norm_type`，DPCRN 預設 `"bN2d"`） |
| `InstantLN` | `Unet` 的輸入 norm，`iLN(channels[0] * input_dim)` |
| `ChanLN`、`LayerNorm2D`、`GlobLN` | `TFGridNet`（intra/inter norm、attention norm、`input_norm` = `LayerNorm2D` 或 `gLN`） |

## 設計說明

串流模型需要 causal 的 norm：`ChanLN`、`InstantLN`、`LayerNorm2D`，或 batch norm
（推論時統計量固定）。`GlobLN` 與 `gGN` 會對整段序列取統計量，只適合離線模型。

## 範例

```python
from puresound.nnet.lobe.norm import get_norm

NormCls = get_norm("cLN")
norm = NormCls(256)
out = norm(features)  # [N, 256, T]
```
