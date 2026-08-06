# puresound.nnet.lobe.group_op

English version: `group_op.md`

Group 結構的算子:一個跨 channel 溝通層(`TAC`),以及一系列以
channel 分組的 GRU/Linear layer——透過把 feature 拆成獨立的 group,
用序列建模能力換取更少的參數量。

## Class: `TAC`

**Transform-Average-Concatenate。** 一個全域溝通層,讓每個
group(例如每支麥克風)透過共享的平均值互相交換資訊,不需要固定的
channel 數或排列順序。

```python
TAC(input_dim: int, hidden_dim: int)
```

**Parameters:**
- `input_dim` – 每個 group 的輸入 feature 維度
- `hidden_dim` – transform/mean/output 各階段使用的 hidden 維度

**Reference:** Luo et al., "End-to-End Microphone Permutation and Number
Invariant Multi-Channel Speech Separation," ICASSP 2020
([code](https://github.com/yluo42/GC3/blob/main/utility/basics.py#L28))。

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, G, C, T]`,`G` 為 group 數(例如麥克風數)

步驟:各個 group 各自獨立做 **transform**(`Linear(input_dim→hidden_dim) + PReLU`)
→ 跨 `G` 做 **average** → 對平均值做 **projection**(`Linear(hidden_dim→hidden_dim) + PReLU`)
→ 把每個 group 的 transform 輸出與(broadcast 過的)投影平均值 **concatenate**
→ **output** 投影回 `input_dim`(`Linear(2*hidden_dim→input_dim) + PReLU`)
→ `nn.BatchNorm1d(input_dim)` → 殘差相加回原始輸入。

注意這裡的 normalization 是一般(非 causal、整批統計)的
`BatchNorm1d`;原始碼裡的註解提到論文原本用的是非 causal 的
`nn.GroupNorm(1, input_size)`——這是刻意的差異,不是疏漏。

**Returns:** `[N, G, C, T]`,與輸入形狀相同。

目前 repository 的 backbone 裡沒有任何呼叫端使用——是個 library 積木。

---

## Class: `GroupedGRULayer`

把 channel 維度平均拆成 `groups` 個獨立的 `nn.GRU`,每個只看
`input_size / groups` 個 channel——相較於一個全寬的 GRU,能減少參數量,
同時保留各 group 內部的 recurrence。

```python
GroupedGRULayer(
    input_size: int,
    hidden_size: int,
    groups: int,
    bidirectional: bool = False,
    bias: bool = True,
    dropout: float = 0.0,
)
```

**Parameters:**
- `input_size` – 總輸入 feature 維度,必須能被 `groups` 整除
- `hidden_size` – 總 hidden 維度,必須能被 `groups` 整除
- `groups` – 獨立 GRU 的數量;每個處理連續的 `input_size/groups` 寬 channel 切片
- `bidirectional` – 傳給每個內部 `nn.GRU`
- `bias` – 傳給每個內部 `nn.GRU`
- `dropout` – 傳給每個內部 `nn.GRU` 自己的 `dropout` 參數(層間 dropout,只有在該 GRU 內部層數 >1 時才有意義——這裡永遠是 1 層,所以透過 `GroupedGRULayer` 本身其實不會有效果)

**沒有 `batch_first` 參數**——每個內部 `nn.GRU` 都寫死 `batch_first=True`。

> **本次已修正:** 這個參數先前被拼錯成 `droupout`,並傳入
> `nn.GRU(..., droupout=droupout, ...)`,而 `nn.GRU` 並不接受這個關鍵字
> ——只要嘗試建構這個 class,`nn.GRU.__init__` 就會立刻拋出
> `TypeError`。repository 裡沒有任何地方明確傳入這個拼錯的關鍵字,
> 所以這次修正單純是改名,不需要更新任何外部呼叫端。已實測驗證:
> `GroupedGRULayer(...)` 現在可以正確建構並執行(`forward`、
> `return_hidden=True`、以及 `flatten_parameters()` 都測試過)。

### `forward(x, h0=None, return_hidden=False)`

**Parameters:**
- `x` – `[N, C, T]`(內部會 permute 成 `[N, T, C]`,batch-first,再沿最後一軸拆成 `groups` 個連續區塊)
- `h0` – 選填的初始 hidden state,形狀 `[groups * num_directions, N, hidden_size / groups]`;每個 group 讀自己的切片(使用前會先 `.detach()`)
- `return_hidden` – 若為 `True`,一併回傳串接後的最終 hidden state

**Returns:** `outputs`(`[N, C, T]`,把所有 group 的輸出串接後 `C = hidden_size`),若 `return_hidden=True` 則回傳 `(outputs, h)`。

### `flatten_parameters()`

對每個內部 `nn.GRU` 呼叫 `flatten_parameters()`(讀取 checkpoint 或搬動
device 後,cuDNN weight-layout 的例行整理)。

---

## Class: `GroupedGRU`

疊 `num_layers` 層 `GroupedGRULayer`,並可選擇在每層之後對 group 之間做
channel shuffle(讓 group 不會在深度方向上永遠彼此隔離——與 ShuffleNet
的 channel shuffle 是同樣的想法)。

```python
GroupedGRU(
    input_size: int,
    hidden_size: int,
    num_layers: int = 1,
    groups: int = 4,
    bidirectional: bool = False,
    bias: bool = True,
    dropout: float = 0.0,
    shuffle: bool = True,
)
```

**Parameters:**
- `input_size` / `hidden_size` – 與 `GroupedGRULayer` 相同;第一層把 `input_size → hidden_size`,之後的層則是 `hidden_size → hidden_size`
- `num_layers` – 疊幾層 `GroupedGRULayer`(必須 `> 0`)
- `groups` – 傳給每一層;若 `groups == 1`,不論傳入什麼值都會強制 `shuffle=False`
- `bidirectional`、`bias`、`dropout` – 傳給每一層
- `shuffle` – 在疊起來的層之間(除了最後一層)交錯 group 之間的 channel

### `forward(x, h0=None, return_hidden=False)`

**Parameters:** 與 `GroupedGRULayer` 相同,`h0` 形狀為
`[num_layers * groups * num_directions, N, hidden_size / groups]`。

每個疊起來的層都是以 `x, s = gru(x, h0[...], return_hidden=True)` 的形式驅動;
各層的 hidden state 會被收集起來串接,而 channel shuffle(當 `shuffle=True` 時)
則在層與層之間執行,最後一層除外。

**Returns:** `x`(`[N, hidden_size, T]`),若 `return_hidden=True` 則回傳
`(x, outstates)`,其中 `outstates` 是所有層的 hidden state 疊成的
`[num_layers * groups * num_directions, N, hidden_size / groups]`——與 `h0`
的排列完全相同,所以可以直接餵回去處理下一個 chunk。

> **本次已修正:** `forward` 先前呼叫內部層時沒有傳入 `return_hidden=True`,
> 卻仍然以 `x, s = gru(...)` 的形式把結果 unpack。`GroupedGRULayer` 在沒有被
> 要求回傳 hidden state 時只會回傳單一個 `Tensor`,所以那個 unpack 等於是試圖
> 沿著 batch 軸把 tensor 拆開——除了 batch size 剛好是 2 以外,任何 batch size
> 都會拋出 `ValueError`;而 batch size 為 2 時則會悄悄丟掉 batch 軸,幾行之後
> 在 channel-shuffle 的 `.permute` 當掉。現在兩條呼叫路徑(`return_hidden`
> 開與關)都已驗證可正常運作。

---

## Class: `GroupedLinear`

Channel 分組的 `nn.Linear`:把 channel 軸拆成 `groups` 個切片,各自套用
獨立的 `Linear`,並可選擇把輸出 channel 跨 group 打散。

```python
GroupedLinear(
    input_size: int,
    hidden_size: int,
    groups: int = 1,
    shuffle: bool = True,
)
```

**Parameters:**
- `input_size` / `hidden_size` – 總維度,各自能被 `groups` 整除
- `groups` – 獨立 `Linear` layer 的數量
- `shuffle` – 把輸出 channel 跨 group 打散;若 `groups == 1` 會強制為 `False`

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, C, T]`(內部會 permute 成 `[N, T, C]`)

**Returns:** `[N, hidden_size, T]`。

---

## Class: `SqueezedGRU`

「squeeze」模式:先用一個便宜的 `GroupedLinear` 投影到共用的寬度,
用一個一般(非分組)的 `nn.GRU` 在該寬度上運算,再選擇性地用另一個
`GroupedLinear` 投影回去。

```python
SqueezedGRU(
    input_size: int,
    hidden_size: int,
    output_size: Optional[int] = None,
    num_layers: int = 1,
    linear_groups: int = 8,
)
```

**Parameters:**
- `input_size` – 輸入 feature 維度
- `hidden_size` – 同時是 `GroupedLinear` 的投影目標,也是 GRU 的 hidden size
- `output_size` – 若有給值,會用第二個 `GroupedLinear(hidden_size→output_size) + ReLU` 投影 GRU 輸出;若為 `None`,則用 `nn.Identity()`(輸出維持在 `hidden_size`)
- `num_layers` – 唯一那個共用 `nn.GRU` 的層數
- `linear_groups` – 兩個 `GroupedLinear` 階段共用的 group 數

### `forward(x, h0=None, return_hidden=False)`

**Parameters:** `x` – `[N, C, T]`

**Returns:** `[N, output_size or hidden_size, T]`,若 `return_hidden=True` 則回傳 `(output, h)`。

## Wiring

`GroupedLinear` 與 `SqueezedGRU` 被
[`multiframe.DeepFilterDecoder`](multiframe.zh-TW.md) 用來便宜地預測
逐頻率 bin 的 deep-filtering 係數。`TAC`、`GroupedGRULayer`、
`GroupedGRU` 目前在 backbone library 裡都沒有任何呼叫端。

## Example

```python
from puresound.nnet.lobe.group_op import TAC, GroupedGRULayer

tac = TAC(input_dim=64, hidden_dim=128)
gru = GroupedGRULayer(input_size=128, hidden_size=128, groups=4)

out = tac(multi_channel_features)  # [N, G, 64, T]
out = gru(features)                # [N, 128, T]
```
