# puresound.nnet.lobe.rnn

English version: `rnn.md`

Recurrent 積木:一個統一的 RNN/LSTM/GRU wrapper 並帶輸出投影,以及
FSMN——一個以 convolution 取代 recurrence 的方案,並帶有明確、可攜帶狀態
的 memory chaining,方便 streaming 使用。

## Class: `SingleRNN`

包裝 `nn.RNN`/`nn.LSTM`/`nn.GRU`(單層),並且一律把結果投影回
`input_size`,所以不論 `hidden_size` 或 `bidirectional` 是什麼,這個
block 永遠是形狀保持(shape-preserving)的——**沒有 `proj_size` 參數**
可以投影到*其他*寬度;投影目標永遠都是 `input_size`。

```python
SingleRNN(
    rnn_type: str,
    input_size: int,
    hidden_size: int,
    bidirectional: bool = False,
    dropout: float = 0.0,
)
```

**Parameters:**
- `rnn_type` – `"RNN"`、`"LSTM"` 或 `"GRU"`(不分大小寫;透過 `getattr(nn, rnn_type.upper())` 對應)
- `input_size` – 輸入 feature 維度——同時也永遠是**輸出**維度,因為最後有一個 `nn.Linear(hidden_size * num_directions, input_size)`
- `hidden_size` – RNN 的 hidden 維度
- `bidirectional` – 若為 `True`,`num_direction = 2`,投影的輸入寬度會相應加倍
- `dropout` – 套用在 RNN 輸出、投影之前(不是 PyTorch 自己的層間 `dropout`,因為這裡永遠只有 1 層)

### `forward(x: Tensor) -> Tensor`

**Parameters:** `x` – `[N, C, T]`(`C = input_size`)

Permute 成 `[N, T, C]`,跑 RNN,dropout,投影回 `input_size`,再
permute 回來。**Returns:** `[N, C, T]`,**永遠**是 `C = input_size`——
與輸入形狀相同,絕不會是 `hidden_size` 或其他自訂的投影寬度。

---

## Class: `FSMN`

**Feedforward Sequential Memory Network** —— 用 dilated depthwise
Conv1d 取代 recurrence 的方案,並且在 layer/呼叫之間有**明確、帶狀態的
memory chaining**,而不是內部的 hidden state。

```python
FSMN(
    input_dim: int,
    output_dim: int,
    project_dim: int,
    l_context: int,
    r_context: int,
    dilation: int = 1,
    dropout: float = 0.0,
    norm_type: str = "bN1d",
)
```

**Parameters:**
- `input_dim` – 輸入 feature 維度
- `output_dim` – 輸出 feature 維度
- `project_dim` – hidden「memory」寬度——depthwise context conv 與 memory tensor 都在這個寬度上
- `l_context` / `r_context` – 過去/未來的 context frame 數;depthwise conv 的 kernel size 為 `l_context + r_context + 1`
- `dilation` – depthwise context conv 的 dilation
- `dropout` – 在 `norm_type` normalization 之後、最後才套用
- `norm_type` – [`norm.get_norm`](norm.zh-TW.md) 的代碼(預設 `"bN1d"`)

**Reference:** [aps FSMN component](https://github.com/funcwj/aps/blob/c814dc5a8b0bff5efa7e1ecc23c6180e76b8e26c/aps/asr/base/component.py#L310)。

### `forward(x, memory=None) -> Tuple[Tensor, Tensor]`

這**不是**單純的 `forward(x) -> Tensor`。它會在呼叫之間傳遞一個 memory
tensor,符合這個 codebase 裡其他 recurrent module 常用的呼叫慣例
(`(out, new_state) = layer(x, state)`)——這一點對 streaming inference
很重要:每個 chunk 呼叫時,都必須把前一個 chunk 回傳的 memory 傳回去:

```python
in_proj = Conv1d(input_dim, project_dim, kernel_size=1)(x)
ctx     = depthwise_conv(pad(in_proj, l_context, r_context))
proj    = in_proj + ctx
if memory is not None:
    proj = proj + memory              # 把呼叫端給的 memory 串進來
out     = norm(Conv1d(project_dim, output_dim, kernel_size=1)(proj))
return out, proj                      # `proj` 就是下一次呼叫要用的新 memory
```

**Parameters:**
- `x` – `[N, C, T]`(`C = input_dim`)
- `memory` – 前一次的 memory block,`[N, P, T]`(`P = project_dim`);序列/串流中的第一次呼叫則為 `None`

**Returns:** `(out, new_memory)`——`out`:`[N, output_dim, T]`;
`new_memory`:`[N, project_dim, T]`,下次呼叫時要當作 `memory` 傳回去
(例如下一個時間 chunk,或是像下方 `Unet` 的用法那樣,傳給堆疊中的下一層
FSMN——那個例子是在*同一次 forward 裡跨堆疊的 layer* chaining memory,
不只是跨時間 chunk)。

---

## Class: `ConditionFSMN`

`FSMN` 加上 FiLM 風格或串接式的條件化,來自外部 embedding(例如 speaker
向量)。繼承自 `FSMN`,沿用它全部的建構子參數,再加兩個:

```python
ConditionFSMN(
    input_dim: int,
    output_dim: int,
    project_dim: int,
    embed_dim: int,
    l_context: int,
    r_context: int,
    dilation: int = 1,
    dropout: float = 0,
    norm_type: str = "bN1d",
    use_film: bool = False,
)
```

**Parameters:**
- `embed_dim` – 條件化 embedding 的維度
- `use_film` – 若為 `False`(預設),embedding 會被 broadcast 後串接到 context 分支上,再用 `Conv1d` 投影回去;若為 `True`,embedding 改為預測一組 FiLM 的 `(scale, bias)`,在相加之前分別套用到 `proj` 與 `ctx` 上
- 其餘參數 – 與 `FSMN` 相同

### `forward(x, embed, memory=None) -> Tuple[Tensor, Tensor]`

**Parameters:**
- `x` – `[N, C, T]`
- `embed` – `[N, embed_dim]`
- `memory` – 與 `FSMN.forward` 相同

**Returns:** `(out, new_memory)`,形狀與 `FSMN.forward` 相同。

## Wiring

`DPARN` 的 dual-path block(`puresound/nnet/dparn.py`)用 `SingleRNN("LSTM",
...)` 做 inter-chunk 建模(搭配 [`attention.MhaSelfAttenLayer`](attention.zh-TW.md)
做 intra-chunk 建模)。`DPCRN` 的 intra/inter RNN 階段也用 `SingleRNN`。
`Unet` 的 FSMN 強化版本(`puresound/nnet/unet.py`)會疊多層
`FSMN`/`ConditionFSMN`,並把上一層的輸出 `memory` 串接到下一層的輸入
——這與 streaming 呼叫端要跨時間 chunk 傳遞 memory 是同一套機制。

## Example

```python
from puresound.nnet.lobe.rnn import FSMN

fsmn = FSMN(input_dim=256, output_dim=256, project_dim=192, l_context=3, r_context=3)

memory = None
out1, memory = fsmn(chunk1, memory)   # 第一個 chunk memory=None
out2, memory = fsmn(chunk2, memory)   # 把狀態帶到下一步
```
