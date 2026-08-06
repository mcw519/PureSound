# puresound.nnet.lobe.attention

English version: `attention.md`

正弦位置編碼(sinusoidal positional encoding)、一個在 forward 時才決定
causal / local-context masking 的輕量 multi-head-attention wrapper,以及一個
可將 feed-forward sublayer 換成 LSTM 的完整 self-attention encoder block。

## Class: `PositionalEncoding`

為序列加上標準 Transformer 的正弦位置編碼。

```python
PositionalEncoding(d_model: int, dropout: float = 0.1, max_len: int = 5000)
```

**Parameters:**
- `d_model` – feature 維度;必須為偶數(需要成對的 `sin`/`cos`),否則拋出 `ValueError`
- `dropout` – 加上位置編碼後套用的 dropout ——**預設為 `0.1`**,不是 `0.0`
- `max_len` – 預先算好的 `pe` buffer 所支援的最長序列長度

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – `[N, T, C]`

內部會先 permute 成 `[T, N, C]` 以加上 `pe` buffer,套用 dropout 後再
permute 回來。**Returns:** `[N, T, C]`,與輸入形狀相同。

---

## Class: `MHA`

`nn.MultiheadAttention` 的輕量 wrapper,以 **forward 時的參數** 加上
causal / local-context window masking。建構子本身只吃兩個參數——dropout
與 in/out-projection 的 bias 都是寫死的,無法透過建構子調整:

```python
MHA(embed_dim: int, heads: int = 1)
```

```python
self.atten = nn.MultiheadAttention(
    embed_dim=embed_dim, num_heads=heads, dropout=0, batch_first=True, bias=False,
)
```

**Parameters:**
- `embed_dim` – attention 總維度(`embed_dim = head_dim * heads`)
- `heads` – attention head 數量

### `forward(query, key, value, causal: bool = True, context_range: int = None) -> Tuple[Tensor, Tensor]`

Masking 完全是 `forward` 時的選擇。`causal` 與 `context_range` 兩者組合出四種
attention 模式:

| `causal` | `context_range` | 可以看到的範圍 |
|:---:|:---:|---|
| `True` | `None` | 標準 causal mask —— 目前 frame 以及所有過去的 frame |
| `True` | `k` | causal **且**限定範圍 —— 只看最近 `k` 個 frame(更久以前的歷史被排除) |
| `False` | `k` | 非 causal 的局部 window —— 以每個位置為中心 `±(k-1)` 個 frame |
| `False` | `None` | 無 mask —— 完整雙向 attention |

**Parameters:**
- `query`、`key`、`value` – `[N, T, C]`(`batch_first=True`)
- `causal` – **預設為 `True`**;若要雙向 attention,呼叫端必須明確傳入 `causal=False`
- `context_range` – window 長度;依上表與 `causal` 組合

**Returns:** `(attn_output, attn_weights)`,直接來自
`nn.MultiheadAttention.forward` 的回傳;`attn_output` 為 `[N, T, C]`。

---

## Class: `MhaSelfAttenLayer`

一個完整的 self-attention encoder block:self-attention + residual + norm,
接著是 feed-forward(或 LSTM)sublayer + residual + norm。normalization
發生在每次 residual 相加**之後**(post-LN,如原始 Transformer 論文的做法)——
不是 pre-LN。

```python
MhaSelfAttenLayer(
    feats_dim: int,
    hidden_dim: int,
    nhead: int,
    dropout: float = 0.0,
    improved: bool = False,
    bidirectional: bool = False,
    position_encoding: bool = True,
)
```

**Parameters:**
- `feats_dim` – model feature 維度(attention 的 `embed_dim`,也是 residual stream 的寬度)
- `hidden_dim` – feed-forward 的 hidden size;當 `improved=True` 時則是內部 `nn.LSTM` 的 hidden size
- `nhead` – attention heads 數,會傳入 `MHA(feats_dim, heads=nhead)`
- `dropout` – 套用在 attention 輸出、feed-forward 路徑內,以及(若 `position_encoding` 為真)`PositionalEncoding` 內部
- `improved` – 將線性 feed-forward sublayer 換成 `nn.LSTM`(見下方參考文獻)。這個 class **沒有 `causal` 建構子參數**——見下面的 `forward`
- `bidirectional` – 只有在 `improved=True` 時才有意義(控制內部 LSTM);當 `improved=False` 時會被忽略,並印出警告("Ignored bidirectional option since no LSTM here.")
- `position_encoding` – 在 attention 前加上 `PositionalEncoding`;當 `improved=True` 時會被忽略,並印出警告("Ignored position_encoding option here replaced by LSTM modeling."),因為 LSTM 本身已經能編碼順序資訊

**Reference:** Dual-Path Transformer Network —— Direct Context-Aware Modeling
for End-to-End Monaural Speech Separation(`improved` 這個 FF↔LSTM 互換的想法出處)。

### `forward(x, causal=False, context_range=None, return_atten_weight=False)`

**Parameters:**
- `x` – `[N, C, T]` —— **channel-first**,與 `MHA` 本身的 `[N, T, C]` 相反;這個 class 會在內部自行 permute
- `causal` – 會傳入內部的 `MHA.forward` 呼叫。這裡**預設為 `False`**,覆蓋了 `MHA.forward` 自己預設的 `True`——因為這個 layer 每次都會明確傳入 `causal`,所以 `MHA` 內部的預設值其實從未真正生效過
- `context_range` – 原封不動傳給 `MHA.forward`
- `return_atten_weight` – 若為 `True`,一併回傳 attention weights

Block 結構:

```
src = x                                          # 位置編碼之前的 residual
x   = pos(x)   if position_encoding else x       # 位置項只餵給 attention 分支
x   = self_atten(x, x, x, causal, context_range)
x   = norm1(src + dropout(x))
src = x
x   = recurrent(x)   if improved else x          # 只有 LSTM——前面沒有 linear layer
x   = feedforward(x)                             # Linear-ReLU-Drop-Linear-Drop;improved 時為 ReLU-Drop-Linear-Drop
x   = norm2(src + x)
```

注意 `src` 是在 `pos(x)` **之前**擷取的,所以 residual/skip connection
本身不會直接帶有位置資訊——只有 attention sublayer 從中萃取出的結果才會。

**Returns:** `x`,形狀 `[N, C, T]`(與輸入相同),若 `return_atten_weight=True`
則回傳 `(x, w)`。

## Wiring

`DPARN` 的 dual-path block(`puresound/nnet/dparn.py`)疊了兩層
`MhaSelfAttenLayer` 做 intra-chunk 建模——第一層 `position_encoding=True`,
第二層 `position_encoding=False`(位置資訊只注入一次,不重複注入),兩層都是
非 causal、`improved=False`。同一個 block 裡的 inter-chunk 建模則改用
[`rnn.SingleRNN`](rnn.zh-TW.md)。

## Example

```python
from puresound.nnet.lobe.attention import MhaSelfAttenLayer, PositionalEncoding

pos_enc = PositionalEncoding(d_model=256, dropout=0.1)
atten   = MhaSelfAttenLayer(feats_dim=256, hidden_dim=1024, nhead=8)

x = torch.randn(4, 256, 100)   # [N, C, T]
x_out = atten(x, causal=False)
```
