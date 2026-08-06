# puresound.nnet.lobe.pooling

English version: `pooling.md`

把長度可變的序列 feature 聚合成固定大小表示的 pooling layer——主要用於
speaker embedding。

## Function: `length_to_mask`

```python
length_to_mask(
    length: torch.Tensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor
```

依 1D 序列長度建構一個 `[batch, max_len]` 的 binary mask(改寫自
[SpeechBrain](https://github.com/speechbrain/speechbrain/blob/d3d267e86c3b5494cd970319a63d5dae8c0662d7/speechbrain/dataio/dataio.py#L661))。
`max_len` 預設為 `length.max()`;`dtype`/`device` 預設沿用 `length` 自己的。
`AttentiveStatisticsPooling.forward` 內部用它在 attention softmax 之前
把 padding 的 frame 蓋掉。

```python
length_to_mask(torch.Tensor([1, 2, 3]))
# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
```

## Class: `AttentiveStatisticsPooling`

在時間軸上計算 attention 加權的 mean 與標準差,把長度可變的序列
收斂成一個固定大小的向量。

**Reference:** Okabe et al., "Attentive Statistics Pooling for Deep Speaker
Embedding," Interspeech 2018。

```python
AttentiveStatisticsPooling(channels, attention_channels=128)
```

**Parameters:**
- `channels` – 輸入 feature 維度(同時也是 attention score 的維度)
- `attention_channels` – attention 評分用 TDNN 的 hidden 維度(`Conv1d → ReLU → BatchNorm1d → Tanh → Conv1d`)

### `forward(x, lengths=None, return_weight=False)`

**Parameters:**
- `x` – `[N, C, L]`
- `lengths` – 選填,每個 batch item 的**相對**序列長度,範圍 `[0, 1]`(是 `L` 的比例,不是絕對 frame 數——內部會先乘上 `L` 再傳給 `length_to_mask`);若為 `None`,每筆資料都視為完全有效(`lengths = ones(N)`)
- `return_weight` – 若為 `True`,回傳 softmax attention weight,而不是 pooled 統計量

**Returns:**
- 若 `return_weight=False`(預設):pooled 結果 `[N, 2*C, 1]`——weighted mean 與 weighted std 串接,並多一個尾端的 singleton 時間軸
- 若 `return_weight=True`:attention weight tensor `[N, C, L]`(softmax 之後、pooling 之前)

### Attention Mechanism

1. 計分:`tanh(TDNN(x))` → `Conv1d` 投影回 `channels` 寬度
2. 用 `lengths` 把 padding 的 frame 蓋掉(填 `-inf`),再對時間軸做 softmax → `α_t`
3. Weighted mean:`μ = Σ_t α_t · x_t`
4. Weighted std:`σ = sqrt(Σ_t α_t · (x_t - μ)² )`(開根號前先 clamp 到 `eps = 1e-12`)
5. 串接 `[μ, σ]` → `[N, 2*C]`,再加一個尾端時間軸 → `[N, 2*C, 1]`

## Example

```python
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling

pool = AttentiveStatisticsPooling(channels=512, attention_channels=128)

# 長度可變的序列,lengths 以 L 的比例表示
pooled = pool(frame_features, lengths=relative_lengths)  # [N, 1024, 1]
weights = pool(frame_features, return_weight=True)        # [N, 512, L]
```
