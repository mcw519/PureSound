# puresound.nnet.lobe.pooling

English version: [pooling.md](pooling.md)

把變長的特徵序列壓成一個固定大小向量的 pooling，是 speaker-embedding head 需要的。
這是獨立的積木：[`EcapaTdnnExtractor`](../ecapa_tdnn.zh-TW.md) 用的是它自己的
channel-attentive `ChnAttnStatPooling`，不是這個 module。

## Function: `length_to_mask`

```python
length_to_mask(
    length: torch.Tensor,             # 1D 長度
    max_len: Optional[int] = None,    # 預設 length.max()
    dtype: torch.dtype = None,        # 預設 length.dtype
    device: torch.device = None,      # 預設 length.device
) -> torch.Tensor                     # [batch, max_len]
```

位置 `< length[i]` 為 1 的二元 mask（改寫自
[SpeechBrain](https://github.com/speechbrain/speechbrain/blob/d3d267e86c3b5494cd970319a63d5dae8c0662d7/speechbrain/dataio/dataio.py#L661)）。

```python
length_to_mask(torch.Tensor([1, 2, 3]))
# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
```

## Class: `AttentiveStatisticsPooling`

沿時間軸做 attention 加權的平均與標準差（Okabe et al., "Attentive Statistics
Pooling for Deep Speaker Embedding", Interspeech 2018）：

```
e   = Conv1d(tanh(BatchNorm(ReLU(Conv1d(x)))))     # [N, C, L]，每個 channel、每個 frame
α   = e 沿時間軸的 softmax，padding 的 frame 設為 -inf
μ   = Σ_t α_t x_t
σ   = sqrt(clamp(Σ_t α_t (x_t - μ)², 1e-12))
out = concat(μ, σ)                                  # [N, 2C, 1]
```

```python
AttentiveStatisticsPooling(
    channels,                  # 輸入寬度 C；也是 attention 分數的寬度
    attention_channels=128,    # 打分網路的 hidden 寬度
)
```

`forward(x, lengths=None, return_weight=False)`：

- `x`——`[N, C, L]`。
- `lengths`——`[0, 1]` 之間的**相對**長度（`L` 的比例）；`None` 表示每一列都是完整長度。
- `return_weight`——改回傳 attention 權重 `[N, C, L]`，而不是 pooling 後的統計量。
- 回傳 `[N, 2C, 1]`（平均與標準差串接，尾端時間軸長度 1）。

### 設計說明

Attention 權重是每個 channel 各一組，而不是每個 frame 一個權重，所以每個 channel 可以
挑自己有資訊的 frame。標準差補上單純平均缺少的二階資訊；clamp 讓常數輸入時開根號仍
可微分。

## 範例

```python
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling

pool = AttentiveStatisticsPooling(channels=512, attention_channels=128)
pooled = pool(frame_features, lengths=relative_lengths)  # [N, 1024, 1]
weights = pool(frame_features, return_weight=True)        # [N, 512, L]
```
