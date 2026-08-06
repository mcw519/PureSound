# puresound.nnet.lobe.pooling

繁體中文版本：`pooling.zh-TW.md`

Pooling layers for aggregating variable-length sequential features into a
fixed-size representation — primarily for speaker embedding.

## Function: `length_to_mask`

```python
length_to_mask(
    length: torch.Tensor,
    max_len: Optional[int] = None,
    dtype: torch.dtype = None,
    device: torch.device = None,
) -> torch.Tensor
```

Builds a binary `[batch, max_len]` mask from 1D sequence lengths (adapted
from [SpeechBrain](https://github.com/speechbrain/speechbrain/blob/d3d267e86c3b5494cd970319a63d5dae8c0662d7/speechbrain/dataio/dataio.py#L661)).
`max_len` defaults to `length.max()`; `dtype`/`device` default to `length`'s
own. Used internally by `AttentiveStatisticsPooling.forward` to mask padded
frames before the attention softmax.

```python
length_to_mask(torch.Tensor([1, 2, 3]))
# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
```

## Class: `AttentiveStatisticsPooling`

Computes an attention-weighted mean and standard deviation over the time
axis, collapsing a variable-length sequence to one fixed-size vector.

**Reference:** Okabe et al., "Attentive Statistics Pooling for Deep Speaker
Embedding," Interspeech 2018.

```python
AttentiveStatisticsPooling(channels, attention_channels=128)
```

**Parameters:**
- `channels` – input feature dimension (also the output attention-score dimension)
- `attention_channels` – hidden dimension of the attention-scoring TDNN (`Conv1d → ReLU → BatchNorm1d → Tanh → Conv1d`)

### `forward(x, lengths=None, return_weight=False)`

**Parameters:**
- `x` – `[N, C, L]`
- `lengths` – optional **relative** sequence lengths in `[0, 1]` per batch item (fractions of `L`, not absolute frame counts — internally scaled by `L` before being passed to `length_to_mask`); if `None`, every item is treated as fully valid (`lengths = ones(N)`)
- `return_weight` – if `True`, return the softmax attention weights instead of the pooled statistics

**Returns:**
- if `return_weight=False` (default): pooled `[N, 2*C, 1]` — concatenation of the weighted mean and weighted std, with a trailing singleton time axis
- if `return_weight=True`: the attention weight tensor `[N, C, L]` (post-softmax, pre-pooling)

### Attention Mechanism

1. Score: `tanh(TDNN(x))` → `Conv1d` back to `channels` width
2. Mask out padded frames (`-inf` fill) using `lengths`, then softmax over time → `α_t`
3. Weighted mean: `μ = Σ_t α_t · x_t`
4. Weighted std: `σ = sqrt(Σ_t α_t · (x_t - μ)² )` (clamped to `eps = 1e-12` before the square root)
5. Concatenate `[μ, σ]` → `[N, 2*C]`, then unsqueeze a trailing time axis → `[N, 2*C, 1]`

## Example

```python
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling

pool = AttentiveStatisticsPooling(channels=512, attention_channels=128)

# Variable-length sequences, lengths given as fractions of L
pooled = pool(frame_features, lengths=relative_lengths)  # [N, 1024, 1]
weights = pool(frame_features, return_weight=True)        # [N, 512, L]
```
