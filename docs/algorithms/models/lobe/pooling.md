# puresound.nnet.lobe.pooling

繁體中文版本：[pooling.zh-TW.md](pooling.zh-TW.md)

Pooling that collapses a variable-length feature sequence into one fixed-size
vector, as a speaker-embedding head needs. A standalone building block:
[`EcapaTdnnExtractor`](../ecapa_tdnn.md) uses its own channel-attentive
`ChnAttnStatPooling` rather than this module.

## Function: `length_to_mask`

```python
length_to_mask(
    length: torch.Tensor,             # 1D lengths
    max_len: Optional[int] = None,    # defaults to length.max()
    dtype: torch.dtype = None,        # defaults to length.dtype
    device: torch.device = None,      # defaults to length.device
) -> torch.Tensor                     # [batch, max_len]
```

Binary mask with ones at positions `< length[i]` (adapted from
[SpeechBrain](https://github.com/speechbrain/speechbrain/blob/d3d267e86c3b5494cd970319a63d5dae8c0662d7/speechbrain/dataio/dataio.py#L661)).

```python
length_to_mask(torch.Tensor([1, 2, 3]))
# tensor([[1., 0., 0.],
#         [1., 1., 0.],
#         [1., 1., 1.]])
```

## Class: `AttentiveStatisticsPooling`

Attention-weighted mean and standard deviation over time (Okabe et al.,
"Attentive Statistics Pooling for Deep Speaker Embedding", Interspeech 2018):

```
e   = Conv1d(tanh(BatchNorm(ReLU(Conv1d(x)))))     # [N, C, L], per channel and frame
α   = softmax over time of e, padded frames set to -inf
μ   = Σ_t α_t x_t
σ   = sqrt(clamp(Σ_t α_t (x_t - μ)², 1e-12))
out = concat(μ, σ)                                  # [N, 2C, 1]
```

```python
AttentiveStatisticsPooling(
    channels,                  # input width C; also the width of the attention scores
    attention_channels=128,    # hidden width of the scoring network
)
```

`forward(x, lengths=None, return_weight=False)`:

- `x` – `[N, C, L]`.
- `lengths` – **relative** lengths in `[0, 1]` (fractions of `L`); `None` treats
  every row as full length.
- `return_weight` – return the attention weights `[N, C, L]` instead of the
  pooled statistics.
- Returns `[N, 2C, 1]` (mean and std concatenated, trailing time axis of 1).

### Design notes

The attention weights are per channel, not one weight per frame, so each
channel can pick its own informative frames. The standard deviation adds
second-order information that a plain mean misses; the clamp keeps the square
root differentiable on constant input.

## Example

```python
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling

pool = AttentiveStatisticsPooling(channels=512, attention_channels=128)
pooled = pool(frame_features, lengths=relative_lengths)  # [N, 1024, 1]
weights = pool(frame_features, return_weight=True)        # [N, 512, L]
```
