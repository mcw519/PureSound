# puresound.nnet.lobe.pooling

Pooling layers for aggregating sequential features into fixed-size representations, primarily for speaker embedding.

## Class: `AttentiveStatisticsPooling`

Computes an attention-weighted mean and standard deviation over the time dimension, producing a single fixed-size vector from a variable-length sequence.

**Reference:** Okabe et al., "Attentive Statistics Pooling for Deep Speaker Embedding," Interspeech 2018.

### Constructor

```python
AttentiveStatisticsPooling(channel: int, attention_channel: int = 128)
```

**Parameters:**
- `channel` – Input feature dimension
- `attention_channel` – Hidden dimension for the attention scoring MLP

### `forward(x: Tensor, t_len: Optional[Tensor] = None) -> Tensor`

Computes attentive statistics (weighted mean + weighted std) over the time axis.

**Parameters:**
- `x` – Input feature sequence `[batch, channel, T]`
- `t_len` – Optional sequence lengths `[batch]` for masked pooling (ignores padding)

**Returns:** Pooled representation `[batch, 2 * channel]` (concatenation of weighted mean and weighted std).

### Attention Mechanism

1. Compute attention scores from a tanh-activated 2-layer MLP over `x`
2. Apply softmax to get attention weights `α_t` (with masking if `t_len` is provided)
3. Compute weighted mean: `μ = Σ α_t * x_t`
4. Compute weighted std: `σ = sqrt(Σ α_t * x_t² - μ²)`
5. Concatenate: `[μ, σ]` → `[batch, 2*channel]`

## Example

```python
from puresound.nnet.lobe.pooling import AttentiveStatisticsPooling

pool = AttentiveStatisticsPooling(channel=512, attention_channel=128)

# Variable-length sequences with mask
pooled = pool(frame_features, t_len=seq_lengths)  # [B, 1024]
```
