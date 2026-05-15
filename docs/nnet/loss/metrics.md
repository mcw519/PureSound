# puresound.nnet.loss.metrics

Speaker embedding loss functions and basic time-domain regression losses.

## Class: `GE2ELoss`

**Generalized End-to-End (GE2E) Loss** — a contrastive loss for training speaker embedding models using speaker similarity matrices.

**Reference:** Wan et al., "Generalized End-to-End Loss for Speaker Verification," ICASSP 2018.

### Constructor

```python
GE2ELoss(mode: str = "softmax")
```

**Parameters:**
- `mode` – Loss variant:
  - `"softmax"` – Softmax-based similarity loss (recommended for training)
  - `"contrast"` – Contrastive similarity loss

### `forward(embeddings: Tensor) -> Tensor`

Computes GE2E loss over a batch of speaker embeddings arranged as N speakers × M utterances.

**Parameters:**
- `embeddings` – Speaker embedding batch `[N*M, embd_dim]` or `[N, M, embd_dim]`

**Returns:** Scalar loss value.

### How GE2E Works

1. Compute a similarity matrix between all utterance embeddings and speaker centroids
2. The diagonal (same-speaker) similarities should be maximized
3. Off-diagonal (different-speaker) similarities are minimized via softmax or contrastive formulation

---

## Class: `TimeDomainBasicLoss`

Simple time-domain waveform regression losses.

### Constructor

```python
TimeDomainBasicLoss(loss_type: str = "l1")
```

**Parameters:**
- `loss_type` – Loss variant:
  - `"l1"` – Mean Absolute Error
  - `"l2"` – Mean Squared Error

### `forward(est: Tensor, ref: Tensor) -> Tensor`

**Parameters:**
- `est` – Estimated (enhanced) waveform `[batch, T]`
- `ref` – Reference (clean) waveform `[batch, T]`

**Returns:** Scalar loss value.

## Example

```python
from puresound.nnet.loss.metrics import GE2ELoss, TimeDomainBasicLoss

ge2e = GE2ELoss(mode="softmax")
loss = ge2e(embeddings)  # embeddings: [N*M, 192]

l1_loss = TimeDomainBasicLoss(loss_type="l1")
loss = l1_loss(enhanced_wav, clean_wav)
```
