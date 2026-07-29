# puresound.nnet.loss.spk

Speaker classification and verification loss functions.

## Class: `AAMsoftmax`

**Additive Angular Margin Softmax (AAM-softmax)** — a metric learning loss that adds an angular margin to the target class cosine similarity for improved speaker discriminability.

**Reference:** Deng et al., "ArcFace: Additive Angular Margin Loss for Deep Face Recognition," CVPR 2019.

### Constructor

```python
AAMsoftmax(
    in_dim: int,
    num_classes: int,
    margin: float = 0.2,
    scale: float = 30.0,
    sub_center: int = 1,
)
```

**Parameters:**
- `in_dim` – Input embedding dimension
- `num_classes` – Number of speaker classes
- `margin` – Angular margin `m` (in radians); larger values → stricter separation
- `scale` – Logit scaling factor `s`
- `sub_center` – Number of sub-centers per class for sub-center AAM-softmax (default: 1 = standard AAM)

### `forward(embeddings: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]`

**Parameters:**
- `embeddings` – L2-normalized speaker embeddings `[batch, in_dim]`
- `labels` – Integer class labels `[batch]`

**Returns:** `(loss, accuracy)` where:
- `loss` – Scalar cross-entropy loss with angular margin
- `accuracy` – Classification accuracy on the current batch

### AAM-softmax Formulation

$$\mathcal{L} = -\log \frac{e^{s \cdot \cos(\theta_{y_i} + m)}}{e^{s \cdot \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cdot \cos(\theta_j)}}$$

---

## Class: `SphereFace2`

**SphereFace2** — an improved variant of SphereFace with better optimization properties and a dual softmax formulation.

**Reference:** Wen et al., "SphereFace2: Binary Classification is All You Need for Deep Face Recognition," ICLR 2022.

### Constructor

```python
SphereFace2(
    in_dim: int,
    num_classes: int,
    margin: float = 0.2,
    scale: float = 32.0,
    lam: float = 20.0,
)
```

**Parameters:**
- `in_dim` – Input embedding dimension
- `num_classes` – Number of speaker classes
- `margin` – Angular margin
- `scale` – Logit scale factor
- `lam` – Temperature parameter for the bias term

### `forward(embeddings: Tensor, labels: Tensor) -> Tuple[Tensor, Tensor]`

**Returns:** `(loss, accuracy)`

## Comparison

| Loss | Key Advantage | When to Use |
|------|--------------|-------------|
| `AAMsoftmax` | Proven, widely used, stable | General speaker verification |
| `SphereFace2` | Better gradient properties | When AAM training is unstable |

## Example

```python
from puresound.nnet.loss.spk import AAMsoftmax

aam = AAMsoftmax(in_dim=192, num_classes=5994, margin=0.2, scale=30.0)
loss, acc = aam(embeddings, speaker_labels)
```

## Class: `GE2ELoss`

Generalized end-to-end speaker-verification loss (softmax or contrast variant)
over an `[nspks, putts, D]` embedding batch; centroids are recomputed excluding
the current utterance. Ported from `cvqluu/GE2E-Loss`.

```python
GE2ELoss(nspks: int, putts: int, init_w: float = 10.0, init_b: float = -5.0, loss_method: str = "softmax")
```

## Class: `TripletLoss`

Cosine or Euclidean triplet loss on `[N, 3, D]` (anchor, positive, negative)
embeddings with margin.

```python
TripletLoss(margin: float = 0.0, add_norm: bool = True, distance: str = "Euclidean")
```
