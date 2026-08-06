# puresound.nnet.loss.spk

繁體中文版本：[spk.zh-TW.md](spk.zh-TW.md)

Speaker classification and verification loss functions: two margin-based
softmax variants for closed-set classification training (`AAMsoftmax`,
`SphereFace2`), and two metric-learning losses that work directly on batches
of embeddings without a classifier head (`GE2ELoss`, `TripletLoss`).

## Class: `AAMsoftmax`

**Additive Angular Margin Softmax (AAM-softmax / ArcFace)** — adds an angular
margin to the target class's cosine similarity before the softmax, so
training directly optimizes angular separation between speaker classes rather
than just classification accuracy.

### Constructor

```python
AAMsoftmax(
    embedding_dim: int,
    n_classes: int,
    margin: float = 0.2,
    scale: int = 30,
    mp: float = 0.,
    sub_center: int = 1,
    sub_center_topk: Optional[int] = 0,
    sub_center_type: str = "max",
)
```

**Parameters:**
- `embedding_dim` – input embedding dimension.
- `n_classes` – number of speaker classes.
- `margin` – angular margin `m` (radians) added to the target class's angle.
- `scale` – logit scale `s` applied after the margin.
- `mp` – margin penalty applied to hard (confusable) negative classes — see
  `sub_center_topk` below. Rescaled internally by `margin / 0.2` (so `mp` is
  specified relative to the `margin=0.2` reference point, then zeroed out if
  `margin` is ~0).
- `sub_center` – if `> 1`, each class gets `sub_center` weight vectors instead
  of one (sub-center AAM-softmax), letting a class with multi-modal embedding
  clusters (e.g. very different recording conditions for the same speaker)
  still get full credit if a sample matches *any* of its sub-centers.
- `sub_center_topk` – if `> 0`, apply the (softer) `mp` margin to the top-`k`
  hardest *negative* classes for each sample (the non-target classes with the
  highest cosine similarity to it) instead of leaving them as plain cosine —
  a hard-negative-mining pressure on top of the main margin.
- `sub_center_type` – `"max"` (take the closest sub-center) or `"avg"` (a
  softmax-weighted blend across sub-centers) when collapsing `sub_center > 1`
  down to one similarity per class.

References: sub-center ArcFace ([Deng et al.](https://ibug.doc.ic.ac.uk/media/uploads/documents/eccv_1445.pdf)),
a sub-center variant ([arXiv:2407.04291](https://arxiv.org/pdf/2407.04291v1)),
and the top-k hard-negative margin ([arXiv:2110.05042](https://arxiv.org/pdf/2110.05042)).

### `forward(x, label) -> Tensor`

**Returns loss only** — not `(loss, accuracy)`. `x` is `[batch, embedding_dim]`
(L2-normalized internally), `label` is `[batch]` (or `[batch, 1]`, squeezed).

Base case (`sub_center=1`, `sub_center_topk=0`) is the standard AAM-softmax /
ArcFace cross-entropy:

$$\mathcal{L} = -\log \frac{e^{s \cos(\theta_{y_i} + m)}}{e^{s \cos(\theta_{y_i} + m)} + \sum_{j \neq y_i} e^{s \cos(\theta_j)}}$$

With `sub_center_topk > 0`, the top-`k` hardest negative classes additionally
get the `mp`-margin logit (`cos(theta) * cos_mp + sin(theta) * sin_mp`) instead
of a plain `cos(theta)`, before the same cross-entropy is taken over all
classes.

### Config usage

```yaml
# egs/speaker_embedding/conf/PS-spk-v1.yaml / egs/target_speaker_extraction/config/default_config.yaml
loss_func:
  - type: AAMsoftmax
    weighted: 1
    args:
      embedding_dim: 192
      n_classes: 21615
      margin: 0.3
      scale: 30
```

Both recipes in this repo use the plain base case — `mp` / `sub_center` /
`sub_center_topk` are left at their defaults (i.e. off).

---

## Class: `SphereFace2`

An alternative to AAM-softmax that reframes speaker classification as
`n_classes` independent binary (same-class / different-class) decisions
instead of one softmax over all classes — hence the reference paper's title,
"Binary Classification is All You Need."

### Constructor

```python
SphereFace2(
    in_features,
    out_features,
    scale=32.0,
    margin=0.2,
    lanbuda=0.7,
    t=3,
    margin_type="C",
    sub_center: int = 1,
)
```

**Parameters:**
- `in_features` / `out_features` – embedding dimension / number of classes.
- `scale` – logit scale (same role as `AAMsoftmax.scale`).
- `margin` – margin; recommended `0.2` for `margin_type="C"`, `0.15` for `"A"`
  (`0.3` / `0.25` for the paper's "LMF" setting), per the class docstring.
- `lanbuda` – weight on the positive-pair term vs. the (`n_classes - 1`, so
  numerically dominant) negative-pair term — without this, the sum over all
  negative classes would swamp the single positive term.
- `t` – exponent in `fun_g(z) = 2 * ((z + 1) / 2)^t - 1`, a monotone
  reparameterization of the cosine score used to "adjust the score
  distribution" (docstring) before the margin is applied.
- `margin_type` – `"A"` applies the margin inside the angle (ArcFace-style,
  `cos(theta + margin)`, needs `sin(theta)`); `"C"` subtracts/adds the margin
  directly from/to the cosine (CosFace-style, `cos(theta) - margin`).
- `sub_center` – like `AAMsoftmax`, but only `"max"`-pooling over sub-centers
  is implemented here (no `"avg"` option).

An `update(margin=0.2)` method recomputes the margin-dependent trig constants
in place, for changing the margin between training stages without rebuilding
the module.

### `forward(input, label) -> Tensor`

**Returns loss only.** Every class gets both a positive-pair score
(`cos_p_theta`, a softplus loss pulling the target class's margined cosine up)
and a negative-pair score (`cos_n_theta`, pulling every other class's margined
cosine down); the target class contributes its `cos_p_theta`, every other
class contributes its `cos_n_theta`, weighted by `lanbuda` / `1 - lanbuda`
respectively, then summed and averaged over the batch. (The source keeps a
commented-out `# return output, loss` — an accuracy-computing `output` value
that this class no longer returns.)

Not currently used by any recipe in this repo.

---

## Class: `GE2ELoss`

Generalized End-to-End speaker-verification loss (softmax or contrast
variant), ported from
[`cvqluu/GE2E-Loss`](https://github.com/cvqluu/GE2E-Loss/blob/master/ge2e.py).
Centroids are recomputed **excluding** the current utterance (so a speaker's
centroid never includes the sample being scored against it), over an
`[nspks, putts, D]` batch of embeddings.

```python
GE2ELoss(
    nspks: int,
    putts: int,
    init_w: float = 10.0,
    init_b: float = -5.0,
    loss_method: str = "softmax",  # "softmax" | "contrast"
    add_norm: bool = True,
)
```

**Parameters:**
- `nspks` / `putts` – number of speakers per batch / utterances per speaker
  (`forward` reshapes the incoming `[nspks * putts, D]` embeddings into
  `[nspks, putts, D]` using these).
- `init_w` / `init_b` – initial value of the learnable affine scale/bias
  applied to the cosine-similarity matrix (`w`, `b` are `nn.Parameter`s,
  trained jointly with the rest of the model).
- `loss_method` – `"softmax"` (each utterance's loss is a softmax
  cross-entropy over similarities to every speaker's centroid) or
  `"contrast"` (a sigmoid contrast against the closest non-matching
  centroid).
- `add_norm` – L2-normalize each embedding (over the feature dimension) before
  computing centroids/similarities.

`forward(dvecs, label=None) -> Tensor` returns the summed per-utterance loss
(`label` is accepted but unused).

---

## Class: `TripletLoss`

Triplet loss on `[N, 3, D]` (anchor, positive, negative) embeddings.

```python
TripletLoss(margin: float = 0.0, add_norm: bool = True, distance: str = "Euclidean")
```

`forward(x, reduction=True)` splits `x` into anchor/positive/negative along
dim 1, optionally L2-normalizes (`add_norm`), computes `dist_pos` / `dist_neg`
using either `euclidean_distance` or `cosine_similarity` (selected by
`distance`, case-insensitive; anything else raises `NameError`), and returns
`mean(max(0, dist_pos - dist_neg + margin))` (or the unreduced per-row values
if `reduction=False`).
