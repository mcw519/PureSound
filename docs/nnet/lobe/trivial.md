# puresound.nnet.lobe.trivial

Utility and convenience layers for building flexible neural network pipelines.

## Class: `LambdaLayer`

Wraps an arbitrary function as an `nn.Module`.

### Constructor

```python
LambdaLayer(fn: Callable)
```

### `forward(x: Tensor) -> Any`

Applies `fn(x)` and returns the result.

**Use case:** Inserting non-parameterized transformations into `nn.Sequential`.

---

## Class: `Magnitude`

Converts a complex stacked tensor (real + imaginary) to a magnitude spectrum.

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – Stacked real/imaginary tensor `[batch, 2*F, T]`

**Returns:** Magnitude tensor `[batch, F, T]`.

---

## Class: `Gate`

A gating mechanism that modulates features using a conditioning signal.

### Constructor

```python
Gate(in_channel: int, embed_dim: int)
```

**Parameters:**
- `in_channel` – Feature dimension to gate
- `embed_dim` – Conditioning embedding dimension

### `forward(x: Tensor, embed: Tensor) -> Tensor`

Computes gating weights from `embed` via sigmoid and applies them element-wise to `x`.

**Returns:** Gated features `[batch, in_channel, T]`.

---

## Class: `FiLM`

**Feature-wise Linear Modulation (FiLM)** — applies a learned affine transformation `γ * x + β` where `γ` and `β` are predicted from a conditioning embedding.

**Reference:** Perez et al., "FiLM: Visual Reasoning with a General Conditioning Layer," AAAI 2018.

### Constructor

```python
FiLM(in_channel: int, embed_dim: int)
```

**Parameters:**
- `in_channel` – Feature dimension to modulate
- `embed_dim` – Conditioning embedding dimension

### `forward(x: Tensor, embed: Tensor) -> Tensor`

**Parameters:**
- `x` – Feature tensor `[batch, in_channel, T]`
- `embed` – Conditioning embedding `[batch, embed_dim]`

**Returns:** Modulated tensor `[batch, in_channel, T]` = `γ(embed) * x + β(embed)`.

---

## Class: `SplitMerge`

Utility layer for splitting a tensor along the channel dimension and then merging (concatenating) two tensors.

### Constructor

```python
SplitMerge(split_at: int)
```

**Parameters:**
- `split_at` – Channel index at which to split

### Methods

#### `split(x: Tensor) -> Tuple[Tensor, Tensor]`

Splits `x` into `x[:, :split_at, :]` and `x[:, split_at:, :]`.

#### `merge(a: Tensor, b: Tensor) -> Tensor`

Concatenates `a` and `b` along the channel dimension.

---

## Class: `SpecAugment`

Spectral augmentation layer for training-time masking of time and frequency bins.

**Reference:** Park et al., "SpecAugment: A Simple Data Augmentation Method for ASR," Interspeech 2019.

### Constructor

```python
SpecAugment(
    freq_mask_param: int,
    time_mask_param: int,
    num_freq_masks: int = 1,
    num_time_masks: int = 1,
)
```

**Parameters:**
- `freq_mask_param` – Maximum width of frequency masks
- `time_mask_param` – Maximum width of time masks
- `num_freq_masks` – Number of frequency masks to apply
- `num_time_masks` – Number of time masks to apply

### `forward(x: Tensor) -> Tensor`

Applies random frequency and time masking during training. During evaluation (`.eval()` mode), returns input unchanged.

**Parameters:**
- `x` – Spectrum tensor `[batch, F, T]`

**Returns:** Masked spectrum tensor.

## Example

```python
from puresound.nnet.lobe.trivial import FiLM, SpecAugment, LambdaLayer

film = FiLM(in_channel=256, embed_dim=192)
x_cond = film(features, speaker_embedding)

spec_aug = SpecAugment(freq_mask_param=27, time_mask_param=100, num_freq_masks=2)
x_aug = spec_aug(mel_features)  # masked only during training
```
