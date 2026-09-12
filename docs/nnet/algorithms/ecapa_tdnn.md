# puresound.nnet.ecapa_tdnn

繁體中文版本：[ecapa_tdnn.zh-TW.md](ecapa_tdnn.zh-TW.md)

Status: *active* (see [nnet index](../index.md)) — the speaker-embedding
backbone (`egs/speaker_embedding/conf/PS-spk-v1.yaml`).

**Reference:** Desplanques et al., "ECAPA-TDNN: Emphasized Channel
Attention, Propagation and Aggregation in TDNN Based Speaker Verification,"
Interspeech 2020.

Typical wiring: a `FeatureEncoder` with `feats_type: fbank80_16k` produces
80-bank log-Mel features (see [nnet.features](../features.md)), which feed
`EcapaTdnnExtractor(input_size=80, ...)` to produce one embedding per
utterance.

## Class: `SEModule`

Squeeze-and-Excitation channel gate.

```python
SEModule(channels: int, bottleneck: int = 128)
# forward(x) -> x * sigmoid(...): AdaptiveAvgPool1d(1) -> Conv1d -> ReLU ->
#               BatchNorm1d -> Conv1d -> Sigmoid, gate shape [N, channels, 1]
```

## Class: `EcapaBlock`

A Res2Net-style multi-scale residual block with an `SEModule` at its output.

### Constructor

```python
EcapaBlock(inplanes, planes, kernel_size=None, dilation=None, scale=8)
```

**Parameters:**
- `inplanes` / `planes` – input/output channel count (equal in every call site here)
- `kernel_size` / `dilation` – shared by all `scale - 1` internal dilated
  convs. Both default to `None` in the signature, but that default is not
  actually usable — the block divides `kernel_size` by 2 to compute padding,
  so calling with the defaults would crash; every real call site passes
  both explicitly
- `scale` – Res2Net split factor: splits `planes` channels into `scale`
  groups of width `planes // scale`, each subsequent group's conv also
  consumes the previous group's *output* (hierarchical residual, not a flat
  split-transform-concat)

### `forward(x: Tensor) -> Tensor`

```python
forward(x: Tensor) -> Tensor  # [N, inplanes, T] -> [N, planes, T]
```

`1x1 Conv -> Res2Net multi-scale dilated convs -> 1x1 Conv -> SEModule ->
+ x` (residual). There is no `t_len` parameter — every call is one full
sequence at a time.

## Class: `ChnAttnStatPooling`

Attentive statistics pooling: aggregates a `[N, C, T]` frame sequence into
one `[N, 2C]` utterance vector. Used internally by `EcapaTdnnExtractor`; not
exported from `puresound.nnet.__init__`.

### Constructor

```python
ChnAttnStatPooling(input_size: int = 1536)
```

`output_size()` (a plain method, not a property) returns `input_size * 2`.

### `forward(x: Tensor) -> Tensor`

```python
forward(x: Tensor) -> Tensor  # [N, input_size, T] -> [N, 2 * input_size]
```

Builds a `3*input_size`-channel "global context" tensor by concatenating `x`
with its own time-mean and time-std (each broadcast back to `T`), squeezes
that through a small `Conv1d -> ReLU -> BatchNorm1d -> Conv1d ->
Softmax(dim=2)` bottleneck to get a per-channel, per-frame attention weight
`w`, then returns the **attention-weighted** mean and std of `x` (not the
plain statistics computed above) concatenated together:
`cat([sum(x*w, dim=2), sqrt(sum(x^2*w, dim=2) - mean^2)], dim=1)`.

## Class: `EcapaTdnnExtractor`

The exported extractor (`from puresound.nnet import EcapaTdnnExtractor`).

### Constructor

```python
EcapaTdnnExtractor(
    input_size: int,
    embedding_size: int = 192,
    model_scale: int = 8,
    ndim: int = 1024,
    att_size: int = 1536,
)
```

**Parameters:**
- `input_size` – input feature dimension (Mel-bank count, e.g. `80`)
- `embedding_size` – output speaker-embedding dimension
- `model_scale` – `scale` forwarded to every `EcapaBlock` (Res2Net split factor)
- `ndim` – hidden channel width through the three `EcapaBlock`s
- `att_size` – channel width feeding `ChnAttnStatPooling` (the class
  docstring in source calls this parameter `output_size`, but the actual
  constructor argument is `att_size` — there is no `output_size` parameter)

### Architecture

```
[N, input_size, T]
  -> Conv1d(input_size, ndim, kernel=5, pad=2) -> ReLU -> BatchNorm1d
  -> x1 = EcapaBlock(x)          (dilation=2)
  -> x2 = EcapaBlock(x + x1)     (dilation=3)
  -> x3 = EcapaBlock(x + x1 + x2) (dilation=4)
  -> Conv1d(3*ndim, att_size, kernel=1)(cat([x1, x2, x3])) -> ReLU
  -> ChnAttnStatPooling(att_size) -> BatchNorm1d(2*att_size)
  -> Linear(2*att_size, embedding_size)
```

Each `EcapaBlock` consumes the *running sum* of the stem output and every
prior block's output (multi-layer feature aggregation), not just its
immediate predecessor. The three dilations (`2, 3, 4`) are hardcoded, not
configurable.

### `forward(x: Tensor) -> Tensor`

```python
forward(x: Tensor) -> Tensor  # [N, input_size, T] -> [N, embedding_size]
```

### Example (mirrors `egs/speaker_embedding/conf/PS-spk-v1.yaml`)

```python
from puresound.nnet import EcapaTdnnExtractor

model = EcapaTdnnExtractor(
    input_size=80, embedding_size=192, model_scale=8, ndim=1024, att_size=1536,
)
mel_feat = mel_bank(stft_complex)   # [N, 80, T], e.g. FeatureEncoder(feats_type="fbank80_16k")
embedding = model(mel_feat)         # [N, 192]
```
