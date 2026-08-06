# puresound.nnet.ecapa_tdnn

English version: [ecapa_tdnn.md](ecapa_tdnn.md)

Status: *active*（見 [nnet index](../index.zh-TW.md)）—— speaker-embedding
的 backbone（`egs/speaker_embedding/conf/PS-spk-v1.yaml`）。

**Reference:** Desplanques et al., "ECAPA-TDNN: Emphasized Channel
Attention, Propagation and Aggregation in TDNN Based Speaker Verification,"
Interspeech 2020.

典型接法：`FeatureEncoder` 設 `feats_type: fbank80_16k` 產生 80-bank 的
log-Mel features（見 [nnet.features](../features.zh-TW.md)），餵給
`EcapaTdnnExtractor(input_size=80, ...)`，每個 utterance 產出一個 embedding。

## Class: `SEModule`

Squeeze-and-Excitation channel gate。

```python
SEModule(channels: int, bottleneck: int = 128)
# forward(x) -> x * sigmoid(...)：AdaptiveAvgPool1d(1) -> Conv1d -> ReLU ->
#               BatchNorm1d -> Conv1d -> Sigmoid，gate 的 shape 是 [N, channels, 1]
```

## Class: `EcapaBlock`

Res2Net 風格的 multi-scale residual block，輸出端接一個 `SEModule`。

### Constructor

```python
EcapaBlock(inplanes, planes, kernel_size=None, dilation=None, scale=8)
```

**Parameters:**
- `inplanes` / `planes` – 輸入/輸出 channel 數（這裡每個呼叫點兩者都相等）
- `kernel_size` / `dilation` – 內部 `scale - 1` 個 dilated conv 共用同一組值。
  兩者在 signature 上預設都是 `None`，但這個預設值其實用不了 —— block 內部會拿
  `kernel_size` 除以 2 算 padding，用預設值呼叫會直接壞掉;實際上每個呼叫點都是
  兩個都明確傳值
- `scale` – Res2Net 的切分因子：把 `planes` 個 channel 切成 `scale` 組、每組
  寬度 `planes // scale`，後面每一組的 conv 還會吃前一組的*輸出*（是層級式的
  residual，不是單純攤平的 split-transform-concat）

### `forward(x: Tensor) -> Tensor`

```python
forward(x: Tensor) -> Tensor  # [N, inplanes, T] -> [N, planes, T]
```

`1x1 Conv -> Res2Net 的 multi-scale dilated conv -> 1x1 Conv -> SEModule ->
+ x`（residual）。沒有 `t_len` 這個參數 —— 每次呼叫都是一整段序列一次處理完。

## Class: `ChnAttnStatPooling`

Attentive statistics pooling：把 `[N, C, T]` 的 frame 序列聚合成一個
`[N, 2C]` 的 utterance 向量。`EcapaTdnnExtractor` 內部會用到；沒有從
`puresound.nnet.__init__` export 出去。

### Constructor

```python
ChnAttnStatPooling(input_size: int = 1536)
```

`output_size()`（是個一般 method，不是 property）回傳 `input_size * 2`。

### `forward(x: Tensor) -> Tensor`

```python
forward(x: Tensor) -> Tensor  # [N, input_size, T] -> [N, 2 * input_size]
```

先把 `x` 跟它自己在時間軸上的平均值、標準差（兩者都廣播回 `T`）concat 起來，
建出一個 `3*input_size` channel 的「全域上下文」tensor，接著擠過一個小小的
`Conv1d -> ReLU -> BatchNorm1d -> Conv1d -> Softmax(dim=2)` bottleneck，
算出一個逐 channel、逐 frame 的 attention 權重 `w`，最後回傳的是 `x` 的
**attention-加權**平均與標準差（不是上面算的那個單純統計量）串接起來：
`cat([sum(x*w, dim=2), sqrt(sum(x^2*w, dim=2) - mean^2)], dim=1)`。

## Class: `EcapaTdnnExtractor`

被 export 出來的 extractor（`from puresound.nnet import EcapaTdnnExtractor`）。

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
- `input_size` – 輸入 feature 維度（Mel-bank 數，例如 `80`）
- `embedding_size` – 輸出的 speaker-embedding 維度
- `model_scale` – 傳給每個 `EcapaBlock` 的 `scale`（Res2Net 切分因子）
- `ndim` – 三個 `EcapaBlock` 之間的隱藏 channel 寬度
- `att_size` – 餵進 `ChnAttnStatPooling` 的 channel 寬度（原始碼裡 class
  docstring 把這個參數寫成 `output_size`，但實際 constructor 的參數名是
  `att_size` —— 根本沒有 `output_size` 這個參數）

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

每個 `EcapaBlock` 吃的是 stem 輸出加上前面所有 block 輸出的*累加總和*
（multi-layer feature aggregation），不是只吃緊鄰前一個的輸出。三個
dilation（`2, 3, 4`）是寫死的，沒有開放設定。

### `forward(x: Tensor) -> Tensor`

```python
forward(x: Tensor) -> Tensor  # [N, input_size, T] -> [N, embedding_size]
```

### Example（對照 `egs/speaker_embedding/conf/PS-spk-v1.yaml`）

```python
from puresound.nnet import EcapaTdnnExtractor

model = EcapaTdnnExtractor(
    input_size=80, embedding_size=192, model_scale=8, ndim=1024, att_size=1536,
)
mel_feat = mel_bank(stft_complex)   # [N, 80, T]，例如 FeatureEncoder(feats_type="fbank80_16k")
embedding = model(mel_feat)         # [N, 192]
```
