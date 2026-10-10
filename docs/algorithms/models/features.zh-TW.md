# puresound.nnet.features

English version: [features.md](features.md)

位於 waveform encoder 與預測 mask 的 backbone 之間的 feature transform：
`Wav -> Encoder -> Features -> Backbone -> Apply Mask -> Decoder -> Wav`
（`system.siso.EncDecMaskBase`、`system.miso.EncDecCondMaskBase`）。
`FeatureEncoder` 由 `puresound.nnet` export；`MelBank` 與 `WeightedSum` 是輔助
class，從 `puresound.nnet.features` import。

## Class: `FeatureEncoder`

每個 recipe 都依自己的 `model.features` block 建一個
（`puresound/recipes.py` 裡的 `nnet.FeatureEncoder(**model_dict["features"])`）。
它把 encoder 輸出轉成 backbone 預期的表示法，並另外回傳一份沒有 augmentation、
沒有 normalization 的拷貝，預測出的 mask 就乘在這份上。

### Constructor

```python
FeatureEncoder(
    feats_type: str = "complex",           # 見分派表
    drop_stft_first_bin: bool = True,      # 丟掉 DC bin（只作用於 complex / magnitude / log1p）
    include_specaug: bool = False,         # 對 backbone 輸入做 SpecAugment
    specaug_args: Optional[Dict] = None,   # lobe.trivial.SpecAugment 的 kwargs
    peq_module: Optional[FrequencyEQLayer] = None,  # 由 recipes.py 依 `freq_eq:` 建出
    normalized_mode: Optional[str] = None, # None | per_feature | per_channel | all_feature
    trainable: bool = False,               # Mel filterbank 與 PEQ 是否可學習
)
```

- `feats_type` 會轉小寫，且必須是下表八種之一（assert）。
- `drop_stft_first_bin` 只作用於 `complex`、`magnitude`、`log1p`。`fbank*` 分支
  永遠讀完整的 `n_fft // 2 + 1` 頻譜，也就是其 Mel matrix 建立時的 bin 數。
- `specaug_args` 以 keyword arguments 傳給 [`SpecAugment`](lobe/trivial.zh-TW.md)：
  `freq_mask_length`、`time_mask_length`、`fill_value`，以及可選的
  `n_freq_mask`、`n_time_mask`、`prob`。`include_specaug=True` 時必填。
- `peq_module` 是已建好的 [`FrequencyEQLayer`](lobe/dsp.zh-TW.md)。`FeatureEncoder`
  讀它的 `get_args`，把 `trainable` 換成自己的 flag 後重建一個新 instance，因此
  單一個 `trainable` 開關同時決定 PEQ 與 Mel filterbank 是否學習。
- `trainable` 對 `complex`、`magnitude`、`log1p`、`free`、`shrink_channel` 無作用，
  這幾種沒有參數。

### `feats_type` 分派表

| `feats_type` | Transform | channel unsqueeze 之前的輸出 |
|---|---|---|
| `complex` | 視需要丟 DC bin，再 `permute(0, 3, 1, 2)` | `[N, 2, F, T]` |
| `magnitude` | [`Magnitude`](lobe/trivial.zh-TW.md)`(drop_first=drop_stft_first_bin)` | `[N, F, T]` |
| `log1p` | `Magnitude(drop_first=..., log1p=True)` | `[N, F, T]` |
| `fbank80_16k` | `MelBank(sr=16000, n_fft=512, n_banks=80)` | `[N, 80, T]` |
| `logfbank80_16k` | `MelBank(sr=16000, n_fft=512, n_banks=80, apply_log=True)` | `[N, 80, T]` |
| `fbank128_16k` | `MelBank(sr=16000, n_fft=512, n_banks=128)` | `[N, 128, T]` |
| `free` | `nn.Identity()` | 不變 |
| `shrink_channel` | `squeeze(-1)`，在 augmentation 之後才套用 | 去掉最後的 size-1 軸 |

`F` 是 encoder 的 bin 數，丟掉 DC bin 時減一。`fbank*` 固定對應 16 kHz、512 點
STFT（257 bin）。

`complex`、`magnitude`、`log1p` 搭配 STFT 前端 `ConvEncDec`
（[lobe/encoder](lobe/encoder.zh-TW.md)）；`fbank*` 餵給 speaker-embedding 模型，例如
[`EcapaTdnnExtractor`](ecapa_tdnn.zh-TW.md)；`free` 與 `shrink_channel` 搭配學習式
time-domain 前端 `FreeEncDec`，給 1-D backbone 用。

### `forward(x) -> (feats, feats_for_enhanced)`

```python
forward(x: Tensor) -> Tuple[Tensor, Tensor]
# x: [N, C, T, 2]（complex STFT，real/imag 在最後一軸）或 [N, C, T]（unsqueeze 成 [N, C, T, 1]）
# 回傳兩個 [N, CH, C', T] tensor（transform 輸出為 3-D 時 CH = 1）
```

1. 若有 `peq_module`，PEQ 先作用在視為 `[N, 2, C, T]` 的 `x` 上，也就是在
   STFT domain 運作。
2. transform 產生 `feats_for_enhanced`。
3. `feats` 取自 `feats_for_enhanced`，有設 `normalized_mode` 就先 normalize，
   有開 SpecAugment 再套用（非訓練模式下 SpecAugment 不做事）。

backbone 收到的是 `feats`。mask 乘在 `feats_for_enhanced` 上
（[nnet.masker](masker.zh-TW.md)），所以 augmentation 與 normalization 不會改動要重建的訊號。

### Normalization

`normalized_mode` 對 `feats` 做 `(x - mean) / (std + 1e-5)`，統計量取自該模式的
reduce 軸（`keepdim=True`）：

| `normalized_mode` | Reduce 軸 | 共用統計量的軸 | 各自獨立 |
|---|---|---|---|
| `per_feature` | `(1, 2)` | channel + frequency | 每個 item、每個 frame |
| `per_channel` | `1` | channel | 每個 item、每個 bin、每個 frame |
| `all_feature` | `(1, 2, 3)` | channel + frequency + time | 每個 item |

其他非 `None` 的值會 raise `NameError`。`all_feature` 的統計量涵蓋整段 utterance，
無法逐 frame 執行；`per_feature` 與 `per_channel` 是逐 frame 的。checkpoint 只在
訓練時所用的模式下有效。

### `back_forward(x) -> Tensor`

在套用 mask 之後、decoder 之前把 DC bin 補回：`complex`、`magnitude`、`log1p` 且
`drop_stft_first_bin=True` 時，在 `dim=2` 最前面補一個全零 bin。其他 type 原樣通過。
呼叫點是 `EncDecMaskBase._spec_to_wav` 與 `EncDecCondMaskBase.forward`。

### Config 用法

```yaml
# egs/voice_isolate/config/train_dpcrn.yaml
model:
  features:
    feats_type: complex
    drop_stft_first_bin: True
    trainable: False
    include_specaug: False
```

```yaml
# egs/speaker_embedding/conf/PS-spk-v1.yaml
model:
  features:
    feats_type: fbank80_16k
    drop_stft_first_bin: True
    trainable: False
    normalized_mode:
    include_specaug: True
    specaug_args:
      freq_mask_length: 4
      time_mask_length: 3
      fill_value: 0.0
      n_freq_mask: 3
      n_time_mask: 5
      prob: 0.5
```

搭配 512 點的 `ConvEncDec` 時，第一份 config 把 `[N, 257, T, 2]` 轉成兩個
`[N, 2, 256, T]` tensor。

### 設計說明

- DC bin 不帶語音，又讓頻率軸成為奇數（257 bin）。丟掉後剩 256 bin，可被 stride 2
  的 CNN stack 整除；`back_forward` 在 inverse STFT 前把它補回。
- 回傳兩個 tensor，讓訓練期的擾動（SpecAugment、normalization）只作用在預測器的輸入上。

## Class: `MelBank`

magnitude 頻譜乘上 Mel filterbank matrix。`FeatureEncoder` 在 `fbank*` type 下建立它。

```python
MelBank(
    sr: int = 16000,
    n_fft: int = 512,          # matrix 為 [n_fft // 2 + 1, n_banks]
    n_banks: int = 80,
    apply_log: bool = False,   # log(mel + 1e-8)
    utt_norm: bool = False,    # 減去每段 utterance 在時間軸上的平均（只減平均）
    trainable: bool = False,   # filterbank 為 nn.Parameter 而非 buffer
)
# forward: [N, n_fft // 2 + 1, T, 2] -> [N, n_banks, T]
```

`mag = sqrt(re^2 + im^2 + 1e-8)`、`mel = mag^T @ filterbank`，再視設定取 log 與減平均。
matrix 來自 [`lobe.stft.mel_filterbank`](lobe/stft.zh-TW.md)。平方根內的 `1e-8` 讓剛好為零的
bin 梯度仍有限，loss 經過這層 backprop 時需要這一點。

## Class: `WeightedSum`

對單一個已疊好的 tensor 的最後一軸做可學習加權和，用來組合多種表示法（例如 SSL 各層）。
目前沒有 recipe 建立它。

```python
WeightedSum(n_samples: int, trainable: bool = True)
# forward: [..., n_samples] -> [...]   (x * w).sum(-1)
```

`w` 每一項初始為 `1 / n_samples`；`trainable` 時是參數，否則是 buffer。沒有 softmax，
所以權重不受總和為一的限制。
