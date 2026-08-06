# puresound.nnet.features

English version: [features.md](features.md)

位於 waveform encoder 與負責預測 mask 的 backbone 之間的 feature transform：
`Wav -> Encoder -> Features -> Backbone -> Apply Mask -> Decoder -> Wav`
（參見 `system.siso.EncDecMaskBase` / `system.miso.EncDecCondMaskBase`）。
`puresound.nnet` 只 export 了 `FeatureEncoder`（見 `puresound/nnet/__init__.py`）；
`MelBank` 與 `WeightedSum` 是內部輔助 class，需要時得直接從
`puresound.nnet.features` import。

## Class: `FeatureEncoder`

這個 module 真正的入口 —— 每個 recipe 都是透過
`nnet.FeatureEncoder(**model_dict["features"])`（`puresound/recipes.py`）建構出來的。
它把 encoder 的原始輸出轉成 backbone 設計時預期的表示法，同時回傳第二份「未經處理」的
tensor，之後 mask 就是乘在這份 tensor 上。

### Constructor

```python
FeatureEncoder(
    feats_type: str = "complex",
    drop_stft_first_bin: bool = True,
    include_specaug: bool = False,
    specaug_args: Optional[Dict] = None,
    peq_module: Optional[FrequencyEQLayer] = None,
    normalized_mode: Optional[str] = None,
    trainable: bool = False,
)
```

**Parameters:**
- `feats_type` – 只能是 `"free"`、`"complex"`、`"magnitude"`、`"log1p"`、
  `"fbank80_16k"`、`"logfbank80_16k"`、`"fbank128_16k"`、`"shrink_channel"`
  其中之一（建構時會 assert）。見下方表格。
- `drop_stft_first_bin` – 丟掉 DC bin（index 0）。只影響 `complex` /
  `magnitude` / `log1p` 這幾條分支；`fbank*` 分支建立的 `MelBank` 永遠吃完整的
  `n_fft // 2 + 1` 頻譜，不受這個 flag 影響（因為學到的 Mel filter 本身最低那個
  三角濾波器就已經把 DC 附近權重壓低，沒有東西需要另外丟）。
- `include_specaug` – 把要餵給 backbone 的 features 包上
  [`SpecAugment`](lobe/trivial.md)（`specaug_args` 會直接原樣展開餵進它的
  constructor，例如 `freq_mask_length`、`time_mask_length`、`fill_value`、
  `n_freq_mask`、`n_time_mask`、`prob` —— 下方有真實 config 範例）。
- `peq_module` – 一個已經建構好的 `FrequencyEQLayer` instance（由
  `recipes.py` 依 `freq_eq:` config block 建出來）。`FeatureEncoder` 不會直接沿用它：
  而是讀出 `peq_module.get_args`，把 `trainable` 強制改成這個 encoder 自己的
  `trainable` flag，然後用這組 args 重新建一個新的 instance。這樣一來，只要調
  `FeatureEncoder.trainable` 這一個 config 開關，就能同時決定 PEQ 跟下面的
  Mel filterbank 是否可學習，不受 `freq_eq:` block 本身怎麼寫影響。
- `normalized_mode` – `None`（預設，不做 normalization），或三種只作用在「要餵給
  backbone 的那份 tensor」上的標準化模式之一。見下方
  [normalization](#normalization) 一節。
- `trainable` – 會傳給 `MelBank` 的 filterbank，也會傳給重建後的
  `peq_module`（見上）。對 `complex` / `magnitude` / `log1p` / `free` /
  `shrink_channel` 沒有作用，因為這幾種本身就沒有可學習參數。

### `feats_type` 分派表

| `feats_type` | Transform | Output（channel-unsqueeze 之前） |
|---|---|---|
| `complex` | 視需要丟 DC bin，再 `permute(0, 3, 1, 2)` | `[N, 2, F(-1), T]` |
| `magnitude` | [`Magnitude`](lobe/trivial.md)`(drop_first=...)` | `[N, F(-1), T]` |
| `log1p` | `Magnitude(drop_first=..., log1p=True)` | `[N, F(-1), T]` |
| `fbank80_16k` | `MelBank(sr=16000, n_fft=512, n_banks=80)` | `[N, 80, T]` |
| `logfbank80_16k` | `MelBank(..., n_banks=80, apply_log=True)` | `[N, 80, T]` |
| `fbank128_16k` | `MelBank(sr=16000, n_fft=512, n_banks=128)` | `[N, 128, T]` |
| `free` | `nn.Identity()` | 不變 |
| `shrink_channel` | `x.squeeze(-1)`（延後執行，見下） | 把最後那個 size-1 軸丟掉 |

`complex`/`magnitude`/`log1p` 是搭配 `ConvEncDec`（見
[lobe/encoder](lobe/encoder.md)）使用的 STFT-domain 前端；`fbank*` 系列是給
speaker-embedding recipe 用的（`EcapaTdnnExtractor` 吃的就是 80-bank 的輸出 ——
見 [algorithms/ecapa_tdnn](algorithms/ecapa_tdnn.md)）；`free` /
`shrink_channel` 則是搭配學到的 time-domain 前端（`FreeEncDec`）給 1-D
序列型 backbone 用的。`free` 跟 `shrink_channel` 目前都沒有任何 recipe 在用。

### `forward(x: Tensor) -> Tuple[Tensor, Tensor]`

**Parameters:**
- `x` – encoder 輸出，可能是 `[N, C, T, 2]`（complex STFT：real/imag 放在最後一軸）
  或 `[N, C, T]`（學到的實數 encoder；內部會 unsqueeze 成 `[N, C, T, 1]`）。

**Returns:** `(feats, feats_for_enhanced)`，在 STFT/Mel 這幾條分支下兩者都是
`[N, CH, C, T]`（如果 transform 輸出的是 3-D tensor，會補一個 leading 的
singleton channel 軸）：
- `feats` – 會被丟進 backbone 的那份（`EncDecMaskBase` 裡的
  `self.backbone(features)`）。有開 SpecAugment 的話，只會烙進*這一份*。
- `feats_for_enhanced` – transform 輸出、完全沒動過的那份。
  `Masker.apply_*_mask_on_reim`（見 [nnet.masker](masker.md)）實際上就是把
  預測出來的 mask 乘在這份 tensor 上，所以 augmentation 永遠不會弄髒真正要重建的
  訊號 —— 只會弄髒 mask 預測器看到的那份拷貝。

如果有設 `peq_module`，可學習的 EQ 會先跑，吃的是整個 `x` reshape 成的
`[N, 2, C, T]`（real/imag 當作 channel 軸）—— 也就是說它是在 STFT domain
裡運作，不是直接對原始 waveform 處理。

### Normalization

當 `normalized_mode` 不是 `None` 時，`_apply_normalization` 會對要餵給 backbone
的那份 tensor 做標準化：`(x - mean) / (std + 1e-5)`，其中 mean 與 std 是在該模式
指定的 reduce 軸上算出來的（全部都帶 `keepdim=True`，所以結果維持輸入的
`[N, CH, C, T]` shape）：

| `normalized_mode` | Reduce 軸 | 統計量共用於 | 各自獨立於 |
|---|---|---|---|
| `per_feature` | `(1, 2)` | channel + freq | 每個 batch item、每個 time frame |
| `per_channel` | `1` | channel | 每個 batch item、每個 freq bin、每個 time frame |
| `all_feature` | `(1, 2, 3)` | channel + freq + time | 每個 batch item |

其他任何非 `None` 的字串都會 raise `NameError`。

只有 `feats`（backbone 的輸入）會被 normalize。`feats_for_enhanced` 刻意保持原本的
scale，因為預測出來的 mask 是乘回*這一份*上面 —— 如果連它也 normalize，重建出來的
訊號就會被連帶縮放掉。有開 SpecAugment 的話，會在 normalization 之後才套用在
`feats` 上。

當 `normalized_mode` 是 `None` 時 tensor 原樣通過，行為與「整條 pipeline 根本沒有
normalization 這一步」是 bit-identical 的 —— 目前所有現役 recipe 都屬於這種情況。

> **Checkpoint 注意事項。** 這條路徑在被修好之前一直是靜默失效的
>（`_apply_normalization` 有算出結果，卻沒有 `return`）。
> `egs/speaker_embedding/conf/PS-spk-v1.yaml`、`PS-spk-v1-1.yaml` 與
> `egs/target_speaker_extraction/config/default_config.yaml` 以前寫的是
> `normalized_mode: all_feature`，但 normalization 從來沒有真的執行過，所以它們
> 隨附/衍生出來的 checkpoint 事實上都是在「未經 normalize 的 features」上訓練的。
> 這些 config 現在都改成空的 `normalized_mode:` 並加上說明註解，讓已釋出的
> checkpoint 與其 config 保持一致。要打開任何一個都必須連同重新訓練一起做。

### `back_forward(x: Tensor) -> Tensor`

在套用 mask 之後、decoder 做 inverse STFT 之前，把 `drop_stft_first_bin` 的效果反過來：
對 `complex` / `magnitude` / `log1p`，在 `dim=2` 補回一個全零的 DC bin，讓 tensor
的 bin 數對回 encoder 原本的輸出;其他 `feats_type` 則原樣通過不做任何事。
呼叫點在 `EncDecMaskBase._spec_to_wav` / `EncDecCondMaskBase.forward`，就在
mask 套用完之後。

### Example（對照 `egs/voice_isolate/config/train_dpcrn.yaml`）

```python
from puresound.nnet import FeatureEncoder

feats = FeatureEncoder(
    feats_type="complex",
    drop_stft_first_bin=True,
    trainable=False,
    include_specaug=False,
)

complex_spec = encoder(wav)                 # [N, 257, T, 2]（ConvEncDec，512-pt FFT）
features, features_for_enhanced = feats(complex_spec)
# features               -> [N, 2, 256, T]，餵給 backbone
# features_for_enhanced  -> [N, 2, 256, T]，被預測出的 mask 相乘
```

```python
# 對照 egs/speaker_embedding/conf/PS-spk-v1.yaml
feats = FeatureEncoder(
    feats_type="fbank80_16k",
    drop_stft_first_bin=True,
    trainable=False,
    normalized_mode=None,            # 原因見上面的 checkpoint 注意事項
    include_specaug=True,
    specaug_args=dict(
        freq_mask_length=4, time_mask_length=3, fill_value=0.0,
        n_freq_mask=3, n_time_mask=5, prob=0.5,
    ),
)
mel_feat, _ = feats(complex_spec)   # [N, 1, 80, T] -> 餵進 EcapaTdnnExtractor 前先 squeeze(1)
```

## Class: `MelBank`

把 complex STFT tensor 轉成 Mel filterbank 表示法。`FeatureEncoder` 在每種
`fbank*`/`logfbank*` `feats_type` 下都會在內部用到它;只有在想單獨做實驗時才需要
自己直接建一個。

### Constructor

```python
MelBank(
    sr: int = 16000,
    n_fft: int = 512,
    n_banks: int = 80,
    apply_log: bool = False,
    utt_norm: bool = False,
    trainable: bool = False,
)
```

**Parameters:**
- `sr` – 取樣率（Hz），會傳給 `lobe.stft.mel_filterbank`
- `n_fft` – FFT size;filterbank matrix 是依 `n_fft // 2 + 1` 個線性 bin 建出來的
- `n_banks` – Mel filter 數量（輸出 channel 數）
- `apply_log` – 在 filterbank matmul 之後做 `log(melspec + 1e-8)`
- `utt_norm` – 對每個 Mel channel 減掉它在時間軸上的 per-utterance 平均值
  （只有減平均;儘管取了這個名字，實際上並沒有除以標準差 —— 見下方原始碼）
- `trainable` – 若為 `True`，filterbank matrix（`[n_fft//2+1, n_banks]`）會是
  可學習的 `nn.Parameter`，否則是固定的 buffer

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – complex 頻譜，`[N, F, T, 2]`，其中 `F == n_fft // 2 + 1`
  （real/imag 疊在最後一軸 —— 對應 `ConvEncDec` 的原始輸出，*還沒*做過
  DC-bin 的丟棄）

**Returns:** `[N, n_banks, T]`。

```python
spec_imag = x[..., 0]
spec_real = x[..., 1]
mag = torch.sqrt(spec_real.pow(2) + spec_imag.pow(2) + 1e-8)  # epsilon 是無條件加的，
# sqrt 在 0 的梯度是 Inf，所以只要有任何一個 STFT bin 剛好是零，backprop 經過這層就會被毒化。
melspec = torch.matmul(mag.permute(0, 2, 1), self.filterbank)  # [N, T, n_banks]
```

## Class: `WeightedSum`

對單一個已疊好的 tensor 的**最後一軸**做可學習的加權組合。目前整個 codebase
沒有任何地方會去建構它（`FeatureEncoder` 不會，任何 recipe 也不會）——
它是一個現成的 building block，是為了某種「多重表示法加權組合」
（例如 SSL-layer-weighted-sum 風格的 features）而準備的，但目前 repo 裡還沒有
任何東西把它組裝起來使用。

### Constructor

```python
WeightedSum(n_samples: int, trainable: bool = True)
```

**Parameters:**
- `n_samples` – 疊在輸入最後一軸上的項目數
- `trainable` – 若為 `True`，權重向量（`w`，初始化為每個都是 `1/n_samples`）
  是可學習參數；若為 `False`，則是固定在同一組 uniform-average 初始值的 buffer

注意這裡的權重是單純的可學習純量 —— **沒有 softmax normalization**，
跟 SUPERB 那種 layer-weighted-sum module 不一樣。

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – 單一個 tensor，shape 為 `[..., n_samples]`（呼叫端要先自己疊好 ——
  這不是一個吃 `List[Tensor]` 的 API）

**Returns:** `(x * self.w).sum(dim=-1)`，也就是 `[...]`（最後一軸被 reduce 掉）。
