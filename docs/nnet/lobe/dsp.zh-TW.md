# puresound.nnet.lobe.dsp

English version: `dsp.md`

一個可學習的 DSP 啟發式 layer:以固定串接的 biquad filter 構成的可訓練
parametric EQ,直接在頻域上套用。

## Class: `FrequencyEQLayer`

不是一個通用的「N-band」EQ——建構子裡寫死了 low-shelf + peaking-bands +
high-shelf 的 biquad 串接,並附上實際的預設頻率、增益與 Q factor。
沒有獨立的 band 數參數;band 數其實隱含在 `len(eq_band_gain)`
(預設為 7),而 `eq_band_cutoff` / `eq_band_q_factor` 至少要一樣長,
否則 `init_eq_weight` 裡的 index 查找會失敗。

```python
FrequencyEQLayer(
    n_fft: int = 512,
    sample_rate: int = 16000,
    eq_band_gain: Tuple[float] = (0.5, 5.5, -3.25, -2.5, -4, -4, -4.5),
    eq_band_cutoff: Tuple[float] = (500, 1000, 1500, 2500, 3500, 5500, 6000),
    eq_band_q_factor: Tuple[float] = (0.707, 0.707, 0.707, 0.707, 0.707, 0.707, 0.707),
    low_shelf_gain_dB: float = 0.0,
    low_shelf_cutoff_freq: float = 80,
    low_shelf_q_factor: float = 0.707,
    high_shelf_gain_dB: float = 0.0,
    high_shelf_cutoff_freq: float = 7800,
    high_shelf_q_factor: float = 0.707,
    trainable: bool = True,
)
```

**Parameters:**
- `n_fft` – 用來計算每個 biquad 頻率響應的 FFT 長度;結果曲線有 `n_fft // 2 + 1` 個 bin
- `sample_rate` – Hz,`get_biquad_params` 用它把 cutoff 頻率換算成 normalized 角頻率
- `eq_band_gain` / `eq_band_cutoff` / `eq_band_q_factor` – 每個 tuple 元素對應一個 **peaking** filter band(增益 dB、中心頻率 Hz、Q factor);預設是 7 個 band,但實際設定檔用的長度不一定一樣(例如 `egs/default_config.yaml` 設了 8 個)
- `low_shelf_gain_dB` / `low_shelf_cutoff_freq` / `low_shelf_q_factor` – 一個固定的 low-shelf 級
- `high_shelf_gain_dB` / `high_shelf_cutoff_freq` / `high_shelf_q_factor` – 一個固定的 high-shelf 級
- `trainable` – 見下方「Learnable Parameters」

### `init_eq_weight()`

在建構時組出整條 filter 串接:
1. 透過 `get_biquad_params`(`puresound/audio/dsp.py`)算出 low-shelf、每個 peaking band、以及 high-shelf 的 `(b, a)` biquad 係數——總共 `n_eq = 2 + len(eq_band_gain)` 級。
2. 計算每一級的頻率響應 `H = rfft(b, n_fft) / rfft(a, n_fft)`。
3. 把所有級**串接(cascade)**起來,做法是跨級取乘積:`H = prod(H, dim=0)`——這是 series filter chain,不是平行的 band 相加式 EQ。
4. 只取 magnitude(`H.abs()`),捨棄相位,reshape 成 `[n_fft // 2 + 1, 1]`。

### `forward(x: Tensor) -> Tensor`

**Parameters:**
- `x` – **頻域 tensor**,形狀 `[..., C, T]`,其中 `C = n_fft // 2 + 1` 為頻率 bin 數,`T` 為時間/frame 軸——**不是原始波形**。`puresound/nnet/features.py` 會餵給它 `[N, 2, C, T]` 的 complex-STFT tensor(實部/虛部疊在 axis 1);純 magnitude spectrogram `[N, C, T]` 一樣可以運作,因為 `peq` 會對任何前導軸做 broadcast。

**Returns:** `x * self.peq`,依頻率 bin 逐一相乘(broadcast),形狀與輸入相同。

### Learnable Parameters

實際上只有**一個** parameter tensor 是真正可學習的——不是個別的
band 增益/cutoff/Q factor,那些只是建構時用過一次的純 Python float,
之後 autograd 就不會再碰它們了:

| Attribute | Shape | 何時可學習 |
|-----------|-------|----------------|
| `peq` | `[n_fft // 2 + 1, 1]` | `trainable=True` → `nn.Parameter`;`trainable=False` → 註冊為 buffer |

`peq` 是 `init_eq_weight()` 算出來的、**已經串接完成、只剩 magnitude** 的
頻率響應曲線。訓練時會直接更新這條曲線(形狀不變,但不再必然對應任何
shelf/peak 級的組合);`eq_band_gain` 等參數只決定它的*初始值*。

### `get_args`(property)

回傳一個 `Dict`,內含每個建構子關鍵字參數與其目前值——
`puresound/nnet/features.py` 用它以不同的 `trainable` 旗標重新建構一個
等價的 layer(`peq_module.__class__(**peq_args)`)。

## Wiring

`puresound/nnet/features.py` 的 `FeatureEncoder` 接受一個選填的
`peq_module: FrequencyEQLayer`;如果有給,會透過 `get_args` 複製一份,
套用在 encoder 輸出的 complex-STFT(permute 成 `[N, 2, C, T]`)上,
之後才進入其餘的 feature pipeline。可透過 recipe 裡的
`freq_eq: {type: FrequencyEQLayer, eq_args: {...}}` 區塊設定
(見 `egs/default_config.yaml`、`egs/noise_suppression/config/dparn.yaml`);
也會以 `puresound.nnet.FrequencyEQLayer` 的名義重新匯出。

## Example

```python
from puresound.nnet.lobe.dsp import FrequencyEQLayer

eq_layer = FrequencyEQLayer(n_fft=1024, sample_rate=32000, trainable=True)

spec = torch.rand(1, 2, 513, 100)   # [N, 2, C=n_fft//2+1, T],例如 complex STFT
spec_eq = eq_layer(spec)
spec_eq.sum().backward()
```
