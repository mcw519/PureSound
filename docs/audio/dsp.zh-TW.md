# puresound.audio.dsp

English version: [`dsp.md`](dsp.md)

重新取樣,加上 RBJ Audio-EQ-Cookbook biquad 設計/套用,以及建立在這兩者
之上的 `ParametricEQ`(固定、不可訓練的濾波鏈)。可訓練的對應版本是
`puresound.nnet.lobe.dsp.FrequencyEQLayer`,它呼叫同一個
`get_biquad_params`、用同樣的 filter-type 字串,拿來初始化一組頻域 EQ
權重,之後再整個一起微調。

## Functions

### `wav_resampling(wav: Tensor, origin_sr: int, target_sr: int, backend: str = "sox", torch_backend_params: Optional[Dict] = None)`

把 `wav: [..., L]` 從 `origin_sr` 重新取樣到 `target_sr`。注意參數名稱是
`origin_sr`,不是 `orig_sr`。

- `backend="sox"`(預設):透過 `torchaudio.sox_effects` 執行 sox 的
  `rate` 效果。如果裝的 torchaudio 根本沒有 `sox_effects` 這個屬性
  (某些 wheel build 沒有內建),會**靜靜地改用下面的 `torchaudio` 路徑**
  —— 不會報錯,也不會有警告。
- `backend="torchaudio"`:用 `torchaudio.transforms.Resample`,搭配
  **隨機**產生的抗鋸齒濾波器 —— `lowpass_filter_width` 從
  `{6, 16, 32, 64, 128}` 抽、`rolloff` 從 `Uniform(0.8, 0.99)` 抽、
  `resampling_method` 從 `{"sinc_interp_hann", "sinc_interp_kaiser"}` 抽
  —— 除非 `torch_backend_params` 明確提供了 `{"lp_width", "rolloff",
  "window"}`,這時就會直接重複使用那組數值,不再重新隨機。這個機制正是
  讓降取樣再升取樣的來回(見 [augmentation.md](augmentation.md) 的
  `apply_src_effect`)表現得像「一個」一致的低品質 resampler,而不是兩個
  各自獨立隨機的原因。
- `origin_sr == target_sr` 會直接短路成 no-op,但仍然回傳下面對應
  backend 該有的 tuple 形狀。

**回傳值的元素個數是依照你傳入的 `backend` 字串本身決定,不是依照實際
執行了哪一種實作:**

- `backend="sox"` → 永遠是 `(wav, target_sr)`,一個 2-tuple —— *即使*
  是上面那個靜默 fallback 的情況也一樣:那組實際被用掉的 torchaudio
  隨機濾波器參數,內部確實有算出來,但**不會**回傳給你(就直接被丟棄)。
- `backend="torchaudio"` → 永遠是 `(wav, target_sr, torch_backend_params)`,
  一個 3-tuple,讓呼叫端可以把 `torch_backend_params` 接力傳進配對的
  第二次呼叫。

```python
wav_16k, sr = wav_resampling(wav, origin_sr=48000, target_sr=16000, backend="sox")

down, sr_d, params = wav_resampling(wav, origin_sr=16000, target_sr=8000, backend="torchaudio")
up,   sr_u, _      = wav_resampling(down, origin_sr=8000, target_sr=16000,
                                     backend="torchaudio", torch_backend_params=params)
```

---

### `get_biquad_params(gain_dB: float, cutoff_freq: float, q_factor: float, sample_rate: float, filter_type: str)`

RBJ Audio-EQ-Cookbook 的 biquad 設計公式。**5 個參數全部必填 —— 沒有
任何預設值**,而且 `filter_type` 是*最後一個*位置參數。

`filter_type` 必須是下列其中之一:`"high_shelf"`、`"low_shelf"`、
`"peaking"`、`"lpf"`、`"hpf"`、`"bpf"`、`"notch"` —— shelf 的名稱有底線,
通帶濾波器則是 3 個字母的縮寫(不是 `"highpass"`/`"lowpass"`)。
`gain_dB` 只會影響 `high_shelf`/`low_shelf`/`peaking` —— `lpf`/`hpf`/
`bpf`/`notch` 的公式完全沒有引用增益項(`A = 10**(gain_dB/40)` 雖然
無條件都會被算出來,但這四種 filter type 根本沒用到它),所以這四種
情況下 `gain_dB` 傳什麼值都一樣;依慣例傳 `0.0`。

**回傳 `(b, a)`,兩個長度為 3 的 numpy array** —— 不是 5 個元素的
`(b0, b1, b2, a1, a2)`。兩者都已經除以 `a0` 正規化過,所以 `a[0] == 1.0`,
不需要另外保留一個 `a0`。

```python
b, a = get_biquad_params(gain_dB=0, cutoff_freq=100, q_factor=0.707,
                          sample_rate=16000, filter_type="hpf")
```

`puresound.nnet.lobe.dsp.FrequencyEQLayer` 在建構時,會針對每個 band
(`low_shelf`、`peaking` × N、`high_shelf`)各呼叫一次這個函式,拿來初始化
它可訓練的頻域 EQ 權重 —— 用的是跟下面 `ParametricEQ` 一樣的
shelf/peaking 詞彙,只是變成可訓練的版本。

---

### `wav_apply_biquad_filter(wav: Tensor, b_coeff: np.ndarray, a_coeff: np.ndarray)`

用 `scipy.signal.lfilter` 逐一聲道套用濾波器。內部會經過 numpy 來回轉換
(先 clone `wav`、轉成 numpy,1 維輸入會先 unsqueeze 成 `[1, L]`),回傳
一個全新的 `torch.Tensor` —— **不可微分**,梯度不會流過這個呼叫。

## Class: `ParametricEQ`

一串**固定**(不可訓練、不是 `nn.Module`)的 biquad:一個 low shelf、
N 個 peaking band、一個 high shelf,依此順序套用。

### Constructor

```python
ParametricEQ(
    sample_rate: float,
    eq_band_gain: Tuple[float, ...],
    eq_band_cutoff: Tuple[float, ...],
    eq_band_q_factor: Tuple[float, ...],
    low_shelf_gain_dB: float = 0.0,
    low_shelf_cutoff_freq: float = 80,
    low_shelf_q_factor: float = 0.707,
    high_shelf_gain_dB: float = 0.0,
    high_shelf_cutoff_freq: float = 1000,
    high_shelf_q_factor: float = 0.707,
    dtype = np.float32,
)
```

`eq_band_gain`/`eq_band_cutoff`/`eq_band_q_factor` 三者長度必須相等 ——
每個 index 對應一個 `"peaking"` band(建構子裡有 `assert`)。每次呼叫
`forward` 實際套用的濾波器總數是 `len(eq_band_gain) + 2`(含兩個
shelf)。

### Methods

#### `forward(wav: Tensor) -> Tensor`

依序透過 `wav_apply_biquad_filter` 套用每一個 biquad。

#### `plot_eq(savefig: Optional[str] = None)`

畫出整條濾波鏈合併後的 `|H(f)|`(每個 stage 的 `rfft(b) / rfft(a)`
在所有 stage 之間相乘)。**`sample_rate` 必須恰好是 `16000` 或
`32000`** —— 其他數值會直接丟出 `ValueError`(繪圖用的 FFT 大小,
512 對 1024,是依取樣率寫死的,不是動態推算)。直接用
`matplotlib.pyplot` 繪圖(不會建立或回傳 figure;每畫一張圖就呼叫一次,
如果不想要互動式顯示就傳 `savefig=...`)。

## 範例

```python
from puresound.audio.dsp import ParametricEQ, wav_resampling

wav_16k, sr = wav_resampling(wav, origin_sr=48000, target_sr=16000, backend="sox")

eq = ParametricEQ(
    sample_rate=16000,
    eq_band_gain=(3.0,),
    eq_band_cutoff=(1000.0,),
    eq_band_q_factor=(1.0,),
    low_shelf_cutoff_freq=80,
    high_shelf_cutoff_freq=7800,
)
wav_eq = eq.forward(wav_16k)
```
