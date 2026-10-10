# puresound.audio.dsp

English version: [dsp.md](dsp.md)

合成 pipeline 共用的訊號處理基本元件：給會截斷的後端用的線性保護、取樣率轉換、
RBJ biquad 設計、固定的 parametric EQ，以及壓縮器增益曲線。波形一律是
`[..., L]`，時間軸在最後。

## `apply_linear(fn, wav, *, headroom=0.5, max_escalations=8)`

讓一個*線性*後端執行時，它內建的截斷器永遠不會觸發。

torchaudio 的濾波後端會在數位滿刻度（`FULL_SCALE = 1.0`）飽和：
`torchaudio.functional.lfilter` 預設截在 [-1, 1]，所有 `*_biquad` 都建在它上面、
且無法關閉。一個模擬線性運算子的階段（麥克風響應、rumble
filter、增益）遇到大訊號就會默默變成 waveshaper。這對混音／目標配對影響最大：
兩者以相同參數通過同一階段，就是為了保住位準關係；固定天花板會壓扁較大聲的
混音，而較小聲的目標毫髮無傷。

對線性運算子 `H` 與任意 `a > 0`，`H(x) = H(a·x) / a`。因此：

1. `scale = headroom / peak(wav)`；呼叫 `fn(wav · scale)`。
2. 若有任何輸出樣本達到 `|y| ≥ 1 − 1e-6`，把 `scale` 除以 8 再試。
3. 回傳 `fn(wav · scale) / scale`；`max_escalations` 次都失敗就 raise
   `RuntimeError`（該運算子的增益超過鏈中任何線性階段應有的範圍）。

靜音或非有限值的輸入直接交給 `fn`。`fn` 不可消耗亂數：重試會再呼叫它一次，
會移動 dataset 依賴的 seeded RNG 串流。隨機參數要在外面抽好再以 closure 傳入。
後端若能關掉天花板，就直接關（`lfilter(..., clamp=False)`）。

**使用者：** `AudioEffectAugmentor.apply_hpf`（device chain 的 rumble filter）
與 `apply_media_coloring`。

## `wav_resampling(wav, origin_sr, target_sr, backend="sox", torch_backend_params=None)`

取樣率轉換，兩個 backend 的行為刻意不同。

| `backend` | 實際執行 | 回傳 |
| --- | --- | --- |
| `"sox"` | `torchaudio.functional.resample`，固定的 Kaiser 窗 sinc（`lowpass_filter_width=64`、`rolloff≈0.9476`、`beta≈14.77`，即 "kaiser_best" 等級）。確定性、接近透明；不會真的呼叫 sox。 | `(wav, target_sr)` |
| `"torchaudio"` | `torchaudio.transforms.Resample`，濾波器**隨機**：`lowpass_filter_width ∈ {6, 16, 32, 64, 128}`、`rolloff ~ U(0.8, 0.99)`、`resampling_method ∈ {sinc_interp_hann, sinc_interp_kaiser}`，由 Python 的 `random` 抽，除非 `torch_backend_params = {"lp_width", "rolloff", "window"}` 指定。 | `(wav, target_sr, params)` |

`"sox"` 是中性的轉換器：載入語料、噪音與 RIR 重新取樣都用它，而且它絕不能抽
亂數，否則每次讀檔都會變成不受控的增強。`"torchaudio"` 是模擬廉價重新取樣器
的增強；回傳的 `params` 可以傳給降取樣再升取樣的第二段，讓兩段共用同一個
濾波器（見 [augmentation.zh-TW.md](augmentation.zh-TW.md) 的
`AudioEffectAugmentor.apply_src_effect`）。

`origin_sr == target_sr` 時不做任何計算；回傳形狀仍依 backend 而定
（`(wav, sr)` 或 `(wav, sr, torch_backend_params or {})`）。

`"sox"` backend 對沒有任何樣本的 `wav`（`L == 0`）也會原樣回傳，取樣率標為 `target_sr`；
polyphase kernel 無法處理它，而呼叫端把空 waveform 視為不可用的音訊。

## `get_biquad_params(gain_dB, cutoff_freq, q_factor, sample_rate, filter_type) -> (b, a)`

RBJ Audio-EQ-Cookbook 的 biquad 設計（R. Bristow-Johnson）。令
`A = 10^(gain_dB/40)`、`ω0 = 2π·cutoff_freq/sample_rate`、
`α = sin ω0 / (2·q_factor)`，`filter_type` 選擇 cookbook 的公式：

| `filter_type` | 濾波器 | 是否使用 `gain_dB` |
| --- | --- | --- |
| `"low_shelf"`、`"high_shelf"` | shelving | 是 |
| `"peaking"` | peaking EQ | 是 |
| `"lpf"`、`"hpf"`、`"bpf"`、`"notch"` | 低通／高通／帶通（峰值增益固定 0 dB）、notch | 否 |

五個參數都必填。結果是兩個已除以 `a0` 的長度 3 NumPy 陣列，所以 `a[0] == 1`。
`puresound.nnet.lobe.dsp.FrequencyEQLayer` 用它初始化可訓練的 EQ（見
[lobe/dsp](../models/lobe/dsp.zh-TW.md)）。

## `wav_apply_biquad_filter(wav, b_coeff, a_coeff) -> Tensor`

對 NumPy 副本逐聲道套用 `scipy.signal.lfilter`；1-D 輸入回傳 `[1, L]`。不會截斷
（SciPy 沒有 clamp）、在 CPU 上執行、不可微分。

## `ParametricEQ`

固定的（不可訓練、不是 `nn.Module`）串接：一個 low shelf、`N` 個 peaking band、
一個 high shelf，`forward(wav)` 依此順序套用。

```python
ParametricEQ(sample_rate, eq_band_gain, eq_band_cutoff, eq_band_q_factor,
             low_shelf_gain_dB=0.0, low_shelf_cutoff_freq=80, low_shelf_q_factor=0.707,
             high_shelf_gain_dB=0.0, high_shelf_cutoff_freq=1000, high_shelf_q_factor=0.707,
             dtype=np.float32)
```

三個 `eq_band_*` tuple 長度必須相同（每個索引一個 peaking band）。
`plot_eq(savefig=None)` 用 matplotlib 畫 `|H(f)| = Π_k |B_k(f)/A_k(f)|`；只支援
`sample_rate` 16000（512 點 FFT）或 32000（1024 點），其他值 raise `ValueError`。

## `compressor_gain(wav, sample_rate, *, threshold_db, ratio, attack_ms=5.0, release_ms=120.0, makeup=True) -> Tensor[1, T]`

前饋式壓縮器的增益曲線，**只回傳、不套用**。

1. 偵測器：`|x|` 經兩個 one-pole 濾波器平滑，attack 與 release 各用
   `a = 1 − exp(−1000 / (τ_ms · fs))`，`env = max(fast, slow)`。起音時快極點較大
   （包絡以 attack 速率上升）；衰減時慢極點較大。
2. 靜態曲線：`g_dB = −max(20·log10 env − threshold_db, 0) · (1 − 1/ratio)`。
3. Makeup：`makeup=True` 時把線性增益除以其平均，讓壓縮不同時是位準變化。

`ratio` 必須 ≥ 1（1 = 不壓縮）；attack 與 release 必須為正。輸入會被攤平，
請傳單聲道 `[1, T]`。

壓縮器是時變增益，而增益對加法可分配：`g·(near + far) = g·near + g·far`。
呼叫端從混音算出曲線，同時乘到混音與目標上，目標就仍是壓縮後混音的近場成分。
這就是它只回傳曲線的原因，也是這裡不用 `apply_media_coloring` 那個 `|x|^p`
waveshaper 的原因。取兩極點最大值的偵測器與常見的切換係數迴圈形狀相同，
卻只需兩個向量化濾波，而不是逐樣本的 Python 迴圈。

**使用者：** `DeviceChain` 的壓縮器階段（見
[device chain](../augmentation/device_chain.zh-TW.md)）。

## 範例

```python
from puresound.audio.dsp import ParametricEQ, apply_linear, wav_resampling
import torchaudio

wav_16k, sr = wav_resampling(wav, origin_sr=48000, target_sr=16000)          # 中性轉換

down, _, params = wav_resampling(wav_16k, 16000, 8000, backend="torchaudio")  # 增強
up, _, _ = wav_resampling(down, 8000, 16000, backend="torchaudio", torch_backend_params=params)

hpf = apply_linear(lambda w: torchaudio.functional.highpass_biquad(w, 16000, 100.0), wav_16k)

eq = ParametricEQ(16000, eq_band_gain=(3.0,), eq_band_cutoff=(1000.0,),
                  eq_band_q_factor=(1.0,), high_shelf_cutoff_freq=7800)
wav_eq = eq.forward(wav_16k)
```
