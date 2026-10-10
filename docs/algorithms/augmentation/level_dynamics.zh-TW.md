# 位準與動態

English version: [level_dynamics.md](level_dynamics.md)

合成流程中所有與音量相關的部分：位準度量、SNR 與 SIR 混音、成對峰值保護、增益與
削波失真、淡入淡出，以及動態範圍壓縮器。函式位於 `puresound/audio/volume.py`、
`puresound/audio/noise.py` 與 `puresound/audio/dsp.py`；呼叫它們的裝置鏈各級見
[裝置鏈](device_chain.zh-TW.md)。

## 1. 位準度量

`volume.py` 沿最後一軸提供三種振幅度量：

```
rms(x)  = sqrt(mean(x²))      能量位準
avg(x)  = mean(|x|)           平均絕對振幅
peak(x) = max(|x|)            峰值
```

- **RMS** 與能量直接相關，是所有 SNR 與 SIR 計算的基礎。
- **avg** 是一階統計量，對零星的大峰值較不敏感。
- **peak** 只在數位域有物理意義，代表與滿刻度的距離。峰值保護（§3）使用它。

`peak/rms` 是 crest factor，描述訊號的動態程度；語音通常約 12–18 dB，壓縮會
降低它（§7）。

`normalize_waveform(wav, amp_type)` 除以三者之一（`amp_type` 為 `rms`、`avg`、
`peak`，分母加 `1e-14`）；`rescale_waveform(wav, target_lvl, amp_type, scale)`
先正規化，再乘上 `target_lvl`（線性，或 dB 以 `10^(L/20)` 轉換）。

**載入時的位準。** recipe 設定 `dataset.gain_normalized_to`（單位 dB，例如
`-28`）時，每段語音都以 `AudioIO.open(target_lvl=...)` 載入並縮放到該 RMS 位準。
所有來源因此以同一位準進入合成，來源本身的音量不再是線索
（[距離線索](distance_cues.zh-TW.md)）。此欄位留空時，語音保留檔案本身的位準。

## 2. SNR 與 SIR 混音

語音對噪音（SNR）與前景對干擾者（SIR）的混音共用同一個函式
`noise.add_bg_noise(wav, noise, snr_list)` 與同一套推導。

### 2.1 縮放係數

給定語音 `s`、噪音 `n` 與目標比值（dB），求 `α` 使

```
20·log10( rms(s) / rms(α·n) ) = SNR_dB
α = rms(s) / ( rms(n) · 10^(SNR_dB/20) )
```

噪音先做 RMS 正規化（`rms(n) = 1`），因此 `α = rms(s) / 10^(SNR_dB/20)`，
`y = s + α·n`。函式對 `snr_list` 的每個值各回傳一個混音與一個縮放後的噪音。

### 2.2 噪音前處理

- 多聲道噪音只取第一聲道。
- 多段噪音各自 RMS 正規化、串接，再正規化一次。第二次是必要的，因為兩段不等長
  的單位 RMS 訊號串接後，整體 RMS 不再是 1。
- 噪音比語音長時隨機位置裁切；較短時循環平鋪後裁切。

recorded noise 觸發時，`NoiseStage` 以機率 `prob / 4` 傳入兩段噪音（「動態」
噪音），否則傳入一段（[場景建構](scene_construction.zh-TW.md)）。

### 2.3 比值的意義

- **RMS 以整列計算，包含靜音。** 若 6 s 的列中語音只占 2 s，`rms(s)` 比說話時
  的位準低約 `10·log10(3) ≈ 4.8 dB`，說話期間的 SNR 因此比名目值高約 4.8 dB。
  名目 SNR 與難度的關係因而取決於語音密度。改用只算語音段的 RMS 會移動每個
  recipe 的有效 SNR，屬於分佈改變（[工程契約](engineering_contract.zh-TW.md)）。
- **SIR 定義在 overlap gating 之後。** gating 在混音前把每個干擾者的部分區段
  靜音，因此 SIR 指的是干擾者實際產生的能量；同一個名目 SIR 下，被 gating 得
  越多，它說話時就越大聲。
- **白噪音的 SNR 相對於當下的混音。** 白噪音在 recorded noise 之後加入，因此它的
  參考是已含該噪音的混音。

### 2.4 白噪音

`noise.add_bg_white_noise(wav, snr_list)`：

```
σ = rms(s) / 10^(SNR_dB/20),   n[k] ~ N(0, σ²)
```

零均值高斯噪音的 RMS 等於其標準差，因此不需要再正規化。

## 3. 成對峰值保護

有兩處會把混音與目標同除以一個共同峰值；這是線性縮放，保住兩者的位準關係與
疊加性：

- `DynamicBaseDataset.avoid_audio_clipping(wav_list)` 在 `ns.py` 中於前景／
  干擾者混音與 echo 之後、speed 擾動之前執行：若清單中最大峰值超過 1，每個訊號
  都除以它。
- `DeviceChain._analogue_to_digital` 在轉換器處做同樣的事
  （[裝置鏈](device_chain.zh-TW.md)）。

兩者都不削波；蓄意的過載只存在於 volume stage 的削波分支。

## 4. 增益失真

`volume.rand_gain_distortion(wav, sample_rate, start_time=None, duration=None)`
模擬 AGC 跳動或位準事故：隨機一段區間乘上一個增益，結果再削波。

```
g = 4^z,  z ~ N(0, 1)
y = clip(x · mask, −1, 1),   mask 在區間內為 g，其他為 1
```

`g_dB = 20·log10(4)·z ≈ 12.04·z`，因此增益在 dB 域呈常態，中位數 0 dB，68 % 落在
±12 dB 內、95 % 落在 ±24 dB 內；在 dB 上對稱的分佈才讓變大聲與變小聲同樣可能。
未給 `start_time`/`duration` 時，區間在整段中均勻抽取。這裡的削波是模型的一部分：
被推過頭的片段確實會削波。這與 `apply_linear` 要移除的 backend 飽和是兩回事
（[頻譜與通道效果](spectral_channel.zh-TW.md)）。

它可以經由 `AudioEffectAugmentor.apply_gain_distortion` 呼叫；沒有任何任務流程
使用它。裝置鏈的增益是 volume stage 的 `g ~ U(perturbed_range)`。

## 5. 削波

`volume.wav_clipping(wav, min_quantile=0.0, max_quantile=0.9)` 以輸入樣本的
分位數為門檻削波：

```
lo = Q_min_quantile(x),  hi = Q_max_quantile(x),  y = clip(x, lo, hi)
```

固定的絕對門檻對安靜的列毫無作用，對大聲的列則造成嚴重破壞。分位數門檻把參數
對應到失真的**量**：`max_quantile = 0.9` 在任何位準下都削掉最大的 10 % 樣本。
做法取自 URGENT challenge 的模擬流程。函式預期單聲道輸入（`[L]` 或 `[1, L]`）。

在裝置鏈中，volume stage 抽出 `min_q ~ U(clipping_range.min)` 與
`max_q ~ U(clipping_range.max)`，以混音的分位數削混音；目標預設削在混音的門檻，
`target_clipping: own_quantile` 則改在目標自己的分位數削（[裝置鏈](device_chain.zh-TW.md)）。

## 6. 淡入淡出

`volume.wav_fade_in` 與 `volume.wav_fade_out(wav, sr, fade_len_s,
fade_begin_s=0, fade_shape="linear")` 乘上一條增益包絡。令 `f` 在
`fade_len_s · sr` 個樣本內由 0 線性升到 1：

| 形狀 | 淡入 | 淡出 |
|---|---|---|
| `linear` | `f` | `1 − f` |
| `exponential` | `2^(f−1) · f` | `2^(−f) · (1 − f)` |
| `logarithmic` | `log10(0.1 + f) + 1` | `log10(1.1 − f) + 1` |

包絡前後以 1 補齊並夾在 [0, 1]；`fade_begin_s + fade_len_s` 必須落在訊號長度內。
logarithmic 開頭上升快（接近 dB 線性），exponential 開頭慢。這些是函式庫工具，
沒有任何任務流程呼叫它們。

## 7. 動態範圍壓縮器

`dsp.compressor_gain(wav, sample_rate, *, threshold_db, ratio, attack_ms=5.0,
release_ms=120.0, makeup=True)` 模擬「這段錄音經過壓縮器」，廣播、會議與發佈
鏈都會這樣做。它回傳增益曲線 `[1, T]`，而不是壓縮後的訊號。

### 7.1 包絡偵測器

作用在 `|x|` 上的不對稱一階峰值跟隨器：

```
a_att = 1 − exp(−1000 / (attack_ms  · fs))
a_rel = 1 − exp(−1000 / (release_ms · fs))
one_pole(x, a):  y[n] = a·x[n] + (1 − a)·y[n−1]
env[n] = max( one_pole(|x|, a_att), one_pole(|x|, a_rel) )
```

一階濾波器的步階響應為 `1 − (1−a)^n`；令 `(1−a)^(fs·τ) = e^−1` 得
`a = 1 − exp(−1/(fs·τ))`，其中 1000 用來換算毫秒。

教科書式的偵測器逐樣本在 attack 與 release 係數之間切換。取一快一慢兩個極點的
逐點最大值，形狀相同：起音時快極點較高，包絡以 attack 速率上升；衰減時慢極點
較高，包絡以 release 速率下降。兩者不是同一條曲線，但這是兩次向量化的
`lfilter(..., clamp=False)`，而不是在 data loader 裡逐樣本跑 Python 迴圈。增強
真正需要的性質（對起音反應快、衰減時不 pumping）由測試釘住，而不是照抄某一種
硬體設計。

### 7.2 增益

```
E_dB[n] = 20·log10(env[n])
over[n] = max(0, E_dB[n] − T)
g_dB[n] = −over[n] · (1 − 1/R)
g[n]    = 10^(g_dB[n]/20)
```

輸入超過門檻 `over` dB 時，輸出應只超過 `over/R` dB，因此衰減量為
`over·(1 − 1/R)`。`R = 1` 是不壓縮；`R → ∞` 是 limiter。`ratio < 1`（expander）
以及非正的 attack 或 release 會拋出 `ValueError`。

### 7.3 Makeup

`makeup=True` 時 `g ← g / mean(g)`。壓縮器的平均增益必然小於 1，少了這一步，
壓縮與「把整列調小聲」就是同一種增強，模型學哪一個都行。正規化之後，唯一可觀察
的效果是包絡形狀的改變。`mean(g) = 1` 並不限制 `max(g)`，安靜段落可能被拉高，
峰值可能超過輸入；超過滿刻度的部分由下游的轉換器 stage 處理。

### 7.4 為什麼目標可以跟隨壓縮器

壓縮器是時變增益，而增益可以分配到和的每一項：

```
g[n]·(near[n] + far[n]) = g[n]·near[n] + g[n]·far[n]
```

混音與目標乘上同一個 `g` 之後，目標仍然精確等於壓縮後混音的近場成分。media
coloring 的 `|x|^p` waveshaper（[頻譜與通道效果](spectral_channel.zh-TW.md)）
沒有這種分解，只能在混音前模擬房間裡的某個裝置，無法模擬被壓縮過的錄音。

曲線由**混音**計算，因為這才是麥克風後面的真實壓縮器看到的訊號；只由目標算出
的曲線是任何裝置都不會產生的效果。

這一級存在的理由：包絡被壓平會改變模型從錄音讀出的直達聲對殘響比，使被壓縮的
遠方講者看起來像近場講者；絕對位準則不會移動這個估計。單純的增益 stage 無法
取代它。

## 設定

| Knob | Schema | 作用位置 |
|---|---|---|
| `dataset.gain_normalized_to` | `DatasetConfig` | 載入時的 `AudioIO.open(target_lvl=...)` |
| `augmentation_volume` | `VolumeAugmentation`：`perturbed_range`（線性增益）、`clipping_prob`、`clipping_range.min`、`clipping_range.max`（分位數範圍）、`target_clipping` | `DeviceChain._volume`：擲一次決定增益或削波，不會兩者皆做 |
| `augmentation_compressor` | `CompressorAugmentation`：`threshold_db_range`（dB，相對振幅 1）、`ratio_range`、`attack_ms_range`、`release_ms_range`（ms） | `DeviceChain._compressor` |
| `augmentation_noise` | `NoiseAugmentation`（`snr_range`、`snr_bands`、`prob_white_noise`、`white_noise_snr_range` 等） | `NoiseStage`（[場景建構](scene_construction.zh-TW.md)） |
| `augmentation_speech.snr_range` | `SpeechAugmentation` | 前景對干擾者的 SIR 抽樣 |

`CompressorAugmentation` 要求 `ratio_range` 下界 ≥ 1、`attack_ms_range` 下界 > 0。

```yaml
augmentation_compressor:
  used: True
  prob: 0.35
  threshold_db_range: [-34.0, -22.0]
  ratio_range: [1.5, 6.0]
  attack_ms_range: [2.0, 15.0]
  release_ms_range: [60.0, 300.0]
```

## 附註

- 壓縮器位於前級之後、轉換器之前，屬於線性組，因為目標要跟隨它。
- 壓縮曲線與訊號在兩者最短的共同長度上相乘；未被覆蓋的尾端樣本保持原值。
- 改變 §2.3 的 RMS 慣例會改變合成分佈。
