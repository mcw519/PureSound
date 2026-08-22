# 位準與動態

English version: [`level_dynamics.md`](level_dynamics.md)

本章處理 pipeline 中所有與「音量」相關的手法：位準的三種度量、SNR/SIR 混音
的推導、增益失真、削波、fade，以及唯一的時變動態手法——動態範圍壓縮器。

## 演算法面

### 1. 位準的三種度量

`puresound/audio/volume.py` 提供三種振幅度量，用途不同：

```
rms(x)  = sqrt( mean(x²) )      # 能量位準
avg(x)  = mean(|x|)             # 平均絕對振幅
peak(x) = max(|x|)              # 峰值
```

**選用原則**：

* **RMS** 與訊號能量直接相關，是所有 SNR/SIR 計算的基礎。對語音這類非定常
  訊號，RMS 反映的是整段的平均能量。
* **平均絕對振幅**（`avg`）是 `AudioIO` 的預設正規化基準。它比 RMS 對大峰值
  不敏感（一階 vs 二階矩），對含有零星大脈衝的訊號較穩定。
* **峰值** 只有在數位域才有明確意義，對應滿刻度（full scale）與削波風險。
  它是 A/D 邊界（[ch6](device_chain.zh-TW.md)）唯一使用的度量。

三者的比值本身帶有訊息：`peak/rms` 是波峰因數（crest factor），描述訊號的
動態程度。語音的波峰因數通常在 12–18 dB，經過壓縮後會下降——這是 §7 的
壓縮器影響近/遠讀出的機制之一。

`normalize_waveform` 除以其中之一（加 `1e-14` 防除零）；`rescale_waveform`
先正規化再乘上目標位準（linear 或 dB，dB 換算為 `10^(L/20)`）。

**語料進場的位準拉平**：`AudioIO.open` 以 `target_lvl`
（recipe 的 `audio_gain_normalized_to`，dB RMS）rescale 每一條載入的語音。
所有聲源進入合成流程時位準已經一致，這是 [ch3](distance_cues.zh-TW.md) §1.1
「位準線索被移除」的第一個成因。

### 2. SNR / SIR 混音的推導

SIR（前景語者對干擾語者）與 SNR（語音對噪音）使用同一條實作路徑
（`puresound/audio/noise.py::add_bg_noise`），數學相同。

#### 2.1 目標與求解

給定語音 `s`、噪音 `n`、目標 SNR（dB），求縮放係數 `α` 使得

```
20 · log10( rms(s) / rms(α·n) ) = SNR_dB
```

因為 `rms(α·n) = α · rms(n)`：

```
rms(s) / (α · rms(n)) = 10^(SNR_dB/20)
α = rms(s) / ( rms(n) · 10^(SNR_dB/20) )
```

實作先把噪音 RMS 正規化（`rms(n) = 1`），式子簡化為

```
α = rms(s) / 10^(SNR_dB/20)
y = s + α · n
```

#### 2.2 實作中的預處理

多條噪音的處理順序值得留意：先**逐條** RMS 正規化，串接之後**再整段** RMS
正規化。第二次正規化是必要的——兩條各自單位 RMS 的訊號串接後，整段的 RMS
仍為 1 只在兩段等長時成立，長度不等時需要重新正規化才能保證 `rms(n) = 1`。

長度對齊：噪音長於語音時隨機裁一段窗（增加同一條噪音的使用多樣性）；短於
語音時循環平鋪到足夠長度再裁切。

#### 2.3 三個語義細節

**(a) RMS 在整列上計算，包含靜音段。** 這決定了 SNR 的實際意義。若語音在
6 秒的列中只說了 2 秒，`rms(s)` 被靜音段拉低約 `10·log10(3) ≈ 4.8 dB`，
因此「說話期間的實際 SNR」比標稱值高約 4.8 dB。

這不是 bug，但它讓標稱 SNR 與感知難度之間的關係依賴語音密度。若要改成
active-speech RMS（只在語音幀上計算），所有 recipe 的有效 SNR 分佈都會改變，
等同重新定義語料難度——這是分佈斷代（[ch8](engineering_contract.zh-TW.md)）。

**(b) SIR 是對 gating 後的訊號定義的。** overlap gating
（[ch7](scene_construction.zh-TW.md)）在混音之前執行，干擾者已有大段被靜音。
`add_bg_noise` 收到的是 gating 後的訊號，因此 SIR 是對「干擾者實際發聲的
總能量」定義的，而不是對原始語句。gating 越強，同一個標稱 SIR 對應的
「干擾者說話時的瞬時音量」越大。

**(c) 白噪路徑的 SNR 是相對當下混音。** 白噪在錄音噪音之後加入，其參考
`rms(s)` 是「已含錄音噪音的混音」，不是原始語音。

#### 2.4 白噪的生成

```
σ = rms(s) / 10^(SNR_dB/20)
n[k] ~ N(0, σ²)
```

高斯分佈的 RMS 恰等於標準差 σ，所以直接把 σ 設為目標 RMS 即可，不需額外
正規化。

### 3. 增益失真（`rand_gain_distortion`）

模擬 AGC 突跳或錄音位準事故：在隨機起點、隨機長度的區間內乘上一個增益。

```
g = 4^z,    z ~ N(0, 1)
y = clip(x · mask, −1, 1)
```

#### 3.1 增益分佈的設計

`4^z` 在 dB 域是常態分佈：

```
g_dB = 20·log10(4^z) = z · 20·log10(4) ≈ z · 12.04
```

因此 `g_dB ~ N(0, 12²)`。這個參數化的意義是：中位數為 0 dB（不變），約 68%
的列位於 ±12 dB 內，95% 位於 ±24 dB 內。在 dB 域使用常態分佈是增益擾動的
慣例——人耳對音量的感知接近對數，在 dB 域對稱的分佈才代表「一樣機率變大聲
或變小聲」。

#### 3.2 這裡的 clip 是刻意的

`clip(±1)` 在這個函式裡是**失真模型的一部分**：增益推爆的區段本來就會削頂。
這與 `apply_linear`（[ch5](spectral_channel.zh-TW.md)）保護的「無意飽和」
性質不同——後者是線性算子被 backend 的隱含限幅器污染，是缺陷；這裡的削頂
是被建模的現象。

### 4. 削波（`wav_clipping`）

分位數式硬削波：

```
lo = Q_{min_quantile}(x),   hi = Q_{max_quantile}(x)
y = clip(x, lo, hi)
```

#### 4.1 為什麼用分位數而非絕對門檻

絕對門檻（例如固定削在 ±0.8）的效果取決於訊號的位準：一個峰值 0.3 的安靜
訊號完全不受影響，一個峰值 1.0 的訊號被大幅削掉。同一組參數在不同列上產生
的失真量差異極大。

分位數門檻則自適應該列的振幅分佈：`max_quantile = 0.9` 意味著「削掉振幅
最大的 10% 樣本」，不論該列整體多大聲。參數因此直接對應失真的**程度**，而
非門檻的絕對位置。這個做法取自 URGENT challenge 的資料模擬流程。

#### 4.2 target 的削波方式

在 device chain 中，target 不是用自己的分位數削波，而是用**混音實際算出的
門檻值**（`min_`、`max_` 的絕對數值）再削一次。理由是物理的：同一次類比
過載打在麥克風的輸出上，門檻是電路決定的絕對電壓，不是各訊號自己的統計量。
細節見 [ch6](device_chain.zh-TW.md)。

### 5. Fade（`wav_fade_in` / `wav_fade_out`）

以增益包絡實作淡入淡出。設 `f` 從 0 線性遞增到 1（長度為 `fade_len_s · sr`）：

| 形狀 | fade-in 增益 | fade-out 增益 |
|---|---|---|
| linear | `f` | `1 − f` |
| exponential | `2^(f−1) · f` | `2^(−f) · (1 − f)` |
| logarithmic | `log10(0.1 + f) + 1` | `log10(1.1 − f) + 1` |

包絡前後補 1，整條 clamp 到 [0, 1]。

三種形狀的差異在於能量的時間分佈：linear 在振幅上均勻；logarithmic 在
開頭上升快（在 dB 域接近線性，聽感上更「均勻」）；exponential 在開頭上升慢，
適合模擬緩慢的淡入。

### 6. 動態範圍壓縮器（`compressor_gain`）

模擬「這段錄音經過了壓縮器」——廣播、會議軟體、發佈鏈普遍使用。

#### 6.1 回傳增益曲線而非壓縮後訊號

函式回傳的是 `[1, T]` 的增益曲線，由呼叫端自行套用。這個介面設計是為了讓
同一條曲線能同時套在混音與目標上，理由見 §6.5。

#### 6.2 包絡偵測器

壓縮器需要先追蹤訊號的「當前音量」。實作使用非對稱一階遞迴（one-pole）
峰值追蹤：

```
a_att = 1 − exp(−1000 / (attack_ms  · fs))
a_rel = 1 − exp(−1000 / (release_ms · fs))

one_pole(x, a):  y[n] = a·x[n] + (1 − a)·y[n−1]

env[n] = max( one_pole(|x|, a_att), one_pole(|x|, a_rel) )
```

**時間常數的推導**。一階遞迴 `y[n] = a·x[n] + (1−a)·y[n−1]` 的階躍響應為
`y[n] = 1 − (1−a)^n`，其等效時間常數 τ 滿足 `(1−a)^{fs·τ} = e^{−1}`，
解得 `a = 1 − exp(−1/(fs·τ))`。把 τ 從秒換成毫秒即得上式的 `1000` 因子。

**為什麼取兩個極點的最大值**。標準壓縮器的偵測器是逐樣本判斷：訊號高於
當前包絡時用快速的 attack 係數，低於時用慢速的 release 係數。實作改用
「快極點與慢極點的逐點最大值」，行為在形狀上等價：

* 訊號起音（onset）時，快極點上升得比慢極點快，`max` 取到快極點——包絡以
  attack 速率上升。
* 訊號衰減時，慢極點下降得比快極點慢，`max` 取到慢極點——包絡以 release
  速率下降。

兩者不是同一條曲線（相關係數約 0.93，非 1.0）。選擇這個近似的原因是效能：
逐樣本切換的迴圈無法向量化，處理 6 秒音訊需要約 2.4 秒，在 dataloader 中
每個 epoch 會累積數百分鐘；`max(fast, slow)` 用兩次 `lfilter` 完成，約 12 ms。

偵測器的具體形式是壓縮器的設計自由度（真實硬體之間也各不相同），增強真正
需要的性質——起音快速反應、衰減不 pump——由測試斷言，而非靠複製某個特定
實作。

#### 6.3 增益計算

在 dB 域套用標準的 above-threshold 壓縮曲線：

```
E_dB[n] = 20 · log10(env[n])
over[n] = max(0, E_dB[n] − T)              # 超出門檻的量
g_dB[n] = −over[n] · (1 − 1/R)
g[n]    = 10^(g_dB[n]/20)
```

**壓縮比 R 的意義**。輸入超出門檻 `over` dB 時，輸出只超出 `over/R` dB，
因此需要的增益衰減為 `over − over/R = over·(1 − 1/R)`。極限行為：
`R = 1` 時 `g_dB = 0`（無壓縮）；`R → ∞` 時 `g_dB = −over`（限幅器，輸出
被完全壓在門檻上）。

#### 6.4 Makeup 正規化

```
g ← g / mean(g)
```

把增益曲線的平均值正規化為 1。**若不做這一步，壓縮與「整體調小聲」在資料上
不可區分**：壓縮器的平均增益必然小於 1，模型可以只學到「這一列比較小聲」
而不是「這一列的動態被壓縮了」。正規化之後，壓縮的唯一可觀測效果是包絡形狀
的改變。

注意正規化保證 `mean(g) = 1` 但不保證 `max(g) ≤ 1`——安靜段可能被推高到
增益大於 1，因此壓縮後的峰值有可能高於壓縮前。下游的 A/D 邊界
（[ch6](device_chain.zh-TW.md)）負責處理超過滿刻度的情況。

#### 6.5 為什麼壓縮器可以套在 target 上，而 waveshaper 不行

這是本章最重要的設計原理，決定了「錄音被壓縮」與「房間裡有台電視」必須用
兩種不同機制建模。

壓縮器的輸出是**時變增益**與訊號的逐點乘積。乘法對加法有分配律：

```
g[n] · (near[n] + far[n]) = g[n]·near[n] + g[n]·far[n]
```

所以把同一條 `g` 套在混音與目標上之後，目標仍然精確等於「壓縮後混音的近場
成分」。疊加性存活，訓練對的關係不變。

對比 `apply_media_coloring` 的 `|x|^p` waveshaper
（[ch5](spectral_channel.zh-TW.md) §6）：

```
(near + far)^p ≠ near^p + far^p
```

waveshaper 沒有逐源分解。它是「房間裡那台電視喇叭」的正確模型——電視的失真
發生在混音之前，作用在單一聲源上；但它是「整條錄音被壓縮」的錯誤模型，因為
目標若要跟隨，只能被獨立地失真成一個不再是混音成分的訊號。

#### 6.6 曲線由混音導出

`compressor_gain` 的輸入是**混音**。真實鏈中壓縮器接在麥克風之後，它看到的
就是混音。若改由目標導出曲線，得到的是一個沒有任何真實裝置會執行的效果
（壓縮器不可能只「看到」近場語者）。

這個 stage 存在的動機是：發佈與會議鏈普遍施加壓縮，而包絡結構是近/遠讀出的
成分之一（波峰因數的改變，§1），讓訓練分佈涵蓋「被壓縮過的世界」是資料層
的處理方式。動機的量測出處見 `CompressorAugmentation` docstring 與
`egs/voice_isolate/benchmarks/`。

## 工程面

### Config 對照

| Knob | Schema | 落點 |
|---|---|---|
| `augmentation_volume` | `VolumeAugmentation`（`perturbed_range`、`clipping_prob`、`clipping_range.min/max`） | `DeviceChain._volume`：擲一次決定走增益或削波分支 |
| `augmentation_compressor` | `CompressorAugmentation` | `DeviceChain._compressor` |
| `augmentation_noise.snr_range` 等 | `NoiseAugmentation` | `NoiseStage`（[ch7](scene_construction.zh-TW.md)） |
| `augmentation_speech.snr_range` | `SpeechAugmentation` | hard-SIR 抽樣（[ch7](scene_construction.zh-TW.md)） |

`CompressorAugmentation` 的 schema 約束：`ratio_range` 下界必須 ≥ 1.0
（1.0 代表無壓縮，小於 1 是擴張器，不是這個 stage 要模擬的東西）；
`attack_ms_range` 下界必須 > 0（0 會讓時間常數公式除零）。

### 順序

壓縮器在 device chain 中的位置是「前級增益之後、A/D 之前」，對應真實鏈的
擺位。它必須留在線性組內（[ch6](device_chain.zh-TW.md)），因為 target 要
跟隨。

`_volume` 每次開火只走增益或削波其中一支（以 `clipping_prob` 決定），不會
兩者都做。

### 陷阱

* 壓縮器曲線與訊號長度以 `min(gain, noisy, target)` 對齊後逐段相乘。SRC
  stage 可能讓長度相差 1–2 樣本，尾端未覆蓋的樣本保持原值，這是預期行為。
* `rand_gain_distortion` 目前只有 augmentor API（TSE 任務使用），NS 與
  voice isolation 的位準路徑是 device chain 的 `_volume` stage。兩者的
  增益分佈不同（前者 `N(0, 12dB²)`、後者 `U(perturbed_range)`），不要混用。
* §2.3(a) 的 RMS 語義：改動它會造成分佈斷代，修改前先確認願意付重訓對照組
  的成本。
