# 頻譜與通道效果

English version: [spectral_channel.md](spectral_channel.md)

麥克風有頻率響應、裝置有 rumble filter、傳輸路徑會經過重取樣器。本頁說明合成
流程使用的線性濾波器與重取樣器、speed 與 pitch 擾動、media coloring（一個刻意的
非線性），以及讓線性各級保持線性的 `apply_linear`。程式位於
`puresound/audio/dsp.py`、`puresound/audio/impulse_response.py` 與
`puresound/audio/augmentation.py`；呼叫它們的裝置鏈各級見
[裝置鏈](device_chain.zh-TW.md)。

## 1. Biquad

### 1.1 轉移函數

biquad 是二階 IIR 濾波器：

```
H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (a0 + a1·z⁻¹ + a2·z⁻²)
y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2] − a1·y[n−1] − a2·y[n−2]     (a0 = 1)
```

二階是能提供「一個共振或一段過渡帶」的最小階數：一對共軛極點決定頻率與 Q，
兩個零點塑造阻帶。更高階的響應由 biquad 串接而成。

### 1.2 RBJ 設計

`dsp.get_biquad_params(gain_dB, cutoff_freq, q_factor, sample_rate, filter_type)`
實作 RBJ Audio EQ Cookbook，回傳以 `a0` 正規化、長度 3 的 NumPy 陣列 `(b, a)`。
`filter_type` 為 `high_shelf`、`low_shelf`、`peaking`、`lpf`、`hpf`、`bpf`、
`notch` 之一。中間變數為

```
A  = 10^(gain_dB / 40)       振幅增益的平方根
w0 = 2π · fc / fs            正規化角頻率，rad/sample
α  = sin(w0) / (2·Q)         頻寬參數
```

`A` 除以 40 而非 20，是因為它在 shelf 與 peaking 公式中同時出現在分子與分母
（例如 `1 + α·A` 對 `1 + α/A`），響應增益因此是 `A²`。`lpf`、`hpf`、`bpf`、
`notch` 不使用 `gain_dB`。

`Q = f0 / Δf`（中心頻率除以 −3 dB 頻寬）。`Q = 1/√2 ≈ 0.707` 是 Butterworth 值，
通帶最平坦、無過衝；`Q = 0.5` 為臨界阻尼；`Q > 1` 會在 cutoff 附近形成峰。

三組代表性係數：

```
LPF:      b = [(1 − cos w0)/2,  1 − cos w0,  (1 − cos w0)/2]      a = [1 + α, −2·cos w0, 1 − α]
HPF:      b = [(1 + cos w0)/2, −(1 + cos w0), (1 + cos w0)/2]     a = [1 + α, −2·cos w0, 1 − α]
peaking:  b = [1 + α·A, −2·cos w0, 1 − α·A]                       a = [1 + α/A, −2·cos w0, 1 − α/A]
```

LPF 與 HPF 共用極點，差別只在零點的位置（Nyquist 與 DC）。peaking 濾波器中
`gain_dB → −gain_dB` 會交換分子與分母，因此衰減恰好是提升的反向。

cookbook 公式由類比原型經 bilinear transform 而來，`Ω = 2·tan(ω/2)`。公式在 `w0`
做了 pre-warp，cutoff 落在指定位置，但靠近 Nyquist 時頻寬與增益會偏離類比
設計。16 kHz 下的語音頻帶內，這個偏差可以忽略。

### 1.3 `ParametricEQ`

`dsp.ParametricEQ` 串接一個 low shelf、N 個 peaking 頻帶與一個 high shelf，
`H(z) = Π H_i(z)`，以 `scipy.signal.lfilter` 逐級套用。串接時各級的 dB 響應
相加，正是等化器應有的行為；並聯相加則會讓各頻帶的相位在交越處互相干涉。它是
固定、不可微分的工具，沒有任何合成 stage 使用它。可訓練的
`puresound.nnet.lobe.dsp.FrequencyEQLayer` 以同一個 `get_biquad_params` 的
shelf 與 peaking 設計初始化權重。`ParametricEQ.plot_eq` 只支援 16 kHz 與 32 kHz。

## 2. 隨機二階響應（換能器）

`impulse_response.rand_add_2nd_filter_response(wav, a=None, b=None)` 模擬每支
麥克風各有不同頻率響應，做法取自 *A Hybrid DSP/Deep Learning Approach to
Real-Time Full-Band Speech Enhancement*（Valin，RNNoise）：

```
r0..r3 ~ U(−3/8, 3/8)
a = [1, r0, r1]      極點
b = [1, r2, r3]      零點
```

它回傳 `(wav, a, b)`；傳入 `a` 與 `b` 即重用同一個響應，目標就是這樣跟隨混音。

**不需檢查的穩定性。** 二階分母 `1 + a1·z⁻¹ + a2·z⁻²` 的極點落在單位圓內的條件
是 `|a2| < 1` 且 `|a1| < 1 + a2`。當 `a1, a2 ∈ [−3/8, 3/8]`，第一條顯然成立，
第二條因 `1 + a2 ≥ 5/8 > 3/8` 而成立。每一次抽樣都穩定，不需要拒絕抽樣。

**增益範圍。** 在單位圓上 `|B| ≤ 1.75`、`|A| ≥ 0.25`，所以響應上界是
`20·log10(7) ≈ +16.9 dB`（在範圍的角落、於 DC 達到）。整個分佈的峰值中位數約
+3.5 dB。上尾正是 `apply_linear`（§7）需要逐步退讓迴圈的原因。

**`clamp=False`。** 濾波以 `torchaudio.functional.lfilter(..., clamp=False)`
執行。`lfilter` 預設把輸出夾在 [−1, 1]，在訊號很熱時等於一個 waveshaper，而且
只打到混音、打不到較安靜的目標。backend 有開關時，直接關掉既精確又免費。

## 3. High-pass（rumble filter）

`AudioEffectAugmentor.apply_hpf(wav, sr, cutoff_freq, q_factor)` 透過
`apply_linear` 執行 `torchaudio.functional.highpass_biquad`（RBJ high-pass），
因為 biquad 系列函式沒有暴露 `clamp`。裝置鏈從加權的離散清單抽 cutoff，因為
實際裝置只使用少數幾個設計值（例如 100、200、300 Hz），並抽
`Q ~ N(0.707, 0.1²)` 夾在 [0.3, 1.3]：接近 Butterworth，但從不完全等於。

## 4. 重取樣

### 4.1 模擬對象

「訊號在鏈上某處經過較低的取樣率」：codec 內部取樣率、藍牙、驅動層。這一級是
往返 `fs → src_sr → fs`。抗混疊濾波器移除 `src_sr/2` 以上的內容，升取樣無法
恢復，同時留下濾波器本身的痕跡：過渡帶形狀、通帶漣波、殘留鏡像。

### 4.2 兩個 backend

`dsp.wav_resampling(wav, origin_sr, target_sr, backend, torch_backend_params=None)`：

| `backend` | 濾波器 | 回傳 |
|---|---|---|
| `"sox"` | 固定、近乎透明的 windowed sinc：`torchaudio.functional.resample`，`sinc_interp_kaiser`、`lowpass_filter_width=64`、`rolloff≈0.9476`、`beta≈14.77`（"kaiser best" 設定）。確定性。這個名稱指它所取代的 sox `rate` 效果；實際上不呼叫 sox。 | `(wav, target_sr)` |
| `"torchaudio"` | 隨機化 windowed sinc：`lowpass_filter_width ∈ {6, 16, 32, 64, 128}`、`rolloff ~ U(0.8, 0.99)`、window ∈ {`sinc_interp_hann`, `sinc_interp_kaiser`}，除非 `torch_backend_params` 提供 `lp_width`、`rolloff`、`window` | `(wav, target_sr, params)` |

`origin_sr == target_sr` 時原樣回傳，tuple 形狀不變。

- `lowpass_filter_width` 是 sinc 截斷長度（以零交越數計）。核越短，過渡帶越寬、
  阻帶越弱、鏡像洩漏越多；這個範圍涵蓋從廉價到高品質的實作。
- `rolloff` 是抗混疊 cutoff 占 Nyquist 的比例：0.8 提早截止（混疊少、頻寬損失
  多），0.99 幾乎保留全頻帶。
- window 決定截斷時旁瓣與主瓣的取捨。

固定 backend 是中性的取樣率轉換：語料載入（`AudioIO.open(resample_to=...)`）、
噪音與 RIR 的重取樣都用它。隨機化 backend 是一種增強。裝置鏈在兩者之間 50/50
選擇，因為固定一個濾波器等於把單一實作的痕跡寫進訓練資料。

### 4.3 目標用同一個濾波器

隨機化 backend 回傳它抽到的三個參數，裝置鏈把它們傳回給升取樣那一段，以及目標
的兩段。這一對訊號必須經過**同一個濾波器**，而不只是相同的取樣率；否則混音與
目標帶著不同的頻寬限制與鏡像，兩者之差就不再是模型該移除的東西。

## 5. Speed 與 pitch

`AudioEffectAugmentor` 上的 `sox_*` 方法以各自重現的 sox 效果命名；沒有一個會
呼叫 sox。

- **`sox_speed_perturbed(wav, speed, sr)`** 把片段視為 `sr · speed` 取樣，再重
  取樣回 `sr`（`torchaudio.functional.resample`，預設濾波器）。時長縮放
  `1/speed`、音高縮放 `speed`，兩者同時改變：`speed = 1.1` 快 10 %、高
  `12·log2(1.1) ≈ 1.65` 個半音。這是語速與音高的真實共變，不是時間伸縮。
- **`sox_pitch_perturbed(wav, shift_ratio, sr)`** 在時長不變下把音高移動
  `shift_ratio` cents（`torchaudio.functional.pitch_shift`，每八度 1200 bins，
  phase vocoder）。沒有任何任務流程呼叫它。
- **`sox_volume_perturbed(wav, vol_ratio, sr)`** 是單純相乘，因此不會削波。

noise suppression 與 voice isolation（`ContinuousSpeedAugmentation`）從
`speed_range` 上的格點 `arange(lo, hi + 0.025, 0.05)` 抽 speed；多加的半步讓 `hi`
留在格點上，單純的 `arange(lo, hi, 0.05)` 會把它排除。同一個倍率套用在混音與
目標上，位置在混音之後、whole-mix RIR 之前，並記錄在 row plan 的
`speed_factor`，讓逐幀標籤能對應到新的時間格。凍結的 target speaker extraction
任務使用單純的 `arange`。speaker embedding 使用 `DiscreteSpeedAugmentation`：
一個 `speed_change` 清單加上 `treat_as_new_speaker`。

## 6. Media coloring

`AudioEffectAugmentor.apply_media_coloring(wav, sr, hp_cutoff, lp_cutoff,
compress_power)` 模擬房間裡經電視或喇叭播放的語音。它套用在被標為 media 的
干擾者上，且在其 RIR 之前，因為是裝置先播放、再由房間傳遞
（[場景建構](scene_construction.zh-TW.md)）。

```
1. y = LP(HP(x))                          HP 與 LP biquad，Q = 0.707，經 apply_linear
2. y = sign(y) · (|y| / P)^p · P,  P = max|y|      可選，p ∈ (0, 1)
3. y ← y · rms(x) / rms(y)
```

- **頻寬限制。** 小喇叭的低頻受單體尺寸與箱體容積限制，高頻受振膜質量與分音
  設計限制。`hp_cutoff_range` 與 `lp_cutoff_range` 涵蓋電視與筆電喇叭。
- **冪次 waveshaping。** 在峰值正規化的訊號上，`p < 1` 抬高小值
  （`0.1^0.7 ≈ 0.2`）並保持峰值為 1，像廣播內容一樣壓縮動態範圍。這裡用靜態
  曲線就夠，因為它塑造的是干擾源，不是任何被評分的東西。
- **RMS 還原。** 沒有它，coloring 也會變成位準改變，被下游的 SIR 縮放吸收，
  把「有沒有 coloring」與「SIR 多少」糾纏在一起。還原 RMS 把效果限定在頻譜與
  動態。
- **為什麼濾波器要包 `apply_linear`。** 不包的話 biquad 會在滿刻度削波，RMS
  還原再把損傷放大回輸入位準，把它藏起來。

waveshaper 沒有逐源分解：`(near + far)^p ≠ near^p + far^p`。它只能在混音前作用
於單一來源。「整段錄音被壓縮過」需要的是時變增益，也就是
[位準與動態](level_dynamics.zh-TW.md) 中的 compressor，因為增益可以分配到和的
每一項。

## 7. `apply_linear`

### 7.1 問題

每個線性 stage 都依賴同一個性質：以相同參數套用在混音與目標上後，混音仍等於
各來源在抽到的 SIR 下之和。backend 的預設值破壞了它。
`torchaudio.functional.lfilter` 除非以 `clamp=False` 呼叫，否則會把輸出夾在
[−1, 1]；torchaudio 的每個 `*_biquad` 函式都建立在 `lfilter` 上，且沒有暴露這個
旗標。訊號很熱時，一個模擬線性裝置的 stage 就會飽和。由於混音是這一對中較大聲
的一方，被壓扁的是它，目標則毫髮無傷，這一對就不再有 recipe 要求的位準關係。

### 7.2 方法

就用線性的定義：對線性 `H` 與任意 `a > 0`，`H(x) = H(a·x) / a`。

```
peak  = max|x|
scale = headroom / peak                  headroom = 0.5
out   = fn(x · scale) / scale
```

`dsp.apply_linear(fn, wav, *, headroom=0.5, max_escalations=8)` 在輸入的原位準
回傳算子的真實輸出。

### 7.3 Headroom 與逐步退讓

`headroom = 0.5` 為算子本身的增益留下 `20·log10(2) ≈ 6 dB`，涵蓋本套件大多數
濾波器。若輸出仍觸及滿刻度（`|out| ≥ 1 − 1e-6`；飽和的 backend 會把樣本釘在
恰好 ±1，真實音訊只會偶然落在那裡），就把 scale 再除以 8（18 dB）重試，最多
`max_escalations` 次；之後拋出 `RuntimeError`，而不是回傳一個悄悄錯掉的結果。
數位靜音或非有限的峰值會不經縮放直接交給 `fn`。

### 7.4 限制

- **`fn` 不得消耗隨機性。** 退讓時會再次呼叫它，這會平移 RNG stream、破壞
  seeded 重現（[工程契約](engineering_contract.zh-TW.md)）。隨機參數請在外面抽好，
  以 closure 傳入。
- **backend 有開關時優先用開關。** `lfilter(clamp=False)` 精確且零成本；隨機 IIR
  與 compressor 的包絡偵測器都這樣做。`apply_linear` 用於沒有開關的 backend：
  `apply_hpf` 與 `apply_media_coloring` 的兩個濾波器。

## 設定

| Knob | Schema | 作用位置 |
|---|---|---|
| `augmentation_src` | `SourceRateAugmentation`（`src_range` 與 `prob_each` 等長） | `DeviceChain._sample_rate_conversion` |
| `augmentation_ir_response` | `SimpleProbAugmentation` | `DeviceChain._second_order_iir` |
| `augmentation_hpf` | `HighPassAugmentation`（`cutoff` 與 `prob_each` 等長） | `DeviceChain._high_pass` |
| `augmentation_speed` | NS、VI、TSE 用 `ContinuousSpeedAugmentation`（`speed_range`）；speaker embedding 用 `DiscreteSpeedAugmentation`（`speed_change`） | `ns.py`，混音後成對套用 |
| `augmentation_speech.media_voice` | `MediaVoiceConfig`（`prob`、`hp_cutoff_range`、`lp_cutoff_range`、`compress_power_range`） | 每個合成干擾者，在其 RIR 之前 |

```yaml
augmentation_speech:
  media_voice:
    used: True
    prob: 0.30                    # 每個干擾者
    hp_cutoff_range: [200, 400]   # Hz
    lp_cutoff_range: [3500, 7000] # Hz
    compress_power_range: [0.6, 0.9]
```

## 附註

- 呼叫 torchaudio backend 時若沒給 `torch_backend_params`，它會自己抽一個濾波器。
  需要第二個訊號跟隨的呼叫端，必須把回傳的參數傳回去，裝置鏈就是這樣做。
- 隨機 IIR 只記錄 `iir_applied`，不記錄係數；若要依響應形狀分桶，必須先擴充
  provenance scalar。
