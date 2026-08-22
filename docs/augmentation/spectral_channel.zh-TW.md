# 頻譜與通道濾波

English version: [`spectral_channel.md`](spectral_channel.md)

麥克風有頻率響應、裝置有 rumble filter、傳輸鏈經過重取樣。本章整理 pipeline
中所有線性濾波手法的設計公式與推導、變速、一個刻意的非線性（media 上色），
以及讓所有線性手法真正保持線性的 `apply_linear` 機制。

## 演算法面

### 1. Biquad：二階 IIR 濾波器

#### 1.1 轉移函數與差分方程

二階 IIR（雙二次，biquad）是所有等化與濾波的基本單元：

```
H(z) = (b0 + b1·z⁻¹ + b2·z⁻²) / (a0 + a1·z⁻¹ + a2·z⁻²)
```

正規化 `a0 = 1` 後對應的差分方程：

```
y[n] = b0·x[n] + b1·x[n−1] + b2·x[n−2] − a1·y[n−1] − a2·y[n−2]
```

二階是最小能同時提供「一個共振峰或一個過渡帶」的階數：兩個極點可構成一對
共軛複數極點，決定共振頻率與品質因數；兩個零點決定阻帶行為。更高階的響應
（例如 `ParametricEQ`）用多個 biquad 串聯構成。

#### 1.2 RBJ 設計公式

`get_biquad_params` 使用 RBJ Audio EQ Cookbook 的參數化，把「工程師想要的
參數」（增益、截止頻率、Q）轉換成係數。三個中介變數：

```
A  = 10^(gain_dB / 40)        # 振幅增益的平方根
w0 = 2π · fc / fs             # 正規化角頻率（rad/sample）
α  = sin(w0) / (2·Q)          # 頻寬參數
```

**`A` 為何除以 40 而非 20**。shelving 與 peaking 濾波器的係數式中，`A` 以
乘積形式出現在分子與分母（例如 peaking 的 `1 + α·A` 對 `1 + α/A`），最終
響應的增益是 `A²`。因此要達到 `gain_dB` 的響應，`A` 必須是振幅比的平方根：
`A = 10^(gain_dB/20 / 2) = 10^(gain_dB/40)`。

**`w0` 的意義**。數位頻率以每樣本的弧度表示，`fc / fs` 是正規化頻率
（Nyquist 對應 0.5），乘 `2π` 得角頻率。

**`α` 與 Q 的關係**。Q（品質因數）定義為共振頻率除以 −3 dB 頻寬：
`Q = f0 / Δf`。Q 越大頻寬越窄、共振越尖。`α = sin(w0)/(2Q)` 是雙線性轉換
後 Q 在數位域的等效表述；`sin(w0)` 因子來自 s 域到 z 域的頻率翹曲。

常用取值：`Q = 0.707 = 1/√2` 對應 Butterworth 響應（通帶最平坦、無過衝），
這是 HPF/LPF 的預設值。`Q = 0.5` 為臨界阻尼（最快無振鈴）、`Q > 1` 在截止
頻率附近產生峰起。

#### 1.3 七種濾波器類型的係數

實作支援 `high_shelf`、`low_shelf`、`peaking`、`lpf`、`hpf`、`bpf`、`notch`。
代表性的三組：

```
LPF:       b = [(1 − cos w0)/2,  1 − cos w0,  (1 − cos w0)/2]
           a = [1 + α,  −2·cos w0,  1 − α]

HPF:       b = [(1 + cos w0)/2, −(1 + cos w0), (1 + cos w0)/2]
           a = [1 + α,  −2·cos w0,  1 − α]

peaking:   b = [1 + α·A,  −2·cos w0,  1 − α·A]
           a = [1 + α/A,  −2·cos w0,  1 − α/A]
```

觀察 LPF 與 HPF 的結構：兩者的分母相同（極點位置只由 `fc` 與 `Q` 決定），
差別只在分子的零點位置——LPF 把零點放在 Nyquist（`z = −1`，`b` 係數和為
`(1−cos)·2 = ` 在 `z = −1` 處為零），HPF 放在 DC（`z = 1`）。

peaking 的對稱結構也值得注意：分子的 `α·A` 對分母的 `α/A`，兩者互為倒數
關係，因此 `gain_dB → −gain_dB` 恰好對應分子分母互換，響應精確反轉。

所有係數最後除以 `a0` 完成正規化。

#### 1.4 假設與失效條件

RBJ 公式由類比原型經雙線性轉換（bilinear transform）導出。雙線性轉換會造成
頻率翹曲：類比頻率 `Ω` 與數位頻率 `ω` 的關係為 `Ω = 2·tan(ω/2)`。RBJ 公式
已在 `w0` 處做了 pre-warp（讓截止頻率精確落在指定位置），但**頻寬與增益在
接近 Nyquist 時仍會偏離**設計值。

實務影響：16 kHz 取樣時，`fc` 在 6 kHz 以上的 shelving 濾波器實際響應會與
理論曲線有可見偏差。語音頻段（80 Hz – 4 kHz）內可忽略。

#### 1.5 ParametricEQ：串聯結構

`ParametricEQ` 把 low shelf + N 個 peaking + high shelf 串聯：

```
H(z) = Π_i H_i(z)
```

串聯的意義是 dB 響應相加（`20·log10|Π H_i| = Σ 20·log10|H_i|`），這正是
等化器的直覺行為——每一段獨立調整、效果疊加。實作逐級呼叫
`scipy.signal.lfilter`（Direct Form I）。

串聯而非並聯的理由：並聯（相加）會讓各段的相位互相干涉，兩段同時提升時
可能在交界處出現凹陷；串聯只是相位累加，振幅響應可預測。

### 2. 隨機二階 IIR：換能器響應

#### 2.1 演算法

模擬「每一支麥克風的頻率響應都不一樣」（出處：A Hybrid DSP/Deep Learning
Approach to Real-Time Full-Band Speech Enhancement，RNNoise 系列的資料增強
做法）：

```
r0..r3 ~ U(−3/8, 3/8)
a = [1, r0, r1]        # 分母（極點）
b = [1, r2, r3]        # 分子（零點）
```

分子與分母都是隨機的二階多項式，得到一條隨機的、溫和的頻率響應曲線。

#### 2.2 為什麼 ±3/8 這個範圍能保證穩定

IIR 濾波器穩定的條件是所有極點位於單位圓內。對二階分母
`1 + a1·z⁻¹ + a2·z⁻²`，Schur–Cohn 穩定判據為：

```
|a2| < 1
|a1| < 1 + a2
```

代入 `a1, a2 ∈ [−3/8, 3/8]`：

* 第一條：`|a2| ≤ 3/8 < 1` 恆成立。
* 第二條：`|a1| ≤ 3/8`，而 `1 + a2 ≥ 1 − 3/8 = 5/8 > 3/8` 恆成立。

兩條件在整個抽樣範圍內恆滿足，因此**不需要任何拒絕取樣或穩定性檢查**——
這是選擇 3/8 這個上界的原因。

#### 2.3 增益範圍與 apply_linear 的關係

隨機極點靠近單位圓時會產生峰起。實測這組參數的響應峰值中位數約 +3.5 dB、
最壞情況可達 +16 dB。這個長尾是 `apply_linear`（§7）需要逃逸重試迴圈的
直接原因：headroom 只留 6 dB，遇到 +16 dB 的響應仍會撞到滿刻度。

#### 2.4 clamp=False 的必要性

實作以 `lfilter(..., clamp=False)` 套用。頻率響應是線性算子，而
`torchaudio.functional.lfilter` 預設把輸出硬夾在 [−1, 1]。在熱訊號上，這個
硬夾是 recipe 沒有要求的 waveshaper，而且它只打得到位準較高的混音、打不到
用來計分的較安靜目標——SIR 關係因此被破壞。backend 提供了關閉開關就直接
關閉，這比 §7 的縮放包裝精確且零成本。

### 3. 高通濾波（rumble filter）

模擬裝置的低頻截除（電源哼聲、風噪、handling noise）。實作使用
`torchaudio.functional.highpass_biquad`（RBJ HPF），參數：

* cutoff 從離散清單加權抽樣——真實裝置的 HPF 截止頻率是設計選擇（常見
  80/120/150 Hz），離散清單比連續分佈更貼近實際。
* `Q ~ N(0.707, 0.1)` 夾在 [0.3, 1.3]。以 Butterworth 為中心、允許小幅偏離
  ——真實裝置的濾波器不會精確是 Butterworth。

該 backend 沒有 clamp 開關，因此包在 `apply_linear` 內執行（§7）。

### 4. 重取樣

#### 4.1 目的與原理

模擬「訊號在傳輸鏈上經過了不同的取樣率」（codec 內部重取樣、藍牙、驅動
層）。做法是 down-then-up：

```
fs → src_sr → fs
```

淨效果是**頻寬受限**。降取樣時抗鋯疊濾波器把 `src_sr/2` 以上的內容濾除，
升取樣回來也無法恢復。除了頻寬損失，還留下該重取樣器抗鋯疊濾波器的特徵
（過渡帶形狀、通帶漣波、殘留鏡像）。

#### 4.2 兩個 backend

**sox `rate`**：固定的高品質多相（polyphase）濾波器實作。

**torchaudio windowed-sinc**：三個參數可控，且刻意隨機化：

```
lowpass_filter_width ∈ {6, 16, 32, 64, 128}
rolloff ∈ U(0.8, 0.99)
window ∈ {sinc_interp_hann, sinc_interp_kaiser}
```

參數的物理意義：

* `lowpass_filter_width` 是 sinc 核的截斷長度（以零交越點計）。理想的
  重取樣需要無限長的 sinc；截斷越短，過渡帶越寬、阻帶衰減越差、鏡像洩漏
  越多。從 6 到 128 涵蓋了「廉價實作」到「高品質實作」的整個範圍。
* `rolloff` 是抗鋯疊濾波器截止頻率相對 Nyquist 的比例。0.8 代表提早截止
  （保守，鋯疊少但頻寬損失多）、0.99 代表貼著 Nyquist（頻寬保留完整但
  可能有殘留鋯疊）。
* window 決定截斷 sinc 的窗函數。Hann 的旁瓣衰減較快但主瓣較寬；Kaiser
  可調整兩者的取捨。

**為什麼隨機化而非固定**。真實世界的重取樣器實作各不相同。固定一組參數等於
把單一廠商的濾波器特徵烙進訓練分佈，模型可能學到針對該特徵的補償。隨機化
讓 artifact 分佈變寬，模型被迫學習對重取樣痕跡不敏感。

device chain 中兩個 backend 以 50/50 擲選，同樣是為了拉寬分佈。

#### 4.3 target 的參數重用

torchaudio 路徑會回傳實際抽到的三個參數，device chain 用它們對 target 執行
同一次重取樣。這比「用同一組取樣率」更嚴格——必須是**同一顆濾波器**，否則
混音與目標各自帶著不同的頻寬損失與鏡像痕跡，兩者的差不再是「模型該去除的
東西」。

### 5. 變速與音高

三個 sox 效果，都經 `apply_linear` 執行（sox 內部走定點格式，會在滿刻度
飽和）：

* **`speed`**：改變播放速率再重取樣回原 `fs`。時間軸與音高**共同變化**
  （不是 time stretch）——`speed = 1.1` 讓語音快 10% 且音高升高約 1.6 半音。
  這模擬的是語速與音高一起變的真實變異，也是最廉價的語音增強。
* **`pitch`**：只改音高（以 cents 為單位），保持時長。用於語者相關任務
  （TSE/SV）的語者變異增強。
* **`vol`**：純增益。

NS 管線的變速從離散格點抽樣：`arange(lo, hi + 0.025, 0.05)`。上界刻意加上
半個步長——裸 `arange(lo, hi, step)` 不含上界，會讓 `speed_range: [0.9, 1.1]`
實際只抽到 0.9–1.05，加速那一半消失。混音與目標成對套用同一個 speed 值。

### 6. Media 上色：喇叭播放的頻譜與動態模型

模擬場景中「電視或喇叭播放出來的聲音」，套用在被標記為 media 的干擾者上
（[ch7](scene_construction.zh-TW.md)）。三個步驟：

```
1. HP biquad (Q=0.707) → LP biquad (Q=0.707)          # 帶限
2. y = sign(x) · (|x| / P)^p · P,   P = max|x|         # 冪次波形整形
3. y ← y · rms_in / rms_out                            # RMS 還原
```

#### 6.1 帶限的物理依據

小型喇叭有明確的通帶：低頻受限於振膜尺寸與箱體容積（低頻延伸不足）、高頻
受限於振膜質量與分頻設計。config 的 `hp_cutoff_range` 與 `lp_cutoff_range`
涵蓋典型的電視/筆電喇叭範圍。帶限是線性操作，因此包在 `apply_linear` 內。

#### 6.2 冪次波形整形的作用

`|x|^p`（`p ∈ (0, 1]`）是一個**無記憶**的靜態非線性。對峰值正規化後的訊號
（`|x/P| ≤ 1`），`p < 1` 會把小值抬升（例如 `0.1^0.7 ≈ 0.2`）而讓最大值
保持在 1，效果是壓縮動態範圍。

這是廣播鏈壓縮的零階近似：廣播與串流內容普遍經過大量壓縮，播放出來的訊號
動態範圍遠小於原始錄音。用靜態非線性而非真正的壓縮器，是因為這裡不需要
時間常數的精確性——它只是干擾源的一個特徵，不是被計分的對象。

#### 6.3 RMS 還原的必要性

帶限與冪次整形都會改變訊號的 RMS。若不還原，上色本身就變成了一次位準改變，
下游的 SIR 縮放會把它一起吸收——結果是「有沒有上色」與「SIR 多少」兩個
變數糾纏在一起。還原 RMS 讓上色只影響頻譜與動態，不影響位準。

#### 6.4 為什麼帶限必須包 apply_linear

若不包：biquad 在滿刻度削頂，接著 §6.3 的 RMS 還原會把削頂造成的傷害按比例
放大回輸入位準——一個披著濾波器名字的 waveshaper。RMS 還原的存在讓這個
錯誤更難察覺（位準看起來正常），這是必須明確包起來的原因。

#### 6.5 與壓縮器的分工

冪次整形是逐樣本的靜態非線性，**沒有逐源分解**：
`(near + far)^p ≠ near^p + far^p`。因此它只能用在混音之前、作用於單一聲源
（「房間裡那台裝置」），不能用來模擬「整條錄音被壓縮」。後者必須用時變
增益（[ch4](level_dynamics.zh-TW.md) §6），因為只有乘法對加法有分配律。

### 7. apply_linear：讓線性後端保持線性

#### 7.1 問題

本章所有線性手法都建立在一個前提上：線性算子作用在混音與目標上，混音仍然
等於各聲源在指定 SIR 下之和。這個前提被 backend 的實作細節破壞：

* `torchaudio.functional.lfilter` 預設把輸出夾在 [−1, 1]。
* 所有 `biquad` 函式建立在 `lfilter` 之上，且不轉發 `clamp` 參數。
* 所有 sox 效果經過定點取樣格式往返，在滿刻度飽和。

因此當輸入訊號較熱時，一個模擬**線性**現象的 stage（麥克風響應、rumble
filter、前級增益、重取樣）會默默停止線性。

**為什麼這比失真本身更嚴重**。這些 stage 以相同參數同時套用在混音與目標上，
目的正是維持兩者的位準關係。一個固定在絕對位準的天花板破壞了這個對稱性：
混音是兩者中較響的，它被壓縮；目標較安靜，毫髮無傷地通過。結果是混音不再
等於各成分在 recipe 指定的 SIR 下之和。

這不是理論風險，在現役 recipe 上會被頻繁觸發，觸發率統計見 `apply_linear`
的 docstring。

#### 7.2 解法

利用線性的定義本身。對線性算子 `H` 與任意純量 `a > 0`：

```
H(x) = H(a·x) / a
```

因此可以先縮放到 backend 的合法範圍、套用、再縮放回來：

```
peak  = max|x|
scale = headroom / peak          # headroom = 0.5
out   = fn(x · scale) / scale
```

結果是該算子的真實輸出，而飽和從未觸發。

#### 7.3 headroom 的選擇與逃逸迴圈

`headroom = 0.5` 留給算子本身 6 dB 的增益空間
（`20·log10(1/0.5) = 6.02 dB`）。這涵蓋了本套件的多數濾波器——隨機二階響應
的峰值中位數約 +3.5 dB（§2.3）。

但該響應的最壞情況可達 +16 dB，會超出 6 dB 的空間。因此實作檢查輸出是否
仍碰到滿刻度（容許 `1e-6` 的判定誤差），若碰到就把 scale 再除以 8（等於
多留 18 dB）重試，最多 8 次。全部失敗則拋出例外——寧可失敗也不回傳一個
安靜地錯掉的數字。

判定用 `>= 1 − 1e-6` 而非精確等於 1：飽和的 backend 會留下精確釘在 ±1 的
樣本，真實音訊只會偶然落在那裡，誤判的代價僅是多一次不必要的重試。

#### 7.4 對 fn 的約束

**`fn` 不得消耗隨機性**。逃逸重試會再次呼叫 `fn`，若它抽樣就會平移全域 RNG
流，破壞 seeded-item 的位元級重現契約（[ch8](engineering_contract.zh-TW.md)）。
目前所有呼叫點都是純濾波或純增益。

需要隨機參數的情況（例如 §4.2 的重取樣參數）必須在 `apply_linear` **外面**
抽好，以閉包傳入。

#### 7.5 什麼時候不該用它

backend 有 clamp 開關時直接關閉，不要用這個包裝：`lfilter(clamp=False)` 是
精確的，而縮放包裝需要至少一次額外的峰值掃描與兩次乘法。

## 工程面

### Config 對照

| Knob | Schema | 落點 |
|---|---|---|
| `augmentation_src` | `SourceRateAugmentation`（`src_range` 與 `prob_each` 等長） | `DeviceChain._sample_rate_conversion` |
| `augmentation_ir_response` | `SimpleProbAugmentation` | `DeviceChain._second_order_iir` |
| `augmentation_hpf` | `HighPassAugmentation`（`cutoff` 與 `prob_each` 等長） | `DeviceChain._high_pass` |
| `augmentation_speed` | `ContinuousSpeedAugmentation`（增強任務）/ `DiscreteSpeedAugmentation`（語者任務） | `ns.py`，混音後成對套用 |
| `augmentation_speech.media_voice` | `MediaVoiceConfig` | 干擾者取樣時上色，在其 RIR 之前 |

### 順序與互斥

* device chain 內固定 SRC → IIR → HPF（[ch6](device_chain.zh-TW.md)）。
* 變速在混音之後、whole-mix RIR 之前（`ns.py`）。
* media 上色在該干擾者的 RIR **之前**：先有裝置放音，再經房間傳播，符合
  物理順序。
* 隨機二階 IIR 的係數由混音那次呼叫抽出，target 使用同一組——係數共享是
  這個 stage 的契約。

### 陷阱

* `wav_resampling` 的 torchaudio 路徑若未傳入 `torch_backend_params`，會
  自行抽樣三個參數。需要決定性或需要 target 跟隨的路徑必須把回傳的參數
  往下傳（device chain 已正確處理；新增呼叫點時要留意）。
* `ParametricEQ.plot_eq` 的 nfft 只支援 16 k 與 32 k（寫死），是除錯工具，
  不屬於 pipeline。
* 二階 IIR 是 device chain 中少數不記錄參數的 stage（只記
  `iir_applied`），因此無法在 eval 中按響應形狀分桶。若需要這個分析維度，
  要先擴充 provenance scalars。
