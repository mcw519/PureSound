# 距離與時間線索

English version: [`distance_cues.md`](distance_cues.md)

近場/遠場的判斷是 voice isolation 的核心決策。本章逐一推導混音中可用的距離
線索的物理來源，說明 pipeline 保留了哪些、移除了哪些，並解析三個直接操縱
這些線索的手法：DRR contrast、direct-arrival smear、`distance_level` 混音。

## 演算法面

### 1. 距離線索的物理來源

人耳與模型判斷聲源距離時可用的訊息有六類。以下逐項推導，並標註它在
pipeline 中的狀態。

#### 1.1 位準（反平方定律）

自由場中點聲源的聲壓隨距離反比衰減：

```
p(d) ∝ 1/d      →      L(d) = L_ref − 20·log10(d / d_ref)
```

每倍距離下降 6 dB。0.5 m 對 3 m 的位準差為 `20·log10(3/0.5) ≈ 15.6 dB`。

**pipeline 狀態：已移除。** 兩個步驟疊加造成：語料載入時 `AudioIO.open` 以
`target_lvl` 做 RMS rescale，把所有聲源的位準拉平；RIR 卷積時逐通道峰值
正規化（[ch2](room_acoustics.zh-TW.md) §3.1）又抹掉通道間的距離增益比。

這個線索是所有距離線索中最直接、也最容易被真實世界的其他因素污染的（說話
音量、口部指向、麥克風增益）。移除它的取捨是：模型不能靠「小聲就是遠」這條
捷徑，必須學習通道特徵；代價是遺失了一個真實存在的線索。`distance_level`
混音模式（§5）是把它明確加回來的手段。

#### 1.2 聲源之間的到達時差

不同距離的聲源，其語音起始點有時間差 `Δt = Δd / c`。3 m 與 0.5 m 的差約
7.3 ms。

**pipeline 狀態：已移除。** 卷積後各通道按自己的 `argmax|h|` 對齊
（ch2 §3.2），所有聲源的語音起始點被拉到同一時刻。

#### 1.3 直達與殘響的能量比（DRR）

直達聲能量隨 `1/d²` 衰減，而擴散場理論指出殘響能量密度在房間內近似均勻，
與距離無關。因此：

```
DRR_dB(d) ≈ −20·log10(d) + C
```

每倍距離下降約 6 dB，`C` 由房間吸收與體積決定。`DRR = 0` 處為臨界距離，
一般會議室約 1–2 m。

**pipeline 狀態：保留**，且可由 DRR contrast（§3）明確放大。這是移除位準
線索後最主要的距離線索。

#### 1.4 直達窗內的精細時間結構

直達聲到達後的最初幾毫秒內，會陸續到達經地面、桌面、鄰近牆面一次反射的
成分。這些路徑差隨聲源距離變化：近場聲源的一次反射路徑差大（相對延遲長、
振幅弱），遠場聲源的一次反射路徑差小（緊貼直達聲、振幅相對強）。

因此直達窗內的「脈衝叢集形狀」——脈衝之間的間隔與相對振幅——攜帶距離資訊。
這是一個時間域的細結構線索，不等同於窗內的頻譜形狀。

**pipeline 狀態：保留**，direct smear（§4）可刻意破壞它。

#### 1.5 衰減尾形（RT60、EDC）

殘響尾巴的衰減速率主要由房間決定，不由距離決定。它提供的是「房間有多響」
而非「聲源有多遠」，但與 DRR 聯合時可以幫助校準——同樣的 DRR 在不同 RT60
的房間對應不同的距離。

**pipeline 狀態：保留。**

#### 1.6 頻譜傾斜

兩個機制讓遠場聲源的高頻相對較少：

* **空氣吸收**：吸收係數隨頻率上升（在室內距離下，主要影響 4 kHz 以上）。
* **牆面吸收的頻率相依性**：多數建材高頻吸收較強，因此經多次反射的成分
  高頻衰減更多；遠場訊號中反射成分占比高，整體頻譜較暗。

**pipeline 狀態：保留**，但合成 RIR 的頻譜傾斜精確度受 ISM 假設限制
（ch2 §7：反射係數與頻率無關）。

#### 1.7 線索清單總表

| 線索 | 物理來源 | pipeline 狀態 |
|---|---|---|
| 位準 | 反平方定律 `p ∝ 1/d` | 移除（載入 RMS + RIR 峰值正規化）；`distance_level` 可加回 |
| 聲源間到達時差 | `Δd / c` | 移除（逐通道 argmax 對齊） |
| DRR | 直達 `1/d²` vs 均勻殘響場 | 保留；DRR contrast 可放大 |
| 直達窗精細時間結構 | 一次反射的路徑差隨距離變化 | 保留；direct smear 可破壞 |
| 衰減尾形 | 房間吸收與體積 | 保留 |
| 頻譜傾斜 | 空氣吸收 + 牆面高頻吸收 | 保留（受 ISM 假設限制） |

哪些線索被模型實際讀取、各佔多少權重，屬於實驗問題，不在本文件範圍。

### 2. 為什麼要操縱線索

給定 §1 的清單，資料層有兩種方向的操作，pipeline 兩者都提供：

* **放大線索**（DRR contrast、`distance_level`）：讓近/遠在訓練分佈中的區隔
  比自然分佈更明顯，降低學習難度，用於建立基本的近/遠選擇性。
* **移除線索**（direct smear）：讓模型不能只依賴單一線索，強迫它尋找替代
  依據，用於提升對線索被遮蔽情境的韌性。

兩者不是對立的——它們作用在不同線索上，可以同時使用。

### 3. DRR contrast

#### 3.1 演算法

對一條剛生成的 RIR，把直達窗之後的整段尾巴乘上一個固定增益：

```
tail_start = peak + round(w · fs / 1000),    w = direct_window_ms（預設 2.5 ms）
h[n] ← h[n] · 10^(−s/20)    for all n ≥ tail_start
```

`s` 是要施加的 DRR 位移量（dB）。

#### 3.2 為什麼位移量恰好是 s

設原始的直達能量為 `E_d`、尾巴能量為 `E_r`。尾巴振幅乘 `10^(−s/20)` 後，
其能量變為 `E_r · 10^(−s/10)`。新的 DRR：

```
DRR' = 10·log10( E_d / (E_r · 10^(−s/10)) )
     = 10·log10(E_d / E_r) + 10·log10(10^(s/10))
     = DRR + s
```

位移量精確等於 `s`，不需要迭代或校正。這個乾淨的結果依賴一個條件：**尾巴的
分界必須與 DRR 量測的分界一致**。實作因此把 `direct_window_ms` 的預設值
設為 2.5 ms，與 `compute_drr_db` 的預設窗長相同（ch2 §5）。若兩者不一致，
被縮放的區間與被量測的區間不同，位移量就不再等於 `s`。

#### 3.3 為什麼縮放尾巴而不是縮放直達聲

兩種做法都能改變 DRR，但下游還有一次峰值正規化（ch2 §3.1）。縮放直達聲會
改變 `max|h|`，正規化後等於同時改變了尾巴的絕對位準；縮放尾巴則保持峰值
不變，正規化係數不變。實際存活到模型輸入的只有 DRR 比值，因此選擇不動峰值
的做法，讓操作的效果單一且可預測。

#### 3.4 兩種模式

**random 模式**：依聲源角色決定位移方向，並以機率 `prob` 決定是否施加。

```
foreground:              s = +U(near_boost_db)
interferer / media / echo: s = −U(far_cut_db)
其他角色:                 不處理
```

效果是把近場的 DRR 往上推、遠場往下拉，擴大兩者的間距。

**deterministic 模式**：位移量是該通道實現距離的固定函數，不看角色、不擲
機率、不消耗 RNG。

```
s = extra_db_per_decade · log10(d / pivot_m)
```

推導其效果：由 §1.3，自然的 DRR-距離關係是 `DRR ≈ −20·log10(d) + C`。加上
位移後：

```
DRR'(d) = −20·log10(d) + C + extra · log10(d / pivot)
        = −(20 − extra)·log10(d) + C'
```

`extra` 為負值時，梯度的絕對值 `|20 − extra|` 變大——**整個池子的
distance→DRR 梯度變陡**，近/遠的 DRR 間距在所有距離上一致地放大。`pivot_m`
是不受影響的樞紐距離（`d = pivot` 時 `s = 0`）。

#### 3.5 兩種模式的設計差異

deterministic 模式存在的理由記在 `init_drr_contrast` 的註解中：random 模式
的逐次抽樣會讓同一個距離在不同列上得到不同的 DRR 位移，也就是把 DRR 與距離
**解耦**。池子因此失去單一的 distance→DRR 映射，同一個 DRR 值可以對應一個
距離區間而非單一距離。

如果目標是「讓模型學會一條清楚的 distance→DRR 對應關係」，位移量就必須是
距離的函數而非獨立隨機變數。這是 deterministic 模式的設計出發點，也是它
刻意不消耗 RNG 的原因——同一個通道在任何情況下都得到同一個位移。

#### 3.6 套用時機

兩種模式都在 RIR 進入 cache **之前**套用，緊接在生成之後。理由見 ch2 §6：
目標訊號透過 `rir_id` 重取同一條 RIR，若在 cache 之後才套用，混音與目標會
基於兩條不同的脈衝響應，破壞「目標是混音的近場成分」的前提。

metadata 會記錄 `drr_contrast_shift_db` 與重新計算的 `drr_db`，讓 eval 能
按實際 DRR 分桶。

### 4. Direct-arrival smear

#### 4.1 目的

破壞 §1.4 的直達窗精細時間結構，讓依賴該線索的讀出在部分訓練列上失效。
設計意圖是逼模型尋找替代線索（頻譜傾斜、尾形），提升線索被遮蔽時的韌性
——殘響本身就會遮蔽這個線索，因為殘響的近場訊號在直達窗後緊接著強反射。

#### 4.2 演算法

```
n = round(smear_ms · fs / 1000)
窗 = h[peak : peak+n]
E_orig = Σ 窗²

k = randn(n);  k ← k / ||k||₂            # 隨機單位能量核
窗 ← causal_conv(窗, k)                   # 左側 zero-pad，因果卷積

if |窗[0]| < max|窗|:                     # 恢復首樣本的支配地位
    窗[0] ← sign(窗[0]) · max|窗| · 1.001

窗 ← 窗 · sqrt(E_orig / Σ窗²)             # 能量還原
h[peak : peak+n] ← 窗
```

#### 4.3 為什麼用隨機單位能量核

與隨機序列做卷積，會把原本集中在少數樣本上的脈衝叢集「攤開」到整個窗長，
破壞脈衝之間的時間關係。核歸一化到單位 L2 能量（`||k||₂ = 1`）是為了讓卷積
本身不改變訊號能量（Parseval 意義下的近似），減少後續能量還原步驟需要修正
的幅度。

**因果性**：卷積前只在左側 zero-pad，確保輸出不會出現在直達聲之前的樣本上。
非因果的塗抹會讓能量出現在直達聲抵達之前，這在物理上不可能，且會影響後續
以 `argmax` 為基準的所有偏移計算。

#### 4.4 三個刻意的保留

這個手法的設計目標是「**只**改變時間結構」，因此三件事必須保持不變：

**(a) 峰值位置**。`wav_apply_rir` 用 `argmax|h|` 做兩件事：截短窗的起點、
傳播延遲對齊的偏移量（ch2 §3.2–3.3）。峰值一移動，所有下游偏移全部跟著
移動。塗抹後可能出現某個後續樣本大於首樣本的情況，因此需要 §4.2 的支配地位
恢復步驟。

這一步的**順序很關鍵**：必須在能量還原之前做。若先還原能量再修首樣本，
修正動作本身會把能量加回去，窗的總能量不再等於原始值。

**(b) 窗內能量**。塗抹後乘 `sqrt(E_orig / E_smeared)` 還原。這保證操作不是
一次位準改變——位準線索本來就已被移除（§1.1），在這裡引入位準變化會混淆
兩種效果。

**(c) 晚期尾巴**。`smear_ms` 之後的樣本完全不動。因此 RT60 與大致的 DRR
保持不變，操作只作用在直達窗內部。

#### 4.5 套用時機與出貨狀態

與 DRR contrast 同一個位置（RIR 進 cache 之前），同一個理由：目標經 `rir_id`
重取同一條 RIR，`early` 目標才會是被塗抹過的 `full` 混音的近場成分。

這個 knob **預設關閉且無任何 recipe 啟用**。設計動機、開啟前應先閱讀的量測，
以及替代線索是否可學的討論，記在 `DirectSmearConfig` 的 docstring 與
§1.7 提到的 probe README 中。knob 入庫的目的是讓該實驗只需修改一行 config，
而不是預先替所有 recipe 做決定。

### 5. distance_level 混音模式

#### 5.1 演算法

在混音層用場景幾何重建 §1.1 被移除的位準線索：

```
SIR = 20 · log10(d_itf / d_fg) + U(jitter_db)
```

`d_fg` 是前景實現的距離、`d_itf` 是**最近**干擾者的距離。

#### 5.2 推導與參數意義

由 §1.1 的反平方定律，兩個聲源在麥克風處的振幅比為 `d_itf / d_fg`，換成
dB 即 `20·log10(d_itf / d_fg)`。0.5 m 前景對 3 m 干擾者得到約 +15.6 dB，
與自由場點聲源的物理一致。

**為何取最近的干擾者**：在 1/r 定律下，最近的干擾者是最響的，因此它決定了
有效的 SIR。若取平均距離，多個遠距干擾者會把 SIR 拉高，低估實際的干擾強度。

**jitter 的作用**：真實情況下位準不只由距離決定，還受口部指向性、說話者
音量、身體朝向、麥克風極性圖影響。`jitter_db` 為這些未建模的因素提供一個
擾動範圍，避免 SIR 與距離形成完全確定的關係（那會讓模型可以從 SIR 反推
距離，一個真實世界不存在的捷徑）。

#### 5.3 Fallback 行為

當幾何資訊不可用時（folder RIR 無 metadata、bank 未提供距離），
`_distance_level_sir` 回傳 `None`，呼叫端退回 legacy hard-SIR 抽樣。這一列
仍然可用於訓練，只是不帶位準線索。

## 工程面

### Config 對照

| Knob | Schema | 落點 |
|---|---|---|
| `augmentation_reverb.drr_contrast` | `DrrContrastConfig` | `Augmentor._apply_drr_contrast`；bank 與 simulator 兩條路徑都在 cache 前 |
| `augmentation_reverb.direct_smear` | `DirectSmearConfig` | `Augmentor._apply_direct_smear`；同上 |
| `augmentation_speech.mix_mode.modes[].distance_level` | `MixModeEntry` | `VoiceIsolationDataset._distance_level_sir` |

Schema 層的驗證約束：

* `DrrContrastConfig`：`deterministic` 模式必填 `extra_db_per_decade`；
  `random` 模式的 `near_boost_db` / `far_cut_db` 下界不得為負（方向由角色
  決定，不由數值符號決定）。
* `DirectSmearConfig`：`smear_ms_range` 下界必須大於 0（0 ms 是 no-op，該
  情境已由 `prob < 1` 涵蓋）。

### 順序與 RNG

* RIR 上的兩個 knob 順序固定：**DRR contrast → direct smear → 進 cache**。
* 兩者的機率擲點都在 short circuit 內部，關閉的 knob 不消耗 RNG，舊 recipe
  可位元級重現（[ch8](engineering_contract.zh-TW.md)）。
* deterministic DRR contrast 完全不消耗 RNG，這是「同一通道恆得同一位移」
  性質的一部分。
* direct smear 的隨機核使用 `torch.randn`，可選傳入 `generator` 以取得
  獨立的隨機流（目前 pipeline 未使用，走全域流）。

### 陷阱

* folder RIR 無 metadata，deterministic DRR contrast 與 `distance_level`
  都會靜默跳過或回退。除錯時先確認 RIR 來源。
* random 模式只處理已知角色（`foreground` / `interferer` / `media` /
  `echo`）。角色字串為預設值 `"source"` 的呼叫路徑不受影響——非
  source-level 的 whole-mix reverb 就是這種情況。
* real-far 列（[ch7](scene_construction.zh-TW.md)）**跳過整個 mix_mode**，
  包含 `distance_level`：模擬近場與真實遠場錄音之間的位準比不具物理意義，
  該列走 hard-SIR。
