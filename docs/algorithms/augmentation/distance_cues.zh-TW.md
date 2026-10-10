# 距離與時間線索

English version: [distance_cues.md](distance_cues.md)

Voice isolation 保留近場說話者、壓制其餘聲源，因此混音透露了多少聲源距離資訊，
決定了模型能學到什麼。本頁列出物理上的距離線索，說明 pipeline 保留與移除了
哪些，並描述三個操縱它們的手法：DRR contrast、direct-arrival smear 與
`distance_level` 混音。

程式碼：`AudioEffectAugmentor._apply_drr_contrast` / `_apply_direct_smear`
（`puresound/audio/augmentation.py`）、`smear_direct_arrival`
（`puresound/audio/impulse_response.py`）、`VoiceIsolationDataset._distance_level_sir`
（`puresound/task/voice_isolation.py`）。

## 演算法面

### 1. 距離線索

#### 1.1 位準（反距離定律）

```
p(d) ∝ 1/d      →      L(d) = L_ref − 20·log10(d / d_ref)
```

每倍距離下降 6 dB；0.5 m 對 3 m 為 `20·log10(6) ≈ 15.6 dB`。

**Pipeline：已移除。** recipe 設定 `dataset.gain_normalized_to` 時，語料語句在
載入時以 RMS 縮放到該位準；每個聲源的 RIR 又各自做峰值正規化
（[房間聲學](room_acoustics.zh-TW.md) §3.2），聲源之間的距離增益因此消失。
位準也是真實錄音中最容易被污染的線索（說話音量、口部指向、麥克風增益）；移除
它讓模型不能靠「小聲就是遠」，只能依賴通道性質。`distance_level` 混音（§5）在
一部分列上把它加回來。

#### 1.2 聲源之間的到達時差

`Δt = Δd / c`，3 m 與 0.5 m 之間約 7.3 ms。

**Pipeline：已移除。** 每個卷積後的聲源都以自己的直達聲峰值對齊
（[房間聲學](room_acoustics.zh-TW.md) §3.3）。

#### 1.3 直達與殘響能量比（DRR）

直達聲能量隨 `1/d²` 衰減，而擴散殘響能量大致與位置無關，因此

```
DRR_dB(d) = 20·log10(r_c / d)
```

`r_c` 是房間的臨界距離（[房間聲學](room_acoustics.zh-TW.md) §5）。

**Pipeline：保留**；DRR contrast（§3）可以放大它。位準被移除後，這是主要的
距離線索。

#### 1.4 直達窗內的精細時間結構

直達聲之後的最初幾毫秒內，會陸續到達經地面、桌面或鄰近牆面的一次反射。它們
相對直達聲的延遲取決於幾何：近距離聲源的一次反射路徑相對長得多（較晚、較弱），
遠距離聲源的路徑差縮小（較早、相對較強）。這個脈衝叢集的形狀——間隔與相對
振幅——攜帶距離資訊，且與窗內頻譜是不同的線索。

**Pipeline：保留**；direct smear（§4）可以破壞它。

#### 1.5 衰減尾形（RT60、EDC）

衰減速率由房間決定，不由距離決定。它說明房間有多響，可用來校準 DRR：同樣的
DRR 在不同 RT60 的房間對應不同距離。

**Pipeline：保留。**

#### 1.6 頻譜傾斜

遠距離聲源的高頻能量相對較少，因為空氣吸收隨頻率上升（室內主要影響 4 kHz 以上），
多數表面高頻吸收較強，而遠場訊號中反射能量的占比較高。

**Pipeline：保留**，精確度取決於 RIR 來源：影像源模擬器的牆面與頻率無關、也沒有
空氣吸收（[房間聲學](room_acoustics.zh-TW.md) §7）；hybrid bank 兩者都有建模。

#### 1.7 總表

| 線索 | 物理來源 | Pipeline |
|---|---|---|
| 位準 | `p ∝ 1/d` | 移除（載入時 RMS、逐聲源 RIR 峰值正規化）；`distance_level` 可加回 |
| 聲源間到達時差 | `Δd / c` | 移除（逐聲源峰值對齊） |
| DRR | 直達 `1/d²` 對均勻殘響場 | 保留；DRR contrast 可放大 |
| 直達窗精細時間結構 | 一次反射的路徑差 | 保留；direct smear 可破壞 |
| 衰減尾形 | 房間吸收與體積 | 保留 |
| 頻譜傾斜 | 空氣與表面吸收 | 保留，程度視 RIR 來源的建模而定 |

### 2. 兩種操作方向

* **放大線索**（DRR contrast、`distance_level`）：讓近與遠在訓練分佈中比自然
  情況更容易區分。
* **移除線索**（direct smear）：讓模型不能只依賴單一線索，在該線索被遮蔽時必須
  使用其他線索。

兩者作用在不同線索上，可以並用。

### 3. DRR contrast

#### 3.1 操作

對剛送出的 RIR 通道，把直達窗之後的整段尾巴縮放：

```
tail_start = peak + max(1, round(w · fs / 1000)),   peak = argmax|h|,  w = direct_window_ms
h[n] ← h[n] · 10^(−s/20)    for n ≥ tail_start
```

`s` 是 DRR 位移量（dB）。

#### 3.2 位移量恰好是 s

設直達能量 `E_d`、尾巴能量 `E_r`，縮放後尾巴能量為 `E_r · 10^(−s/10)`，因此

```
DRR' = 10·log10( E_d / (E_r · 10^(−s/10)) ) = DRR + s
```

前提是被縮放的區間就是被量測的區間。`direct_window_ms` 預設 2.5 ms，與
`compute_drr_db` 的窗長及取整方式相同；若改動其中一個，兩者要設成同一個值。

#### 3.3 為什麼縮放尾巴而不是直達聲

兩者都會改變 DRR，但 RIR 在下游會做峰值正規化。縮放直達聲會改變 `max|h|`，
正規化後等於同時改變了尾巴的絕對位準；縮放尾巴則峰值與正規化係數都不變。到達
模型輸入的只有比值，操作的效果保持單一。

#### 3.4 模式

**`random`**：方向依聲源角色決定，以機率 `prob` 套用。

```
foreground:                  s = +U(near_boost_db)
interferer / media / echo:   s = −U(far_cut_db)
其他角色:                     不處理
```

**`deterministic`**：位移量是該通道實現距離的固定函數，與角色無關，不擲機率、
不消耗隨機性。

```
s = extra_db_per_decade · log10(d / pivot_m)
```

代入自然關係 `DRR ≈ −20·log10(d) + C`：

```
DRR'(d) = −(20 − extra_db_per_decade)·log10(d) + C'
```

負的 `extra_db_per_decade` 讓整個池子的距離→DRR 斜率變陡；`pivot_m` 是不受影響的
距離（`d = pivot_m` 時 `s = 0`）。

random 模式每列重新抽位移，同一個距離在不同列上會得到不同的 DRR：它把 DRR 與
距離解耦，池子不再有單一的距離→DRR 映射。deterministic 模式保持單一映射，這正是
以 DRR 判讀距離的模型所需要的；這也是它不消耗隨機性的原因——同一條通道永遠得到
同一個位移。

#### 3.5 參數

| Key | 模式 | 預設 | 意義 |
|---|---|---|---|
| `mode` | — | `random` | `random` 或 `deterministic` |
| `direct_window_ms` | 兩者 | 2.5 | 直達窗，ms（> 0） |
| `prob` | random | 0.5 | 每條符合條件通道的機率，(0, 1] |
| `near_boost_db` | random | `[0, 4]` | `foreground` 的 DRR 提升範圍，dB，`0 ≤ low ≤ high` |
| `far_cut_db` | random | `[0, 4]` | 遠場角色的 DRR 降低範圍，dB，`0 ≤ low ≤ high` |
| `extra_db_per_decade` | deterministic | 必填 | 斜率變化，每十倍距離的 dB 數 |
| `pivot_m` | deterministic | 1.0 | 位移為零的距離，m（> 0） |

屬於另一模式的 key 會被拒絕。

#### 3.6 套用位置

在 `apply_rir` 的 bank 與模擬器路徑上，通道一送出就套用，在進入 RIR cache 之前：
目標以 `rir_id` 重取同一條脈衝響應，混音與目標因此看到同一個位移。metadata 記錄
`drr_contrast_shift_db` 與重新計算的 `drr_db`，讓評估能依實際 DRR 分桶。這兩條
路徑上送出的每條通道都符合條件，包括回聲通道與用來替噪音上色的通道
（角色 `interferer`）。

### 4. Direct-arrival smear

#### 4.1 目的

在一部分列上破壞 §1.4 的直達窗時間結構，讓只依賴它的距離讀出在這些列上失效。
殘響本身就會遮蔽這個線索（殘響房間中強反射緊接在直達窗之後）；頻譜傾斜之類的
替代線索是否因此可學仍是開放問題，因為時間與頻譜是被聯合讀取的。沒有任何
已發布的 recipe 啟用它。

#### 4.2 操作

`smear_direct_arrival(impaulse, sample_rate, smear_ms=..., generator=None)`，
逐通道：

```
n = round(smear_ms · fs / 1000)             （n < 2 時不處理）
window = h[peak : peak + n],  peak = argmax|h|
E = Σ window²

k = randn(n);  k ← k / ||k||₂                # 隨機單位能量核
window ← causal_conv(window, k)             # 只在左側 zero-pad

if |window[0]| < max|window|:                # 讓峰值留在原位
    window[0] ← sign(window[0]) · max|window| · 1.001

window ← window · sqrt(E / Σ window²)        # 還原窗內能量
h[peak : peak + n] ← window
```

#### 4.3 設計

* **隨機單位能量核**。與隨機序列卷積會把少數集中的脈衝攤開到整個窗、破壞它們的
  間隔。單位 L2 範數讓卷積大致保持能量，還原步驟只需小幅修正。
* **因果**。只在左側補零，直達聲之前不會出現能量；否則既不符物理，也會移動所有
  以峰值推得的偏移。
* **保留峰值位置**。`wav_apply_rir` 以峰值截目標窗並對齊傳播延遲；峰值一動，
  下游所有偏移都跟著動。支配地位的修正要在能量還原之前：之後才修會把能量加回去。
* **保留窗內能量**。操作改變的是時間，不是位準。
* **尾巴不動**。`smear_ms` 之後的樣本不變，因此 RT60 不變，DRR 也幾乎不變。

#### 4.4 參數與套用位置

| Key | 意義 |
|---|---|
| `prob` | 每條送出通道的機率，(0, 1]（必填） |
| `smear_ms_range` | `[low, high]` ms，`0 < low ≤ high`（必填）；`smear_ms ~ U(low, high)` |

緊接在 DRR contrast 之後、進 cache 之前套用，理由相同：`early` 目標必須是被塗抹過
的 `full` 混音的近場成分。metadata 記錄 `direct_smear_ms`。

### 5. `distance_level` 混音

#### 5.1 操作

在混音階段依場景幾何把 §1.1 的位準線索加回來：

```
SIR_dB = 20 · log10(d_itf / max(d_fg, 1e-3)) + U(jitter_db)
```

`d_fg` 是前景的實現距離，`d_itf` 是**最近**干擾者的距離。前景與干擾者總和再以
這個 SIR 混合，機制與每個 hard-SIR 列相同，都是 `add_bg_noise`
（[位準與動態](level_dynamics.zh-TW.md)）。

#### 5.2 各項的意義

* `20·log10(d_itf / d_fg)` 是兩個點聲源在麥克風處的自由場振幅比：0.5 m 對 3 m
  約 +15.6 dB。
* **最近的干擾者**：在 `1/r` 下它最響，決定有效 SIR；取平均距離會讓多個遠距
  干擾者把 SIR 拉高，低估干擾強度。
* **`jitter_db`**（預設 `[-3, 3]` dB）代表位準除了距離之外還取決於的因素——口部
  指向、說話音量、頭部朝向、麥克風指向圖。沒有它，SIR 就是距離的確定函數，這是
  真實錄音不會提供的捷徑。

#### 5.3 Fallback

任一距離缺失時（資料夾 RIR、沒有 source-level reverb 的列），`_distance_level_sir`
回傳 `None`，該列退回以 `augmentation_speech.snr_range` 抽的 hard SIR。這一列
仍然參與訓練，只是不帶位準線索。

## 工程面

### Config 對照

| Knob | Schema | 落點 |
|---|---|---|
| `augmentation_reverb.drr_contrast` | `DrrContrastConfig` | `AudioEffectAugmentor._apply_drr_contrast`，bank 與模擬器路徑，cache 之前 |
| `augmentation_reverb.direct_smear` | `DirectSmearConfig` | `AudioEffectAugmentor._apply_direct_smear`，同一位置 |
| `augmentation_speech.mix_mode.modes[].distance_level` | `MixModeEntry` | `VoiceIsolationDataset._distance_level_sir` |

兩個 RIR knob 只有在其區塊設 `used: true` 時才啟用。`mix_mode` 只存在於
voice-isolation 任務；一個模式條目結合 `name`、`prob` 與 `distance_level: true`：

```yaml
augmentation_speech:
  mix_mode:
    used: true
    modes:
      - {name: physical,       prob: 0.4, physical: true}
      - {name: distance_level, prob: 0.2, distance_level: true, jitter_db: [-3, 3]}
      - {name: moderate,       prob: 0.4, sir_range: [-3, 6]}
```

驗證規則：deterministic DRR contrast 必填 `extra_db_per_decade`；random 模式的
範圍不得為負（正負號由角色決定）；`smear_ms_range` 必須從大於 0 ms 開始（0 ms 是
no-op，已由 `prob < 1` 涵蓋）。

### 順序與 RNG

* 每條送出的通道：**DRR contrast → direct smear → cache**。
* 兩者的機率擲點都在 short circuit 之內，關閉的 knob 不消耗隨機性，其他 recipe
  可位元級重現（[工程契約](engineering_contract.zh-TW.md)）。
* random DRR contrast 與 smear 的抽樣（`prob`、`smear_ms`）使用 Python `random`；
  smear 的核使用全域 `torch` 流（可傳 `generator` 取得獨立的流，但 pipeline 沒有
  傳）。deterministic DRR contrast 不抽任何值。
* `distance_level` 的 jitter 使用 `torch` 流，在模式選擇之後抽取。

### 陷阱

* 資料夾 RIR 沒有 metadata：deterministic DRR contrast 會跳過，`distance_level`
  會退回。knob 看起來沒作用時先確認 RIR 來源。
* random DRR contrast 忽略 `foreground`、`interferer`、`media`、`echo` 以外的
  角色。整段混音 reverb 的角色是 `source`，random 模式不會動它；deterministic
  模式與角色無關，只要通道有距離仍會位移它。
* 帶有真實遠場干擾者的列（real-far 列，以及從 real-far 池取干擾者的 real-near 列）
  完全跳過 `mix_mode`，包含 `distance_level`：兩個以不同方式錄製與正規化的訊號之間
  的位準比沒有物理意義，這些列使用 hard SIR。session 列使用自己的 SIR 抽樣
  （[場景構成](scene_construction.zh-TW.md) §4）。
