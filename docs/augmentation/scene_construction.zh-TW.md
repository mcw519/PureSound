# 場景構成

English version: [`scene_construction.md`](scene_construction.md)

一列訓練資料描述一個場景：誰在說話、位在哪裡、什麼時候說、多大聲，以及環境
有什麼聲音。本章整理場景層的手法——row 類型、干擾者來源、overlap gating、
混音模式、回聲殘餘、噪音三源。

實作分佈在 `puresound/task/ns.py`（合成骨架）、
`puresound/task/voice_isolation.py`（近/遠場 row 類型）、
`puresound/task/overlap_gating.py`、`puresound/task/noise_stage.py`。

## 演算法面

### 1. Row 類型：一列資料的劇本

`_plan_row` 在每列開始時擲一次，決定該列的類型（`RowPlan` /
`VoiceIsolationRowPlan`）。之後的合成骨架依 plan 分岔。

#### 1.1 一般列

前景語者 + （以機率決定的）合成干擾者 + 噪音。這是基準路徑。

#### 1.2 target-absent 列

模擬「沒有近場使用者」的情境。實作方式是**減除法**而非「不加入」：

```
target_in_mix = 前景對混音的貢獻（在加入干擾者之前快照）
...正常合成整個場景...
noisy  ← noisy − target_in_mix
target ← 0
```

**為什麼用減除而非不加入**。SIR 抽樣（§4）與 overlap gating（§3）都需要
一個參考訊號來定義「相對於前景多大聲」、「在前景的靜音段填多少」。若一開始
就不放前景，這兩個機制失去參考，該列的干擾者位準與活動分佈會與其他列不
一致。先正常合成、最後減除，可以讓這列的干擾者統計與一般列完全對齊。

減除必須在任何位準縮放之前執行，否則減不乾淨（縮放後的混音減去未縮放的
快照會留下殘差）。

`force_interferer` 可保證這列在拿掉前景後場景不是空的。

#### 1.3 real-far 列（voice isolation）

遠場通道**不做 RIR 卷積**，直接使用池子中已完成的「喇叭 → 空氣 → 麥克風」
真實錄音。

設計理由：卷積出來的遠場通道只帶有收音鏈的 LTI 部分
（[ch2](room_acoustics.zh-TW.md) §1）。一段真實的端到端錄音同時帶有非 LTI
的部分——喇叭的非線性、裝置的 AGC 行為、實際的底噪結構。用真實錄音當干擾
源，是讓訓練分佈涵蓋這些成分的直接手段。

`lone_far_prob` 讓一部分列整列只有遠場聲、完全沒有近場語者，構成「無近場
錨」的形狀。

#### 1.4 real-near 列（voice isolation）

**前景**整條換成真實近場錄音。這類列必須跳過前景 RIR，也跳過 whole-mix
RIR（`skip_whole_mix_reverb`）——通道必須保持錄音的原樣，再卷一次 RIR 會
變成「錄音又被放到另一個房間」。

real-near 列可攜帶 room 與 speaker 標籤。同一列的 real-far 干擾者會優先從
**同一個房間**抽取、並排除同一位語者。前者讓一列內的近場與遠場屬於同一個
聲學環境；後者避免前景與干擾者是同一個人。

#### 1.5 turn-taking 覆寫

real 列可各自帶 `turn_taking_prob` 覆寫（§3.3）。覆寫發生在 row 層而非
config 層，因此不影響任何其他列的分佈——這是「只讓某類 row 有更多遠場獨白」
的實作方式。

### 2. 干擾者的取樣與上色

#### 2.1 語者與語句取樣

* 干擾者人數由 `add_n_cases` 決定，可為定值或 `[lo, hi]` 區間隨機。
* 語者池排除前景語者。
* 取樣率不符的語者以**有界重試**處理：最多重抽 5 次，仍不符則跳過該干擾者
  （人數本身已是隨機的，少一個不影響列的有效性）。

**為什麼重試必須有界**。無界重試在「某語者完全沒有該取樣率的語句」時會
永久迴圈。DataLoader worker 卡住之後，DDP 的某個 rank 永遠到不了下一個
collective 操作，整個訓練 deadlock。同樣的教訓在 `align_audio_list` 的
防靜音裁窗重試中也出現（上限 10 次）——那裡的極端情況是整條語音都接近
靜音，沒有任何 offset 能滿足條件。

#### 2.2 media 上色

每個干擾者以 `media_voice.prob` 獨立擲骰決定是否成為「媒體聲源」（電視、
喇叭播放）。上色內容為帶限加冪次波形整形，數學見
[ch5](spectral_channel.zh-TW.md) §6。

順序上，上色在該干擾者的 RIR **之前**執行：先有裝置放音，再經房間傳播，
符合物理因果。

角色差異：on-the-fly 模擬器會把 media 角色貼牆放置
（[ch2](room_acoustics.zh-TW.md) §8），因此 media 的早期反射相對更強；
pre-generated bank 沒有這個機制，media 與一般干擾者共用同一個遠場通道池，
差別只剩上色本身。

### 3. Overlap gating：誰在什麼時候說話

#### 3.1 問題

合成資料若不做時間結構處理，得到的是「兩個人整列不間斷同時說話」——目標
語句與干擾語句都是連續語音，重疊率天然偏高（超過 80%）。這個分佈缺少決定
實際行為的兩種情境：干擾者在目標的靜音段說話、遠場語者獨自持有發言權而
整列沒有近場語者。

`OverlapGating` 提供兩種機制。它們不是彼此的變體，產生的時間結構本質不同。

#### 3.2 逐幀 Bernoulli（預設）

每個干擾者獨立抽一個 overlap regime，然後逐幀決定是否發聲。

**regime 抽樣**用一次擲骰對兩個累積門檻：

```
roll ~ U(0, 1)
roll < no_overlap_prob                        → p = 0
roll < no_overlap_prob + high_overlap_prob    → p = U(high_overlap_range)
否則                                           → p = U(mid_overlap_range)
```

三個 regime 的用意是讓一個 batch 涵蓋整個難度譜，而不是全部集中在平均值
附近。同一列的不同干擾者各自抽 regime，因此一列可以同時有一個容易的與一個
困難的干擾者。

**逐幀決定**對目標的活動段與靜音段用不同的機率：

```
目標活動幀:  發聲機率 = p（該 regime 抽到的 overlap 率）
目標靜音幀:  發聲機率 = U(fill_on_silent_range)
```

分開設定的理由：一個只在目標說話時才發聲的干擾者，是比真實對話窄得多的
分佈。真實情況下，干擾者在目標的停頓中繼續說話是常態，`fill_on_silent`
通常設得比 overlap 率高。

這個機制**不 gate 目標本身**，只改變「誰蓋在目標上」。

#### 3.3 Turn-taking

近場與遠場以長對話輪交替，模擬真實的話輪轉換。

**輪次腳本生成**（`_turn_script`）在 VAD 幀網格上進行：

```
fps = fs / hop                                    # 幀率
far_first = (rand < far_first_prob)                # 是否以遠場輪開場

while pos < n_frames:
    turn_s = U(turn_far_seconds 或 turn_near_seconds)
    turn_f = round(turn_s · fps)
    對應的 mask[pos : pos+turn_f] = True

    if rand < 0.5:                                 # 輪間留空隙
        pos = end + round(U(turn_gap_seconds) · fps)
    else:                                          # 輪間輕微搶話
        pos = max(pos+1, end − round(U(turn_overlap_seconds) · fps))

    切換近/遠
```

設計要點：

* **輪長分近遠兩組範圍**。近場與遠場的自然發言長度不同，分開設定讓兩者
  可獨立控制。
* **空隙與搶話各 50%**。真實對話的話輪交界有時留白、有時重疊，兩者都存在。
  純空隙會讓模型看不到交界處的重疊。
* **`far_first_prob` 決定開場方**。以遠場輪開場的列會產生「多秒的遠場獨白，
  前面沒有任何近場錨」——這是 Bernoulli 填充永遠造不出的形狀（Bernoulli
  不 gate 目標，目標一定從頭就在），也是 streaming 情境下最困難的開局。

**目標的標籤與貢獻必須用同一條包絡**。turn-taking 是唯一會 gate 目標的
機制。實作對 `target`（標籤來源）與 `target_mix`（目標對混音的貢獻）套用
同一條 `near_mask` 包絡。若只 gate 其中一個，這列就會宣稱近場語者在混音中
是靜音的幀裡說過話——一個自相矛盾的訓練樣本。

#### 3.4 防 click 包絡

幀遮罩在單一樣本內從 0 跳到 1，在波形上是一個階躍不連續，聽起來是一聲
click，頻譜上是寬頻脈衝。`_Envelope` 的處理：

```
1. 幀遮罩以 hop 長度上採樣到樣本率（repeat_interleave）
2. 與歸一化 Hann 窗卷積（長度 2·fade_samples + 1）
3. clamp 到 [0, 1]
```

Hann 窗歸一化為單位和，因此在遮罩的平坦區內卷積結果仍為 1（不改變位準），
只在邊緣產生平滑過渡。過渡長度為 `fade_samples`，預設 400 樣本
（16 kHz 下 25 ms），與語音的音節時間尺度相當——足夠平滑到不產生 click，
又短到不會明顯改變語音的起始時間。

這個操作不消耗隨機性。

#### 3.5 回報的 overlap_fraction

回報值是**實現值**而非抽樣的機率：干擾者活動幀與目標活動幀的交集，除以
目標活動幀數。抽樣機率不等於實現值，因為目標自身的靜音分佈會影響結果。

turn-taking 分支的分母是 **gate 後**的目標活動幀。近場語者在遠場輪中依
構造是靜音的，把那些幀算成「目標在說話」會回報出不存在的重疊。

Bernoulli 分支不會除以零，因為 `apply` 在目標完全無活動幀時已提早返回；
turn-taking 分支則需要額外保護——近場包絡有可能完全錯過所有目標活動幀。

### 4. 混音：前景對干擾者的位準關係

#### 4.1 基準：hard SIR

```
SIR ~ U(augmentation_speech.snr_range)
```

混音數學見 [ch4](level_dynamics.zh-TW.md) §2。這是 NS 任務的唯一模式，也是
voice isolation 的 fallback。

#### 4.2 mix_mode（voice isolation）

以機率權重在多種位準關係之間抽樣。`_sample_mix_mode` 用累積權重選擇，
權重不需總和為 1。

| 模式 | 位準關係 |
|---|---|
| `physical` | 不做任何相對縮放，直接相加 |
| `distance_level` | 由場景幾何重建反距離定律（[ch3](distance_cues.zh-TW.md) §5） |
| 其他（`moderate`、`counter_level` 等） | 各自帶明確的 `sir_range` 抽樣 |

**`physical` 模式的重要注意事項**。這個名稱容易誤解：它不做額外縮放，但
**不代表**得到物理正確的 1/r 位準定律。原因是上游已經兩次拉平位準——載入
時的 RMS rescale 與 RIR 的逐通道峰值正規化（[ch2](room_acoustics.zh-TW.md)
§3.1）。因此「自然相加」的結果是 SIR 接近 0 dB，與距離無關。實作照實計算
並回報實現 SIR：

```
realized_SIR = 10 · log10( Σ fg² / Σ itf² )
```

要得到真正的距離位準定律，必須使用 `distance_level` 模式。

#### 4.3 real-far 列跳過 mix_mode

real-far 列固定走 hard-SIR。理由是「模擬近場語音」與「真實遠場錄音」之間的
位準比不具物理意義——兩者的位準各自由不同的錄製與正規化流程決定，套用
幾何公式只會產生一個沒有物理對應的數字。

### 5. 回聲殘餘（echo playback）

模擬裝置自身喇叭漏回麥克風、且上游 AEC 未完全消除的殘餘。

建模方式：另取一段語句、經過**同一個房間**的一條通道、以低於當下混音
`U(erle_db_range)` dB 的位準加入。實作直接重用 SNR 混音機制：

```
noisy = add_bg_noise(noisy, [echo], snr_list=[erle_db])
```

**ERLE**（echo return loss enhancement）是 AEC 的性能指標，定義為 AEC 消除
的回聲能量比。以 ERLE 當作 SNR 參數，意義是「AEC 之後還剩多少回聲」——
25 dB 的 ERLE 代表殘餘回聲比混音低 25 dB。

三個約束：

* **永不進入目標**。回聲是要被消除的東西。
* **需要 source-level 房間**：它要自己的 RIR 通道。
* **通道 metadata 刻意丟棄**：回聲不屬於 RIR lineage 的一部分。

已知限制：使用 pre-generated bank 時，`distance_range_override` 無法被滿足
（bank 只能挑最接近請求距離帶的既有通道），回聲通道會落在遠場池中。真正的
近場回聲通道需要 on-the-fly 模擬器。物理上回聲的路徑應該很短（裝置喇叭到
裝置麥克風），這個限制讓 bank 路徑的回聲比實際更「遠」。

### 6. 噪音三源

`NoiseStage` 提供三個非語音來源。它們是三個而非一個，因為各自回答不同的
問題。

#### 6.1 錄音噪音（SNR 相對）

從噪音語料抽取，以 `U(snr_range)` 相對當下混音混入。

**dynamic type**（1/4 機率）：串接兩條不同的噪音再正規化，讓一列之內噪音
場景切換。

**room coloring**（可選）：噪音先經過**同一個房間**的一條通道再混入。
若不做，乾的噪音配上濕的語音會讓模型免費得知「哪個是語音」——殘響本身
就成了語音的標記。這是一個資料洩漏的修補，不是額外的真實性提升。

實作以 `noise_transform` callable 注入，在 SNR 縮放**之前**卷積，因此
上色不影響最終的 SNR（SNR 是對卷積後的噪音定義的）。

#### 6.2 白噪（SNR 相對）

在錄音噪音未走 dynamic 分支的列上，以 `prob_white_noise` 疊加高斯白噪，
SNR 相對**當下混音**（已含錄音噪音）。

用途是提供錄音噪音池不一定涵蓋的廉價寬頻覆蓋。真實裝置的底噪往往接近
白噪或粉紅噪，而錄音噪音池多為具體場景（咖啡廳、街道、風扇）。

#### 6.3 絕對底噪（capture floor）

```
level_dbfs ~ U(level_dbfs_range)
floor = randn_like(noisy) · 10^(level_dbfs / 20)
noisy ← noisy + floor
```

關鍵差異：位準**不隨語音縮放**。麥克風的自雜訊與房間本底噪音不管誰在說話
都坐在同一個位準上，SNR 相對式的來源無法表達這件事——SNR 固定意味著語音
小聲時噪音也跟著小聲。

**「dBFS」是抽樣時的位準，不是交付時的位準**。device chain 末端的 A/D
增益調度（[ch6](device_chain.zh-TW.md) §3）會整列縮放，因此抽在 −45 dBFS
的底噪，在該列被降低 6 dB 之後實際交付的是 −51 dBFS。

這對它模擬的對象是**正確的**：電容自雜訊與房間本底都位於前級放大器的
上游，因此會跟著前級的增益調整一起變化。轉換器自身的電子雜訊才不會——
那個雜訊要加在 A/D 之後，但其量級（約 −90 dBFS）比這個 knob 的抽樣範圍
低 40 dB 以上，因此 pipeline 中沒有對應的 stage。

#### 6.4 順序與共同約束

固定順序：**錄音噪音 → 白噪 → 絕對底噪**，全部在 device chain 之前。

底噪放在 device chain 之前也是刻意的：電容自雜訊與語音一樣要經過裝置的
頻率響應，因此它應該與語音一起穿過類比路徑。

三源都**只加在混音上，不加在目標上**——把噪音加進目標等於要求模型重現它。

## 工程面

### Config 對照

| 區塊 | Schema | 內容 |
|---|---|---|
| `augmentation_speech` | `SpeechAugmentation` | `prob`、`add_n_cases`、`snr_range`（hard SIR）、`is_target` |
| `augmentation_speech.media_voice` | `MediaVoiceConfig` | 帶限範圍、壓縮次冪範圍 |
| `augmentation_speech.overlap_control` | `OverlapControlConfig` | 三 regime 機率與範圍、`fill_on_silent_range`、`fade_samples`、turn-taking 的輪長/gap/overlap/`far_first_prob` |
| `augmentation_speech.echo_playback` | `EchoPlaybackConfig` | `distance_range`、`erle_db_range` |
| `augmentation_speech.mix_mode` | `MixModeConfig` / `MixModeEntry` | 模式清單與權重（僅 voice_isolation 任務接受） |
| `augmentation_target_absent` | `TargetAbsentAugmentation` | `prob`、`force_interferer` |
| `augmentation_realfar` / `augmentation_realnear` | `RealFarAugmentation` / `RealNearAugmentation` | `pool_manifest`、`prob`、`lone_far_prob`、`turn_taking_prob` |
| `augmentation_noise` | `NoiseAugmentation` | `snr_range`、`prob_white_noise`、`white_noise_snr_range`、`room_coloring`、`absolute_floor` |

`mix_mode` 在純 NS dataset 上會直接拋錯（建構子檢查），因為近/遠位準關係
只有 voice isolation 實作。這是「一個區塊只存在於真的消費它的任務」原則的
執行點。

### 順序

`__getitem__` 的場景段順序（完整 16 站表見
[ch8](engineering_contract.zh-TW.md)）：

```
row 計畫 → 前景通道 → 干擾者（media 上色 → RIR）→ overlap gating
  → 混音（hard SIR / mix_mode）→ target-absent 減除 → echo
  → 防削波 → 變速 → whole-mix RIR → 噪音三源
```

### RNG 紀律的兩個細節

**Bernoulli regime 的抽樣消耗量依 regime 而異。** `_draw_overlap_rate` 先擲
一次選 regime，只有兩個非零 regime 才會再抽一次範圍內的值。因此 regime
門檻的順序**不可交換**：把 `no_overlap_prob` 與 `high_overlap_prob` 互換，
同一個 seed 產生的不是「權重不同的分佈」，而是**完全不同的列**——因為
RNG 流的消耗量在該點之後全部錯位。

**turn-taking 對 target-absent 列強制關閉**（`allow_turn_taking=False`）。
前景是用 gating **之前**的快照減除的（§1.2）。若 gate 了前景，減除時用的
是未 gate 的快照，會留下「這列宣稱不存在的那個聲音」的殘差。

### 陷阱

* `is_target: true` 會把 gating 後的混音直接複製為目標（NS 的「所有語者都是
  目標」用法）。這與 voice isolation 的所有近/遠機制互斥，不可同時啟用。
* VAD 參照的快照時點是變速**之後**、噪音與 device chain **之前**：時間軸
  必須與混音一致（所以在變速後），但標籤要打在乾淨語音上（所以在噪音前）。
* `far_target`（gate 後、SIR 後的干擾者總和）沒有任何 loss 消費它。它是
  eval 的洩漏診斷輸出，不是訓練目標。`ns.py` 中該欄位上方的註解記錄了這段
  歷史——它曾被註解為某個不存在的 loss 的輸入。
* real-near 列的前景不經過任何 RIR。若同時啟用 whole-mix reverb，
  `skip_whole_mix_reverb` 是唯一防止該列被二次卷積的機制，不要繞過它。
