# 場景構成

English version: [scene_construction.md](scene_construction.md)

一列訓練資料描述一個場景：誰在說話、在哪裡、什麼時候說、多大聲，以及環境中還有
什麼聲音。本頁涵蓋場景層的階段——row 類型、干擾者來源、overlap gating、前景與
干擾者的混音、回聲殘餘、開頭環境段與三個噪音來源。

程式碼：`puresound/task/ns.py`（合成骨架，`NoiseSuppressionDataset.__getitem__`）、
`puresound/task/voice_isolation.py`（近/遠場 row 類型）、
`puresound/task/session_rows.py`（對話列）、`puresound/task/overlap_gating.py`、
`puresound/task/noise_stage.py`、`puresound/audio/noise.py`（`add_bg_noise`）。

## 演算法面

### 1. Row 類型

`_plan_row` 在每列開始時決定一次這一列的類型，回傳 `RowPlan`（voice isolation 中
為 `VoiceIsolationRowPlan`）。之後的骨架依 plan 的旗標分岔：

| 旗標 | 效果 |
|---|---|
| `target_absent` | 最後把前景從混音中移除；目標歸零 |
| `force_interferer` / `force_speech_interferers` | 干擾者區塊不擲機率直接觸發 |
| `skip_whole_mix_reverb` | 這一列不卷整段混音 RIR |
| `skip_overlap_gating` | 這一列自帶輪次腳本 |
| `speed_perturb_companions` | 變速也套用在 `background_speech_reference` 上 |

Voice isolation 的決策順序是 **session 列 → real-near 列 → real-far 列 → 合成
target-absent**；每個決策都有防護，關閉的區塊不抽任何值。

#### 1.1 一般列

前景語者、以 `augmentation_speech.prob` 機率加入的合成干擾者，再加噪音。其他
row 類型都是這條路徑的變化。

#### 1.2 Target-absent 列

`augmentation_target_absent`（`prob`、`force_interferer`）模擬「沒有近場使用者」。
做法是減除而非省略：

```
target_in_mix = 前景對混音的貢獻（加入干擾者之前）
... 正常合成整個場景 ...
noisy  ← noisy − target_in_mix
target ← 0
```

SIR（§4）與 overlap gating（§3）都以前景為參考。先正常合成、最後減除，讓這些列的
干擾者位準與活動分佈與一般列完全一致。減除發生在混音任何重新縮放之前（混音步驟
只縮放干擾者），因此能完全抵消。`force_interferer` 保證拿掉前景後這一列不是空的。

#### 1.3 Real-far 列（voice isolation）

`augmentation_realfar` 以池子 manifest 中已完成的「喇叭 → 空氣 → 麥克風」錄音取代
合成干擾者，插入時**不做 RIR**。卷積出的遠場通道只帶有收音鏈的 LTI 部分
（[房間聲學](room_acoustics.zh-TW.md) §1）；真實錄音還帶有卷積做不到的部分——
換能器非線性、指向性、真實的底噪、位準與頻譜傾斜。前景仍使用模擬通道。

* `prob`：成為 real-far 列的比例。
* `lone_far_prob`：這些列中完全沒有近場語者的比例（`target_absent` 與
  `force_interferer`），與 `augmentation_target_absent` 獨立抽樣，讓兩種壓力可
  分別調整。
* 干擾者人數取自 `augmentation_speech.add_n_cases`；每段錄音都像語料語音一樣載入
  （RMS 位準、重採樣），量測的 `distance_m` 以 `origin: real` 隨 metadata 帶出。

Manifest 每行一個 JSON 物件：`wav_path`、`distance_m`、`room`、`speaker`、`mic`
（由 `egs/voice_isolate/scripts/build_real_recording_pool.py` 產生）。

#### 1.4 Real-near 列（voice isolation）

`augmentation_realnear` 以真實的近講錄音取代**前景**，目標就是這段錄音本身。這一列
不模擬房間：沒有前景 RIR，`skip_whole_mix_reverb` 也擋掉整段混音 RIR——再卷一次
RIR 等於把錄音放進第二個房間。real-far 池有載入時，干擾者取自該池：優先取與近場
錄音同一個 `room`（該房間條目足夠時），並排除同一個 `speaker`，讓近與遠主要只差
距離，收音鏈身分無法當作線索。這些列的目標一定存在。

`stitch_to_length`（兩個 real 區塊皆有）：比列長短的池子錄音會被補零，在 keep 列上
會讓目標有一部分是數位靜音。開啟後會串接同一個 `(speaker, room, mic)` 的其他錄音，
直到覆蓋整列。

#### 1.5 Session 列（voice isolation）

`augmentation_session_rows` 把同一套機制排成一段對話：一位使用者、一到兩位旁人，
以及一份輪次腳本，並附上標示每一輪由誰發言的標籤。一般列的目標幾乎都在前半秒內
開口、很少讓干擾者先發言、也從不讓同一位說話者出現兩次；session 列補上這些情境。

* **資格**。只有長度至少 `min_seconds`（預設 12 s）的列；長度檢查在機率擲點之前，
  因此 recipe 的短長度桶抽到的值與沒有此區塊時完全相同。腳本最多涵蓋
  `max_seconds`。
* **腳本形狀**（`shape_probs`，權重）：`user_first`、`bystander_first`（旁人先說
  `bystander_open_seconds`，錯錨情境）、`user_gap`（使用者在 `user_gap_seconds`
  之後回來；空檔內旁人以 `bystander_in_gap_prob` 機率說話）、`overlap`（輪次交界
  重疊：雙講）。列長不足以容納的形狀會降級（`user_gap` → `bystander_first` →
  `user_first`），回報的是實現的形狀。輪次交替，長度取自 `user_turn_seconds` 與
  `bystander_turn_seconds`，交界以 `turn_gap_seconds` 分開或以 `overlap_seconds`
  重疊（機率 `boundary_overlap_prob`，`overlap` 形狀一律重疊）。腳本至少有一個
  使用者輪與一個旁人輪。
* **說話者**。使用者與旁人取自與一般前景相同的語者池，因此「使用者」是這一列的
  角色，而不是某個聲音的屬性。
* **通道**。有 source-level reverb 時，使用者經過 `user_distance_range`（預設
  0.3–1.0 m）內的近場通道；以 `rir_move_prob` 機率，後段的使用者輪改用同一房間的
  第二條近場通道（使用者移動了）。旁人使用 `bystander_distance_range`（預設
  1.5–4.0 m），但以 `distance_matched_bystander_prob` 機率會有一位旁人放在使用者的
  距離範圍內，使距離本身無法辨認使用者。每位說話者以自己的輪次遮罩 gate，包絡與
  turn-taking 相同（§3.4）。
* **位準**。旁人匯流排以 `sir_range` 抽的 SIR 混入，或以 `sir_low_tail_prob` 機率
  改從 `sir_low_tail_range` 抽，且不低於 `augmentation_speech.snr_range` 的下界。
  之後一定會在混音上加一層取自 `floor_dbfs_range` 的收音底噪，因此使用者的空檔
  是房間聲音而非數位靜音。使用者空檔從不標成 target-absent。
* **標籤**（`SESSION_LABEL_KEYS`，位於 `vad_target` 幀網格上，並依該列的變速倍率
  縮放）：`user_active`、`bystander_active`、`turn_id`（每個單人輪 1..K，雙講處為
  0；每輪只保留最長的連續段）、`turn_role`、`turn_speaker`、`turn_chain`、
  `turn_distance` 與 `row_source_id`。啟用 session 的 recipe 中每一列都帶有這些 key
  （非 session 列沒有輪次，`row_source_id = −1`），另有 `SESSION_SCALAR_KEYS`
  診斷值。
* **配對列**（兩者互斥）：`pair_prob` 在一個結束後會還原 RNG 流的 scope 中，以 slot
  seed 渲染來源素材，因此兩列 `row_source_id` 相同時，是同一份來源經過兩次獨立的
  chain 與噪音抽樣；`paired_view_prob` 則對完全相同的已完成混音再跑一次 device
  chain，收在 `paired_view` 下（列長至少 `paired_view_min_seconds`）。

Plan 會設定 `skip_overlap_gating`、`skip_whole_mix_reverb`、
`force_speech_interferers` 與 `speed_perturb_companions`。啟用的 session 區塊若與
`augmentation_speech.is_target: true`（旁人會變成目標）或啟用的
`augmentation_row_initial_ambient`（開頭段會遮掉腳本宣稱存在的語音）同時出現，
會在載入 config 時被拒絕。

#### 1.6 逐列的 turn-taking 比例

Real-near 與 real-far 區塊各有選用的 `turn_taking_prob`，只對自己的列覆寫
`overlap_control.turn_taking_prob`（§3.3）。覆寫以列為單位而非寫在共用 config，
因此其他 row 類型的分佈——以及 RNG 消耗——都不變。

### 2. 干擾者

#### 2.1 語者與語句取樣

* 人數取自 `add_n_cases`，可為整數或 `[lo, hi]` 範圍，並夾到 `[1, 可用語者數]`。
* 語者池排除前景語者。
* 取樣率不符的抽樣最多重試 5 次，之後跳過該干擾者。無界重試會在某語者沒有所需
  取樣率的語句時永久迴圈；DataLoader worker 卡住會讓某個 DDP rank 到不了下一個
  collective，整個訓練 deadlock。`align_audio_list` 尋找非靜音裁窗時也因同樣理由
  設有上限（10 次）。

#### 2.2 Media 上色

每個干擾者以 `media_voice.prob` 機率獨立成為媒體聲源（電視、喇叭播放）：以
`hp_cutoff_range` / `lp_cutoff_range` 帶限、以 `compress_power_range` 做冪次波形
整形，並還原 RMS（[頻譜與通道效應](spectral_channel.zh-TW.md)）。上色在該干擾者的
RIR 之前——先由裝置播放，再由房間傳遞。On-the-fly 模擬器會把 `media` 聲源貼牆放置
（[房間聲學](room_acoustics.zh-TW.md) §8）；bank 則從與其他干擾者相同的遠場池提供
`media` 通道，差別只剩上色。

### 3. Overlap gating

#### 3.1 問題

不做時間處理的合成列是兩個人整列不停地說話，接近全程重疊主導了分佈。這個分佈
缺少決定實際行為的情境：干擾者在目標的停頓中說話，以及遠場說話者獨自發言、整列
沒有近場說話者。

`OverlapGating`（`augmentation_speech.overlap_control`）提供兩種產生不同時間結構
的機制。兩者都在由 `vad_label` 區塊建立的廉價 energy VAD 幀網格上運作；沒有該區塊
時 gating 停用。

#### 3.2 逐幀 Bernoulli（預設）

每個干擾者用一次擲骰對兩個累積門檻抽 overlap regime：

```
roll ~ U(0, 1)
roll < no_overlap_prob                        → p = 0
roll < no_overlap_prob + high_overlap_prob    → p = U(high_overlap_range)
否則                                           → p = U(mid_overlap_range)
```

再逐幀 gate：

```
目標活動幀：以機率 p 開啟
目標靜音幀：以機率 U(fill_on_silent_range) 開啟
```

三個 regime 讓一個 batch 涵蓋整個難度範圍；同一列的干擾者各自抽樣，因此一列可以
同時有容易與困難的干擾者。靜音幀另設比例，是因為真實干擾者會在目標停頓時繼續
說話；只在目標說話時才開口的干擾者是窄得多的分佈。目標本身不被 gate。

預設值：`no_overlap_prob` 0.25、`high_overlap_prob` 0.25、
`mid_overlap_range` [0.1, 0.5]、`high_overlap_range` [0.5, 1.0]、
`fill_on_silent_range` [0.3, 0.5]。

#### 3.3 Turn-taking

以 `turn_taking_prob` 機率（預設 0，逐列覆寫見 §1.6），近場與遠場改以長輪次交替。
`_turn_script` 在 VAD 幀網格上寫腳本：

```
fps = fs / hop
far_first = rand < far_first_prob

while pos < n_frames:
    turn = U(far 時用 turn_far_seconds，否則 turn_near_seconds) · fps   （≥ 1 幀）
    標記目前這一方的 mask[pos : pos + turn]
    if rand < 0.5:  pos = end + U(turn_gap_seconds) · fps              # 停頓
    else:           pos = max(pos + 1, end − U(turn_overlap_seconds) · fps)  # 重疊
    換另一方
```

* 近場與遠場輪長各有範圍（`turn_near_seconds` 預設 [1.5, 3.0] s、
  `turn_far_seconds` [2.0, 4.5] s）。
* 交界以相同機率停頓或重疊；只有停頓的話，模型永遠看不到輪次交界的重疊。
* `far_first_prob`（預設 0.5）比例的列以遠場輪開場：數秒的遠場獨白、前面沒有任何
  近場錨，這是 Bernoulli 填充造不出的形狀（它從不 gate 目標），也是 streaming 模型
  最困難的開局。

Turn-taking 是唯一會 gate 目標的機制。同一條近場包絡同時套在 `target`（標籤來源）
與 `target_mix`（目標對混音的貢獻）上；只 gate 其中一個，會宣稱近場語者在混音靜音
處說過話。Target-absent 列停用 turn-taking：前景以 gating 之前的快照減除（§1.2），
被 gate 過的前景會留下「這一列宣稱不存在的那個聲音」的殘差。

#### 3.4 防 click 包絡

在單一樣本內切換的幀遮罩是波形上的階躍：一聲 click、一個寬頻脈衝。`_Envelope` 把
遮罩依 hop 上採樣（`repeat_interleave`），與長度 `2 · fade_samples + 1`、歸一化為
單位和的 Hann 窗卷積，再夾到 [0, 1]。平坦區內增益維持 1，只有邊緣在 `fade_samples`
（預設 400 樣本，16 kHz 下 25 ms）內漸變。它不消耗隨機性。

#### 3.5 回報的 `overlap_fraction`

回報實現值而非抽樣的機率：目標活動且至少一個干擾者的 gate 開啟的幀數，除以目標
活動幀數。Turn-taking 分支的分母是 **gate 後**的目標活動，因為近場語者在遠場輪中
依構造是靜音的。Bernoulli 分支不會除以零（目標沒有任何活動幀時 `apply` 已提前
返回）；turn-taking 分支則明確防護，因為近場包絡可能錯過所有活動幀。

### 4. 前景與干擾者的混音

#### 4.1 Hard SIR

```
SIR ~ U(augmentation_speech.snr_range)
noisy = fg + (rms(fg) / 10^(SIR/20)) · itf / rms(itf)
```

`add_bg_noise` 把干擾者匯流排在整列上正規化為單位 RMS，再對前景的整列 RMS 縮放
（[位準與動態](level_dynamics.zh-TW.md)）。位準在 gating 之後才設定，因此 gate 得
很稀疏的干擾者在說話時會比較大聲。這是 noise suppression 唯一的模式，也是 voice
isolation 的 fallback。

#### 4.2 `mix_mode`（voice isolation）

`augmentation_speech.mix_mode.modes` 是一組依權重抽選的位準關係
（`_sample_mix_mode`；權重不需總和為 1）：

| 模式 | 位準關係 |
|---|---|
| `physical: true` | 不做相對縮放，直接相加 |
| `distance_level: true` | 依該列幾何以反距離定律決定 SIR（[距離線索](distance_cues.zh-TW.md) §5） |
| 其他條目 | 從自己的 `sir_range` 抽 SIR |

**`physical` 不是 `1/r` 定律**。位準在上游已被拉平——載入時的 RMS 縮放與逐聲源的
RIR 峰值正規化（[房間聲學](room_acoustics.zh-TW.md) §3.2）——因此「自然」相加的
結果 SIR 接近 0 dB，與距離無關。實現的 SIR 會被計算並回報：

```
realized_SIR = 10 · log10( Σ fg² / Σ itf² )
```

真正的距離位準定律需要 `distance_level`。Noise-suppression recipe 中啟用的
`mix_mode` 會在載入 config 時被拒絕（dataset 建構子也會再檢查一次）。

#### 4.3 跳過 `mix_mode` 的列

帶有真實遠場干擾者的列（real-far 列，以及 real-far 池有載入時的 real-near 列）一律
使用 hard SIR：兩個以不同方式錄製與正規化的訊號之間的位準比沒有物理對應，套用
幾何公式只會得到沒有意義的數字。

#### 4.4 Session 列

Session 列經由 `SessionRowBuilder.mix` 混音：同樣是 `add_bg_noise`，但使用 session
自己抽的 SIR（§1.5），之後在混音上加上強制的收音底噪。

### 5. 回聲殘餘

`augmentation_speech.echo_playback` 模擬裝置自身喇叭經上游 AEC 後仍漏進麥克風的
聲音。另一段語句經過**同一個房間**的一條通道（`distance_range_override =
distance_range`，預設 0.2–1.0 m），以低於當下混音 `U(erle_db_range)` dB 的位準
加入（預設 20–35 dB）：

```
noisy = add_bg_noise(noisy, [echo], snr_list=[erle_db])
```

ERLE（echo return loss enhancement）是 AEC 消除的回聲能量，以它當 SNR 的意義是
「AEC 之後還剩多少回聲」。回聲永不進入目標、需要 source-level reverb（它需要該列
房間的一條通道），其通道 metadata 不列入 RIR lineage。使用 pre-generated bank 時，
回聲通道以 `interferer` 角色請求，因此來自遠場池（有帶內通道就用帶內，否則用最接近
帶中心的）；真正短的裝置喇叭到麥克風路徑需要 on-the-fly 模擬器。

### 6. 列開頭的環境段

`augmentation_row_initial_ambient` 讓一列以場景聲音開場。以機率 `prob`，一段長
`U(lead_seconds_range)` s（預設 1–4 s）的開頭從每個語音成分——混音、目標與背景
參照——中遮掉，邊緣以 `fade_ms`（預設 50 ms）的 raised-cosine 漸變；若開頭段會讓
該列剩下不到 0.5 s 則跳過。它在變速與整段混音 reverb 之後（時間已定）、噪音 stage
之前執行，噪音 stage 再以該列自己的噪音與底噪填滿開頭段；噪音沒有觸發的列得到
靜音的開頭，這也是真實的串流起始情境。它同樣套用在 keep 與 suppress 列上，因此
開頭段不帶任何標籤資訊。

### 7. 噪音來源

`NoiseStage`（`augmentation_noise`）加入三個非語音來源，因為各自回答不同的問題。

#### 7.1 錄音噪音（相對混音）

以機率 `prob`，把一段噪音以 `snr_range` 均勻抽出的 SNR 混入；設定 `snr_bands` 時
改為分段均勻（先依 `prob` 選一個帶，再於帶內均勻抽）。SNR 相對於當下混音的整列
RMS（前景、干擾者、回聲）。來源：`noise_folder`（每個檔案機率相同）或
`noise_sources`（先依 `weight` 選語料，再於其中均勻選檔案）。

* **Dynamic type**（錄音噪音觸發的列上機率為 `prob / 4`）：兩段不同的噪音各自
  RMS 正規化、串接後再正規化，讓噪音在一列之內變換。
* **Room coloring**（`room_coloring.prob`，只在有房間 scene 的列上）：每段噪音先
  經過**同一個房間**的一條通道（角色 `interferer`）。沒有它，乾的噪音配上有殘響的
  語音會讓模型免費得知誰是誰。卷積在 SNR 縮放之前，因此 SNR 定義在上色後的噪音上。

回報的 `noise_snr` 就是這個 SNR；該來源未觸發時為 NaN。

#### 7.2 白噪（相對混音）

在錄音噪音觸發且未走 dynamic type 的列上，以機率 `prob_white_noise` 加入高斯
白噪，SNR 取自 `white_noise_snr_range`，相對於已含錄音噪音的混音。它提供錄音噪音池
（多為具體場景）不一定涵蓋的廉價寬頻覆蓋。

#### 7.3 絕對收音底噪

```
level_dbfs ~ U(absolute_floor.level_dbfs_range)          （預設 −55 到 −35）
noisy ← noisy + randn_like(noisy) · 10^(level_dbfs / 20)
```

位準**不隨語音縮放**。麥克風自雜訊與房間本底噪音不管誰在說話都在同一個位準，
SNR 相對式的來源無法表達這件事。它以 `absolute_floor.prob` 觸發，與錄音噪音的
擲點無關（但 `augmentation_noise` 區塊本身必須啟用）。

「dBFS」是抽樣時的位準而非交付時的位準：[device chain](device_chain.zh-TW.md) 末端
的轉換器會對整列做增益調度，因此在一列被調低 6 dB 時，抽在 −45 dBFS 的底噪交付時
是 −51 dBFS。這對它模擬的對象是正確的——電容自雜訊與房間本底位於前級放大器上游，
會跟著前級增益變化。轉換器自身的電子雜訊則不會，必須加在 A/D stage 之後；它約在
−90 dBFS，遠低於這個範圍，因此沒有對應的 stage。

#### 7.4 順序與共同規則

**錄音噪音 → 白噪 → 收音底噪**，全部在 device chain 之前，因此底噪也像語音一樣
經過裝置的頻率響應。三者都只加在混音上、從不加在目標上：目標中有噪音等於要求
模型重現它。

## 工程面

### Config 對照

| 區塊 | Schema | 內容 |
|---|---|---|
| `augmentation_speech` | `SpeechAugmentation` | `prob`、`add_n_cases`、`snr_range`（hard SIR，dB）、`is_target` |
| `augmentation_speech.media_voice` | `MediaVoiceConfig` | `prob`、`hp_cutoff_range`、`lp_cutoff_range`（Hz）、`compress_power_range` |
| `augmentation_speech.overlap_control` | `OverlapControlConfig` | regime 機率與範圍、`fill_on_silent_range`、`fade_samples`、`turn_taking_prob`、輪長、gap/overlap 範圍、`far_first_prob` |
| `augmentation_speech.echo_playback` | `EchoPlaybackConfig` | `prob`、`distance_range`（m）、`erle_db_range`（dB） |
| `augmentation_speech.mix_mode` | `MixModeConfig` / `MixModeEntry` | 模式清單：`name`、`prob`、`physical`、`distance_level`、`sir_range`、`jitter_db`（僅 voice isolation） |
| `augmentation_target_absent` | `TargetAbsentAugmentation` | `prob`、`force_interferer` |
| `augmentation_realfar` | `RealFarAugmentation` | `pool_manifest`、`prob`、`lone_far_prob`、`turn_taking_prob`、`stitch_to_length` |
| `augmentation_realnear` | `RealNearAugmentation` | `pool_manifest`、`prob`、`turn_taking_prob`、`stitch_to_length` |
| `augmentation_session_rows` | `SessionRowsConfig` | `enabled`、`prob`，以及腳本、距離、SIR、底噪與配對 knob（§1.5） |
| `augmentation_row_initial_ambient` | `RowInitialAmbientAugmentation` | `prob`、`lead_seconds_range`（s）、`fade_ms` |
| `augmentation_noise` | `NoiseAugmentation` | `prob`、`noise_folder` / `noise_sources`、`snr_range`、`snr_bands`、`prob_white_noise`、`white_noise_snr_range`、`room_coloring`、`absolute_floor` |
| `vad_label` | `VadLabelConfig` | gating 與標籤使用的幀網格 |

Real 列與 session 區塊只存在於 voice-isolation 的 recipe schema；config model 禁止
未知的 key，因此任務不消費的區塊會在載入 config 時被拒絕。Curriculum 可以在 epoch
之間調整 `prob` 值；區塊改變時 dataset 會重新推導各 stage
（`rebind_augmentation_blocks`）。

### 順序

```
載入並裁切前景
→ row 計畫（session / real-near / real-far / target-absent）
→ 前景通道（source-level 擲點、房間 scene、full + 目標 RIR）
→ 干擾者（取樣 → media 上色 → RIR 或真實錄音）
→ overlap gating → 加總 → 混音（hard SIR / mix_mode / session）→ is_target 複製
→ target-absent 減除 → 回聲殘餘 → 防削波
→ 變速 → 整段混音 RIR → 列開頭環境段
→ 噪音（錄音 → 白噪 → 底噪）→ VAD 參照快照
→ device chain（+ 選用的 paired view）→ 裁到列長、標籤
```

完整的 stage 表與每個區塊的 RNG 行為見[工程契約](engineering_contract.zh-TW.md)。

### RNG 細節

* **Bernoulli regime 的抽樣次數依 regime 而異**。`_draw_overlap_rate` 擲一次選
  regime，只有兩個非零 regime 會再抽一個值。因此交換 `no_overlap_prob` 與
  `high_overlap_prob` 不只是重新加權分佈：同一個 seed 會產生不同的列，因為下游的
  流已經錯位。
* **沒有 gate 任何東西的列不抽任何值**：gating 關閉、沒有 VAD 區塊、沒有干擾者、
  目標無聲，或 `turn_taking_prob` 為 0，都會在擲點之前返回。
* **Session 列以固定順序抽樣**，順序列在 `session_rows.py` 的模組 docstring 中；
  配對列只 seed 來源素材的 scope，之後還原各個流。

### 陷阱

* 沒有啟用 `vad_label` 時 `overlap_control` 不起作用：gating 的 labeler 由該區塊
  建立。
* `is_target: true` 會把混好的語音複製成目標（noise suppression 的用法：所有說話者
  都是目標）。它與 voice isolation 的每個近/遠機制相矛盾，不要併用。
* VAD 參照是在變速、整段混音 reverb 與開頭環境段之後（時間與混音一致）、device
  chain 之前（標籤在乾淨語音上計算；目標從不帶噪音）對目標做的快照。
* 除了 session 列（`speed_perturb_companions`）之外，`background_speech_reference`
  是變速前的快照；在有變速的一般列上，由它推得的逐幀標籤會差一個變速倍率。
* 混音峰值超過 1 時，防削波以同一係數同時除混音與目標，保持兩者比例；
  `background_speech_reference` 不會跟著縮放。
* `far_target`（縮放後的干擾者匯流排）沒有任何 loss 使用；它是遠場語音洩漏的評估
  輸出（`egs/voice_isolate/scripts/eval_indomain.py`）。
* Real-near 列沒有房間。若 real-far 池沒有載入，它的干擾者會退回為對齊但未卷積的
  合成語料語音——真實錄音配上乾的干擾者。啟用 `augmentation_realnear` 時也要啟用
  `augmentation_realfar`（其 `prob` 可以是 0）。
* 啟用整段混音 reverb 時，`skip_whole_mix_reverb` 是唯一防止 real-near 或 session
  列被二次卷積的機制；不要繞過它。
