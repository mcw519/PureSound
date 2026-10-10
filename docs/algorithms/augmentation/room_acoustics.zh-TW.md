# 房間聲學（RIR 軸）

English version: [room_acoustics.md](room_acoustics.md)

房間把一個乾聲源變成麥克風收到的訊號。本頁說明這個轉換的物理模型、pipeline
從哪裡取得房間脈衝響應（RIR），以及卷積步驟除了卷積之外對訓練對做了什麼。

程式碼：`puresound/audio/impulse_response.py`（`wav_apply_rir`、
`compute_drr_db`）、`puresound/audio/augmentation.py`
（`AudioEffectAugmentor.apply_rir`）、`puresound/audio/room_simulator.py`
（`RoomImpulseResponseSimulator`）、`puresound/audio/rir/bank/loader.py`（bank
loader）、`puresound/dataset/dynamic_base.py`（逐聲源 reverb helper）。

## 演算法面

### 1. 線性非時變模型

小振幅下波動方程式是線性的；若房間幾何與邊界不隨時間變化，系統也是非時變
的。「房間 + 收音」因此是一個 LTI 系統，由單一脈衝響應 `h[n]` 描述，麥克風
訊號是乾聲源與它的卷積：

```
y[n] = (h * x)[n] = Σ_k h[k] · x[n − k]
```

一條 `h[n]` 就能把任何乾語音放進那個房間的那個位置，不需要錄音。

**涵蓋**：傳播延遲、直達聲、早期反射、晚期殘響、房間模態造成的頻率響應、
聲源與麥克風的線性頻率響應。

**不涵蓋**：

| 現象 | 原因 | 改由何處建模 |
|---|---|---|
| 喇叭非線性、AGC、壓縮 | 非線性 | [位準與動態](level_dynamics.zh-TW.md)、[device chain](device_chain.zh-TW.md) |
| 說話者移動、AEC 收斂 | 非時變不成立 | 卷積出的通道是靜止的說話者；session 列在輪次之間換通道（[場景構成](scene_construction.zh-TW.md) §1.5） |
| 加性雜訊 | 不是通道效應 | 噪音 stage（[場景構成](scene_construction.zh-TW.md) §7） |
| 空氣紊流、溫度梯度 | 非時變不成立 | 室內距離下可忽略 |

### 2. RIR 的時間結構

```
振幅
 │    ┃ 直達聲（單一脈衝，t = d/c）
 │    ┃
 │    ┃  ╿ ╿  早期反射（可數的鏡面反射，路徑可個別辨識）
 │    ┃  │ │ ╿ ╷
 │    ┃  │ │ │ │ ╷╷.,·-.,_ 晚期殘響（密度高到統計上等同指數衰減的高斯噪音）
 └────┸──┴─┴─┴─┴─┴────────────────────────→ 時間
      0        ~50–80 ms
```

* **直達聲**：延遲 `t = d/c`（`c ≈ 343 m/s`），振幅 `∝ 1/d`。
* **早期反射**：經一次或少數幾次牆面鏡面反射。約 50 ms 內的反射與直達聲融合
  成單一聽覺事件（precedence 或 Haas effect）並提升清晰度，這是清晰度指標
  C50 以 50 ms 為分界的原因（ISO 3382-1）。
* **晚期殘響**：路徑不再可辨，聲場趨近擴散狀態，空間上均勻、時間上指數衰減。
  RT60 是聲源停止後位準下降 60 dB 所需的時間。

這三段決定了下面兩組時間窗：目標窗（§4）與 DRR 窗（§5）。

### 3. `wav_apply_rir` 在卷積前後做的事

`wav_apply_rir(wav, impaulse, sample_rate, rir_mode)` 接收 `wav` `[C, L]`、
RIR `[R, L_h]` 與 `rir_mode ∈ {full, early, direct}`，回傳與輸入同長 `L` 的
訊號（多通道 RIR 要求 `wav` 為單通道，並回傳 `R` 個通道）。FFT 卷積前後有三個
不中性的步驟。

#### 3.1 截窗

```
peak   = argmax h              （最大的正值樣本）
direct: h ← h[:, 0 : peak + 6 ms]
early:  h ← h[:, 0 : peak + 50 ms]
full:   h 不變
```

#### 3.2 峰值正規化移除距離的位準線索

```
h ← h / max|h|
```

目的是讓濕訊號維持在輸入訊號的量級。每個聲源各用自己的單通道 RIR 卷積，因此
每個聲源都以自己的直達聲峰值正規化。自由場中直達聲峰值正比於 `1/d`：磁碟上
3 m 的 RIR 峰值比 1 m 的低約 9.5 dB，正規化後兩者峰值都是 1.0。**距離造成的
位準差被移除。**

影響：

* 混音不繼承 `1/r` 位準定律；近場與遠場語者的相對音量由混音階段決定
  （[場景構成](scene_construction.zh-TW.md) §4）。
* 殘存的距離線索是 DRR、衰減尾形與頻譜傾斜（[距離線索](distance_cues.zh-TW.md)）。
* `distance_level` 混音模式明確把位準線索加回來
  （[距離線索](distance_cues.zh-TW.md) §5）。
* `mix_mode: physical` 與 recipe 中所有 hard-SIR 範圍都以這個正規化為前提；
  改動它，這些設定的意義全部跟著改變。

#### 3.3 移除傳播延遲

卷積後輸出取 `y[delay : delay + L]`，其中 `delay` 為第一個 RIR 通道的
`argmax|h|`。濕訊號與乾訊號逐樣本對齊，這是訓練對需要的；代價是聲源之間的
相對飛行時間被移除：3 m 的聲源本應比 0.5 m 的晚約 7.3 ms 到達
（`2.5 m / 343 m/s`），對齊後兩者同時開始。時間線索只剩 RIR 內部（峰值之後）
的部分。

#### 3.4 三種模式共用同一個正規化係數

截窗保留直達聲峰值，因此同一條 RIR 在 `full`、`early`、`direct` 下的
`max|h|` 相同。`full` 混音與 `early` 目標的直達聲位準完全一致，只差殘響能量。
「目標是混音的近場成分」這個前提由此成立；若截窗後重新正規化，目標與混音之間
會多出一個未知增益，模型除了去殘響還得校正增益。

### 4. 目標窗

`augmentation_reverb.target_rir_type` 決定目標保留多少殘響：

| 設定 | 目標 | 模型被要求做的事 |
|---|---|---|
| `full` | 完整 RIR 卷積 | 分離與降噪，不去殘響 |
| `early` | `peak + 50 ms` | 去除晚期殘響，保留早期反射 |
| `direct` | `peak + 6 ms` | 只保留直達聲與最貼近的反射 |
| `anechoic` | 乾聲源 | 完全去殘響 |

目標的建構方式是以混音 RIR 的 `rir_id` 重取同一條脈衝響應、改用選定模式卷積
（`anechoic` 直接回傳乾聲源）。`early` 依循 §2 的感知分界：50 ms 內的反射有助
可懂度，把它們當成失真只會讓任務變難。`direct` 的 6 ms 約對應 2 m 路徑差，只
保留緊貼直達聲的反射，例如桌面或地面。

### 5. DRR：直達與殘響的能量比

```
DRR_dB = 10 · log10( Σ_{peak ≤ n < peak+w} h[n]²  /  Σ_{n ≥ peak+w} h[n]² )
```

`compute_drr_db(rir, sample_rate, direct_window_ms=2.5)` 以 `peak = argmax|h|`、
`w = round(2.5 ms · fs)` 實作。峰值之前的能量不計；尾巴為空時（消聲或被截斷的
RIR）回傳 `+inf`。模擬器與 bank 送出的每條 RIR 都在 metadata 中帶有 `drr_db`。

**為什麼 DRR 能量測距離**。直達聲能量隨 `1/d²` 衰減，而擴散場理論中穩態殘響
能量密度在房間內均勻、與聲源位置無關。因此

```
DRR_dB(d) = 20 · log10(r_c / d)
```

每倍距離約下降 6 dB；臨界距離 `r_c`（DRR = 0 dB）為
`r_c ≈ 0.057 · sqrt(Q · V / T60)` m，其中 `V` 為體積（m³）、`T60` 單位為 s、
`Q` 為聲源指向性因子（Kuttruff，*Room Acoustics*）。辦公室與會議室的 `r_c` 約為
一公尺量級，recipe 的近場與遠場距離範圍正是以此為界。這條關係也是
deterministic DRR contrast 寫成 `log10(d)` 形式的原因
（[距離線索](distance_cues.zh-TW.md) §3）。

**兩組窗長不可互換**。目標建構用 6 ms / 50 ms（感知融合）；DRR 量測用 2.5 ms
（只取直達聲本身，讓量測值不受第一批反射落點影響）。DRR contrast 的尾巴分界
也取 2.5 ms，使其位移恰好落在這個量測上。

### 6. RIR 的來源

`augmentation_reverb` 在 dataset 建構時（`DynamicBaseDataset.init_augmentor`）
選定一個來源，`apply_rir` 從中取用：

| Config | 來源 |
|---|---|
| `simulator.used: true`、`simulator.pregenerated.used: true` | Pre-generated bank |
| `simulator.used: true`、無啟用的 `pregenerated` | On-the-fly 影像源模擬器 |
| 無 `simulator` 或已關閉 | RIR WAV 資料夾（`rir_folder`） |

**(a) Pre-generated bank**（`simulator.pregenerated`）。一個 bank scene 是一個
房間；前景與每個干擾者從同一個 scene 取各自的通道，房間一致性由結構保證。依
角色選通道：`foreground` 取近場標籤（`near_labels`，預設 `near_0, near_1`），
`interferer`/`media`/`echo` 取遠場標籤（`far_0, far_1, far_2`），其他角色取
全部通道；同一個 scene 在還有未用通道時不會重複發同一條。
`distance_range_override` 只保留距離帶內的通道，帶內沒有時退回最接近帶中心
的通道。三種 loader：

* `PreGeneratedRoomBank`——已渲染房間的資料夾（`bank_type: room`）。
* `PreGeneratedReleaseBank`——release manifest 中的一個 recipe
  （`bank_type: release`，設了 `recipe_id` 時為預設）。建構時稽核 release，
  可要求 production certificate（`require_production`），拒絕尚未 ready 的
  recipe，並依 recipe 凍結的權重依序抽 origin、variant、item。它的 `split` 必須
  等於 dataset 的 pipeline role（由 dataset 以 `usage_role` 注入），確保
  train、validation、test 的 RIR 互不重疊。
* `UnionRoomBank`——`banks:` 列表以各成員的 `weight` 合成一個池（權重是抽樣
  機率而非 item 數）。curriculum 可在 epoch 之間調整權重（`set_weights`）。

Bank 格式與生成：[RIR bank](../audio/rir_bank_v2.zh-TW.md)；loader API：
[bank loaders](../audio/rir_bank.zh-TW.md)。

**(b) On-the-fly 模擬器**（`RoomImpulseResponseSimulator`）。每次呼叫依抽樣的
房間尺寸、RT60 與距離即時生成（§7–§8）。參數連續可控且不占儲存空間，代價是
每列的生成時間與僅限 shoebox 幾何。

**(c) 資料夾 RIR**。檔案均勻抽取。不帶 metadata（沒有距離、RT60、角色），所有
依賴幾何的手法都會跳過它，只能用於整段混音的殘響。

**Cache**。bank 與模擬器的 RIR 以 `rir_id`（`bank-N`、`simulated-N`）存入 32 條
的 LRU cache；資料夾 RIR 以檔案 key 重新讀取。目標以同一個 id 重取同一條脈衝
響應、只改變截窗（§3.4、§4）。因此 DRR contrast 與 direct smear 必須在 RIR 進
cache 之前套用；若在之後套用，混音與目標會由兩條不同的脈衝響應建構。

### 7. 影像源法

`rir_generator`（Habets 對 Allen & Berkley 影像源法的實作）計算矩形房間的
RIR。一次牆面反射等同於牆後的一個鏡像聲源；重複鏡像得到更高階反射：

```
h(t) = Σ_i  (β^{n_i} / (4π·d_i)) · δ(t − d_i/c)
```

`d_i` 是鏡像聲源到接收器的距離、`n_i` 是該路徑的反射次數、`β` 是牆面反射
係數；`1/d_i` 是球面擴散，`β^{n_i}` 是累積吸收。

**由 RT60 到 β**。config 給的是 RT60 而非 `β`。假設六面共用一個吸收係數 `α`，
由 Sabine 公式

```
RT60 = 0.161 · V / (S · α)
```

（`V` 單位 m³、總表面積 `S` 單位 m²）解出 `α`，再得 `β = sqrt(1 − α)`（`α` 是
能量係數，`β` 作用在振幅上）。RT60 短於 `0.161 · V / S` 時需要 `α > 1`，
generator 會拋錯。Sabine 假設擴散場，在低吸收時準確；高吸收時 Eyring 公式較
接近，因此非常乾的模擬房間其衰減只近似於要求的 RT60。

**假設與其後果**：

| 假設 | 失效條件 | 後果 |
|---|---|---|
| `β` 與頻率無關 | 真實表面高頻吸收較強 | 高頻衰減太慢，殘響偏亮 |
| 只有鏡面反射 | 家具與粗糙表面造成散射 | 晚期尾巴缺乏擴散場統計性質 |
| 矩形房間 | 其他幾何、家具遮擋 | 沒有繞射與遮蔽 |
| 點聲源、全指向接收器 | 口部與麥克風有指向性 | 沒有方向相關的頻譜著色 |
| 無空氣吸收 | 長路徑的高頻 | 遠場高頻能量偏多 |

這些限制是 hybrid 生成器存在的原因：低頻用波動法取得正確的模態行為，高頻用
頻率相依材質的幾何法，兩者以 crossover 拼接（[hybrid RIR](../audio/hybrid_rir.zh-TW.md)）。

`hp_filter: true` 套用 generator 的高通，移除影像源總和的直流成分；
`order: -1` 表示不限制反射階數；`nsample` 是 RIR 長度，預設為 `RT60 · fs` 個樣本。

### 8. 場景幾何取樣

`sample_scene` 抽房間與接收器：每一軸尺寸從 `room_dim_range` 均勻抽樣、RT60
從 `rt60_range` 均勻抽樣、接收器在房內均勻抽樣且距每面牆至少
`receiver_margin`。`generate` 每次呼叫再依角色放置一個聲源。

**依角色決定距離範圍**：有 `distance_range_override` 時用它；`foreground` 用
`foreground_distance_range`；`media` 用 `media_distance_range`，沒有時退回
`interferer_distance_range`；`interferer` 用 `interferer_distance_range`；其餘用
`source_receiver_distance_range`。

**階段一：房內均勻取樣加拒絕**。在房內（距牆至少 `source_margin`）均勻取一點，
距離落在範圍內就採用；最多 64 次。這保持空間分佈均勻，但範圍窄時（例如
`[0.3, 0.5]` m）符合條件的球殼只占房間極小比例，拒絕幾乎都失敗。

**階段二：球殼取樣**。直接構造一個距離符合的點（均勻半徑、均勻方向），只對房間
邊界做拒絕；最多 256 次。若球殼與房間完全沒有交集，則朝房內最遠的角落放置，
並把半徑夾到可用範圍內——取最接近的可達距離。

球殼取樣在房內不是空間均勻的（朝向近牆的方向較常被拒絕）。這是可接受的，因為
實現的距離會寫入 metadata，並被下游當作真值使用（deterministic DRR contrast、
`distance_level` 混音、純量標籤）；距離標籤錯誤比擺位偏差更糟。

**media 貼牆放置**。`media` 代表電視或喇叭，通常靠牆擺放。階段一會把聲源貼到
距 x 或 y 牆 `source_margin + U(0, media_wall_offset_max)` 之內。牆面鏡像因此
離聲源很近，一道強反射緊接在直達窗之後到達，DRR 低於同距離自由站立的說話者。
階段二放棄貼牆：距離語義優先於擺位。

回傳的 metadata：`room_dim`、`receiver`、`source`、`rt60`、`source_role`、
`source_receiver_distance`（m）與 `drr_db`。

### 9. 延伸閱讀

| 文件 | 內容 |
|---|---|
| [Hybrid RIR](../audio/hybrid_rir.zh-TW.md) | 波動法低頻加幾何法高頻、crossover |
| [RIR scene](../audio/rir_scene_v2.zh-TW.md) | 材料優先的場景 schema |
| [RIR bank](../audio/rir_bank_v2.zh-TW.md) | 生產用 bank 格式：生成、QC、split、release |
| [Bank loaders](../audio/rir_bank.zh-TW.md) | §6 使用的訓練端 loader |
| [RIR metrics](../audio/rir_metrics.zh-TW.md) | RT60、DRR、EDC 等量測 |
| [Spatial RIR](../audio/spatial_rir.zh-TW.md) | 陣列、FOA 與雙耳渲染 |
| [Audio 與 RIR 索引](../audio/index.zh-TW.md) | 其餘 RIR 文件：晚期場、阻抗、模態阻尼、校準 |

## 工程面

### Config 對照

| 區塊 | Schema | 消費者 |
|---|---|---|
| `augmentation_reverb` | `ReverbAugmentation` | `init_augmentor`；`NoiseSuppressionDataset.__getitem__` 的整段混音分支 |
| `augmentation_reverb.simulator` | `RoomSimulatorConfig` | `RoomImpulseResponseSimulator` |
| `augmentation_reverb.simulator.pregenerated` | `PreGeneratedBankConfig` | Bank loader（room / release / union） |
| `augmentation_reverb.drr_contrast` | `DrrContrastConfig` | [距離線索](distance_cues.zh-TW.md) §3 |
| `augmentation_reverb.direct_smear` | `DirectSmearConfig` | [距離線索](distance_cues.zh-TW.md) §4 |

只有 recipe 寫出的 key 會轉交給模擬器與 bank 的建構子，其餘沿用它們自己的預設
（模擬器：`receiver_margin` 0.4 m、`source_margin` 0.4 m、
`media_wall_offset_max` 0.2 m、`sound_speed` 343 m/s、`order` −1、
`hp_filter` true；bank：`drr_window_ms` 2.5、`cache_size` 64）。

```yaml
augmentation_reverb:
  used: true
  prob: 1.0
  target_rir_type: early
  simulator:
    used: true
    source_level: true          # 每個聲源一條通道（近/遠場列的前提）
    pregenerated:
      used: true
      banks:
        - {name: core, weight: 0.5, bank_type: room, folder: /path/to/bank/core}
        - {name: wide, weight: 0.5, bank_type: room, folder: /path/to/bank/wide}
```

### 逐聲源路徑與整段混音路徑

* **Source-level**（`simulator.source_level: true`）：每列在
  `should_apply_source_level_reverb()` 擲一次 `torch.rand(1) < augmentation_reverb.prob`。
  命中時該列抽一個房間 scene，前景取一條 `foreground` 通道（混音用 `full`、
  目標用 `target_rir_type`），每個干擾者從同一個 scene 取一條 `interferer` 或
  `media` 通道。
* **整段混音**：沒有 source-level reverb 的列（且 row plan 未設
  `skip_whole_mix_reverb`）在混音與變速之後，以同一個 `prob` 再擲一次，命中就把
  完成的語音混音整段卷上一條 RIR。RIR 取自已設定的來源，角色為 `source`
  （bank 的全部通道；模擬器的預設距離範圍）。多通道結果只保留第 0 通道。
* 兩條路徑不會在同一列同時發生。`source_level: true` 時，第一次沒擲中的列仍可能
  在第二次擲中而走整段混音路徑。
* 關閉的區塊不消耗隨機性（[工程契約](engineering_contract.zh-TW.md)）。RIR 的
  選取會用到 NumPy（模擬器幾何）與 Python `random`（bank 的 scene 與通道），
  因此 seeded item 必須重設三個亂數產生器。
* RIR lineage（`rir_release_id`、`rir_variant_id`、`rir_renderer_profile_id`
  及 `puresound/task/ns.py` 中其餘的 `RIR_PROVENANCE_KEYS`）以字串隨每列輸出，
  僅供追溯；沒有 loss 讀取它。

### 陷阱

* 資料夾 RIR 沒有 metadata，所有依賴幾何的手法在這條路徑上都會靜默跳過。距離
  相關的 knob 看起來沒效果時，先確認 RIR 來源。
* Cache 只有 32 條。若某個流程在前景與其目標重取之間插入大量其他 `apply_rir`
  呼叫，條目可能被淘汰，目標會拿到新抽的 RIR。目前的呼叫順序不會發生。
* 截窗以 `argmax h` 為錨，正規化與對齊以 `argmax |h|` 為錨。直達聲是最大正值
  樣本時兩者一致，模擬 RIR 即是如此；極性反轉的 RIR 其目標窗會錨在較晚的樣本上。
* `_last_rir_meta` 只保留最近一次呼叫的 metadata，供 REPL 檢視用。函式庫程式
  一律從回傳值 `RirApplied.detail.metadata` 取 metadata；只要中間有其他卷積，
  這個側通道就會錯誤歸屬。
