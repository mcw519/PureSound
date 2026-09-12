# 房間聲學（RIR 軸）

English version: [`room_acoustics.md`](room_acoustics.md)

房間把一個乾聲源變成麥克風收到的訊號。本章說明這個轉換的物理模型、pipeline
如何實作它，以及實作中每個看似平凡的步驟對訓練資料的實際影響。

## 演算法面

### 1. 線性非時變模型

聲波在空氣中的傳播由波動方程式描述。在小振幅（線性聲學）條件下，波動方程式
是線性的，且若房間幾何與邊界條件不隨時間變化，系統也是非時變的。因此
「房間 + 收音」構成一個 LTI 系統，可以用單一脈衝響應 `h[n]` 完整描述。麥克風
收到的訊號是乾聲源與該脈衝響應的卷積：

```
y[n] = (h * x)[n] = Σ_k h[k] · x[n − k]
```

這條式子是整個 RIR 軸的基礎。它的意義是：只要取得一條 `h[n]`，任何乾語音都
可以被「放進」那個房間的那個位置，不需要真的去錄音。

**模型涵蓋**：傳播延遲、直達聲、早期反射、晚期殘響、房間模態造成的頻率
響應、麥克風與喇叭的線性頻率響應。

**模型不涵蓋**：

| 不涵蓋的現象 | 原因 | 後果 |
|---|---|---|
| 喇叭非線性失真、AGC、壓縮 | 違反線性 | 需要 ch4/ch5 的非線性 stage 個別建模 |
| 說話者移動、AEC 收斂 | 違反非時變 | 卷積出的通道是「靜止的人」 |
| 加性雜訊 | 不是通道效應 | 由 ch7 的噪音 stage 單獨加入 |
| 空氣紊流、溫度梯度 | 違反非時變 | 對室內短距離可忽略 |

合成資料與真實錄音的差距落在「LTI 可解釋」與「不可解釋」的哪一側，是實驗
問題，量測紀錄保留於內部。

### 2. RIR 的時間結構

一條典型的房間脈衝響應在時間軸上分成三段，分段依據是反射路徑的密度：

```
振幅
 │    ┃ 直達聲（單一脈衝，t = d/c）
 │    ┃
 │    ┃  ╿ ╿  早期反射（可數的鏡面反射，路徑清晰可辨）
 │    ┃  │ │ ╿ ╷
 │    ┃  │ │ │ │ ╷╷.,·-.,_ 晚期殘響（反射密度過高，統計上等同高斯噪音的指數衰減）
 └────┸──┴─┴─┴─┴─┴────────────────────────→ 時間
      0        ~50–80 ms
```

* **直達聲**：從聲源直線到達，延遲 `t = d/c`（`d` 距離、`c ≈ 343 m/s`），
  振幅隨 `1/d` 衰減（球面波的反平方定律作用在能量上，振幅是其平方根）。
* **早期反射**：經一次或少數幾次牆面反射到達。在感知上，50 ms 內到達的
  反射會與直達聲融合成單一聲音事件而不被聽成回聲（precedence effect，又稱
  Haas effect），並且提升語音清晰度——這是語音清晰度指標 C50 以 50 ms 為
  分界的原因。
* **晚期殘響**：反射次數多到路徑不可辨識，聲場趨近擴散（diffuse）狀態，
  能量在空間中近似均勻分佈，時間上呈指數衰減。衰減速率以 RT60 描述：
  聲源停止後聲壓級下降 60 dB 所需的時間。

這個三段結構直接決定了 pipeline 的兩組時間窗（見 §4、§5）。

### 3. 卷積實作的三個副作用

`puresound/audio/impulse_response.py::wav_apply_rir` 執行卷積，但在卷積前後
做了三件在數學上不中性的處理。這三件事對訓練資料的影響比卷積本身更大。

#### 3.1 峰值正規化移除了距離的位準線索

```
h ← h / max|h|
```

目的是讓濕訊號停在與輸入訊號相近的量級，避免下游位準失控。代價是抹掉了
距離的位準資訊。

推導：自由場中距離 `d` 處的聲壓 `p ∝ 1/d`，而 RIR 的直達聲峰值正比於該
聲壓。因此磁碟上一條 3 m 的 RIR 峰值大約是 1 m 的 1/3（約 −9.5 dB）。逐通道
除以各自的峰值之後，所有通道的直達聲都變成 1.0——**距離的位準差被完全
移除**。

影響範圍：

* 混音不繼承 1/r 位準定律。近場與遠場語者在混音中的相對音量由 SIR 抽樣
  決定，與它們的幾何距離無關。
* 殘存的距離線索只有 DRR、衰減尾形、頻譜傾斜（完整清單見
  [ch3](distance_cues.zh-TW.md)）。
* 要取回位準線索，必須在混音層明確重建，這是 `distance_level` 混音模式的
  由來（ch3 §5）。
* `mix_mode: physical` 與所有 hard-SIR 範圍的設定都建立在這個前提上。修改
  這一行，這些設定的語義全部連動改變。

#### 3.2 傳播延遲移除了聲源間的到達時差

卷積後取 `y[delay : delay + L]`，其中 `delay = argmax|h|`（直達聲峰值位置）。

這讓濕訊號與乾訊號逐樣本對齊，方便建構訓練對。副作用是**各聲源之間的相對
飛行時差被移除**：3 m 的聲源本應比 0.5 m 的晚約 7.3 ms 到達
（`(3−0.5)/343 s`），對齊後兩者的語音起始點一致。時間結構線索只剩下 RIR
內部（峰值之後）的部分。

#### 3.3 三種 rir_mode 共用同一個正規化係數

`early` 與 `direct` 模式的做法是把 `h` 截短再卷積：

```
direct: h[0 : peak + 6 ms]
early:  h[0 : peak + 50 ms]
```

截短保留了直達聲峰值，所以 `max|h|` 不變，三種模式對同一條 RIR 得到**同一個**
正規化係數。結果是 `full` 混音與 `early` 目標的直達聲位準完全一致，兩者只差
晚期殘響能量。

這個性質是「目標是混音的近場成分」這個訓練假設成立的基礎。若截短後重新
正規化，目標與混音之間會多出一個未知的增益差，模型被要求同時做去殘響與
增益校正。

### 4. 目標訊號的時間窗選擇

`target_rir_type` 決定訓練目標保留多少殘響：

| 設定 | 目標訊號 | 模型被要求做的事 |
|---|---|---|
| `full` | 完整 RIR 卷積 | 不去殘響（只做分離/降噪） |
| `early` | `peak + 50 ms` | 去除晚期殘響，保留早期反射 |
| `direct` | `peak + 6 ms` | 只保留直達聲，接近完全去殘響 |
| `anechoic` | 乾訊號 | 完全去殘響 |

`early` 是預設選擇，理由是 §2 的感知分界：50 ms 內的早期反射對聽感與清晰度
有正面貢獻，把它們列為「要去除的失真」既不必要也讓任務變難。`direct` 的
6 ms 窗約對應 2 m 的路徑差，只涵蓋緊貼直達聲的地面或桌面反射。

### 5. DRR：直達與殘響的能量比

DRR（direct-to-reverberant ratio）以能量比量化「這個聲源聽起來多近」：

```
DRR_dB = 10 · log10( Σ_{n∈[peak, peak+w]} h[n]²  /  Σ_{n>peak+w} h[n]² )
```

實作在 `compute_drr_db`，預設窗長 `w = 2.5 ms`。尾巴無能量時回傳 `+inf`
（消聲室或被截斷的 RIR）。

**為什麼 DRR 是距離的代理量**。直達聲能量隨 `1/d²` 衰減；擴散場理論指出，
在穩態下殘響場的能量密度在房間內近似均勻，與聲源距離無關。因此

```
DRR(d) ∝ (1/d²) / const   →   DRR_dB ≈ −20·log10(d) + C
```

每倍距離 DRR 下降約 6 dB，常數 `C` 由房間的吸收與體積決定。這條關係是
deterministic DRR contrast 使用 `log10(d)` 形式的物理依據（ch3 §3）。

**臨界距離**（critical distance）是 `DRR = 0 dB` 的距離，即直達與殘響能量
相等處。一般辦公室或會議室的臨界距離約在 1–2 m，這也解釋了為何近場/遠場的
決策邊界落在這個區間：越過臨界距離之後，殘響成分開始主導。

**兩組窗長不可混用**。目標建構用 6 ms / 50 ms（依感知融合分界），DRR 量測
用 2.5 ms（只取直達聲本身，讓量測對早期反射的擺位不敏感）。DRR contrast 的
尾巴分界刻意對齊 2.5 ms，理由見 ch3 §3。

### 6. RIR 的三個來源

`AudioEffectAugmentor.apply_rir` 依優先序選用其中一個，三者互斥：

**(a) Pre-generated room bank**（`pregenerated` 區塊）

離線產生好的多聲源房間 bank。一個 bank scene 對應一個房間，前景與所有干擾者
從同一個 scene 取各自的通道，因此房間一致性是結構上保證的，不需要額外約束。
`release` 型 bank 另外掛上生產契約：`recipe_id`、`split` 必須等於 dataset 的
`usage_role`（防止 train/test 用到同一批 RIR）、QC 與 production gate。
`banks:` 列表可加權聯集多個 bank（`UnionRoomBank`），用來組合不同產生器或
不同參數區間的覆蓋。

**(b) On-the-fly 影像源模擬器**（`RoomImpulseResponseSimulator`）

逐列即時生成，參數（房間尺寸、RT60、距離）由 config 的範圍抽樣。優點是
參數連續可控且不占磁碟；缺點是每列都要付生成成本，且只支援 shoebox 幾何。
唯一支援 per-role 距離覆寫（`distance_range_override`）的來源。

**(c) Folder RIR**

資料夾內的 wav 逐條隨機抽取。無 metadata（沒有距離、RT60、角色資訊），因此
只能用於 whole-mix 殘響，無法參與任何需要幾何的手法。

**共用的 cache 機制**。前兩者生成的 RIR 以 `rir_id` 存入 LRU cache（32 條）。
目標訊號透過同一個 `rir_id` 重取同一條 RIR，只改變截短窗。這是 §3.3 所述
「目標是混音的近場成分」的實作機制，也是 DRR contrast 與 direct smear 必須
在 RIR 進 cache **之前**套用的原因——若在之後套用，混音與目標會拿到兩條
不同的脈衝響應。

### 7. 影像源法的模型與假設

`rir_generator`（Habets 實作）以影像源法（Image Source Method, ISM）求解
矩形房間的脈衝響應。

**原理**。把牆面反射等效成「鏡像聲源」：一個聲源對一面牆的反射，等於在牆
另一側對稱位置放一個虛擬聲源，其訊號直接傳到接收器。多次反射對應多次鏡像。
RIR 由所有影像源的貢獻疊加：

```
h(t) = Σ_i  (β^{n_i} / (4π·d_i)) · δ(t − d_i/c)
```

`d_i` 是第 i 個影像源到接收器的距離、`n_i` 是該路徑的反射次數、`β` 是牆面
反射係數。`1/d_i` 是球面波衰減、`β^{n_i}` 是多次反射的累積吸收。

**反射係數的推導**。config 給的是 RT60，不是 `β`。轉換用 Sabine 公式：

```
RT60 = 0.161 · V / (Σ_i S_i · α_i)
```

`V` 房間體積（m³）、`S_i` 各面牆面積（m²）、`α_i` 各面吸收係數。假設六面
牆吸收係數相同，可解出單一 `α`，再由能量守恆得反射係數
`β = sqrt(1 − α)`（`α` 定義在能量上、`β` 作用在振幅上）。

Sabine 公式本身的假設是**擴散場**：聲能在房間內均勻分佈、各方向入射機率
相等。這在吸收係數低（`α < 0.2`）且房間比例接近立方時較準；長條形房間或
單面強吸收（例如只有一面牆貼吸音棉）時，Sabine 會低估 RT60。

**其他假設與失效條件**：

| 假設 | 失效條件 | 觀察到的偏差 |
|---|---|---|
| 反射係數與頻率無關 | 真實牆面高頻吸收明顯較強 | 高頻衰減比實際慢，殘響「太亮」 |
| 鏡面反射（無散射） | 家具、書架、粗糙表面造成擴散 | 晚期尾巴缺乏真實擴散場的統計性質 |
| 矩形房間 | 任何非矩形幾何、家具遮擋 | 缺少繞射與遮蔽效應 |
| 點聲源、全指向接收器 | 人的口部有指向性、麥克風有極性圖 | 缺少方向相關的頻譜著色 |
| 無空氣吸收 | 高頻在長距離明顯衰減 | 遠場高頻偏多 |

這些偏差是 hybrid RIR 生成路線存在的原因：低頻用波動方程式數值解（FDTD）
取得正確的模態行為、高頻用幾何法配上頻率相依的材質吸收，兩段以
Linkwitz–Riley crossover 拼接（見 §9 導讀）。

`hp_filter=True` 會加上高通濾波，去除 ISM 離散化產生的低頻直流假象；
`order=-1` 表示不限制反射階數（算到 RIR 長度用盡）。

### 8. 場景幾何的取樣策略

`RoomImpulseResponseSimulator.sample_scene` 先抽房間與接收器，
`_sample_source` 再依角色抽聲源位置。

**場景層**：房間三軸尺寸各自均勻抽樣、RT60 均勻抽樣、接收器位置在房間內
均勻抽樣（距牆保留 `receiver_margin`）。

**聲源層**依角色取距離範圍（`foreground_distance_range`、
`interferer_distance_range`、`media_distance_range`，可被
`distance_range_override` 覆寫），然後用兩階段策略滿足距離約束：

*階段一：房內均勻取樣 + 拒絕*。在房間內（距牆 `source_margin`）均勻取點，
檢查與接收器的距離是否落在範圍內，最多試 64 次。這個策略保持空間分佈均勻，
但當距離範圍很窄時（例如 `[0.3, 0.5]`），符合條件的球殼體積只占房間極小
比例，拒絕率過高。

*階段二：球殼直接取樣*。改為「均勻半徑 + 均勻方向」直接構造符合距離的點，
只對房間邊界做拒絕，最多 256 次。若球殼與房間幾何完全無交集（距離要求
在該房間不可能達成），則朝房內最遠的角落走，把半徑夾到可用範圍內——取
最接近的可達距離。

這個 fallback 的設計取捨值得留意：球殼取樣的空間分佈**不是**房內均勻的
（近牆的方向被拒絕比例較高）。這裡刻意選擇犧牲空間均勻性，因為距離是會被
寫入 metadata 並被下游手法（deterministic DRR contrast、`distance_level`
混音）當作真值使用的標籤，標籤錯誤比取樣偏差有害得多。

**media 角色的貼牆處理**。`media` 代表電視或喇叭，這類聲源通常靠牆放置。
實作把聲源在 x 或 y 軸上貼到牆面附近
（`source_margin + U(0, media_wall_offset_max)`）。物理效果是**一次反射的
路徑差變小**，早期反射相對直達聲更強，DRR 因此低於同距離的自由站立說話者。
注意進入階段二（球殼取樣）時會放棄貼牆，同樣是距離語義優先於擺位語義。

### 9. Hybrid RIR 生成：文件導讀

RIR 生成的深入內容已有專門文件（皆在 `docs/audio/`，含 zh-TW 版）。依閱讀
目的分組：

**生成主線**

| 文件 | 內容 |
|---|---|
| `hybrid_rir` | 低頻 FDTD（pytARD）與高頻幾何法的混合架構、crossover 設計 |
| `rir_realism_algorithm` | 演算法與程式碼的對應總表，適合當索引 |
| `multiband_fdn` | 晚期殘響的多頻帶回饋延遲網路合成 |
| `rir_late_coupling` | PathEvent 早期場與 FDN 晚期場的能量與時間耦合 |

**材質與低頻物理**

| 文件 | 內容 |
|---|---|
| `modal_damping` | 由材質阻抗推導低頻模態阻尼 |
| `impedance_priors` | 相位感知的低頻阻抗先驗 |
| `impedance_measurements` / `impedance_tube_protocol` | 阻抗量測資料與量測協定 |
| `complex_impedance_source_audit` | 阻抗數據來源審查 |
| `modal_validation` | 模態行為的驗證方法 |

**Bank 與量測**

| 文件 | 內容 |
|---|---|
| `rir_scene_v2` | 材料優先的場景 schema |
| `rir_bank` / `rir_bank_v2` | bank 讀取器 API、M6 production bank 契約 |
| `rir_measurement_campaign` | 受控房間量測與反演校準契約 |
| `spatial_rir` | 空間 RIR 與 BRIR |

**度量**

| 文件 | 內容 |
|---|---|
| `rir_metrics` | RT60、DRR、EDC 等度量的定義與實作 |
| `rir_attribution` | direct/early/late 的能量歸因方法 |

已知的實作陷阱（詳見 `hybrid_rir` 與 ）：pytARD 必須使用
有損模式，無損或 Unit 設定會讓可用頻寬上限砍半；pyroomacoustics 的 RT60
定義慣例與固定 40-sample 延遲需要對齊補償；兩段頻帶的拼接使用
Linkwitz–Riley crossover 以保證相加後振幅平坦。

## 工程面

### Config 對照

| 區塊 | Schema | 消費者 |
|---|---|---|
| `augmentation_reverb` | `ReverbAugmentation` | `dynamic_base` 初始化 + `ns.py` whole-mix 分支 |
| `augmentation_reverb.simulator` | `RoomSimulatorConfig` | `RoomImpulseResponseSimulator` |
| `augmentation_reverb.simulator.pregenerated` | `PreGeneratedBankConfig` | bank loader（room / release / union） |
| `augmentation_reverb.drr_contrast` | `DrrContrastConfig` | [ch3](distance_cues.zh-TW.md) |
| `augmentation_reverb.direct_smear` | `DirectSmearConfig` | [ch3](distance_cues.zh-TW.md) |

主要 knob：

* `target_rir_type ∈ {full, early, direct, anechoic}`——目標的殘響量（§4）。
* `simulator.source_level`——`true` 時每個聲源有自己的通道（近/遠場景的
  前提）；`false` 時整個混音共用一條 RIR。
* per-role 距離範圍——控制近/遠分佈的主要 knob。
* release bank 的 `usage_role == split` 硬約束——防止 RIR 洩漏。

### 順序與互斥

* `source_level` 列走「每聲源一通道」路徑，並**跳過** whole-mix RIR 分支。
  非 source-level 列則在混音與變速之後才卷 whole-mix RIR。兩條路徑互斥，
  由 `should_apply_source_level_reverb()` 每列擲一次決定。
* 機率擲點位於 short circuit 內部：關閉 reverb 區塊不消耗 RNG，詳見
  [ch8](engineering_contract.zh-TW.md)。
* RIR lineage（`release_id`、`variant_id`、`renderer_profile_id` 等）以字串
  形式隨 sample 發出（`RIR_PROVENANCE_KEYS`），僅供追溯，不參與訓練。

### 陷阱

* folder RIR 無 metadata，所有需要幾何的手法在該路徑上靜默跳過。這是刻意
  設計（寧可少做也不錯標），但除錯時容易誤判為 knob 沒生效——先確認 RIR
  來源。
* LRU cache 只有 32 條。若某個流程在前景與目標之間插入了大量其他
  `apply_rir` 呼叫，目標重取時可能已被淘汰，導致目標拿到新抽的 RIR。目前
  的呼叫順序不會觸發，但新增 stage 時要留意。
* `_last_rir_meta` 是 REPL 除錯用的側通道，只保留最後一次呼叫的 metadata。
  正式路徑一律從回傳值 `RirApplied.detail.metadata` 取，否則在多次卷積之間
  會錯誤歸屬。
