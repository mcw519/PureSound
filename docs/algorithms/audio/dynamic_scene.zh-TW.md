# 移動聲源渲染 — `puresound.audio.rir.render.dynamic`

English: [dynamic_scene.md](dynamic_scene.md)

渲染 `DynamicSceneSpec`（`puresound.audio.rir.scene.dynamic`）：靜態的
`RoomSceneV2` 房間、一支固定的單聲道麥克風，以及最多八個聲源——目標講者、其他
講者與噪音——每個聲源可以靜止不動，或沿關鍵影格軌跡移動。輸出麥克風混音、各聲源
分軌與三種評估參考。Web 工作台的[聲學世界](../../usage/world.zh-TW.md)畫面建立在
它之上。
渲染器版本：`DYNAMIC_RENDER_VERSION = "puresound.dynamic_geometric_fdn.v2.1"`。

```python
from puresound.audio.rir.scene.world_presets import world_preset
from puresound.audio.rir.render.dynamic import render_dynamic_scene

scene = world_preset("approach", seed=4)
render = render_dynamic_scene(scene, assets)  # asset_id -> 16 kHz 單聲道陣列
render.mixture, render.stems, render.references["target"], render.references["near"]
```

## 場景與角色

`DynamicSceneSpec`（schema `puresound.dynamic_scene.v2`）有 1 到 8 個聲源
（`MAX_SOURCES`），每個 `DynamicSource` 帶一個角色：

| 角色 | 意義 | 房間 transducer 的指向性 |
| --- | --- | --- |
| `target` | 模型應保留的講者 | `speech_human` |
| `interferer` | 其他講者 | `speech_human` |
| `noise` | 噪音源 | `omnidirectional` |

渲染器從房間 transducer 讀取指向性；預設場景與 Web 編輯器依角色設定它。每個聲源
都依關鍵影格移動，只有一個關鍵影格就是靜止不動——噪音也一樣。速度、牆面、障礙物
與麥克風距離檢查適用於每個聲源。`reference_rir` 決定參考音保留什麼（見下文）。

語音只播放完整音檔。設定 `repeat` 時，講者播放放得下的完整音檔次數，之後保持
安靜，但剩餘時間的傳播與殘響仍照常渲染；連一次都放不下的音檔會被拒絕。噪音可以
在最後填入不完整的一輪。

較舊的 `puresound.dynamic_scene.v1` schema 場景（`fixed_source_id` 與角色
`speech` / `noise`）在讀取時轉換：固定的講者成為 `target`，其他講者成為
`interferer`，參考類型為 `early`。

## 訊號模型

每條同調路徑 `p`——直達聲，以及 `max_order`（≤ 2）以內的每條鏡像反射——都拆成
一段短的**路徑濾波器** `h_p` 和一段**傳播時間** `τ_p`。在時間 `e` 發出的 sample
於 `e + τ_p(e)` 抵達：

```
y(t) = Σ_p  (h_p,e ∗ s)(e)   於   t = e + τ_p(e)
```

`h_p,e` 依聲源在**發聲**時刻 `e` 的姿態計算：球面擴散、聲源指向性、麥克風指向、
牆面反射濾波與障礙物插入損失。濾波器本身不含延遲、延遲交給連續的時間映射處理，
路徑長度改變時才不會像「交叉淡化含延遲的脈衝響應」那樣產生梳狀濾波與雙重到達；
Doppler 頻移由時間映射自然產生。

| 網格 | 步距 | 計算內容 |
| --- | --- | --- |
| 幾何 | 10 ms（加上每個關鍵影格時刻） | 姿態、距離、近場權重、每條路徑的傳播時間 |
| 路徑濾波 | 40 ms（加上關鍵影格與最後一格） | 路徑事件、牆面濾波、指向性、遮擋 |

兩次路徑濾波更新之間，濾波器 taps 線性內插，每條路徑的鏡像聲源位置也線性內插。
後者是精確的：鏡像聲源是聲源位置的仿射函數，而聲源在關鍵影格之間等速直線移動，
關鍵影格又都是更新點。所以由內插後鏡像位置算出的傳播時間，與每 10 ms 重算的結果
相同。濾波器本身（入射角、指向性、Fresnel 數）在步行速度（≤ 2 m/s）下每 40 ms
只變幾度。

路徑以其類型與鏡像階數追蹤。二階路徑撞上兩面牆的先後順序會隨聲源移動而對調，所以不
列入身份；某次更新時反射點剛好落在房間稜邊上，該次的濾波器會由前後更新內插，而不是
讓路徑消失。

### 時變濾波

兩個更新點之間的段落 `j`，交叉淡化兩端濾波器的輸出。濾波對 taps 是線性的，所以
等同於用線性內插的 taps 濾波。每個輸入訊號的所有段落用一次批次 overlap-save FFT
處理（`_SegmentFilter`）；從不移動的聲源就是一般的 FFT 卷積。

### 小數傳播時間

輸出 sample `r` 讀取滿足 `e + τ(e) = r` 的發聲時刻 `e`，這個讀取就是小數延遲。
Kaiser 窗 sinc（32 taps、β = 6；`path_events/fractional_delay.py`）在任何小數
部分下，7 kHz 以內的振幅誤差都小於 0.01 dB（Laakso 等，"Splitting the unit
delay"，IEEE Signal Processing Magazine，1996）。低階內插的高頻損失會隨小數部分
改變：

| 內插方式 | 4 kHz | 6 kHz | 7 kHz |
| --- | --- | --- | --- |
| 線性 | 0 … −3.0 dB | 0 … −8.3 dB | 0 … −14.2 dB |
| 因果三階 Lagrange（靜態渲染器預設） | −0.1 … +0.7 dB | −3.2 … +1.3 dB | −8.4 … +1.4 dB |
| 窗 sinc，32 taps | ±0.01 dB | ±0.01 dB | ±0.01 dB |

移動聲源的小數部分以 `v / c · fs` Hz 循環（1 m/s 時為 47 Hz），表中的起伏就變成
高頻的振幅調變；改用 sinc 後消失（6 kHz 實測 0.00 dB p-p）。對稱的核在抵達前會有
15 個 sample（< 1 ms）的預振鈴。沒有套用 Doppler 振幅因子；Mach < 0.006 時它小於
0.05 dB。

## 聲源指向性

預設場景的講者使用 `speech_human`：正常音量說話的水平面倍頻帶位準，0–180° 每 15°
一筆，取自 Monson、Hunter & Story，"Horizontal directivity of low- and
high-frequency energy in speech and singing"，JASA 132(1), 433–441 (2012)，
表 I。相對於嘴部正前方：

| 角度 | 125 Hz | 500 Hz | 1 kHz | 2 kHz | 4 kHz | 8 kHz |
| --- | --- | --- | --- | --- | --- | --- |
| 90° | −1.6 | −1.6 | −1.6 | −7.0 | −6.3 | −8.7 |
| 180° | −3.6 | −5.9 | −6.3 | −13.1 | −18.9 | −26.3 |

假設指向性對嘴部軸對稱（仰角沿用水平資料）。它以 `PathBandGain` 隨每條路徑攜帶，
以最小相位 FIR 套用（`path_events/band_filter.py`）。`speech_cardioid` 是
不分頻率的理想 cardioid，正後方為零點；指定它的場景仍照樣渲染。

## 障礙物

物件改為連續衰減路徑，而不是直接把路徑關掉
（`occlusion_model="fresnel_kirchhoff"`，`path_events/occlusion.py`）。每個物件
從每段路徑看出去，視為不透明的矩形屏風；依 Babinet 原理，屏風後方的聲場為

```
U / U0 = 1 − (1 − t) · [F(a2) − F(a1)] · [F(b2) − F(b1)] / (2j),   F(v) = C(v) + j S(v)
```

其中 `[a1, a2] × [b1, b2]` 是屏風以 Fresnel 單位
`v = h · sqrt(2 (d1 + d2) / (λ d1 d2))` 表示的範圍，`t = sqrt(SceneObject.transmission)`
（Born & Wolf，*Principles of Optics*，§8.7–8.9；Pierce，*Acoustics*，第 9 章）。
陰影邊界上是 −6 dB，頻率越高、越深入陰影損失越大，而且跨越邊界時連續。立在地板上
或頂到天花板的物件，聲音不能從下方或上方繞過。每個 1/3 倍頻帶中心的增益取該處一個
倍頻程內的平均能量，三條邊之間的干涉零點才不會變成在頻譜上滑動的凹口。

在「經過障礙物」預設（0.4 × 0.8 m、高 2 m 的屏風）中，屏風後方的直達聲在 250 Hz
損失 2.4–7.3 dB（陰影邊緣附近比正後方多）、1 kHz 損失 6–16 dB、4 kHz 損失 6–19 dB、
8 kHz 損失 6–24 dB（邊緣附近較少）。講者走過屏風後方時，損失每 50 ms 最多變化：
250 Hz 0.1 dB、8 kHz 5 dB（陰影邊緣在高頻最銳利）。限制：物件視為薄屏風（厚物件實際衰減會再多一些）、沒有模擬
物件表面的反射，而且 Kirchhoff 理論會高估遠小於波長的物件的影響，所以約 250 Hz
以下的損失應視為上限。

## 晚場

擴散場沿用專案的能量匹配 PathEvent–FDN 耦合（[早／晚場耦合](rir_late_coupling.zh-TW.md)），
所以移動場景與同一房間的靜態渲染一致。在某個姿態上，先算靜態 PathEvent RIR
（二階、窗 sinc 延遲），再耦合到多頻帶 FDN，其尾段能量依材質加上空氣吸收的倍頻帶 RT60 決定（與靜態 FDN backend 的目標相同）；把它的
擴散成分重新對齊到直達聲抵達時刻，就是該姿態的晚場響應 `L_k`。同調路徑依耦合的
等功率權重，在直達聲後 16 到 32 ms 之間淡出；權重依每個 tap 的抵達時間逐 tap
計算。

晚場響應在每個關鍵影格，以及兩個關鍵影格之間等距插入的姿態上計算，使相鄰姿態相距
不超過 1 公尺、45°。每個發出的 sample 驅動其發聲時間前後的兩個響應：

```
late(t) = Σ_k  (g(u) · w_k(u) · s(e)) ∗ L_k,   w = (1 − u, u)
g(u)² = ((1 − u) E_a + u E_b) / ((1 − u)² E_a + u² E_b + 2 u (1 − u) C_ab)
```

`E` 是兩個響應的能量，`C` 是它們的內積。相鄰響應只有部分相關（相關係數 0.35–0.66），
單純的線性權重在中點最多會損失 3 dB；`g` 讓混合後的能量在兩者之間呈線性。沿 3.6 公尺
的行走路徑，中點的晚場位準與站在該處的聲源相差在 1 dB 以內。

即使場景只同調渲染較少階反射，晚場能量仍由二階路徑場決定（`LATE_FIELD_ORDER`），因為擴散位準屬於房間本身。每個聲源有自己的
FDN seed（`SeedSequence([scene.seed, index])`），兩位講者的殘響尾巴不再是同一個
濾波器。

## 評估參考

每個聲源都會得到一條場景 `reference_rir` 類型的參考分軌，位於麥克風的時間軸上。
類型與時間窗沿用訓練的 `target_rir_type`（`wav_apply_rir`），從每次發聲的直達聲
起算（`REFERENCE_WINDOWS_S`）：

| `reference_rir` | 每次發聲保留的部分 |
| --- | --- |
| `early`（預設） | 直達聲之後 50 ms 內抵達的同調路徑，加上擴散場的前 50 ms |
| `direct` | 直達聲之後 6 ms 內的同調路徑；擴散場開始得更晚，所以完全不含 |
| `full` | 全部：參考等於該聲源的分軌 |
| `anechoic` | 乾聲移到直達聲抵達的時間、單位增益——沒有距離衰減、指向性、空氣吸收或牆面 |

參考分軌加總成三種參考（`render.references`）：

- **`target`**：目標講者。場景沒有目標講者時為靜音。
- **`speech`**：所有講者，不論是否為目標。
- **`near`**：每位講者在發聲時刻依 `near_radius_m` 附近 0.2 m 寬的 raised cosine
  （`near_weights`）加權，再像聲源一樣傳播。區域內沒有人時為靜音。

所有訊號共用一個增益，使混音峰值不超過 0.95。`render.audio()` 的命名為 `input`、
`source-{id}`、`reference-source-{id}` 與 `reference-{target,speech,near}`。

`puresound.evaluation.world.world_metrics` 以每種參考為輸出評分；參考為靜音時，
改為逐窗回報輸出的殘留位準，而非 SI-SDR。Web 工作台在使用者未指定時，降噪模型以
`speech`、人聲分離模型以 `near` 評分（`puresound.web.world.POLICY_DEFAULTS`）；
兩者皆非時，場景有目標講者就用 `target`，否則用 `speech`。

## 驗證

`test/rir/test_dynamic_scene.py` 與 `test/rir/test_rir_path_band_effects.py`：

| 性質 | 檢查 |
| --- | --- |
| 靜止聲源與靜態管線一致 | 與「窗 sinc PathEvent RIR + 耦合」相比，無晚場誤差低於 −120 dB（實測 −149 dB）、有晚場低於 −40 dB（實測 −44 dB） |
| 晚場跟著房間走 | 三種材質、兩種房間尺寸下，50 ms 後能量與靜態耦合差 0.6 dB 以內 |
| 沒有高頻調變 | 移動聲源 6 kHz 包絡起伏小於 0.2 dB |
| Doppler | 以 1 m/s 遠離的 1 kHz 純音實測為 `f · c / (c + 1)` |
| 遮擋連續 | 沿預設路徑，直達聲在 250 Hz、1 kHz、4 kHz 的損失每 20 ms 變化 < 1 dB；低頻能繞過屏風 |
| 晚場跟著移動 | 行走路徑中點的晚場位準與站在該處的聲源相差在 1 dB 以內 |
| 指向性 | 頻帶增益等於實測表；最小相位 FIR 誤差 0.15 dB 以內 |
| 殘響尾巴各自獨立 | 兩位講者晚場尾巴的相關係數低於 0.3 |
| 參考類型 | 脈衝的 `early` 與 `direct` 參考分別在直達聲後 50 ms 與 6 ms 結束；`full` 等於分軌；`anechoic` 是位於直達聲抵達時間、單位增益的脈衝 |
| 參考依角色加總 | 兩位目標講者加一位其他講者時，`target` 是目標的總和，`speech` 再加上其他講者，`near` 為每位講者加權 |
| 多聲源 | 兩位目標講者、兩位其他講者與兩個噪音源（其一移動）可通過驗證；第九個聲源會被拒絕 |

## 限制

靜態房間、一支全指向麥克風；不支援移動麥克風、雙耳渲染或開門。最多八個聲源，
渲染時間大致隨聲源數線性增加。多位目標講者以總和評分，不分別評分。同調反射只到二階，
其後的能量是統計式 FDN 場。路徑濾波每 40 ms 更新。指向性是水平面資料、以軸對稱
套用。障礙物是 Kirchhoff 薄屏風。
