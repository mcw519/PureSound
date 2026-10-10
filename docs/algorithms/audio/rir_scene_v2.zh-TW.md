# RIR scene schema — `puresound.audio.rir.scene`

English version: [rir_scene_v2.md](rir_scene_v2.md)

`RoomSceneV2`（schema 字串 `rir_scene.v2`）用物理成因描述一個 shoebox 房間——
幾何、表面材質、環境、transducer 與家具。衰減由這些成因導出，從來不是輸入。
這讓 scene 的 RT60 與其吸收一致、位準與距離和聲源功率一致，也讓每個渲染出的
item 都能追溯到一個可序列化的 scene。

## 型別

| 型別 | 欄位（單位） |
|---|---|
| `MaterialSpectrum` | `center_frequencies_hz`（遞增）、`values`、可選 `uncertainty_std`；`at(f)` 在 log2 頻率上線性內插，邊界外維持端點值 |
| `SurfaceMaterial` | `absorption`、`scattering`、`transmission` 頻譜，值域 [0, 1]；`provenance`；可選 `impedance_real`、`impedance_imag`（Pa·s/m） |
| `SceneSurface` | 六個邊界之一 `west, east, south, north, floor, ceiling`；頂點（m）；基底 `material_id`；`patches` |
| `SurfacePatch` | 窗、門或其他 patch：`role`、`area_fraction` 值域 [0, 1)、`material_id` |
| `EnvironmentConfig` | `temperature_c`（−50..60）、`relative_humidity_percent`、`pressure_pa`、`sound_speed_m_s` |
| `Pose` | `position_m` [x, y, z]、`orientation_ypr_deg` [yaw, pitch, roll] |
| `TransducerConfig` | `kind`（`source`／`receiver`）、`pose`、`directivity_id`、`calibration_gain_db`、`power_db_spl_at_1m`（聲源必填）、`array_id`、`channel_index` |
| `SceneObject` | 垂直柱體：`footprint` 多邊形（m）、`z_min`、`z_max`、`material_id`、寬頻 `absorption`、`scattering`、`transmission` |
| `RoomSceneV2` | `scene_id`、`room_type`、`dimensions_m`、`surfaces`、`materials`、`environment`、`sources`、`receivers`、`objects`、`catalog_version` |

建構時會驗證 scene：剛好六個具名邊界、patch 面積比例總和小於一、所有引用的
材質都存在、物件在房間內、沒有 transducer 位於物件內部。

## 導出量

**聲速。** 未給 `sound_speed_m_s` 時，
`c = 331.3 + 0.606 · T + 0.0124 · RH` m/s（T 單位 °C，RH 單位 %）。

**等效邊界材質**（`effective_boundary_materials`）。有 patch 的邊界以面積加權
化為每個邊界一個材質：吸收、散射與穿透取面積加權平均，不確定度以平方和開根號
合成。阻抗透過導納混合，因為同一面牆上的 patch 感受到幾乎相同的聲壓，其法向
導納以並聯方式相加：

```text
Y_eff = Σ_i  a_i / Z_i          （a_i = 面積比例）
Z_eff = 1 / Y_eff               （實部截到 ≥ 0）
```

只要有任一組成缺少阻抗，等效阻抗就維持未知；吸收、散射與穿透仍照常混合。

**預測 RT60**（`predicted_octave_rt60_s`）。逐 octave 頻帶的 Sabine 公式：

```text
RT60(f) = 24 ln(10) · V / (c · Σ_b S_b · α_b(f))，截到 [0.02, 20] s
```

`V` 為體積，`S_b`、`α_b` 為邊界 `b` 的面積與等效吸收。Sabine 假設擴散聲場且只
計入邊界，所以它是 metadata 與晚場目標，不是保證：吸收集中在單一表面或家具
主導時最不可靠。

**`rt60`** 是 500 Hz 與 1 kHz 預測值的中位數。它的存在是讓為 RT60 驅動 scene
寫的程式能讀 `RoomSceneV2`；它從不複製自某個要求的 RT60。

**Transducer 增益**（`transducer_channel_gains`）。每個聲源為
`10^((power_db_spl_at_1m − reference + receiver calibration_gain_db) / 20)`，
供 hybrid renderer 的 `calibrated` 輸出使用。

## 序列化

`to_json()`／`from_json()` 讓 canonical scene 來回轉換；`from_json` 接受 JSON 字串
或路徑。`to_metadata()` 另加 bank loader 與 QC 讀取的相容欄位：`room_dim`、
`rt60`（附 `rt60_origin`）、`mic_pos`、`source_pos`、`source_labels`、
`source_distances`，以及 `channel_map` 清單（`channel`、`label`、`source_pos`、
`distance_m`、`horizontal_distance_m`、`power_db_spl_at_1m`、`directivity_id`）。
`load_room_scene(dict)` 解析 `rir_scene.v2` mapping，其他 mapping 原樣回傳。

## 材質目錄

`puresound.audio.rir.scene.materials`（`MATERIAL_CATALOG_VERSION =
"puresound-materials.v1"`）以七個 octave 頻帶、125 Hz – 8 kHz
（`MATERIAL_BANDS_HZ`）存放常見飾面的母體先驗：磚牆、石膏板、木襯板、玻璃窗、
木門、16 mm 木板、亞麻地板、兩種地毯、棉窗簾、三種天花板飾面與軟墊家具。
吸收係數改編自 Pyroomacoustics 材質資料庫；散射、穿透與不確定度是工程先驗，
不是已安裝產品的認證量測。目錄不存阻抗：吸收係數無法決定反射相位，所以不
憑空編造。

`ROOM_TYPE_RECIPES` 為 `office`、`meeting_room`、`classroom`、`living_room`
給出牆、地板、天花板飾面的加權選項。`sample_materialized_shoebox` 為每類表面
抽一種飾面，抽一個窗 patch（一面牆的 6–22 %）與一個門 patch（另一面牆的
2.5–7.5 %），並在 logit 空間擾動每個係數：

```text
logit(α') = logit(α) + s · (0.22 · z_room + ε_level + ε_slope · u(f))
```

`z_room ~ N(0, 1)` 由房間內所有材質共用，所以飾面會一起變得較吸音或較不吸音；
`ε_level ~ N(0, 0.18)` 與 `ε_slope ~ N(0, 0.12)` 逐材質抽取，`u(f)` 在各頻帶間
由 −1 到 1，`s` 是 `material_variation_scale`；散射只用一半的房間項。家具依其
類別取軟墊或木質材質。

## 抽樣一個 scene

`sample_material_first_rir_scene(config, seed=...)` 先依 `HybridRIRConfig` 抽房間
幾何、聲源與 receiver 位置、家具，再呼叫 `upgrade_hybrid_scene_to_v2`：加上材質，
並抽環境（18–25 °C、30–70 % RH、98–103 kPa）、一個房間層級的聲源功率
`N(70, 2)` dB SPL @ 1 m 再加每個聲源 `N(0, 1.5)` dB、聲源朝向（yaw 均勻、pitch
在 ±15° 內）與 `speech_cardioid` 指向性，以及一個全指向 receiver `mic_0`。輸入
幾何的 RT60 不會帶過來。

在 `generate_hybrid_rir.py` 中，`--scene-version v0` 渲染 RT60 驅動的
`HybridRIRScene`，`--scene-version v1` 渲染 material-first 的 `RoomSceneV2`
（`--room-type`、`--material-variation-scale`）。

## Backend 如何使用 scene

| Backend | 讀取的內容 |
|---|---|
| Pyroomacoustics | 每個邊界的等效吸收與散射頻譜、溫度、濕度、聲速、聲源指向性；家具透過事後遮擋模型（從直達聲起、在 `obstacle_occlusion_recovery_ms` 內回升的衰減斜坡，加一個散射 tap） |
| PathEvents | 明確的 image-source 路徑與被動邊界濾波、聲源與 receiver 指向性、物件遮擋與穿透、一階邊緣繞射與擴散散射、空氣吸收 |
| PathEvents + FDN | 早場同上，再加一個 multiband 晚場，其 octave 目標為經空氣吸收修正的預測 RT60 |

限制：在 shoebox backend 中 patch 是面積加權而非實際擺放的多邊形；繞射是有界
的一階模型，不是網格求解器；receiver 指向性只在 backend 宣告支援時才生效。

## 複數阻抗

材質可帶實部與虛部表面阻抗，單位 Pa·s/m。兩條頻譜必須都存在且中心頻率相同，
實部必須非負（被動表面）。低頻阻抗 backend 會使用它們；見
[阻抗先驗](impedance_priors.zh-TW.md) 與 [阻抗量測](impedance_measurements.zh-TW.md)。
