# 材料優先的 RIR 場景 schema

English version: `rir_scene_v2.md`

`puresound.audio.rir.scene.schema` 定義 M1 generator 使用的版本化 `rir_scene.v2`
metadata。scene 儲存的是物理輸入，而不是一個被要求的 broadband RT60：

```text
six named surface meshes + material spectra + window/door area patches
  + temperature / humidity / pressure / sound speed
  + source and receiver pose / pattern / calibration
  + interior objects
```

`RoomSceneV2.to_json()` 與 `RoomSceneV2.from_json()` 會 round-trip 這個 canonical
representation。`to_metadata()` 另外會輸出 `room_dim`、`mic_pos`、`source_pos`、
`channel_map`，以及一個由材料推導出的 `rt60` 相容性數值，讓既有的
`PreGeneratedRoomBank` 讀取端可以繼續運作。

## Schema types

| Type | Purpose |
|------|---------|
| `MaterialSpectrum` | 在 octave 或 third-octave 中心頻率上的數值與不確定度；頻率內插採對數尺度。 |
| `SurfaceMaterial` | Absorption、scattering、transmission、provenance，以及選填、單位為 Pa·s/m 的 complex impedance。 |
| `SceneSurface` | `west/east/south/north/floor/ceiling` 六面之一的具名多邊形網格。 |
| `SurfacePatch` | 邊界上一扇窗或門的面積比例與材料。 |
| `EnvironmentConfig` | 溫度、相對濕度、氣壓，以及推導或明確指定的音速。 |
| `Pose` | 位置加上 yaw/pitch/roll 方向。 |
| `TransducerConfig` | Source/receiver 身分、pose、directivity、電平、校正與 array 身分。 |
| `SceneObject` | 家具的封閉多邊形足跡、高度區間、族別、材料、absorption/scattering，以及能量穿透率。 |
| `RoomSceneV2` | 完整 scene：驗證、序列化、有效牆面混合，以及預測 octave RT60。 |

`RoomSceneV2.rt60` 屬性只是一座遷移用的橋樑：它是由序列化 surfaces 算出的
500/1000 Hz Sabine 預測值中位數，絕不會從 v0 版「被要求的 RT60」欄位取樣或
複製而來。

## Material catalog

`puresound.audio.rir.scene.materials` 內含 `puresound-materials.v1`。初始的
absorption priors 是 Pyroomacoustics 0.10.1 materials 資料庫中可檢視的一個子集。
PureSound 額外加上明確的 uncertainty、scattering 與 transmission engineering
priors。這些是給模擬用的母體 priors，不是針對特定安裝產品的認證量測。

catalog 涵蓋：

- 磚石、石膏板與木造牆體構造；
- 硬質、木質與地毯地板；
- 硬質與吸音天花板；
- 玻璃、木門、窗簾，以及布面/木製家具。

每個抽樣出的房間共享一個房間層級的 latent absorption shift，之後再各自疊加
較小的材料層級電平與頻譜斜率擾動。Room-type 配方會讓建材彼此相關：

| Room type | Typical correlation |
|-----------|---------------------|
| `office` | 石膏板/磚牆，地毯或塑膠地板，通常是吸音天花板 |
| `meeting_room` | 石膏板/木牆，大多是地毯，吸音天花板 |
| `classroom` | 磚/石膏板，大多是硬地板，硬質與吸音天花板混合 |
| `living_room` | 灰泥/磚/木牆，地毯或木地板，大多是硬質天花板 |

每個房間也會在不同牆面各拿到一個窗戶 patch 與一個門 patch。目前的 shoebox
backend 使用的是底牆與 patches 的面積加權材料頻譜；它們個別的組成材料與比例
仍保留在 metadata 中，供之後的 explicit mesh/path backend 使用。

## Complex impedance contract

`SurfaceMaterial.impedance_real` 與 `impedance_imag` 儲存 surface impedance 的
實部與虛部，單位是 SI 制的 Pa·s/m。兩條頻譜都必須存在、共享相同的頻率中心，
且對一個被動表面而言實部不可為負。`impedance_at(frequency_hz)` 執行與其他
材料頻譜相同的對數頻率內插。

Absorption 無法唯一決定 impedance，因為它只指定了反射係數的大小，沒有指定
相位。因此材料 catalog 在有 phase-aware 量測或明確版本化的 prior 之前，會讓
impedance 保持缺席狀態，不會從 absorption 表悄悄合成出 impedance。

當一個 area-patched 邊界的每個組成都已宣告 impedance，
`effective_boundary_materials()` 會依面積把 normal admittance 並聯混合：

```text
Y_effective = sum(area_fraction_i / Z_i)
Z_effective = 1 / Y_effective
```

若任何組成的 impedance 未知，有效 impedance 就維持未知。Absorption、
scattering 與 transmission 則繼續使用它們既有的面積加權混合方式。

[`impedance_priors.md`](impedance_priors.md) 中 phase-aware 的參考資料維持在
一個獨立的實驗性 catalog 裡，不會自動對應到 `SurfaceMaterial`：一筆 100 mm
rigid-backed glass-wool 的參考量測，不能拿來替代一片未指定的地毯、天花板或
已安裝的牆體系統。

## High-frequency rendering

對一個 v2 scene，`PyroomacousticsHighFrequencyBackend` 會為每個邊界建構一個
頻率相依的 `pra.Material`，同時包含 absorption 與 scattering 頻譜。溫度與
濕度會傳入房間，且兩個 hybrid bands 共用同一個 scene 音速以對齊 direct-path。

M3 另外提供一個明確 opt-in 的 `PathEventHighFrequencyBackend`。它把每個
`SceneObject` 當成一個封閉的垂直稜柱，對 source/reflections/receiver 折線的
每一段做可見性測試，並把被擋住的 events 保留在 metadata 中以供檢視。一條被
擋住的 direct path 可能產生：

- 一個直穿事件（straight-through event），其 pressure gain 等於各物件
  energy transmission 乘積的平方根；
- 每個 blocker 最多兩條最短的可見垂直邊 diffraction 繞射路徑；
- first-order 的 deterministic wall-scattering 分支，其 `1-s` 與 `s/N`
  能量分配總和恰好等於母事件。

這個交互作用模型刻意收斂在有限範圍內：它不是一個通用三角網格或完整 UTD
實作。Diffraction 是一個受限的 1 kHz 參考模型，controlled scattering 目前也
只套用在 first-order 牆面路徑上。

Source 的 `speech_cardioid` directivity 會依 source 朝向與出射方向逐 event
計算。Receiver 在這個 backend 中維持 omnidirectional；不支援的 pattern 會
直接 raise，而不會悄悄退回 omni。

M4（`PathEventFDNHighFrequencyBackend`）繼承 M3 backend，把它稀疏的
late field 換成一條 deterministic 的 multiband feedback-delay network
（FDN）尾段。實作上會先算出 coherent 的 M3 early response，再以一個以
可設定的 mixing time（`--fdn-mixing-time-ms`，預設 24 ms）為中心、跨越
可設定 transition duration（`--fdn-transition-ms`，預設 16 ms）的
equal-power transition，把它 crossfade 進 FDN 尾段。FDN 每個頻帶的 RT60
目標來自 scene 自身預測的 octave decay，而不是一個固定常數；delay-line
數量是 2 的冪次（`--fdn-delay-lines`，預設 16），並依房間/source
決定性地 seed（`--fdn-seed`）。一個 positive-quadratic-root 的 gain 解會讓
transition 之後的能量，等於原本 PathEvent response 原本會帶有的能量，因此
M4 只改變 late-field 的性格，不會改變 M3 已經產生的 direct/early samples。

### 沒有單一的 default backend

這個系統沒有單一一個「the default backend」——預設值取決於呼叫的是哪一個
進入點：

| Entry point | Flag | Choices | Default |
|---|---|---|---|
| `generate_hybrid_rir.py`（底層 generator） | `--high-backend` | `pyroomacoustics`、`path-events-m3`、`path-events-m4` | `pyroomacoustics` |
| `generate_m6_bank.py`（M6 one-command wrapper） | `--backend` | `pyroomacoustics`、`path-events-m4` | `path-events-m4` |

`generate_hybrid_rir.py` 自己的預設值刻意釘死在 `pyroomacoustics`，預期不會
改變：M5 measured-calibration 的 exit gate 會檢查 parsed 出來的預設 backend
型別，M4 coupling validator 的 `production_default_unchanged` 檢查會讀取
FDN metadata，而且有一個專門的測試
（`test_m6_emission_is_opt_in_and_default_backend_is_unchanged`）直接斷言
parsed 出來的預設值。改動這一層底層預設值，等於回頭改寫已經關閉的
milestones 的 exit contracts，因此這次改動是往上一層做的。

`generate_m6_bank.py`——訓練資料建議使用的進入點，見
[`egs/rir_generation/README.md`](../../egs/rir_generation/README.md)——把
自己的 `--backend` 預設成 `path-events-m4`（自 commit `3525008`，
2026-08-04 起），並且每次呼叫都會明確把 `--high-backend` 往下傳給底層
腳本，所以 M6 的預設值完全活在 wrapper 這一層，從未觸碰
`generate_hybrid_rir.py` 自己的預設值。這個改動是根據量測證據做的：對照
五個語料庫共 1465 筆量測 RIR，M4 的 normalized octave decay shape 與量測
衰減的偏差是 `0.081`，而 Pyroomacoustics 的偏差是 `0.628`
（`RIR_EXP_LOG.md` §6.6.6）——Pyroomacoustics 在 2 kHz 的高頻殘響大約長了
2.2 倍，方向完全錯誤，五個參考語料庫全部一致指向相反方向。Pyroomacoustics
在兩層都仍然是第一 class 的明確選項，也是 A/B arm；它偏長的成因目前仍是
未解問題。

啟用 `record_realized_metrics=True` 時，每個輸出的 sidecar 都會用
`puresound.audio.rir.metrics` 記錄 broadband 與有效 octave-band 的 DRR、
C50/C80、EDT、T20、T30、fit quality 與 spectral tilt。預測出的材料衰減與
實際算出的 RIR 衰減是分開的欄位。

## Calibrated output

`HybridRIRConfig.output_mode` 有兩種模式：

- `peak_normalized` 保留 v0 的行為，把整個五聲道 item 縮放到 `normalize_peak`；
- `calibrated` 絕不做逐 item 的 peak normalization。1 公尺處的 source SPL 與
  receiver calibration 會被換算成相對於 94 dB SPL（約 1 Pa）參考值的
  逐聲道 amplitude gain。

這保留了距離、source 電平與跨房間的增益差異。下游 recipe 可以再對
convolved 後的訓練混音做正規化，但那是一個獨立且會被記錄下來的決定。

## Generator

M1 的參考設定是
[`egs/rir_generation/phases/m1_material/config/material_v1.json`](../../egs/rir_generation/phases/m1_material/config/material_v1.json)。

```bash
python egs/rir_generation/generate_hybrid_rir.py \
  --output-dir egs/rir_generation/exp/rir_realism/m1/hybrid_rir_material_v1_16k \
  --scene-version v1 --room-type mixed \
  --output-mode calibrated --record-realized-metrics \
  --n-rooms 1000 --rir-per-room 10 \
  --sample-rate 16000 --duration 1.6 \
  --low-backend pytard --num-workers 22
```

舊的指令維持 v0 為預設值。這是刻意的：選擇 `--scene-version v1` 是一個明確
的 bank 版本變更。

## Current boundary

M1 把高頻衰減變成 material-first。M2 加入了實驗性的 per-mode 與
complex-impedance 低頻路徑。M3 現在透過一個 opt-in 的 PathEvent backend
算出家具可見性、scalar reference-frequency 的
transmission/diffraction/scattering，以及 source cardioid。M4 為同一個
mono/per-channel 的 hybrid generator 加上一個 deterministic 的 multiband
FDN late field（opt-in，見上文），而透過另一個獨立的
`puresound.audio.rir.render.spatial` API——如
[`spatial_rir.zh-TW.md`](spatial_rir.zh-TW.md)
所述——則加上了 synchronized receiver arrays、first-order Ambisonics，
以及一個選用的 binaural decoder。

Surface patches 在 shoebox backend 上仍是面積加權的，還不是明確的多邊形。
General meshes、frequency-dependent diffraction，以及量測得來（而非理想化
first-order）的 source/receiver directivity patterns，都還是之後的
milestones。
