# PureSound RIR 模組化重構計畫

狀態：R0–R7 全部完成；遷移收尾  
版本：v1.0  
日期：2026-08-02  
範圍：`puresound/audio` 底下與 RIR 生成、物理模型、空間渲染、校準、bank 管理相關的程式

本文件規劃並記錄模組化重構。R0–R7 已全部執行完畢，各階段結果附在對應小節；重構未改變任何 M 系列算法的物理假設。既有的整體算法與 milestone 狀態，仍以 [`RIR_REALISM_PLAN.md`](RIR_REALISM_PLAN.md) 為準；新舊 import 路徑對照見 [`docs/audio/rir_package_migration.md`](docs/audio/rir_package_migration.md)。

## 1. 重構目的

目前 RIR 功能已經從單純的 `rir_generator` 擴展到：

```text
場景與材料
  → 阻抗／模態／FDTD 物理模型
  → 低頻與高頻 renderer
  → path event 與 late field
  → crossover／空間／雙耳組裝
  → metrics／calibration
  → M6 bank、QC、release、production decision
```

功能已能運作，但目前大多數模組仍直接放在 `puresound/audio` 根目錄，造成：

1. `hybrid_rir.py` 同時負責資料模型、場景取樣、backend、交叉濾波、障礙物幾何、metadata 與檔案輸出。
2. `rir_path_events.py` 同時負責 schema、幾何 visibility、image-source 路徑生成與波形 renderer。
3. `rir_metrics.py` 同時包含時間、頻譜、衰減、雙耳與陣列空間分析。
4. bank 的 manifest、item QC、release、evaluation 與 production decision 有清楚的概念差異，但目前以互相 import 的平面檔案表示。
5. 核心物理演算法被 `torch`、`torchaudio`、filesystem 與 optional backend 細節牽連，難以單獨測試或重用。

重構的目標不是一次性重寫算法，而是建立清楚的分層，使每個模組只有一種主要責任，並保留現有 import 路徑與 M6 bank contract 的相容性。

## 2. 現況盤點

### 2.1 主要大型模組

| 現有檔案 | 約略行數 | 主要問題 |
|---|---:|---|
| `puresound/audio/hybrid_rir.py` | 3,240 | 低頻、高頻、場景、幾何、crossover、輸出全部混在一起 |
| `puresound/audio/rir_path_events.py` | 2,396 | schema、幾何、路徑生成、directivity、fractional delay、render 混在一起 |
| `puresound/audio/fdtd_reference.py` | 1,870 | 解析邊界參考、FDTD solver、source convention、測試診斷集中 |
| `puresound/audio/rir_metrics.py` | 1,516 | temporal、spectral、decay、echo density、IACC、spatial coherence 混在一起 |
| `puresound/audio/rir_bank_qc.py` | 1,251 | item 分析、multiprocessing、index 寫入、release audit 混在一起 |
| `puresound/audio/rir_bank_release.py` | 1,031 | variant materialization、distribution、recipe、release audit 混在一起 |
| `puresound/audio/rir_bank_manifest.py` | 880 | schema、hash、split policy、filesystem 操作與 audit 混在一起 |

### 2.2 依賴集中點

目前最常被其他 RIR 模組使用的核心為：

- `rir_metrics`：9 個下游模組使用。
- `acoustic_impedance`：7 個物理與 path-event 模組使用。
- `rir_bank_manifest`：5 個 bank 模組使用。
- `rir_path_events`、`rir_scene`：各被 5 個 rendering/calibration 模組使用。

直接 import cycle 目前不明顯，但尚未有架構規則阻止未來形成 cycle。語意上的高耦合主要在：

```text
hybrid_rir
  ├─ scene sampling
  ├─ low-frequency backends
  ├─ high-frequency backends
  ├─ crossover
  ├─ obstacle geometry
  ├─ metadata
  └─ dataset I/O

bank loader
  ├─ manifest
  ├─ release
  └─ production certificate

release
  └─ QC

production decision
  ├─ evaluation
  ├─ release
  └─ manifest
```

### 2.3 必須保留的現有 API

下列 API 已被生成腳本、phase validator 與測試使用，不能在第一階段直接刪除：

- `puresound.audio.hybrid_rir.HybridRIRConfig`
- `HybridRIRScene`、`PolygonObstacle`
- `AnalyticModalLowFrequencyBackend`
- `ImpedanceModalLowFrequencyBackend`
- `GpuARDPytARDBackend`、`GpuARDPytARDCuPyBackend`
- `PyroomacousticsHighFrequencyBackend`
- `PathEventHighFrequencyBackend`、`PathEventFDNHighFrequencyBackend`
- `generate_hybrid_rir`
- `sample_hybrid_rir_scene`、`sample_material_first_rir_scene`
- `upgrade_hybrid_scene_to_v2`
- `hybrid_crossover`
- `write_hybrid_rir_dataset_item`

目前也有少數測試與 CLI 直接使用 `hybrid_rir` 的私有 helper，例如 `_solve_modal_ard`、`_calibrate_pytard_signal`、`_apply_rt60_decay_envelope`、`_align_high_band_direct`。遷移前必須把這些 helper 分類成：正式 API、測試專用 API，或只保留在 compatibility shim 的 legacy symbol。

## 3. 目標架構

目標是在 `puresound/audio/rir/` 建立真正的 domain package。初期不移除舊的平面檔案；舊檔案先變成 re-export compatibility shim。

```text
puresound/audio/rir/
├── __init__.py
├── api.py                 # 對外穩定入口
├── contracts.py           # RIR tensor、metadata、backend protocol
├── scene/
│   ├── __init__.py
│   ├── schema.py          # RoomSceneV2、Pose、SurfaceMaterial 等
│   ├── sampling.py        # room/source/obstacle sampling
│   ├── materials.py       # material catalog 與 realization
│   └── geometry.py        # polygon、room、visibility 基礎幾何
├── physics/
│   ├── __init__.py
│   ├── propagation.py     # air absorption、sound speed、travel time
│   ├── impedance/
│   │   ├── admittance.py
│   │   ├── priors.py
│   │   ├── measurements.py
│   │   ├── tube.py
│   │   ├── fitting.py
│   │   ├── modes.py
│   │   └── residues.py
│   └── wave/
│       ├── fdtd.py
│       ├── low_frequency.py
│       └── source_convention.py
├── path_events/
│   ├── __init__.py
│   ├── schema.py          # ComplexPathGainSpectrum、PathEvent、PathEventSet
│   ├── geometry.py        # visibility、scene interaction、image source
│   ├── generator.py       # shoebox/scene path event generation
│   ├── directivity.py     # orientation、source/receiver directivity
│   └── renderer.py        # fractional delay、path-event waveform
├── render/
│   ├── __init__.py
│   ├── backend.py         # backend protocol、registry、capability metadata
│   ├── low_frequency/
│   │   ├── pytard.py
│   │   ├── analytic_modal.py
│   │   └── impedance_modal.py
│   ├── high_frequency/
│   │   ├── pyroomacoustics.py
│   │   ├── path_event.py
│   │   └── fdn.py
│   ├── crossover.py       # causal LR crossover、energy matching
│   ├── coupling.py        # path-event 與 FDN coupling
│   ├── spatial.py         # receiver array、Ambisonics
│   ├── binaural.py        # BRIR decoder/renderer
│   └── hybrid.py          # orchestration，不放底層幾何或 solver
├── metrics/
│   ├── __init__.py
│   ├── temporal.py        # direct、DRR、clarity、arrival、decay
│   ├── spectral.py        # octave band、tilt、頻譜 response
│   ├── density.py         # echo density、mixing time、noise floor
│   ├── spatial.py         # IACC、array coherence、diffuse coherence
│   └── report.py          # analyze_rir 與統一輸出 schema
├── calibration/
│   ├── __init__.py
│   ├── loss.py
│   ├── inverse_m4.py
│   ├── inverse_m5.py
│   ├── synthetic_recovery.py
│   ├── measured_campaign.py
│   └── residual.py
└── bank/
    ├── __init__.py
    ├── schema.py          # manifest、item、split、release schema
    ├── storage.py         # WAV/metadata/index filesystem adapter
    ├── loader.py          # training-time PreGeneratedRoomBank
    ├── qc.py
    ├── release.py
    ├── evaluation.py
    └── production.py
```

### 3.1 分層規則

依賴只能由上往下：

```text
api / CLI adapter
        ↓
render、calibration、bank
        ↓
path_events、scene、metrics
        ↓
physics、contracts
        ↓
numpy/scipy 基礎運算
```

具體規則：

1. `contracts` 不得 import renderer、bank 或 filesystem。
2. `scene` 不得 import `torch`、`torchaudio` 或 Pyroomacoustics。
3. `physics` 只負責物理量與 solver，不寫 WAV、manifest 或 training metadata。
4. `path_events` 可以使用 scene 與 physics，但不能依賴 bank。
5. `render` 可以組裝 scene/path/physics，但不能直接修改 bank manifest。
6. `metrics` 接受 array-like 與明確的分析設定，不能依賴特定 renderer。
7. `bank` 可以使用 metrics，但 bank schema 不得反向依賴 hybrid renderer。
8. optional backend（PyTARD、CuPy、Pyroomacoustics）必須 lazy import。
9. 所有 filesystem 與 CLI 行為放在 adapter 層；核心函數優先回傳 immutable dataclass 或 NumPy array。
10. 舊的 `puresound.audio.*` import 路徑在遷移完成前必須維持可用。

## 4. 分階段遷移計畫

### R0：API 與 contract freeze

目的：在移動任何程式前，固定行為與輸出格式。

工作項目：

- 列出 `hybrid_rir` 的正式 public symbols 與 legacy/private symbols。
- 固定 `[channels, samples]`、sample rate、dtype、causality 與 metadata contract。
- 為 `RoomSceneV2`、`PathEventSet`、M6 manifest/release 建立 contract fixtures。
- 建立 import boundary 測試，禁止新模組跨層 import。
- 確認 optional dependency 缺失時，scene/metrics/schema 仍可 import。

完成條件：現有測試全數通過，且每個保留 API 都有相容性測試。

#### R0 實作結果（2026-08-02）

- [x] 建立 `puresound/audio/rir/` package 與 `contracts.py`。`contracts` 只 import
  stdlib 與 NumPy，不碰 torch、torchaudio、renderer、bank 或 filesystem。
- [x] 定義 `RIRArray`（`[channels, samples]` 佈局驗證、dtype 轉換、因果邊界檢查）、
  `RenderContext`（backend 真正需要的 render 參數子集，與 scene sampling knobs 分離）、
  `BackendCapabilities`（含顯式 `deterministic_for_fixed_seed`）與
  `validate_rir_metadata`。
- [x] 凍結 API inventory：`hybrid_rir` 22 個 public symbols 分成
  egs 使用（15）、僅測試使用（5）、無使用者（2）三類；16 個被外部引用的 private
  helper 分成「應提升為 public」（6，egs CLI 直接 import）與「僅測試」（8）。
- [x] 建立 `RoomSceneV2`、`PathEventSet`、M6 manifest 的 golden fixtures，
  同時以檔案與原始碼中的 SHA-256 常數雙重釘住。
- [x] 建立 layer-direction lint 與 optional-dependency 守門測試。

新增檔案：

- `puresound/audio/rir/{__init__,contracts}.py`；
- `test/test_rir_r0_{api_inventory,contracts,golden_fixtures,import_boundaries}.py`；
- `test/fixtures/rir_r0/{room_scene_v2,path_event_set,bank_manifest}.json`。

R0 完全是新增，沒有修改任何既有檔案。115 個 R0 測試通過；M6 validator 套件
30/30、RIR 相關回歸 218/218 維持通過。

R0 過程中確認並記錄的兩件事：

1. **NumPy/Torch 邊界是單一一行**——`hybrid_rir.py:2162` 的
   `torch.as_tensor(rir, dtype=torch.float32)`。其餘全程 NumPy。契約已據此凍結，
   R2 拆 renderer 時不得增加第二個轉換點。
2. **`RoomSceneV2` 的物件往返不是 byte-stable**——`from_dict(to_dict())` 會把
   整數座標 `[0, 0, 0]` 加寬成 `[0.0, 0.0, 0.0]`，canonical JSON hash 因此改變。
   數值相等，且 M6 resume 不受影響（它 hash 的是從 JSON 載入的原始 dict，而 JSON
   文字往返會保留 int/float 區別）。但任何未來「重建 scene 物件再重算
   `scene_sha256`」的程式都會拿到不同的 hash。R0 以測試釘住現況；R1 若要修正，
   必須是明示決定並確認沒有既有 digest 依賴它。

### R1：抽出 contracts 與 scene

來源：`rir_scene.py`、`rir_materials.py`、`hybrid_rir.py` 的 scene sampling/obstacle data。

拆分：

- `RoomSceneV2`、`SurfaceMaterial`、`Pose` 等移到 `scene/schema.py`。
- room/source/obstacle sampling 移到 `scene/sampling.py`。
- polygon 與 room geometry 基礎工具移到 `scene/geometry.py`。
- material catalog/realization 移到 `scene/materials.py`。
- `HybridRIRConfig` 與 backend protocol 移到 `contracts.py` 或 `render/backend.py`。

完成條件：scene/schema 不需載入 torch 或 optional renderer；舊 import 仍可用。

#### R1 實作結果（2026-08-02）

分五個可獨立驗證的小步執行，每步都以 R0 golden fixture 的 SHA-256 當關卡：

| 步驟 | 內容 | 結果 |
|---|---|---|
| R1a | `rir_scene.py` → `rir/scene/schema.py` | 836 行，`git mv` 保留歷史 |
| R1b | `rir_materials.py` → `rir/scene/materials.py` | 381 行 |
| R1c | `HybridRIRConfig` → `contracts.py` | 純資料，落 layer 0 |
| R1d | 純幾何 → `rir/scene/geometry.py` | 10 個函式，177 行 |
| R1e | scene sampling → `rir/scene/sampling.py` | 18 個定義，619 行 |

`hybrid_rir.py` 從 3,240 行降到 **2,585 行**（−20%）。舊的 `rir_scene` /
`rir_materials` 保留為 re-export shim；`hybrid_rir` 以別名保留全部既有名稱。

**`HybridRIRConfig` 為何落在 layer 0**：`scene/sampling` 需要它，而 layer 2 不得
import layer 3 的 `render`，所以計畫提供的兩個位置只有 `contracts.py` 可行。
`RIRBackend` protocol 則刻意留在 `hybrid_rir` 等 R2——它的簽名要引用 scene 型別，
放 layer 0 只能弱化成 `Any`，等 `render/backend.py` 出現才有正確的家。

**私有 helper 就地正名**：R0 分類為「應提升為 public」的 6 個，以及測試用的
`_obstacle_floor_coverage`，在新模組裡都取得正式名稱（`sample_point`、
`min_feasible_rt60`、`polygon_distance` 等），`hybrid_rir` 則以
`x as _x` 別名維持舊呼叫端。新 package 沒有底線命名的公開 API，舊 recipe 也不必改。

**R0 測試在此發揮作用，並據此修正判準**：inventory 測試原本檢查「定義於此模組」，
搬移後兩次正確地擋下改動。契約其實是「可從舊路徑 import」，因此改為
(a) `hybrid_rir.__all__` 必須等於凍結清單、(b) 定義於此的 public symbol 不得超出
清單、(c) 被外部引用的 private helper 必須仍可 import。同時替 `hybrid_rir` 補上
先前缺少的 `__all__`（§2 現況盤點列為問題之一）。

**過程中被測試攔下的兩個真實錯誤**（皆已修正）：

1. 以 `str.replace` 改名時沒有詞界保護，`max_obstacle_floor_coverage` 被誤傷成
   `maxobstacle_floor_coverage`；
2. `ast` 節點的 `lineno` 指向 `def`/`class` 那行而不含裝飾器，導致 `@dataclass`
   留在原檔成為孤兒，並疊加到後面的 `PytARDWaveBackend`。修正為取
   `min(node.lineno, decorator_list[*].lineno)`。

第二點值得記住：任何以 AST 行號搬移程式碼的工具都必須處理裝飾器，否則會產生
「能 import、但行為錯誤」的靜默損壞。

驗證：R0 三份 golden digest 完全未變（scene `d951…`、path events `772a…`）；
RIR 相關 187 項、M6 整合 30/30、全套 552 passed。全套仍有 6 個**既有**失敗，
與 R1 無關——它們引用 `phases/` 重整前的扁平腳本路徑，並缺少
`egs/rir_generation/measurements/` 資料，建議單獨修。

### R2：拆解 hybrid renderer

來源：`hybrid_rir.py`。

拆分責任：

- PyTARD/GPU adapter → `render/low_frequency/pytard.py`。
- analytic/impedance modal backend → `render/low_frequency/`。
- Pyroomacoustics backend → `render/high_frequency/pyroomacoustics.py`。
- path-event high backend → `render/high_frequency/path_event.py`。
- FDN backend → `render/high_frequency/fdn.py`。
- crossover、alignment、causal clipping → `render/crossover.py`。
- `generate_hybrid_rir` 保留為高階 orchestration。
- dataset WAV/JSON 寫入移到 CLI/storage adapter。

完成條件：`hybrid.py` 只負責 pipeline composition；任何 backend 可以以 protocol 注入並獨立測試。

#### R2 實作結果（2026-08-02）

`hybrid_rir.py` 從 2,585 行降到 **497 行**，只剩三個定義：`generate_hybrid_rir`
（241 行 orchestration）、`_realized_acoustics_metadata`、`_config_metadata`。
其餘全部成為 re-export，舊 import 路徑不變。

新增 14 個模組：

```text
rir/render/arrays.py              37   共用 [channels, samples] 整形
rir/render/backend.py             34   RIRBackend protocol
rir/render/crossover.py          262   LR crossover、energy match、對齊、causal clip
rir/render/low_frequency/
    modal_damping.py             167   材料 → per-mode 損耗（pytard 與 analytic 共用）
    pytard.py                    701   pytARD CPU/CuPy backend
    analytic_modal.py            158
    impedance_modal.py           293
rir/render/high_frequency/
    obstacles.py                 224   post-hoc 遮蔽與散射
    pyroomacoustics.py           212   production default
    path_event.py                175   M3 coherent
    fdn.py                       168   M4 early+FDN late
rir/bank/storage.py                    dataset WAV/JSON 寫檔 adapter
```

**兩處為了達成分層而做的最小行為保持改動**：

1. `PathEventHighFrequencyBackend` 原本以
   `isinstance(self, PathEventFDNHighFrequencyBackend)` 選 metadata 字串，
   使 base class 依賴自己的 subclass，兩者無法分檔。改為未加型別註解的 class
   attribute `_LATE_PATH_AIR_ABSORPTION_POLICY`，由 subclass 覆寫。未加註解是關鍵：
   `dataclass` 只收 `__annotations__`，所以它不會變成 field。行為完全等價。
2. `RIRBackend` protocol 落在 `render/backend.py` 而非 `contracts.py`。放 layer 0
   的話簽名只能寫成 `Any`（contracts 不得 import scene）；放 layer 3 才能正確
   引用 `HybridRIRScene | RoomSceneV2`。

**被測試抓到的一個搬移專屬破壞**：`_default_pytard_root()` 以
`Path(__file__).parents[1]` 定位 vendored pytARD。在舊位置那是 `puresound/`，
搬到 `rir/render/low_frequency/` 後變成 `rir/render/`，pytARD 直接找不到。已改為
錨定套件本身（`Path(puresound.__file__).parent`），對未來搬移免疫。

這一類 `__file__` 相對路徑是 AST 搬移工具**看不到**的破壞，golden fixture 也蓋
不到（fixture 走 analytic backend）。搬移含檔案系統路徑的模組時要主動 grep
`__file__`。

**工具化**：R1e 的兩個錯誤（`str.replace` 無詞界、`ast.lineno` 不含裝飾器）已寫成
scratchpad 的可重用抽取工具，R2 五個步驟共用，未再發生同類錯誤。工具另含
`check_module()` 偵測疊加裝飾器、`audit_no_mangled_identifiers()` 比對搬移前後的
識別字集合。

驗證：R0 三份 golden digest 未變；RIR 相關 151 項、M6 整合 30/30、全套 552 passed，
6 個既有失敗不變。

### R3：拆解 path-event pipeline

來源：`rir_path_events.py`。

- dataclass/schema 與 JSON round-trip → `path_events/schema.py`。
- polygon visibility、segment intersection、scene interaction → `path_events/geometry.py`。
- shoebox image-source 與 scene path generation → `path_events/generator.py`。
- directivity → `path_events/directivity.py`。
- fractional-delay kernel、render、arrival partition → `path_events/renderer.py`。

完成條件：可以只建立 PathEventSet、只做 geometry audit，或只 render 已存在的 PathEventSet。

#### R3 實作結果（2026-08-02）

`rir_path_events.py`（2,245 行）拆成六個模組而非計畫的五個：
`schema` 663、`geometry` 479、`interactions` 636、`generator` 398、
`directivity` 60、`renderer` 256。多出的 `interactions.py` 是因為
`augment_scene_path_events_with_interactions` 單一函式就有 536 行，硬併入
`geometry.py` 會讓該檔逼近 1,100 行，違反 §5 的規模準則。

搬移時漏掉 `_BOUNDARY_GEOMETRY`——它是 `AnnAssign`（帶型別註解的賦值），而我的
常數蒐集只掃 `ast.Assign`。已補；此後的模組級常數改為同時掃兩種節點。

### R4：拆解 metrics 與 spatial renderer

來源：`rir_metrics.py`、`multiband_fdn.py`、`spatial_late_field.py`、`spatial_rir.py`、`binaural_renderer.py`。

- metrics 按 temporal/spectral/density/spatial 分開。
- `analyze_rir` 保留為 report facade。
- FDN design/render 與 spatial late-field assembly 分開。
- Ambisonics decoder 與 BRIR renderer 不依賴 bank 或 calibration。

完成條件：mono、array、Ambisonics、binaural 路徑共用同一套低層 metrics 與 contracts。

#### R4 實作結果（2026-08-02）

`rir_metrics.py`（1,516 行）拆成
`core`/`temporal`/`spectral`/`density`/`spatial`/`report` 六個模組。
`analyze_rir` 留在 `report.py` 作為 facade，維持所有 bank 工具的單一入口。

FDN、coupling、spatial、binaural 四個模組本來就是單一職責，這一階段對它們是
**重新定位**而非拆分，且四個都直接落在 §3 目標樹的
`render/{coupling,spatial,binaural}.py` 與 `render/multiband_fdn.py`。
`rir_attribution.py` 不屬 calibration，歸入 `metrics/attribution.py`。

### R5：拆解 calibration 與 measured pipeline

來源：`rir_calibration.py`、`rir_inverse_calibration.py`、`rir_m4_inverse_calibration.py`、`rir_m5_pipeline.py`、`rir_measured_calibration.py`、`rir_measurement_campaign.py`、`rir_constrained_residual.py`。

- measurement schema 與 filesystem audit 分開。
- loss/metrics 只處理數值輸入。
- M4/M5 inverse fit 各自成為 calibration strategy。
- measured runner 只負責載入資料、呼叫策略、輸出 fit report。
- residual model 維持 causal/decay contract，不直接依賴 CLI。

完成條件：synthetic recovery、measured fit、M4/M5 validator 可以使用相同的 renderer/metrics contract。

#### R5 實作結果（2026-08-02）

七個 calibration 模組整檔搬入 `rir/calibration/`：`loss`（M5.1 loss 契約）、
`synthetic_recovery`（M5.2）、`inverse_m4`、`inverse_m5`、`measured_runner`
（M5.3 fail-closed runner）、`measured_campaign`（M5.1 acquisition 契約）、
`residual`（M5.5）。全部原本就有 `__all__` 且職責單一，不需再拆。

### R6：拆解 M6 bank pipeline

來源：`rir_bank.py`、`rir_bank_manifest.py`、`rir_bank_qc.py`、`rir_bank_release.py`、`rir_bank_evaluation.py`、`rir_bank_production.py`。

分層：

1. `bank/schema.py`：純 dataclass、hash、split policy、serialization。
2. `bank/storage.py`：WAV/metadata/index 的 filesystem 操作。
3. `bank/loader.py`：training-time loader 與 cache。
4. `bank/qc.py`：item QC 與 quarantine，不建立 production decision。
5. `bank/release.py`：variant/recipe materialization。
6. `bank/evaluation.py`：distribution、throughput、listening/downstream evidence。
7. `bank/production.py`：只讀取各項 evidence 並產生 fail-closed decision。

完成條件：loader 可以只依賴 schema/storage；M6 的 manifest hash、split disjointness、QC hash、release lineage 與 production certificate 行為完全不變。

#### R6 實作結果（2026-08-02）

六個 bank 模組搬入 `rir/bank/`：`schema`（原 manifest）、`loader`、`qc`、
`release`、`evaluation`、`production`，與 R2 已建立的 `storage` 併齊。

**`bank/__init__.py` 刻意不做任何 import。** 第一版讓它 re-export `loader`，
結果把 `torch` 拉進每一個 manifest 讀取者，`rir_bank_manifest` 從 torch-free
變成 torch-dependent——R0 的 boundary 測試立刻擋下。`schema` 必須能在沒有
torch/torchaudio 的環境載入，這個性質比 `__init__` 的便利重要。

M6 全套 30/30 通過，manifest hash、split、QC、release lineage 與 certificate
行為皆不變。

### R7：compatibility shim 與舊模組退場

- `puresound/audio/hybrid_rir.py` 改為 re-export facade。
- `puresound/audio/rir_scene.py`、`rir_path_events.py`、bank 相關舊檔案保留相容入口。
- 更新 egs 與 tests 到新路徑。
- 對私有 helper 設定 deprecation 或移為測試 fixture。
- 加入 migration guide，最後才評估刪除舊實作。

#### R7 實作結果（2026-08-02）

- **`physics/` 層補齊**：目標樹 §3 列了 `physics/`，但 R1–R6 沒有任何階段指派它。
  11 個模組搬入 `rir/physics/{impedance,wave}/` 與 `physics/propagation.py`。
- **`hybrid_rir.py` 搬成 `render/hybrid.py`**，扁平路徑改為手寫 shim。
- **33 個 compatibility shim** 覆蓋所有舊路徑。自動產生的 shim 只轉 `__all__`，
  漏了兩處外部實際依賴的私有名稱（`hybrid_rir` 的 14 個相容別名、
  `rir_bank_evaluation._paired_t_confidence_interval`），由一支「掃描全 repo
  對扁平模組的 import，逐一驗證 shim 是否具備該屬性」的檢查抓出並補齊。
- **`rir/api.py`**：37 個名稱的穩定門面。docstring 明講它會載入 renderer stack
  （含 torch），要省 import 成本就直接取用分層模組。
- **migration guide**：`docs/audio/rir_package_migration.md`，含完整新舊對照表、
  私有 helper 更名表、分層規則，以及「shim 何時該退場」的前提條件。

R0 inventory 測試在 R1–R7 期間共擋下 4 次改動，每次都不是搬錯，而是測試的判準
需要隨遷移精確化。四次修正都收斂到同一個原則：**契約是「可從舊路徑 import」，
不是「定義於該檔」**：

1. 公開介面改以 `hybrid_rir.__all__` 為準（並補上它原本缺少的 `__all__`）；
2. 私有 helper 改檢查 `hasattr` 而非定義位置；
3. 同一 package 內姊妹模組共用私有 helper 屬正常，跨 package 才算違規；
4. 自述為 `Compatibility shim` 的模組整體豁免——它存在的目的就是 re-export。

另外把「不得拉入 torch/torchaudio/pyroomacoustics/cupy」的守門從扁平 shim 擴充到
12 個正規模組（`PURE_PACKAGE_MODULES`）。這個性質原本只在會消失的 shim 上被驗證，
現在釘在遷移後大家真正會 import 的位置。

**R7 驗收**：

- 全套 564 passed / 6 failed，6 個失敗與遷移前完全相同（既有的缺檔與
  `phases/` 重整遺留的扁平腳本路徑）；
- M6 整合 30/30；R0 三份 golden digest 未變；
- 端到端實跑 `generate_m6_bank.py`（6 房）：生成 → QC 6/6 → release audit PASS
  → `status=candidate`，證明搬完全部模組後 CLI 仍可用；
- `contracts`、`scene.*`、`metrics`、`path_events`、`physics.*`、`bank.schema`
  全部可在不載入 torch 的情況下 import。

**遷移後規模**：`puresound/audio/rir/` 共 70 個模組、28,701 行；
`puresound/audio/` 下留 35 個 compatibility shim。`egs/`、`test/` 尚有 216 處
沿用舊路徑的 import——全部照常運作，依 §R7 的設計刻意不強制改寫。

## 5. 測試與驗收策略

每個階段都必須同時通過以下四類測試：

### 行為相容

- 現有 RIR 單元測試不退化。
- 同 seed 的 scene、path event、RIR metadata 保持 deterministic。
- 舊 import path 與新 import path 產生相同結果。

### 數值相容

- crossover 前後的 causality、direct arrival、energy matching 不變。
- 低頻 modal/impedance/FDTD 的 reference fixture 不變。
- path-event delay、gain、visibility 與 reconstruction error 不變。

### bank contract

- M6 manifest/release/QC/evaluation/production validator 全數通過。
- train/validation/test split 不混用。
- hash、release lineage、QC report identity 不變。

### 架構品質

- `contracts`、`scene`、`physics` 可在沒有 optional renderer 的環境 import。
- 新 package 不允許反向依賴 CLI 或 bank storage。
- 每個 production module 的 public API 有 docstring 與 type hints。
- 單一模組建議不超過約 500–700 行；超過時必須有明確理由。

## 6. 非目標

本次重構不包含：

- 改變 M4/M5/M6 的物理模型或參數。
- 把 Pyroomacoustics 替換成新的預設 backend。
- 在沒有 benchmark 的情況下調整音響效果。
- 改變 M6 release schema 或 training data split 規則。
- 將所有舊 API 一次刪除。
- 在本階段加入 neural RIR generator。

## 7. 主要風險與處理方式

| 風險 | 影響 | 處理方式 |
|---|---|---|
| 大量既有程式直接 import `hybrid_rir` | 高 | 先保留 facade 與 re-export |
| 測試依賴私有 helper | 中高 | 先分類 helper，再建立明確測試 API |
| optional backend 在 import 時被載入 | 中 | lazy import 與 capability check |
| metadata/hash 行為細微改變 | 高 | 先建立 golden fixtures 與 canonical JSON tests |
| 大檔案移動造成 git review 困難 | 中 | 每個階段只移動一個 bounded domain |
| bank schema 與 loader 同時改動 | 高 | schema/storage/loader 分三階段遷移 |
| 物理核心與 torch dtype 混用 | 中 | contracts 明確規定 NumPy/Torch 邊界 |

## 8. 第一個實作批次

第一批只做 R0，不拆任何演算法：

1. 建立 API inventory 與 import compatibility tests。
2. 建立 `RoomSceneV2`、`PathEventSet`、M6 manifest 的 golden fixtures。
3. 建立 package boundary lint/test，禁止新的跨層依賴。
4. 定義 `RIRArray`、`RenderContext`、`BackendCapabilities` 與 metadata contract。
5. 針對 `hybrid_rir.py` 的 private helper 做 public/legacy/test-only 分類。

R0 完成並通過後，才開始 R1 的實際檔案拆分。這樣可以先確定「搬家不改行為」，再逐步改善模組邊界。

## 9. 當前決策

- 採用新 `puresound/audio/rir/` domain package。
- 舊 flat modules 暫時保留為 compatibility shim。
- `hybrid_rir` 只保留高階 orchestration facade。
- core contracts 與 scene/physics 不得依賴 torch、torchaudio、CLI 或 filesystem。
- M6 bank schema 與現有 release/QC 行為視為 frozen contract。
- R0（contract freeze）、R1（contracts 與 scene）、R2（hybrid renderer）已完成。
- 正規位置為 `puresound/audio/rir/{contracts.py,scene/,render/,bank/storage.py}`；
  `rir_scene`、`rir_materials` 與 `hybrid_rir` 的對應名稱都是 re-export。
- `hybrid_rir.py` 已收斂為 497 行的 orchestration facade。
- R3（path-event pipeline）尚未開始。`rir_path_events.py`、`rir_metrics.py` 與
  bank 相關模組仍在舊位置。
