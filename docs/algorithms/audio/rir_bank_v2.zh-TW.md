# RIR bank 格式 — `puresound.audio.rir.bank`

English version: [rir_bank_v2.md](rir_bank_v2.md)

本頁是 RIR bank 格式（「M6」bank；manifest schema `puresound.rir_bank.v2`）的
權威說明：bank 如何排版、切分、檢查、發佈與升級。訓練工作如何讀取已發佈的 bank
見 [bank loader](rir_bank.zh-TW.md)。

一個 WAV 目錄無法保證 train 與 test 的房間不混用、某個 item 就是當初生成的那一個、
或它通過了任何檢查。Bank 格式把這些都變成 bank 本身可用雜湊驗證、fail-closed
的性質。

## 排版

```text
<bank>/
├── rir_bank_manifest.json         # 唯一可信來源
├── rir_bank_generation_audit.json # 生成紀錄
├── rir_bank_qc_summary.json       # QC 之後
├── indexes/{train,validation,test}.jsonl
├── qc/items/<item_id>.json        # 每個 item 一份 QC 報告
├── qc/candidate_indexes/{split}.jsonl
├── qc/quarantine/index.jsonl
└── room_000000/room_000000_000000.{wav,json}   # 32-bit float WAV + sidecar
```

## Manifest

`RIRBankManifest` 包含：

| 欄位 | 內容 |
|---|---|
| `bank_id`、`release_status` | 身分；狀態為 `draft`、`candidate` 或 `production` |
| `split_policy` | `BankSplitPolicy`：policy id、seed、各 split 比例 |
| `generator` | `BankGeneratorProvenance`：產生器 id 與版本、`code_revision`、`config_sha256`、`task_plan_sha256`、seed |
| `renderer_profiles` | `BankRendererProfile`：低／高頻 backend、scene schema、renderer 設定雜湊、`evidence_tier`（`development`、`empirical_candidate`、`production_approved`），以及可選的校準、殘差與核准報告雜湊 |
| `items` | 每個 item 一筆 `RIRBankItem`（見下） |
| `split_indexes` | 各 split JSONL 索引的路徑、SHA-256 與筆數 |
| `manifest_sha256` | 其餘所有內容之 canonical JSON 的 SHA-256 |

每筆 `RIRBankItem` 記錄 `item_id`、`room_id`、`acoustic_space_id`、`scene_id`、
`split`、`generation_seed`、`origin`（`synthetic`、`real`、`mixed`）、
`renderer_profile_id`、`signal_variant`（`physical`、`physical_residual`、
`measured`）、`level_policy`（`calibrated`、`peak_normalized`、
`native_measured`）、WAV 與 JSON 的路徑及 SHA-256、canonical scene 的 SHA-256、
音訊標頭（取樣率、channel 數、frame 數），以及 QC 狀態（`pending`、`pass`、
`fail`）與報告路徑和雜湊。

雜湊使用 canonical JSON（排序鍵、不允許 NaN）與 `canonicalize_float_wav_header`，
讓相同內容永遠產生相同位元組。

## Split

Split 以**聲學空間（acoustic space）**而非 item 為單位指派
（`puresound.m6_split.sha256_acoustic_space.v1`）：

```text
u = int(SHA-256(policy_id ‖ seed ‖ acoustic_space_id)[:8]) / 2^64
split = u < f_train 則 train；u < f_train + f_val 則 validation；否則 test
```

預設比例為 0.8 / 0.1 / 0.1。合成房間的 acoustic space id 是對房間層級 scene——
尺寸、表面、材質、環境、物件——取雜湊，不含聲源與 receiver 擺位，所以同一房間
渲染的所有 item 都落在同一個 split。指派只取決於 id，與 item 順序或 bank 大小無關。
若 task plan 會讓任何 split 變空，生成會拒絕執行。

`audit_rir_bank_manifest` 驗證：manifest 雜湊、每個 split 非空、每筆指派符合
policy、acoustic space 與 room id 在各 split 間互斥、每個資產存在且雜湊與標頭相符、
sidecar 身分與 scene 雜湊相符、split 索引與 item 相符、task plan 雜湊相符，以及
`production` 狀態有已核准的 renderer profile、通過的 QC 與固定（非 dirty）的
code revision 作為依據。

## 生成

`generate_hybrid_rir.py --emit-m6-manifest` 在所有 item 完成後寫出 manifest、
split 索引與生成紀錄；`generate_m6_bank.py` 把生成、QC 與 release 打包包成一個
指令。所有會影響內容的選項（backend、scene 版本、取樣率、長度、seed、範圍）
都進入設定雜湊，改其中一個就會得到不同的 bank。相同參數下兩次全新執行產生
相同的 `manifest_sha256`。

Resume 只在 item 記錄的 task 身分、WAV 雜湊、scene 雜湊與音訊標頭都符合目前計畫
時才沿用它；缺失或被修改的 item 會重新生成。變更過的 renderer、revision、長度或
其他生成輸入無法混入既有 bank。

## 品質控管（QC）

`run_rir_bank_qc` 以 `RIRBankQCPolicy`（`puresound.rir_bank_qc.physical.v1`）
評估每個 item，並為每個 item 寫一份報告。每項檢查有四種狀態之一：

| 狀態 | 意義 |
|---|---|
| `pass` | 可量測且在界限內 |
| `fail` | 違反硬性不變量；severity 為 `quarantine` 時該 item 被隔離 |
| `not_evaluable` | 訊號或 metadata 不足以支持可靠估計 |
| `not_applicable` | 此檢查不適用於此 item |

缺少的證據永遠不會被轉成 pass 或 fail。

**結構檢查：** 資產存在且雜湊相符；metadata 可讀；身分與 scene 雜湊相符；
音訊可讀、形狀符合宣告、有限且非靜音；峰值符合 level policy（`peak_normalized`
與 `native_measured` 要求 `|peak| ≤ 1`，`calibrated` 只要求峰值有限）；channel map 恰好涵蓋每個
channel 一次；聲速合乎物理。

**逐 channel 物理檢查**（失敗即隔離 item）：`prearrival_energy`
（`floor(d / c · fs)` 之前的能量超過峰值的 `maximum_prearrival_relative_peak`）、
`direct_arrival_timing`（偵測到的直達路徑與 `d / c` 相差 1 ms 以內）、
`insufficient_tail_energy`（50 ms 之後）、`implausible_spectral_tilt`
（超過 ±18 dB/octave）、`implausible_t20`（RT60 超出 0.02–20 s）、衰減擬合
R² ≥ 0.70 與衰減涵蓋率、octave 衰減涵蓋率，以及晚期回音密度 ≥ 0.4。

**僅供參考：** 近／遠場 DRR 關係，以及空間成對指標；後者為 `not_applicable`，
因為 bank item 的各 channel 是彼此獨立的聲源到麥克風路徑，不是同步 receiver。

QC 寫出 `qc/candidate_indexes/`（各 split 通過的 item）與
`qc/quarantine/index.jsonl`，更新 manifest 中每個 item 的狀態與報告雜湊，並把
policy 連同其內容雜湊一起存下，讓門檻改變時產生可區分的證據。失敗的 item
永遠不會進入 release 索引。

## Release

`build_m6_variant_release(source_bank, output, measured_bank_root=None, ...)`
從一個已 QC 的 bank 建出不可變的 release（`rir_bank_release.json`，schema
`puresound.rir_bank_release.v1`）。它拒絕寫入已存在的目錄。

| Variant | 內容 |
|---|---|
| `synthetic_calibrated` | 已 QC bank 的複本（`variants/calibrated/`） |
| `synthetic_peak_normalized` | 相同 item，每個 item 一個共同增益縮放到峰值 0.98（`variants/peak_normalized/`） |
| `measured_native` | 有提供時，一個通過 QC 的量測 bank（`variants/measured_native/`） |

| Recipe | Variant | Origin 權重 |
|---|---|---|
| `synthetic_calibrated` | calibrated | synthetic 1.0 |
| `synthetic_peak_normalized` | peak-normalized | synthetic 1.0 |
| `real_native` | measured | real 1.0 |
| `mixed_calibrated_real` | calibrated + measured | 預設 synthetic 0.5、real 0.5 |

沒有量測 bank 時，`real_native` 與 `mixed_calibrated_real` 會以 `blocked` 狀態
寫出並附原因。每個可用的 recipe 在 `recipes/<recipe_id>/` 下有各 split 的索引，
每個 variant 有一份聲學統計 `distribution.json`。Origin 逐 item 保持可見，而不是
併入一個沒有標記的目錄。`audit_m6_variant_release` 會從 parent 重新推導每個
child variant，並把判定快取在 release 旁（`.m6_release_audit_cache.json`）。

## 量測 RIR

`bank/measured_ingest.py`（`build_measured_m6_bank`）把公開的量測 RIR 轉成一個
`measured` bank。公開語料以直達到達為時間零點，而 bank 定義 `t = 0` 為發聲瞬間，
所以原始語料會在 `prearrival_energy` 失敗。Ingest 在不改動響應本身的前提下把
傳播延遲補回去
（`puresound.measured_time_origin.iso3382_onset_to_geometric_arrival.v1`）：

1. 找出 ISO 3382-1 起始點：第一個高於峰值下 20 dB、且高出噪音底 20 dB 的 sample。
2. 平移 channel，讓起始點落在依公開聲源–receiver 距離算出的
   `floor(d / c · fs)`，並把它之前的區段靜音。
3. 遇到以下情況時拒收該 channel 而不是修補：起始點前有獨立的更早到達
   （`earlier_arrival`）、靜音會移除超過 1 % 的能量（`removed_energy`）、或平移量
   不合理（`implausible_shift`）。一個 channel 被拒收即整個 item 被拒收。

沒有公開環境資料的語料假設為 20 °C、50 % RH，並記在每個 item 的 scene 中，讓 QC
用同一個聲速重算到達時間。對齊後的 item 與合成 item 經過相同的 QC。量測資料只
在授權與來源允許衍生訓練使用時才可使用。

## 升級為 production

技術上有效的 bank 是 `candidate`。升級需要單元測試無法提供的證據，由
`evaluate_m6_release` 收集、`build_m6_production_decision` 認證
（`rir_bank_production_decision.json`）：

- `empirical` 層級的受控聽測，至少 `MINIMUM_EMPIRICAL_PARTICIPANTS`（20）位
  受試者，含 hidden reference 與 degraded anchor；沒有受試者的執行是
  `contract_fixture`，不算數；
- 房間互斥的下游訓練與評估；
- renderer profile 在 QC 之前就核准（`production_approved`），因為 QC summary
  綁定 manifest 雜湊；
- 四個 recipe 全部可用、所有 variant item 通過 QC、revision 固定；
- 聲學、ML 與 release 的簽核，以及可稽核的證據包。

憑證會對照不可變的 release 驗證；`require_production=True` 的 loader 會拒絕沒有
有效且已核准憑證的 release。不要繞過缺少的證據，或把 development profile 重新
標記來滿足 production 旗標。

## Release 檢查清單

1. 以固定的 recipe 與乾淨的 code revision 生成或 resume。
2. 執行 manifest 稽核與 QC。
3. 從已 QC 的 bank（以及剪除後的量測 bank，如有）建 release。
4. 稽核量測來源的授權與出處。
5. 把實證證據與實作測試分開記錄。
6. 訓練前，從最終 release 載入每個已發佈的 split。

各步驟的指令見 [RIR generation 指南](../../../egs/rir_generation/README.zh-TW.md)。
