# RIR bank loader — `puresound.audio.rir.bank.loader`

English version: [rir_bank.md](rir_bank.md)

預先生成 RIR bank 的訓練端介面：先挑一個房間，再依聲源角色各挑一個 channel。
這個模組是訓練執行時唯一會 import 的 `puresound.audio.rir` 部分。Bank 格式本身
（manifest、split、QC、release）定義在 [RIR bank 格式](rir_bank_v2.zh-TW.md)。

所有 loader 都提供與即時 [room simulator](room_simulator.zh-TW.md) 相同的兩個
呼叫，所以兩者的資料增強路徑相同：

```python
scene = bank.sample_scene()                  # 整個混音共用一個房間
impulse, meta, sr = bank.select_channel(
    scene, source_role="foreground",         # 或 interferer / media / echo / 其他
    distance_range_override=None,            # 可選 [lo, hi]，單位公尺
)                                            # impulse：[1, L]
```

## `PreGeneratedRoomBank`

```python
PreGeneratedRoomBank(
    folder, near_labels=("near_0", "near_1"), far_labels=("far_0", "far_1", "far_2"),
    drr_window_ms=2.5, wav_name="rir_5ch.wav", meta_name="metadata.json",
    cache_size=64, split=None, manifest_name="rir_bank_manifest.json",
    include_failed_qc=False, allow_legacy_layout=True,
)
```

它讀兩種排版：

- **Manifest bank**（存在 `rir_bank_manifest.json`）。必須給 `split`——多 split 的
  bank 不選一個就不能讀。Loader 會驗證 manifest 內容雜湊，只提供該 split 的
  item，跳過 QC 失敗的 item（除非 `include_failed_qc`），並在 `candidate` 或
  `production` bank 中跳過 QC 仍在 pending 的 item。被提供的 item 若 RIR 檔
  遺失，或 metadata 無法讀取、缺少 loader 讀取的 scene 結構，會在建構時就拋出
  例外，而不是訓練到一半才失敗。
- **目錄 bank**（沒有 manifest）。每個房間子目錄放一組 `rir_5ch.wav` +
  `metadata.json`，或同名的 `<item>.wav` + `<item>.json`。沒有 manifest 的 bank
  都以此方式讀取，例如 `egs/rir_generation/tools/measured/real_rir_to_bank.py`
  寫出的量測語料 bank，以及由多個來源組成的訓練 view。這裡不接受 `split=`，
  因為沒有任何東西記錄 split。若資料夾看起來是遺失了 manifest 的 manifest bank——
  有完整的 `indexes/` 或 item metadata 帶有 bank 欄位——loader 會拒絕讀取，
  而不是把各 split 混在一起。`allow_legacy_layout=False` 會完全停用這種排版。

**Scene 與 channel 的選擇。** `sample_scene()` 均勻挑一個房間，並依 label 把它的
channel 分成近場池與遠場池；沒有 label 相符時，以距離中位數切分。
`select_channel` 對 `foreground` 從近場池抽，對 `interferer`、`media`、`echo`
從遠場池抽，其他角色從全部 channel 抽，並避免在同一個 scene 內重複使用 channel。
`distance_range_override` 只保留範圍內的 channel，沒有時退而取最接近範圍中心的
channel。回傳的 metadata 帶有聲源–receiver 距離、在 `drr_window_ms` 上計算的
DRR、RT60、label、房間與 split 身分、origin、renderer profile 與 QC 報告。WAV 以
容量 `cache_size` 個房間的 LRU cache 保存。增強器會用透明的重採樣器把 impulse
轉到資料集的取樣率。

## `PreGeneratedReleaseBank`

```python
PreGeneratedReleaseBank(
    folder, *, recipe_id, split,
    release_manifest_name="rir_bank_release.json",
    require_production=False, production_decision_name="rir_bank_production_decision.json",
    audit=True, audit_cache=True, **bank_kwargs,
)
```

一次提供一個 release 中某個 recipe 的一個 split。建構時會稽核 release，任何不符
都 fail closed（`audit_cache` 重用存在 release 旁的判定；已經稽核過的呼叫端，
例如第二個 DDP rank，可用 `audit=False` 跳過）。`require_production=True` 時
還要求一份有效且已核准的 production 憑證。Recipe 必須是 `ready`；它的每個
variant 都成為請求 split 上的一個 `PreGeneratedRoomBank`，且它們的 item 總數必須
等於 recipe 索引的筆數。

`sample_scene()` 分三步抽樣——依 recipe 固定的 origin 權重抽 origin，再抽該 origin
的一個 variant，最後抽一個 item——所以合成／量測混合 recipe 依循其權重，而不是
依磁碟上的檔案數。Scene 與 channel metadata 帶有 `release_id`、`release_sha256`、
`release_recipe_id`、`release_variant_id`、`release_origin` 與 production 憑證雜湊。

## `UnionRoomBank`

以固定的各 bank 抽樣機率把多個 bank 當成一個池（權重是機率，不是 item 數）。每個
成員保有自己的 label、DRR 窗與 cache，每個 scene 都會被送回產生它的 bank。
`set_weights({name: weight})` 重新設定部分成員的權重後重新正規化；curriculum 用它
在 epoch 之間移動房間池（`puresound.config.curriculum`）。權重可以是零，但不能
全部為零。

## 設定

Bank 設在 `augmentation_reverb.simulator.pregenerated`，與即時 simulator 的設定
並列（`puresound.config.augmentation`：`PreGeneratedBankConfig`、
`RoomBankMemberConfig`）。`folder` 與 `banks` 必須恰好給其中一個。

```yaml
augmentation_reverb:
  used: true
  simulator:
    used: true
    pregenerated:
      used: true
      bank_type: release              # 預設：有 recipe_id 時為 release，否則為 room
      folder: <release root>
      recipe_id: synthetic_calibrated # 可用時為 real_native / mixed_calibrated_real
      split: train
      usage_role: train
      require_production: false
```

Union 改為列出成員；頂層只能共用 `usage_role`：

```yaml
    pregenerated:
      used: true
      banks:
        - {name: simulated, weight: 0.6, bank_type: room, folder: <bank>}
        - {name: measured,  weight: 0.4, bank_type: room, folder: <bank>}
```

**防止 split 外洩。** 資料集以自己的 pipeline role 填入 `usage_role`，並拒絕與之
不符的 `usage_role` 設定（`puresound/dataset/dynamic_base.py`）。接著 `release`
bank 要求 `split` 等於 `usage_role`，所以訓練資料集不能被指向 validation 或 test
的房間；兩項檢查都在讀任何 RIR 之前就 raise。`room` bank 不會把 `split` 與角色
交叉檢查，且會拒絕 release 專用選項（`recipe_id`、`require_production`、
`audit`……）。

## 樣本中的出處

資料集會把 RIR 身分複製進每個樣本（`puresound/task/ns.py` 的
`RIR_PROVENANCE_KEYS`）：`rir_release_id`、`rir_release_sha256`、`rir_recipe_id`、
`rir_variant_id`、`rir_split`、`rir_origin`、`rir_renderer_profile_id`、
`rir_production_certificate_sha256` 與 `rir_interferer_variant_ids`。Bank 沒有
提供的欄位為空字串。
