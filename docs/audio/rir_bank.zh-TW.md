# RIR bank 讀取器 — `puresound.audio.rir.bank`

English version: `rir_bank.md`

訓練端存取預先生成之 RIR bank 的介面。生成／QC／release 的完整流程記錄在
[`rir_realism_algorithm.zh-TW.md`](rir_realism_algorithm.zh-TW.md) §10，M6
契約則記錄在 [`rir_bank_v2.zh-TW.md`](rir_bank_v2.zh-TW.md)。

## `PreGeneratedReleaseBank`（`bank/loader.py`）

讀取一個 M6.4 release（`rir_bank_release.json`），只服務單一 recipe、單一
split 底下的項目。

- `recipe_id` 與 `split` 為必填；讀取器在建構時會重新 audit 整個 release，
  一旦 hash 對不上就直接 fail closed。
- `split` 必須與 dataset 的 `usage_role` 一致，但這個檢查是在上一層執行
  （見下方）——`PreGeneratedReleaseBank` 本身並沒有 `usage_role` 這個參數。
- 若 release 缺少 manifest（或雖然有 M6 metadata，但 index 檔案被移除），
  會直接丟例外，而不是退化成掃描目錄。

## `PreGeneratedRoomBank`

給 pre-M6 bank 使用的 legacy 目錄式 WAV 讀取器。它會偵測沒有 manifest 的
M6 layout，並拒絕讀取，而不是把不同 split 混在一起。

## Training configuration（訓練端設定）

透過 `AudioEffectAugmentor.init_room_bank`
（`puresound/audio/augmentation.py`）接線，這個方法本身是由
`puresound/dataset/dynamic_base.py` 的 `init_augmentor` 呼叫的。YAML 設定放在
`augmentation_reverb.simulator.pregenerated` 底下，是 on-the-fly
`room_simulator` 路徑的同層手足，而不是一個獨立的頂層 key：

```yaml
augmentation_reverb:
  used: true
  simulator:
    used: true
    pregenerated:
      used: true
      bank_type: release          # 或 "room"，走 legacy 讀取器
      folder: <release root>
      recipe_id: synthetic_calibrated   # 之後可用 real_native / mixed_calibrated_real
      split: train
      usage_role: train
      require_production: false
```

有兩道檢查可以防止 split 洩漏：`init_augmentor` 會拒絕與 dataset 自身
train／validation／test 角色不一致的 `usage_role`（若 YAML 沒填，則直接
用 dataset 的角色補上）；`init_room_bank` 接著會拒絕與（此時已確定的）
`usage_role` 不一致的 `split`。任何一個對不上，都會在讀到任何一筆 RIR
之前就丟例外——一個 train 用的 dataset 物件，既不能指向 `split: test`，
也不能帶著對不上的 `usage_role`。

Provenance 會隨著每一筆樣本一起傳遞：`rir_release_sha256`、
`rir_recipe_id`、`rir_variant_id`、`rir_split`、`rir_origin`、
`rir_renderer_profile_id`，以及（存在時的）production certificate hash
（`puresound/task/ns.py`）。
