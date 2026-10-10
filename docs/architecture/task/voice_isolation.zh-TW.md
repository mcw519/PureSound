# puresound.task.voice_isolation

English version: [`voice_isolation.md`](voice_isolation.md)

近場前景 voice isolation dataset：保留靠近裝置的說話者，壓掉更遠的一切。它透過
row-type hooks 特化 [NoiseSuppressionDataset](ns.zh-TW.md) 的合成骨架；合成路徑
只有一條，所有與近／遠判斷有關的東西都放在覆寫裡。

## Class: `VoiceIsolationDataset`

在 noise-suppression 的集合上再加三種 row type 與一個混音區塊。每一個都有自己的
config 區塊，不存在或未啟用的區塊不會從 RNG stream 抽任何東西，所以沒有這些區塊的
recipe 會位元級一致地重新產生。

| keyword（recipe key） | row type |
| --- | --- |
| `augmentation_realfar_args`（`augmentation_realfar`） | **real-far 列**：遠場干擾者是 pool manifest 裡「喇叭→空氣→麥克風」的完成錄音，直接插入、不套 RIR。卷積出來的遠場通道只帶有收音鏈的線性非時變部分；真實錄音還帶有位準、頻譜傾斜、換能器非線性與底噪。`prob` 是這類列的比例；`lone_far_prob` 是其中沒有近場說話者的比例（target = 靜音，也就是「只有遠場人聲 = 壓掉」的範例）；`turn_taking_prob` 是列層級的 turn-taking 機率。 |
| `augmentation_realnear_args`（`augmentation_realnear`） | **real-near keep 列**：前景是真實的近距離錄音，target 就是錄音本身。干擾者來自 real-far pool，優先同一個房間、絕不同一位 speaker，讓真實收音的語音落在「遠場那一側也是真實錄音」的同一種混音的 keep 側，使 keep/suppress 的邊界建立在距離線索上，而不是收音鏈的身分。 |
| `augmentation_session_rows_args`（`augmentation_session_rows`） | **session 列**：一位使用者、一到兩位旁人與該列自己的底噪，排成一份輪流說話的 script，並帶逐 frame 與逐 turn 的身分 labels（`puresound.task.session_rows`）。 |
| `augmentation_speech_args.mix_mode` | 明確的前景／干擾者位準關係（見下）。 |

### 決定 row type

`_plan_row` 依下列順序決定，每次抽樣都有對應區塊的 guard：

1. **Session 列**——只有區塊啟用且該列長度至少 `min_seconds` 時才考慮，而且長度
   檢查在機率抽樣之前，所以較短的長度分桶抽到的東西與沒有這個區塊時完全相同。整列
   在這裡一次渲染完成；plan 會跳過 overlap gating 與 whole-mix RIR、強制加入干擾者，
   並把 speed 變化也套到背景 reference 上。
2. **Real-near 列**——以 `prob` 抽樣。上游抽到的語料 utterance 會被丟棄、換成 pool
   錄音；不模擬合成房間，也跳過 whole-mix RIR。有載入 real-far pool 時，這一列的
   干擾者會被強制加入並取自該 pool；target 永遠存在。
3. **Real-far 列**——以 `prob` 抽樣，接著用區塊自己的 `lone_far_prob` 決定 lone-far
   （target-absent），與合成的 target-absent 比例互相獨立，讓兩種遠場來源可以分開
   調整。
4. 否則走基底 plan（合成的 target-absent 抽樣）。

### 真實錄音 pools

Pool manifest 是 JSON Lines，每筆錄音一個物件：`wav_path`（必填）、`distance_m`、
`room`、`speaker`、`mic`。由 `egs/voice_isolate/scripts/build_real_recording_pool.py`
產生。設 `stitch_to_length: true` 時，比該列短的錄音會接上同一個
`(speaker, room, mic)`——同一位說話者、同一條鏈——的其他錄音，而不是補零；在 keep
列上補零會讓一半的 target 變成數位靜音。Real-far 干擾者回報的通道 metadata 是
`{"source_receiver_distance": distance_m, "origin": "real"}`；干擾者數量來自
`augmentation_speech.add_n_cases`。

### `mix_mode`

`augmentation_speech.mix_mode.modes` 是一串 `MixModeEntry`（`name`、`prob`、
`physical`、`distance_level`、`sir_range`、`jitter_db`）。每列依 `prob` 權重抽一個
mode（權重不必加總為 1；空 list 代表 `physical`）。

- **`physical`**：把套完 RIR 的訊號直接相加、不重新縮放。來源在載入時做了 RMS
  正規化，每條 RIR 也逐通道做了峰值正規化，所以比例落在 0 dB 附近；留下來的距離
  線索是 DRR、殘響尾巴形狀與頻譜傾斜，不是位準。
- **`distance_level`**：把位準線索加回來：SIR = 20·log10(d_itf / d_fg) + jitter，
  取自場景的實際幾何與最近的干擾者。任一距離未知時，該列退回基底的單一 SIR 抽樣。
- 其他 mode 從自己的 `sir_range` 抽 SIR。

Real-far 列不走 `mix_mode`、改用基底的 SIR 抽樣，因為模擬近場通道與真實遠場錄音之間
的位準比沒有物理意義。Session 列用自己的 SIR 分佈混音，之後再加上強制的收音底噪。

### 輸出的 labels

`_emit_task_metadata` 加上 per-row 的 float scalars（`VOICE_ISOLATION_SCALAR_KEYS`，
未定義時為 NaN），只從模擬 metadata 計算——它們是監督與評估用的 labels，絕不是
推論輸入：

`foreground_distance`、`foreground_drr`、`nearest_interferer_distance`、
`strongest_interferer_drr`、`drr_gap`、`rt60`、`n_interferers`、
`target_absent`、`target_present`、`has_background_speech`、`near_count`、
`far_count`、`mix_mode`（`MIX_MODE_CODES` 的代碼：none 0、legacy 1、physical 2、
moderate 3、counter_level 4、distance_level 5、session 6）、
`realized_speech_sir`、`noise_snr`、`overlap_fraction`、`turn_taking`。

它們供輔助 head（例如 [`DistHeadRegressionLoss`](../../algorithms/losses/dist.zh-TW.md)）
與評估分桶使用。啟用 session 區塊時，每一列——不論是否為 session 列——都會帶齊
session label 契約（`SESSION_LABEL_KEYS`：`user_active`、`bystander_active`、
`turn_id`、`turn_role`、`turn_speaker`、`turn_chain`、`turn_distance`、
`row_source_id`；以及 `SESSION_SCALAR_KEYS` 診斷值），由 identity 與 proximity
losses 讀取。長度達到 `paired_view_min_seconds` 的 session 列會以
`paired_view_prob` 被選去做第二個收音 view。

## Class: `VoiceIsolationRowPlan`

擴充自 `RowPlan`，多了 `use_realnear`、`use_realfar`、real-near 抽到的錄音資訊
（`realnear_room`、`realnear_speaker`、`realnear_fg_metadata`）與 `session`（渲染好的
`SessionRender`，其他 row type 上為 `None`）。

## Class: `VoiceIsolationCollateFunc`

`NoiseSuppressionCollateFunc` 再加上：

- 上述 scalar labels 與 session scalars，每個 key 一個 tensor；
- session 的逐 frame 與逐 turn labels（`collate_session_labels`）；
- `far_target`，跟其他 waveform 一樣補齊（沒有的列補零）；
- `background_vad_target` / `background_vad_reference`：batch 裡只要有任何一列帶
  有，其餘沒有的列就補零。

## Recipe 設定

`task: voice_isolation` 是 recipe 的頂層 key；它選中 `VoiceIsolationRecipe`，也是
唯一帶有真實錄音與 session 區塊的 schema。

```yaml
task: voice_isolation      # driver: egs/voice_isolate/main.py
augmentation_realfar:
  used: True
  prob: 0.2
  lone_far_prob: 0.15
  turn_taking_prob: 0.3
  pool_manifest: data/realfar_pool/voices.train.jsonl
  stitch_to_length: True
augmentation_realnear:
  used: True
  prob: 0.15
  turn_taking_prob: 0.5
  pool_manifest: data/realfar_pool/voices.near.train.jsonl
  stitch_to_length: True
augmentation_session_rows:
  enabled: True            # 這個區塊用 `enabled`，不是 `used`
  prob: 0.5
  min_seconds: 12.0
```

Recipe 會拒絕 `augmentation_session_rows` 與 `augmentation_speech.is_target: true`
同時出現（旁人會變成 target 的一部分），也拒絕它與
`augmentation_row_initial_ambient` 同時出現（前導區會抹掉 turn script 宣稱存在的
語音）。完整設定見 `egs/voice_isolate/config/` 裡已發布的 recipes。
