# puresound.task.ns

English version: [`ns.md`](ns.md)

通用的 noise-suppression dataset：從一份 clean-speech metafile 即時（on-the-fly）
合成 (noisy, clean) pairs。它同時是 voice-isolation task 透過 row-type hooks 特化
的**合成骨架（synthesis skeleton）**——見 [task.voice_isolation](voice_isolation.zh-TW.md)。
合成路徑只有一條；task 只能靠覆寫 hook 改變它。

## Class: `NoiseSuppressionDataset`

繼承自 [`DynamicBaseDataset`](../dataset/dynamic_base.zh-TW.md)。每個 item 的流程：

1. 解析 sampler key，為抽到的 speaker 挑一段 clean utterance，裁切或補齊到該列
   長度；
2. 規劃這一列（`_plan_row`：target-absent 或 task 自己的 row type）；
3. 給前景一個通道（`_prepare_foreground`：來自 room simulator 或 pre-generated
   bank 的 source-level RIR，或原樣保留）；
4. 選擇性加入干擾語音（`_sample_interferers`）、決定誰在何時說話
   （`OverlapGating`），再混合前景與干擾者（`_mix_foreground_with_interferers`）；
5. target-absent 列：從混音中減去前景，並把 target 歸零；
6. 選擇性加入殘留的 playback echo（需要 source-level reverb）；
7. 任一訊號削波時兩者一起縮放，接著做 speed perturbation 與 whole-mix RIR
   （source-level reverb 已套用或 plan 禁止時跳過）；
8. 選擇性在列首的環境音前導區把所有語音成分靜音；
9. 加入噪音（`NoiseStage`：依 SNR 的錄製噪音、白噪音、絕對位準的收音底噪），只加
   在混音上；
10. 把 target 存成 VAD reference，接著跑收音與傳輸鏈（`DeviceChain`：SRC、IIR、
    HPF、volume、compressor、A/D 邊界、codec、packet loss），target 跟著線性級
    一起變換；
11. 裁到該列長度並輸出 labels。

每個區塊都是 optional，不存在或未啟用的區塊不會從 RNG stream 抽任何東西——每個
機率抽樣都放在它的 guard 裡——所以新增旋鈕不會改變既有 seeded recipe 的產出。
步驟 3-10 背後的演算法見
[Scene construction](../../algorithms/augmentation/scene_construction.zh-TW.md) 與
[Device chain](../../algorithms/augmentation/device_chain.zh-TW.md)。

### Constructor

```python
NoiseSuppressionDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    audio_gain_normalized_to: Optional[int] = None,
    dataset_role: str = "train",
    pipeline_role: Optional[str] = None,
    curriculum=None,
    **augmentation,   # AUGMENTATION_BLOCKS 列出的區塊
)
```

Constructor 就是 `DynamicBaseDataset` 的。增強區塊以 `<block>_args` 形式的
keyword arguments 傳入，只接受 class 屬性 `AUGMENTATION_BLOCKS` 列出的名稱；未列出
的名稱會 raise `TypeError`。每個值是驗證過的 config model，或一個會在這裡被驗證成
該 model 的 mapping。這個 class 在基底集合上再加四個區塊：

| keyword | config model | 作用 |
| --- | --- | --- |
| `augmentation_speech_args`（基底） | `SpeechAugmentation` | 干擾者：`prob`、`add_n_cases`、`snr_range`、`is_target`、`media_voice`、`echo_playback`、`overlap_control` |
| `augmentation_noise_args`（基底） | `NoiseAugmentation` | 錄製噪音（`noise_folder` 或加權的 `noise_sources`）、SNR 範圍或分段、白噪音、`room_coloring`、`absolute_floor` |
| `augmentation_reverb_args`（基底） | `ReverbAugmentation` | room simulator、pre-generated bank 或 RIR 資料夾；`target_rir_type`；source-level 或 whole-mix |
| `augmentation_speed_args`（基底） | `ContinuousSpeedAugmentation` | 在 `speed_range` 內以 0.05 為步距的 speed perturbation，兩端都包含 |
| `augmentation_ir_response_args`、`augmentation_src_args`、`augmentation_hpf_args`、`augmentation_volume_args`、`augmentation_compressor_args`（基底） | 見 `puresound.config.augmentation` | 裝置鏈的類比級 |
| `vad_label_args`（基底） | `VadLabelConfig` | frame labels：energy backend，或延後到 GPU 做的 Silero |
| `augmentation_codec_args` | `CodecAugmentation` | codec 級（只作用在混音） |
| `augmentation_packet_loss_args` | `PacketLossAugmentation` | packet-loss 級（只作用在混音） |
| `augmentation_target_absent_args` | `TargetAbsentAugmentation` | target-absent 列：`prob`、`force_interferer` |
| `augmentation_row_initial_ambient_args` | `RowInitialAmbientAugmentation` | 環境音前導：`prob`、`lead_seconds_range`、`fade_ms` |

由這些區塊組出的三個元件——`device_chain`（`device_chain_from_blocks`）、
`noise_stage`（`NoiseStage`）與 `overlap_gating`（`OverlapGating`）——都在
`rebind_augmentation_blocks()` 裡建立，constructor 會呼叫它，所以 curriculum 在
訓練途中改動某個區塊時，這些元件也會拿到新值。

Task 專屬的區塊會被拒絕而不是被忽略：啟用的 `augmentation_speech.mix_mode` 在這裡
會 raise `ValueError`（`NoiseSuppressionRecipe` 在載入時也會拒絕）。真實錄音與
session 區塊不是 `NoiseSuppressionRecipe` 的欄位，所以設定了它們的 recipe 除非宣告
`task: voice_isolation`，否則會在 schema 驗證時失敗。

### `__getitem__(key) -> Dict`

`key` 是 sampler 給的 tuple `(speaker, sample_rate[, seed[, seconds[, epoch]]])`，
由 `DynamicBaseDataset.parse_item_key` 解析。帶 seed 時，合成用到的每個 RNG 都會
重新 seed，所以同一個 item 不論 epoch、run、worker 如何分配，都能位元級一致地重新
產生（見 [task.sampler](sampler.zh-TW.md)）。

Per-item dict：

| key | 內容 |
| --- | --- |
| `noisy_speech`、`clean_speech` | `[1, L]` 的混音與 target |
| `consistency_noise` | `noisy_speech - clean_speech` |
| `far_target` | SIR 混音後的干擾語音加總，沒有干擾者時為全零。沒有任何隨附的 loss 讀它；它是評估遠場語音洩漏時的參考訊號 |
| `speaker_id`、`audio_sr`、`audio_length` | 整數 speaker index、取樣率、長度 |
| `vad_target` 或 `vad_reference` | frame labels；Silero 延後到 GPU 標註時則是 clean reference |
| `background_vad_target` 或 `background_vad_reference` | 背景語音的同一組東西，只有真的混入干擾者時才有 |
| `DEVICE_CHAIN_SCALARS` | 裝置鏈每一級實際做了什麼（`*_applied` 為 0/1，該級沒觸發時參數為 NaN），以 float tensor 輸出 |
| `RIR_PROVENANCE_KEYS` | RIR 來源字串，取自前景通道的 metadata（沒有時取第一個干擾者的），都沒有時為空 |
| `paired_view`、`row_source_id` | 只有 task 選中這一列做第二個收音 view 時才有（見下） |

### 追蹤一列

`puresound.task.trace.recording(dataset)` 為 web 的管線檢視器逐階段記錄一列。在每個
階段邊界，骨架、`NoiseStage` 與 `DeviceChain` 會把當下的 pair 連同該階段抽到的參數交給
`dataset.trace`；每一次 `apply_rir` 也連同它的脈衝響應一起保留。記錄只複製、不抽亂數，
所以被追蹤的一列與同 seed 未追蹤的一列逐位元相同。`STAGES` 依合成順序列出階段 id；沒有
作用在這一列上的階段不留紀錄。訓練從不安裝 trace，且只追蹤主要的 device-chain view。

### Row-type hooks

骨架在每個 task 可能換上自己 row type 的地方呼叫這些 hook。每個基底實作就是通用
行為，只抽通用列需要的東西、順序不變：

| hook | 決定 |
| --- | --- |
| `_plan_row(target_speech)` | row type（`RowPlan`）；可以換掉前景 |
| `_prepare_foreground(target_speech, plan)` | 前景的通道；回傳 `(source_level_reverb, room_scene, fg_metadata, noisy, target)` |
| `_sample_interferers(...)` | 干擾語音從哪裡來；回傳 `(target, interferers, metadata)` |
| `_turn_taking_override(plan)` | 列層級的 turn-taking 機率（`None` = 用 `overlap_control` 的值） |
| `_mix_foreground_with_interferers(...)` | 前景與干擾者的位準關係；基底從 `snr_range` 抽一個 SIR，mix mode 回報為 `"legacy"` |
| `_emit_task_metadata(sample, ...)` | 列層級 labels；基底寫入 RIR provenance keys |
| `_emit_row_labels(sample, plan, ...)` | 逐 frame 或逐 turn 的 labels，與 `vad_target` 同一個格點；基底沒有 |
| `_auxiliary_chain_view_probability(plan)` | 對同一個完成的混音再抽一次裝置鏈的機率；基底回傳 0 且不抽任何東西 |
| `_emit_auxiliary_chain_view_labels(sample, plan)` | 第二個 view 的 task labels |

抽到第二個 view 時（`puresound.task.paired_views.apply_chain_views`），該列會帶
`paired_view`——第二條鏈的 `noisy_speech` 與 `clean_speech`，裁到主列長度——以及
共用的隨機 `row_source_id`。與主 view 完全相同的第二個 view 會被丟棄。

## Class: `RowPlan`

Task 在合成開始前做的 per-item 決定（dataclass）；task 子類別可以擴充它。

| 欄位 | 效果 |
| --- | --- |
| `target_absent` | 從混音中移除前景並把 target 歸零 |
| `force_interferer` | 不論機率，干擾者區塊一定觸發 |
| `force_speech_interferers` | 同上，但不消耗那次機率抽樣 |
| `skip_whole_mix_reverb` | 通道必須維持前景提供的樣子時，不套 whole-mix RIR |
| `skip_overlap_gating` | 給自己寫好 turn script 的 row type 用 |
| `speed_perturb_companions` | speed 變化也套到背景語音 reference 上 |
| `speed_factor` | 由骨架寫入：實際套用的速度，用來把 speed 前的 script 對應到 speed 後的 label 格點 |

## Class: `NoiseSuppressionCollateFunc`

把 waveform keys 補齊並堆疊，同時重新命名三個 per-item scalar：

| `__getitem__` key | batch key |
| --- | --- |
| `speaker_id` | `spkid` |
| `audio_sr` | `sr` |
| `audio_length` | `length` |

- `noisy_speech`、`clean_speech`、`consistency_noise` 保留原名，補齊到最長的 item。
- 至少一個 item 帶有 `vad_target` / `vad_reference` 時，才會補齊並放進 batch。
- `DEVICE_CHAIN_SCALARS` 每個 key 串成一個 tensor（每列都帶齊所有 key）。
- `RIR_PROVENANCE_KEYS` 以 Python list 原樣傳遞，每個 item 一個元素。
- 帶 `paired_view` 的列由 `collate_paired_views` 收成巢狀的
  `batch["paired_view"]`，其中 `source_indices` 把每個 view 對應回主列，另加一個
  per-row 的 `row_source_id` tensor。輔助 view 絕不附加到主 batch，所以一般 loss
  看到的仍是 B 個獨立列。

這裡不收：`far_target`、`background_vad_target`、`background_vad_reference`。
[`VoiceIsolationCollateFunc`](voice_isolation.zh-TW.md) 會把它們連同自己的 scalar
labels 一起加上。

### 在 `system.siso` 中的使用

[`EncDecMaskBase`](../system/siso.zh-TW.md) 在 training 與 validation step 讀
`batch["noisy_speech"]`、`batch["clean_speech"]` 與 `batch.get("vad_target")`，
並把整個 batch 傳給 `compute_loss`。Loss 透過宣告 `required_inputs` 要求額外輸入
（例如 `"batch"`、`"vad_target"`、`"background_vad_target"`、`"inactive_labels"`）；
module 經由 `invoke_loss` 只用這些輸入呼叫它（見 [system.base](../system/base.zh-TW.md)）。
`BaseLightningModule.ensure_vad_targets` 讀改名後的 `sr` key，在 GPU 上標註延後的
`vad_reference` / `background_vad_reference`；`test_step` 與 `predict_step` 讀 `sr`
來做逐指標重取樣，以及以輸入取樣率存檔。`spkid` 不會被訓練迴圈讀取；它是為了需要
逐列語者身分的工具而保留。
