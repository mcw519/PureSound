# puresound.task.ns

English version: [`ns.md`](ns.md)

通用的 noise-suppression dataset：從一份 clean-speech metafile 即時（on-the-fly）
合成 (noisy, clean) pairs。這同時也是 voice-isolation task 所特化（specialize）
的**合成骨架（synthesis skeleton）**——見 [task.voice_isolation](voice_isolation.zh-TW.md)。

## Class: `NoiseSuppressionDataset`

繼承自 `DynamicBaseDataset`。每個 item 的流程：為抽到的 speaker 挑一段 clean
utterance，選擇性地給它一個房間聲學通道（room channel，可以是 on-the-fly
simulator 產生的 source-level RIR，也可以來自 pre-generated bank），選擇性地
加入干擾 speaker／playback echo／噪音，接著跑過裝置鏈（speed、whole-mix RIR、
SRC、IIR、HPF、codec、packet loss、volume），並讓 clean target 跟著一致地變形
（warp）。

### Constructor（所有區塊皆為 optional；未啟用／不存在的區塊絕不消耗 RNG）

```python
NoiseSuppressionDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    audio_gain_normalized_to: Optional[int] = None,
    augmentation_speech_args=None,        # interferers: prob/add_n_cases/snr_range/
                                          #   media_voice/echo_playback/overlap_control
    augmentation_noise_args=None,         # background + white noise, SNR ranges;
                                          #   optional room_coloring (noise gets a
                                          #   channel of the speech's room) and
                                          #   absolute_floor (dBFS-anchored capture
                                          #   floor, mixture only)
    augmentation_reverb_args=None,        # simulator / pre-generated bank / whole-mix RIR
    augmentation_speed_args=None,
    augmentation_ir_response_args=None,
    augmentation_src_args=None,
    augmentation_hpf_args=None,
    augmentation_volume_args=None,
    augmentation_codec_args=None,
    augmentation_packet_loss_args=None,
    augmentation_target_absent_args=None, # silence-injection rows (target = zeros)
    vad_label_args=None,                  # frame labels: energy backend or deferred Silero
)
```

Task-specific 的區塊在這裡會被拒絕（`mix_mode` 會 raise；recipe 的 main 在沒有
`dataset.task: voice_isolation` 時會拒絕 `augmentation_realfar/realnear`）。

### `__getitem__((speaker, sr) | (speaker, sr, item_seed)) -> Dict`

3-tuple 形式帶有一個 per-item seed（用於 deterministic validation，由 seeded 過
的 [`SpeakerSampler`](sampler.zh-TW.md) 設定）：synthesis 用到的每一個 RNG 都會
被重新 seed，所以同一個 item 不論在哪個 epoch、哪次 run、worker 如何分配，都能
重新產生位元級一致（bit-exact）的結果。

回傳一個 per-item dict：`noisy_speech`、`clean_speech`、`added_noise`（若沒有加
噪音則為 `None`）、`consistency_noise`（`noisy - clean`）、`far_target`（SIR 混音
後的干擾者訊號加總，沒有干擾者時為全零——只有選用的 far decoder 會用到它，見
[task.voice_isolation](voice_isolation.zh-TW.md)）、`speaker_id`、`audio_sr`、
`audio_length`、VAD labels（`vad_target`，或者當 `vad_label_args.backend` 是
`silero` 時的一個 deferred `vad_reference`），以及背景語音對應的同一組
target/reference（`background_vad_target`/`background_vad_reference`，只有在
真的混入過干擾者時才會出現），再加上 `_emit_task_metadata` hook 產生的
RIR-provenance 純量欄位（`RIR_PROVENANCE_KEYS`：該 item 沒有 RIR metadata 時為
空字串／空 tuple）。

這些 per-item keys 之中有幾個會被 `NoiseSuppressionCollateFunc` **改名或直接
丟棄**——見下文。

### Row-type hooks

Skeleton 把每一個 task 可以替換自己 row type 的位置都委派（delegate）出去；
每個 base 實作本身*就是*通用行為，也不會從 RNG stream 多draws任何東西：

| hook | 決定的內容 |
|---|---|
| `_plan_row(target_speech)` | row type（`RowPlan`）；可能替換 foreground |
| `_prepare_foreground(target_speech, plan)` | foreground 的聲學通道（room sim 或原樣） |
| `_sample_interferers(...)` | 干擾語音的來源 |
| `_turn_taking_override(plan)` | row 層級的 turn-taking rate |
| `_mix_foreground_with_interferers(...)` | foreground/interferer 的 level 關係 |
| `_emit_task_metadata(...)` | 額外的 per-sample labels |

## Class: `RowPlan`

per-item 決策的 dataclass：`target_absent`、`force_interferer`、
`force_speech_interferers`、`skip_whole_mix_reverb`。Task 的 subclass 可以繼承
並擴充它。

## Class: `NoiseSuppressionCollateFunc`

把 waveform 相關的 keys 做 padding 並疊成 batch tensors，過程中還會**改名**三個
per-item 的純量 key：

| `__getitem__` key | collate 後的 batch key |
|---|---|
| `speaker_id` | `spkid` |
| `audio_sr` | `sr` |
| `audio_length` | `length` |

`noisy_speech`、`clean_speech`、`consistency_noise` 保留原名（各自 padding 到
batch 中最長的 item）。`vad_target`/`vad_reference` 只要 batch 中至少有一個
item 帶有它們，就會被 padding 並包含進來。`RIR_PROVENANCE_KEYS`
（`rir_release_id`、`rir_release_sha256`、`rir_recipe_id`、`rir_variant_id`、
`rir_split`、`rir_origin`、`rir_renderer_profile_id`、
`rir_production_certificate_sha256`、`rir_interferer_variant_ids`）只要有任何
item 帶有它們，就會以原生 Python list 的形式直接傳遞——每個 batch item 一筆，
*不會* padding／疊成 tensor。

**在這一步被丟棄的欄位**（存在於 `__getitem__` 的 per-item dict，但
`NoiseSuppressionCollateFunc` 不會讀取）：`added_noise`、`far_target`、
`background_vad_target`、`background_vad_reference`。需要它們的呼叫端得自己
寫一個 collate function，或改用
[`VoiceIsolationCollateFunc`](voice_isolation.zh-TW.md)——它在這個 base 行為之上
疊加了 `far_target`、`background_vad_*` 這一對，以及自己的純量 labels。

### 在 `system.siso` 裡如何被消費（consume）

[`EncDecMaskBase`](../system/siso.md)（SISO 的 trainer）在
`training_step`/`validation_step` 裡直接讀取 `batch["noisy_speech"]`、
`batch["clean_speech"]` 和 `batch.get("vad_target")`，並把整個 batch dict 傳給
`compute_loss`，所以一個註冊進來的 loss 可以透過在自己身上設定一個 flag
屬性（`uses_batch`、`uses_vad_logits`、`uses_background_vad_logits`、
`uses_dist_preds`、`uses_vad_target`、`uses_inactive_labels`）來選擇讀取額外的
key，而不需要固定的呼叫簽章（call signature）。
`BaseLightningModule.ensure_vad_targets`（見 [system.base](../system/base.md)）
會讀取被改名後的 `sr` key，用來在 GPU 上延遲（lazily）標記一個 deferred 的
`vad_reference`/`background_vad_reference` 時挑選正確的取樣率（`silero`
backend 的情況）。`test_step`/`predict_step` 也會讀取 `sr`，用於逐 metric 的
resampling，以及以原始取樣率存下推論輸出。改名後的 `spkid` key 會一路被帶到
collate 之後，但 `siso.py` 本身完全不會讀它——它是留給下游工具用的（例如把
training samples 依 speaker 傾印出來），不是給 training loop 用的。
