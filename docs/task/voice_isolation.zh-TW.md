# puresound.task.voice_isolation

English version: [`voice_isolation.md`](voice_isolation.md)

近場 foreground voice isolation dataset：把 speaker 保留在約 1 公尺以內，其餘更
遠的通通抑制掉。它透過自己的 row-type hooks 特化了
[NoiseSuppressionDataset](ns.zh-TW.md) 的 synthesis skeleton——整個系統只有
一條 synthesis path，所有特化都活在 override 裡。

## Class: `VoiceIsolationDataset`

在各自獨立的 config knob 之後新增了以下能力（未啟用／不存在的區塊絕不消耗
RNG，所以純粹的 noise-suppression recipe 仍能重新產生位元級一致的結果）：

| block | row type |
|---|---|
| `augmentation_realfar` | far 干擾者是從 pool manifest 抽出的**成品 loudspeaker→air→mic 錄音**，直接插入、不套用任何 RIR（用卷積模擬出的 far channel 只帶得出一條 capture chain 中屬於 LTI 的部分）。`prob` = 佔全體 items 的比例；`lone_far_prob` = 其中沒有 near speaker 的比例（target = silence，也就是絕對的「只有遠端聲音＝該壓制」範例）；`turn_taking_prob` = row 層級的比率。 |
| `augmentation_realnear` | foreground 是一段**真實的近距離麥克風錄音**（target = 它自己），干擾者優先取自同一個房間、且絕不會是同一位 speaker——真實語音被放在跟 far 側同樣真實的那組 mixture 的 KEEP 那一側，讓 keep/suppress 的分界線繼續建立在距離線索上，而不是 capture-chain 的身分線索上。 |
| `augmentation_speech.mix_mode` | 明確指定 foreground/interferer 的 level 關係：`physical` 不做任何 rescale 直接相加——實務上會落在接近 0 dB 的比例，因為 sources 在載入時做過 RMS-normalized、每個 channel 的 RIR 又做過 peak-normalized，所以真正留下來的距離線索是 DRR／殘響尾形狀／音色（tilt），不是 level；其餘 rescale 模式會抽一個該模式專屬的 SIR range，而 `distance_level` 模式會把 level 線索明確地找回來（SIR = 20·log10(d_itf/d_fg) + 依場景實際幾何抽出的 jitter）。Real-far 的 row 會跳過這個機制（因為模擬近端 vs. 真實 far 錄音的 level 比例本來就不具物理意義）。 |

Pool manifest 是一行一個 JSON object（`wav_path` / `distance_m` / `room` /
`speaker`），由 `egs/voice_isolate/scripts/build_real_recording_pool.py` 建置。

### 產出的 labels

`_emit_task_metadata` 會附加 per-sample 的純量欄位（未定義時為 NaN）：
foreground 與最強干擾者的距離／DRR、`drr_gap`、`rt60`、`n_interferers`、
`target_absent`/`target_present`、`has_background_speech`、
`near_count`/`far_count`、`mix_mode` 代碼、`realized_speech_sir`、
`noise_snr`、`overlap_fraction`、`turn_taking`。這些提供給輔助任務的監督訊號
使用（例如 [`DistHeadRegressionLoss`](../nnet/loss/dist.md)），也用於 eval 分桶
（bucketing）。

## Class: `VoiceIsolationRowPlan`

`RowPlan` 擴充了 `use_realnear` / `use_realfar`，以及 real-near 抽樣結果
（room / speaker / metadata）。

## Class: `VoiceIsolationCollateFunc`

在 `NoiseSuppressionCollateFunc` 之上，再加上前述純量 labels、選用的
`far_target` waveform，以及 background-VAD labels/references 的 collation。

## Recipe wiring

```yaml
dataset:
  task: voice_isolation      # this recipe: egs/voice_isolate/main.py
augmentation_realfar:  {used: True, prob: 0.20, lone_far_prob: 0.15,
                        pool_manifest: data/realfar_pool/voices.train.jsonl,
                        turn_taking_prob: 0.3}
augmentation_realnear: {used: True, prob: 0.15, turn_taking_prob: 0.5,
                        pool_manifest: data/realfar_pool/voices.near.train.jsonl}
```

參考用的 recipe 是 `egs/voice_isolate/config/train_dpcrn.yaml`。
