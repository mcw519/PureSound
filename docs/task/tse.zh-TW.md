# puresound.task.tse

English version: [`tse.md`](tse.md)

> **Status: legacy** —— 維持可運作但已凍結：不新增功能、不重寫。

Target Speaker Extraction（TSE）dataset：跟 [task.ns](ns.zh-TW.md) 一樣會建出
一個 target-vs-interferer-vs-noise 的 mixture，但每個 item 還會額外帶一段
獨立增強過的 **enrollment** utterance（同一位 target speaker 的另一段錄音，
用來讓模型知道該抽取哪個聲音）。

這個 dataset **不會**繼承 [`NoiseSuppressionDataset`](ns.zh-TW.md)、也不使用
它的 row-type-hook skeleton——它直接繼承 `DynamicBaseDataset`，用自己獨立
的 `__getitem__`（沒有 `RowPlan`、沒有 overlap-gating/turn-taking、也沒有
real-recording rows）。它只重用了 `DynamicBaseDataset` 同樣有提供給
`task.ns` 的那幾個底層 per-source-RIR 輔助方法
（`should_apply_source_level_reverb`、`apply_source_level_target_reverb`、
`apply_source_level_interferer_reverb`）。

## Class: `TargetSpeakerExtractDataset`

### Constructor

```python
TargetSpeakerExtractDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    enroll_speech_args: EnrollmentConfig | Mapping | None = None,
    audio_gain_normalized_to: Optional[int] = None,
    augmentation_speech_args=None,   # interferers: used/prob/add_n_cases(int)/snr_range/is_target
    augmentation_noise_args=None,
    augmentation_reverb_args=None,
    augmentation_speed_args=None,
    augmentation_ir_response_args=None,
    augmentation_src_args=None,
    augmentation_hpf_args=None,
    augmentation_volume_args=None,
    vad_label_args=None,
    dataset_role: str = "train",
    pipeline_role: Optional[str] = None,
)
```

**沒有 `role_based_rir` 這個參數**——這個字串在這份 788 行的檔案裡完全沒有
出現過，整個 repo 裡也找不到。Foreground 跟 interferer 的 reverb 用的是跟
`task.ns` 一樣的機制：當 `augmentation_reverb.simulator.source_level` 開啟時
是 source-level 的 per-source RIR，否則就是混音後套用一個 whole-mix
RIR。這裡的 `add_n_cases` 同樣必須是純量 `int`（不是
`task.ns`/`task.voice_isolation` 還額外支援的 `[low, high]` range）。

### `enroll_speech_args` 是必填的

dataset 接受 `EnrollmentConfig` 或 mapping，並透過與 recipe 相同的 Pydantic
model 驗證。若傳入 `None`，會在初始化前直接丟出
`ValueError("enroll_speech_args is required")`。

| Key | 用途 |
|---|---|
| `enroll_length_seconds` | enrollment 片段的裁切／補零長度（秒） |
| `gain_normalized_to` | 若為 truthy，套用在 enrollment 片段上的 RMS 目標音量（dB） |
| `add_noise` | `{used, prob, noise_folder, snr_range, prob_white_noise, white_noise_snr_range}` |
| `add_reverb` | `{used, prob, rir_folder, target_rir_type}`，或者用一個 `simulator: {used: True, ...}` 子區塊取代 `rir_folder` |
| `add_volume` | `{used, prob, clipping_prob, clipping_range: {min, max}, perturbed_range}` |
| `add_inactive_target` | `{used, prob}`——見[下方說明](#add_inactive_target) |

（`egs/target_speaker_extraction/config/default_config.yaml` 裡的
`enroll_speech:` 區塊,就是這個形狀一份完整、可運作的範例。）

### `get_enroll_speech(target_speaker, batch_sr)`

從 `target_speaker` 抽一段 utterance，接著透過它**自己專屬**的
`AudioEffectAugmentor`（`self.enroll_augmentor`，其資料夾／simulator 是從
`enroll_speech_args` 的 `add_noise`/`add_reverb` 讀進來的——跟建構 mixture
用的 `self.augmentor` 完全分開）獨立做裁切／正規化／reverb／加噪音／調整
音量。這一步會先跑，在 mixture 本身被建出來之前，所以 enrollment 片段的
augmentation 抽樣結果跟 mixture 的抽樣結果不會共用 RNG 狀態。

### `__getitem__((speaker, sr)) -> Dict`

只有 2-tuple 這種形式——這裡沒有 seeded/deterministic 的 3-tuple item（相對
於 [task.ns](ns.zh-TW.md)）；這個 dataset 沒有實作
[`SpeakerSampler`](sampler.zh-TW.md) 的 `item_seed` 重新 seed 機制。

順序（由上到下逐行確認過）：為要求的 speaker 抓 enrollment 語音 -> 為同一位
speaker 挑 foreground utterance -> 選擇性地把 foreground 換成一位不相干
speaker 的語音（`add_inactive_target`，見下文）-> 裁切 -> foreground 的
reverb（source-level RIR，或者在 whole-mix 路徑下維持乾聲）-> 選擇性地混入
干擾者（各自在 source level 做 reverb，或者維持乾聲；加總後以一個從
`augmentation_speech_args.snr_range` 抽出的 SIR 混入）-> clip guard ->
speed -> whole-mix RIR（只有在 foreground 的 reverb *不是* source-level 時
才會做）-> 背景噪音（+ 選擇性的白噪音）-> SRC -> 2nd-order IIR -> HPF ->
volume -> 最終裁切。

回傳：

| Key | 說明 |
|---|---|
| `noisy_speech` | `Tensor [1, T]`，完整的 mixture |
| `clean_speech` | `Tensor [1, T]`，target 的 direct-path 參考訊號（`add_inactive_target` 的 row 會是全零，見下方說明） |
| `enroll_speech` | `Tensor [1, T_enroll]`，獨立增強過的 enrollment 片段 |
| `added_noise` | 實際加入的背景噪音，或 `None` |
| `consistency_noise` | `noisy_speech - clean_speech` |
| `speaker_id` | `self.spk2idx[target_speaker]`——永遠是真正 enrolled 的那位 speaker，`add_inactive_target` 的 row 也一樣 |
| `audio_sr`、`audio_length` | 跟 `task.ns` 一樣 |
| `vad_target` | 只有在設定了 `vad_label_args` 時才會出現 |

### `add_inactive_target`

當 `enroll_speech_args["add_inactive_target"]["used"]` 觸發時（機率為
`prob`），dataset 會把 foreground utterance 換成一位不相干、隨機選出的
speaker 的語音——用意是模擬「enrolled 的那個聲音在這段裡完全沒有出現」，
一個教會模型輸出靜音的負樣本（negative example）。Enrollment 片段
本身不受影響：它是在這次替換之前,就已經從真正的 `target_speaker` 抓好的。

在 `__getitem__` 的尾端，參考 waveform 會被歸零：

```python
# Warp target speech to zeros
if inactive_target_speaker is not None:
    target_speech = torch.zeros_like(target_speech)
```

跟 [`task.ns`](ns.zh-TW.md)（`puresound/task/ns.py`，約第 441 行）是同一套寫法。
只要這個分支被觸發：

- `clean_speech` 會是全零，所以這些 row 在 waveform 層級的訓練目標確實就是靜音。
- `consistency_noise`（`noisy_speech - target_speech`）因此會等於完整的 mixture。
- `vad_target` 會是全零，由 `create_empty_vad_target(target_speech)` 算出來——
  這個函式只讀 `target_speech.shape`，完全不會讀它的數值。
- `speaker_id` 仍然是真正 enrolled 那位 speaker 的 index：`target_speaker` 這個
  id 字串是刻意保持不動的，因為幾行之後還要拿它當
  `self.spk2idx[target_speaker]` 的查表 key。

> **本次已修正：** 這行歸零的程式碼先前賦值的對象是 `target_speaker`（那個
> speaker-id 字串）而不是 `target_speech`，跟它自己的註解相反。結果是
> `clean_speech` 沒被歸零，*而且*任何抽到這個分支的 row 都會在 `spk2idx` 查表時
> raise `KeyError`、讓 worker 當掉——這大概正是為什麼這個 repo 裡的每一份
> recipe config 都設定 `add_inactive_target: {used: False}`。

## Class: `TargetSpeakerExtractCollateFunc`

把四個 waveform key 做 padding，並改名兩個純量 key，跟
[`NoiseSuppressionCollateFunc`](ns.zh-TW.md) 是同一套慣例：

| `__getitem__` key | collate 後的 batch key |
|---|---|
| `enroll_speech` | `conditional_speech` |
| `speaker_id` | `target` |
| `audio_sr` | `sr` |
| `audio_length` | `length` |

`noisy_speech`、`clean_speech`、`consistency_noise` 保留原名。`added_noise`
不會被 collate（跟 `task.ns` 是同樣的缺口）。`vad_target` 只有在 batch 裡
至少有一個 item 帶有它時才會被 padding 並包含進來。

## Recipe wiring

參考用的 recipe 是 `egs/target_speaker_extraction/main.py` +
`egs/target_speaker_extraction/config/default_config.yaml`，搭配的是
`EncDecCondMaskBase`（[system.miso](../system/miso.md)）——enrollment 片段是
第二條需要自己 encoder 的音訊串流，這正是
[system.siso](../system/siso.md) 的 `EncDecMaskBase` docstring 用來把自己
跟 MISO 情境區分開來的那個案例。

```python
from puresound.task.tse import TargetSpeakerExtractDataset, TargetSpeakerExtractCollateFunc

dataset = TargetSpeakerExtractDataset(
    metafile_path="data/libri_train",
    min_utt_length_in_seconds=4.0,
    min_utts_in_each_speaker=5,
    target_sr=16000,
    training_sample_length_in_seconds=6.0,
    audio_gain_normalized_to=-28,
    enroll_speech_args={
        "enroll_length_seconds": 6,
        "gain_normalized_to": -28,
        "add_inactive_target": {"used": False, "prob": 0.1},
        "add_noise": {
            "used": True, "prob": 0.5, "noise_folder": "musan_noise/",
            "snr_range": [10, 30], "prob_white_noise": 0.1,
            "white_noise_snr_range": [10, 30],
        },
        "add_reverb": {"used": False, "prob": 0.5, "rir_folder": "rirs/",
                       "target_rir_type": "full"},
        "add_volume": {"used": False, "prob": 0.5,
                       "perturbed_range": [0.2, 1.2], "clipping_prob": 0.1,
                       "clipping_range": {"min": [0.0, 0.1], "max": [0.9, 1.0]}},
    },
    augmentation_speech_args={"used": True, "is_target": False, "prob": 1.0,
                               "add_n_cases": 2, "snr_range": [-20, 20]},
)
```
