# puresound.task.sv

English version: [`sv.md`](sv.md)

狀態：凍結的 legacy——為既有 recipe 保留，不再開發。

Speaker-embedding dataset：裁切一段 target utterance，選擇性地混入干擾
speaker（以隨機 SIR），接著套用 speed / reverb / noise / SRC / IIR / HPF /
volume augmentation——這些直接寫在 `sv.py` 裡，而不是經過
[task.ns](ns.zh-TW.md) 所用的 `DeviceChain`。輸出只有一段增強過的 waveform，
加上一個整數的 speaker label——沒有其他東西。這裡沒有 enrollment 輸入；以
enrollment 為條件的抽取（extraction）是 [task.tse](tse.zh-TW.md) 的工作，
不屬於這個模組。

## Class: `SpeakerEmbeddingDataset`

繼承自 [`DynamicBaseDataset`](../dataset/dynamic_base.zh-TW.md)。

### Constructor

```python
SpeakerEmbeddingDataset(
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

constructor 就是 `DynamicBaseDataset` 的，跟
[`NoiseSuppressionDataset`](ns.zh-TW.md) 一樣。augmentation 區塊是 keyword
引數，只接受類別屬性 `AUGMENTATION_BLOCKS` 裡列出的名稱；未列出的名稱會丟出
`TypeError`。這個類別沿用 base 的整組區塊，只替換其中一項：

| keyword | config model | 作用 |
| --- | --- | --- |
| `augmentation_speed_args` | `DiscreteSpeedAugmentation` | `speed_change`（以索引挑選的速度倍率 list）與 `treat_as_new_speaker`（見下文），取代 base 的 `ContinuousSpeedAugmentation` |

`__getitem__` 會讀 `augmentation_speech_args`（干擾者；`add_n_cases` 是純量
`int`）、`augmentation_noise_args`、`augmentation_reverb_args`、
`augmentation_speed_args`、`augmentation_ir_response_args`、
`augmentation_src_args`、`augmentation_hpf_args` 與
`augmentation_volume_args`。base 另外也註冊了 `augmentation_compressor_args` 與
`vad_label_args`；它們會被接受但不會被讀取，所以這個 dataset 不套用 compressor，
也不輸出 VAD label。noise-suppression 專屬的區塊（`augmentation_codec_args`、
`augmentation_packet_loss_args`、`augmentation_target_absent_args`……）沒有註冊。
也沒有 `enroll_speech_args` 參數。

### `__getitem__(key) -> Dict`

item key 由 `parse_item_key` 解析，與 [task.ns](ns.zh-TW.md) 相同：一組
`(speaker, sample_rate)`，可再延伸出 [`SpeakerSampler`](sampler.zh-TW.md) 能提供的
逐 item seed、列長度、epoch 與 utterance。

只回傳兩個 key：

| Key | 說明 |
|---|---|
| `noisy_speech` | `Tensor [1, T]`，增強過的 waveform |
| `speaker_id` | `int`，`self.spk2idx[target_speaker]`，可能會被位移（見下文） |

沒有 `clean_speech`（或任何其他）key——分類 loss（`AAMsoftmax`、GE2E、……）
只需要增強過的 waveform 跟它的整數 label。

### Augmentation 順序

每個區塊各自由自己的 `used`/`prob` 獨立控制，跟 `task.ns` 是同一套慣例：

crop -> interferer SIR-mix -> clip guard -> speed -> reverb -> noise -> SRC ->
2nd-order IIR -> HPF -> volume -> 最終裁切到 `training_sample_length`

備註：
- 跟 [task.ns](ns.zh-TW.md)/[task.tse](tse.zh-TW.md) 不同，這裡沒有
  source-level（依每個 source 的距離／房間）reverb 路徑——只有一個
  whole-mix RIR 區塊，在干擾者混音之後套用到 `noisy_speech` 上。這個
  dataset 完全沒有 near/far 距離的概念。
- 干擾者區塊會混入恰好
  `augmentation_speech_args["add_n_cases"]` 位其他 speaker，以一個從
  `snr_range` 抽出的 SIR 混音。這裡的 `add_n_cases` 必須是純量 `int`——跟
  `task.ns`/`task.voice_isolation` 不同，它們還接受 `[low, high]` 這種
  range，用來抽出隨機的干擾者人數。
- Reverb/noise/SRC/IIR/HPF/volume 全部都只會轉換 `noisy_speech`——這裡沒有
  平行的 clean-speech 路徑需要保持同步，因為根本沒有 clean 輸出。

### 把 speed perturbation 拿來當 speaker-label 的技巧

`augmentation_speed_args["speed_change"]` 是一個**速度倍率的 list**（例如
`[0.9, 1.1]`），不是像 `task.ns` 的 `speed_range` 那種 `[low, high]`
range。實際使用時是用索引（`sp_idx`）挑其中一個。當
`augmentation_speed_args["treat_as_new_speaker"]` 為真時，回傳的
`speaker_id` 會被位移 `(sp_idx + 1) * len(self.spk2idx)`——也就是說，同一個
speaker 的每一種速度變化版本，都會變成一個*獨立*的分類類別。這正是
Kaldi 風格「把 speed augmentation 當成新 speaker」的技巧；可參考
`egs/speaker_embedding/conf/PS-spk-v1.yaml` 的 class-loss 設定：
`n_classes: 21615  # equal to 7205 * 3 (speed up / slow down)`。

## Class: `SpeakerEmbeddingCollateFunc`

```python
{
    "noisy_speech": Tensor,  # [N, T], padded
    "target": Tensor,        # [N], int64, renamed from speaker_id
}
```

collate 出來的 batch 就只有這樣——沒有 `sr`、`length`，也沒有 waveform
target key（對照 [`NoiseSuppressionCollateFunc`](ns.zh-TW.md)，它還會把
`audio_sr -> sr`、`audio_length -> length` 改名）。

## Recipe wiring

參考用的 recipe 是 `egs/speaker_embedding/main.py` +
`egs/speaker_embedding/conf/PS-spk-v1.yaml`，搭配的是
`EncPredClassBase`（[system.siso](../system/siso.zh-TW.md)）跟一個 `AAMsoftmax`
loss：

```python
from puresound.task.sv import SpeakerEmbeddingDataset, SpeakerEmbeddingCollateFunc
from puresound.task.sampler import SpeakerSampler

train_dataset = SpeakerEmbeddingDataset(
    metafile_path="data/vox12_train",
    min_utt_length_in_seconds=4.0,
    min_utts_in_each_speaker=10,
    target_sr=16000,
    training_sample_length_in_seconds=4.0,
    audio_gain_normalized_to=-22,
    augmentation_noise_args={
        "used": True, "prob": 0.5, "noise_folder": "musan_noise/",
        "snr_range": [10, 30], "prob_white_noise": 0.2,
        "white_noise_snr_range": [20, 40],
    },
    augmentation_reverb_args={
        "used": True, "prob": 0.5, "rir_folder": "rirs/",
        "target_rir_type": "full",
    },
    augmentation_speed_args={
        "used": True, "prob": 0.5, "treat_as_new_speaker": True,
        "speed_change": [0.9, 1.1],
    },
)
train_sampler = SpeakerSampler(
    data=train_dataset.meta, total_batch=4000, n_spks=64, n_per=2,
    select_by_sr_first=False,
)
```
