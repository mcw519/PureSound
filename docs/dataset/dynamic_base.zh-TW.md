# puresound.dataset.dynamic_base

English version: [dynamic_base.md](dynamic_base.md)

給語音任務用的 base dataset class，內建可組合的動態增強 pipeline。
`puresound.task.ns.NoiseSuppressionDataset` 與
`puresound.task.voice_isolation.VoiceIsolationDataset`（以及 legacy 的
`puresound.task.sv` / `puresound.task.tse` dataset）全都繼承自這裡，並覆寫
`__getitem__`。

## Class: `DynamicBaseDataset`

繼承 `torch.utils.data.Dataset`。負責 metafile 的載入/過濾、共用的
`AudioEffectAugmentor`，以及 VAD-labeler 的分派邏輯；實際每筆資料的合成則由
subclass 實作。

### Constructor

```python
DynamicBaseDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    audio_gain_normalized_to: Optional[int] = None,
    augmentation_speech_args: AugmentationArg = None,
    augmentation_noise_args: AugmentationArg = None,
    augmentation_reverb_args: AugmentationArg = None,
    augmentation_speed_args: AugmentationArg = None,
    augmentation_ir_response_args: AugmentationArg = None,
    augmentation_src_args: AugmentationArg = None,
    augmentation_hpf_args: AugmentationArg = None,
    augmentation_volume_args: AugmentationArg = None,
    vad_label_args: AugmentationArg = None,
    dataset_role: str = "train",
    pipeline_role: Optional[str] = None,
    curriculum: AugmentationArg = None,
)
```

每一個增強/VAD/role 參數都是有預設值的明確 keyword argument —— **沒有**
`**kwargs` 這種萬用參數。每個長度單位都是**秒**，不是 samples：

- `metafile_path` – CSV metafile 路徑（由
  [`MetafileParser.read_from_metafile`](parser.md) 解析）
- `min_utt_length_in_seconds`（預設 `3.0`）– 短於這個秒數的 utterance 會在
  `gen_meta()` 裡被丟掉
- `min_utts_in_each_speaker`（預設 `5`）– 篩完之後 utterance 數不足這個值的
  speaker 會整個被丟掉
- `target_sr` – 若有設定，`self.training_sample_length = int(target_sr *
  training_sample_length_in_seconds)`（單位 samples）；若為 `None`，
  `self.training_sample_length` 就維持 `None`
- `training_sample_length_in_seconds`（預設 `6.0`）
- `audio_gain_normalized_to` – 目標 dBFS，會傳給這個 base class 內部每次開檔的
  `AudioIO.open(target_lvl=...)`
- `augmentation_speech_args`、`augmentation_noise_args`、
  `augmentation_reverb_args`、`augmentation_speed_args`、
  `augmentation_ir_response_args`、`augmentation_src_args`、
  `augmentation_hpf_args`、`augmentation_volume_args` – 經過驗證的 Pydantic
  capability model；也可傳 mapping，由這個邊界負責驗證。這裡只會真正用到
  `augmentation_noise_args` 跟 `augmentation_reverb_args`（在
  `init_augmentor()` 裡）—— 其餘的由 subclass 的 `__getitem__` 自行讀取
- `vad_label_args` – 見下方 [VAD-labeler 分派邏輯](#vad-labeler-分派邏輯-init_vad_labeler)
- `dataset_role`（預設 `"train"`）– 真正的 dataset stage。
- `pipeline_role`（預設等於 `dataset_role`）– 供 RIR bank 等 stage-sensitive
  capability 選擇資料來源分布。
- `curriculum` – 已驗證的
  [`CurriculumConfig`](../configuration.md#curriculum-knobs-that-move-with-the-epoch)，
  或 `None`（旋鈕都是常數）。給了排程之後，被點名的旋鈕會跟著 sampler 附在
  每個 item 上的 epoch 走。只有 training dataset 會拿到；validation dataset
  不會，這樣 validation loss 仍可跨 epoch 比較。

### Initialization sequence

`__init__` 會呼叫 `self.init_necessary()`，依序執行：

1. `self.gen_meta(...)` —— 把它回傳的 4-tuple 拆開存進 `self.meta`、
   `self.gender_meta`、`self.gender_spks`、`self.sr_meta`
2. 建立 `self.total_spks`（排序過的 speaker id 清單）與 `self.spk2idx`
3. `self.init_augmentor()` —— 建立 `self.augmentor`
4. `self.init_vad_labeler()` —— 建立 `self.vad_labeler` / `self.gating_vad_labeler`
5. `self.apply_curriculum_epoch(0)` —— 先把排程的起始值就位，讓「還沒有人宣告
   epoch」之前抽出來的列也看得到排程開頭，而不是檔案裡的常數

---

#### `parse_item_key(key) -> ItemKey`

讀入一個 sampler key —— `(speaker, sample_rate)`，以及這次 run 有要求時才會
出現的 per-item seed、該 batch 的列長、當前 epoch（見
[task.sampler](../task/sampler.zh-TW.md)）—— 並依序套用每個 task 共用的三件事：

1. 這一列的長度，`sample_length` 會讀它；
2. 這個 epoch 的排程旋鈕，接著才是
3. per-item seed，讓 seed 管轄的每一次抽樣都已經看到該 epoch 要求的值。

每個 dynamic dataset 的 `__getitem__` 都從這裡開始。這正是讓 key 形狀不會變成
task-specific 的原因：seeded validation、混合列長、curriculum 對每個 task 的
行為都一致，而不是每個 `__getitem__` 各自長出一套 tuple arity 階梯。

---

#### `apply_curriculum_epoch(epoch)`

把該 epoch 的排程旋鈕就位；該 epoch 已生效時就是 no-op。它只重建幾個小的 config
物件，且絕不動 RNG——在這裡抽一次會讓整列後續的抽樣依「當時是第幾個 epoch」而位移。
遇到無法套用的目標只會 warning 不會 raise：recipe 在載入時就已拒絕未知目標，而在
epoch 中途殺掉 DataLoader worker 會讓 DDP 卡在下一次 all-reduce。

---

#### `rebind_augmentation_blocks()`

重新推導所有「由 augmentation block 組出來」的元件。block 是每列現讀的，但由它
組成的元件——capture chain、noise stage、gating helper——只組一次，並持有當時拿到
的那份 block。會組這些元件的 dataset 在這裡重組它們，讓 run 途中改變的值也能抵達。
Subclass 把元件建構寫在這個方法裡並由 `__init__` 呼叫，構造與重推導就不會走岔。
只做組裝：不動 RNG、不讀磁碟。

---

#### `gen_meta(metafile_path, min_utts_in_spk=10, min_utt_length=3.0)`

這裡的預設值（`min_utts_in_spk=10`）跟 constructor 自己的預設值
（`min_utts_in_each_speaker=5`）不一樣——但實務上這無關緊要，因為
`init_necessary()` 每次呼叫時都會明確傳入
`metafile_path=self.metafile_path,
min_utt_length=self.min_utt_length_in_seconds,
min_utts_in_spk=self.min_utts_in_each_speaker`，所以最後生效的一定是
constructor 的值。`gen_meta` 自己的預設值只有在你直接呼叫它時才會用到。

透過 `MetafileParser.read_from_metafile(f_path=metafile_path,
use_speaker_as_key=True)` 解析 metafile，然後：

- 把短於 `min_utt_length` 秒的個別 utterance 丟掉（`length / sr <
  min_utt_length`）
- 把篩完剩下 utterance 數不足 `min_utts_in_spk` 的 speaker 丟掉
- 從 speaker id 開頭到第一個 `_` 之前的部分，推導出每個 speaker 的
  `corpus_id`（metafile 預期 speaker 命名成 `{corpus}_{speaker}`）
- 依性別（`"m"` / `"f"` / `"other"`）與取樣率把 speaker 分桶

**回傳一個 4-tuple** `(meta, gender_meta, gender_spks, sr_meta)`：

| 回傳值 | Shape |
|---|---|
| `meta` | `{spkid: {"gender", "channels", "corpus_id", "utts": {uttid: {"path", "length", "channels", "sr"}}}}` |
| `gender_meta` | `{"m"/"f"/"other": {corpus_id: [spkid, ...]}}` |
| `gender_spks` | `{"m"/"f"/"other": [spkid, ...]}` |
| `sr_meta` | `{sample_rate: {spkid: [uttid, ...]}}`（一個 `defaultdict`） |

**副作用**：同時會把 `self.all_corpus_id`（看到的 corpus id 集合）直接設到
`self` 上——這是在 4-tuple 回傳值*之外*額外做的事。因此以 bound method 的方式
呼叫 `gen_meta`（`init_necessary()` 就是這樣做的）會有超出回傳值本身的效果。
注意這裡**沒有** corpus 層級的過濾：每個 corpus id 都會進 `all_corpus_id`。
原本這裡寫過一個過濾器，但條件是 `len(...) < 0`，永遠不成立，所以從來沒有濾掉
任何東西；它被刪除而不是修正，因為真的加上 min-speaker 過濾會改變訓練分佈。
speaker 與 utterance 層級的過濾（`min_utts_in_spk`、`min_utt_length`）是實際
有作用的，發生在上面。

---

#### `init_augmentor()`

**不吃任何參數**——讀取 `self.augmentation_noise_args` 與
`self.augmentation_reverb_args`（在 `__init__` 裡設定），建立
`self.augmentor = AudioEffectAugmentor()`。

- 若有設定 `augmentation_noise_args`：
  `self.augmentor.load_bg_noise_from_folder(augmentation_noise_args["noise_folder"])`
- 若有設定 `augmentation_reverb_args`，會走一個**三選一分支**來決定 reverb
  backend：

  | `augmentation_reverb_args["simulator"]` 的條件 | Backend |
  |---|---|
  | `simulator.used` 且 `simulator.pregenerated.used` | pre-generated RIR bank —— `self.augmentor.init_room_bank(pregenerated_config)` |
  | `simulator.used`，但沒有 `pregenerated` block（或 `pregenerated.used` 為 false） | 物理模擬的 room simulator —— `self.augmentor.init_room_simulator(simulator_args)` |
  | 沒有 `simulator` key，或 `simulator.used` 為假值 | 靜態 RIR 資料夾 —— `self.augmentor.load_rir_from_folder(augmentation_reverb_args["rir_folder"])` |

  **Pre-generated bank 的 `usage_role` contract**：pregenerated config 自己可能
  設了 `usage_role`（例如把某個 bank split 釘死給
  `"train"`/`"validation"`/`"test"` 用）。如果有設定、但跟這個 dataset
  instance 的 `pipeline_role` 對不上，`init_augmentor()` 會丟出
  `ValueError`。如果沒設定，呼叫 `init_room_bank()` 前會填成
  `self.pipeline_role`。共用 runner 會明確區分 train/validation dataset role，
  並從 typed dataset config 讀取各 stage 的 pipeline role。

---

#### VAD-labeler dispatch: `init_vad_labeler()`

讀取 `self.vad_label_args`（`cfg`），永遠會設定三個屬性：

- `self.gating_vad_labeler` – 若 `cfg` 為假值或 `cfg["used"]` 為假值則是
  `None`；否則一律是 `EnergyVADLabeler(frame_length=...,
  hop_length=...)`（便宜的 energy-based VAD）。這是設計給**在 DataLoader
  worker 內部**做粗略 overlap-gating 決策用的，所以不管 `cfg["backend"]`
  設什麼，它都刻意不用 Silero。
- `self.defer_vad_to_gpu` – 只有 `cfg["backend"] == "silero"`（不分大小寫）
  時才是 `True`；告訴 subclass 略過在 worker 裡逐筆做 VAD 標記，改成直接送出
  未經處理的 clean waveform，把 VAD target 的計算留給訓練模組自己在 GPU 上
  做一次 batch 化的 Silero pass。
- `self.vad_labeler` – subclass 拿來算真正 VAD loss target 用的 labeler：
  - 若 VAD 關閉，**或** `backend == "silero"`（如上、延後到 GPU 做）則是
    `None`
  - 否則就是 `create_vad_labeler(cfg)`（來自 `puresound.audio.vad`），它會
    重新讀一次 `cfg["backend"]`，據此建出 `EnergyVADLabeler` 或
    `SileroVADLabeler`

`frame_length`/`hop_length` 會先從 `cfg["args"]` 讀，讀不到再退回頂層的
`cfg["frame_length"]`/`cfg["hop_length"]`，最後才是 `400`/`160`。

---

### Source-level reverb helpers

這幾個函式只有在 `init_augmentor()` 選到 room-simulator 分支（pre-generated
bank 或物理模擬器）時才有意義——它們讓 subclass 可以一次對單一個 source
waveform 做 reverb，而不是只能混合已經處理好殘響的訊號。

#### `should_apply_source_level_reverb() -> bool`

只有在 `augmentation_reverb_args["used"]`、它的 `simulator` block 有啟用，
*而且* `simulator["source_level"]` 有設定時才可能是 `True`——接著才會抽
`torch.rand(1) < augmentation_reverb_args["prob"]`。只要上述任何一個
config 開關沒開，就一定回傳 `False`（不會消耗任何 RNG）。

#### `apply_source_level_target_reverb(wav, sr, room_scene, distance_range_override=None) -> ForegroundReverb(noisy, clean, metadata)`

把 room scene 的 RIR 以**foreground** source 的身份套用到 `wav` 上
（`source_role="foreground"`、`rir_mode="full"`），做出 `noisy_target`。接著
建構 `clean_target`：

- 若 `augmentation_reverb_args["target_rir_type"] == "anechoic"`：
  `clean_target = wav`（完全不加殘響）
- 否則：用**同一個實際抽到的 RIR**（`rir_id`）、以
  `rir_mode=target_rir_type` 重新套用一次（例如只取 direct-path 或
  early-reflections 的模式）

`rir_metadata` 是 simulator 實際抽到的擺位資訊 dict（例如
`source_receiver_distance`）；若這個 RIR 其實是從資料夾來的而不是
simulator，則是 `None`。

#### `apply_source_level_interferer_reverb(wav, sr, room_scene, distance_range_override=None, source_role="interferer") -> ReverbedSource(wav, metadata)`

概念跟上面一樣，但用在非 target 的 source：以指定的 `source_role` 標籤套用
room scene 的 `"full"` RIR（讓 room scene 可以獨立擺放多個 interferer）。
沒有 clean 配對，但這個 channel 的 `metadata` 會跟 waveform 一起回傳——呼叫端
需要每個 interferer 的擺放資訊來記 RIR lineage，而在此之前唯一的取得方式是去
讀 augmentor 的私有屬性 `_last_rir_meta`。

---

### VAD target helpers

#### `create_vad_target(clean_speech, sample_rate)`

若 `self.vad_labeler` 為 `None` 則回傳 `None`。否則，如果 `clean_speech` 是
全零的 tensor，會直接回傳 `create_empty_vad_target(clean_speech)`，而不是
呼叫 labeler——**這是為了迴避 energy-based VAD 裡一個真實存在的地雷**：
`EnergyVADLabeler` 是相對於*該筆 utterance 自己的最大值*去算每個 frame 的
dB 能量，所以一段全靜音的輸入會讓 `reference == eps`，導致每個 frame 都被
讀成「有活動」（`0 dB > -40 dB` 的門檻）。Target-absent 的訓練列（刻意設成
全零的 clean reference）如果沒有這個特例處理，就會得到一個假的全 1 VAD
target。

#### `create_empty_vad_target(wav) -> Tensor`

回傳一個長度為 `frame_count(wav.shape[-1], self.vad_labeler.frame_length,
self.vad_labeler.hop_length)`（來自 `puresound.audio.vad.frame_count`）的
全零 tensor；若 `self.vad_labeler` 為 `None` 則回傳 `None`。

---

### Utterance selection and shaping

#### `choose_an_utterance_by_speaker_name(target_speaker_name, ignoring_utt_list=None, select_channel=None, select_with_sr_as_key=None) -> (wav, sr, (speaker_name, uttid))`

為 `target_speaker_name` 隨機抽一筆 utterance，透過
`AudioIO.open(target_lvl=self.audio_gain_normalized_to,
resample_to=self.target_sr)` 開檔。

- `ignoring_utt_list` – 要從候選中排除的 utterance id（例如同一筆樣本裡
  已經用過的那個）
- `select_channel` – 若有設定，且開出來的 waveform 的 channel 數比這個
  index 多，就只保留那個 channel（強制轉單聲道）
- `select_with_sr_as_key` – 若有設定，候選池會限縮成
  `self.sr_meta[select_with_sr_as_key][target_speaker_name]`，而不是這個
  speaker 完整的 utterance 清單，並且會 assert 開出來的檔案取樣率吻合。兩個
  分支的候選池都是**metafile 順序**，這正是固定 seed 能跨 process 抽到同一筆的
  原因；這個分支以前會把池子過一次 `set`，因此依賴 `PYTHONHASHSEED`。只有
  `target_sample_rate: null` 的 recipe 會走到這裡（`runner.py` 由它決定
  `select_by_sr_first`），所以出貨的 recipe 都沒受影響——見
  `test_sr_keyed_utterance_pool_keeps_metafile_order`
- 若抽到的 utterance 是全靜音，最多遞迴重試 5 次，之後丟出
  `RuntimeError("Timeout, can't find a useful utterance.")`

#### `align_audio_list(wav_list, length, padding_type="zero") -> List[Tensor]`

把 `wav_list` 裡每個 waveform 裁切（隨機 offset）或補齊（前後補零長度隨機
分配）到剛好 `length` 個 samples。

- 裁切時如果抽到的 window 是靜音會重抽 offset，但最多只重試 10 次，之後就
  算是靜音也照樣接受——這個上限是刻意設計的，不是 bug：如果沒有這個
  上限，一段真的全靜音的 source（安靜的 interferer，或 target-absent 的
  列）會讓重試迴圈跑到天荒地老，卡住一個 DataLoader worker，進而讓 DDP
  死鎖（因為某個 rank 永遠到不了下一個 collective）。
- `padding_type="zero"` – 補零
- `padding_type="normal"` – 補完之後，對整段補好的 waveform 加上 40 dB-SNR
  的背景白噪音（透過 `add_bg_white_noise`）

#### `avoid_audio_clipping(wav_list) -> List[Tensor]`

只要清單裡有任何一個 waveform 的絕對值峰值超過 1，就用**同一個共用峰值**
去除清單裡的**每一個**waveform（不是各自除各自的峰值）——這樣才能保留例如
target 跟它的 interferer 之間的相對音量關係。如果沒有任何一個 clip 就原樣
回傳。

### Abstract / unimplemented methods

- `__len__` – 丟出 `NotImplementedError`；subclass 必須覆寫。動態合成沒有固定
  的 epoch 大小，而訓練路徑的迭代是由自帶長度的 `batch_sampler` 驅動，所以
  沒有人會去問 dataset 要長度
- `__getitem__` – 丟出 `NotImplementedError`
- `apply_audio_augmentation()` – 丟出 `NotImplementedError`

## Example

```python
from puresound.dataset.dynamic_base import DynamicBaseDataset

class MyDataset(DynamicBaseDataset):
    def __len__(self):
        return len(self.total_spks)

    def __getitem__(self, idx):
        spk = self.total_spks[idx]
        wav, sr, (spk, uttid) = self.choose_an_utterance_by_speaker_name(spk)
        wav = self.align_audio_list([wav], self.training_sample_length)[0]
        vad_target = self.create_vad_target(wav, sr)
        return {"speech": wav, "sr": sr, "vad_target": vad_target}
```
