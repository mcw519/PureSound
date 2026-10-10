# puresound.dataset.dynamic_base

English version: [dynamic_base.md](dynamic_base.md)

動態合成用的 base dataset：每一筆訓練資料都在讀取當下，由乾淨語音的 metafile
加上噪音、殘響與裝置鏈增強即時組出來，磁碟上不存任何混音。
`puresound.task.ns.NoiseSuppressionDataset` 直接繼承它，
`puresound.task.voice_isolation.VoiceIsolationDataset` 經由 noise-suppression
dataset 繼承，legacy 的語者 embedding 與目標語者萃取 dataset
（`puresound.task.sv`、`puresound.task.tse`）也繼承它。每個 subclass 實作自己的
`__getitem__`。

## Class: `DynamicBaseDataset`

繼承 `torch.utils.data.Dataset`。負責 metafile 的載入與過濾、augmentation block
的驗證、共用的 `AudioEffectAugmentor`、VAD labeler 的分派、sampler key 的解析與
curriculum 掛鉤；每筆資料的實際合成由 subclass 實作。

### Constructor

```python
DynamicBaseDataset(
    metafile_path: str,
    min_utt_length_in_seconds: float = 3.0,
    min_utts_in_each_speaker: int = 5,
    target_sr: Optional[int] = None,
    training_sample_length_in_seconds: float = 6.0,
    audio_gain_normalized_to: Optional[int] = None,
    dataset_role: str = "train",
    pipeline_role: str | None = None,
    curriculum: AugmentationArg = None,
    **augmentation: AugmentationArg,
)
```

`AugmentationArg` 是已驗證的 Pydantic model、一般 mapping 或 `None`。長度單位
都是**秒**，不是 samples。

- `metafile_path` – CSV metafile，由
  [`MetafileParser.read_from_metafile`](parser.zh-TW.md) 解析
- `min_utt_length_in_seconds` – 短於這個長度的 utterance 會在 `gen_meta()` 被丟掉
- `min_utts_in_each_speaker` – 篩完後 utterance 數不足的 speaker 會被丟掉
- `target_sr` – 有設定時，每個波形載入時都重取樣到這個取樣率，且
  `self.training_sample_length = int(target_sr * training_sample_length_in_seconds)`；
  為 `None` 時，每一列維持來源檔案的取樣率，`self.training_sample_length` 為 `None`
- `audio_gain_normalized_to` – 目標 dBFS，傳給這個 class 內每一次開檔的
  `AudioIO.open(target_lvl=...)`
- `dataset_role` – dataset stage：`"train"`、`"validation"` 或 `"test"`；其他值會
  丟出 `ValueError`
- `pipeline_role` – 預設等於 `dataset_role`；stage-sensitive 的 capability（例如
  預先生成的 RIR bank split）依它選擇資料來源分布。允許值相同。
- `curriculum` – `CurriculumConfig`（見
  [Recipe 設定](../../usage/configuration.zh-TW.md)），或 `None`（旋鈕都是常數）。
  runner 只把它交給 training dataset，這樣 validation loss 仍可跨 epoch 比較。
- `**augmentation` – 以 keyword 傳入的 augmentation block（見下方）

#### Augmentation block：`AUGMENTATION_BLOCKS`

dataset 接受哪些 keyword argument，列在 class attribute `AUGMENTATION_BLOCKS`
這個 `keyword -> config model` mapping 裡。base 註冊的是：

| keyword | model（`puresound.config.augmentation`） |
|---|---|
| `augmentation_speech_args` | `SpeechAugmentation` |
| `augmentation_noise_args` | `NoiseAugmentation` |
| `augmentation_reverb_args` | `ReverbAugmentation` |
| `augmentation_speed_args` | `ContinuousSpeedAugmentation` |
| `augmentation_ir_response_args` | `SimpleProbAugmentation` |
| `augmentation_src_args` | `SourceRateAugmentation` |
| `augmentation_hpf_args` | `HighPassAugmentation` |
| `augmentation_volume_args` | `VolumeAugmentation` |
| `augmentation_compressor_args` | `CompressorAugmentation` |
| `vad_label_args` | `VadLabelConfig` |

每個註冊過的 keyword 都會成為一個 attribute，呼叫端沒給就是 `None`，讀取端因此
可以直接寫 `if self.augmentation_noise_args:`。recipe 把寫成 `used: false` 的
block 也當成 `None` 轉交（`BaseRecipe.augmentation_kwargs`）。每個值都經過
`as_block(value, model)`：model instance 原樣保留，mapping（或其他型別的 model）
會被驗證成註冊的 model。不在 registry 裡的 keyword 會丟出 `TypeError`，並列出
可接受的名稱。

用 registry 而不是參數列表，是因為 recipe 那一半是從 schema 推導出來的
（`BaseRecipe.augmentation_kwargs`）；手寫的列表面對推導出來的列表終究會走岔。
Subclass 擴充這個 mapping，也可以替換某個 entry 來綁定同一個 block 的另一種
方言（語者 embedding dataset 用離散的變速值，分離類任務用連續範圍）。在這裡驗證
mapping，代表在測試或腳本裡手動建出來的 dataset 也會經過與 recipe 相同的 schema
檢查。

base class 只會用到 `augmentation_noise_args`、`augmentation_reverb_args` 與
`vad_label_args`（在 `init_augmentor()` 與 `init_vad_labeler()`）；其餘由 subclass
在合成一列時讀取。

### Initialization sequence

`__init__` 儲存參數、驗證 block 與 role，然後呼叫 `self.init_necessary()`，依序
執行：

1. `self.gen_meta(...)` —— 拆開存進 `self.meta`、`self.gender_meta`、
   `self.gender_spks`、`self.sr_meta`
2. `self.total_spks`（排序過的 speaker id）與 `self.spk2idx`
3. `self.init_augmentor()` —— 建立 `self.augmentor`
4. `self.init_vad_labeler()` —— 建立 `self.vad_labeler` 與
   `self.gating_vad_labeler`
5. `self.apply_curriculum_epoch(0)` —— 先把排程的起始值就位，讓還沒有人宣告
   epoch 之前抽出來的列（dump、稽核腳本、一般 PyTorch loop）看到的是排程的起點，
   而不是檔案裡的常數

---

#### `parse_item_key(key) -> ItemKey`

sampler 交給 dataset 的 tuple 會隨這次 run 的要求變長：
`(speaker, sample_rate[, seed[, seconds[, epoch]]])`（見
[task.sampler](../task/sampler.zh-TW.md)）。其他形狀會丟出 `TypeError`。這個
method 回傳 `ItemKey(speaker, sample_rate, seed, seconds, epoch)` named tuple
（缺的欄位為 `None`），並依序套用每個 task 共用的三件事：

1. 這一列的長度，`sample_length` 會讀它（秒數乘上 `target_sr`；沒有目標取樣率時改乘上 key
   自己的 `sample_rate`，因為此時還沒打開 utterance）；
2. 這個 epoch 的排程旋鈕（`apply_curriculum_epoch`），接著才是
3. per-item seed，重設 `random`、`numpy` 與 `torch` 的 seed，讓 seed 管轄的每次
   抽樣都已經看到該 epoch 要求的值。

每個 dynamic dataset 的 `__getitem__` 都從這裡開始。在同一處解析 key，key 的形狀
就不會變成 task-specific：seeded validation、混合列長與 curriculum 對每個 task
的行為都一致。

#### 衍生的音訊參數

- `audio_sr` – 有設 `target_sr` 就是它，否則是 `self.ori_audio_sr`，也就是
  `__getitem__` 開啟前景 utterance 時記下的取樣率。
- `sample_length` – `audio_sr` 下的裁切長度（samples）：key 帶了列長時用
  `parse_item_key` 設下的 per-item 覆寫值，否則用 `training_sample_length`，再否則
  用 `ori_audio_sr * training_sample_length_in_seconds`。

兩者都會退回來源檔案的取樣率，所以只在合成一筆資料的過程中有意義。每個
DataLoader worker 擁有自己的 dataset 副本、一次只合成一筆，因此 per-item 覆寫值
不會被共用。

---

#### `apply_curriculum_epoch(epoch)`

把 `epoch` 的排程旋鈕就位；沒有使用 curriculum 或該 epoch 已生效時是 no-op。它會
解析排程、把被點名的 augmentation block 換成驗證過的副本（`with_overrides`）、
有 block 改變時呼叫 `rebind_augmentation_blocks()`，並在 room bank 是多個 bank 的
聯集時把排程的 bank 權重交給它的 `set_weights`。

它絕不動 RNG：每一列是 per-item seed 的，在這裡抽一次會讓整列後續的抽樣依「當時
是第幾個 epoch」而位移。無法套用的目標（這個 dataset 沒有的 block、不是聯集 bank
卻排了 bank 權重）只記 warning 不 raise：recipe 在載入時就已拒絕未知目標
（`BaseRecipe.curriculum_targets_exist`），而在 epoch 中途殺掉 DataLoader worker
會讓 DDP 卡在下一次 all-reduce。

#### `rebind_augmentation_blocks()`

重新推導所有由 augmentation block 組出來的元件。block 是每列現讀的，但由它組成的
元件——capture chain、noise stage、gating helper——只組一次，並持有當時拿到的那份
block。會組這些元件的 subclass 把建構寫在這個方法裡、由 `__init__` 呼叫，構造與
重推導就不會走岔。只做組裝：不動 RNG、不讀磁碟。base 沒有需要重推導的東西。

---

#### `gen_meta(metafile_path, min_utts_in_spk=10, min_utt_length=3.0)`

`init_necessary()` 一律用 constructor 的值呼叫它，所以這裡的預設值只在直接呼叫時
才生效。

用 `MetafileParser.read_from_metafile(f_path=..., use_speaker_as_key=True)` 解析
metafile，接著：

- 丟掉短於 `min_utt_length` 秒（`length / sr`）的 utterance
- 丟掉剩下不到 `min_utts_in_spk` 個 utterance 的 speaker
- 由 speaker id 第一個 `_` 之前的前綴推出每個 speaker 的 `corpus_id`（metafile
  的 speaker 命名為 `{corpus}_{speaker}`）
- 依性別（`m`/`male` → `"m"`、`f`/`female` → `"f"`、其他 → `"other"`）與取樣率
  把 speaker 分組

回傳 4-tuple `(meta, gender_meta, gender_spks, sr_meta)`：

| 回傳值 | 形狀 |
|---|---|
| `meta` | `{spkid: {"gender", "channels", "corpus_id", "utts": {uttid: {"path", "length", "channels", "sr"}}}}` |
| `gender_meta` | `{"m"/"f"/"other": {corpus_id: [spkid, ...]}}` |
| `gender_spks` | `{"m"/"f"/"other": [spkid, ...]}` |
| `sr_meta` | `{sample_rate: {spkid: [uttid, ...]}}`；在該取樣率下 utterance 不足 `min_utts_in_spk` 的 speaker，以及因此沒有任何 speaker 的取樣率，都會被移除 |

它同時設定 `self.all_corpus_id`（出現過的 corpus id 集合）。每個 corpus 不論
speaker 多少都保留：最少 speaker 數的過濾會改變訓練分布。

---

#### `init_augmentor()`

不接受參數；依 `self.augmentation_noise_args` 與 `self.augmentation_reverb_args`
建立 `self.augmentor = AudioEffectAugmentor()`。

- **噪音**（block 有設定時）：有 `noise_sources`（具名、帶權重的資料夾）就用
  `load_bg_noise_sources` 載入；否則用 `load_bg_noise_from_folder` 載入
  `noise_folder`。
- **殘響**（block 有設定時）：恰好選一個 backend。

  | 條件 | backend |
  |---|---|
  | `simulator.used` 且 `simulator.pregenerated.used` | 預先生成的 RIR bank —— `init_room_bank(...)` |
  | `simulator.used`，沒有 `pregenerated` block 或 `pregenerated.used` 為 false | 物理房間模擬器 —— `init_room_simulator(...)` |
  | 沒有 `simulator` block，或 `simulator.used` 為 false | 靜態 RIR 資料夾 —— `load_rir_from_folder(rir_folder)` |

  backend 之後，若 `direct_smear` 與 `drr_contrast` 子 block 有啟用，會以
  `init_direct_smear`、`init_drr_contrast` 安裝。

交給這些元件的 block 都經過 `delegated_kwargs`，只轉交 recipe 實際寫出的 key，
其餘沿用各元件自己的預設值。

**預先生成 bank 的 `usage_role`**：bank block 可以自行固定 `usage_role`
（`"train"`/`"validation"`/`"test"`）。有設定且與這個 dataset 的 `pipeline_role`
不同時，`init_augmentor()` 會丟出 `ValueError`；沒設定時，會在
`init_room_bank()` 之前填入 `pipeline_role`。這就是 validation dataset 不會抽到
training RIR 的原因。

---

#### VAD labeler 分派：`init_vad_labeler()`

讀取 `self.vad_label_args`（`VadLabelConfig`），並一定會設定三個 attribute：

- `self.gating_vad_labeler` – block 不存在或沒有 `used` 時為 `None`；否則是
  `EnergyVADLabeler`。它負責在 DataLoader worker 內做的粗略 overlap-gating 判斷，
  所以不論 `backend` 寫什麼，永遠是便宜的 energy VAD，絕不用 Silero。
- `self.defer_vad_to_gpu` – 只有 `backend == "silero"` 時為 `True`。此時 subclass
  跳過每列的 VAD 標註、改輸出乾淨參考波形，由 training module 在 GPU 上以 Silero
  一次標註整個 batch，讓 Silero 離開每筆資料的 worker 路徑。
- `self.vad_labeler` – subclass 用來產生 VAD loss 目標的 labeler：VAD 關閉或交給
  GPU 時為 `None`，否則是 `puresound.audio.vad` 的 `create_vad_labeler(...)`
  （energy 或 Silero）。

`frame_length`、`hop_length`（以及 gating labeler 的 `eps_mode`）先從 block 的
`args` 讀，再退回 block 自己的 `frame_length` / `hop_length` 欄位（預設
`400` / `160`）；`args` 是 labeler constructor 的命名空間，優先。

---

### Source-level 殘響 helper

這些只在 `init_augmentor()` 選了房間模擬分支（預先生成 bank 或物理模擬器）時
適用。它們讓 subclass 能替每個聲源各自套上房間通道，而不只是混合事先加好殘響的
訊號。

#### `should_apply_source_level_reverb() -> bool`

reverb block 有啟用、其 `simulator` 有啟用且 `simulator.source_level` 有設定時，
以 `augmentation_reverb_args.prob` 的機率回傳 `True`。任一道關卡關閉時直接回傳
`False`，不動 RNG。

#### `apply_source_level_target_reverb(wav, sr, room_scene, distance_range_override=None) -> ForegroundReverb(noisy, clean, metadata)`

把 room scene 的 `"full"` RIR 以**前景**聲源套到 `wav` 上，得到 noisy target，
再產生 clean target：

- `target_rir_type == "anechoic"`：乾的 `wav`
- 其他：以 `rir_mode=target_rir_type`（`"full"`、`"early"` 或 `"direct"`）重新
  渲染**同一個實現出來的 RIR**（`rir_id`）

`metadata` 是實際的擺位（例如 `source_receiver_distance`）；RIR 來自資料夾時為
`None`。結果可以直接 `noisy, clean, meta = ...` 拆開。

#### `apply_source_level_interferer_reverb(wav, sr, room_scene, distance_range_override=None, source_role="interferer") -> ReverbedSource(wav, metadata)`

同樣的事套在非目標聲源上，並標上 `source_role`，讓 room scene 能獨立擺放多個
干擾者。通道的 `metadata` 隨波形一起回傳，因為呼叫端要把每個干擾者的擺位記進它的
RIR lineage；在兩次呼叫之間去讀 augmentor 的內部狀態，只要中間有別的東西做了卷積
就會歸錯。

---

### VAD target helper

#### `create_vad_target(clean_speech, sample_rate)`

`self.vad_labeler` 為 `None` 時回傳 `None`。全零的參考訊號（目標缺席的列）直接
回傳 `create_empty_vad_target(clean_speech)`，不呼叫 labeler：`EnergyVADLabeler`
以該 utterance 自己的最大值衡量每個 frame，靜音輸入否則會每個 frame 都讀成有聲。
這樣每種 backend 對缺席的目標都一致回報「沒有活動」。

#### `create_empty_vad_target(wav) -> Tensor`

長度為 `frame_count(wav.shape[-1], frame_length, hop_length)` 個 frame 的全零
`float32` tensor（`frame_count` 來自 `puresound.audio.vad`）；`self.vad_labeler`
為 `None` 時回傳 `None`。

---

### Utterance 選取與整形

#### `choose_an_utterance_by_speaker_name(target_speaker_name, ignoring_utt_list=None, select_channel=None, select_with_sr_as_key=None) -> (wav, sr, (speaker_name, uttid))`

隨機抽 `target_speaker_name` 的一個 utterance，並以
`AudioIO.open(target_lvl=self.audio_gain_normalized_to, resample_to=self.target_sr)`
開啟。

- `ignoring_utt_list` – 要排除的 utterance id（例如同一列已經用過的）
- `select_channel` – 波形的聲道數多於這個 index 時，只保留該聲道
- `select_with_sr_as_key` – 把候選池限制在
  `self.sr_meta[select_with_sr_as_key][target_speaker_name]`，並 assert 開出來的
  檔案是這個取樣率。只有 `target_sample_rate: null` 的 recipe 會走到這個分支
  （runner 由它推導 sampler 的 `select_by_sr_first`）。
- 兩個分支都從**依 metafile 順序**排列的候選池抽，絕不經過 `set`，所以固定 seed
  在每個 process 都抽到同一個 utterance，不受 `PYTHONHASHSEED` 影響。
- 抽到全靜音的檔會重試最多 5 次，之後丟出
  `RuntimeError("Timeout, can't find a useful utterance.")`。

#### `align_audio_list(wav_list, length, padding_type="zero") -> List[Tensor]`

把每個波形裁切（隨機位移）或補齊（補零長度隨機分配在前後）到剛好 `length` 個
samples。

- 裁切到靜音視窗時會重抽位移，最多 10 次，之後接受靜音視窗。這個上限是刻意的：
  真正靜音的聲源（很小聲的干擾者、目標缺席的列）否則會無限重試、卡住 DataLoader
  worker，進而讓 DDP 死結。
- `padding_type="zero"` – 補零
- `padding_type="normal"` – 補齊後，在整段波形上加 40 dB SNR 的白噪音
  （`add_bg_white_noise`）

#### `avoid_audio_clipping(wav_list) -> List[Tensor]`

任一波形的峰值超過 1 時，把**每一個**波形都除以同一個共同峰值，保留目標與干擾者
之間的相對音量。沒有 clipping 時原樣回傳。

### Subclass 要實作的 method

- `__getitem__(key)` – 每筆資料的合成；base 直接丟出 `NotImplementedError`
- `__len__` – base 丟出 `NotImplementedError`。動態合成沒有固定的 epoch 大小：
  迭代由自帶長度的 `batch_sampler` 驅動，而 Lightning 會把 `__len__` 丟出
  `NotImplementedError` 的 dataset 視為 unsized。
- `apply_audio_augmentation()` – 丟出 `NotImplementedError`
- `rebind_augmentation_blocks()` – 選用，見上方

## Example

```python
from puresound.dataset.dynamic_base import DynamicBaseDataset

class MyDataset(DynamicBaseDataset):
    def __getitem__(self, key):
        item = self.parse_item_key(key)
        wav, self.ori_audio_sr, _ = self.choose_an_utterance_by_speaker_name(
            item.speaker, select_channel=0, select_with_sr_as_key=item.sample_rate
        )
        wav = self.align_audio_list([wav], self.sample_length)[0]
        vad_target = self.create_vad_target(wav, self.audio_sr)
        return {"speech": wav, "vad_target": vad_target}

dataset = MyDataset(
    metafile_path="data/train_speech.csv",
    target_sr=16000,
    vad_label_args={"used": True, "backend": "energy"},
)
```
