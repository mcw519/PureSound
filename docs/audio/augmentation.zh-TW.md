# puresound.audio.augmentation

English version: [`augmentation.md`](augmentation.md)

給語音資料集用的可組合式 audio augmentation：噪音注入、殘響(資料夾 RIR、
即時物理模擬器、或預先產生的 bank)、sox 風格的音量/速度/音高擾動、通道
染色(biquad、codec 來回轉換、封包遺失、媒體裝置 EQ)、以及取樣率轉換
artifact。`puresound/dataset/*.py` 和 `puresound/task/*.py` 都是透過每個
dataset split 各一個 `AudioEffectAugmentor` instance 來驅動這一切。

本模組貫穿全文的慣例:
- 波形都是 `[..., L]`(時間軸在最後一維);大多數呼叫點傳入的是單聲道
  `[1, L]`。
- 每個 augmentation method 都回傳 **`(augmented_wav, info)`** —— `info`
  是實際被隨機取樣出來的值(id、係數、增益...),讓呼叫端可以記錄下來,或
  是把同一個擾動重新套用到成對的另一個訊號上(例如把同一個 RIR 分別
  卷積到混音訊號跟它的乾淨 target 上 —— 見下面 `apply_rir` 的 `rir_id`)。
  這不是裝飾性設計,有好幾個 recipe 實際依賴這個機制。

## Class: `AudioEffectAugmentor`

### Constructor

```python
AudioEffectAugmentor()
```

不吃任何參數。在呼叫下面任何一個載入方法之前,內部狀態都是空的:

| attribute | 初始值 | 由誰填入 |
|---|---|---|
| `bg_noise` | `{}` | `load_bg_noise_from_folder` |
| `rir` | `{}` | `load_rir_from_folder` |
| `room_simulator` | `None` | `init_room_simulator` |
| `room_bank` / `room_bank_kind` | `None` / `None` | `init_room_bank` |
| `simulated_rir` | `OrderedDict()` | `apply_rir`(LRU cache,見下文) |
| `_last_rir_meta` | `None` | 每一次 `apply_rir` 呼叫 |

`_last_rir_meta` 讓已經把 `apply_rir` 回傳 tuple 解構掉的呼叫端,依然能夠
取回最後一次的 RIR metadata(`getattr(augmentor, "_last_rir_meta", None)`)
—— `puresound/task/ns.py` 就是這樣在不更動 `apply_rir` 本身回傳簽章的情況
下,記錄每個 interferer 的 RIR 資訊。

### 載入 noise / RIR pool

#### `load_bg_noise_from_folder(folder: str, suffix: str = ".wav")`
#### `load_rir_from_folder(folder: str, suffix: str = ".wav")`

兩者都會(透過 `puresound.utils.recursive_read_folder`)遞迴掃描 `folder`
底下副檔名為 `suffix` 的檔案,並且**只登記路徑** —— 這時候還沒有把音訊
載入記憶體。`bg_noise`/`rir` 最終長成 `{uttid: {"wav_path": path}}`,其中
`uttid` 是去掉副檔名後的檔名(如果去掉副檔名後還剩其他的點,會用 `_`
重新接起來,所以 `Room042.00093.wav` 的 key 會變成 `Room042_00093`)。

### 房間聲學:3 種互斥的來源,依優先順序檢查

`apply_rir` 依下列優先順序取得 impulse response:(1)預先產生的 bank、
(2)即時物理模擬器、(3)以 `rir_id` 為 key 的本 instance 模擬結果快取、
(4)上面載入的靜態資料夾 pool。一份 recipe 最多只會接上(1)/(2)其中
一種;(3)/(4) 則永遠是備援路徑。

#### `init_room_simulator(config: dict)`

```python
aug.init_room_simulator({
    "room_dim_range": [[3.0, 8.0], [3.0, 8.0], [2.4, 3.5]],
    "rt60_range": [0.15, 0.8],
    "source_receiver_distance_range": [0.3, 4.0],
    "foreground_distance_range": [0.3, 1.2],
    "interferer_distance_range": [1.2, 4.0],
    "receiver_margin": 0.4,
    "source_margin": 0.4,
    "nsample": 8192,
    "hp_filter": True,
})
```

在把下列這些「對 training recipe 有意義、但 dataclass 本身不接受」的
key 剔除之後,建出 `RoomImpulseResponseSimulator(**config)`(見
[room_simulator.md](room_simulator.md)):`used`、`source_level`、
`pregenerated`。這些 key 是被 dataset 層
(`puresound/dataset/dynamic_base.py`)拿去判斷「要不要呼叫這個方法」,
以及「殘響要不要用 per-source(『source-level』—— foreground 跟每個
interferer 各自用同一個房間抽出的不同結果做卷積)的方式套用」,而不是
只在混好的混音訊號上套用一次。

#### `init_room_bank(config: dict)`

接上預先產生的 RIR bank,取代即時模擬 —— bank 類別本身以及完整的
training-config YAML 契約請見 [rir_bank.md](rir_bank.md)。這個方法自己
的工作是決定「該建哪一種 bank class」:

- 如果 `config` 裡有 `recipe_id`,`bank_type` 預設為 `"release"`,否則
  預設為 `"room"`(舊版的目錄式 WAV bank)。
- `bank_type: release` 要求要有 `usage_role`(`train`/`validation`/
  `test` 三選一)、非空的 `recipe_id`,而且 `split` 必須跟 `usage_role`
  字串相等 —— 這樣可以擋掉 train 時期的 reader 不小心讀到 test split
  房間的情況。
- `bank_type: room` 會直接拒絕 release 專屬的 key(`recipe_id`、
  `release_manifest_name`、`require_production`、
  `production_decision_name`、`usage_role`),讓舊版 config 不會意外
  「半套」用到 release 語意。

設定 `self.room_bank` 跟 `self.room_bank_kind`(`"release"` 或 `"room"`)。

#### `sample_room_scene() -> Optional[dict]`

如果接了 bank,回傳 `room_bank.sample_scene()`;如果接了 simulator,回傳
`room_simulator.sample_scene()`;都沒有就回傳 `None`。一個「scene」固定了
房間幾何跟收音端(麥克風)位置,**但還沒有 source 位置** —— 原因見
[room_simulator.md](room_simulator.md):`apply_rir` 的 `source_role`
參數會在卷積當下決定 source 的距離範圍,所以同一個 scene 可以重複拿來
在同一個房間裡,把 foreground 講者放近、interferer 放遠。

### 基於 Sox 的單一數值擾動

跟 `add_bg_noise`/`apply_rir` 不同,底下這三個方法各自只吃一個「已經
決定好」的數值,不是一個範圍 —— recipe 層每個 item 從設定的範圍抽樣
一次,再把抽出的值傳進來。

#### `sox_volume_perturbed(wav: Tensor, vol_ratio: float, sr: int) -> (Tensor, vol_ratio)`

Sox 的 `vol` 效果:**線性**振幅倍率(不是 dB)。原始 docstring 引用的典型
範圍是 `[0.125, 2]`。如果安裝的 torchaudio 沒有 `sox_effects` 後端,會
退回單純的 `wav * vol_ratio`。

#### `sox_speed_perturbed(wav: Tensor, speed: float, sr: int) -> (Tensor, speed)`

Sox 的 `speed`+`rate` 效果鏈:會同時改變時長**和**音高(不像 tempo 效果
只改時長)。典型範圍 `[0.8, 1.2]`。備援(沒有 sox 時):用
`torchaudio.functional.resample` 在改變過的取樣率上重新取樣 —— 一樣的
時長/音高 trade-off,但濾波器品質較差。

#### `sox_pitch_perturbed(wav: Tensor, shift_ratio: int, sr: int) -> (Tensor, shift_ratio)`

Sox 的 `pitch` 效果:以 **cents** 為單位位移(100 cents = 一個半音)。
典型範圍 `[-100, 100]`。**沒有備援** —— 如果 torchaudio 沒有 sox 後端,
會直接原封不動回傳 `wav`(跟上面兩個方法不同,那兩個在 sox 不可用時
還會用別的方式近似效果)。

### 噪音注入

#### `add_bg_noise(wav, snr_list: List[float], sr: int, dynamic_type: bool = False, noise_id: Optional[List[str]] = None, noise_transform=None) -> (List[Tensor], (added_noise, noise_id, snr_list))`

在 [`noise.add_bg_noise`](noise.md) 外面包了一層 pool 管理:從
`self.bg_noise` 選一個(或接受呼叫端指定的)noise id,載入並重新取樣到
`sr`,在 SNR 混音**之前**視需要先丟給 `noise_transform` 處理(例如把
noise 跟房間通道做卷積,讓它跟語音共享同一個聲學空間),接著在一次呼叫
裡對 `snr_list` 裡**每一個** SNR 都做混音。

- `dynamic_type=True` 會抽 **2** 個 noise id 而不是 1 個;
  `noise.add_bg_noise` 會把它們頭尾接起來,變成同一個 noise bed 再混音。
- 如果明確指定 `noise_id`,必須是一個 `list` —— 當呼叫端需要指定、可
  重現的噪音時會用到(例如讓同一個混音的兩個版本共用同一個 noise bed)。
- 回傳一個 noisy 波形的 **list**,每個 `snr_list` 項目對應一筆 —— 不是
  一個跟 `wav` shape 相同的單一 tensor。

#### `add_bg_white_noise(wav, snr_list: List[float]) -> (List[Tensor], (noise, snr_list))`

[`noise.add_bg_white_noise`](noise.md) 的薄包裝;同樣是多 SNR / list 輸出
的合約,沒有牽涉到任何 pool。

### 殘響

#### `apply_rir(wav, rir_mode: str = "image", sr: int = 16000, rir_id: Optional[str] = None, room_scene: Optional[dict] = None, source_role: str = "source", distance_range_override: Optional[List[float]] = None) -> (Tensor, (rir_id, {"mode": str, "metadata": Optional[dict]}))`

> **`rir_mode="image"` 本身不是一個合法的 mode。** 這個 repo 裡每一個
> 真正的呼叫點都會明確傳入 `rir_mode` —— `"full"`、`"direct"`、或
> `"early"`(各自裁切什麼請見 [impulse_response.md](impulse_response.md))。
> 如果不帶 `rir_mode` 呼叫 `apply_rir(wav)`,會在 `wav_apply_rir` 內部的
> assertion 丟出 `AssertionError`。recipe YAML 裡(`target_rir_type`)還
> 會出現第四種值 `"anechoic"`,但那是由**呼叫端**
> (`puresound/dataset/dynamic_base.py`)處理成「跳過殘響、直接回傳乾聲」
> —— 這個值不會被傳到 `apply_rir` 本身。

RIR 來源,依優先順序(第一個符合的就用):

1. **Bank**(`self.room_bank` 已設定、`rir_id is None`):如果
   `room_scene` 是帶 bank 標記的 scene(`room_scene["_bank"]` 為真)就
   重複使用,否則重新抽一個;針對 `source_role` 選一個通道,可用
   `distance_range_override` 縮小範圍 —— 不過預先產生的 bank 只能挑
   *池內最接近*那個範圍的通道,沒辦法精確取樣(見
   [room_simulator.md](room_simulator.md));把結果用新的 `"bank-{n}"`
   id 快取起來。
2. **Simulator**(`self.room_simulator` 已設定、`rir_id is None`):呼叫
   `room_simulator.generate(sample_rate=sr, scene=room_scene,
   source_role=source_role, distance_range_override=...)`(`source_role`
   如何決定距離範圍、以及這裡的 `distance_range_override` 為什麼能被
   *精確*遵守 —— 跟 bank 路徑不同 —— 請見
   [room_simulator.md](room_simulator.md));把結果用新的
   `"simulated-{n}"` id 快取起來。
3. **命中模擬 RIR 快取**(`rir_id` 指到 `self.simulated_rir` 裡已經有的
   項目):重複使用那個確切的 impulse response(如果 `sr` 跟快取當時不同
   會重新取樣)。這就是 dataset 如何建出共用同一個房間的 noisy/clean
   pair 的方式:先用 `rir_id=None` 呼叫一次拿到一個 id,再用同一個
   `rir_id` 搭配*不同*的 `rir_mode`(例如混音用 `"full"`,clean target
   用 recipe 的 `target_rir_type` —— 通常是 `"early"` 或 `"direct"`)
   再呼叫一次,把*同一個* RIR 用兩種方式卷積。確切的作法可參考
   `puresound/dataset/dynamic_base.py` 裡的
   `apply_source_level_target_reverb`。這個快取是一個上限
   `simulated_rir_cache_size`(32)筆的 `OrderedDict` LRU。
4. **靜態資料夾 pool**(備援):如果 `rir_id is None`,從 `self.rir`
   (由 `load_rir_from_folder` 載入)隨機挑一個 key,否則直接查找。

回傳 `(reverb_wav, (rir_id, {"mode": rir_mode, "metadata": rir_metadata}))`。
資料夾 pool 來的 RIR,`rir_metadata` 是 `None`;其餘情況則是
simulator/bank 的 metadata dict(房間尺寸、receiver、source、rt60、
`source_role`、`source_receiver_distance`、`drr_db` —— 見
[room_simulator.md](room_simulator.md))。

### 通道染色 / 失真

#### `apply_2nd_iir_response(wav, a_coeffs: Optional[Tensor] = None, b_coeffs: Optional[Tensor] = None) -> (Tensor, (a_coeffs, b_coeffs))`

隨機二階 IIR 染色(模擬麥克風/通道),出自 *A Hybrid DSP/Deep Learning
Approach to Real-Time Full-Band Speech Enhancement*。當 `a_coeffs`/
`b_coeffs` 為 `None` 時,會從 `[-3/8, 3/8]` 均勻抽出(各 3 個自由係數;
隱含 `a[0]=b[0]=1`)。把回傳的係數傳回去,就能把*同一個*隨機濾波器套用
到第二個訊號上。

#### `apply_gain_distortion(wav, sr: int) -> (Tensor, (start_sample, duration_samples, gain))`

委派給 `volume.rand_gain_distortion`(見 [volume.md](volume.md))——
對 `wav` 裡隨機的一個*片段*(不是整段訊號)套用隨機增益,再裁切到
`[-1, 1]`。這裡沒有 min/max-gain 參數;片段的位置、長度、跟增益值全部都
是在 `rand_gain_distortion` 內部隨機決定的。

#### `apply_clipping_distortion(wav, min_quantile: float, max_quantile: float) -> (Tensor, (min_quantile, max_quantile))`

委派給 `volume.wav_clipping`(見 [volume.md](volume.md))—— 注意該模組
的預設邊界是不對稱的;在這裡兩個邊界都是必填參數。

#### `apply_src_effect(wav, sr: int, src_sr: int, src_backend: str) -> (Tensor, info_list)`

降取樣/升取樣的取樣率轉換來回(`sr → src_sr → sr`),模擬頻寬受限的傳輸。
`info_list` 是 [`dsp.wav_resampling`](dsp.md) 除了波形以外回傳的其他東西
—— 當 `src_backend="torchaudio"` 時,這包含了隨機決定的抗鋸齒濾波器
參數,而這組參數會刻意被重複用在升取樣那一段(同樣的
`lp_width`/`rolloff`/`window`),讓這一來一回的重新取樣行為像是「一個」
一致的低品質 resampler,而不是兩個各自獨立隨機的。

#### `apply_hpf(wav, sr: int, cutoff_freq: int, q_factor: float) -> (Tensor, (cutoff_freq, q_factor))`

`torchaudio.functional.highpass_biquad`,單一 cutoff —— 不是範圍,跟上面
sox 系列擾動一樣是「recipe 負責決定範圍」的慣例。

#### `apply_media_coloring(wav, sr: int, hp_cutoff: float, lp_cutoff: float, compress_power: Optional[float] = None) -> (Tensor, (hp_cutoff, lp_cutoff, compress_power))`

模擬經由電視/喇叭類裝置播放:先限制頻帶在 `[hp_cutoff, lp_cutoff]`
(兩個串接的 biquad),接著視需要做 `compress_power in (0, 1]` 的
peak-normalized waveshaping(`|x|^p`;`1.0`/`None` = 不壓縮,用來模擬
廣播鏈路常有的輕度動態範圍壓縮)。染色完後會把 RMS 還原回染色前的
水準,這樣下游的 SIR/level scaling 才不會被染色這個步驟本身干擾。
`puresound/task/ns.py` 用這個方法為標記成媒體來源的 interferer 上色 ——
見 [room_simulator.md](room_simulator.md) 裡的 `source_role="media"`。

#### `apply_codec(wav, sr: int, codec_name: str, bit_rate: Optional[int] = None) -> (Tensor, (codec_name, bit_rate))`

透過 `torchaudio.io.AudioEffector` 做編碼/解碼來回,模擬 VoIP/電話網路的
codec。`codec_name` 必須是 `supported_codecs()` 其中之一(目前是
`"libopus"`、`"g722"`,各自對應到編碼器需要的容器格式 —— 分別是
`ogg`/`matroska`,這個選擇是為了來回轉換的可靠度,避開 Opus/AAC 包在
MKV 裡的一些怪癖)。輸出會被裁切或補零回輸入原本的確切取樣數(codec
可能因為內部的重新取樣/分幀而改變長度)。對沒有位元率旋鈕的 codec
(`g722`)來說,`bit_rate` 會被忽略。

#### `supported_codecs() -> List[str]`(staticmethod)

回傳 `apply_codec` 接受的 codec 名稱。

#### `apply_packet_loss(wav, sr: int, packet_ms: int = 20, loss_rate: float = 0.05) -> (Tensor, (packet_ms, loss_rate, n_dropped))`

把隨機、大小為 `packet_ms` 的區塊歸零,模擬 VoIP 斷續(20 ms 是 WebRTC
的預設值;60 ms 則是低頻寬 Opus 常見的值)。每個封包各自獨立以機率
`loss_rate` 被丟棄。回傳資訊裡的 `n_dropped` 是*實際*被歸零的封包數
(不管是隨機抽樣結果剛好一個都沒丟、還是 `loss_rate <= 0`,都會是 0 ——
這兩種情況都會回傳一份複製過、但沒被更動的波形)。

## 範例

```python
from puresound.audio.augmentation import AudioEffectAugmentor

aug = AudioEffectAugmentor()
aug.load_bg_noise_from_folder("/data/musan/noise")
aug.load_rir_from_folder("/data/rirs")

noisy_list, (added_noise, noise_id, snr_list) = aug.add_bg_noise(
    wav=clean_wav, snr_list=[0.0], sr=16000
)
reverberant, (rir_id, rir_info) = aug.apply_rir(
    wav=noisy_list[0], rir_mode="full", sr=16000
)
```
