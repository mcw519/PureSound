# puresound.audio.augmentation

English version: [augmentation.md](augmentation.md)

`AudioEffectAugmentor` 是合成 pipeline 取用的「每個 dataset 一份」工具箱：噪音池、
RIR 來源（資料夾、即時模擬器或預先產生的 bank）、兩個 RIR 通道增強，以及位準、
速度、音高、濾波、codec 與封包遺失等運算子。dataset 層在 `init_augmentor`
（`puresound/dataset/dynamic_base.py`）為每個 split 建一個實例；task dataset、
`NoiseStage` 與 `DeviceChain` 呼叫它的方法。每個運算子在一筆訓練資料中的位置見
[資料增強](../augmentation/index.zh-TW.md)。

慣例：

- 波形是 `[C, L]`，幾乎都是單聲道 `[1, L]`。
- 每個運算子都回傳 `(wav, info)`，`info` 是被隨機化的內容（id、係數、增益）。把
  `info` 傳回去就能對配對訊號套用完全相同的擾動，混音與目標就是這樣拿到同一個
  濾波器或同一個 RIR。

## 狀態

| 屬性 | 由誰填入 |
| --- | --- |
| `bg_noise`、`bg_noise_groups` | `load_bg_noise_from_folder`、`load_bg_noise_sources` |
| `rir` | `load_rir_from_folder` |
| `room_simulator` | `init_room_simulator` |
| `room_bank`、`room_bank_kind`（`"room"`、`"release"`、`"union"`） | `init_room_bank` |
| `drr_contrast`、`direct_smear` | `init_drr_contrast`、`init_direct_smear` |
| `simulated_rir` | `apply_rir`：已提供脈衝的 LRU 快取，32 筆 |
| `_last_rir_meta` | `apply_rir`；僅供互動檢查，沒有程式讀它 |

## 噪音池

- `load_bg_noise_from_folder(folder, suffix=".wav")`：遞迴登記 `folder` 下每個檔案
  的路徑，此時還不載入。key 是去掉副檔名的檔名，中間的點以 `_` 連接。YAML：
  `augmentation_noise.noise_folder`。
- `load_bg_noise_sources([(name, folder, weight), ...])`：由多個語料組成一個池。
  key 加上 `name/` 前綴，同名檔案不會互相覆蓋。抽樣時先依 `weight` 比例選來源，
  再在來源內均勻選檔，小語料也保有它的份額。YAML：
  `augmentation_noise.noise_sources: [{name, folder, weight}]`。

## RIR 來源

`apply_rir` 依序取第一個可用的脈衝來源：bank、模擬器、快取（傳入已知 `rir_id`
時）、資料夾池。一個 recipe 最多啟用 bank 與模擬器其中之一。

- `load_rir_from_folder(folder)`：靜態 WAV 池，`augmentation_reverb.rir_folder`。
- `init_room_simulator(config)`：由 `augmentation_reverb.simulator` 建立
  [`RoomImpulseResponseSimulator`](room_simulator.zh-TW.md)，先移除 dataset 層
  管理的 key（`used`、`source_level`、`pregenerated`）。
- `init_room_bank(config)`：由 `augmentation_reverb.simulator.pregenerated` 建立
  預先產生的 bank，內容是單一 bank（`folder`）或 `banks:` 串列。類別見
  [RIR bank loaders](rir_bank.zh-TW.md)。
  - 有 `recipe_id` 時 `bank_type` 預設為 `"release"`，否則為 `"room"`（目錄 bank）。
  - release bank 需要 `usage_role ∈ {train, validation, test}`、非空的
    `recipe_id`，且 `split == usage_role`，訓練端因此拿不到 test 房間。dataset 層
    依 pipeline role 設定 `usage_role`，設定值不一致就拒絕。
  - room bank 拒絕 release 專用的 key（`recipe_id`、`release_manifest_name`、
    `require_production`、`production_decision_name`、`audit`、`audit_cache`）。
  - `banks:` 串列以相同方式建立每個成員，透過 `UnionRoomBank` 依成員的 `weight`
    （抽樣機率，不是筆數）提供。串列旁只能另設 `usage_role`。
- `sample_room_scene()`：有 bank 就回傳 bank 場景，否則回傳模擬器場景，都沒有則
  `None`。場景固定房間與接收點、不含聲源，所以一個場景可服務同一筆資料的前景與
  所有干擾者。

## RIR 通道增強

兩者都作用在剛取出的脈衝上，**在進快取之前**。目標會以同一個 `rir_id` 取回同一個
脈衝、用較短的窗做 convolution，所以 `"early"` 目標仍是 `"full"` 混音的近場成分。
兩者在關閉時都不消耗亂數。

**`init_drr_contrast(config)`**（`augmentation_reverb.drr_contrast`）把
`peak + direct_window_ms`（預設 2.5 ms，即 pipeline 的 DRR 窗）之後的尾段乘上
`10^(−Δ/20)`，使該通道的 DRR 恰好移動 `+Δ` dB；metadata 會加上
`drr_contrast_shift_db` 與重算的 `drr_db`。

| `mode` | Δ | 亂數 |
| --- | --- | --- |
| `"random"`（預設） | 以機率 `prob`（0.5）：`foreground` 為 `+U(near_boost_db)`，`interferer`/`media`/`echo` 為 `−U(far_cut_db)`（預設皆 `[0, 4]` dB）；其他角色不動 | Python `random` |
| `"deterministic"` | 由通道實際距離 `d` 計算 `extra_db_per_decade · log10(d / pivot_m)`（`pivot_m` 預設 1 m）；與角色無關 | 無 |

random 模式每次抽樣都拉大近／遠場的 DRR 差距，但同時讓 DRR 與距離脫鉤；
deterministic 模式讓池子的距離→DRR 斜率變陡，且保持單一、一致的對應。

**`init_direct_smear(config)`**（`augmentation_reverb.direct_smear`）：以機率
`prob` 套用 [`smear_direct_arrival`](impulse_response.zh-TW.md)，
`smear_ms ~ U(smear_ms_range)`，並在 metadata 記錄 `direct_smear_ms`。

## `apply_rir(wav, rir_mode="full", sr=16000, rir_id=None, room_scene=None, source_role="source", distance_range_override=None) -> RirApplied`

`rir_mode` 必須是 `"full"`（預設）、`"early"` 或 `"direct"`；其他值會讓
[`wav_apply_rir`](impulse_response.zh-TW.md) 的 assertion 失敗。
（recipe 中的 `target_rir_type: anechoic` 由 dataset 層處理，它會直接略過呼叫。）

1. **Bank**（有 `room_bank`、`rir_id is None`）：`room_scene` 是 bank 場景
   （`room_scene["_bank"]`）就沿用，否則抽一個；依 `source_role` 選通道，有給
   `distance_range_override` 就挑最接近者；重新取樣到 `sr`；套用通道增強；以
   `bank-<n>` 快取。
2. **模擬器**（有 `room_simulator`、`rir_id is None`）：`generate(sr, scene,
   source_role, distance_range_override)`，override 精確遵守；套用通道增強；以
   `simulated-<n>` 快取。
3. **快取**（`rir_id` 在 `simulated_rir` 中）：取出已存的脈衝，`sr` 不同就重新取樣。
4. **資料夾**：`rir` 中隨機一個 key，或給定的 `rir_id`。

配對目標的做法是兩次呼叫：第一次 `rir_id=None`、`rir_mode="full"` 會回傳一個 id；
第二次帶這個 id 與 recipe 的 `target_rir_type`。dataset 層的
`apply_source_level_target_reverb` 就是這樣做。

回傳 `RirApplied(wav, RirDetail(rir_id, {"mode": rir_mode, "metadata": ...}))`，
兩層 `NamedTuple`，所以 `wav, (rir_id, info) = aug.apply_rir(...)` 可以直接拆解。
`result.detail.metadata` 是模擬器或 bank 的擺放資訊（房間、接收點、聲源、`rt60`、
`source_role`、`source_receiver_distance`、`drr_db`，以及增強欄位），資料夾 RIR
則為 `None`。metadata 請從回傳值取；`_last_rir_meta` 每次呼叫都會被覆寫，中間只要
有別的 convolution 就會對錯對象。

## 位準、速度與音高

每個方法接受一個已決定的值；由 recipe 層從範圍中抽。

| 方法 | 運算 |
| --- | --- |
| `sox_volume_perturbed(wav, vol_ratio, sr)` | `wav · vol_ratio`，線性增益，不會截斷 |
| `sox_speed_perturbed(wav, speed, sr)` | `torchaudio.functional.resample(orig=int(sr·speed), new=sr)`：長度乘 `1/speed`、音高乘 `speed`（sox `speed` 的語意） |
| `sox_pitch_perturbed(wav, shift_ratio, sr)` | `torchaudio.functional.pitch_shift`，`bins_per_octave=1200`，因此 `shift_ratio` 單位是 cent；phase vocoder，長度不變；0 不做事 |

名稱沿用 sox 的用語；實際上都不呼叫 sox。

## 噪音

- `add_bg_noise(wav, snr_list, sr, dynamic_type=False, noise_id=None, noise_transform=None)`：
  抽一個噪音 id（或使用傳入的 `noise_id`：先前呼叫回傳的 id，或 id 串列），載入並重新取樣到 `sr`，套用
  `noise_transform`（例如一個房間通道，讓噪音與語音同處一室），再以
  [`noise.add_bg_noise`](noise.zh-TW.md) 對 `snr_list` 中每個 SNR 混合。
  `dynamic_type=True` 抽兩段 clip 接成一條床。回傳
  `(noisy_list, (added_noise_list, noise_id, snr_list))`。
- `add_bg_white_noise(wav, snr_list)`：[`noise.add_bg_white_noise`](noise.zh-TW.md)；
  回傳 `(noisy_list, (noise_list, snr_list))`。

## 濾波與通道損傷

| 方法 | 運算 | info |
| --- | --- | --- |
| `apply_2nd_iir_response(wav, a_coeffs=None, b_coeffs=None)` | 隨機穩定 biquad，[`rand_add_2nd_filter_response`](impulse_response.zh-TW.md) | `(a, b)` |
| `apply_hpf(wav, sr, cutoff_freq, q_factor)` | 在 [`apply_linear`](dsp.zh-TW.md) 下的 `highpass_biquad` | `(cutoff, Q)` |
| `apply_src_effect(wav, sr, src_sr, src_backend)` | 以 [`wav_resampling`](dsp.zh-TW.md) 做 `sr → src_sr → sr`；`"torchaudio"` 時第二段沿用第一段抽到的隨機濾波器 | 重新取樣器的回傳值 |
| `apply_gain_distortion(wav, sr)` | [`rand_gain_distortion`](volume.zh-TW.md)：隨機片段增益，再截到 [-1, 1] | `(start, length, gain)` |
| `apply_clipping_distortion(wav, min_quantile, max_quantile)` | [`wav_clipping`](volume.zh-TW.md)，在訊號自己的分位數截斷 | 兩個分位數比例 |
| `apply_media_coloring(wav, sr, hp_cutoff, lp_cutoff, compress_power=None)` | 電視／喇叭播放：在 `apply_linear` 下做高通與低通 biquad（Q 0.707），`p < 1` 時再做 `sign(x)·abs(x/peak)^p·peak` waveshaping；RMS 還原為輸入值 | `(hp, lp, p)` |
| `apply_codec(wav, sr, codec_name, bit_rate=None)` | 透過 TorchCodec 編碼再解碼 | `(codec, bit_rate)` |
| `apply_packet_loss(wav, sr, packet_ms=20, loss_rate=0.05)` | 把整個封包清零，每個封包以 `loss_rate` 獨立遺失（torch generator） | `(packet_ms, loss_rate, n_dropped)` |

`apply_media_coloring` 還原 RMS，所以之後的 SIR 縮放不受染色影響。它的 waveshaper
模擬的是房間裡播放裝置的失真，在混合前套在干擾者上；它不是錄音鏈壓縮的模型，
後者是一條增益曲線（見 [`compressor_gain`](dsp.zh-TW.md)）。

`apply_codec` 支援 `supported_codecs() == ["libopus", "g722"]`。TorchCodec 的編碼器
以容器選 codec，所以來回轉換會寫一個暫存的 `.opus` 或 `.g722` 檔；以
`sample_rate=sr` 解碼可抵銷 codec 內部的取樣率（Opus 以 48 kHz、G.722 以 16 kHz
編碼），輸出再裁切或補零到輸入長度。G.722 忽略 `bit_rate`。

## 範例

```python
from puresound.audio.augmentation import AudioEffectAugmentor

aug = AudioEffectAugmentor()
aug.load_bg_noise_from_folder("/path/to/noise")
aug.init_room_simulator({
    "room_dim_range": [[3.0, 8.0], [3.0, 8.0], [2.4, 3.5]],
    "rt60_range": [0.15, 0.8],
    "source_receiver_distance_range": [0.3, 4.0],
})

scene = aug.sample_room_scene()
mix = aug.apply_rir(clean, rir_mode="full", sr=16000, room_scene=scene, source_role="foreground")
target = aug.apply_rir(clean, rir_mode="early", sr=16000, rir_id=mix.detail.rir_id).wav
(noisy,), _ = aug.add_bg_noise(mix.wav, snr_list=[5.0], sr=16000)
```
