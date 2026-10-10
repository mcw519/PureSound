# 工程契約

English version: [engineering_contract.md](engineering_contract.md)

讓合成可重現、可比較、可安全擴充的規則。它們適用於 `DeviceChain`、
`NoiseStage`、`OverlapGating`、`AudioEffectAugmentor` 與各 dataset 的
`__getitem__`；各模組 docstring 就地陳述，本頁集中整理。

## 1. RNG 契約

合成從三個全域產生器抽樣：Python `random`、NumPy `np.random` 與 PyTorch 全域
generator。

### 1.1 不觸發就不抽樣

每個機率門的形狀相同：

```python
block is not None and block.used and torch.rand(1) < block.prob
```

機率擲點位於 short circuit 內部。停用某個區塊，或任務根本沒有它，後面每一級抽到
的值都與原本完全相同，因此**新增一個 knob 不會改變任何未啟用它的 recipe 的資料**。
若擲點放在外面（先擲再看 `used`），即使是預設關閉的 knob 也會平移之後所有抽樣，
改變每個既有 recipe 的輸出。這條契約是 knob 能不斷累加的前提。

### 1.2 抽樣順序是輸出的一部分

對調兩級，或一級內的兩次抽樣，都會平移共用 stream；即使每一級本身都沒變，同一個
seed 也會產生不同資料。有些 stage 消耗的抽樣數會變動：
`OverlapGating._draw_overlap_rate` 在零重疊 regime 取一個值、其他 regime 取兩個，
因此連 regime 門檻的順序都不能改（[場景建構](scene_construction.zh-TW.md)）。

### 1.3 重試

- **`apply_linear` 的逐步退讓不抽樣。** 退讓會再次呼叫 backend，因此傳入的函式
  必須是純濾波器或增益；隨機參數在外面抽好、以 closure 傳入
  （[頻譜與通道效果](spectral_channel.zh-TW.md)）。
- **有上限的抽樣重試會抽樣，這是刻意的。** 重抽取樣率不符的干擾者語句（最多 5
  次）、在 `align_audio_list` 中重裁全靜音視窗（最多 10 次）、重開空白語句（最多
  5 次，之後報錯），都是抽樣邏輯的一部分，不是錯誤復原。次數有上限，消耗量也就
  有上限。

### 1.4 帶 seed 的項目重設三個 stream

`DynamicBaseDataset.parse_item_key` 接受
`(speaker, sample_rate[, seed[, seconds[, epoch]]])`，依序套用：列長、該 epoch 的
curriculum 值、最後是 seed：

```python
random.seed(seed)
np.random.seed(seed % (2**32))
torch.manual_seed(seed)
```

三個都必須重設：房間幾何經 NumPy 抽樣，語者與語句經 `random`，大多數機率門與範圍
抽樣經 PyTorch。`% 2**32` 是 NumPy 的 seed 範圍。先套 curriculum，代表每一次
seeded 抽樣都已看到該 epoch 的值；為新 epoch 重新組合各級
（`rebind_augmentation_blocks`）不抽樣。這就是驗證集能跨 epoch、跨 run、跨 worker
配置保持確定的機制。語句池是有序 list，從不是 set，因此同一個 seed 在任何 process
都選到同一段語句，與 `PYTHONHASHSEED` 無關。

## 2. 合成順序

`NoiseSuppressionDataset.__getitem__` 的一列（voice isolation 透過表中的 hook
替換成自己的列型）：

| # | 步驟 | 實作 | 頁面 |
|---|---|---|---|
| 1 | 載入，可選的 RMS 縮放 | `choose_an_utterance_by_speaker_name` → `AudioIO.open(target_lvl=...)` | [位準與動態](level_dynamics.zh-TW.md) |
| 2 | 裁切或補齊到列長（有上限的防靜音重試） | `align_audio_list` | [場景建構](scene_construction.zh-TW.md) |
| 3 | 列規劃（target-absent；VI 另有真實錄音列與 session 列） | `_plan_row` | [場景建構](scene_construction.zh-TW.md) |
| 4 | 前景通道（source-level RIR：混音用 `full`，目標用目標視窗） | `_prepare_foreground` | [房間聲學](room_acoustics.zh-TW.md) |
| 5 | 干擾者：抽樣、media coloring、逐源 RIR | `_sample_interferers` | [場景建構](scene_construction.zh-TW.md)、[頻譜與通道效果](spectral_channel.zh-TW.md) |
| 5a | RIR 抽取內、快取之前：DRR contrast，然後 direct smear | `AudioEffectAugmentor.apply_rir` | [距離線索](distance_cues.zh-TW.md) |
| 6 | overlap gating（Bernoulli 或 turn taking），除非列規劃跳過 | `OverlapGating.apply` | [場景建構](scene_construction.zh-TW.md) |
| 7 | 前景 × 干擾者混音（hard SIR；VI 為 `mix_mode`） | `_mix_foreground_with_interferers` | [場景建構](scene_construction.zh-TW.md) |
| 8 | target-absent 扣除 | `__getitem__` | [場景建構](scene_construction.zh-TW.md) |
| 9 | 以 ERLE 加入播放回音 | `__getitem__` | [場景建構](scene_construction.zh-TW.md) |
| 10 | 成對峰值保護 | `avoid_audio_clipping` | [位準與動態](level_dynamics.zh-TW.md) |
| 11 | 成對的 speed 擾動 | `sox_speed_perturbed` | [頻譜與通道效果](spectral_channel.zh-TW.md) |
| 12 | whole-mix RIR（沒有 source-level reverb 的列） | `AudioEffectAugmentor.apply_rir` | [房間聲學](room_acoustics.zh-TW.md) |
| 13 | 列首環境音前導：遮掉開頭數秒的所有語音 | `__getitem__` | [場景建構](scene_construction.zh-TW.md) |
| 14 | 噪音：recorded、白噪音、capture floor | `NoiseStage.apply` | [場景建構](scene_construction.zh-TW.md) |
| 15 | 擷取目標的 VAD 參考 | `__getitem__` | [裝置鏈](device_chain.zh-TW.md) |
| 16 | 裝置鏈（類比組、轉換器、傳輸組），加上可選的成對視圖 | `apply_chain_views` → `DeviceChain.apply` | [裝置鏈](device_chain.zh-TW.md) |
| 17 | 裁到列長、VAD 標籤、組 sample dict、provenance | `__getitem__` | — |

順序限制：

- **5a 在快取之前。** 目標以 `rir_id` 重新取出同一個 RIR 的另一個視窗；若在快取
  之後才修改，混音與目標會拿到不同的 impulse。
- **8 在任何縮放之前。** 扣除用的是任何位準改變之前擷取的前景貢獻。
- **13 在 11、12 之後，14、15 之前。** 時間軸已定，前導由該列自己的環境音填滿而
  不是數位靜音，標籤也繼承遮罩。
- **15 在 16 之前。** 標籤在乾淨目標上計算：在 speed 之後（時間軸與混音一致）、
  在任何鏈上損傷之前。
- **14 在 16 之前。** 噪音底與語音一起經過裝置響應。

## 3. 設定與程式

所有增強 schema 都是 `puresound/config/augmentation.py` 中的 Pydantic model，
建立在 `StrictConfig`（`extra="forbid"`）上。

- **未知 key 是錯誤。** 拼錯的 key 在載入時失敗，而不是讓區塊停在預設值；仍帶著
  已移除機制之 knob 的 recipe 會明確失敗。見
  [Configuration](../../usage/configuration.zh-TW.md)。
- **啟用時必填的欄位在載入時驗證。** 每個區塊的 `enabled_contract`（經
  `require_fields_when_enabled`）拒絕沒有參數的 `used: true`，而不是讓 `None` 在
  訓練途中才冒出來。
- **schema 預設值就是行為預設值。** dataset 以屬性讀取區塊並直接使用該值；沒有
  第二套預設值。
- **停用的區塊以 `None` 抵達 dataset。** `BaseRecipe.augmentation_kwargs` 把每個
  `augmentation_*` 區塊（以及 `vad_label`）以 `<name>_args` 轉交，`used: false`
  的區塊被換成 `None`。

### 區塊

| YAML 區塊 | Schema | 使用者 | 任務 |
|---|---|---|---|
| `augmentation_speech`（`media_voice`、`echo_playback`、`overlap_control`、`mix_mode`） | `SpeechAugmentation` | `ns.py`、`voice_isolation.py`、`OverlapGating` | NS、VI（`mix_mode` 僅 VI） |
| `augmentation_reverb`（`simulator`、`simulator.pregenerated`、`drr_contrast`、`direct_smear`） | `ReverbAugmentation` | `DynamicBaseDataset` 初始化、`AudioEffectAugmentor` | 全部 |
| `augmentation_noise`（`room_coloring`、`absolute_floor`、`noise_sources`、`snr_bands`） | `NoiseAugmentation` | `NoiseStage` | 全部 |
| `augmentation_src`、`augmentation_ir_response`、`augmentation_hpf`、`augmentation_volume`、`augmentation_compressor` | 各級 schema | `DeviceChain` | NS、VI、TSE |
| `augmentation_codec`、`augmentation_packet_loss` | `CodecAugmentation`、`PacketLossAugmentation` | `DeviceChain` | NS、VI |
| `augmentation_speed` | `ContinuousSpeedAugmentation`；speaker embedding 用 `DiscreteSpeedAugmentation` | `ns.py`、各任務 dataset | 全部 |
| `augmentation_target_absent` | `TargetAbsentAugmentation` | `_plan_row` | NS、VI |
| `augmentation_row_initial_ambient` | `RowInitialAmbientAugmentation` | `__getitem__` | NS、VI |
| `augmentation_realfar`、`augmentation_realnear` | `RealFarAugmentation`、`RealNearAugmentation` | `voice_isolation.py` | VI |
| `augmentation_session_rows` | `SessionRowsConfig`（開關名為 `enabled`） | `task/session_rows.py` | VI |
| `vad_label` | `VadLabelConfig`（backend 選項放在 `args`） | VAD labeler | 全部 |
| `curriculum` | `CurriculumConfig`（`puresound/config/curriculum.py`） | 逐 epoch 的 knob 排程，在 `parse_item_key` 中套用 | 全部 |

### 歸屬

- **區塊只存在於使用它的任務。** 每個任務的 recipe model 宣告自己的區塊；
  noise suppression recipe 上的 `augmentation_speech.mix_mode` 會被拒絕，
  `augmentation_realfar` 只是 voice isolation recipe 的欄位。
  `augmentation_session_rows` 與 `augmentation_row_initial_ambient` 互斥。
- **交給元件建構子的區塊只轉交 recipe 寫出的 key。** 房間模擬器、bank loader、
  VAD labeler、DRR contrast 與 direct smear 收到的是
  `delegated_kwargs(block)`（`exclude_unset=True`），其餘部分仍由元件自己的預設值
  決定。

## 4. 分佈改變與可比性

合成輸出就是訓練分佈。改變輸出就是新的分佈，合成域指標跨越它便不可直接比較：
差異可能來自模型，也可能來自資料。「改變輸出」包括修 bug、把近似改成精確、調整
抽樣順序、改預設值。

- **優先用預設關閉的 knob。** 依 §1.1，未啟用它的 recipe 不受影響，新分佈是
  opt-in。
- **無法避免的改變視為斷點。** 斷點兩側訓練的 checkpoint 只能在不經過合成的評估
  （真實錄音）上比較；合成指標需要重訓的對照組。commit message 要說明分佈已改變，
  以及哪些 checkpoint 落在哪一側。
- 哪個斷點何時發生，記錄在 commit 歷史，不在這裡。

## 5. 測試

| 測試 | 守護的內容 |
|---|---|
| `test/task/test_synthesis_fingerprint.py` | 完整寫出所有 knob 的 recipe 與最精簡的 recipe 合成出相同音訊（預設值只存在一處）；帶 seed 的項目是其 seed 的純函數 |
| `tools/rng_fingerprint.py` | 對真實 recipe 做前後比較，用於不得改變合成的重構；能抓到讓上一個測試兩個 recipe 一起移動的改變 |
| `test/task/test_device_chain.py` | stage 順序、每級動到哪些訊號、停用時不抽樣、類比路徑的線性、provenance key |
| `test/task/test_noise_stage.py`、`test/task/test_overlap_gating.py` | 來源或 gate 沒做事時不抽樣、回報的 SNR 與重疊率 |
| `test/audio/test_dsp.py` | `apply_linear`、重取樣器的線性、`compressor_gain` 的性質 |
| `test/config/test_config_schema.py` | 各層級拒絕未知 key、啟用時必填欄位、區塊歸屬正確的任務、停用區塊以 `None` 抵達 dataset |
| `test/audio/`、`test/task/` | 個別 knob（DRR contrast、direct smear、noise sources、row-initial ambient、session rows） |
| `test/rir/` | RIR 生成、校準與 bank 契約 |

某個改動讓 fingerprint 測試失敗時，作者必須明確決定這是有意的斷點（更新期望值並
記錄斷點），還是意外（還原）。

## 6. 新增一個 knob

1. **Schema。** 加欄位與它的 enabled contract；`used: false` 時不得要求任何其他
   欄位。
2. **RNG。** 把機率擲點放進 short circuit，並以 `tools/rng_fingerprint.py` 確認
   不含此 knob 的 recipe 沒有改變。
3. **位置。** 放進 §2 的順序，並檢查那裡列出的限制（快取、扣除、標籤）。
4. **Provenance。** 若評估需要依它分桶，加進對應的 scalar 清單，並在加入前說明
   誰會讀它。
5. **線性。** 模擬線性裝置、且作用在混音／目標對上的 stage，要用 `clamp=False` 或
   `apply_linear`。
6. **文件。** 在 `docs/algorithms/augmentation/` 對應頁面說明演算法，在
   `docs/architecture/task/` 說明 dataset 行為，兩種語言都要。註解與文件遵守
   [Repository layout](../../repository_layout.zh-TW.md) 的規則：不寫結果、日期或
   版本名稱；需要證據時，指向固定住該行為的測試或論文。
