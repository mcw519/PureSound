# 工程契約

English version: [`engineering_contract.md`](engineering_contract.md)

前面各章說明手法本身；本章說明讓這些手法可重現、可比較、可安全擴充的橫切
規則。這些規則分散在各模組的 docstring 中，此處集中整理。

## 1. RNG 決定性契約

整條合成管線共用三個全域隨機數產生器：Python `random`、NumPy 的
`np.random`、PyTorch 的全域 generator。契約有四條，`DeviceChain`、
`NoiseStage`、`OverlapGating`、`AudioEffectAugmentor` 全部遵守。

### 1.1 不開火就不抽樣

每個機率門的形狀固定：

```python
block is not None and block.used and torch.rand(1) < block.prob
```

機率擲點位於 short circuit 的**內部**。

**效果**：關閉某個區塊（或 dataset 根本沒有該區塊）之後，後面每一級抽到的
隨機數與原來完全相同。因此**新增一個 knob 不會改變任何未啟用它的 recipe
產生的資料**——舊 recipe 位元級重現。

**為什麼這條契約值得如此嚴格**。若機率擲點在 short circuit 外面（先擲再看
`used`），關閉一個 stage 會平移之後所有 stage 的隨機數。結果是：新增一個
預設關閉的 knob 也會改變所有既有 recipe 的輸出，任何 checkpoint 都無法與
它之前訓練的 checkpoint 比較。這條契約是「knob 可以安全累積」的前提。

### 1.2 抽樣順序是 load-bearing 的

交換兩個 stage、甚至交換單一 stage 內兩次抽樣的先後，都會平移共享的 RNG
流，使同一個 seed 產生不同的資料。「每個 stage 各自的實作沒變」不構成順序
可以調整的理由。

一個容易忽略的案例：`OverlapGating._draw_overlap_rate` 的**抽樣消耗量依
regime 而異**（[ch7](scene_construction.zh-TW.md) §3.2）。零重疊 regime 只
消耗一個隨機數，另兩個消耗兩個。因此連 regime 門檻的語義順序都不可交換。

### 1.3 重試路徑的隨機性規則

有兩類重試，規則不同：

* **`apply_linear` 的逃逸重試不得消耗隨機性**。它會重新呼叫 backend，因此
  規定傳入的 `fn` 必須是純濾波或純增益（[ch5](spectral_channel.zh-TW.md)
  §7.4）。需要隨機參數時，在外面抽好以閉包傳入。
* **有界的取樣重試會消耗隨機性，且這是預期行為**。取樣率不符的語者重抽、
  防靜音的裁窗重抽（[ch7](scene_construction.zh-TW.md) §2.1）都屬於抽樣
  邏輯本身，不是錯誤恢復。它們的次數有上界，因此消耗量有界。

### 1.4 seeded item 必須重設三個流

item key 帶 `item_seed` 時，`DynamicBaseDataset.parse_item_key` 會為所有 task
重設三個流：

```python
random.seed(item_seed)
np.random.seed(item_seed % (2**32))
torch.manual_seed(item_seed)
```

三個流都要重設，因為 pipeline 三者都用：房間幾何取樣走 NumPy、語者與語句
選擇走 Python `random`、多數機率門與範圍抽樣走 PyTorch。漏掉任一個，該列
就無法位元級重現。

`% (2**32)` 是 NumPy 的 seed 範圍限制。

這是決定性驗證集的實作機制：同一個 item 在不同 epoch、不同 run、不同
worker 佈局下產生完全相同的資料。

## 2. 合成順序總表

一列資料的完整站序：

| # | 步驟 | 實作位置 | 章 |
|---|---|---|---|
| 1 | 載入 + RMS rescale | `AudioIO.open(target_lvl=...)` | ch4 §1 |
| 2 | 裁切 / 對齊（含防靜音有界重試） | `align_audio_list` | ch7 §2.1 |
| 3 | Row 計畫（target_absent / realnear / realfar） | `_plan_row` | ch7 §1 |
| 4 | 前景通道（source-level RIR：full → 混音、early → 目標） | `_prepare_foreground` | ch2 §4 |
| 5 | 干擾者取樣 → media 上色 → 各自 RIR | `_sample_interferers` | ch7 §2 / ch5 §6 |
| 5a | （RIR 進 cache 前：DRR contrast → direct smear） | `Augmentor.apply_rir` | ch3 §3–4 |
| 6 | Overlap gating（Bernoulli / turn-taking） | `OverlapGating.apply` | ch7 §3 |
| 7 | 前景 × 干擾混音（hard SIR / mix_mode） | `_mix_foreground_with_interferers` | ch7 §4 |
| 8 | target-absent 減除 | `__getitem__` | ch7 §1.2 |
| 9 | Echo playback（ERLE） | `__getitem__` | ch7 §5 |
| 10 | 防削波成對縮放 | `avoid_audio_clipping` | ch4 §1 |
| 11 | 變速（成對） | `sox_speed_perturbed` | ch5 §5 |
| 12 | Whole-mix RIR（僅非 source-level 列） | `Augmentor.apply_rir` | ch2 §6 |
| 13 | 噪音三源（錄音 → 白噪 → 底噪） | `NoiseStage.apply` | ch7 §6 |
| 14 | VAD 參照快照 | `__getitem__` | ch7 §6 / ch6 |
| 15 | Device chain（類比組 → A/D → 傳輸組） | `DeviceChain.apply` | ch6 |
| 16 | 裁到 `sample_length`、組成 sample dict | `__getitem__` | — |

幾個順序的理由集中在此：

* **步驟 5a 必須在 cache 之前**：目標透過 `rir_id` 重取同一條 RIR，見
  ch2 §6。
* **步驟 8 必須在任何縮放之前**：減除用的是縮放前的快照，見 ch7 §1.2。
* **步驟 14 在變速之後、噪音之前**：時間軸要與混音一致，標籤要打在乾淨
  語音上。
* **步驟 13 在 15 之前**：底噪要與語音一起經過裝置響應，見 ch7 §6.3。

## 3. Config 與程式碼的對照

所有 schema 定義在 `puresound/config/augmentation.py`，使用 Pydantic strict
模式。兩條驗證原則：

**未知的 key 是錯誤，不是預設值。** 促成這個決定的是兩類失效：拼字錯誤
（`porb: 0.5`）過去會讓該區塊以機率 0 靜默執行；以及刪除機制不會刪除它的
knob（歷史上有 10 個 knob 在程式碼消失後仍留在 33 份 config 裡）。完整
討論見 `docs/configuration.md`。

**`used: true` 時必填欄位由 `enabled_contract` 驗證。** 這讓「啟用了但沒
設參數」在載入時就失敗，而不是在訓練中途拿到 `None`。

### 區塊對照表

| Config 區塊 | Schema | 消費者 |
|---|---|---|
| `augmentation_speech`（含 `media_voice`、`echo_playback`、`overlap_control`、`mix_mode`） | `SpeechAugmentation` | `ns.py` / `voice_isolation.py` / `OverlapGating` |
| `augmentation_reverb`（含 `simulator`、`pregenerated`、`drr_contrast`、`direct_smear`） | `ReverbAugmentation` | `dynamic_base` 初始化 + `Augmentor` |
| `augmentation_noise`（含 `room_coloring`、`absolute_floor`） | `NoiseAugmentation` | `NoiseStage` |
| `augmentation_src` / `ir_response` / `hpf` / `volume` / `compressor` / `codec` / `packet_loss` | 各自 schema | `DeviceChain` |
| `augmentation_speed` | `ContinuousSpeedAugmentation`（增強任務）/ `DiscreteSpeedAugmentation`（語者任務） | `ns.py` / 語者任務 |
| `augmentation_target_absent` | `TargetAbsentAugmentation` | `_plan_row` |
| `augmentation_realfar` / `augmentation_realnear` | `RealFarAugmentation` / `RealNearAugmentation` | `voice_isolation.py`（僅該任務） |
| `augmentation_vad_label` | `VadLabelConfig` | VAD labeler（backend 自有選項走 `args`） |

### 區塊歸屬原則

**一個區塊只存在於真的消費它的任務。** `mix_mode` 在純 NS dataset 上直接
拋錯；`augmentation_realfar` 只是 `voice_isolation` 的欄位。task discriminator
決定使用哪一個 model。

**整包轉交建構子的區塊只轉發 recipe 實際寫過的 key。** bank loader、room
simulator、VAD labeler 屬於這類。這讓元件自身的預設值仍然生效，而不是被
config model 的預設值覆蓋——`delegated_kwargs` 是這個行為的實作。

**Schema 的欄位預設值就是行為預設值。** dataset 以屬性存取讀取它們，因此
schema 上寫的預設值就是 pipeline 實際使用的值，不存在第二份預設值。

## 4. 分佈版本與 checkpoint 可比性

這條管線的輸出**就是訓練分佈**。任何改變輸出的修改都產生一個新的分佈，而
**跨分佈的合成域指標不可直接比較**——差異可能來自模型，也可能來自資料。

「改變輸出」的範圍比直覺更廣，包含：

* 修正一個 bug；
* 把不精確的近似改成精確；
* 調整抽樣順序；
* 改變任何 stage 的預設參數。

### 4.1 規則

**視同斷代。** 會改變合成輸出的修改，其前後訓練的 checkpoint 只能在**不
經過合成管線的評測**上互相比較（真實錄音關卡）。合成域指標要重訓對照組
才有意義。斷代點以 commit 記錄。

**優先做成預設關閉的 knob。** §1.1 的契約保證未啟用該 knob 的 recipe 不受
影響，因此新分佈是選擇加入的，不會強制斷代。這是擴充 pipeline 的首選方式。

**不可避免的斷代要明確標記。** 修改 commit 的訊息要說明分佈變了、哪些
checkpoint 落在界線的哪一側。

### 4.2 實驗紀錄的位置

哪個斷代發生在何時、哪些 checkpoint 屬於哪個分佈，屬於實驗紀錄，不在本
手冊，也不隨本次發佈。

## 5. 測試防線

| 測試 | 守護對象 |
|---|---|
| `test/test_utils/test_synthesis_fingerprint.py` | 合成輸出的指紋。分佈被無意改變時第一個失敗 |
| `test/test_utils/test_device_chain.py` | stage 順序、target 跟隨、RNG 契約、provenance scalars |
| `test/test_rir_*.py` 家族 | RIR 生成、校準、bank 契約（清單見 [ch2](room_acoustics.zh-TW.md) §9） |
| 各 task dataset 測試 | row 類型、gating 行為、seeded 重現 |

指紋測試是這套防線的核心：它把「分佈有沒有變」從人的判斷變成 CI 的判斷。
一個修改若讓指紋測試失敗，作者必須明確決定這是預期的斷代（更新指紋並記錄）
還是意外（修回去）。

## 6. 新增 knob 的檢查清單

1. **Schema**：加欄位，補 `enabled_contract`（`used: false` 時不應要求其他
   欄位）。
2. **RNG 契約**：機率擲點放進 short circuit 內部；補一個「關閉時舊 recipe
   指紋不變」的測試。
3. **套用位置**：確認它在 §2 順序表中的正確位置，特別是與 cache、減除、
   縮放相關的三個約束（§2 末段）。
4. **Provenance**：若 eval 需要按它分桶，加進對應的 scalar 清單；先確認
   誰會讀它（[ch6](device_chain.zh-TW.md) 的 `RIR_PROVENANCE_KEYS` 是反例）。
5. **線性檢查**：若這個 stage 模擬線性現象且會作用在 mixture/target 對上，
   確認它包在 `apply_linear` 內或使用了 backend 的 clamp 開關
   （[ch5](spectral_channel.zh-TW.md) §7）。
6. **文件**：本手冊對應章節補一節（演算法 + 工程兩面）；API reference
   （`docs/audio/`、`docs/task/`）補簽名。

## 7. 本手冊自身的編寫規則

* 收錄範圍是**手法本身**：數學推導、模型假設、參數的物理意義、實作契約、
  已知陷阱。
* **實驗結論、判定、量測數字一律不收錄**，保留於內部。手法的設計動機出自
  量測時，只給出處（docstring），不複述數字。
* 程式碼註解同樣不留實驗痕跡（repo 慣例）。docstring 中既有的量測引用是
  該 knob 的設計依據紀錄，遷移或刪除前先確認 benchmarks 已收錄同樣的內容。
