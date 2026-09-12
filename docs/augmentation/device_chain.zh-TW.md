# 裝置鏈

English version: [`device_chain.md`](device_chain.md)

混音完成之後、進入模型之前，剩下的一切都屬於**收音與傳輸**：重取樣、麥克風
響應、rumble filter、前級增益、壓縮、A/D 轉換、VoIP codec、封包遺失。這些
處理不涉及語者也不涉及房間，因此集中在
`puresound/task/device_chain.py::DeviceChain`，而非散落在 `__getitem__` 中。

## 演算法面

### 1. 鏈的結構

```
SRC → 2nd-order IIR → HPF → volume/clipping → compressor    # 類比路徑（線性組）
  ────────────────── _analogue_to_digital ──────────────────  # A/D 邊界
  → codec → packet loss                                      # 數位傳輸損傷
```

分組的依據是**目標訊號是否跟隨**，而這又由該 stage 模擬的物理現象決定。

#### 1.1 類比組：目標跟隨

類比組模擬的是換能器頻率響應、濾波、前級增益、壓縮，全部是（時變）線性
算子。訓練對的定義是：**目標是模型應該「穿過這條通道」恢復出來的訊號**。
因此通道對混音做了什麼，也必須對目標做同樣的事。

線性保證這件事在數學上成立。對線性算子 `H`：

```
H(near + far + noise) = H(near) + H(far) + H(noise)
```

混音經過 `H` 之後仍然等於各成分經過 `H` 之後的和，且比例不變——recipe 指定
的 SIR 在通道之後依然成立。這是所有類比 stage 都以**相同參數**同時套用在
混音與目標上的原因。

#### 1.2 傳輸組：目標不跟隨

codec 失真與封包遺失是**損傷**，不是通道特性。模型應該學會修復它們，而不是
複製它們。因此這兩個 stage 只作用在混音上，目標維持為未受損的參考。

這個區分不是慣例問題，而是任務定義：如果目標也帶著 codec 失真，模型學到的
最佳行為是保留那些失真；如果目標乾淨，模型被要求去除它們。

### 2. 各 stage 的內容與 target 跟隨方式

| Stage | 模擬對象 | 數學 | target 如何跟隨 |
|---|---|---|---|
| `_sample_rate_conversion` | 鏈上經過不同取樣率 | `fs → src_sr → fs` 的頻寬受限與濾波器痕跡（[ch5](spectral_channel.zh-TW.md) §4） | 同一 backend；torchaudio 路徑重用混音抽到的三個濾波器參數 |
| `_second_order_iir` | 換能器頻率響應 | 隨機二階 IIR（[ch5](spectral_channel.zh-TW.md) §2） | 同一組 `(a, b)` 係數 |
| `_high_pass` | rumble filter | RBJ HPF，cutoff 加權抽樣、`Q ~ N(0.707, 0.1²)` 夾 [0.3, 1.3]（[ch5](spectral_channel.zh-TW.md) §3） | 同 cutoff、同 Q |
| `_volume` | 前級增益或類比過載 | 擲一次決定：增益 `U(perturbed_range)` 或分位數削波（[ch4](level_dynamics.zh-TW.md) §3–4） | 增益用同一比例；削波用混音**實現出的分位數數值** |
| `_compressor` | 廣播式動態壓縮 | 時變增益曲線（[ch4](level_dynamics.zh-TW.md) §6） | 曲線由混音導出，同一條曲線乘在兩者上 |
| `_analogue_to_digital` | 增益調度（gain staging） | 見 §3 | 成對同除以同一峰值 |
| `_codec` | VoIP / 電話編解碼 | encode → decode 往返 | 不跟隨（混音 only） |
| `_packet_loss` | 封包遺失 | 每包 Bernoulli 丟棄後歸零 | 不跟隨（混音 only） |

#### 2.1 削波的 target 處理值得單獨說明

`_volume` 的削波分支對混音使用抽樣得到的分位數 `(min_q, max_q)`，計算出
實際的門檻值 `(lo, hi)`；對 target 則直接使用**這兩個絕對數值**再削一次，
而不是用 target 自己的分位數。

理由是物理的：類比過載發生在麥克風輸出端，門檻由電路決定，是一個絕對的
電壓值。混音與目標經歷的是同一次過載，因此門檻相同。若各自用自己的分位數，
較安靜的 target 會被削在較低的絕對位準上，等於它經歷了一次更嚴重的過載。

### 3. A/D 邊界：整條 pipeline 唯一有滿刻度的位置

`_analogue_to_digital` 的語義定義了上游所有 stage 的位準觀，值得單獨一節。

#### 3.1 兩側的語義

**上游是聲學與類比路徑。** 那裡的位準是聲壓經過換能器與前級放大器。聲壓
沒有滿刻度——峰值超過 1.0 不是錯誤，而是「這個房間很大聲」。因此上游任何
stage 都不得把 1.0 當作天花板，這正是 `apply_linear`
（[ch5](spectral_channel.zh-TW.md) §7）存在的理由。

**下游是數位世界。** codec 無法編碼超過滿刻度的樣本；模型輸出也被夾在
[−1, 1]，因此一個峰值超過 1.0 的目標是模型無論分離得多好都達不到的。

#### 3.2 跨越邊界的操作是增益調度，不是削波

```
peak = max( max|noisy|, max|target| )
if peak > 1.0:
    noisy  ← noisy  / peak
    target ← target / peak
```

這對應的現實動作是**錄音師把前級增益調到不會打爆轉換器**。它是一次線性
縮放，因此疊加性存活、混音仍然等於各成分在指定 SIR 下之和。

用兩者的**共同**峰值而非各自的峰值，是維持位準關係的必要條件：分別正規化
會改變兩者的相對音量。

#### 3.3 蓄意的過載為什麼不放在這裡

蓄意的過載失真是另一件事，活在 `_volume` 的削波分支中：它有機率控制、有
provenance 記錄（`volume_clipped`）、以相同的絕對門檻打在兩個訊號上。

如果 A/D 邊界也做削波，recipe 中被 `_volume` 削過的列會被削第二次，而且
第二次沒有任何記錄——eval 無法區分「這列被削了一次」與「兩次」。

#### 3.4 其他性質

* 這一步**不消耗隨機性**。
* 可用 `overload_guard=False` 關閉。TSE 任務歷史上沒有這一級，把它做成
  knob 而不是共用元件強加的行為，是為了不改變該任務訓練的分佈。
* `overload_rescaled` 記錄該列是否觸發，讓 eval 能識別被整體降低過位準的列。

### 4. Codec 與封包遺失

#### 4.1 Codec

實作以 `torchaudio.io.AudioEffector` 執行真實的 encode → decode 往返，不是
用濾波器近似。因此得到的失真包含該 codec 的實際行為：量化噪音、頻寬限制、
心理聲學模型造成的頻譜修改。

每個 codec 綁定一個容器格式（`libopus` → ogg、`g722` → matroska），選擇
標準是語音頻寬用途下的往返可靠性（避免 MKV 內嵌 Opus 之類的相容性問題）。

兩個 codec 的特性：

* **libopus**：現代 VoIP 主流，bitrate 可調（config 的 `bitrate_range`
  逐 codec 指定），低 bitrate 時以頻寬限制與參數化編碼為主。
* **g722**：ITU-T 子頻帶 ADPCM，寬頻電話標準。內部強制 16 kHz、無 bitrate
  knob（固定 64 kbps），因此該 codec 的 `bitrate_range` 設定會被忽略。

輸出長度處理：codec 往返可能改變樣本數（編碼延遲、frame 對齊），實作截短或
零填補回原始長度，保證下游的長度假設不變。

#### 4.2 封包遺失

```
packet_samples = round(fs · packet_ms / 1000)
每個封包以 Bernoulli(loss_rate) 決定是否丟棄
被丟棄的封包區間歸零
```

`packet_ms` 預設 20 ms，是 WebRTC 的預設封包長度；60 ms 是低頻寬 Opus 的
常見設定。

實作直接歸零，不做任何隱藏（packet loss concealment, PLC）。這是刻意的
最壞情況模型：真實 VoIP 端點的 PLC 會用外插或波形重複填補缺口，但那是
端點的行為，各家實作不同。訓練的目標是讓上游模型對「訊號真的缺了一段」
有韌性，而不是模擬某個特定端點的 PLC 演算法。

**獨立假設的限制**：實作對每個封包獨立擲骰。真實網路的封包遺失有突發性
（burst loss，連續丟失數個封包），獨立模型低估了長缺口的機率。目前沒有
突發模型；若需要，可用 Gilbert–Elliott 兩狀態馬可夫鏈擴充。

## 工程面

### 兩條契約

這兩條是模組 docstring 明文列出的契約，任何修改都必須維持。

#### 契約一：stage 順序是 load-bearing 的

每個 stage 都從共享的 RNG 流抽樣。交換兩個 stage 的順序會改變後續所有
stage 抽到的隨機數，因此同一個 seed 產生不同的資料——即使每個 stage 各自
的實作完全沒變。

現行順序就是已發佈 checkpoint 訓練時使用的順序。順序本身也有物理依據
（換能器響應在前級之前、壓縮在轉換器之前），但即使物理上可以互換的兩級，
也不應該為了整齊而交換。

#### 契約二：關閉的 stage 完全不接觸 RNG 流

每一級的守衛形狀固定：

```python
block is not None and block.used and torch.rand(1) < block.prob
```

機率擲點放在 short circuit 的**內部**。因此關閉一級（或 dataset 根本沒有
該區塊）之後，後面每一級抽到的隨機數與原來完全相同。

這是「新增 knob 之後舊 recipe 仍能位元級重現」的機制。詳細討論見
[ch8](engineering_contract.zh-TW.md)。

### Provenance scalars

每一列記錄鏈實際執行了什麼（`DEVICE_CHAIN_SCALARS`），供 eval 按「這列
走過哪條通道」分桶分析——與 `eval_indomain.py --by-bucket` 已用於 SIR /
overlap / DRR 的做法相同。

慣例：

* `*_applied`、`*_clipped`、`*_rescaled` 為 0.0 / 1.0。
* 參數欄在該級未開火的列上為 `NaN`。
* **每一列都攜帶每一個 key**（值可能是 NaN）。這是它們能搭上既有 scalar
  collate（每個 key 一次 `torch.cat` 0 維張量）的前提；缺 key 的列會讓
  batch 無法組成單一張量。
* codec 種類以 float code 發出（`CODEC_CODES`），與 `MIX_MODE_CODES` 同一
  慣例——字串無法進入 scalar collate。

字串型的 `RIR_PROVENANCE_KEYS` 走另一條 collate 路徑。模組註解把它標為
警世案例：為了追溯而加入，但從未被任何分析讀取。新增記錄欄之前，先確定
誰會讀它。

### Config 對照

| Stage | Config 區塊 | Schema |
|---|---|---|
| SRC | `augmentation_src` | `SourceRateAugmentation` |
| 2nd IIR | `augmentation_ir_response` | `SimpleProbAugmentation` |
| HPF | `augmentation_hpf` | `HighPassAugmentation` |
| volume | `augmentation_volume` | `VolumeAugmentation` |
| compressor | `augmentation_compressor` | `CompressorAugmentation` |
| codec | `augmentation_codec` | `CodecAugmentation`（僅 NS / voice isolation） |
| packet loss | `augmentation_packet_loss` | `PacketLossAugmentation`（同上） |

`device_chain_from_blocks` 從 dataset 已驗證的區塊建構鏈。缺少某區塊的任務，
該級直接缺席且零成本——這是 NS 與 TSE 能共用同一條鏈的機制（`getattr` 取
不到的區塊傳入 `None`，`_fires` 的第一個條件就會短路）。

### 陷阱

* **對這條鏈的任何行為修改都會改變合成分佈**，包括「把不精確的近似修正確」
  這類明顯正向的修改。修改前後訓練的 checkpoint 在合成域指標上不可直接
  比較。詳見 [ch8](engineering_contract.zh-TW.md) 的分佈版本一節。
* VAD 參照在鏈**之前**快照（`ns.py`）。VAD 標籤打在乾淨（早期殘響）訊號
  上，因為 Silero 這類 VAD 在重度失真的語音上不可靠。鏈的損傷因此不會
  改變活動標籤。
* 壓縮器曲線與訊號長度以 `min()` 對齊。SRC stage 可能造成 1–2 樣本的長度
  差異，尾端未覆蓋的樣本保持原值。
* `_fires` 的 `probability_draw=False` 參數存在但目前無呼叫端使用；它讓
  某個 stage 可以「只看 `used` 不擲機率」。若要使用，注意它會改變該列的
  RNG 消耗量。
