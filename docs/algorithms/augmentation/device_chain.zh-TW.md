# 裝置鏈

English version: [device_chain.md](device_chain.md)

`puresound.task.device_chain.DeviceChain` 負責混音完成之後、進入模型之前的
一切：重取樣、換能器響應、rumble filter、前級增益或過載、壓縮、A/D 轉換、
VoIP codec 與封包遺失。它不涉及語者也不涉及房間。noise suppression、voice
isolation 與 target speaker extraction 三個 dataset 以
`device_chain_from_blocks(augmentor, dataset)` 建立一條鏈，並在 noise stage
之後對每一列呼叫一次 `apply(noisy, target, sample_rate=...)`。

## 結構

```
SRC → 2nd-order IIR → HPF → volume / clipping → compressor     類比路徑：目標跟隨
──────────────────── _analogue_to_digital ────────────────────   A/D 邊界
→ codec → packet loss                                           傳輸損傷：只作用於混音
```

stage 屬於哪一組，決定乾淨目標是否也要經過它。

**類比路徑：目標跟隨。** 換能器響應、濾波、前級增益與壓縮都是（時變）線性
算子。目標的定義是模型應該「穿過這條通道」恢復出來的訊號，因此每一級都以
相同參數同時套用在混音與目標上。線性保證這一對訊號保持一致：

```
H(near + far + noise) = H(near) + H(far) + H(noise)
```

經過 `H` 之後，混音仍然是各成分之和，且 SIR 與 SNR 維持 recipe 抽到的值。

**傳輸：只作用於混音。** codec 失真與封包遺失是模型應該修復的損傷，不是應該
重現的通道。若目標也帶著它們，最佳的模型就會保留它們。

## 各級 stage

| Stage | 模擬對象 | 計算 | 目標 |
|---|---|---|---|
| `_sample_rate_conversion` | 鏈上某處經過較低的取樣率 | `fs → src_sr → fs`；`src_sr` 依權重 `prob_each` 從 `src_range` 抽出；重取樣器在固定與隨機化兩個 backend 之間 50/50 選擇（[頻譜與通道效果](spectral_channel.zh-TW.md)） | 同一 backend；隨機化 backend 重用混音的濾波器參數 |
| `_second_order_iir` | 換能器頻率響應 | 隨機、穩定的二階 pole/zero 濾波器 | 同一組 `(a, b)` |
| `_high_pass` | rumble filter | RBJ high-pass；cutoff 依權重 `prob_each` 從 `cutoff` 抽出；`Q ~ N(0.707, 0.1²)` 夾在 [0.3, 1.3] | 同 cutoff、同 Q |
| `_volume` | 前級增益或類比過載 | 對 `clipping_prob` 擲一次決定分支：增益 `g ~ U(perturbed_range)`（單純相乘），或分位數削波（[位準與動態](level_dynamics.zh-TW.md)） | 增益：同一個 `g`；削波：依 `target_clipping`（見下） |
| `_compressor` | 廣播或會議系統的壓縮 | 由 `compressor_gain` 在混音上算出的時變增益曲線 | 同一條曲線 |
| `_analogue_to_digital` | 轉換器前的增益調度 | 共同峰值超過 1 時，兩者同除以該峰值 | 同一個除數 |
| `_codec` | VoIP 或電話編碼 | 真實的 encode → decode 往返 | 只作用於混音 |
| `_packet_loss` | 網路封包遺失 | 每個封包 Bernoulli 丟棄、補零 | 只作用於混音 |

### 削波與目標

削波分支抽出分位數水準 `min_q ~ U(clipping_range.min)` 與
`max_q ~ U(clipping_range.max)`，把混音削在
`[Q_min_q(noisy), Q_max_q(noisy)]`。目標怎麼削由
`augmentation_volume.target_clipping` 決定：

| `target_clipping` | 目標削在 | 效果 |
|---|---|---|
| `mixture_level`（預設） | 混音的門檻 `[Q_min_q(noisy), Q_max_q(noisy)]` | 比門檻安靜的目標不受影響 |
| `own_quantile` | `[Q_min_q(target), Q_max_q(target)]`，同一組分位數水準、在自己的樣本上算 | 不論位準，目標都被削得和混音一樣重；出貨的降噪 recipe 用的是這個 |

兩種模式下混音的削法完全相同，分支抽的亂數也相同。這是類比路徑唯一允許的
非線性；在被削波的列上，目標並不精確等於混音中的近場成分。voice isolation 的
recipe 設 `clipping_prob: 0`；noise suppression 的 recipe 會啟用這個分支。

## A/D 邊界

`_analogue_to_digital` 是整個合成流程中唯一存在數位滿刻度的位置。

- **上游是聲學與類比路徑。** 那裡的位準是經過換能器與前級的聲壓；峰值超過
  1.0 代表房間很大聲，不是錯誤。上游任何一級都不得把 1.0 當作天花板，這正是
  濾波器要經過 `apply_linear` 或以 `clamp=False` 執行的原因
  （[頻譜與通道效果](spectral_channel.zh-TW.md)）。
- **下游是數位。** codec 無法編碼超過滿刻度的樣本，模型輸出也被夾在 [−1, 1]，
  因此超過滿刻度的目標永遠無法達到。

跨越邊界的動作是增益調度，也就是錄音師把前級調到不會打爆轉換器：

```
peak = max(max|noisy|, max|target|)
if peak > 1:  noisy ← noisy / peak;  target ← target / peak
```

共用一個除數，同時保住兩者的位準關係與疊加性。蓄意的過載則放在 `_volume`：
它受機率控制，並記錄為 `volume_clipped`；若這裡也削波，那些列會被削第二次，
而且沒有任何記錄。

這一步不消耗隨機性，並記錄 `overload_rescaled`。
`DeviceChain(..., overload_guard=False)` 可移除這一步；所有任務建鏈時都開啟它，
測試則用這個旗標隔離類比路徑的各級。

## Codec

`AudioEffectAugmentor.apply_codec` 以 TorchCodec 的 `AudioEncoder` 編碼到暫存檔，
再以 `AudioDecoder(path, sample_rate=sr)` 解碼回來。encoder 沒有 codec 參數，
因此由容器副檔名選擇 codec：`libopus` → `.opus`、`g722` → `.g722`。得到的
失真是 codec 的真實行為（量化、頻寬限制、心理聲學塑形），不是濾波器近似。

- **libopus** 內部以 48 kHz 編碼；bitrate 是從 `bitrate_range.libopus` 均勻
  抽出的整數，單位 bit/s。
- **g722** 是 ITU-T G.722 子頻帶 ADPCM，16 kHz、固定 64 kbit/s，沒有 bitrate
  可調。

codec 依權重 `prob_each` 從 `codecs` 中選出（省略時均勻選擇）。encoder 延遲與
frame 對齊會改變長度，因此解碼結果截短或補零回輸入長度。

## 封包遺失

```
packet_samples = max(1, round(fs · packet_ms / 1000))
n_packets      = floor(T / packet_samples)
drop_k ~ Bernoulli(loss_rate),  k < n_packets；被丟棄的封包歸零
```

`packet_ms` 從 `packet_ms_choices` 抽出（20 ms 是 WebRTC 預設，60 ms 常見於
低 bitrate Opus），`loss_rate ~ U(loss_rate_range)`。結尾不足一包的部分永遠
不會被丟棄。

兩個刻意的簡化：

- **補零、不做補償。** 封包遺失補償（PLC）是端點行為，各實作不同；模型應該
  能應付真正缺失的音訊。
- **獨立遺失。** 真實網路遺失是成串的，因此長缺口的機率被低估。自然的延伸是
  Gilbert–Elliott 雙狀態模型；目前沒有實作。

## 契約

**stage 順序固定。** 每一級都從共用的 RNG stream 抽樣，因此即使兩級本身都沒
改，對調它們也會改變 seeded recipe 的產出。這個順序也符合實體鏈：換能器在
前級之前、壓縮在轉換器之前、轉換器在 codec 之前。

**停用的 stage 不抽樣。** 每個 guard 都是
`block is not None and block.used and torch.rand(1) < block.prob`，因此停用某一
級，或任務根本沒有該區塊，後面每一次抽樣都不變。見
[工程契約](engineering_contract.zh-TW.md)。

**沒有區塊的 stage 就不存在。** `device_chain_from_blocks` 讀取 dataset 的
`augmentation_src_args`、`augmentation_ir_response_args`、
`augmentation_hpf_args`、`augmentation_volume_args`，並以 `getattr` 讀取
compressor、codec 與 packet-loss 區塊；沒有該區塊的任務得到 `None`，該級不
產生任何成本。因此 target speaker extraction 沒有 codec 與 packet-loss 兩級。

## Provenance

`apply` 回傳 `ChainResult(noisy, target, applied)`；`applied` 對
`DEVICE_CHAIN_SCALARS` 的每個 key 各存一個 float：

| Key | 值 |
|---|---|
| `src_applied`、`iir_applied`、`hpf_applied`、`volume_applied`、`volume_clipped`、`compressor_applied`、`codec_applied`、`packet_loss_applied`、`overload_rescaled` | 0.0 或 1.0 |
| `src_target_sr`、`hpf_cutoff`、`volume_gain`、`compressor_ratio`、`compressor_threshold_db`、`codec_bitrate`、`packet_loss_rate` | 抽到的值；該級未觸發時為 NaN |
| `codec_kind` | `CODEC_CODES`：libopus 1.0、g722 2.0 |

每一列都帶齊所有 key，collate 才能對每個 key 串接成一個 tensor；字串無法走
這條路徑，所以 codec 以 float 代碼表示。隨機 IIR 只記錄是否觸發，不記錄係數。
評估時可依這些值分桶，例如
`egs/voice_isolate/scripts/eval_indomain.py --by-bucket`。

## 成對的通道視圖

`puresound.task.paired_views.apply_chain_views` 可以在完全相同的加噪後訊號對
上，再跑一次獨立抽樣的裝置鏈，用於一致性監督。voice isolation 透過
`augmentation_session_rows.paired_view_prob` 在 session 列上啟用（列長至少
`paired_view_min_seconds`）；第二個視圖在 collate 時放在 `paired_view` 之下。
機率為 0 時不做任何 clone，也不抽樣。

## 設定

| Stage | YAML 區塊 | Schema（`puresound.config.augmentation`） | 任務 |
|---|---|---|---|
| SRC | `augmentation_src`：`src_range`、`prob_each` | `SourceRateAugmentation` | NS、VI、TSE |
| IIR | `augmentation_ir_response` | `SimpleProbAugmentation` | NS、VI、TSE |
| HPF | `augmentation_hpf`：`cutoff`、`prob_each` | `HighPassAugmentation` | NS、VI、TSE |
| volume | `augmentation_volume`：`perturbed_range`、`clipping_prob`、`clipping_range.min`、`clipping_range.max`、`target_clipping` | `VolumeAugmentation` | NS、VI、TSE |
| compressor | `augmentation_compressor`：`threshold_db_range`、`ratio_range`、`attack_ms_range`、`release_ms_range` | `CompressorAugmentation` | NS、VI、TSE |
| codec | `augmentation_codec`：`codecs`、`prob_each`、`bitrate_range` | `CodecAugmentation` | NS、VI |
| packet loss | `augmentation_packet_loss`：`packet_ms_choices`、`loss_rate_range` | `PacketLossAugmentation` | NS、VI |

```yaml
augmentation_src:
  used: True
  prob: 0.5
  src_range: [8000, 16000]
  prob_each: [0.2, 0.8]
augmentation_hpf:
  used: True
  prob: 0.25
  cutoff: [100, 200, 300]
  prob_each: [0.4, 0.4, 0.2]
```

`src_range` 中等於該列取樣率的項目是 no-op 重取樣，但仍會設 `src_applied`，
並消耗相同的抽樣。

## 附註

- VAD 參考訊號在裝置鏈之前擷取（`ns.py`），因此鏈上的損傷永遠不會改變活動
  標籤。
- 壓縮曲線與訊號以最短長度對齊；未被覆蓋的尾端樣本保持原值。
- 對裝置鏈的任何行為修改都會改變合成分佈，修 bug 也一樣；見
  [工程契約](engineering_contract.zh-TW.md)。
- `_fires(block, probability_draw=False)` 只檢查 `used` 而不擲機率。目前沒有
  任何一級使用它；一旦使用，會改變該列的 RNG 消耗量。
