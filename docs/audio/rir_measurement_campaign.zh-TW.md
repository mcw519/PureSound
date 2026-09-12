# RIR 量測與校正

English: [rir_measurement_campaign.md](rir_measurement_campaign.md)

本文件定義 inverse calibration 可接受的實測房間資料。目標是擬合房間行為，
避免把 transducer response、幾何誤差、clock delay 或資料洩漏誤認成材料參數。

Schema：`puresound.rir_measurement_campaign.v1`

Loss policy：`puresound.rir_calibration_loss.v1`

## 必要量測資產

每組 source/receiver configuration 至少要有兩次 exponential sine sweep（ESS）
錄音，並保留：

- 原始 sweep recordings；
- deconvolution 使用的 inverse filter；
- background-noise recording；
- deconvolution、harmonic separation 與 latency 設定；
- 最終 RIR。

同一 capture 的資產必須使用相同 sample rate。重複量測用來估計 repeatability
並排除不穩定資料，不只是為了取平均。

## 房間紀錄

每個 measured room 都需要：

- 穩定的 `room_id` 與 room type；
- 有文件說明的右手座標系；
- geometry uncertainty；
- dimensions，或附 SHA-256 的保留 mesh；
- source 與 receiver poses；
- 環境資料；
- 保留資產的 hashes。

同一實體房間不能因 session 或位置不同而改名。

### Transducer 與 pose

Source 和 receiver 分開記錄 manufacturer、model、serial number、reference
axis、calibration date、calibration response 與 provenance。

Pose 包含 position、yaw/pitch/roll，以及位置和角度的不確定度。只有距離不足以
識別反射幾何與 directivity。

### 環境與同步

保存溫度、相對濕度與氣壓，因為它們會影響音速與空氣吸收。

Multi-channel spatial measurement 必須 sample-synchronized，並代表同一次激發
被多個 receivers 接收。若 channels 是不同 source positions，只能做獨立 mono
metrics，不能計算 coherence 或 IACC。

### 資產 identity

Asset path 必須相對於 campaign root，且不能包含 `..`。每個保留檔案都有
SHA-256。Campaign audit 會檢查檔案存在、hash 正確，以及同一路徑是否被宣告成
不同內容。

## Data split

Split 必須 room-disjoint：

- `train`：校正位置與 deterministic position holdout；
- `validation`：整個房間不得參與 fitting；
- `test`：整個房間不得參與 fitting 與 model selection。

`deterministic_position_assignments()` 由 campaign、room 與 measurement ID
推導 position holdout，因此重跑會得到相同 split。

Train room 需要足夠的不同位置與方向，才能識別所選參數。同一 pose 的 repeated
takes 能改善 uncertainty estimate，但不算新的 configuration。

## 校正目標

Reference report 合併多種互補項目：

```text
L = w_stft L_stft
  + w_edc L_edc
  + w_arr L_arr
  + w_oct L_oct
  + w_sp L_sp
  + w_causal R_causal
  + w_decay R_decay
```

| 項目 | 檢查內容 |
|---|---|
| Multiresolution STFT | Early structure 與頻譜細節 |
| Energy-decay curve | 對齊 direct arrival 後的 decay shape |
| Arrival timing | Direct 與 early reflection timing |
| Octave acoustics | 各頻帶 decay 與 level |
| Spatial coherence | 同步 receivers 的關係 |
| Causality regularization | 物理到達前的能量 |
| Decay regularization | 不合理或不穩定的 tail |

`analyze_rir_calibration_loss()` 會分別報告各項 loss 與 diagnostics。它是
NumPy/SciPy reference metric，不是 autograd loss。

非同步資料的 spatial term 應為 `not_applicable`，不能填成零後算作成功比較。

## 校正流程

1. Audit campaign contract 與保留資產。
2. 固定 room 與 position assignments。
3. 只 fitting 已啟用的 parameter groups。
4. 用 synthetic fixtures 檢查 identifiability 與 recovery。
5. 評估 train room 的 position holdout。
6. 評估完整 validation 與 test rooms。
7. Spatial parameters 只使用同步陣列資料。
8. Physical fit 穩定後才加入 constrained residual model。

Parameter group 應對應可觀察原因，例如 material、directivity、timing 與
late-field behavior。若 campaign 無法分開識別兩組參數，應固定其中一組或補做
更適合的量測。

Residual correction 必須 bounded、causal 且可追溯，不得用來掩蓋錯誤幾何、
asset audit failure 或 room leakage。

## 執行工具

從 template 開始：

```text
egs/rir_generation/phases/m5_calibration/config/
  m5_measurement_campaign_template.json
```

執行 fitting：

```bash
python egs/rir_generation/phases/m5_calibration/scripts/fit_m5_measured_campaign.py --help
```

常用驗證入口：

```bash
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measurement_contract.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measured_runner.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_group_identifiability.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_spatial_calibration.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_constrained_residual.py
python egs/rir_generation/phases/m5_calibration/scripts/validate_m5_exit.py
```

產生的 reports 預設是本機實驗輸出；只有 release evidence bundle 明確要求時
才納入版控。

## 解讀限制

通過 schema 與 synthetic recovery 代表管線實作一致，不代表已證明 real-room
generalization。Production evidence 仍需 controlled measurements、
room-disjoint evaluation，以及適合該 release 的 listening 或 downstream-task
結果。
