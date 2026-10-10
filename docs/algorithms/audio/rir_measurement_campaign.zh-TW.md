# RIR 量測 campaign 與校正

English version: [rir_measurement_campaign.md](rir_measurement_campaign.md)

`puresound.audio.rir.calibration` 把 RIR 生成器的物理參數擬合到實測房間。這裡的
一切設計都是為了不把房間行為和 transducer response、幾何誤差、時鐘延遲或資料
洩漏混為一談：campaign 契約規定量測必須保留什麼，audit 拒絕不完整的 campaign，
split 讓房間彼此不相交，loss 則把每一項分開回報。

| 模組 | 內容 |
|---|---|
| `measured_campaign.py` | campaign schema `puresound.rir_measurement_campaign.v1`、`audit_measurement_campaign` |
| `measured_runner.py` | `deterministic_position_assignments`、`run_measured_campaign_fit` |
| `loss.py` | `analyze_rir_calibration_loss`，policy `puresound.rir_calibration_loss.v1` |
| `inverse_m4.py`、`inverse_m5.py` | parameter-profile 與 grouped-path-gain 反解擬合、局部可辨識性 |
| `synthetic_recovery.py` | 在合成 fixture 上加入受控擾動做參數還原 |
| `residual.py` | 受限的因果 residual |

## Campaign 契約

`RIRMeasurementCampaign` 包含 `campaign_id`、`rooms`、`transducers`、`records`、
`room_splits`（每個房間恰好被指派到 `train`、`validation` 或 `test` 之一）與
`provenance`。所有 JSON mapping 都必須是 strict JSON（不含 NaN/Inf）。

**資產。** `MeasurementAsset` 是相對於 campaign 的路徑（不可為絕對路徑、不可含
`..`）、64 位十六進位 SHA-256、media type，以及音訊資產需同時提供的 sample rate、
channel 數與 frame 數。

**Capture。** `SweepCapture` 是 exponential sine sweep（唯一接受的方法），含
起訖頻率（終點低於 Nyquist）、長度、fade、靜音、播放位準，以及下列保留資產，
全部使用同一 sample rate：

- 至少兩次原始 sweep 錄音，用來估計重複性並排除不穩定的量測，而不只是取平均；
- deconvolution 使用的 inverse filter；
- 背景噪音錄音；
- deconvolved RIR（已校正、已修正延遲的線性 RIR，也就是 fit 所使用的資料）；
- `latency_correction_samples` 與 `deconvolution_config`。

**房間。** `MeasuredRoom` 有穩定的 `room_id`（同一實體房間不因 session 或位置
不同而改名）、`room_type`、有文件說明的 `coordinate_frame`、
`geometry_uncertainty_m`、provenance，以及 `dimensions_m` 或保留的 `mesh_asset`
其中之一。

**Transducer。** `CalibratedTransducer`（`kind` 為 source 或 receiver）分開保存
製造商、型號、序號、參考軸、校正日期、保留的校正響應與 provenance，確保
transducer response 不會被併入房間參數。

**Record。** `MeasuredRIRRecord` 是一個聲源 pose 被一組接收陣列觀測的結果：
measurement、room、session 與 source id、時間戳、聲源與接收端 pose、capture、
溫度、相對濕度與氣壓（它們決定聲速與空氣吸收），以及 `synchronized_receivers`。
`MeasurementPose` 含位置、yaw/pitch/roll，以及兩者各自的一倍標準差不確定度；
只有距離無法辨識反射幾何或指向性。只有在各通道是同一次激發、sample 同步的
觀測時才能做空間比較；包含不同聲源位置的通道可用於獨立的單聲道指標，但不能
用於 coherence 或 IACC。

## Audit

`audit_measurement_campaign(campaign, root, *, verify_hashes=True,
minimum_records_per_room=12)` 檢查：沒有路徑被宣告兩種 hash、每個資產都存在且
hash 相符、三個 split 都有房間、每個房間的 record 數與相異 source/receiver
組態數都達到下限、每個 capture 都有重複的原始 sweep 以及噪音與 inverse filter
資產，並且至少有一筆同步的多接收端 capture。同一 pose 的重複量測能改善不確定度
估計，但不算相異組態。除非所有檢查都通過（`ready_for_m5_inverse_calibration`），
fit 會拒絕執行（`CampaignNotReadyError`）。

## Split

各 split 的房間互不相交：`validation` 與 `test` 房間不參與擬合，`test` 房間也
不參與模型選擇。在每個 train 房間內，
`deterministic_position_assignments(campaign, holdout_fraction=0.25)` 依
`campaign_id:room_id:measurement_id` 的 SHA-256 排序 record，把前
`round(0.25 n)` 筆（至少一筆、絕不全部）標為 `position_holdout`，其餘為
`position_fit`；其他房間的 record 標為 `room_validation` 或 `room_test`。因此
重跑流程會得到相同的 split（policy `puresound.m5_position_split.sha256.v1`）。
Train 房間至少需要兩筆 record。

## 校正目標

`analyze_rir_calibration_loss(measured, synthetic, sample_rate, *, ...)` 是以
NumPy/SciPy 實作的參考指標，不是 autograd loss：

$$L = \sum_i w_i L_i$$

| 項目 | 檢查內容 | 預設權重 |
|---|---|---|
| `multiresolution_stft` | 早期結構與頻譜細節（FFT 大小 256/512/1024） | 1 |
| `energy_decay` | 對齊直達聲後的衰減曲線形狀 | 1 |
| `arrival_timing` | 直達聲抵達時間 | 1 |
| `octave_acoustics` | 分頻帶的衰減與位準（125–4000 Hz 八度頻帶） | 1 |
| `spatial_coherence` | 每對接收端的晚期複數 coherence | 1 |
| `causality` | 合成結果在物理首次抵達前的能量 | 10 |
| `decay_regularization` | 遞增或不穩定的尾巴 | 1 |

報告（`CalibrationLossReport`）保留每一項、其診斷資訊與權重。通道少於兩個時，
空間項對總和貢獻 0，但會以 `evaluable: false` 加上原因回報；請讀 `evaluable`，
不要把這個 0 當成空間比對成功。

## 擬合

`run_measured_campaign_fit(campaign, root, *, minimum_records_per_room=12,
position_holdout_fraction=0.25, mixing_time_candidates_s=(0.020, 0.024,
0.032), max_order=4, maximum_evaluations=60)`：

1. audit campaign 並固定位置指派；
2. 對每筆 record 的每個接收端，依 `dimensions_m` 建立 shoebox path-event 模型
   （image order 為 `max_order`，排除 edge 與 corner；此 runner 不支援 mesh
   房間）；
3. 對每個 train 房間，在 mixing-time 候選值上擬合 `M4InverseParameters`
   profile（mixing time、coherent reflection gain、500–4000 Hz 有效中心頻率的
   八度 RT60 目標），再以 `fit_grouped_path_gains` 精修六個逐邊界的反射調整量；
4. 評估每個 train 房間的 fit 位置與 holdout 位置；
5. 以各 train 房間參數的中位數評估 `validation` 與 `test` 房間，對應模型用於
   未見房間時的情況。

報告記錄 audit、split、各房間的 fit、母體參數與彙總；只有在 split 保持不相交、
每個 train 房間都有 fit 與 holdout 位置、profile 與 grouped fit 都收斂、所有
holdout 群組都被評估，且每個 loss 都是 finite 時才通過。報告一律回報
`production_enabled: false`。

參數群組對應可觀測的成因（材料、指向性、時間、晚期聲場）。若 campaign 無法把兩個
群組分開辨識，就固定其中一個或收集更好的量測；在擬合真實資料之前，
`analyze_local_identifiability` 與 synthetic-recovery 模組會先在 fixture 上檢查
這件事。空間參數只能從同步陣列擬合。

Residual 修正（`fit_causal_decay_residual`，policy
`puresound.m5_constrained_residual.v1`）只有在物理 fit 穩定後才套用。它從訓練
位置學出一個共用、相對於直達聲的樣板，從直達聲抵達處開始，受 residual 對物理
能量比（預設 0.25）與最大 RT60（預設 0.8 s）限制，並由
`evaluate_residual_ablation` 比較有無 residual 的結果。它不得掩蓋無效幾何、
失敗的 audit 或房間洩漏。

## 執行工具

```bash
python egs/rir_generation/phases/m5_calibration/scripts/fit_m5_measured_campaign.py \
  --campaign campaign.json --asset-root /path/to/assets --output-report report.json
```

選項：`--minimum-records-per-room`、`--position-holdout-fraction`、
`--mixing-time-ms`、`--max-order`、`--maximum-evaluations`。只有報告通過時
exit code 才是 0。可從
`egs/rir_generation/phases/m5_calibration/config/m5_measurement_campaign_template.json`
開始；它是含 placeholder hash 的 schema 範本，不是量測證據。
`egs/rir_generation/phases/m5_calibration/scripts/` 下的 `validate_m5_*.py` scripts
分別檢查各步驟（量測契約、合成與 robust 還原、經 M4 渲染器的反向對應、群組
可辨識性、空間候選 profiling、受限 residual、runner 在合成 campaign 上的執行，以及
彙總的 exit 檢查）；皆支援 `--help`。產生的報告屬於本機實驗輸出，除非 release 證據包明確納入。

測試：`test/rir/test_rir_measurement_campaign.py`、`test_rir_calibration.py`、
`test_rir_inverse_calibration.py`。

## 解讀限制

通過 schema、audit 與合成還原檢查，代表流程的實作前後一致；不代表能推廣到真實
房間。後者仍需要受控量測、房間不相交的評估，以及符合該次 release 的聆聽或下游
任務結果。
