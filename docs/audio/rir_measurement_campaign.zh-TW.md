# M5 受控房間 RIR 量測與反演校準契約

English version: `rir_measurement_campaign.md`

狀態：M5.1–M5.6 implementation 已完成；受控 measured／listening／downstream empirical exit 尚未就緒
Schema：`puresound.rir_measurement_campaign.v1`
Loss policy：`puresound.rir_calibration_loss.v1`

## 1. 我們正在解什麼問題

M1–M4 已經能從房間幾何、材料、環境、direct/early paths 與 multiband
late field 合成可用 RIR。M5 的問題不再只是「能不能產生殘響」，而是：

> 給定少量但完整可追溯的真實房間量測，能否反推出 renderer 的物理參數，
> 並讓未量測位置、甚至未見過的房間也比未校準模型更接近真實？

形式上，量測 RIR 記為 (h^{\mathrm{meas}}_{r,s,m}(t))，其中 (r) 是實體
房間、(s) 是 source pose、(m) 是 receiver。Renderer 為

\[
\hat h(t)=F(\theta_{\mathrm{room}},\theta_{\mathrm{mat}},
\theta_{\mathrm{dir}},\theta_{\mathrm{late}};
g,s,m,e),
\]

其中 (g) 是幾何，(e) 是溫度、濕度與氣壓。M5 要找的是參數

\[
\theta^*=\arg\min_\theta
\sum_{(s,m)\in\mathcal T_r}
\mathcal L\!\left(h^{\mathrm{meas}}_{r,s,m},F(\theta)\right),
\]

然後只在沒有參與 fitting 的位置與房間上評估。若量測把喇叭、麥克風、
幾何誤差、時鐘延遲與房間響應混在一起，最佳化即使收斂，也可能只是把
裝置誤差錯認為牆面材料。因此 M5.1 先凍結「什麼資料才有資格進反演」。

## 2. 為何使用 repeated exponential sine sweep

每個 source／receiver configuration 至少錄兩次 exponential sine sweep
（ESS）。ESS 的瞬時頻率以指數方式由 (f_1) 掃到 (f_2)：

\[
x(t)=\sin\left[
2\pi f_1\frac{T}{\ln(f_2/f_1)}
\left(e^{t\ln(f_2/f_1)/T}-1\right)
\right].
\]

錄音 (y(t)) 與對應 inverse filter (x^{-1}(t)) 做反卷積：

\[
\tilde h(t)=y(t)*x^{-1}(t).
\]

這條流程的價值是頻帶能量可控，而且裝置的 harmonic distortion 在反卷積
後會與線性 impulse response 分離。方法源頭可參考 Farina 的
[AES ESS 論文](https://angelofarina.it/Public/Papers/134-AES00.PDF)。本專案不只
保存最後的 `rir.wav`，還強制保存：

- 至少兩個 raw sweep recordings；
- 播放 sweep 的 inverse filter；
- 同一 configuration 的 background-noise recording；
- deconvolution window、harmonic separation、latency correction 等設定；
- 最終 deconvolved RIR。

重複量測不是為了把兩條 WAV 平均掉而已。它讓我們估計 repeatability、
噪聲底、時變干擾與 clipping，並能拒絕「單次錄音恰好看起來合理」的資料。

## 3. 必須保存的物理證據

### 3.1 實體房間與座標系

每個 `MeasuredRoom` 必須有穩定 `room_id`、房間類型、右手座標系說明、
geometry uncertainty，以及下列至少一項：

- 可量測的長寬高；或
- 有 SHA-256 的 coarse／detailed mesh asset。

同一物理房間不得因換了一組 source position 就被重新命名，否則
room-disjoint test 會洩漏。

### 3.2 Source 與 receiver

`CalibratedTransducer` 分開記錄 source／receiver 的 manufacturer、model、
serial number、reference axis、校正日期、校正響應 asset 與 provenance。
房間傳遞函數和 transducer response 不可以只靠一條合成 EQ 一起吸收掉。

每個 pose 都包含：

- `position_m = [x,y,z]`；
- `orientation_ypr_deg = [yaw,pitch,roll]`；
- position／orientation 的一個標準差不確定度。

只有距離、沒有絕對位置或方向，不足以擬合牆面反射、directivity 或
spatial coherence。

### 3.3 環境與同步

每筆 capture 保存 temperature、relative humidity 與 pressure，因為它們會
改變 sound speed 與空氣吸收。多 receiver capture 必須 sample-synchronized；
每個 WAV channel 必須代表同一 source excitation 到不同 receiver，而不是
把多個 source positions 包成 channel。後者仍可比較 mono acoustic metrics，
但不能拿來估 inter-channel coherence 或 IACC。

### 3.4 不可變 asset 身分

每個 retained asset 使用 campaign-relative safe path 與 SHA-256。Audit 會
檢查檔案存在、hash 相符，而且同一路徑不能出現互相衝突的 hash。這使
「metadata 沒變但 WAV 被重做」成為可偵測的錯誤。

## 4. 資料切分不是 item-disjoint，而是 room-disjoint

Schema 只允許 `train`、`validation`、`test` 三種 room split，而且每個
`room_id` 只能被指派一次。對一個 train room，仍需保留未參與 fitting 的
source／receiver positions，形成 position holdout；test room 則整間完全不
參與 parameter fitting、loss weighting 或 early stopping。

最少資料量先設為每個代表房間 12 筆，建議 12–30 個 configurations。這不是
聲學上的神奇常數，而是第一個 operational gate：資料太少時，多組材料、
scattering、directivity 與 late-field parameters 常能產生近似 RIR，反問題
不可識別。M5.2 會先以 synthetic recovery 實際測哪些參數能被找回，再決定
正式 campaign 是否要增加位置、方向或 receiver spacing。

## 5. 多目標 calibration loss

單一 waveform L1/L2 會被極小時間偏移支配；只比 RT60 又會忽略 early
reflection、頻率 coloration 與空間結構。因此 reference loss 為：

\[
\mathcal L =
w_{\mathrm{stft}}L_{\mathrm{stft}}+
w_{\mathrm{edc}}L_{\mathrm{edc}}+
w_{\mathrm{arr}}L_{\mathrm{arr}}+
w_{\mathrm{oct}}L_{\mathrm{oct}}+
w_{\mathrm{sp}}L_{\mathrm{sp}}+
w_{\mathrm{causal}}R_{\mathrm{causal}}+
w_{\mathrm{decay}}R_{\mathrm{decay}}.
\]

每一項都單獨輸出，不能只留下 total scalar。

### 5.1 Multiresolution STFT

在多個 FFT size 比較 spectral convergence 與 log-magnitude L1：

\[
L_{\mathrm{SC}}=
\frac{\lVert|S|-|M|\rVert_F}{\lVert|M|\rVert_F},
\qquad
L_{\log}=\operatorname{mean}
\left|\log(|S|+\epsilon)-\log(|M|+\epsilon)\right|.
\]

短窗對 early/transient 較敏感，長窗提供較細頻率解析度。這種多解析度
spectral objective 也常見於 neural waveform generator；可參考
[Parallel WaveGAN](https://arxiv.org/abs/1910.11480)。目前 NumPy/SciPy 版本是
metric oracle，不是 autograd training loss。

### 5.2 Direct-relative energy-decay curve

先分別找出 measured／synthetic direct arrival，從 direct sample 對齊後做
Schroeder backward integration：

\[
E(t)=\sum_{\tau=t}^{T}h^2(\tau),\qquad
D(t)=10\log_{10}\frac{E(t)}{E(0)}.
\]

比較的是 decay curve RMSE，而不是只比一個線性斜率。因此雙斜率衰減、
過強 early energy 或尾端 flattening 都不容易被單一 RT60 隱藏。

### 5.3 Arrival timing

每個 channel 比較 direct-arrival sample，換算成毫秒，並以明確 tolerance
正規化。這一項保持絕對飛行時間；EDC 與 octave metrics 的 direct-relative
對齊不會把幾何 timing error 消掉。

### 5.4 Octave acoustic metrics

在有效 125 Hz–4 kHz octave bands 比較 direct-relative band energy 與
qualified T20。沒有足夠 decay range 的 T20 必須記成 unqualified，不能以
零或預設 RT60 代替。

### 5.5 Spatial coherence

只有同步多 receiver capture 才可評估。在 late window 對每個 receiver pair
比較 normalized complex cross spectrum：

\[
\Gamma_{ij}(f)=
\frac{S_{ij}(f)}{\sqrt{S_{ii}(f)S_{jj}(f)}}.
\]

Mono 或 source-channel bank 會回報 `evaluable=false` 與原因，而不是得到
看似完美的 spatial loss 0。

### 5.6 物理 regularization

`causality` 懲罰由幾何／measured physical-arrival bound 以前的 synthetic
energy fraction。`decay_regularization` 則量 late-window energy 是否持續
反常增長。兩者的功能是排除「metric 變近，但物理上創造 pre-echo 或不穩定
尾場」的解。

## 6. M5.1 已完成的實作

（本節原為結果紀錄，已遷至 [`RIR_EXP_LOG.md`](../../RIR_EXP_LOG.md) 附錄。）

## 9. 重現 M5.1–M5.6 implementation

```bash
PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measurement_contract.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_synthetic_recovery.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_robust_recovery.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_m4_parameter_mapping.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_group_identifiability.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measured_runner.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_spatial_calibration.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_constrained_residual.py

PYTHONPATH=. .venv/bin/python \
  egs/rir_generation/phases/m5_calibration/scripts/validate_m5_exit.py

.venv/bin/pytest -q \
  test/test_rir_measurement_campaign.py \
  test/test_rir_calibration.py \
  test/test_rir_inverse_calibration.py \
  test/test_rir_m4_inverse_calibration.py \
  test/test_rir_m5_pipeline.py \
  test/test_rir_measured_calibration.py \
  test/test_rir_constrained_residual.py \
  test/test_m5_measurement_contract_validator.py \
  test/test_m5_synthetic_recovery_validator.py \
  test/test_m5_robust_recovery_validator.py \
  test/test_m5_m4_parameter_mapping_validator.py \
  test/test_m5_completion_validators.py
```

預期顯示：

```text
M5.1 implementation exit: PASS
controlled measurement readiness: OPEN
M5.2 synthetic recovery: PASS
M5.2b robust synthetic recovery: PASS
M5.2c actual M4 parameter mapping: PASS
M5.2d grouped material/path identifiability: PASS
M5.3 runner implementation: PASS
M5.3 measured inverse fit: BLOCKED ON CONTROLLED CAMPAIGN
M5.4 synchronized spatial calibration implementation: PASS
M5.5 constrained residual implementation: PASS
M5 implementation exit: PASS
M5 empirical/production exit: OPEN
```

兩者並不矛盾：前者代表程式、schema 與 loss 已準備好；後者代表尚未取得
符合契約的真實 campaign，因此 M5.3 measured-room fit 還不能宣稱完成。
