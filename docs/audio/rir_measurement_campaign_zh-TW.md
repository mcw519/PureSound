# M5 受控房間 RIR 量測與反演校準契約

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

主要入口：

- `puresound/audio/rir_measurement_campaign.py`：schema、strict JSON、hash
  與 campaign audit；
- `puresound/audio/rir_calibration.py`：七項 reference loss 與 term-level
  diagnostics；
- `egs/rir_generation/phases/m5_calibration/scripts/validate_m5_measurement_contract.py`：deterministic
  loss probes、template 與 legacy bank readiness audit；
- `egs/rir_generation/phases/m5_calibration/config/m5_measurement_campaign_template.json`：只有結構
  範例，不是實測證據；
- `egs/rir_generation/phases/m5_calibration/reports/m5_measurement_contract_report.json`：正式
  M5.1 report。

驗證 probe 確認：identity fidelity terms 為零、8-sample delay 可被偵測、
pre-arrival energy 會受罰、spatial perturbation 可被偵測，而且 mono spatial
term 明確不可評估。針對 schema、hash corruption、同步語意與 loss 的 10 個
測試全數通過。

## 7. 現有 measured bank 的真實狀態

Validator 分別抽查既有 train 與 held-out view 各 64 個 metadata。兩者目前
只保存 `channel_map`、distance、`origin=real` 與 RT60 類資訊；以下九類 M5
證據在抽樣中都是 0/64：

1. 穩定實體 room identity；
2. geometry／mesh；
3. source position 加 orientation；
4. receiver position 加 orientation；
5. source／receiver response calibration；
6. temperature／humidity／pressure；
7. repeated raw ESS；
8. deconvolution inverse／config／noise；
9. synchronized receiver channel semantics。

結論不是「舊資料沒用」。它仍可做 DRR、C50、decay、頻譜與 echo-density
distribution reference；但缺失的 acquisition evidence 無法從最後 RIR WAV
逆推出來，所以不能被升格成 M5 controlled inverse-calibration dataset。

## 8. M5.2 synthetic recovery 已完成什麼

在等待／規劃受控量測時，M5.2 先完成第一個 synthetic recovery baseline：

1. 從已知 scene 生成 target RIR；
2. 隱藏 material、mixing-time、late-decay 等一小組參數；
3. 從錯誤初值開始最小化同一 loss；
4. 在未參與 fitting 的 source／receiver positions 比較 recovery；
5. 用 multi-start 與 scaled Jacobian singular values 檢查局部不可識別方向；
6. 只有能在 synthetic holdout 穩定找回的參數，才進 M5.3 measured fit。

實作 `puresound.rir_synthetic_recovery.v1` 是一個刻意簡化、保持因果的
approximate renderer。Geometry 與 direct response 視為已知，只開放十個
shared-room parameters：

- 一個 mixing time；
- 一個 coherent early-reflection gain；
- 500／1000／2000／4000 Hz 四個 RT60；
- 同四個 octave bands 的 late gain。

Direct component 永遠不被 crossfade；early 與 late 使用連續 equal-power
transition。每個 late band 的 pressure envelope 為

\[
a_b(t)=10^{-3t/T_{60,b}},
\]

所有 mixing time、gain 與 RT60 都受明確 box bounds 限制。三個 train
positions 共用同一組房間參數，但各有不同 distance、early path fixture 與
deterministic band-limited late excitation，因此 optimizer 必須找出可跨位置
解釋資料的參數。

正式 validator 從三組差異很大的初值做 bounded nonlinear least squares。
三次都回到同一組隱藏 ground truth，最大 parameter spread 為
`2.35e-12`；scaled Jacobian condition number 是 `17.22`，且具有完整 local
column rank。獨立 M5.1 oracle 在兩個 unseen positions 的 mean loss 由
`2.26933` 降到 `1.69e-14`，所有輸出保持 direct arrival 前嚴格為零。

這個結果是必要但很弱的第一關，屬於 **inverse-crime baseline**：target 與
fitter 使用同一個 noise-free model family，所以精確 recovery 是預期結果。
它證明的是 parameter serialization／bounds、multi-position objective、
multi-start、local sensitivity、holdout 與 M5.1 oracle 接線都正確；它沒有
證明：

- measurement noise 或 clock drift 下仍穩定；
- approximate renderer 能吸收完整 M4 renderer 的 model mismatch；
- 真實材料、scattering 或 directivity 已被找回；
- global identifiability；
- measured-room fit 已完成。

報告為 `egs/rir_generation/phases/m5_calibration/reports/m5_synthetic_recovery_report.json`；同一
holdout position 的 target／錯誤初值／recovered WAV 位於
`egs/rir_generation/exp/rir_realism/m5/rir_m5_synthetic_recovery/`。下一個可平行開發切片是加入 controlled
noise/model mismatch；下一節已完成這個 robust synthetic gate。真正的
M5.3 仍必須等待符合前述契約的 controlled campaign。

### 8.1 M5.2b：噪聲、已知 nuisance 與未知 model mismatch

M5.2b 不再讓 fitter 看到乾淨 target。每個 train／holdout position 都加入：

- 32–38 dB SNR 的 broadband acquisition noise；
- -0.8 至 +1.2 dB 的 per-position gain calibration error；
- -6 至 +11 samples 的 deconvolution latency offset；
- nominal model 沒有的 early taps；
- 一個獨立 stochastic late component，其 RT60 為 nominal band 的 1.25 倍。

系統同時保存 `raw_rir` 和 `corrected_rir`。只有 campaign metadata 已知的 gain
與 latency 會被移除；noise、extra paths 和 secondary decay 刻意留在 fitting
target。這是在測 robust estimation，不是用 ground truth 把所有誤差清乾淨。

新的 `puresound.rir_robust_recovery_objective.v1` 使用四組 smooth residual：

\[
r(\theta)=
\left[
\sqrt{w_w}r_{\mathrm{wave}},
\sqrt{w_e}r_{\mathrm{early}},
\sqrt{w_b}r_{\mathrm{broadband\ decay}},
\sqrt{w_o}r_{\mathrm{octave\ decay}}
\right].
\]

Decay residual 以 8 ms window 的 log energy 計算；只有高於估計 noise energy
20 dB 的 windows 參與 fitting。這個門檻非常重要：若把已進入 noise floor 的
高頻尾端也當成房間衰減，optimizer 會把 noise plateau 解讀成較長 RT60。

它稱為 M4-consistent proxy，是因為 direct／coherent early／broadband late／
octave late 的分工與 M4 相同；但目前仍以 SciPy finite-difference least
squares 最佳化 surrogate，不是 autograd，也尚未對完整 PathEvent＋FDN
renderer 求導。

正式結果：

- mixing time absolute error：`0.0235 ms`；
- early gain error：`0.0214 dB`；
- 最大 octave RT60 relative error：`1.43%`；
- 最大 late gain error：`0.106 dB`；
- 兩個遠距初值的最大 parameter spread：`1.03e-6`；
- scaled-Jacobian condition number：`8.12`，local full rank；
- held-out M5.1 total：相對初始值下降 `61.4%`；
- held-out octave error：robust objective `0.00846`，waveform-only ablation
  `0.03341`。

另外，五個擾動案例中有兩個的 global absolute peak 並不是 direct arrival。
這不是小細節：高 DRR 以外的 RIR、未建模反射或 noise spike 都可能比 direct
大。正式量測應以 geometry (d/c) 建立 arrival search window，或使用另行
驗證的 onset detector；不能直接 `argmax(abs(rir))`。

M5.2b 的 15/15 gates 全數通過，報告位於
`egs/rir_generation/phases/m5_calibration/reports/m5_robust_recovery_report.json`，六個 holdout
RIR artifacts 位於 `egs/rir_generation/exp/rir_realism/m5/rir_m5_robust_recovery/`。它仍不是 measured-room fit；
下一節已把第一組參數映射到 actual M4 renderer。

### 8.2 M5.2c：actual PathEvent＋multiband FDN parameter profile

M5.2c 不再用 smooth surrogate 產生 candidate，而是直接通過 M4 的
`PathEvent -> equal-power transition -> multiband FDN`。由於 mixing time 會
離散改變 prime delay topology，演算法以 `20/24/28 ms` 做 outer profile；每個
profile 內再以 bounded least squares 估 coherent-reflection aggregate gain 和
500／1000／2000 Hz RT60。

Target 保留 8% alternate-FDN-seed mismatch 與 42 dB SNR noise。兩個 order-4
PathEvent positions 用於 fitting，另兩個 positions 只做 holdout。14/14 gates
通過：正確選到 hidden `24 ms`、best／second cost ratio `0.0820`、best
Jacobian condition number `3.14`、coherent gain error `0.661 dB`、最大 RT60
error `4.23%`，held-out M5.1 total 下降 `71.7%`。所有 candidate 維持
physical-arrival causality，M4 transition 前樣本完全不變，Pyroomacoustics
production default 也沒有改動。

報告位於
`egs/rir_generation/phases/m5_calibration/reports/m5_m4_parameter_mapping_report.json`，三個 holdout
RIR artifacts 位於 `egs/rir_generation/exp/rir_realism/m5/rir_m5_m4_parameter_mapping/`。這只識別 aggregate
coherent gain，不代表已從 RIR 分離出單一牆面的 absorption／scattering；
也不是 measured-room fit。下一個 M5.2d 是逐 material／path group 的
identifiability ablation。

### 8.3 M5.2d：哪些 material/path groups 真的可辨識

每條 PathEvent 都保留撞到的 surface sequence。M5.2d 對每次 boundary hit
施加一個 effective pressure adjustment，因此同一參數會一致影響所有包含該
牆面的高階路徑。三個 order-4 train positions 可找回 west／east／south／
north／floor／ceiling 六組：condition number `2.74`、最大 gain error
`0.00123 dB`、held-out M5.1 total 下降 `64.8%`。

但若把每面牆同時開放 `absorption loss` 和 `specular scattering loss`，兩者在
mono coherent amplitude 上的 Jacobian columns 完全相同；rank 由應有的 12
只有 6。演算法因此保留六個 `effective_reflection`，拒絕六個 scattering
duplicates。這不是最佳化失敗，而是資料本身沒有足夠觀測；scattering 必須等
M5.4 的 synchronized receiver evidence。

### 8.4 M5.3：runner 已完成，但不合格資料不會開始 fitting

`fit_m5_measured_campaign.py` 的執行順序是：

1. 驗全部 retained assets 與 SHA-256；
2. 確認每個 room 的 repeated ESS、noise、inverse、calibration、geometry、
   environment 與 synchronized channel semantics；
3. 以 `campaign_id + room_id + measurement_id` 的 SHA-256 固定選出
   train-room position holdout；
4. 只在 `position_fit` 估每個 train room 的 M4 topology、RT60 與六面
   effective reflection；
5. 分開回報 train-position、position-holdout、validation-room 與 test-room；
6. 未見房間只用 train-room population median parameters，不偷 fit test room。

完整 synthetic campaign fixture 已走通 runner 的所有階段與 8/8 gates，而且
metadata 明確標成 `qualifies_as_measured_evidence=false`。現有 template／legacy
bank 則在 readiness 階段退出，M4 optimizer 完全不會被呼叫。正式狀態報告為
`m5_measured_fit_status_report.json`：M5.3 runner implementation PASS，真實
measured fit 仍 BLOCKED。

目前 reference runner 支援有 `dimensions_m` 的 shoebox campaign。只有 mesh 的
campaign 必須先註冊能產生 PathEvent 的 mesh backend，不能把 mesh 悄悄縮成
shoebox。

### 8.5 M5.4：同步 receiver 才能校正 spatial groups

`select_spatial_calibration_candidate` 至少要求兩個同步 channel。它用同一份
M5.1 loss 比較 actual M4 的 scattering、receiver directivity 與 late-field
candidates；mono 直接拋錯，不會得到假的 spatial zero loss。四候選 fixture
正確選回 hidden scattering＋opposed-cardioid 組合，11/11 gates 通過。

證據邊界也要保留：現行 M4 scattering 只分配 first-order coherent PathEvent
energy，80 ms 後的 FDN field 尚未依 scattering 改變。因此 scattering 是由
synchronized early／spectral／octave total 選中；late spatial coherence 主要
驗 shared field 與 directivity，不能把相同的 late term 說成 scattering 證據。

### 8.6 M5.5：learn residual，不重學 basic physics

新的 residual model 先算 `target - physical`，再把每條 residual 對齊自己的
direct arrival，除以 physical tail norm，從多個 train rooms 取 robust median。
它受到三個硬限制：

- direct arrival 前永遠為零；
- 50 ms 後每 10 ms block 不得比最大 `RT60=0.8 s` 的 pressure decay 更慢；
- normalized residual energy 不得超過 physical energy 的 `0.15`。

M5.5 同時輸出 physical-only、residual-only、combined，避免只報最好的一條。
在完全未參與 fitting 的第三個 synthetic room，combined total 相對
physical-only 下降 `67.9%`，residual-only 明顯較差；causality、decay、energy
budget 與 parameter interpolation 共 12/12 gates 通過。這證明 residual
contract／ablation plumbing 可用，尚不代表已在真實房間訓練 neural model。

### 8.7 M5.6：完成的是 implementation，不是捏造 empirical PASS

`validate_m5_exit.py` 彙整 M5.1、M5.2／2b／2c／2d、M5.3 runner、M5.4、
M5.5、必要 WAV/campaign artifacts 與禁止 false claim 的 invariants。結果為：

- **M5 implementation exit：PASS**；
- **M5 empirical／production exit：OPEN**；
- production enablement：`ready=false`，Pyroomacoustics default 不變。

empirical exit 還缺：合格 repeated-ESS campaign、measured position holdout、
measured physical-room holdout、measured synchronized spatial calibration、
measured residual training、controlled listening 與 room-disjoint downstream
task。這些是必須真的取得／執行的外部證據，不能由 synthetic fixture 生成。

近期 inverse-acoustic rendering 研究也採用 differentiable rendering 與稀疏
觀測來估 room parameters，例如
[AV-DAR](https://openaccess.thecvf.com/content/ICCV2025/html/Jin_Differentiable_Room_Acoustic_Rendering_with_Multi-View_Vision_Priors_ICCV_2025_paper.html)
與 [DiffRIR / Hearing Anything Anywhere](https://masonlwang.com/hearinganythinganywhere/)。
PureSound 的策略更保守：先用現有可稽核物理 renderer 做 recovery baseline，
確認 identifiability，最後才加入 learned residual。

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
