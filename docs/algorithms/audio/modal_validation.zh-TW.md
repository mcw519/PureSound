# 低頻 modal validation

English version: [modal_validation.md](modal_validation.md)

本頁說明低頻阻抗與模態實作如何彼此交叉檢查，並與一個獨立的波動 solver 比對。
內容是驗證方法；通過這些檢查不代表模型已適用於 production 房間。

## 驗證分層

1. 阻抗／反射的代數 round trip；
2. 被動、因果的數位邊界響應；
3. 一維複數腔體模態；
4. 可分離的三維阻抗模態；
5. 獨立的三維 FDTD reference；
6. held-out 的聲源／接收端位置。

各層彼此獨立：scene metadata 從不被拿來當預期的聲學答案，因此任兩層一致就是
對兩者的證據。

## 阻抗基本式

令 $Z_0 = \rho c$：

$$\Gamma = \frac{Z - Z_0}{Z + Z_0},\qquad \alpha = 1 - |\Gamma|^2,\qquad Z = Z_0\,\frac{1+\Gamma}{1-\Gamma}$$

測試（`test/rir/test_acoustic_impedance.py`）涵蓋複數 round trip、相位與
magnitude、matched 與剛性極限，以及拒絕負實部阻抗。吸音率本身無法提供相位，
因此 `SurfaceMaterial` 只有在明確給定 `impedance_real` 與 `impedance_imag`
兩個頻譜時才帶有阻抗；沒有任何程式會由吸音率補上它們。

## 被動時域邊界

Normalized admittance 模型是 positive-real，並以 bilinear transform 離散化
（見[複數聲學阻抗](impedance_measurements.zh-TW.md)）；數位極點必須位於單位圓
內。檢查項目：輸出與狀態皆 finite、嚴格因果、宣告頻帶內的 passivity、
zero-relaxation 極限與實數阻抗邊界一致，以及數位反射符合類比目標。

## FDTD reference

`puresound.audio.rir.physics.wave.fdtd.simulate_fdtd_reference(config, ...)` 是
只為驗證而寫的小型 staggered 壓力／質點速度 solver；它和被檢查的模態遞迴不共用
任何程式碼。

- `FDTDReferenceConfig` 設定房間尺寸、網格間距、時長、$c$、$\rho$、內部 CFL
  （預設 0.92）、聲源／接收端位置，以及 Ricker 聲源（預設中心 180 Hz）。
  `spatial_derivative_order` 為 2 或 4，需搭配相容的 `near_wall_closure` 與
  `boundary_pressure_scheme`；不相容的組合會被拒絕。
- 邊界只能擇一給定：`boundary_absorption`（頻率無關的實數阻抗）或
  `boundary_admittance`（relaxation、multi-pole 或 resonant 模型，每個牆面
  cell、每個 branch 各有獨立狀態）。
- 內部 CFL 條件無法約束在邊與角累積的顯式 admittance 項，因此時間步長另受邊界
  Courant number 限制：$c\,\Delta t \sum_\text{axes} \max(y_-, y_+)/\Delta x \le 0.9$，
  其中使用各模型的全頻 admittance 上界（四階 `face_quadratic_time_quadratic`
  scheme 為 0.25；centered quadratic scheme 沒有這個上限）。兩個上限都會序列化
  到結果中（`interior_cfl_time_step_s`、`boundary_time_step_limit_s`）。
- `simulate_reciprocal_fdtd_reference` 求解交換後的問題，用於互易性檢查。

`physics.wave.low_frequency.estimate_low_frequency_modes(rir, fs, ...)` 從 gate
過的響應讀取模態峰值：zero-padded FFT（倍數 8）、prominence ≥ 6 dB 且間隔
≥ 3 Hz 的峰值，以及 $Q = f_\text{peak}/\Delta f_{-3\,\text{dB}}$。它會記錄 gate
長度決定的原生解析度；zero padding 並不能提升這個解析度。
`egs/rir_generation/compare_modal_acoustics.py TAG=PATH ...` 在每個通道的直達聲
之後套用同一個估計器，比較不同 RIR bank 的模態峰值與 Q 分佈。

## 一維模態

長度 $L$、兩端為相同 locally reacting 邊界的腔體，
`solve_1d_impedance_cavity_modes` 求解

$$1 - \Gamma(s)^2 e^{-2sL/c} = 0,\qquad s = -\gamma + j\omega,\qquad Q = \frac{\omega}{2\gamma}$$

起點是剛性模態 $f = nc/(2L)$，以及該處 $|\Gamma|$ 所隱含的衰減。頻率無關的
實數邊界會和解析解比較；phase-aware 邊界必須相對於 magnitude 相同的實數邊界
產生模態位移。

## 三維模態

牆面上，normalized admittance $y(s)$ 給出 Robin 條件

$$\partial_n p + \frac{s}{c}\,y(s)\,p = 0$$

對長度 $L$ 的單一軸，令 $h_\pm = (s/c)\,y_\pm(s)$，空間波數滿足

$$(h_-h_+ - k^2)\sin(kL) + k(h_- + h_+)\cos(kL) = 0$$

三個軸共用同一個時間極點：

$$k_x^2 + k_y^2 + k_z^2 + (s/c)^2 = 0$$

`solve_rectangular_impedance_modes(room_dim_m, boundary_config, *,
mode_indices, sound_speed_m_s=343.0, continuation_steps=8)` 在
`continuation_steps` 步內把 admittance 從弱逐步加到完整值，追蹤指定的剛性牆
分支，避免跳到無關的根。特徵函數以複數 bilinear 體積 norm 正規化。檢查項目：
還原精確的 1D 情況、均勻立方體預期的簡併、頻率與 Q 對照 FDTD reference，以及
聲源／接收端特徵函數求值。

## Residue 校正

`ImpedanceModalLowFrequencyBackend` 除了極點還需要複數 residue。沒有校正時，
它使用 engineering scale 乘上特徵函數耦合，並在 metadata 中如實標示。有校正時
（`--impedance-residue-calibration`），residue 來自
`fit_fixed_pole_modal_residues`：固定已解出的極點與特徵函數，只對 FDTD 響應
擬合一個複數 scale 與頻率冪律 $(f_\text{ref}/f)^p$（預設
$f_\text{ref} = 100$ Hz）：

- 訓練案例先以 RMS 正規化，避免大聲的位置主導；holdout 位置從不參與擬合；
- 接受條件為 holdout 平均相關 ≥ 0.9 且 NRMSE ≤ 0.35；
- 擬合出的 residue 由 FDTD pressure-cell 聲源慣例轉換成自由場 $1/r$ RIR 慣例
  （`convert_pressure_state_modal_residue`，係數 $4\pi c^2/(f_s s)$）；
- 結果以 schema `puresound.impedance_modal_residue_calibration.v2` 儲存（v1
  檔案仍可讀入）；
- 所要求的頻帶若超出邊界設定或校正檔的 `valid_frequency_range_hz`，backend
  會拒絕執行。

Metadata 會區分 FDTD 校正的 residue（`modal_residue_fdtd_validated: true`）與
engineering fallback，且一律回報 `modal_residue_production_validated: false`：
production 驗證需要實測房間的轉移函數，合成 FDTD fit 得好並不能給予這個資格。

## Evidence tiers

| Tier | 意義 |
|---|---|
| direct complex measurement | 阻抗實部與虛部皆為實測 |
| measured property plus model | 實測性質輸入具名物理模型 |
| engineering prior | 為模擬或診斷選定的參數 |
| synthetic reference | 以獨立 solver 檢查某個實作 |

由實測 flow resistivity 建立的多孔模型，相位仍是模型推得的（見
[阻抗 priors](impedance_priors.zh-TW.md)）。公開的 grazing-duct liner 資料保留
原始幾何，不會被改標成正入射牆面資料（見
[公開資料來源準則](complex_impedance_source_audit.zh-TW.md)）。

## 執行檢查

實作：`puresound/audio/rir/physics/impedance/`、
`puresound/audio/rir/physics/wave/`、`puresound/audio/rir/render/low_frequency/`。

測試：`test/rir/test_fdtd_reference.py`、`test_impedance_modes.py`、
`test_impedance_residues.py`、`test_acoustic_impedance.py`。

Validator 與校正工具位於 `egs/rir_generation/phases/m2_impedance/scripts/`
（例如 `calibrate_impedance_modal_residues.py --boundary-config <json>
--output-calibration <json> --output-report <json>`）與
`egs/rir_generation/phases/m3_wave_path/scripts/`（FDTD 邊界與高階檢查）；各自
支援 `--help`。產生的報告屬於本機實驗輸出，除非 release 流程明確要求保留。

Production 邊界模型還需要可追溯的真實房間證據、held-out 房間與位置、明確宣告的
適用範圍，以及經 review 的材料 catalog 映射。
