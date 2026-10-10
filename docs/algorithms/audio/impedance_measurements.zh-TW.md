# 複數聲學阻抗

English version: [impedance_measurements.md](impedance_measurements.md)

這條路徑把保留相位的阻抗量測轉成被動、因果的邊界模型，供 path-event
renderer、impedance modal solver 與 FDTD reference 使用。它屬於研究與驗證
工具：這裡沒有任何東西會寫入 scene 材料 catalog。

| 模組 | 內容 |
|---|---|
| `physics/impedance/admittance.py` | 反射／阻抗換算、被動 admittance 模型、bilinear 離散化 |
| `physics/impedance/measurements.py` | SI 與 normalized 量測契約 |
| `physics/impedance/fitting.py` | 被動 multi-pole 與 resonant fitting |
| `physics/impedance/modes.py` | 1D 與可分離 3D 阻抗模態、六面牆邊界設定 |

（皆位於 `puresound/audio/rir/` 之下）

## 為何需要複數阻抗

對空氣 characteristic impedance $Z_0 = \rho c$ 中、阻抗為 $Z(f)$ 的
locally reacting 表面：

$$\Gamma(f) = \frac{Z(f) - Z_0}{Z(f) + Z_0},\qquad \alpha(f) = 1 - |\Gamma(f)|^2,\qquad Z = Z_0\,\frac{1+\Gamma}{1-\Gamma}$$

吸音率只保留 $|\Gamma|$。它丟掉的相位決定反射延遲、模態頻率位移與模態 Q，
因此本路徑要求阻抗的實部與虛部，絕不從擴散場吸音率重建相位。
`impedance_from_absorption_and_phase()` 正是因此把相位列為明確參數。被動性即
$\operatorname{Re} Z \ge 0$，也就是 $|\Gamma| \le 1$；
`normal_incidence_reflection_coefficient()` 會拒絕負實部，並把無限大阻抗映到
剛性 $\Gamma = 1$。

材料名稱相同不代表厚度、背板、air gap、安裝方式或入射條件相同；下面的契約
會把這些全部記錄下來。

## SI 量測契約

`ComplexImpedanceMeasurement.from_csv_and_metadata(csv, json)` 讀取 schema
`puresound.complex_impedance_measurement.v1`，只支援正入射。

CSV 欄位（Pa·s/m）；可選的不確定度欄位 `impedance_real_std_pa_s_m` 與
`impedance_imag_std_pa_s_m` 必須成對出現：

```csv
frequency_hz,impedance_real_pa_s_m,impedance_imag_pa_s_m
60.0,421.2,-880.4
80.0,398.1,-631.7
120.0,376.5,-402.8
```

最小 sidecar：

```json
{
  "schema_version": "puresound.complex_impedance_measurement.v1",
  "measurement_id": "lab_sample_001",
  "method": "ISO 10534-2 two-microphone transfer function",
  "incidence": "normal",
  "environment": {"air_density_kg_m3": 1.204, "sound_speed_m_s": 343.0},
  "sample": {"material_label": "100 mm glass wool", "thickness_m": 0.1,
             "backing": "rigid", "air_gap_m": 0.0},
  "provenance": {"source_url": "https://example.org/measurement/001",
                 "license": "CC-BY-4.0"}
}
```

Loader 會拒絕：其他 schema 或入射條件；少於三點；非正或非嚴格遞增的頻率；
non-finite 值；負的實部阻抗；不成對或為負的不確定度；缺少
`sample.material_label`、`provenance.source_url` 或 `provenance.license`；
以及任何反射 magnitude 大於一的點。實際量測還應記錄管子幾何、麥克風間距、
樣品批次與公差、溫度、濕度、校正與重複性。
[阻抗管量測流程](impedance_tube_protocol.zh-TW.md) 的輸出就是這個格式。

ISO 10534-2 的正入射阻抗管資料不能與混響室擴散入射吸音率互換。

## Normalized 量測契約

有些文獻只提供 $z = Z/(\rho c)$，沒有保留當時用來 normalize 的 $\rho$ 與 $c$。
若用假設的大氣條件乘回去，等於把假設冒充成量測環境，因此
`NormalizedComplexImpedanceMeasurement` 以 schema
`puresound.normalized_complex_impedance_measurement.v1` 保留無因次值：

```csv
frequency_hz,normalized_impedance_real,normalized_impedance_imag
500,0.785,-3.596
600,0.474,-3.213
```

（可選 `normalized_impedance_real_std` / `normalized_impedance_imag_std`）。
Sidecar 必須提供 `acoustic_field_geometry`（`normal_incidence_tube` 或
`grazing_duct`）、`phasor_convention`（`exp(+i*omega*t)`）、
`conditions.mean_flow_mach`（≥ 0）、`conditions.source_spl_db`（> 0）、
`sample.material_label`、`provenance.source_url`、`provenance.license` 與
`applicability.scope`。

它的有界 fitting 座標是 Cayley transform：

$$\mathcal{C}(z) = \frac{z - 1}{z + 1}$$

只有在 locally reacting 正入射的解讀下，它才等於正入射壓力反射係數。對
grazing-duct 資料它只是有界的數值座標；這類資料絕不會被改標成阻抗管量測。

## 被動邊界模型

時域 solver 需要因果模型，而不是取樣點。所有模型都寫成無因次 admittance
$y(s) = \rho c / Z(s)$，正入射時 $\Gamma = (1 - y)/(1 + y)$。

**Multi-pole relaxation**（`PassiveMultiPoleAdmittance`）：

$$y(s) = g_s + \sum_k \frac{g_{L,k}}{1 + s\tau_k} + \sum_k g_{H,k}\,\frac{s\tau_k}{1 + s\tau_k},\qquad \tau_k = \frac{1}{2\pi f_{p,k}}$$

每個 gain 都非負、每個 branch 都是 positive-real，所以 parallel sum 在結構上
就是被動的；不存在事後的 passivity 修補。`FirstOrderRelaxationAdmittance` 是
單極點特例，由[阻抗 priors](impedance_priors.zh-TW.md) 使用。

**Resonant**（`PassiveResonantAdmittance`），series-RLC branch 的 parallel
sum，令 $u = s/\omega_0$：

$$y_k(s) = \frac{g_{\mathrm{peak},k}}{Q_k}\,\frac{u}{u^2 + u/Q_k + 1}$$

共軛極點對可以表示實數 relaxation 極點做不到的 Helmholtz 式共振。只有在實測
reactance 顯示共振時才使用。

**離散化。** `digital_normalized_admittance_filter()` 以 bilinear transform
映射 relaxation section，以各自 prewarp 的 bilinear biquad 映射 resonant
branch；parallel sum 以 $z^{-1}$ 多項式精確合併。Bilinear transform 保持
positive-real，所以實現角度相依反射

$$\Gamma_\theta = \frac{\cos\theta - y}{\cos\theta + y}$$

的 `digital_locally_reacting_reflection_filter(model, cos_theta, fs)` 是
bounded-real（單位圓上 $|\Gamma_\theta| \le 1$）。結果是
`DigitalBoundaryReflectionFilter`（schema
`puresound.digital_boundary_reflection_filter.v1`）。

## Fitting

| 函式 | 模型 | 預設 |
|---|---|---|
| `fit_passive_multi_pole_admittance(f, Z, *, air_density_kg_m3, sound_speed_m_s, pole_frequencies_hz, reflection_weights=None, acceptance_threshold=0.05)` | multi-pole，固定實數極點 | gain 限制在 [0, 100]；三個起點 |
| `fit_complex_impedance_measurement(measurement, *, num_poles=3, pole_frequencies_hz=None, acceptance_threshold=0.05)` | multi-pole | 極點在 $[0.5 f_\min, 2 f_\max]$ 上 log 間隔（單極點：$\sqrt{f_\min f_\max}$） |
| `fit_passive_single_resonance_admittance(f, z, *, training_frequency_indices=None, acceptance_threshold=0.1)` | 一個 RLC branch + static conductance，極點對自由 | 未參與訓練的 index 報 held-out 指標 |
| `fit_normalized_complex_impedance_measurement(measurement, *, alternating_frequency_holdout=True, acceptance_threshold=0.1)` | resonant | 偶數 bin 訓練、奇數 bin held out |

所有 fitting 都最小化堆疊後的反射實部與虛部誤差，並回報 RMS 與最大複數誤差、
最大 magnitude 與 phase 誤差；`accepted` 以最大複數誤差和門檻比較。量測帶有
不確定度時，`fit_complex_impedance_measurement` 會把它傳遞到反射域：

$$\sigma_\Gamma = \left|\frac{\partial\Gamma}{\partial Z}\right|\sqrt{\tfrac12(\sigma_R^2 + \sigma_I^2)},\qquad \frac{\partial\Gamma}{\partial Z} = \frac{2Z_0}{(Z + Z_0)^2}$$

並以 $1/\max(\sigma_\Gamma, \text{floor})$ 加權每個點，其中
floor $= \max(10^{-6}, 0.1\cdot\operatorname{median}\sigma_\Gamma)$。

固定極點是刻意的保守選擇：避免不穩定的極點重定位步驟。fit 沒過門檻時，
調整模型階數、極點位置或頻帶；絕不拿掉 passivity 限制。

## 模型的使用位置

- **Path events** —— 給定 `surface_admittance_models` 時，
  `render_path_events` 會以 `digital_locally_reacting_reflection_filter`
  （依表面模型與入射角快取）實現反射或散射路徑上的每一次邊界碰撞。被濾波路徑
  上的每個表面都需要模型，且 event 儲存的 gain spectrum 必須在容許誤差內與
  模型一致。
- **Impedance modal backend** —— `RectangularImpedanceBoundaryConfig`
  （schema `puresound.rectangular_impedance_boundary.v1`）為 shoebox 六面牆
  各指定一個被動模型，並要求 `source.evidence_tier` 與 `applicability.scope`。
  `ImpedanceModalLowFrequencyBackend` 求解由此產生的非線性特徵值問題
  （見 [modal validation](modal_validation.zh-TW.md)）；以
  `generate_hybrid_rir.py --low-backend analytic-impedance
  --impedance-boundary-config <json>` 選用。
- **FDTD reference** —— `simulate_fdtd_reference(..., boundary_admittance=...)`
  在每個牆面 cell 保存各 branch 的邊界狀態。

對 $L_x \times L_y \times L_z$ 的矩形房間，剛性參考頻率為

$$f_{n_x n_y n_z} = \frac{c}{2}\sqrt{\left(\frac{n_x}{L_x}\right)^2 + \left(\frac{n_y}{L_y}\right)^2 + \left(\frac{n_z}{L_z}\right)^2}$$

複數邊界會使其位移並加入衰減。只有在幾何、環境、牆面方向與 phasor
convention 都一致時，才比較 modal 與 FDTD 結果。對單一正入射樣品 fit 得好，
不代表擴散、角度相依的房間邊界已被驗證。

## 工具

```bash
# 原始雙麥克風 H12 sweep -> complex_impedance_measurement.v1
python egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py --help
# fit normalized 量測、密集頻格 passivity、1D 模態、可選的重複量測比較
python egs/rir_generation/phases/m2_impedance/scripts/validate_impedance_measurement.py --help
```

資料範本位於
`egs/rir_generation/phases/m2_impedance/measurements/impedance_tube_template/`。
`.../measurements/zenodo_15195587/` 下的公開 liner 資料用來演練 normalized
grazing-duct 匯入與 resonant fitting；它們是驗證資料，不是牆面材料（見
[公開資料來源準則](complex_impedance_source_audit.zh-TW.md)）。

測試：`test/rir/test_acoustic_impedance.py`、
`test_impedance_measurements.py`、`test_impedance_modes.py`、
`test_impedance_tube.py`。

## 發布一個 fitted boundary 之前

1. 驗證來源檔案、provenance 與再散布授權。
2. 保留量測幾何與 phasor convention。
3. 在 fit 頻帶內外檢查 passivity。
4. 回報 held-out 的複數、magnitude 與 phase 誤差。
5. 交叉比對 modal 與 FDTD 行為。
6. 保留厚度、背板、air gap、安裝方式與環境限制。
7. 只能透過明確、經 review 的映射把模型加入材料 catalog。
