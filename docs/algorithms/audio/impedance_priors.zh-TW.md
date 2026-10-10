# Phase-aware 低頻阻抗 priors

English version: [impedance_priors.md](impedance_priors.md)

`puresound.audio.rir.physics.impedance.priors` 收錄少量複數表面阻抗參考值，
用來驗證複數邊界路徑（被動 fitting、FDTD boundary、impedance modal solver）。
它從不把阻抗套到 scene 材料上：每個 prior 的 metadata 都回報
`production_material_mapping_enabled: false`。

## Evidence tier

本模組區分三種證據：

| Tier | 意義 |
|---|---|
| 實測複數阻抗 | 直接量測 $Z(f)$ 的實部與虛部（見[複數聲學阻抗](impedance_measurements.zh-TW.md)） |
| 實測參數 + 已發表模型 | 把實測的非聲學性質代入具名物理模型 |
| engineering prior | 為模擬選定的參數，不是證據 |

catalog 只收第二種，以
`EVIDENCE_TIER_MEASURED_PARAMETER_MODEL = "measured_flow_resistivity_plus_miki_model"`
標示：airflow resistivity 是實測值，複數阻抗（尤其是相位）是模型預測，
不可描述成複數阻抗量測。`MikiPorousLayerPrior` 會拒絕其他 tier，且至少
需要一個 provenance URL。

更強的 prior 需要 phase-aware 的正入射量測，例如 ISO 10534-2
（[overview](https://www.iso.org/standard/81294.html)）；該標準本身就把它和
混響室的擴散入射吸音率區分開來。這類資料的匯入路徑見
[阻抗管量測流程](impedance_tube_protocol.zh-TW.md)。

## Miki 剛性背板多孔層

`MikiPorousLayerPrior` 計算 Miki 對 Delany–Bazley 模型的 positive-real 修正
（[Miki 1990](https://doi.org/10.1250/ast.11.19)），對象是厚度 $d$（m）、
剛性背板的多孔層。令 airflow resistivity 為 $\sigma$（Pa·s/m²）、
$X = 1000 f/\sigma$，採 $e^{+j\omega t}$ convention：

$$Z_c = \rho c\,\bigl[1 + 5.50X^{-0.632} - j\,8.43X^{-0.632}\bigr],\qquad
k = \frac{\omega}{c}\bigl[1 + 7.81X^{-0.618} - j\,11.41X^{-0.618}\bigr]$$

$$Z_s = -j\,Z_c \cot(kd),\qquad
\Gamma = \frac{Z_s - \rho c}{Z_s + \rho c},\qquad
\alpha = 1 - |\Gamma|^2$$

| 方法 | 回傳 |
|---|---|
| `characteristic_impedance_pa_s_m(f, rho, c)` | $Z_c$，Pa·s/m |
| `complex_wavenumber_rad_m(f, c)` | $k$，rad/m |
| `surface_impedance_pa_s_m(f, rho, c)` | $Z_s$，Pa·s/m |
| `reflection_coefficient(f, rho, c)` | 正入射複數 $\Gamma$ |
| `absorption_coefficient(f, rho, c)` | $\alpha$，裁切到 [0, 1] |
| `metadata()` | catalog 版本、provenance、有效頻帶、模型與 convention 字串 |

超出 $0.01 \le f/\sigma \le 1$（`validity_frequency_range_hz` property）時每個
計算都會丟例外：這個經驗模型不會被悄悄外推。

## Catalog

`reference_impedance_priors()` 回傳三個以 `prior_id` 為 key 的不可變 prior
（catalog 版本 `puresound-impedance-priors.v0`）。Flow resistivity 取自
Tarnow 對玻璃棉板的量測（[Tarnow 2002](https://doi.org/10.1121/1.1476686)）：

| `prior_id` | $\sigma$ | $d$ | 有效頻帶 |
|---|---|---|---|
| `glass_wool_14kgm3_50mm_normal` | 5 880 Pa·s/m² | 0.05 m | 58.8–5 880 Hz |
| `glass_wool_14kgm3_100mm_normal` | 5 880 Pa·s/m² | 0.10 m | 58.8–5 880 Hz |
| `glass_wool_30kgm3_100mm_normal` | 15 500 Pa·s/m² | 0.10 m | 155–15 500 Hz |

50 mm 這一列是模型推導的厚度變體，不是實測「已安裝 50 mm」的阻抗。這些值
都不能代表地毯、天花板磚或其他纖維材料。

## 擬合時域邊界

時域 solver 無法直接使用 $Z_s(f)$，需要因果且被動的邊界。
`fit_first_order_relaxation(prior, f_min, f_max, *,
air_density_kg_m3=1.204, sound_speed_m_s=343.0, num_frequencies=128,
acceptance_threshold=0.08)` 擬合 `FirstOrderRelaxationAdmittance`，即單極點
normalized admittance（$y = \rho c\,Y_\text{surface}$）：

$$y(s) = g_\infty + \frac{g_0 - g_\infty}{1 + s\tau},\qquad \tau = \frac{1}{2\pi f_r},\qquad \Gamma = \frac{1-y}{1+y}$$

欄位為 `normalized_admittance_infinite`（$g_\infty$）、
`normalized_admittance_relaxation`（$g_0 - g_\infty$）與
`relaxation_frequency_hz`（$f_r$）。

- 在 `num_frequencies` 個 log 間隔頻點上，於複數壓力反射域擬合；實部與虛部
  誤差一起堆疊，magnitude 與 phase 同時對齊。
- $g_0$、$g_\infty$、$f_r$ 在 log 空間最佳化，因此兩端 admittance 恆為非負，
  模型即為 positive-real。會試三個起點，取 cost 最低者。
- 擬合頻帶必須落在 prior 的有效頻帶內，且至少需要八個頻點。
- `ImpedancePriorFit` 回報 RMS 與最大複數誤差、最大 magnitude 與 phase 誤差，
  以及 `accepted`（最大複數誤差 $\le$ `acceptance_threshold`）。

$g_0 < g_\infty$（relaxation 強度為負）的 fit 是合法的：兩端仍非負，而遞增的
admittance 正是剛性背板多孔層的 compliant phase 在此模型中的表現。

## 使用方式

- `egs/rir_generation/phases/m2_impedance/config/` 下的六面牆設定
  `impedance_reference_glass_wool_14kgm3_{50,100}mm.json`（schema
  `puresound.rectangular_impedance_boundary.v1`）存放這類 fit，作為
  `generate_hybrid_rir.py --low-backend analytic-impedance
  --impedance-boundary-config <json>` 以及
  [modal validation](modal_validation.zh-TW.md) 中 FDTD validator 的輸入。
- 測試：`test/rir/test_impedance_priors.py`（provenance、有效頻帶、
  passivity、fitting）。

## 限制

這些 prior 用來驗證 solver，不描述房間。材料若要在 production 帶有阻抗，
需要：已安裝組態（厚度、背板、air gap、安裝方式、不確定度）的直接正入射
量測、經[量測契約](impedance_measurements.zh-TW.md)的被動 fitting、明確且經
review 的相容 scene 材料映射，以及對斜入射的處理——正入射 locally reacting
的值本身無法提供斜入射行為。
