# Phase-aware 低頻阻抗 priors

English version: `impedance_priors.md`

`puresound.audio.rir.physics.impedance.priors` 收錄用來驗證 M2 複數邊界
路徑的實驗性參考資料。它不會自動把阻抗套用到 production scene 材料上。

## Evidence contract

每一個 prior 都記錄 evidence tier、來源 URL、有效頻帶、layer 組態，以及
是否啟用 production 材料映射。目前唯一的 evidence tier 是:

```text
measured_flow_resistivity_plus_miki_model
```

這表示 airflow resistivity 是實際量測值，而複數表面阻抗則是由一個已發表
的模型預測而來。它不可以被描述成直接的複數阻抗量測。

要取得更強的 prior，量測方法必須是 phase-aware 的 normal-incidence 方法，
例如 ISO 10534-2。該標準明確區分 normal-incidence 阻抗管結果與
diffuse-incidence 混響室吸收率:

- [ISO 10534-2:2023 overview](https://www.iso.org/standard/81294.html)

## Miki 剛性背板多孔層模型

這個 phase-aware 參考採用 Miki 對 Delany-Bazley 多孔材料模型的
positive-real 修正版本:

- [Miki 1990, DOI 10.1250/ast.11.19](https://doi.org/10.1250/ast.11.19)

令 `X = 1000 f / sigma`，並採用 `exp(+j omega t)` convention:

```text
Zc = rho*c [1 + 5.50 X^-0.632 - j 8.43 X^-0.632]
k  = omega/c [1 + 7.81 X^-0.618 - j 11.41 X^-0.618]
Zs = -j Zc / tan(k d)
```

`sigma` 是 airflow resistivity，單位 Pa·s/m²；`d` 是層厚，單位公尺。程式
保守地限制 `0.01 <= f/sigma <= 1.0`，不會悄悄外推到這個經驗模型原本有效
範圍之外。

## 初始參考資料

Flow-resistivity 參數來自 Tarnow 對 100 mm 玻璃棉板的量測:

- [Tarnow 2002, DOI 10.1121/1.1476686](https://doi.org/10.1121/1.1476686)

| Prior | Flow resistivity | Thickness | Model-valid range | Production mapping |
|-------|------------------|-----------|-------------------|--------------------|
| 14 kg/m³ glass wool, normal direction | 5.88 kPa·s/m² | 50 mm model-derived variant | 58.8–5880 Hz | disabled |
| 14 kg/m³ glass wool, normal direction | 5.88 kPa·s/m² | 100 mm source configuration | 58.8–5880 Hz | disabled |
| 30 kg/m³ glass wool, normal direction | 15.5 kPa·s/m² | 100 mm source configuration | 155–15500 Hz | disabled |

Flow resistivity 是在來源樣品上實測得到。50 mm 這一列是明確的 Miki
剛性背板厚度實驗，不是直接量測「已安裝 50 mm」的阻抗。這些數值都不能
通用到任意地毯、天花板磚或纖維物件。

## 時域 fitting

`fit_first_order_relaxation()` 直接在複數壓力反射域對被動、因果的 FDTD
boundary 做 fitting。它同時最佳化零頻 admittance、無限頻 admittance 與
relaxation frequency，並保持兩個端點 admittance 皆為非負值。

| Prior | Fit band | RMS complex error | Maximum complex error | Maximum phase error |
|-------|----------|-------------------|-----------------------|---------------------|
| 14 kg/m³, 50 mm | 60–300 Hz | 0.00313 | 0.00506 | 0.00413 rad |
| 14 kg/m³, 100 mm | 60–300 Hz | 0.0189 | 0.0571 | 0.0710 rad |
| 30 kg/m³ | 155–300 Hz | 0.00782 | 0.0160 | 0.00741 rad |

兩者都通過診斷用的最大誤差門檻 0.08。這些 fit 中出現的負
relaxation 差值是合理的:低頻與高頻 admittance 仍維持非負，而遞增的
admittance 代表剛性背板多孔層的 compliant phase。

## FDTD 模態診斷

一個 2.0 × 1.2 × 1.0 m 的參考房間比較:

1. 14 kg/m³ 的 phase-aware fit;
2. 一個頻率無關的實數 boundary，在 80 Hz 有相同的反射量值但相位為零。

主要響應峰值從 magnitude-only 情況下的 85.7 Hz，移到 phase-aware
boundary 下的 64.0 Hz。估計的 Q 值則從約 10.5 變為 15.8。這不是一個
measured-room 準確度的宣稱;它只是說明即使某個頻率的反射量值被固定，
反射相位仍會實質改變模態頻率與 Q。

高瞬時 admittance 也揭露了一個 interior CFL 條件未涵蓋的
corner/edge 穩定性上限。FDTD solver 現在會持續縮小時間步，直到加總後的
normalized boundary Courant number 低於 0.9，並同時序列化 interior 與
boundary 兩個時間步上限。

## 程式入口

舊的扁平模組 `puresound/audio/impedance_priors.py` 已在 RIR 套件遷移中
移除(見 [`rir_package_migration.md`](rir_package_migration.md));這份
catalog 現在位於 `physics.impedance` sub-package 之下。

| 檔案 | 職責 |
|------|------|
| `puresound/audio/rir/physics/impedance/priors.py` | 本模組:`MikiPorousLayerPrior`、`reference_impedance_priors()`、`fit_first_order_relaxation()` |
| `puresound/audio/rir/physics/impedance/admittance.py` | `FirstOrderRelaxationAdmittance`,即 `fit_first_order_relaxation()` 擬合的被動因果 boundary |
| `test/test_impedance_priors.py` | provenance／validity、passivity、fitting 與 FDTD modal diagnostic 測試 |

## 尚待補齊的 gate

在 production 使用之前，還需要:

- 針對 room-finish 組態取得直接的 normal-incidence 量測;
- 表示 mounting、厚度、背板、air gap 與不確定度;
- 讓這些量測通過嚴格的 CSV/JSON 契約，以及
  [`impedance_measurements.zh-TW.md`](impedance_measurements.zh-TW.md)
  所述的被動 relaxation／resonant fitting;
- 只映射相容的 scene 材料;
- 把已驗證的 1D 複數 eigenvalue 連接點，延伸到 production 的 3D 模態
  問題;
- 重新執行 room-disjoint measured-bank 與下游語音測試。
