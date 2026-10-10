# 正入射複數阻抗量測與匯入

English version: [impedance_tube_protocol.md](impedance_tube_protocol.md)

`puresound.audio.rir.physics.impedance.tube` 把重複的雙麥克風阻抗管 sweep
化簡成[量測契約](impedance_measurements.zh-TW.md)所使用的複數表面阻抗。
本頁定義座標慣例、化簡演算法、品質門檻、原始檔案格式與量測流程。

## 為何量 $H_{12}$ 而不是吸音率

低頻房間邊界除了吸收能量，也會改變反射相位。$\alpha(f) = 1 - |\Gamma(f)|^2$
只給出 magnitude，因此無法決定表面阻抗、模態頻率位移或因果的時域邊界。
管中兩支固定麥克風量到的是複數轉移函數

$$H_{12}(f) = \frac{P(x_2, f)}{P(x_1, f)}$$

它同時保留振幅與相位。ISO 10534-2 的範圍正是正入射吸音率與表面阻抗；其結果
不等於混響室的擴散入射吸音率。

## 座標慣例與反射係數

- 樣品表面在 $x = 0$；$+x$ 由樣品指向聲源；$x_1, x_2 > 0$ 是兩支麥克風到
  樣品的距離。
- $H_{12} = P(x_2)/P(x_1)$；phasor convention 為 $e^{+i\omega t}$。

只有平面波時，$p(x) = A e^{ikx} + B e^{-ikx}$，$k = 2\pi f/c$；$A$ 朝樣品前進、
$B$ 為反射波，$\Gamma = B/A$。消去 $A$、$B$
（`reflection_from_two_microphone_transfer`）：

$$\Gamma = \frac{e^{ikx_2} - H_{12}e^{ikx_1}}{H_{12}e^{-ikx_1} - e^{-ikx_2}},\qquad Z_s = Z_0\,\frac{1+\Gamma}{1-\Gamma},\quad Z_0 = \rho c$$

$Z_s(f)$ 是被動 rational fitting、FDTD 邊界與複數模態特徵值問題的共同輸入。
`transfer_from_surface_reflection` 是用於校正與 round-trip 測試的正向模型。

## 麥克風互換校正

兩個通道的複數靈敏度不會完全相同。設不匹配為 $C(f)$，同一校正聲場在原位與
互換位置各量一次，得到 $H_\mathrm{I} = CH$、$H_\mathrm{II} = C/H$，因此

$$C(f) = \sqrt{H_\mathrm{I}(f)\,H_\mathrm{II}(f)},\qquad H_{12,\text{corrected}} = H_{12,\text{raw}}/C$$

`microphone_switch_calibration_factor` 先對 $H_\mathrm{I}H_\mathrm{II}$ 的相位
unwrap 再開根號，讓分支跨頻率保持連續，並選擇實部中位數為正的符號。除非
metadata 設 `acquisition.microphone_switch_calibration_required: false`，
化簡 script 都要求這項校正；同型號的兩支麥克風不會被假設為已匹配。

## 有效頻帶門檻

`TwoMicrophoneTubeGeometry.validate_frequencies` 會拒絕違反任一條件的頻帶：

- **平面波上限。** 直徑 $D$ 的剛性圓管，第一個橫向模態在
  $f = 1.841c/(\pi D)$ 出現；所選的每個頻率都必須低於它。
- **間距條件數。** 間距 $s = |x_1 - x_2|$，$|\sin(ks)|$ 太小時兩支麥克風幾乎
  分不出入射波與反射波，反解會放大誤差。整個頻帶都必須滿足
  $|\sin(ks)| \ge$ `minimum_spacing_sine`（預設 0.05）。這是數值防護，不能
  取代實驗室依標準的選頻程序。

## 化簡、不確定度與被動性

`reduce_two_microphone_repeats(frequencies, h12_repeats, coherence_repeats,
geometry, *, air_density_kg_m3, microphone_correction=None,
minimum_coherence=0.95, passivity_tolerance=1e-6)`：

1. 要求所有重複量測使用相同頻格；
2. 要求每個頻率的平均 magnitude-squared coherence 達到 `minimum_coherence`；
3. 除以麥克風校正後，逐次計算 $\Gamma$ 與 $Z_s$；
4. 對重複量測平均 $Z_s$，並回報實部與虛部的樣本標準差（只有一次時為零）；
5. 平均 $\operatorname{Re} Z_s$ 低於 $-$`passivity_tolerance`，或結果
   $|\Gamma| > 1 +$ `passivity_tolerance` 時拒絕；落在容許範圍內的負實部會被
   截為零。

下游的 `fit_complex_impedance_measurement` 以
$\partial\Gamma/\partial Z = 2Z_0/(Z + Z_0)^2$ 把標準差傳遞到反射域，作為反比
權重。重複量測必須是真正的「拆下－重裝－重量」循環：同一次安裝重放訊號，
抓不到密封、壓縮或邊緣洩漏造成的變異。單次 sweep 仍可產生阻抗，但沒有重複性
不確定度（輸出就不含不確定度欄位），也不應通過 production 材料門檻。

## 原始資料契約

長格式轉移函數 CSV（`load_transfer_repeats_csv`）：

```csv
repeat_id,frequency_hz,h12_real,h12_imag,coherence
install_01,200,0.91,-0.13,0.997
install_01,250,0.88,-0.17,0.998
install_02,200,0.90,-0.14,0.996
install_02,250,0.87,-0.18,0.997
```

麥克風互換 CSV（`load_microphone_switch_csv`，至少三列，頻格與轉移函數 CSV
相同）：

```csv
frequency_hz,h12_original_real,h12_original_imag,h12_swapped_real,h12_swapped_imag
200,1.02,0.03,1.01,0.02
250,1.02,0.03,1.01,0.02
```

JSON sidecar，schema `puresound.impedance_tube_transfer_measurement.v1`，由
化簡 script 檢查：

| 區塊 | 必要欄位 |
|---|---|
| 頂層 | `schema_version`、`measurement_id`、`method`、`phasor_convention: "exp(+i*omega*t)"` |
| `environment` | `air_density_kg_m3`、`sound_speed_m_s`、`temperature_c`、`relative_humidity_percent`（0–100）、`pressure_pa` |
| `tube` | `shape: "circular"`、`diameter_m`、`microphone_1_distance_from_sample_m`、`microphone_2_distance_from_sample_m` |
| `acquisition` | `analysis_frequency_range_hz` `[min, max]`、`minimum_coherence`、`source_spl_db`；可選 `minimum_spacing_sine`（0.05）、`microphone_switch_calibration_required`（true）、`passivity_tolerance`（1e-6） |
| `sample` | `sample_id`、`material_label`、`mounting`、`backing`、`thickness_m`、`air_gap_m` |
| `provenance` | `source_url`（或實驗記錄位置）、`license` |
| `applicability` | 可選；預設 `scope: "measurement_validation_pending"` 與 `automatic_scene_catalog_mapping: false`，在量測被接受前都維持 false |

可重用的範本（`metadata.template.json`、`raw_transfer.template.csv`、
`microphone_switch.template.csv`）位於
`egs/rir_generation/phases/m2_impedance/measurements/impedance_tube_template/`。

## 執行化簡

```bash
python egs/rir_generation/phases/m2_impedance/scripts/reduce_impedance_tube_measurement.py \
  --transfer-csv path/to/raw_h12.csv \
  --metadata path/to/raw_h12.json \
  --microphone-switch-csv path/to/microphone_switch.csv \
  --output-csv path/to/complex_impedance.csv \
  --output-metadata path/to/complex_impedance.json
```

`--minimum-frequency` / `--maximum-frequency` 可覆寫設定的分析頻帶。輸出是一組
`puresound.complex_impedance_measurement.v1` 檔案，可由
`ComplexImpedanceMeasurement.from_csv_and_metadata` 讀入，並直接交給
`fit_complex_impedance_measurement()`。Sidecar 保留原始檔與校正檔的 SHA-256、
管子幾何與頻帶門檻、repeat id、coherence、座標與轉移函數定義，以及轉換清單。

## 量測流程

先從一種安裝方式明確、可裁成可重複樣品的常見多孔吸音材開始（例如 50 或
100 mm 玻璃棉或岩棉、剛性背板、無 air gap）：

1. 至少三個獨立樣品；
2. 每個樣品至少重新安裝三次；
3. 兩個聲源位準（例如 75 與 85 dB），檢查位準相依性；
4. 記錄厚度公差、密度或面密度、批次、裁切直徑與周邊密封；
5. 由間距與管子截止頻率兩個限制的交集選頻帶；
6. fitting 後以 held-out 樣品驗證，而不只是 held-out 頻率；
7. 只有在反射 magnitude/phase、passivity、FDTD 與 1D 模態檢查都對「完全相同的
   安裝組態」通過後，才把該組態映射到 scene catalog。

一律保留原始 $H_{12}$ 頻譜：之後加入管損或幾何修正時必須從原始資料重跑化簡，
只保留最終吸音曲線是不夠的。

## 化簡的限制

- 管內傳播使用實數波數 $k = 2\pi f/c$，沒有套用窄管的 thermoviscous 損耗修正。
- 不確定度只來自樣品與重新安裝的重複量測；麥克風位置、空氣參數與校正頻譜的
  誤差沒有被傳遞。
- 高 coherence 只代表線性頻譜估計穩定，不代表沒有邊緣洩漏、樣品壓縮、側向
  約束或樣品間變異。
- 只支援圓管與正入射平面波。
- 正入射、locally reacting 的阻抗無法說明房間模型中的斜入射、有限面積 patch
  或 non-local reaction。

測試：`test/rir/test_impedance_tube.py`。
