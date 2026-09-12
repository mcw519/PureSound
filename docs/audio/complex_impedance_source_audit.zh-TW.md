# 複數聲學阻抗公開資料審核

English version: `complex_impedance_source_audit.md`

本文件記錄 M2.5 對公開資料的逐項審核。目的不是收集越多
absorption curve 越好，而是只納入能保留反射相位、安裝條件與授權的
直接 complex impedance 證據。

## 1. 接受門檻

一份資料至少必須符合：

1. 直接提供複數表面阻抗 \(Z(f)\)、複數反射係數 \(\Gamma(f)\)，或提供
   可重建它們的 calibrated complex two-microphone transfer function；
2. 頻率、實部、虛部和 phasor convention 明確；
3. 樣品厚度、背板／air gap、量測幾何與重要操作條件可追溯；
4. 來源 URL、版本與 license 可追溯；
5. 不把 diffuse-field absorption 或只有
   \(\alpha(f)=1-|\Gamma(f)|^2\) 的資料補猜成唯一相位；
6. 自動映射到 room material catalog 前，量測配置必須與實際安裝相容。

## 2. 審核結果

| 來源 | 結果 | 理由 |
|------|------|------|
| [Zenodo 15195587](https://zenodo.org/records/15195587) | 接受，但只限 pipeline validation | CC BY 4.0 HDF5 直接提供逐頻率 normalized resistance/reactance；論文描述樣品幾何、130/145 dB、無流與 grazing-flow 條件、NASA/UFSC 試驗台和 eduction 方法。 |
| [FOAM 02](https://zenodo.org/records/15407780) | 拒絕作為 complex source | Apache-2.0 與樣品標籤清楚，但公開檔只有 absorption coefficient、直徑與分類，沒有 reflection phase 或 complex impedance。 |
| [FOAM 01](https://zenodo.org/records/10551344) | 拒絕作為 complex source | 公開檔只有 absorption coefficient 與樣品標籤；不能從 \(\alpha\) 唯一恢復相位。 |
| [vyhyb/imptube](https://github.com/vyhyb/imptube) | 方法參考，不是量測資料源 | MIT 實作可由 calibrated transfer function 計算 reflection 與 surface impedance，但 repository 沒有可直接納入的已量測樣品資料。 |
| [MDPI Mathematics 10(18), 3264](https://www.mdpi.com/2227-7390/10/18/3264) | 拒絕作為 production room-finish source | 文章包含六個燒結金屬纖維樣品的 normalised surface-impedance 圖與量測配置，但未提供可追溯的逐頻率原始數值；材料也不是本階段優先的常見室內 finish。 |
| [Frontiers in Physics 14:1785611](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2026.1785611/full) | 方法參考，不是實體量測 source | 研究測試 complex surface-impedance deduction，但本文驗證是 numerical experiment，結論也把 physical validation 列為後續工作。 |

「拒絕」只表示不符合這一條 complex-impedance 路徑，不表示資料品質差。
FOAM 01/02 仍可用於 normal-incidence absorption 分布、分類或其他不需要
phase 的工作。

## 3. 第一份接受資料

M2.5 納入 Zenodo 15195587 Figure 6 的兩組無流、130 dB KT eduction：

- NASA GFIT：`Resistance-NASA-KT`、`Reactance-NASA-KT`；
- UFSC：`Resistance-UFSC-KT`、`Reactance-UFSC-KT`；
- 使用論文共同比較範圍 500–2500 Hz；
- 數值原樣保留為 \(z=Z/(\rho c)\)，沒有插值；
- 原始 `paper_data.hdf5`：
  SHA-256
  `ba4cf7cf293d2b20ed590eb78ed8c133484771acbd011c680466d289abdc1a72`。

兩份已轉換檔位於：

- `egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/nasa_gfit_noflow_130db_kt.csv`
- `egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/ufsc_noflow_130db_kt.csv`

各自的 JSON sidecar 記錄 HDF5 dataset path、checksum、文章、授權、實際掃描
幾何、量測條件與 transformation。

## 4. 為什麼新增 normalized contract

來源發表的是無因次：

\[
z(f)=\frac{Z(f)}{\rho c}=r(f)+j x(f)
\]

但 HDF5 沒有逐筆保存當時用來 normalization 的精確 \(\rho\) 與 \(c\)。
若任選標準大氣再乘回 Pa·s/m，會把「參考換算環境」誤寫成「量測環境」。
因此 repository 保留原始 dimensionless 數值，並另設
`puresound.normalized_complex_impedance_measurement.v1`。需要在指定空氣
環境使用時，再由該場景自己的 \(\rho c\) 轉成 SI。

## 5. 適用範圍

這是高聲壓、穿孔、背腔式 aircraft liner，資料由 grazing duct 的 acoustic
field 反推表面阻抗。它很適合驗證：

- phase-aware ingestion；
- Helmholtz resonance fitting；
- passivity 與因果數位化；
- 同一 rational boundary 的 FDTD／eigenproblem 接線。

它不適合直接代表：

- 油漆牆、地毯、窗簾、天花板；
- 低聲壓室內語音條件；
- 未匹配的 incidence、流速、孔徑、背腔深度或 backing。

因此兩份 sidecar 都設為
`automatic_scene_catalog_mapping: false`。第一份可直接映射到一般房間材料
的資料，仍應是正常聲壓、normal-incidence、安裝配置完整的 porous 或 room
finish 量測。

## 6. M2.6 第二輪結論與處置

截至 2026-07-30，第二輪公開來源審核仍未找到同時具備下列條件、可直接
納入的資料：

- 常見 room-finish 或 porous absorber；
- normal-incidence 的逐頻率 complex \(Z\)、complex \(\Gamma\) 或
  calibrated complex \(H_{12}\)；
- 厚度、backing、air gap、環境與正常室內聲壓級；
- 可重用授權與 machine-readable raw values。

不能用論文圖的像素 digitization 冒充原始量測，也不能用 FOAM 01/02 的
\(\alpha\) 補猜相位。因此本輪沒有新增「已接受的 room material」，而是
補齊可重現的 two-microphone acquisition/reduction：

- raw repeated \(H_{12}=P(x_2)/P(x_1)\) contract；
- microphone-switch complex calibration；
- circular-tube plane-wave cutoff 與 microphone-spacing conditioning；
- coherence gate、跨安裝 repeatability 與被動性 gate；
- \(Z\) uncertainty 到 complex reflection fit 的權重傳播；
- 可直接輸出 `puresound.complex_impedance_measurement.v1`。

完整算法、原始檔格式與第一批實驗設計見
[`impedance_tube_protocol.zh-TW.md`](impedance_tube_protocol.zh-TW.md)。
ISO 10534-2:2023 的公開說明確認 two-microphone complex transfer
technique 可取得 normal surface impedance，也明確指出阻抗管 normal
incidence 和混響室 random/diffuse incidence 結果不能直接比較：
[ISO 10534-2:2023](https://www.iso.org/standard/81294.html)。
