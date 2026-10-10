# 公開複數阻抗資料：接受準則

English version: [complex_impedance_source_audit.md](complex_impedance_source_audit.md)

複數邊界路徑（[複數聲學阻抗](impedance_measurements.zh-TW.md)）只接受保留
反射相位、安裝條件與授權的證據。本頁說明公開資料必須符合的規則、列出依此
規則檢查過的公開來源，並描述唯一保留作為驗證資料的資料集。

## 接受準則

一份資料必須同時：

1. 直接提供複數表面阻抗 $Z(f)$、複數反射係數 $\Gamma(f)$，或可重建它們的
   calibrated 複數雙麥克風轉移函數；
2. 明確標示頻率、實部、虛部與 phasor convention；
3. 樣品厚度、背板與 air gap、量測幾何與操作條件可追溯；
4. 來源 URL、版本與 license 可追溯；
5. 不以擴散場吸音率或只有 $\alpha(f) = 1 - |\Gamma(f)|^2$ 的資料代替唯一相位；
6. 映射到房間材料之前，量測組態必須與實際安裝相容。

從論文圖表以像素數位化取值不能取代原始數值，也不能用 $\alpha$ 猜測相位。

## 已檢查的公開來源

| 來源 | 狀態 | 理由 |
|---|---|---|
| [Zenodo 15195587](https://zenodo.org/records/15195587) | 接受，僅限 pipeline 驗證 | CC BY 4.0 HDF5，逐頻率提供 normalized resistance/reactance；論文記載樣品幾何、SPL、流場條件、試驗台與 eduction 方法 |
| [FOAM 02](https://zenodo.org/records/15407780) | 不是複數來源 | 只有吸音係數、直徑與分類；沒有相位或阻抗 |
| [FOAM 01](https://zenodo.org/records/10551344) | 不是複數來源 | 只有吸音係數與標籤；相位無法由 $\alpha$ 還原 |
| [vyhyb/imptube](https://github.com/vyhyb/imptube) | 方法參考 | MIT 授權的轉移函數化簡實作；沒有已量測樣品資料 |
| [MDPI Mathematics 10(18), 3264](https://www.mdpi.com/2227-7390/10/18/3264) | 不是室內裝修材料來源 | 燒結金屬纖維樣品的阻抗圖，沒有可追溯的逐頻率數值 |
| [Frontiers in Physics 14:1785611](https://www.frontiersin.org/journals/physics/articles/10.3389/fphy.2026.1785611/full) | 方法參考 | 阻抗推導方法只做了數值驗證 |

「不是複數來源」與資料品質無關：FOAM 01/02 仍可用於正入射吸音率分佈、分類
或其他不需要相位的工作。

目前找到的公開資料中，沒有任何一份能以機器可讀的原始值，提供常見室內裝修材料
或多孔吸音材在正入射、室內 SPL 下的量測。這類資料要依
[阻抗管量測流程](impedance_tube_protocol.zh-TW.md) 自行產生。

## 保留的驗證資料集

從 Zenodo 15195587 的 Figure 6，取兩組無流、130 dB 的 Kumaresan–Tufts
eduction 資料，轉成
[normalized 契約](impedance_measurements.zh-TW.md#normalized-量測契約)：

| 檔案（位於 `egs/rir_generation/phases/m2_impedance/measurements/zenodo_15195587/`） | HDF5 dataset |
|---|---|
| `nasa_gfit_noflow_130db_kt.csv` / `.json` | `/Figure6/Resistance-NASA-KT`、`/Figure6/Reactance-NASA-KT` |
| `ufsc_noflow_130db_kt.csv` / `.json` | `/Figure6/Resistance-UFSC-KT`、`/Figure6/Reactance-UFSC-KT` |

- 頻帶：論文共同比較範圍 500–2500 Hz。
- 數值依發表原樣複製為 $z = Z/(\rho c)$，沒有插值。
- 來源 `paper_data.hdf5`，SHA-256
  `ba4cf7cf293d2b20ed590eb78ed8c133484771acbd011c680466d289abdc1a72`。
- 每個 sidecar 記錄 HDF5 dataset 路徑、checksum、引用文獻、授權、掃描幾何、
  條件與轉換步驟，並標示 `acoustic_field_geometry: grazing_duct`。

數值保持無因次，是因為 HDF5 沒有保留每筆資料 normalize 時使用的 $\rho$ 與
$c$；自行選一個標準大氣，等於把換算假設冒充成量測環境。

**適用範圍。** 這是高 SPL、穿孔、背腔式的飛機 liner，阻抗由 grazing-duct 聲場
推導。它適合驗證 phase-aware 匯入、Helmholtz 共振 fitting、passivity 與因果
數位化，以及把同一個 rational 邊界接到 FDTD solver 與特徵值問題。它不能代表
油漆牆面、地毯、窗簾或天花板、低 SPL 的室內語音條件，也不能套用到入射、流速、
穿孔尺寸、背腔深度或背板不相符的情況。因此兩個 sidecar 都設定
`applicability.scope: pipeline_validation_only` 與
`automatic_scene_catalog_mapping: false`。
