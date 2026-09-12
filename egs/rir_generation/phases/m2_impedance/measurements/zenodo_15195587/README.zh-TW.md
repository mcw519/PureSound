# Zenodo 15195587 normalized liner 阻抗

English version: [`README.md`](README.md)

這個目錄放的是以下論文所發布之 normalized 複數阻抗資料中，
500–2500 Hz 的子集：

> Nicolas T. Quintino, Lucas A. Bonomo, Julio A. Cordioli, Michael G. Jones,
> Brian M. Howerton, Douglas M. Nark, and Francesco Avallone,
> “Comparison of Impedance Eduction Test Rigs with Different Boundary-Layer
> Profiles,” *AIAA Journal* 63(11), 2025.

- 資料集 DOI：<https://doi.org/10.5281/zenodo.15195587>
- 論文 DOI：<https://doi.org/10.2514/1.J065173>
- 資料集授權：CC BY 4.0
- 來源檔案：`paper_data.hdf5`
- 來源 MD5：`ba222c796afe37e4afe911f33df6a7c4`
- 來源 SHA-256：
  `ba4cf7cf293d2b20ed590eb78ed8c133484771acbd011c680466d289abdc1a72`

相對來源的改動：

- 複製 Figure 6 的頻率、KT resistance、KT reactance 陣列；
- 保留論文共同的 500–2500 Hz 比較頻帶；
- 把 NASA 與 UFSC 拆成嚴格的 CSV/JSON 量測配對；
- 以無因次的 \(Z/(\rho c)\) 保留數值；
- 新增機器可讀的樣品幾何、provenance、適用範圍；
- 未做插值、平滑，或 SI 大氣條件換算。

這些高聲壓 grazing-duct liner 資料是為了 pipeline validation 而納入的，
不會自動對應到 room-material catalog。
