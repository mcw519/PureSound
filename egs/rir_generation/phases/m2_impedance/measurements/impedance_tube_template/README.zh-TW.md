# 雙麥克風阻抗管輸入範本

English version: [`README.md`](README.md)

量測前，先複製這三個範本檔案：

- `raw_transfer.template.csv`：每個 repeat 與每個頻率各一列；
- `microphone_switch.template.csv`：正常與交換麥克風後的校正；
- `metadata.template.json`：環境、管、擷取、樣品、provenance、適用範圍。

這些 CSV 檔案刻意只有 header，沒有其他內容。不要把這個目錄當成已量測
的材質資料。reduction 指令與繁體中文協定文件見
`docs/audio/impedance_tube_protocol.zh-TW.md`。

在獨立 specimen、被動性、fit、FDTD、模態這五道 gate 都通過之前，
`automatic_scene_catalog_mapping` 保持 false。
