# 內部 RIR 工具

這些輔助工具支援實驗與 bank 準備工作，但刻意不納入公開的根目錄 CLI
介面。

English version: [`README.md`](README.md)

- `audition/`：卷積、preview-bank、room-scene、README 樣本等輔助工具；
- `measured/`：實測 RIR 語料掃描與 bank 產出；
- `bank/`：純 symlink 的 bank view，以及 curriculum 切片。

一般的模擬 RIR 生成、M6 bank 封裝、繪圖、資料夾統計，請改用上層目錄的
七個指令。
