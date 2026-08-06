# puresound.dataset

English version: [`README.md`](README.md)

兩種 manifest 風格：

* **Dynamic**（`dynamic_base.py` + `parser.py`，active）：用一份 CSV metafile
  列出乾淨（clean）語音；每個項目的混音（mixture）都是即時（on the fly）合成。
  `puresound/task/` 底下的任務 dataset 都建立在這之上。
* **Kaldi-form**（`kaldi_base.py`，legacy）：預先混好的 `wav.scp` 風格清單，
  供已凍結（frozen）的 SV/TSE 專案使用。

完整 API 參考文件：`docs/dataset/`。
