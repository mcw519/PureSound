# 儲存庫佈局

English: [repository_layout.md](repository_layout.md)

## 邏輯放在函式庫，recipe 保持很薄

`egs/<task>/` 是**驅動層**。它指定語料、recipe 與一串評測關卡，但不實作它們。
一個 recipe 目錄只放：

| 檔案 | 是什麼 |
| --- | --- |
| `main.py` | 建好 task dataset，交給 `puresound.system.runner` |
| `config/*.yaml`、`config/eval/*.yaml` | 已發布的訓練、推論與評測 recipe |
| `run_full_gate.sh` | 評測關卡清單 —— 只有呼叫與彙總，沒有邏輯 |
| `README.md`、`README.zh-TW.md` | 這個 recipe 怎麼跑 |
| `pretrained_ckpt/` | 發布的 checkpoint、ONNX 產物，以及逐版判定表 |
| `benchmarks/` | 評測定義與結果紀錄 —— 只有數字，不放音訊 |

其他含有邏輯的東西，都屬於某個函式庫模組：

| 職責 | 模組 |
| --- | --- |
| 語料準備與 manifest | `puresound.dataset.corpus` |
| 合成與資料增強 | `puresound.audio`、`puresound.task` |
| 訓練驅動、DDP、各 stage | `puresound.system.runner` |
| 評測協定、統計、紀錄 | `puresound.evaluation` |
| 串流匯出與 runtime | `puresound.streaming` |
| 發布模型註冊與推論 | `puresound.inference` |

**當第二個 recipe 需要第一個 recipe 已有的工具時，那個工具要搬進 `puresound/`，
兩邊都用 import 的，絕不複製一份。** 一個 task 專屬的 `scripts/` Python 目錄，
正是一個儲存庫最後會把同一套評測維護兩遍、然後在下一個版本才發現兩邊結果不一致的
原因。`egs/voice_isolate/scripts/` 早於這條規則，是磁碟上唯一的例外；它採按需搬遷，
哪一支被另一個 recipe 用到就搬哪一支。

`puresound.cli` 是 model zoo 與推論的介面（`puresound models`、`puresound infer`、
`puresound web`）。import 它不會載入 Torch；只有指令讀寫音檔時才會載入。訓練與評測
工具是函式庫模組，用 `python -m puresound.evaluation.tools.<name>` 執行，不掛在那支
CLI 底下。

## run 的命名，與 release 的命名

run 目錄和 catalog id 是給人在數個月後、在幾十個同輩旁邊讀的。兩者都要寫出**實際訓練的
那個架構**，不是它出身的家族。

**run 目錄** —— `exp/<task>_<arch>_<variable>_<stage>`，例如
`ns_dpcrn-mamba_activebin_s2`。

| 區段 | 理由 |
| --- | --- |
| `<task>` | 不同 recipe 的 run 目錄最後會並排在一起——共用的 `exp/`、封存區、結果表——沒有這段的話，降噪的 run 和語音分離的 run 無法分辨 |
| `<arch>` | 骨架**以及與骨架不同的地方**。降噪這些模型是 DPCRN 把時間路徑換成 ERB 帶上的 Mamba；叫它 `dpcrn` 就是謊報訓練了什麼 |
| `<variable>` | 這次 run 在測什麼——那正是純版號從來不帶的資訊。`v3_s2` 什麼都沒說，`activebin_s2` 說了改了什麼 |
| `<stage>` | 課程階梯用 `s0`..`sN`，沒有階梯就省略；微調探針用 `ft` |

**Catalog id** —— `<task>-<arch>-<version>`，例如 `noise-suppression-dpcrn-mamba-v0`。
zoo id 是使用者會打進去、會釘在自己程式裡的東西，所以它跟 run 目錄不同：保留一個單純
可排序的版號，把意義放在 `display_name` 與 `description`。架構那一段遵守同一條準確性規則。
**改名是破壞性變更**：把舊寫法加進 `puresound/inference/zoo.py` 的 alias 表並用測試釘住，
不要期待呼叫端自己遷移。

## 發布產物與實驗產物

分界的判準是：別人 clone 這個 repo，需不需要這個檔案才能重現一個已發布的模型。

| 留 | 不留 |
| --- | --- |
| `puresound/`、`test/`、`docs/`、`sdk/` | `exp/` —— run、記錄、中間 checkpoint |
| `main.py`、`config/*.yaml`、`config/eval/*.yaml` | `config/exp/` —— 實驗 recipe |
| `run_full_gate.sh`、recipe README | `data/` —— 語料、metafile、list（本機絕對路徑） |
| `benchmarks/` —— 紀錄，零音訊 | `data_report/` —— 評測音訊 |
| 發布的 ONNX 產物 + `model_zoo/catalog.yaml` | `proc/`、`dummy_samples/`、`backup/` |
| `pretrained_ckpt/README.md` —— 判定表 | `pretrained_ckpt/*.ckpt` —— 直到某一版被升上去 |

已發布的 recipe 是從 `config/exp/` 升上來的；已發布的 checkpoint 是從 run 目錄升上來
的，並且與支撐它的 catalog 條目和判定表那一列一起。每一版憑什麼被判定，看判定表；
實驗日誌不屬於這個儲存庫（見下）。

紀錄只帶數字，不帶算出這些數字的音訊，也不帶任何由私有錄音衍生的材料。這就是為什麼
`egs/noise_suppression/benchmarks/` 留、`egs/voice_isolate/benchmarks/` 不留：前者評的
是公開語料，後者使用不對外散布的私有錄音。

## 註解、文件與測試

**註解與 docstring** 說明程式做什麼、為什麼這樣設計、假設了什麼或處理不了什麼。
不寫日期、版本或 run 名稱（`v8`、`ns_dpcrn-mamba_*_s1`）、實驗結果（指標數值、
前後對比、「在某某上量到」）、歷史（「以前會」、「原本是」），也不引用實驗日誌。
設計決定若建立在實驗上，註解用文字寫出決定與理由；讀者需要證據時，指向固定住該行為
的測試或論文。屬於設計本身性質的數字——由算術推得的上限、緩衝區大小、取樣率——保留。

**實驗日誌**——逐次 run 的結果、日期、版本之間的比較、每個實驗試了什麼又排除了什麼
——放在公開儲存庫之外。程式、文件與 recipe README 都不指向它們。

**文件**（`docs/`）維持英文與繁體中文兩份，同一個 commit 一起改，分成三部分：

| 部分 | 內容 |
| --- | --- |
| `docs/architecture/` | 各套件如何組合；資料如何流經語料準備、合成、訓練、評估、匯出與部署；設定系統與 model zoo |
| `docs/algorithms/` | 每個模型、loss、合成與增強階段、RIR 產生器、指標與統計在算什麼、為什麼 |
| `docs/usage/` | 準備資料、訓練與課程階梯、跑閘門、匯出與部署、串流 SDK、web playground |

**測試**對應套件結構（`test/<package>/test_<module>.py`）。一個測試釘住呼叫端依賴的
一個行為：輸出契約、已出貨的路徑、必須大聲失敗的錯誤。同一個行為的多個情境寫成一個
參數化測試；只為重現某一次實驗設定的測試不留。超過幾秒的測試標成 `slow`。

**凍結的 legacy**——`puresound/task/tse.py`、`puresound/task/sv.py`、
`puresound/system/miso.py`、`puresound/dataset/kaldi_base.py` 與它們的 recipe——
不照這些規則改寫；文件只標明它們的狀態。

Web SDK 位於 `sdk/web/`，包含 TypeScript 原始碼、固定依賴版本的 npm lockfile，
以及建置與測試工具。`node_modules` 與 SDK 編譯產物由本機產生。`npm run assets` 把編譯後的
runtime、ONNX Runtime Web 與已發布模型的裝置版本連同雜湊目錄寫入
`puresound/web/static/device/`，旁邊是納入版控、供試聽台 *這台裝置* 使用的 `device.js`、
`worker.js` 與第三方授權聲明；建置 Python 套件時，package data 會一併包含這些檔案。
