# 評測與閘門

English: [evaluation.md](evaluation.md)

`puresound.evaluation` 放的是 benchmark 協定、它的統計與它的紀錄。recipe 的閘門腳本只是
這些工具的清單；協定住在函式庫裡，這樣兩個 recipe 不會漂移成兩套協定。本頁說明噪音抑制
的閘門 `egs/noise_suppression/run_full_gate.sh`；每一關能決定什麼、不能決定什麼，論證在
[`egs/noise_suppression/benchmarks/stages.zh-TW.md`](../../egs/noise_suppression/benchmarks/stages.zh-TW.md)。
voice isolation recipe 則以同一批工具驅動自己的關卡清單
`egs/voice_isolate/run_full_benchmark.sh`；其中田野錄音的關卡使用不隨儲存庫散布的
私有錄音。

| 模組 | 說明 |
|---|---|
| `evaluation.statistics` | bootstrap 區間、配對比較、block 統計、判定 |
| `evaluation.records` | `StageResult` / `GateRecord`——benchmark 目錄保存的那份紀錄 |
| `evaluation.systems` | 一道關卡把音訊送進什麼：一顆 checkpoint、「什麼都不做」的基線（`Passthrough`），或另一個系統已產出的音訊目錄（`PrecomputedSystem`） |
| `evaluation.transcribers` | 本機 Whisper、Azure Speech、ElevenLabs 統一成一個呼叫（web playground 的逐字檢查用） |
| `evaluation.tools.preflight` | 證明 checkpoint 能完整載入閘門用到的每一份 recipe |
| `evaluation.tools.build_eval_set` | 用 recipe 自己的合成凍結一份 (mix, clean) 測試集 |
| `evaluation.tools.mix_paired_set` | 把帶字稿的語音資料夾與 held-out 噪音池混成 WER 集 |
| `evaluation.tools.reference` | 在凍結集上算 PESQ、STOI、ESTOI、SI-SDR 與兩項頻譜細節量，配對且分帶 |
| `evaluation.tools.noreference` | 沒有乾淨參考時用 DNSMOS P.835，並依 tag 分解 |
| `evaluation.tools.wer` | 對語料字稿算詞錯誤率，拆成刪除、插入、替換 |
| `evaluation.tools.rtf` | CPU real-time factor 對預算 |
| `evaluation.tools.collect` | 把各關卡檔案併成一份紀錄並回傳判定 |

每個工具都以 `python -m puresound.evaluation.tools.<name>` 執行；`--help` 列出旗標。

## 測試集只建一次

閘門的測試集建一次就凍結。它們放在 `data_report/` 底下（不進版控）；閘門從下列預設路徑
讀取，每條路徑都能用環境變數覆寫。

```bash
# 第 1 關：用 recipe 自己的合成建出的合成集                        (NS_TESTSET)
python -m puresound.evaluation.tools.build_eval_set \
    egs/noise_suppression/config/eval/ns_testset.yaml \
    --out-dir egs/noise_suppression/data_report/ns_testset --n 500 --seed 1234

# 第 2、4 關：VCTK-DEMAND，連同字稿一起匯入                         (NS_VCTK)
python -m puresound.dataset.corpus.vctk_demand testset /path/to/audio/vctk_demand \
    --out-dir egs/noise_suppression/data_report/vctk_demand_test

# 第 3 關：DNS-5 dev set 的 inventory                               (NS_DEVSET)
python -m puresound.dataset.corpus.dns_challenge devset /path/to/audio/dns-5 \
    --output-dir egs/noise_suppression/data/dns5

# 第 4b 關：困難 WER 集——LibriTTS 底下疊 held-out 的 DEMAND 噪音    (NS_WERSET_HARD)
python -m puresound.dataset.corpus.vctk_demand noise /path/to/audio/vctk_demand \
    --out-dir egs/noise_suppression/data/demand_noise
python -m puresound.evaluation.tools.mix_paired_set \
    --speech-dir /path/to/audio/LibriTTS/test-clean --transcript-suffix .normalized.txt \
    --noise-dir egs/noise_suppression/data/demand_noise \
    --snr -10 5 --min-duration 8 --n 500 --seed 1234 \
    --out-dir egs/noise_suppression/data_report/libritts_demand_hard
```

`build_eval_set` 從 `noisy - clean` 量出每個項目的 SNR，並把建集 commit、recipe SHA-256
與 seed 記進測試集的 provenance；`--min-active` 會丟掉乾淨目標幾乎無聲的項目。
`mix_paired_set` 接受 SNR 窗（`--snr LOW HIGH`）、最短與最長句長，噪音池則是一個檔案一條
音軌。

## 執行閘門

從 repo 根目錄執行。

```bash
# 什麼都不做的基線，每一關都要對照它來讀——不帶 checkpoint
bash egs/noise_suppression/run_full_gate.sh baseline

# 一顆 checkpoint，加上建它的 recipe
bash egs/noise_suppression/run_full_gate.sh <tag> <ckpt> [recipe]

# 另一個系統已產出的音訊，每個輸入一個檔，以輸入的 stem 命名
PRECOMPUTED_DIR=/path/to/enhanced bash egs/noise_suppression/run_full_gate.sh <tag>
```

`<tag>` 是紀錄的名字：寫到 `egs/noise_suppression/benchmarks/records/<tag>.json`。
`[recipe]` 預設是 `egs/noise_suppression/config/dpcrn.yaml`；請給 checkpoint 訓練時用的
recipe，已發布的 checkpoint 則給只含模型的 `config/infer_dpcrn.yaml`。checkpoint 與
`PRECOMPUTED_DIR` 是兩個系統；腳本拒絕兩者同時給。

| 變數 | 預設 | 作用 |
|---|---|---|
| `DEVICE` | `cpu` | 第 1、2、4、4b 關模型跑在哪個裝置 |
| `DRY_BLEND` | `1.0` | 推論操作點，以 `--dry-blend` 傳入並記為 `inference.dry_blend` |
| `SKIP_WER` | `0` | `1` 跳過第 4 與 4b 關 |
| `SKIP_WER_HARD` | `0` | `1` 跳過第 4b 關 |
| `NS_DATA` | `egs/noise_suppression/data/dns5` | 找 dev set inventory 的位置 |
| `NS_TESTSET` | `egs/noise_suppression/data_report/ns_testset` | 第 1 關的集合 |
| `NS_VCTK` | `egs/noise_suppression/data_report/vctk_demand_test` | 第 2 關的集合 |
| `NS_WERSET` | `$NS_VCTK` | 第 4 關的集合 |
| `NS_WERSET_HARD` | `egs/noise_suppression/data_report/libritts_demand_hard` | 第 4b 關的集合 |
| `NS_DEVSET` | `$NS_DATA/dns5_devset.jsonl` | 第 3 關的 inventory |
| `RTF_BUDGET` | `0.5` | 第 5 關的預算 |
| `GATE_JOBS` | 核心數的一半 | 第 1–3 關的評分 worker 行程數 |
| `ASR_MODEL` / `ASR_DEVICE` | `large-v3` / `cuda` | 第 4、4b 關的辨識器 |
| `GATE_OUT` | `$TMPDIR` 或 `/tmp` | 每次執行的 log 目錄放在哪裡 |
| `PRECOMPUTED_DIR` / `PRECOMPUTED_NAME` | 未設 / 目錄名稱 | 改評預先產出的音訊，而不是 checkpoint |

## 各關卡

| # | 關卡 | 集合 | 指標 | 角色 | 紀錄中的關卡名 |
|---|---|---|---|---|---|
| 0 | preflight | 閘門用到的每一份 recipe | 參數是否載入 | 中止 | -- |
| 1 | 凍結合成集 | `NS_TESTSET` | PESQ-WB；STOI、ESTOI、SI-SDR、諧波間隙、暫態相關；依 SNR 分帶 | PESQ-WB 為 **gate**，其餘 monitor | `frozen_testset` |
| 2 | VCTK-DEMAND | `NS_VCTK` | 同第 1 關 | PESQ-WB 為 **gate**，其餘 monitor | `vctk_demand` |
| 3 | DNS-5 dev set | `NS_DEVSET` | DNSMOS SIG、BAK、OVRL、P.808；依類別與裝置 | monitor | `dns_devset` |
| 4 | WER | `NS_WERSET` | 刪除；WER、插入、替換 | 刪除為 **gate**，其餘 monitor | `wer` |
| 4b | WER，困難集 | `NS_WERSET_HARD` | 同第 4 關，依 SNR 與噪音類型分帶，保留假設字串 | 刪除為 **gate**，其餘 monitor | `wer_hard` |
| 5 | CPU real-time factor | 合成音訊 | RTF，單執行緒 | 對 `RTF_BUDGET` 為 **gate** | `cpu_rtf` |

- **第 0 關**只在有 checkpoint 時執行。checkpoint 無法完整載入 recipe 時，閘門在第 1 關
  之前就停：部分未訓練的模型只會報出一個沒人解釋得了的退步。
- **第 3 關**不論 `DEVICE` 為何都在 CPU 上跑模型——dev set 的 clip 夠長，多個 worker 的
  GPU forward 會用完記憶體，而且模型在這一關只佔一小部分，大頭是 DNSMOS 自己的 session。
- **第 4 關**用辨識器轉寫混音與輸出，兩者都對語料字稿評分。`wer.del` 是 gate；
  `wer.wer`、`wer.ins`、`wer.sub` 以 monitor 報告，永遠不會否決發版。
- **第 4b 關**在 `SKIP_WER=1`、`SKIP_WER_HARD=1`，或 `$NS_WERSET_HARD/manifest.jsonl`
  不存在時跳過；log 會寫明原因。執行時會加上 `--band snr_band --band noise`，並把每一句
  假設字串寫到 log 目錄的 `4b_wer_hard_hyp.jsonl`。
- **第 5 關**在 `PRECOMPUTED_DIR` 時跳過——沒有模型可以計時。

接著把各關檔案併成一份紀錄，並要求這次執行應該產出的關卡都在：`cpu_rtf`（precomputed
除外），有 checkpoint 或 precomputed 音訊時再加上 `frozen_testset.pesq_wb`、
`vctk_demand.pesq_wb`、`dns_devset.dnsmos_ovr`、`wer.del`（`SKIP_WER=1` 除外）與
`wer_hard.del`（有跑第 4b 關時）。缺少必要關卡會讓彙整失敗。任一關的指令失敗、或紀錄的
判定不是 `pass` 時，腳本以非零碼結束，並印出 log 所在位置。

## 每一關都評兩個系統

候選模型，以及**什麼都不做**。絕對分數單獨看什麼都不代表：一個未處理混音本來就得分很高的
測試集根本沒有進步空間，而從很糟的起點拉出很大的改善也不代表終點好。判定讀的是對
`Passthrough` 的配對差。

這也表示整套 benchmark 在模型還不存在時就能跑——這正是閘門可以先建起來的原因：
`run_full_gate.sh baseline` 會印出每一關未處理的分數。

可以把第三個系統放到同一條軸上：先用任何別的模型離線跑過測試集的輸入檔，再把
`--precomputed DIR` 交給 `reference`、`noreference`、`wer`（或把 `PRECOMPUTED_DIR=DIR`
交給閘門）。檔案用輸入檔的 stem 對應，工具自己在輸出檔名加的後綴在不含糊時接受，兩個候選
就是錯誤而不是猜測。這就是把公開模型放到**我們的**測試集上跟我們比的方法，而不只是在它
自己的集上。

閘門是在一個**推論操作點**上給 checkpoint 打分，並把它記在 `inference` 欄位，讓任何數字都
不會脫離它量測時的旋鈕被讀。`reference`、`noreference`、`wer`、`rtf` 吃同一個旗標
`--dry-blend`（見 [`docs/architecture/system/siso.zh-TW.md`](../architecture/system/siso.zh-TW.md)）；
閘門腳本以 `DRY_BLEND=` 暴露。`SKIP_WER=1` 是給品質關卡迭代用的，不能用來決定發版：那樣
紀錄裡就沒有 WER 關卡，而 WER 正是品質關卡補不上的那道護欄。

## 三種「宣稱得比量到的多」

每一種在這裡都有對應的防線。

**測試集分辨不出那個差距。** 當區間蓋過 0，或區間沒有可讀的寬度（只有一筆資料、或端點
不是有限值）時，`verdict()` 回傳 `"no-resolution"`，`GateRecord.unresolved` 會列出哪些
gate 關卡是這個狀態。整體紀錄會保留 `no-resolution`，而 `collect` 會回傳非零；它不是退步，
但同樣不能支持發版。

```python
from puresound.evaluation.statistics import paired_bootstrap_ci, verdict

delta = paired_bootstrap_ci(enhanced_scores, unprocessed_scores)
decision = verdict(delta, direction="higher_is_better")   # pass / fail / no-resolution
```

`verdict(..., tolerance=x)` 讓一個 do-no-harm 關卡吸收整個區間都落在 `x` 以內的退步；它
永遠不會把分辨不出的區間變成 pass。WER 工具以 `--tolerance` 暴露它，作用在刪除上。

**把單一 checkpoint 當成一次量測。** 同一個 run 相鄰 epoch 在某個指標上的擺幅，可能比要
比較的版本差距還大。`block_summary()` 把一個 run 當成末尾一組 checkpoint 來評分，而且報
全距不報標準差——五個點撐不起一個標準差。

**沒有配對的比較。** 同一批項目會經過兩個系統，所以配對差的離散度只有單邊的一小部分。
把兩邊各自獨立重抽，報的是指標本身的離散度——那正是真實效果被誤判成雜訊的方式。

## 一份紀錄必須能跟另一份比

`GateRecord` 強制要求那些一旦缺少就會默默毀掉比較的欄位：`recipe`（recipe 與 checkpoint
不合會部分載入然後照樣產出數字）、評分時的 `chain_commit`、`inference` 操作點，以及每一關
的 `n` 與區間。凍結的合成關卡另外會在 `extra.set_provenance` 保存建集時的 commit、recipe
SHA-256 與 seed；辨識生成音訊的合成鏈要看這些值，而不是之後評分時的 commit。紀錄的格式
約定，包括每份紀錄歸檔時附上的 `lineage` 區塊，在
[`egs/noise_suppression/benchmarks/README.zh-TW.md`](../../egs/noise_suppression/benchmarks/README.zh-TW.md)。

`load_checkpoint_system` 在載入當下就強制第一項：**缺少**參數是錯誤而不是警告——而參數被
改名也正是以「缺少」的形式出現的。反過來，checkpoint 帶的東西比 recipe 建的多時，會被報告
並記進紀錄，因為音訊路徑仍然是完整的。

## 角色：gate 還是 monitor

**gate** 可以否決發版；**monitor** 只報告、永遠不否決——要嘛是這個集分辨不出我們各個
checkpoint 之間的差異，要嘛是這個數字存在的目的是對外可比而非對內決策。角色取決於這一關
**量得到什麼**，不是誰有多在乎它。

WER 關卡在同一關裡就示範了這個劃分。刪除是有方向的失敗——模型刪掉語音時，對固定的參考
它就會上升——也是品質指標看不到的那一種，所以 `del` 是 gate。WER、插入與替換跟著辨識器
動的程度不亞於跟著模型，兩顆 checkpoint 之間的差距可以在統計上分辨得出、實際上卻只是每千
字幾個字，所以它們是 monitor：要讀，但絕不拿來排名或否決。版本排名交給有進步空間的品質
關卡。

有 gate 關卡 fail 或分辨不出時 `collect` 會以非零碼結束，所以驅動腳本會跟著失敗。
