# voice_isolate —— benchmarks

English version: [`README.md`](README.md)

這裡的東西全部進版控，而且**不含任何音訊**。定義（區段標記、window 檔、建置腳本）與測試紀錄
（逐 checkpoint 的成績單）放這裡；它們描述的音訊 payload 放在被忽略的 `../data_report/` 樹下。

這條界線就是分家的理由。這裡有一部分素材是公開的——LibriTTS、DNS-5、BUT ReverbDB、VOiCES、
RealMAN、ai-coustics Dawn Chorus——用這裡的腳本重建它們是預期用法。另一部分是不得外流的內部
田野素材。把兩者放在不同目錄，界線就由 `.gitignore` 強制執行，而不是靠記得；私有路徑底下也就
永遠不需要加負向規則。

| benchmark | 語料 | 音訊 | 進版控的內容 |
|---|---|---|---|
| [`field_test_vector/`](field_test_vector/) | **內部田野錄音** | 永不 | 區段標記、`windows.json`、`build_cases.py`、`RESULTS.md`、逐版本成績單 |
| [`qvf22/`](qvf22/) | **內部**，且附帶商用系統的輸出 | 永不 | `windows.json` |
| [`wer_sets/`](wer_sets/) | LibriTTS test-clean + DNS-5 + BUT ReverbDB（公開） | 可重建 | 四種殘響 regime 各自的重建方式 |
| [`dawn_chorus/`](dawn_chorus/) | `ai-coustics/dawn_chorus_en`（公開） | 外部託管 | 逐 checkpoint 的報告 |
| [`probes/`](probes/) | VOiCES、RealMAN（CC-BY） | 外部託管 | 支撐跨鏈結論的探針輸出 |
| [`full_gate/`](full_gate/) | 以上全部一次跑完 | — | 每個 checkpoint 一個檔：`../run_full_benchmark.sh` 的完整九關摘要，發版決定實際是根據它下的 |

**沒有任何單一 benchmark 能決定發版。** `dpcrn_v10` 在田野 benchmark 的冷啟動欄位居首，卻沒過
部署閘門；只讀其中一邊都會做出錯誤決定。`full_gate/` 的存在就是為了讓完整圖像留在一起。

## 換了合成鏈之後，哪些數字還算數

這九關有一半讀磁碟上固定的音訊，另一半在評測當下經由 `puresound` 的裝置鏈合成。
只有後者會隨著鏈的變動而移動；跨鏈把兩邊混在一起比，就是把資料的變動讀成模型的變動。

| 關 | 音訊來源 | 依賴合成鏈 |
|---|---|---|
| 1 真實成績單（QVF + 田野） | 真實錄音 | 否 |
| **2 in-domain SI-SDRi + buckets** | `eval_indomain.py`，逐 batch 合成 | **是** |
| **3-5 far-only 探針**（expand / high / boundary） | 同上 | **是** |
| 6 Dawn Chorus WER | 真實語料 | 否 |
| 7a / 7b / 8 WER sets | `build_wer_set.py` 預先建好，自己做卷積 | 否 |
| 9 turn-taking 成績單 | 磁碟上凍結的 set | 否——但 `build_turntaking_set.py` **依賴合成鏈**，重建這個 set 就會移動 |

**界線是 `9c56e02`（2026-08-18）**，它讓裝置鏈的類比路徑真正變成線性。原本有六個級
在滿刻度靜默飽和；修正後 9.0% 的 mixture 與 2.8% 的 target 逐點誤差超過峰值的 1%。
輸出位準幾乎沒動（RMS 中位數 −20.5 → −20.8 dBFS），抽樣序列完全沒動（同一個 seed
仍抽到同一批語料）——但第 2-5 關在界線兩側是**不同的考卷**。

`run_full_benchmark.sh` 會把 `chain=<sha>` 蓋在摘要標題上。**在這一節出現之前 commit
的紀錄全部早於這次修正**：標題裡沒有 `chain=` 的紀錄一律視為 `9c56e02` 之前，它的合成
關卡不可與修正後的 run 排名。真實錄音那幾關——部署閘門與 QVF 跨鏈牆都在那裡——不受
影響，一路往回都可比。


## 田野 benchmark

`field_test_vector/` 是決定部署問題的那一個：兩支手工標記的真實錄音、27 個 clip，用兩種**永不
平均**的模式評分——STREAM（連續錄音、近場錨在場，等同對話中的行為）與 COLD-START（每段切出來
從第零秒餵入，等同機器閒置時的行為）。成績與方法見
[`field_test_vector/RESULTS.md`](field_test_vector/RESULTS.md)。

重建 clip（需要私有錄音已放在 `../test_vector/`）：

```bash
uv run python egs/voice_isolate/benchmarks/field_test_vector/build_cases.py
```

評分一個 checkpoint：

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True uv run python egs/voice_isolate/scripts/eval_realcase.py \
    egs/voice_isolate/config/infer_dpcrn.yaml \
    --ckpt egs/voice_isolate/pretrained_ckpt/dpcrn_v8.ckpt \
    --cases-dir egs/voice_isolate/data_report/field_cases/test_vector_cases \
    --device cuda:0 --dry-blend 0.9
```

## 新增一筆紀錄

一筆紀錄就是評分器自己的輸出、未經編輯、以它所描述的 checkpoint 命名，放進該 benchmark 的
`records/`。紀錄很便宜、不含音訊，而且是這個 repo 裡唯一持久的證據——`EXPERIMENT_LOG.md` 是刻意
不進版控的，所以只存在於那裡的數字撐不過一次乾淨 clone。

## 新增一個 benchmark

先決定語料，因為它決定版面：公開語料附上建置腳本與重建說明；私有素材只放標記與紀錄，音訊放到
`../data_report/` 底下並加上忽略規則。若某個定義檔會帶到擷取檔名、裝置識別碼或時戳，提交前先改寫
該欄位——`field_test_vector/spans/` 就是去識別後的樣子。
