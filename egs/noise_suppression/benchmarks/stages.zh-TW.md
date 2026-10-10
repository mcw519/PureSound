# 閘門關卡：每一關是什麼、能決定什麼

English: [`stages.md`](stages.md)

驅動程式：[`../run_full_gate.sh`](../run_full_gate.sh)。怎麼執行、有哪些變數、彙整怎麼
運作：[`docs/usage/evaluation.zh-TW.md`](../../../docs/usage/evaluation.zh-TW.md)。紀錄格式見
[`README.zh-TW.md`](README.zh-TW.md)。

**gate** 關卡可以否決一次發版；**monitor** 關卡只報告、永遠不否決——要嘛是這個測試集
分辨不出我們各個 checkpoint 之間的差別，要嘛是這個數字存在的目的是對外可比，而不是對內
決策。一關的角色取決於它**量得到什麼**，不是我們多在乎它。

每一關都會把**未處理的混音**跟候選模型一起評分，讀的是配對後的差。驅動程式不帶
checkpoint 執行時只評那個基線——這就是「在還沒有模型可以擋之前，先確認閘門本身會動」的
做法。候選也可以是另一個系統產出的音訊目錄（`PRECOMPUTED_DIR=`）——這是把公開模型放到這條
軸上、而不只在它自己的集上比的方法；此時 preflight 與 RTF 會跳過。

## 各關卡

| # | 關卡 | 測試集（n） | 指標 | 角色 | 為什麼 |
|---|---|---|---|---|---|
| 0 | preflight | -- | 參數是否載入 | 中止 | recipe 與 checkpoint 不合會部分載入，然後照樣產出數字 |
| 1 | 凍結合成集 | `data_report/ns_testset`（500），`NS_TESTSET=` | PESQ-WB；STOI、ESTOI、SI-SDR、諧波間隙、暫態相關 | PESQ-WB 為 **gate**，其餘 monitor | 我們自己的合成、語者 held-out、涵蓋部署在乎的 SNR |
| 2 | VCTK-DEMAND | `data_report/vctk_demand_test`（824），`NS_VCTK=` | 同第 1 關 | PESQ-WB 為 **gate**，其餘 monitor | 真實錄音噪音、固定在磁碟上，也是多數文獻報告的那份集 |
| 3 | DNS-5 dev set | `data/dns5/dns5_devset.jsonl`（921），`NS_DEVSET=` | DNSMOS SIG / BAK / OVRL / P.808，依類別 | monitor | 沒有乾淨參考；passthrough 什麼都不做也能拿好分數 |
| 4 | WER | VCTK-DEMAND 切片（824），`NS_WERSET=` | 刪除；WER、插入、替換 | 刪除為 **gate**，其餘 monitor | 品質指標看不見的失敗：吃掉字 |
| 4b | WER，困難集 | `data_report/libritts_demand_hard`（500），`NS_WERSET_HARD=`；`SKIP_WER_HARD=1` 跳過 | 同第 4 關，依 SNR 與噪音類型分帶，保留假設字串 | 刪除為 **gate**，其餘 monitor | 字數夠多、句子夠長、SNR 夠低，解析得出 VCTK 切片解析不出的東西 |
| 5 | CPU real-time factor | 合成 | RTF，單執行緒 | **gate** | 即時性是硬性要求，不是取捨 |

`SKIP_WER=1` 跳過第 4 與 4b 關；第 4b 關在集合還沒建時也會跳過。此時紀錄裡不會有它們的
關卡，並會寫明。

### 1. 凍結合成集 —— gate

由 `puresound.evaluation.tools.build_eval_set` 用 recipe 自己的合成管線、固定 seed 產生，
所以音訊是釘死的，相隔數週評分的兩個 checkpoint 才可比。語者來自 metafile 的 valid
split，與訓練語者互斥。

```bash
python -m puresound.evaluation.tools.build_eval_set \
    egs/noise_suppression/config/eval/ns_testset.yaml \
    --out-dir egs/noise_suppression/data_report/ns_testset --n 500 --seed 1234
```

每一筆的 SNR 是從 `noisy - clean` **量出來的**，不是從那份要求它的 config 抄來的，所以
逐 SNR 帶的分解才有意義。一定要看那張分解表：一個系統可以在高 SNR 多賺 PESQ，同時在
0 dB 附近把語音毀掉，而兩者的平均什麼都沒報。

這一關是合成的，所以只有 `extra.set_provenance` 裡的建集 commit 與 recipe digest
相同時，數字才能互相比較。

### 2. VCTK-DEMAND —— gate

真實錄音噪音混 VCTK 語音，磁碟上本來就配對好，由
`puresound.dataset.corpus.vctk_demand testset` 匯入成 16 kHz。它給了兩件合成集給不了的東西。

它**對外可比**：VCTK-DEMAND 上的 PESQ 是多數噪音抑制論文報的那個數字，所以這裡的結果能
跟公開工作對讀，而不只是跟我們自己的前一版比。

它**不隨我們的合成而動**。第 1 關的音訊是這個 repo 的裝置鏈產生的，所以它的數字只在同一次
建集內可比；這一關是磁碟上固定的音訊，跨越我們做的每一項改動都能比。

它的 SNR 分佈比第 1 關溫和，這在這裡是優點：兩關取樣問題的不同一半，只在其中一半有效的
模型會在另一半失敗。

### 3. DNS-5 dev set —— monitor

真實 clip，沒有乾淨參考，用 DNSMOS P.835 評分。它是 monitor 有兩個理由：一個完全不失真的
passthrough 會因為什麼都沒做而拿到好分數；而且綜合品質預測器可能漏掉聽者或辨識器抓得到的
真實缺陷。

SIG 與 BAK 分開報，OVRL 絕不單獨讀——OVRL 分不出「把噪音拿掉」與「把語音拿掉」，而那正是
這個任務的兩個失敗方向。

逐噪音類別的分解才是這一關的重點。噪音抑制是分類別壞掉的：它可能在風扇上撐得住、在嬰兒
哭聲上垮掉。類別來自 dev set 自己的檔名。

### 4. WER —— 刪除為 gate

品質指標最補不上的那道護欄。一個把噪音連同部分語音一起拿掉的模型，可以守住 PESQ 而丟掉
整句話。

三條規則讓這個數字誠實：

**讀 delta，不讀絕對率。** 在不知道同一批切句上未處理混音的 WER 之前，任何 WER 都毫無
意義。兩邊都會轉寫，判定讀的是配對差。

**刪除單獨報**，因為它是有方向的那個失敗模式。替換與插入可能來自辨識器本身；對著固定參考
而上升的刪除率，就是模型在移除語音。

**參考是語料自己的逐字稿**，絕不是辨識器跑乾淨音訊的輸出——那量的是辨識器跟自己的一致性，
而且會偏袒聽起來最像它訓練資料的那個系統。

辨識器預設用大模型是刻意的：弱辨識器可能正好蓋住強辨識器揭露的過度抑制，而這一關的目的
是暴露它，不是求快。

**辨識器必須是音訊的函數。** faster-whisper 預設的解碼是一道 temperature fallback 階梯：
品質檢查沒過的 segment 會改用採樣重解，所以同一段音訊每次轉寫結果不同，配對閘門沒辦法
透過這種雜訊讀出小差距。`evaluation.tools.wer` 釘住 `--asr-temperature`（預設 0.0），
讓同一份音訊兩次跑出相同結果。

**VCTK 切片能分辨與不能分辨的。** 句子短、SNR 溫和、參考字數少，所以未處理混音本來就接近
辨識器的地板，而我們各版本之間差異的整個範圍，只是辨識器本身 run-to-run 擺動的小倍數。
在這份集上：

- `wer.wer` 與 `wer.sub` 是 **monitor**，不得用來排兩個版本的高下。兩顆 checkpoint 在這裡
  的差距可以統計上可解析，實務上卻只是每一千字多聽錯一兩個字——可解析不等於重要。
- `wer.del` 仍是 **gate**。刪除是品質指標看不見的失敗，會移除語音的模型會明確地把它推離
  地板。把它當一個護欄問題讀——「這一版會不會吃字？」——不要讀出別的。
- **排名屬於有進步空間的那幾關。** 同一個 STFT 上的 oracle 遮罩在 PESQ、STOI、DNSMOS 上都
  遠高於每一個模型，而這裡的 WER 已經接近地板。需要一個能排名的 ASR 數字的版本，需要的是
  更難的音訊，也就是第 4b 關。

`--band`（預設 `snr_band`）把逐句的錯誤率照第 1 關的方式分帶；這份集僅有的可用範圍在最低
SNR 那一帶。

### 4b. 困難集上的 WER —— 刪除為 gate

第 4 關要求的那份更難的音訊，角色相同：刪除為 gate，其餘 monitor。

```bash
python -m puresound.dataset.corpus.vctk_demand noise /path/to/audio/vctk_demand \
    --out-dir egs/noise_suppression/data/demand_noise
python -m puresound.evaluation.tools.mix_paired_set \
    --speech-dir /path/to/audio/LibriTTS/test-clean --transcript-suffix .normalized.txt \
    --noise-dir egs/noise_suppression/data/demand_noise \
    --snr -10 5 --min-duration 8 --n 500 --seed 1234 \
    --out-dir egs/noise_suppression/data_report/libritts_demand_hard
```

四個性質讓它解析得出 VCTK 切片解析不出的東西：

- **參考字數多出好幾倍**，固定數量的錯字佔雜訊的比例更小。
- **句子長**（至少 8 秒），一次刪除不會是整句的十分之一。
- **SNR 介於 −10 到 5 dB**，大部分在 0 dB 以下，那正是 VCTK-DEMAND 幾乎不到的範圍。
- **這裡沒有任何模型聽過的噪音**：VCTK-DEMAND 當初混進去的 DEMAND 噪音，以
  `noisy - clean` 還原——一個對所有用 DNS 訓練的模型都是 held-out 的噪音池。

刻意**不加殘響**：難度只有 SNR 一個軸，所以量到的某一帶就是那一帶，不是那一帶加一個房間。

這裡的解析度也有極限。它分得出「會吃字的系統」與「不會吃字的系統」，也分得開我們與公開
模型之間那種大小的差距；但像我們自己兩顆相鄰 checkpoint 那種大小的差距，仍可能落在區間
之內。要在本關排我們自己的兩顆 checkpoint，一定要先看配對區間與符號計數（變差對變好的
句數）。

**讀 delta 之前先修剪迴圈。** whisper 遇到解不開的片段會重複同一句，而一筆這種假設字串的
插入率遠高於語料平均。本關因此會逐系統印出迴圈計數（假設字串超過參考的 1.5 倍），閘門也
把每一句假設字串保留在 `4b_wer_hard_hyp.jsonl`。各系統的迴圈次數不同，所以未修剪的差距有
一部分是在排「誰的輸出比較會弄壞辨識器」。

`--band noise` 一定要開：語料層級的平均值會蓋掉傷害是否集中在與語音頻譜重疊的那幾種噪音。

兩道 WER 關卡不互相取代。VCTK 那關對外可比、解析度低；這一關有解析度，但沒有對外可比性。

### 5. CPU real-time factor —— gate

在 CPU 上、單執行緒、用合成音訊量。不是在 GPU 上量，也不是用多核機器的全部核心量：那兩
種數字都無法說明這東西能不能出貨。要在閒置的機器上量；load average 接近核心數時，扭曲的
不只是數字，連排名都會變。

預設預算（`RTF_BUDGET`，0.5）替裝置上其他所有東西留了餘裕。要調高就要有寫在紀錄裡的理由。

## 怎麼讀一份結果

1. **`no-resolution` 的 gate 關卡不算通過。** 那表示這個集分不出候選與「什麼都不做」。
   紀錄會保留 `no-resolution`，collect 也會回傳非零，因為它不能支持發版。
2. **delta 旁邊的絕對值要一起讀。** 從很糟的起點拉出很大的改善，不代表終點好。
3. **單一 checkpoint 不構成一次量測。** 相信版本之間的差異之前，先評分整個 run 末尾的
   一組 checkpoint。
4. **比對第 1 關之前先確認 `extra.set_provenance`**；比對執行時間量測則確認評分時的
   `chain_commit`。
