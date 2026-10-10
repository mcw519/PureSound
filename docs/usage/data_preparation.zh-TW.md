# 準備資料

English: [data_preparation.md](data_preparation.md)

語料準備在 `puresound.dataset.corpus`：把音訊掃描成 record、轉成單一取樣率、無洩漏地
切分，並寫出 recipe 要讀的 metafile。這些工具與語料無關——各語料之間的差異是一個參數
或一個小的語料模組，而不是在 `egs/` 底下再複製一份這些程式碼。

有命令列的模組一律以 `python -m puresound.dataset.corpus.<module>` 從 repo 根目錄執行，
`--help` 會列出完整旗標。下面的指令只列出重要的旗標；路徑是範例。

## 模組

| 模組 | 做什麼 | 命令列 |
|---|---|---|
| `records` | `AudioRecord`、七欄 metafile、JSONL inventory | 函式庫 |
| `scan` | 把資料夾掃成 record；語者互斥切分 | 函式庫 |
| `resample` | 把語料鏡射成固定取樣率的樹，只做一次 | 各語料指令上的 `--resample-*` 旗標 |
| `dns_challenge` | DNS Challenge 的乾淨語音、噪音與 dev set | `speech`、`noise`、`devset` |
| `vctk_demand` | VCTK-DEMAND 的測試集、訓練語音與還原出的噪音 | `testset`、`speech`、`noise` |
| `librilight` | LibriLight 章節：挑出最乾淨的，再切成片段 | `select`、`segment` |
| `kaldi` | Kaldi `wav.scp` / `utt2spk` 清單轉 metafile | 有 |
| `paired` | 把磁碟上已配對的 (noisy, clean) benchmark 寫成評分集格式 | 函式庫（`vctk_demand testset` 背後用它） |
| `concat_short` | 把同一語者的短句接成夠長、可訓練的片段 | 有 |
| `clean_speech` | 讓語音語料通過一顆 checkpoint，寫到另一棵樹 | 有 |
| `target_policy` | 逐檔決定訓練目標用清理後的還是原始的 | 有 |
| `noise_corpora` | FSD50K、CochlScene、MUSAN 轉成 16 kHz 噪音資料夾，依授權與人聲過濾 | 有 |
| `speech_screen` | 找出噪音資料夾裡聽得懂的人聲並移出去 | 有 |
| `ssn` | 從訓練語音合成語音形狀噪音 | 有 |
| `pool` | 把多份 metafile 併成一個訓練池，並報出 batch 比例 | `merge` |

## 兩個檔案，兩種職責

metafile 是七欄，形狀固定——那是 `DynamicBaseDataset` 在解析的東西：

```
uttid, spkid, gender, path, length, sample rate, channels
```

語料知道的其他一切（噪音類別、收音裝置、授權）放進 `AudioRecord.tags`，寫到旁邊的
**JSONL inventory**。把兩者分開，才能讓語料帶任意 metadata，而不必每個 parser 都長出
一個新欄位。

```python
from puresound.dataset.corpus import scan_folder, split_records, write_metafile

records = scan_folder("/corpus/read_speech", id_prefix="dns5",
                      speaker_pattern=r"reader_(\d+)")
train, valid = split_records(records, valid_ratio=0.05, split_by="speaker")
write_metafile("data/train.csv", train)
```

## 切分：這樣做擋掉了什麼洩漏

`split_by="speaker"` 是預設，因為只有這個模式的兩邊，才是一個 dev set 必須具備的互斥。
用 `"utterance"` 的話，每個 dev 語者同時也是 train 語者，dev loss 報的就是「模型把訓練
時聽過的聲音背得多熟」。記得呼叫 `assert_disjoint(train, valid)`——切分要被驗證，不是被
假設。

有兩種失敗模式被做成明確錯誤，而不是驚喜：

- 目錄型策略（`parent`、`grandparent`、`path-prefix`）套在沒有那層目錄的樹上會**丟出
  錯誤**。退回用檔名會讓每一句話都變成自己的語者，而在「每個語者只有一句」上做語者互斥
  切分，互斥只存在於名義上。扁平樹但檔名帶語者的，用 `speaker_pattern=`；真的每個檔案
  就是一個語者的，用 `speaker_id_strategy="per-file"`。
- `speaker_pattern` 沒有匹配到某個路徑時也會**丟出錯誤**，理由相同。

## utterance id：要唯一，還是要可解析

`utt_id_style="digest"`（預設）會接上相對路徑的雜湊，所以兩個語料永遠不會撞 id。
`"stem"` 則保留語料自己的檔名，撞到重複時丟出錯誤，而不是讓兩個檔案共用一個 id。

當下游會從 id 裡解析結構時就要用 `"stem"`：`egs/voice_isolate` 的
`build_chapter_corpus.py` 會從 `..._seg_2` 讀出 segment 編號，而 digest 會接在它後面，
所以該 recipe 的 setup 要帶 `--utt-id-style stem`。

每個語料指令也都接受 `--id-prefix`。前綴會出現在指令寫出的每個 `uttid` 與 `spkid` 開頭，
這是語料併池時彼此不相混的原因，也是 `trainer.speaker_source_weights` 比對的對象。

## 重取樣：只做一次，不是每個 epoch

語料指令接受 `--resample-to RATE`（搭配 `--resample-root DIR` 與 `--jobs N`）：音訊只轉
一次，寫進鏡射的樹，metafile 指向轉好的檔案。從 Python：

```python
from puresound.dataset.corpus import build_resampled_tree

converted, report = build_resampled_tree(
    records, source_root="/corpus", dest_root="/corpus_16k",
    target_sample_rate=16000, jobs=16,
)
print(report.summary())    # "N file(s): ... converted, ... reused, ... failed"
```

值得知道的三個性質：

- **與訓練同一個重取樣器。** 轉檔走 `AudioIO`，也就是合成管線用的那個。用別的重取樣器
  轉出的語料是一份略微不同的語料，而差異會落進模型，不會落進 log。
- **長度取自寫出的檔案**，絕不是把原始數字等比換算。
- **用行程，不用執行緒。** `torchaudio` 的 sox effects 依賴 libsox 的全域狀態，從多個執行緒
  呼叫會不定期讓直譯器 abort。`jobs=1` 直接 inline 執行，在 debugger 底下正需要這樣。

轉檔可續跑——已存在且非空的目的檔會被重用。轉檔失敗的檔案會從回傳的 record 中剔除並列
進報告，所以一個壞檔既不會中斷整個執行，也不會變成指向不存在檔案的 metafile 列。
兩個會鏡像到同一個目的檔的來源檔（同資料夾的 `a.wav` 與 `a.flac`）會在一開始就被拒絕，
否則第二個會被當成第一個已轉好的副本。

## 語音語料

### DNS Challenge

```bash
python -m puresound.dataset.corpus.dns_challenge speech <dns_root> --output-dir DIR \
    --subset read_speech --valid-ratio 0.05 \
    --resample-to 16000 --resample-root <dns_root>/datasets_fullband_16k/clean_fullband
python -m puresound.dataset.corpus.dns_challenge noise <dns_root> --output-dir DIR \
    --resample-to 16000 --resample-root <dns_root>/datasets_fullband_16k/noise_fullband
python -m puresound.dataset.corpus.dns_challenge devset <dns_root> --output-dir DIR
```

| 子指令 | 寫出 |
|---|---|
| `speech` | 從 `clean_fullband` 產出語者互斥的 `<id-prefix>_train.csv` / `<id-prefix>_valid.csv`（或用 `--train-metafile` / `--valid-metafile` 指定） |
| `noise` | `augmentation_noise` 要指向的資料夾（訓練取樣率），以及 `<id-prefix>_noise.jsonl` |
| `devset` | `<id-prefix>_devset.jsonl`：官方 dev set 的 inventory，每個 clip 標上噪音類別與收音裝置 |

語料的三個部分用法不同，所以是三個指令。噪音從不列在 metafile 裡——合成管線是從資料夾
抽噪音——所以 `noise` 的存在就是為了產出訓練取樣率的那個資料夾。

`speech` 的 `--subset` 每個 subset 給一次（省略則是整個 clean 資料夾），而且知道每個官方
subset 把語者放在哪裡：

- `read_speech` 是一個扁平目錄，檔名是 `book_..._reader_<id>_..._seg_N.wav`，所以語者從
  檔名取出。以 M-AILABS 為底的法、義、俄、德、西語 subset、`emotional_speech` 與
  `VocalSet_48kHz_mono` 各有自己的 pattern（`SUBSET_SPEAKER_PATTERNS`）；
  `vctk_wav48_silence_trimmed` 是一個語者一個目錄。表裡沒有的 subset 用
  `--speaker-pattern` 或 `--speaker-id-strategy` 補上。
- 德語 Spoken Wikipedia 部分的檔名是條目而不是朗讀者。一個條目就是一個朗讀者，所以拿它
  來切分是可靠的單位，但它的「語者數」不是真的語者數——這個來源要用比例加權
  （`trainer.speaker_source_weights`），絕不能照它看起來有多少語者來算。
- `vctk_wav48_silence_trimmed` 永遠不收 p232 與 p257，也就是 VoiceBank-DEMAND 的兩位測試
  語者，這樣 VCTK-DEMAND 那道閘門關卡才維持域外。`--exclude-speaker ID`（可重複）再排除
  更多。
- 同一 subset 內檔名與大小都重複的檔案會被丟掉，因為 DNS 把好幾個 subset 整份附了第二
  份；`--keep-duplicates` 則保留。

#### dev set 帶著唯一的逐類別軸

dev set 沒有乾淨參考——只能無參考評分。它有的是**每個檔名**裡的噪音類別與收音裝置，
`parse_devset_name` 會把它們解析出來。診斷需要這條軸：噪音抑制是逐類別失敗的，而類別間
的平均會把是哪一類藏起來。

標籤是人工寫的，所以同一個來源會以 `fan`、`fan_noise`、`fannoice` 出現。
`CATEGORY_ALIASES` 把明確的情況收斂，`category_raw` 保留原字串。表裡沒有的拼法保留它自己
正規化後的形式，而不是被硬併進鄰居：錯誤的合併比長尾更糟。沒有任何錨點可依的檔名回傳
`{}`——老實的空缺，而不是去猜裝置在哪裡結束、噪音從哪裡開始。

**訓練用**噪音完全沒有標籤——`noise_fullband` 的檔名是 AudioSet clip id。這裡不會發明
標籤；`scan_folder` 的 `tagger` 參數就是將來有標籤來源時的掛鉤。

### VCTK-DEMAND

```bash
python -m puresound.dataset.corpus.vctk_demand testset <root> --out-dir DIR
python -m puresound.dataset.corpus.vctk_demand speech  <root> --out-dir DIR --resample-to 16000
python -m puresound.dataset.corpus.vctk_demand noise   <root> --out-dir DIR
```

Valentini 的 VCTK + DEMAND 是大多數噪音抑制文獻報 PESQ 的集合，所以在它上面的結果可以
對照已發表的成果來讀，而不只是對照我們自己先前的 run。

`testset` 是**匯入**測試集而不是合成——語料本身就附上已配對的兩邊——並保留字稿，這讓
同一份集合能同時服務參考指標關卡與 WER 關卡。另外組一份 WER 集，回答的會是在略微不同的
切段上略微不同的問題。

`speech` 從乾淨的訓練半部建出 train/valid metafile。語句都在一個扁平目錄，語者寫在檔名裡
（`p232_001.wav`）；id 預設 `--id-prefix vctk --utt-id-style stem`，預設優先用 28 語者的
訓練切分，因為已發表的 baseline 就是在它上面訓練的。兩種訓練切分都與測試語者互斥。
VCTK 是 48 kHz：`--resample-to 16000` 會一次轉進同層的 `<root>_16k/` 樹（或用
`--resample-root` 指定）。

`noise` 還原 DEMAND 那一邊。語料沒有噪音資料夾，但兩半是逐樣本對齊的，所以
`noisy - clean` 就是噪音；子指令為每種噪音類型寫出一條 `<type>.wav`，類型取自語料附的
`log_*.txt`（唯一寫下類型的地方）。`--splits` 選測試切分、訓練切分或兩者；兩邊的噪音
類型不同。用 DNS 噪音訓練的模型一種都沒聽過，所以這是 held-out 噪音池：
`evaluation.tools.mix_paired_set` 用它建出困難 WER 集。不要拿它來訓練。

### LibriLight

```bash
python -m puresound.dataset.corpus.librilight select /path/to/audio/LibriLight \
    --hours 4000 --max-hours-per-speaker 4 \
    --exclude-speakers-in /path/to/audio/LibriTTS/test-clean \
    --out data/librilight/selection.jsonl
python -m puresound.dataset.corpus.librilight segment data/librilight/selection.jsonl \
    --source-root /path/to/audio/LibriLight \
    --dest-root /path/to/training_set/ns_speech/librilight_segments \
    --out data/librilight/ll_train.csv
```

LibriLight 是 LibriVox 有聲書，一個章節一個 16 kHz FLAC，旁邊的 JSON 記著朗讀者、
voice-activity 清單與估計的 SNR。

- `select` 依該 SNR 排序所有章節，從最乾淨的開始取，直到用完 `--hours` 小時的有聲音訊；
  `--max-hours-per-speaker` 讓少數多產的朗讀者不會佔掉語料的大半，另可用 `--min-snr` 設
  下限。沒有有限 SNR 的章節略過。`--exclude-speakers-in DIR`（一個內含 `<speaker>/` 子
  目錄的資料夾）指定的 held-out 集合，其朗讀者會最先被拒收：LibriLight 與 LibriSpeech、
  LibriTTS 共用 LibriVox 的朗讀者 id，而困難 WER 集就是 LibriTTS `test-clean`。
  `--exclude-speaker ID` 拒收單一朗讀者。
- `segment` 沿著 voice activity 把每個選中的章節切成 `--min-seconds` 到 `--max-seconds`
  的片段（超過 `--max-gap` 的停頓一定結束一段；兩側各保留 `--context` 秒的前後文），這樣
  一次幾秒的訓練裁切就不必解碼整個章節。語者寫成 `ll_<reader>`。

### Kaldi 格式清單

語料本來就附 `wav.scp` / `utt2spk`（以及選用的 `utt2gender`）時：

```bash
python -m puresound.dataset.corpus.kaldi data/train.csv \
    data/train_wav2scp.txt data/train_utt2spk.txt \
    --utt2gender_path data/train_utt2gender.txt \
    --insert_root_path /corpus/root --separator " "
```

| 參數 | 意義 |
|---|---|
| `output_path`（位置參數） | 要寫出的 CSV metafile |
| `wav2scp_path`（位置參數） | 每行 `<uttid> <wav_path>` |
| `utt2spk_path`（位置參數） | 每行 `<uttid> <spk_id>` |
| `--utt2gender_path` | 每行 `<uttid> <gender>`；省略時每列都是 `None`，sampler 視為性別未知 |
| `--separator` | **輸入**清單的欄位分隔符；輸出一律是逗號分隔 |
| `--insert_root_path` | 加在 `wav2scp` 讀到的每個路徑前面的前綴 |
| `--on-error {skip,raise}` | `skip`（預設）丟掉讀不出音訊的列，`raise` 則停止 |

`utt2spk` 裡沒有的列（有給 `utt2gender` 時，`utt2gender` 裡沒有的也算）會被丟掉，每一種
丟棄都會在摘要裡計數。轉換器一次寫一個檔，所以每個切分跑一次。`dataset.test_folder`
（recipe 的 `--scoring` / `--inference` 讀的）是另一種格式：一個目錄，裡面有
`wav2scp.txt`（`--scoring` 另需 `wav2ref.txt`），格式與此轉換器讀的輸入相同，都是
`<uttid> <path>`。

## 整理語音語料

### 接合短句

```bash
python -m puresound.dataset.corpus.concat_short data/vctk/vctk_train.csv \
    --source-root /path/to/audio/vctk_16k --dest-root /path/to/training_set/ns_speech/vctk_concat \
    --out data/vctk/vctk_concat_train.csv
```

recipe 會丟掉每一句短於 `dataset.filter_min_utterance_length` 的語句。對切成長段的語料
這是對的，但對由單句組成的語料，會默默刪掉大半。調低門檻會改到每個來源的資料列；接合則
只改需要的語料。同一語者的短句依 id 順序、以 `--gap` 秒靜音接起來，直到一段達到
`--min-seconds`（絕不超過 `--max-seconds`）。本來就夠長的語句原封不動通過，最後剩下
不夠長的一段照樣保留，交給 recipe 的門檻決定。接合出的列 uttid 以 `_cat<N>` 結尾，
inventory 的 `joined` tag 列出原句。

### 讓訓練目標通過一顆 checkpoint 清理

```bash
python -m puresound.dataset.corpus.clean_speech \
    --metafile data/dns5/dns5_train.csv \
    --source-root /path/to/audio/dns-5/datasets_fullband_16k \
    --dest-root /path/to/training_set/ns_speech_cleaned/dpcrn_mamba_v2/dns5 \
    --recipe egs/noise_suppression/config/infer_dpcrn.yaml \
    --ckpt egs/noise_suppression/pretrained_ckpt/dpcrn_mamba_v2.ckpt \
    --out-metafile data/dns5/dns5_train.v2clean.csv
```

「乾淨」語音語料並不乾淨：朗讀有聲書帶著房間底噪、交流聲與錄音器材的噪音底，而被訓練
去重現目標的模型也會把這些一起重現。`clean_speech` 讓每個檔案通過任何
`puresound.evaluation.systems` 載得了的 checkpoint，把輸出寫進**它自己的樹**——以清理器
命名、鏡射原始目錄結構——絕不放在原檔旁邊，這樣經過模型的目標不會被誤認為錄音，兩個清理
器的輸出也不會混在一起。它寫出一份指向清理後檔案的 metafile（預設：輸入 metafile 改成
`.clean.csv` 後綴）與一份統計檔（預設：輸出 metafile 改成 `.stats.jsonl` 後綴），每個檔案
記錄：

| 欄位 | 意義 |
|---|---|
| `removed_db` | `input - output` 相對於輸入的能量：清理器改了多少 |
| `floor_in_db` / `floor_out_db` | 清理前後最安靜 20 % 的 32 ms 音框的位準；兩者之差就是清理器拿掉的噪音底 |

實務重點：

- `--dry-blend` 預設 1.0，也就是模型自己的輸出。不要用匯出預設的 0.9 來清理，那會保留
  十分之一的輸入。
- 檔案依長度分批（`--max-batch-seconds`、`--max-batch-items`），只在尾端補零；模型是因果
  的，所以一個項目的輸出與它被分到哪一批無關。`--verify N`（預設 4）會在開跑前用真實檔案
  驗證這一點，若分批與單檔輸出不同就拒絕執行。TF32 會被關掉，讓輸出只取決於清理器與輸入。
- 可續跑：統計檔裡已有、且輸出存在的檔案會略過。`--limit N` 只清理前 N 列，供試跑。

### 決定用清理後的還是原始的目標

```bash
python -m puresound.dataset.corpus.target_policy \
    --original data/dns5/dns5_train.csv \
    --cleaned data/dns5/dns5_train.v2clean.csv \
    --policy floor-drop --min-floor-drop 3 \
    --out data/pool/dns5_train.csv
```

清理與採用清理後的檔案是兩個獨立的決定，因為清理器並不透明：連本來就乾淨的語音回來也會
有可量測的改變，而替換一個本來就沒有噪音可去的檔案，只會把清理器的染色寫進目標。

| `--policy` | 目標 |
|---|---|
| `all` | 每個檔案都用清理後的 |
| `none` | 每個檔案都用原始的（對照組） |
| `floor-drop` | 只在 `floor_in_db - floor_out_db >= --min-floor-drop` 時用清理後的——也就是真的有東西可清的地方 |

沒有清理版本、或在 `floor-drop` 下沒有統計的檔案，保留原始版本：缺少清理永遠不是弄丟一個
檔案的理由。原始檔分成好幾次清理時，`--cleaned`（與 `--stats`）可以重複給；統計檔預設是
每份清理後 metafile 旁邊的 `.stats.jsonl`。指令會印出被替換的目標比例。

## 噪音

噪音從不列在 metafile 裡。合成管線會（遞迴地）讀取資料夾下的每一個 `.wav`，所以一個噪音
來源就是一個訓練取樣率的資料夾：

| 來源 | 產生方式 |
|---|---|
| DNS Challenge 噪音 | `dns_challenge noise` |
| FSD50K、CochlScene、MUSAN | `noise_corpora`，再過 `speech_screen` |
| 語音形狀噪音 | `ssn` |
| DEMAND（held-out，只用於 WER 集） | `vctk_demand noise` |

### 公開噪音語料

```bash
python -m puresound.dataset.corpus.noise_corpora fsd50k /path/to/audio/FSD50K \
    --dest-root /path/to/training_set/ns_noise/fsd50k
python -m puresound.dataset.corpus.noise_corpora cochlscene /path/to/audio/CochlScene \
    --dest-root /path/to/training_set/ns_noise/cochlscene
python -m puresound.dataset.corpus.noise_corpora musan /path/to/audio/musan \
    --dest-root /path/to/training_set/ns_noise/musan
```

每個語料先連同決定 clip 能否使用的 tag 一起列出、過濾，再用共用的重取樣器轉成 16 kHz。
指令會印出保留幾個 clip，以及其餘各因為什麼被剔除；`--dry-run` 印完這份統計就停。過濾
的項目：

- **授權。** FSD50K 逐 clip 授權；只保留 CC0 與 CC BY，因為商用模型不能用 NC 與
  Sampling+ 的 clip 訓練。
- **有標註的人聲。** 噪音抑制會保留輸入中的每一個人聲，所以一段內容是有人在講話的噪音
  clip 教的是刪除。FSD50K 中標在 AudioSet 人聲子樹任何位置的 clip 都會被丟掉。群眾與
  交談聲是刻意保留的——聽不出字的 babble 正是模型必須處理的餐廳與車站噪音——並和所有
  場景錄音一樣過一次 `speech_screen`，抓出標籤漏掉的講話者。
- **音樂裡的人聲。** MUSAN 的標註會標出哪些音樂曲目有人聲；這些丟掉。MUSAN 的語音部分
  是 LibriVox，也就是 LibriTTS 測試語者的來源，永遠不用。
- **過短的 clip**（`--min-seconds`，預設 2）：管線會把 clip 重複鋪滿整列長度，一聲短促的
  敲擊鋪滿一整列就成了節拍器，不是噪音。

它會在資料夾旁寫出帶 tag 的 `<folder>.inventory.jsonl`，若有需要署名的授權，另寫
`<folder>.attribution.jsonl`。

### 篩掉聽得懂的人聲

```bash
python -m puresound.dataset.corpus.speech_screen /path/to/training_set/ns_noise/cochlscene \
    --out /path/to/training_set/ns_noise/cochlscene.screen.jsonl --device cuda
python -m puresound.dataset.corpus.speech_screen /path/to/training_set/ns_noise/cochlscene \
    --out /path/to/training_set/ns_noise/cochlscene.screen.jsonl \
    --apply --rejected-root /path/to/training_set/ns_noise_rejected/cochlscene
```

界線畫在**聽得懂**，分兩道：Silero VAD 標出任何像人聲的地方，再由 faster-whisper
（`--whisper-model`，預設 `large-v3`）只轉寫那些區段。whisper 既認為有語音、又對至少幾個
字有信心時，這個檔案才算有人聲；babble 過不了信心那一關，清楚的講話者會過。第一個指令為
每個檔案寫一列 JSONL 數字（依路徑可續跑），所以門檻事後還能調整。`--apply` 接著把判定為
聽得懂的檔案**移到** `--rejected-root` 那棵樹——是移動不是刪除，這個判斷才能被稽核。
需要 `silero-vad` 與 `faster-whisper` 套件。

### 語音形狀噪音

```bash
python -m puresound.dataset.corpus.ssn data/dns5/dns5_train.csv \
    --dest /path/to/training_set/ns_noise/ssn --clips 300 --seconds 30
```

語音形狀噪音具有語音的長時頻譜，所以在遮罩能分離它的每一處都與語音重疊，而沒有任何錄音
語料提供它。每個 clip 取 `--utts-per-clip` 句隨機語句的長時頻譜（每個 clip 重抽一次），
套到白噪音上；每隔一個 clip 還會再乘上另一句語音的包絡，加入穩態遮罩太容易處理掉的音節
速率起伏。只能用**訓練** metafile 來建，絕不用測試集。`<dest>.manifest.jsonl` 記錄每個
clip 的來源與 seed。

## 合併語音語料

```bash
python -m puresound.dataset.corpus.pool merge \
    --source data/dns5/dns5_train.csv \
    --source data/vctk/vctk_train.csv \
    --out data/pool/pool_train.csv
```

recipe 只指定一個 `train_metafile`，所以用多個語料訓練就是一次合併。這之所以是一個模組，
是因為那份報告：**sampler 是均勻地抽語者**，所以一個語料佔 batch 的比例是它的*語者數*
佔整個池的比例，而不是它的時數；指令會逐來源印出這個比例：

```
  dns5_train     <spk> spk  <utt> utt  <hours> h  -> <share> of batches
  vctk_train     <spk> spk  <utt> utt  <hours> h  -> <share> of batches
```

- `--max-speakers STEM=N` 以丟掉整個語者的方式替某個來源（以 metafile 的 stem 指名）設
  上限——語者正是 sampler 運作的單位。改成限制列數會改變時數，batch 比例卻原地不動。
- `--exclude-speakers FILE`（可重複）列出任何來源都不准貢獻的 spkid，一行一個——例如以
  聲紋比對到測試集的朗讀者。
- 合併會拒絕同一個 `uttid` 或 `spkid` 出現在兩個來源：前者是 dataset 永遠抽不到的檔案，
  後者會把兩個語料的語者融成同一個 sampler 類別。各語料指令的 `--id-prefix` 就是用來避免
  這兩件事的。
- 每一列的音檔都會讀一次，全為零、空檔或讀不開的檔案會被丟掉，並逐來源印出數量。coverage
  sampler 會直接指定 dataset 要載入哪一句，而 dataset 遇到這種檔案會報錯、不會改抽別句，
  所以池裡只要留一個，訓練跑了幾個小時後就會停掉。光是 DNS-5 德文 Wikipedia 就有 88 個全
  零檔。`--jobs N` 設定讀檔的行程數（預設為核心數的一半；185 萬個檔案用 22 個行程約 9 分
  鐘）；`--skip-audio-check` 可略過這一步。若篩選後沒有任何列，合併會失敗，不會覆蓋輸出
  metafile。

若要自己決定比例，而不是讓語者數決定，設定 `trainer.speaker_source_weights`（見下）。

## 這些工具餵給哪些 recipe 旋鈕

### 語音：`dataset` 與 `trainer.speaker_source_weights`

```yaml
dataset:
  train_metafile: data/pool/pool_train.csv
  valid_metafile: data/pool/pool_valid.csv

trainer:
  speaker_source_weights: {dns5_: 0.6, ll_: 0.3, vctk_: 0.1}
```

`speaker_source_weights` 把一個 **spkid 前綴** 對應到一個權重。訓練時每個 batch 位置先依
權重抽一個來源，再在其中均勻抽一位語者；沒有這個區塊時每位語者機率相同，所以一個語料的
比例就是它的語者比例。以下規則都在建 sampler 時檢查：

- 每位語者必須恰好落在一個前綴下；有多個符合時取最長的前綴。沒有落在任何前綴下的語者、
  或沒有匹配任何語者的前綴，都是錯誤。
- 權重必須為正，會被正規化；log 會印出每個來源的語者數與佔 batch 位置的比例。
- 不能與「先依取樣率選」的模式併用（也就是沒有設 `dataset.target_sample_rate` 的 recipe）。
- 驗證仍對自己的 metafile 均勻抽樣，所以無論訓練比例怎麼調，驗證 loss 量的都是同一件事。

### 噪音：`augmentation_noise.noise_folder` 或 `noise_sources`

```yaml
augmentation_noise:
  used: True
  prob: 0.9
  noise_sources:
    - {name: dns5,   folder: /path/to/audio/dns-5/datasets_fullband_16k/noise_fullband, weight: 0.6}
    - {name: fsd50k, folder: /path/to/training_set/ns_noise/fsd50k, weight: 0.3}
    - {name: ssn,    folder: /path/to/training_set/ns_noise/ssn, weight: 0.1}
  snr_range: [-5, 40]
  prob_white_noise: 0.05
  white_noise_snr_range: [10, 30]
```

`noise_folder` 指定一個資料夾，裡面每個檔案機率相同，所以一個語料的比例就是它的檔案數。
`noise_sources` 以指定比例給多個資料夾：每次抽取先依 `weight`（預設 1.0）選來源，再在其中
均勻選一個檔案。兩者擇一，不能同時給。來源名稱必須唯一，沒有任何 `.wav` 的來源資料夾是
錯誤。檔案以「來源名稱加檔名」為鍵，所以兩個語料重用同一個檔名也不會互相覆蓋。

### SNR：`augmentation_noise.snr_bands`

```yaml
augmentation_noise:
  snr_range: [-5, 40]
  snr_bands:
    - {low: -5, high: 5,  prob: 0.4}
    - {low: 5,  high: 15, prob: 0.4}
    - {low: 15, high: 40, prob: 0.2}
```

沒有 `snr_bands` 時，SNR 在 `snr_range` 上均勻分布。有的話，先依 `prob` 選一個帶，再在帶內
均勻抽 SNR——一個分段均勻分布，讓範圍可以延伸到高 SNR，又不會稀釋困難的混音。
`snr_range` 仍是外框：每個帶都必須落在它裡面，每個帶都要 `high > low`，機率總和必須是 1。
