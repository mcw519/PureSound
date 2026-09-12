# puresound.dataset.parser

English version: [parser.md](parser.md)

給 dataset 建構用的 CSV metafile 解析器。

## Class: `MetafileParser`

**不是一般意義下可以 instantiate 的 class**——`MetafileParser` 完全沒有
`__init__`；每個 method 都是 `@staticmethod`。要直接在 class 上呼叫 method：
`MetafileParser.read_from_metafile(...)`，絕對不要寫成
`MetafileParser(...)`。

### Metafile format

逗號分隔，7 個依位置決定的欄位（不需要欄位名稱）：

```
uttid, spkid, gender, path, length, sr, channels
116-288045-0000, 116, M, dev-other/116/288045/116-288045-0000.flac, 170400, 16000, 1
```

- Header 列是**可選的**，且不管出現在哪裡都會被自動偵測並跳過（不只限於
  第一行）：只要某一列的第 0 欄是 `uttid`、第 1 欄是 `spkid`、第 4 欄是
  `length`（不分大小寫、去除空白後比對），就會被當成 header 跳過。
- 空白列會被跳過。
- 其餘每一列都必須剛好有預期的欄位數，否則 `read_from_metafile` 會丟出
  `ValueError`，並指出是哪一行出的問題。
- `length`、`sr`、`channels` 讀出來都是**字串**，不是 int——parser 內部完全
  沒有做型別轉換（像
  [`dynamic_base.gen_meta()`](dynamic_base.zh-TW.md) 這種呼叫端，使用前都會
  明確地 `float(...)` 轉型）。

兩個可選的尾端欄位可以延伸這一列：

```
uttid, spkid, gender, path, length, sr, channels, label_path
uttid, spkid, gender, path, length, sr, channels, label_path, start_time
```

**`with_start_time_column=True` 必須搭配 `with_label_column=True`。**
欄位數檢查是各自獨立地依每個 flag 各加一欄，但列拆解的分支只有在
`with_label_column` 也是 `True` 時才會去讀 `start_time`。單獨設定
`with_start_time_column=True` 會讓每一列都丟出「too many values to
unpack」的 `ValueError`（欄位數檢查預期有 8 欄，但 `not with_label_column`
分支裡的 7 個名稱解構還是照樣執行）。如果需要 start-time 欄位，請務必兩個
flag 一起傳。

### Methods

#### `read_from_metafile(f_path, use_speaker_as_key=False, insert_corpus_root_path=None, with_label_column=False, with_start_time_column=False) -> Dict`

- `f_path` – CSV metafile 路徑
- `use_speaker_as_key` – 會改變回傳的形狀（見下方）
- `insert_corpus_root_path` – 若有設定，會透過 `os.path.join` 加到 `path`
  前面（`with_label_column` 時也會加到 `label_path` 前面）
- `with_label_column` / `with_start_time_column` – 見上方說明

**`use_speaker_as_key=False`**（預設）——以 `uttid` 為 key 的扁平 dict：

```python
{
    "utt001": {"spkid": ..., "gender": ..., "path": ..., "length": ...,
               "sr": ..., "channels": ..., # 若有要求，還會有 "label_path"/"start_time"
               },
    ...
}
```

**`use_speaker_as_key=True`**——以 `spkid` 為 key 的巢狀 dict，
`gender`/`channels` 記錄的是該 speaker 遇到的第一筆 utterance 的值：

```python
{
    "spk001": {
        "gender": ...,
        "channels": ...,
        "utts": {
            "utt001": {"path": ..., "length": ..., "channels": ..., "sr": ...,
                       # 若有要求，還會有 "label_path"/"start_time"
                       },
            ...
        },
    },
    ...
}
```

[`dynamic_base.gen_meta()`](dynamic_base.zh-TW.md) 用的就是這個模式。

---

#### `create_scp_files(metafile_path, out_folder, insert_corpus_root_path=None, with_label_column=False, with_start_time_column=False, rename_uttid=False, add_prefix=None) -> None`

parser → [`kaldi_base`](kaldi_base.md) 的橋接函式：用
`read_from_metafile(use_speaker_as_key=False, ...)` 讀進 metafile，再把它
寫成 Kaldi 風格、空白分隔的 manifest 檔案到 `out_folder`：

| 一定會寫出 | 只有 `with_label_column` 時才會寫 | 只有 `with_start_time_column` 時才會寫 |
|---|---|---|
| `wav2scp.txt`、`wav2spk.txt`、`wav2gender.txt`、`wav2duration.txt` | `wav2label.txt` | `wav2start.txt` |

- `wav2duration.txt` 存的是 `length / sr`，單位**秒**（這裡現場算出來的，
  不是從 metafile 直接抄過來）
- `rename_uttid=True` 會把每個 uttid 換成補零的流水號（`00000`、
  `00001`、……）；`add_prefix`（在改名之後才套用）會在每個 uttid 前面加上
  `{prefix}_`
- **不會寫出 `wav2ref.txt`。** 光是 `wav2scp.txt` 就滿足
  [`KaldiFormBaseDataset`](kaldi_base.zh-TW.md) 唯一必要的檔案，但如果你
  要用 `mode="train"`/`"dev"`（會讀 `wav2ref`），得自己另外準備或改名成
  `wav2ref.txt`——這個函式不會幫你產生。

## Example

```python
from puresound.dataset.parser import MetafileParser

metadata = MetafileParser.read_from_metafile(
    "train_meta.csv",
    insert_corpus_root_path="/data/librispeech",
)
# metadata["utt001"] = {
#     "spkid": "spk001", "gender": "m",
#     "path": "/data/librispeech/audio/utt001.wav",
#     "length": "32000", "sr": "16000", "channels": "1",
# }

MetafileParser.create_scp_files("train_meta.csv", out_folder="data/train")
```
