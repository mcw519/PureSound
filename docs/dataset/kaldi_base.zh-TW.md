# puresound.dataset.kaldi_base

English version: [kaldi_base.md](kaldi_base.md)

> **Status: legacy** — 維持可運作但已凍結：不新增功能、不重寫。供已凍結的
> SV（[task.sv](../task/sv.zh-TW.md)）與 TSE
> （[task.tse](../task/tse.zh-TW.md)）recipes 使用。

Kaldi 風格的 dataset 讀取器：讀的是一個裝有 `<uttid> <path>` manifest 檔案
的**資料夾**，而不是逐一傳入檔案路徑給 constructor。

## Class: `KaldiFormBaseDataset`

繼承 `torch.utils.data.Dataset`。

### Constructor

```python
KaldiFormBaseDataset(
    folder,
    resample_to: Optional[int] = None,
    mode: str = "train",
    audio_gain_normalized_to: Optional[int] = None,
    split_to_chunks_with_size: Optional[float] = None,
)
```

- `folder` – 一個資料夾，裡面預期要有寫死的 manifest 檔名（見下方
  [Manifest files](#manifest-files)）——**不是**個別的檔案路徑
- `resample_to` – 若有設定，所有開啟的音訊都會 resample 到這個取樣率
- `mode` – `"train"`（預設）/ `"dev"` / `"eval"`（其他值會讓 `assert`
  失敗）；`"eval"` 不會載入 clean reference，但會啟用切塊；
  `"train"`/`"dev"` 會載入 clean reference、且絕不切塊
- `audio_gain_normalized_to` – 目標 dBFS，會傳給
  `AudioIO.open(target_lvl=...)`
- `split_to_chunks_with_size` – 單位是**秒**，不是 samples（內部會乘上開啟
  檔案的取樣率：`chunk_length = int(sr * split_to_chunks_with_size)`）。只有
  在 `mode == "eval"` 時才會生效。

### Manifest files

`_folder_content` 寫死了兩個要在 `folder` 裡找的檔名：

| Key | 檔名 | 是否必要？ |
|---|---|---|
| `wav2scp` | `wav2scp.txt` | 是——找不到會丟出 `FileNotFoundError` |
| `wav2ref` | `wav2ref.txt` | 只有在 `mode != "eval"` 時才真的需要（見下方陷阱） |

兩者都是空白分隔的 `<uttid> <value>` 檔案，用
[`load_text_as_dict`](../utils.zh-TW.md) 載入。想加更多檔案（例如
`wav2enroll.txt`）要透過 `folder_content` 的 setter——見下方。

**wav2scp.txt**：
```
utt001 /path/to/noisy/utt001.wav
utt002 /path/to/noisy/utt002.wav
```

**wav2ref.txt**（clean reference，只有 train/dev 需要）：
```
utt001 /path/to/clean/utt001.wav
utt002 /path/to/clean/utt002.wav
```

**wav2enroll.txt**（可選的 enrollment/conditional speech——只有在透過
`folder_content` setter 加上 `"wav2enroll"` 之後才會被讀取）：
```
utt001 /path/to/enroll/spk001.wav
```

### `__getitem__(idx) -> Dict`

| Key | 內容 |
|---|---|
| `noisy_speech` | waveform tensor，永遠都有 |
| `clean_speech` | `mode != "eval"` 時從 `wav2ref` 開出來；否則是**`torch.empty(0)`**（不是 `None`） |
| `conditional_speech` | **只有**當 `wav2enroll` 這個 key 已經透過 `folder_content` 加入、且這一列剛好有對應資料時，才會從 `wav2enroll` 開出來；否則是**`torch.empty(0)`** |
| `sr` | 開啟的 noisy waveform 的取樣率 |
| `name` | uttid（manifest 的 key） |

還有兩個值得知道的行為：

- 如果 `_sr`（clean reference 原生的取樣率）跟 `sr`（noisy waveform 的取樣
  率）不一樣，reference 會透過 Sox backend resample 到 `sr`——只會印出一句
  `print()` 警告，不會丟例外。
- 在 `mode="eval"` 下，若有設定 `split_to_chunks_with_size`、而且這筆
  utterance 比算出來的 `chunk_length` 還長，`noisy_speech` 會用
  `torch.nn.functional.unfold` 重新切成 overlap 的區塊（50% hop：
  `stride=chunk_length // 2`），變成 2-D 的 `[n_chunks, chunk_length]`
  tensor，而不是 1-D waveform。`clean_speech`/`conditional_speech` 永遠不會
  被切塊。

**陷阱**：在 `mode="train"`/`"dev"` 下，`__getitem__` 是無條件讀取
`self.df[key]["wav2ref"]`——跟 `wav2enroll` 不同，這裡沒有先做存在性檢查。
如果 `folder` 裡沒有 `wav2ref.txt`，`_load_df` 只會在建構當下印一句警告；
真正的 `KeyError` 要等到第一次呼叫 `__getitem__` 才會冒出來。

### `folder_content` property — read-resets-to-default footgun

```python
@property
def folder_content(self):
    self._folder_content = {"wav2scp": "wav2scp.txt", "wav2ref": "wav2ref.txt"}
    return self._folder_content

@folder_content.setter
def folder_content(self, dct):
    self._folder_content.update(dct)
    self._load_df(self.folder)   # 立刻重新載入
```

**Setter** 是原本設計要拿來加 manifest 檔案的方式——它會把 `dct` 併入現有的
對應表，並重新載入 `self.df`：

```python
dataset.folder_content = {"wav2enroll": "wav2enroll.txt"}
```

但**getter 每次讀取都會把 `self._folder_content` 靜默重設**回寫死的
`{wav2scp, wav2ref}` 預設值——包括看起來人畜無害的讀取，例如只是想印出來
檢查目前設定：

```python
dataset.folder_content = {"wav2enroll": "wav2enroll.txt"}   # 現在追蹤 3 個檔案
print(dataset.folder_content)   # 印出 {wav2scp, wav2ref} -- wav2enroll 不見了
# 之後任何會再觸發 _load_df() 的程式路徑（例如再呼叫一次 setter）
# 重新載入時都不會帶著 wav2enroll。
```

這是真實存在、目前就是這樣運作的行為——不是排隊等修的 bug——所以要把這個
getter 當成「讀了就不安全」：一旦透過 setter 客製化過 `folder_content`，
就不要再讀這個 property 來檢視內容。

## Example

```python
from puresound.dataset.kaldi_base import KaldiFormBaseDataset
from torch.utils.data import DataLoader

# folder 裡必須有 wav2scp.txt（train/dev 模式還要加上 wav2ref.txt）
dataset = KaldiFormBaseDataset("data/test", mode="dev")
loader = DataLoader(dataset, batch_size=1, shuffle=False)
for batch in loader:
    noisy, clean = batch["noisy_speech"], batch["clean_speech"]

# 在預設檔案之上再加一個 enrollment/conditional-speech manifest
dataset.folder_content = {"wav2enroll": "wav2enroll.txt"}
```
