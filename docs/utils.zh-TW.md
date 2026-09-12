# puresound.utils

English version: [utils.md](utils.md)

給檔案 I/O、config 載入，以及 tensor 運算用的通用工具函式。

## Functions

### `str2bool(v: str) -> bool`

```python
def str2bool(v: str):
    return v.lower() in ("true", "yes")
```

只有（不分大小寫的）字串 `"true"` 跟 `"yes"` 會回傳 `True`。**其餘所有
情況——`"1"`、`"t"`、`"y"`、`"false"`、`"no"`、`"0"`、打錯字、空字串——都會
回傳 `False`。** 它絕對不會丟出例外；對輸入完全沒有做任何驗證。

---

### `str2list(s: str) -> List`

把一個以空白分隔的字串切成 Python list（`s.strip().split()`——任何一段
連續空白都算分隔符，不限單一個空格）。

---

### `load_text_as_dict(file_path: str, separator: str = " ", coding: str = "utf8") -> Dict[str, List[str]]`

把一個分隔符文字檔載入成 dict。每一行的第一欄變成 key；**剩下的欄位一律
變成字串組成的 list**——就算剩下只有一欄，值也是一個單元素的 list，不是
單純的字串（是 `{"aaa": ["bbb"]}`，不是 `{"aaa": "bbb"}`）。

**Parameters:**
- `file_path` – 文字檔路徑
- `separator` – 欄位分隔符（預設 `" "`）
- `coding` – 檔案編碼（預設 `"utf8"`）

---

### `recursive_read_folder(folder: str, file_type: str, output: Optional[List]) -> None`

```python
def recursive_read_folder(folder: str, file_type: str, output: Optional[List]) -> None:
```

儘管型別標注寫的是 `Optional[List]`，**`output` 其實沒有預設值**——它是
必要的 positional/keyword 參數。要傳入一個既有的 list；這個函式會
**原地修改它、然後回傳 `None`**：

```python
found = []
recursive_read_folder("corpus", ".flac", found)
# found 現在已經被填好資料了；函式本身的回傳值是 None
```

如果傳 `output=None`（或乾脆不傳），一旦找到符合的檔案就會丟出例外
（`None.append(...)`）。

比對方式是單純的**子字串測試**（`file_type in file`），不是後綴檢查——
`file_type=".wav"` 也會匹配到像 `"backup.wav.old"` 這樣的檔名。

---

### `load_hparam(file_path: str) -> Dict`

透過 `yaml.load_all`（不是 `yaml.load`）載入 YAML 檔——所以它支援一個檔案
裡有多個用 `---` 分隔的文件，會依序把每個文件的頂層 key 合併進同一個扁平
dict（後面文件的 key 會覆蓋前面文件裡相同的 key）。

---

### `create_folder(folder_name: str) -> None`

透過 `os.makedirs(folder_name, exist_ok=True)` 建立一個目錄（連同中間所有
必要的目錄），如果它還不存在的話。原始碼自己的 docstring 宣稱它「folder
不存在時會 raise FileExistsError」，但實作其實是把呼叫包在
`try/except FileExistsError: print(...)` 裡——**它實際上從不會真的
raise**；碰到 race-condition 的衝突只會被接住並印出訊息。

---

### `convolve(x: torch.Tensor, filter: torch.Tensor) -> torch.Tensor`

直接在時域上做的 1-D 卷積，左側補了 `len(filter) - 1` 個零樣本，讓輸出是
causal 的、且長度跟輸入一樣。

**Shapes**：`x` 是 `[1, T]`（單一 channel）、`filter` 是 `[K]`
（一個 1-D kernel）；回傳 `[1, T]`。

---

### `next_fast_len(size: int) -> int`

回傳大於等於 `size`、且質因數只有 2、3、5 的下一個整數（一個高效的 FFT
長度)——等同於 `scipy.fftpack.next_fast_len`。結果會依請求的 `size` 為
key，快取在一個 module 層級的 cache（`_NEXT_FAST_LEN`）裡。

---

### `fftconvolve(x: torch.Tensor, kernel: torch.Tensor, mode: str = "full") -> torch.Tensor`

基於 FFT 的卷積（透過 `torch.fft.rfft`/`irfft`，內部會先無條件進位到一個
快速 FFT 長度）。

- `"full"` – 長度 `len(x) + len(kernel) - 1`
- `"same"` – 長度 `max(len(x), len(kernel))`——是**比較長**那個輸入的
  長度，如果剛好是 `kernel` 比較長，就不一定等於 `len(x)`
- `"valid"` – 長度 `max(len(x), len(kernel)) - min(len(x), len(kernel)) +
  1`（只取兩個訊號完全重疊的區段）

原始碼裡留下來的使用備註：只要 RIR 的峰值不在第 0 個 sample，用 waveform
去跟它卷積就會引入一段傳播延遲——可以用 `rir.abs().argmax(dim=-1)` 找出
peak 位置，據此裁切。
