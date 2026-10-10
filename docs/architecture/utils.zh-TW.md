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

每個元素是一個 `"<檔名> <完整路徑>"` 字串（scp 風格的一行），不是單純的路徑。

如果傳 `output=None`（或乾脆不傳），一旦找到符合的檔案就會丟出例外
（`None.append(...)`）。

比對方式是單純的**子字串測試**（`file_type in file`），不是後綴檢查——
`file_type=".wav"` 也會匹配到像 `"backup.wav.old"` 這樣的檔名。

`iter_files_recursive(folder, file_type)` 是同一個遍歷（深度優先、依 `os.listdir`
順序、同樣的子字串比對），以 generator 產出 `(檔名, 完整路徑)` 配對。檔名或目錄可能含
空白時請用它，因為 scp 風格的一行無法再被正確拆開。

---

### `load_hparam(file_path: str) -> Dict`

透過 `yaml.safe_load_all` 載入 YAML 檔——所以它支援一個檔案裡有多個用 `---`
分隔的文件，會依序把每個文件的頂層 key 合併進同一個扁平 dict（後面文件的 key 會覆蓋
前面文件裡相同的 key）。空文件會被略過。檔案以 UTF-8 讀取，回傳時即關閉。Python 專用
tag（`!!python/...`）會被拒絕並丟出 `yaml.YAMLError`。

---

### `pin_thread_pools() -> None`

把呼叫端行程可能產生的每個 thread pool 都釘在一條執行緒：torch 的 intra-op pool
（`set_num_threads`）與 inter-op pool（`set_num_interop_threads`，行程已跑過平行區段時
略過）、numba，以及透過 `threadpoolctl` 的 BLAS/OpenMP pool；也會為之後才 import 的
函式庫設定 `OMP_NUM_THREADS`。供 worker 行程使用：worker 是 fork 出來的，繼承父行程已決定好
的執行緒數，否則每個 worker 都會跑「每核心一條」執行緒。訓練的 DataLoader worker
（`system.runner.seed_worker`）與評分用的 pool（`evaluation.parallel`）都呼叫它。numba 與
threadpoolctl 為選用，缺少時略過。

---

### `create_folder(folder_name: str) -> None`

透過 `os.makedirs(folder_name, exist_ok=True)` 建立一個目錄（連同中間所有
必要的目錄），如果它還不存在的話。它不會丟出 `FileExistsError`：與同時建立
同一個資料夾的其他建立者（多個 DataLoader worker 或 DDP rank）撞在一起時，只會以
debug 等級記錄後忽略。

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
- `"same"` – 長度 `max(len(x), len(kernel))`，取完整結果的中央段——是
  **比較長**那個輸入的長度，如果剛好是 `kernel` 比較長，就不一定等於 `len(x)`
- `"valid"` – 長度 `max(len(x), len(kernel)) - min(len(x), len(kernel)) +
  1`（只取兩個訊號完全重疊的區段）

以 `"full"` 模式把 waveform 與 RIR 卷積時，只要 RIR 的峰值不在第 0 個
sample，就會引入一段傳播延遲——可以用 `rir.abs().argmax(dim=-1)` 找出 peak
位置，據此裁切。
