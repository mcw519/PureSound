# puresound.audio.io

English version: [`io.md`](io.md)

音訊檔案 I/O:載入 + 重新取樣 + 調整音量水準、儲存、以及長度正規化裁切。
每個方法都是 `AudioIO` 上的 `@staticmethod`;這個 repo 裡沒有任何地方會
真的把這個 class 實例化 —— 一律呼叫 `AudioIO.open(...)`,而不是
`AudioIO().open(...)`。

## Class: `AudioIO`

### Constructor

```python
AudioIO(verbose: bool = False)
```

雖然存在,但形同虛設:`self.verbose` 有被存起來,但沒有任何方法會去讀它
(所有方法都是 `@staticmethod`,需要 `verbose` 的地方都是各自吃自己的
參數)。這個 repo 裡沒有任何呼叫點會真的建立 `AudioIO` 的實例。

### Static Methods

#### `audio_info(f_path: str) -> Tuple[int, int, float, int]`

透過 `torchaudio.info` 讀取檔案的 metadata,**不會載入波形本身**。回傳一
個**單純的 tuple**,不是 dict —— 順序很重要:

```python
sample_rate, total_samples, total_seconds, num_channels = AudioIO.audio_info(f_path)
```

`total_seconds` 是 `round(num_frames / sample_rate, 2)`。

---

#### `open(f_path: str, resample_to: Optional[int] = None, normalized: bool = False, target_lvl: Optional[float] = None, verbose: bool = False) -> Tuple[Tensor, int]`

用 `torchaudio.load` 載入檔案,視需要透過
[`dsp.wav_resampling(..., backend="sox")`](dsp.md) 重新取樣,再視需要調整
音量水準。`resample_to` 是在調整水準**之前**發生的,所以 `target_lvl` 是
在已經重新取樣過的訊號上量測的。

> **調整水準這裡有個地雷:單獨設 `normalized=True` 什麼事都不會發生。**
> 實際的邏輯是
> ```python
> if normalized:
>     if target_lvl is not None and verbose:
>         wav = normalize_waveform(wav=wav, amp_type="avg")
> elif target_lvl is not None:
>     wav = rescale_waveform(wav=wav, target_lvl=target_lvl, amp_type="rms", scale="dB")
> ```
> 所以 peak/avg 正規化只有在 `normalized=True` **而且**同時設了
> `target_lvl` **而且** `verbose=True` 的時候才會真的發生。這個 repo 裡
> 每一個呼叫點都完全不用 `normalized`,一律改用 `target_lvl` —— 這才是
> 該用的路徑:

```python
wav, sr = AudioIO.open(f_path, target_lvl=-28.0)   # RMS 正規化到 -28 dBFS
```

單獨設 `target_lvl`(常見用法,`normalized` 維持預設的 `False`)會透過
`rescale_waveform(..., amp_type="rms", scale="dB")` 做縮放 —— 見
[volume.md](volume.md)。`target_lvl=None`(同樣是預設值)搭配
`normalized=False` 完全不會做任何水準調整 —— `open` 就只是載入(視需要
重新取樣)而已。

---

#### `save(wav: Tensor, f_path: str, sr: int, **kwargs)`

**`wav` 是第一個位置參數,`f_path` 是第二個** —— 跟 `open`/`audio_info`
(路徑放第一個)的順序相反。1 維的 `wav` 在儲存前會先被 unsqueeze 成
`[1, L]`。`**kwargs` 會轉傳給 `torchaudio.save`(例如 `encoding`、
`bits_per_sample`)。

```python
AudioIO.save(wav=wav, f_path="output.wav", sr=16000)
```

---

#### `audio_cut(wav: Tensor, sr: int, length_s: float) -> Tuple[Tensor, Tuple[int, int]]`

便利包裝:`cut_audio(wav, sr, length_s, padding=True)`。回傳
`(wav, (offset, end_offset))`。

---

#### `cut_audio(wav: Tensor, sr: int, length_s: int, padding: bool = False) -> Tuple[Tensor, int, int]`

**隨機偏移、裁到固定目標長度**(`sr * length_s` 個樣本)—— 不是一個
確定性的 `[start, end)` 切片:

| 條件 | 結果 |
|---|---|
| `wav.shape[-1] > target_len` | 在 `[0, len(wav) - target_len]` 內隨機取一個 offset,裁成剛好 `target_len` |
| `wav.shape[-1] <= target_len` 且 `padding=True` | 在尾端補零到 `target_len` |
| `wav.shape[-1] <= target_len` 且 `padding=False` | 原封不動回傳 —— **會比 `target_len` 短**,呼叫端要自己處理 |

回傳**3 個值**,`(wav, offset, end_offset)`,不是 1 個。

`audio_cut` 跟 `cut_audio` 目前在這個 repo 裡都沒有外部呼叫端,也沒有
測試(已用 grep 確認)。dataset 層自己的長度對齊工具 ——
`puresound/dataset/dynamic_base.py` 裡的 `align_audio_list` —— 是另一個
獨立實作的「隨機裁切或補零到目標長度」邏輯,並不會呼叫這兩個函式。

## 範例

```python
from puresound.audio.io import AudioIO

wav, sr = AudioIO.open("speech.wav", resample_to=16000, target_lvl=-28.0)
sample_rate, total_samples, duration_s, num_channels = AudioIO.audio_info("speech.wav")
AudioIO.save(wav=wav, f_path="output.wav", sr=sr)
```
