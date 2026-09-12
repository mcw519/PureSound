# puresound.audio.volume

English version: [`volume.md`](volume.md)

振幅域的工具函式:RMS 量測、正規化/縮放,以及三個貫穿 `augmentation.py`
跟 `task/*.py` dataset pipeline 的失真/淡入淡出效果。全部都是對
`wav: [..., L]`(時間軸在最後)操作,而且都會保持原本的 shape。

## Functions

### `calculate_rms(wav: Tensor, to_log: bool = False) -> Tensor`

`sqrt(mean(wav**2, dim=-1))`,只在最後一個軸上做 reduce —— 對
`wav: [C, L]` 來說,回傳的是 `[C]`,**是一個 tensor,不是 Python
scalar**,除非 `wav` 本身是 1 維的。`to_log=True` 會回傳
`20 * log10(rms)`(類似 dBFS 的形式),而不是線性數值。

---

### `normalize_waveform(wav: Tensor, amp_type: str = "avg") -> Tensor`

用三種逐訊號振幅量測值其中一種去除 `wav`(分母加上 `1e-14` 避免除以
零):

| `amp_type` | 分母 | reduction |
|---|---|---|
| `"avg"`(預設) | `mean(\|wav\|, dim=-1)` | `keepdim=True` |
| `"peak"` | `max(\|wav\|, dim=-1)` | `keepdim=True` |
| `"rms"` | `sqrt(mean(wav**2, dim=-1))` | **`keepdim=False`** |

> **Shape 地雷,已驗證**:因為 `"rms"` 這個分支的 reduce 沒有帶
> `keepdim`,所以只有在 `wav` 的第一維大小恰好是 1(常見的 `[1, L]`
> 單聲道情況)時,才能正確 broadcast 回去 —— 真正多聲道的
> `wav: [C, L]`(`C > 1`)在做 `wav / den` 時會丟出 `RuntimeError`
> (已驗證:`normalize_waveform(torch.randn(2, 16000), amp_type="rms")`
> 會失敗;同一個呼叫換成 `amp_type="avg"`/`"peak"` 則會成功,因為它們
> `keepdim=True` 算出來的分母 shape 是 `[C, 1]`,不管 `C` 是多少都能
> broadcast)。這個 repo 裡所有真實語料都是單聲道,所以實務上還沒踩到
> 這個雷,但不要把真正的多聲道音訊餵進 `amp_type="rms"`。

---

### `rescale_waveform(wav: Tensor, target_lvl: float, amp_type: str = "avg", scale: str = "linear") -> Tensor`

**是兩個步驟,不是一個**:先做 `normalize_waveform(wav, amp_type)`
(讓訊號落在 `amp_type` 對應的單位水準上),*然後*再乘上 `target_lvl`
(如果 `scale="db"`,會先把 `target_lvl` 換算成線性值:
`target_lvl = 10**(target_lvl/20)`)。`amp_type` 決定 `target_lvl` 是
相對於*哪一種*振幅量測值;`scale` 決定 `target_lvl` 本身是用什麼單位
表示。因為內部會呼叫 `normalize_waveform`,上面提到的
`amp_type="rms"` + 多聲道 shape 地雷,在這裡一樣適用。

這個 repo 裡最常見的用法,是以 RMS 為基準的 dB 水準調整(
[`AudioIO.open`](io.md) 的 `target_lvl` 參數做的就是這件事):

```python
wav_m28dBFS = rescale_waveform(wav, target_lvl=-28.0, amp_type="rms", scale="dB")
```

`scale` 預設是 `"linear"`,**不是** `"dB"` —— 上面那種 dBFS 用法要明確
傳入 `scale="dB"`。

---

### `rand_gain_distortion(wav: Tensor, sample_rate: int = 16000, start_time: Optional[float] = None, duration: Optional[float] = None, return_info: bool = False) -> Tensor | Tuple[Tensor, Tuple]`

**不是對整段訊號做增益縮放** —— 是對 `wav` 裡隨機一個*連續片段*乘上
隨機增益,再把結果裁切到 `[-1, 1]`。這就是
[`augmentation.apply_gain_distortion`](augmentation.md) 對外公開的功能;
這裡沒有 min/max-gain 範圍參數,所有東西都在內部隨機決定:

- `distortion_gain = 4 ** N(0, 1)` —— 類似 log-normal 分布,中心在
  unity gain(增益為 1),在對數空間中,每一個標準差對稱地對應
  `×4`/`÷4`。
- 如果沒有給 `start_time`/`duration`(單位秒),片段的起點跟長度會各自
  在整段剩餘訊號上均勻抽樣(`randint(0, len(wav))`,再
  `randint(0, len(wav) - start)`)—— 完全隨機的位置。
- 如果有給,兩者都不會被精確使用,而是會被「dither」過:實際的
  起點/長度(取樣點數)是
  `int((value + Uniform(0, 1) * value) * sample_rate)` —— 所以
  `start_time=0.1` 實際上可能落在 `[0.1s, 0.2s]` 之間的任何地方,也就是
  一個「下限」被拉伸最多 2 倍,不是精確的位置(已驗證:在
  `sample_rate=16000` 下,`start_time=0.1, duration=0.2` 實際產生的
  起點是 1827 個取樣點、長度是 5366 個取樣點,兩者都落在各自
  `[1×, 2×]` 的範圍內)。

`return_info=True` 會回傳 `(distorted_wav, (start_sample,
duration_samples, gain))`,讓呼叫端可以記錄下來,或是重新套用一模一樣
的失真。

---

### `wav_fade_in(wav: Tensor, sr: int, fade_len_s: int, fade_begin_s: int = 0, fade_shape: str = "linear") -> Tensor`
### `wav_fade_out(wav: Tensor, sr: int, fade_len_s: int, fade_begin_s: int = 0, fade_shape: str = "linear") -> Tensor`

在 `fade_len_s` 秒內,從(fade-in)或往(fade-out)0 做振幅漸變,從
`fade_begin_s` 開始。`fade_shape`(以下是 fade-in 的曲線;fade-out
則是每條曲線的鏡像,`1 - fade`):

| 形狀 | 曲線,`t = linspace(0, 1, fade_len_s * sr)` |
|---|---|
| `"linear"` | `t` |
| `"exponential"` | `2**(t - 1) * t` |
| `"logarithmic"` | `log10(0.1 + t) + 1` |

這段漸變曲線,前後會分別接上一段全為 1 的區塊,再整體裁切到
`[0, 1]`,然後乘進 `wav` —— `[fade_begin_s, fade_begin_s + fade_len_s]`
範圍以外的部分完全不受影響(相當於乘上 1)。雖然帶 `_s` 後綴的參數在
函式簽章裡型別標的是 `int`,但實際上是直接當作秒數使用
(`int(sr * fade_len_s)`),所以小數秒數(例如 `fade_len_s=0.01`)實務上
也能正常運作。

---

### `wav_clipping(wav: Tensor, min_quantile: float = 0.0, max_quantile: float = 0.9) -> Tensor`

在 `wav` 自己的**經驗分位數**上做硬裁切,每次呼叫都重新計算 —— 不是
固定的 dB 門檻。`torch.quantile(wav, [min_quantile, max_quantile])` 算出
`(min_, max_)`,再做 `torch.clip(wav, min_, max_)`。

> 預設邊界是**不對稱的**:`min_quantile=0.0` 就是訊號自己的最小值
> (等於沒做事的下限 —— 預設情況下底部完全不會被裁切),而
> `max_quantile=0.9` 會裁掉頂部 10%。已驗證:用預設值時,
> `clipped.min() == wav.min()`,但 `clipped.max() < wav.max()`。如果要
> 對稱裁切,要明確傳入例如 `min_quantile=0.01, max_quantile=0.99`。

## 範例

```python
from puresound.audio.volume import rescale_waveform, wav_fade_in, wav_clipping

wav_m28 = rescale_waveform(wav, target_lvl=-28.0, amp_type="rms", scale="dB")
wav_faded = wav_fade_in(wav_m28, sr=16000, fade_len_s=0.01, fade_shape="linear")
wav_clipped = wav_clipping(wav_faded, min_quantile=0.05, max_quantile=0.95)
```
