# puresound.audio.volume

English version: [volume.md](volume.md)

振幅域工具：RMS 量測、正規化到目標位準，以及三種位準效果（片段增益、淡入淡出、
分位數截斷）。全部接受 `wav: [..., L]`，只沿最後一軸縮減或作用；輸出保持輸入
形狀。

## `calculate_rms(wav, to_log=False) -> Tensor`

`rms = sqrt(mean(wav², dim=-1))`，每個前導索引一個值（`[C, L]` 得 `[C]`，`[L]`
得 0 維 tensor）。`to_log=True` 回傳 `20·log10(rms)`，即 dBFS 位準。靜音在 log
形式下為 `-inf`。

## `normalize_waveform(wav, amp_type="avg") -> Tensor`

除以一個逐訊號的振幅量再加 `1e-14`，使用 `keepdim=True`，因此任何聲道數都能
broadcast：

| `amp_type` | 分母 |
| --- | --- |
| `"avg"` | `mean(abs(wav))` |
| `"peak"` | `max(abs(wav))` |
| `"rms"` | `sqrt(mean(wav²))` |

## `rescale_waveform(wav, target_lvl, amp_type="avg", scale="linear") -> Tensor`

`normalize_waveform(wav, amp_type) · g`，`scale="linear"` 時 `g = target_lvl`，
`scale="dB"`（大小寫不拘）時 `g = 10^(target_lvl/20)`。預設是線性，所以 repo 中
常用的 dBFS 位準調整要明確寫出：

```python
rescale_waveform(wav, target_lvl=-28.0, amp_type="rms", scale="dB")   # RMS 在 -28 dBFS
```

[`AudioIO.open(..., target_lvl=...)`](io.zh-TW.md) 呼叫的就是它。

## `rand_gain_distortion(wav, sample_rate=16000, start_time=None, duration=None, return_info=False)`

把一段連續片段乘上隨機增益，再把整段訊號硬截到 [-1, 1]；它模擬的是瞬間的增益
跳動加上轉換器飽和，而不是整段的位準變化。

- 增益：`4^z`，`z ~ N(0, 1)` 由 Python 的 `random` 抽，即以 1 為中心的
  log-normal，標準差 `20·log10 4 ≈ 12 dB`。
- 片段起點（樣本）：`start_time` 為 `None` 時為 `randint(0, L)`，否則為
  `int(start_time·(1 + u)·sample_rate)`，`u ~ U(0, 1)`，所以起點落在
  `[start_time, 2·start_time)` 秒。
- 片段長度：`duration` 為 `None` 時為 `randint(0, L − start)`，否則為
  `int(duration·(1 + u)·sample_rate)`，以同樣方式抖動。

`return_info=True` 回傳 `(wav, (start_sample, duration_samples, gain))`，方便記錄
或重現同一個失真。對 recipe 以 `AudioEffectAugmentor.apply_gain_distortion`
提供（見 [augmentation.zh-TW.md](augmentation.zh-TW.md)）。

## `wav_fade_in(wav, sr, fade_len_s, fade_begin_s=0, fade_shape="linear")` / `wav_fade_out(...)`

把 `wav` 乘上一個包絡：除了 `[fade_begin_s, fade_begin_s + fade_len_s)` 秒這段
窗之外都是 1；窗內依 `t = linspace(0, 1, int(sr·fade_len_s))` 走三種曲線之一，
並截在 [0, 1]：

| `fade_shape` | 淡入 | 淡出 |
| --- | --- | --- |
| `"linear"` | `t` | `1 − t` |
| `"exponential"` | `2^(t−1)·t` | `2^(−t)·(1 − t)` |
| `"logarithmic"` | `log10(0.1 + t) + 1` | `log10(1.1 − t) + 1` |

每條淡出曲線都是對應淡入曲線的時間反轉。參數單位是秒，儘管型別標註為 `int`，
仍可給小數。窗必須落在訊號內（`fade_begin_s + fade_len_s ≤ L / sr`），否則包絡
長度檢查會失敗。

## `wav_clipping(wav, min_quantile=0.0, max_quantile=0.9) -> Tensor`

以訊號自己的經驗分位數硬截：
`torch.clip(wav, quantile(wav, min_quantile), quantile(wav, max_quantile))`，
每次呼叫重新計算。截斷位準因此相對於訊號本身，而不是固定的 dBFS 門檻。預設值
不對稱：`0.0` 是最小樣本（下方不截），`0.9` 則截掉最上面 10 % 的樣本。請傳單聲道
`[1, L]` 或 `[L]`；`C > 1` 的 `[C, L]` 輸入，其逐聲道邊界無法 broadcast。

`clipping_thresholds(wav, min_quantile, max_quantile)` 只回傳兩個位準 `(lo, hi)`、
不截斷，讓第二個訊號能削在第一個訊號的位準上；`DeviceChain` 對目標就是這樣做
（見 [位準與動態](../augmentation/level_dynamics.zh-TW.md)）。

## 範例

```python
from puresound.audio.volume import rescale_waveform, wav_clipping, wav_fade_in

wav = rescale_waveform(wav, target_lvl=-28.0, amp_type="rms", scale="dB")
wav = wav_fade_in(wav, sr=16000, fade_len_s=0.01)
wav = wav_clipping(wav, min_quantile=0.01, max_quantile=0.99)   # 對稱
```
